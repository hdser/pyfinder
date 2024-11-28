import pandas as pd
from typing import List, Tuple, Dict
from sqlalchemy import create_engine
import urllib.parse
import os
from pathlib import Path
import gc

class DataIngestion:
    def __init__(self, df_trusts: pd.DataFrame, df_balances: pd.DataFrame, chunk_size: int = 100_000):
        """Initialize data ingestion with chunked processing."""
        # Create initial mappings
        unique_addresses = pd.concat([
            df_trusts['trustee'].str.lower(), 
            df_trusts['truster'].str.lower(), 
            df_balances['account'].str.lower(),
            df_balances['tokenAddress'].str.lower()
        ]).drop_duplicates().reset_index(drop=True)
        
        self.address_to_id = {addr: str(idx) for idx, addr in enumerate(unique_addresses)}
        self.id_to_address = {str(idx): addr for idx, addr in enumerate(unique_addresses)}
        
        # Initialize containers for edges
        self.edges = []
        self.capacities = []
        self.tokens = []
        
        # Process data in chunks
        self._process_chunks(df_trusts, df_balances, chunk_size)
        
        # Clean up
        gc.collect()

    def _process_chunks(self, df_trusts: pd.DataFrame, df_balances: pd.DataFrame, chunk_size: int):
        """Process data in chunks to manage memory."""
        # Process trusts in chunks
        for start in range(0, len(df_trusts), chunk_size):
            # Get chunk of trust relationships
            trust_chunk = df_trusts.iloc[start:start + chunk_size].copy()
            trust_chunk['truster'] = trust_chunk['truster'].str.lower()
            trust_chunk['trustee'] = trust_chunk['trustee'].str.lower()
            
            # Add self-trust for this chunk
            self_trust = pd.DataFrame({
                'truster': trust_chunk['truster'].unique(),
                'trustee': trust_chunk['truster'].unique()
            })
            
            # Process balances for relevant trustees
            trustees = pd.concat([trust_chunk['trustee'], self_trust['trustee']]).unique()
            
            # Get relevant balances
            relevant_balances = df_balances[
                df_balances['tokenAddress'].str.lower().isin(trustees)
            ].copy()
            
            if not relevant_balances.empty:
                relevant_balances['account'] = relevant_balances['account'].str.lower()
                relevant_balances['tokenAddress'] = relevant_balances['tokenAddress'].str.lower()
                
                # Process edges for this chunk
                self._process_chunk_edges(
                    pd.concat([trust_chunk, self_trust]),
                    relevant_balances
                )
            
            # Clean up chunk data
            del trust_chunk
            del self_trust
            del relevant_balances
            gc.collect()

    def _process_chunk_edges(self, trust_chunk: pd.DataFrame, balance_chunk: pd.DataFrame):
        """
        Process edges for a single chunk of data, ensuring no duplicate edges are created.
        
        The function creates two types of edges:
        1. Account -> Intermediate node (representing token holding)
        2. Intermediate node -> Truster (representing trust relationships)
        """
        # First, process the balance information to create account -> intermediate edges
        balance_edges = balance_chunk.copy()
        balance_edges['account'] = balance_edges['account'].str.lower()
        balance_edges['tokenAddress'] = balance_edges['tokenAddress'].str.lower()
        
        # Convert balances and filter out zero balances
        balance_edges['balance'] = balance_edges['demurragedTotalBalance'].apply(self._convert_balance)
        balance_edges = balance_edges[balance_edges['balance'] > 0]
        
        # Create the first set of edges (account -> intermediate)
        balance_edges['account_id'] = balance_edges['account'].map(self.address_to_id)
        balance_edges['token_id'] = balance_edges['tokenAddress'].map(self.address_to_id)
        balance_edges['intermediate_node'] = balance_edges['account_id'] + '_' + balance_edges['token_id']
        
        # Create holder -> intermediate edges
        holder_edges = balance_edges[[
            'account_id', 'intermediate_node', 'token_id', 'balance'
        ]].rename(columns={
            'account_id': 'from',
            'intermediate_node': 'to',
            'token_id': 'token',
            'balance': 'capacity'
        })
        
        # Now process trust relationships
        trust_edges = trust_chunk.merge(
            balance_edges[['account', 'tokenAddress', 'intermediate_node', 'balance']],
            left_on='trustee',
            right_on='tokenAddress',
            how='inner'
        )
        
        # Filter out self-trust relationships
        trust_edges = trust_edges[trust_edges['account'] != trust_edges['truster']]
        
        # Create intermediate -> truster edges
        trust_edges['truster_id'] = trust_edges['truster'].map(self.address_to_id)
        outgoing_edges = trust_edges[[
            'intermediate_node', 'truster_id', 'tokenAddress', 'balance'
        ]].rename(columns={
            'intermediate_node': 'from',
            'truster_id': 'to',
            'tokenAddress': 'token',
            'balance': 'capacity'
        })
        
        # Combine all edges and ensure uniqueness
        all_edges = pd.concat([holder_edges, outgoing_edges])
        all_edges['edge_key'] = all_edges['from'] + '_TO_' + all_edges['to']
        
        # For duplicate edges, keep the one with maximum capacity
        unique_edges = all_edges.sort_values('capacity', ascending=False).drop_duplicates(
            subset=['edge_key'], 
            keep='first'
        )
        
        # Convert to final format and extend main lists
        edge_tuples = list(zip(unique_edges['from'], unique_edges['to']))
        capacities = unique_edges['capacity'].tolist()
        tokens = unique_edges['token'].tolist()
        
        # Verify no duplicates before extending
        existing_edges = set((f, t) for f, t in self.edges)
        new_edges = []
        new_capacities = []
        new_tokens = []
        
        for (f, t), cap, tok in zip(edge_tuples, capacities, tokens):
            if (f, t) not in existing_edges:
                new_edges.append((f, t))
                new_capacities.append(cap)
                new_tokens.append(tok)
                existing_edges.add((f, t))
        
        # Extend the main lists with verified unique edges
        self.edges.extend(new_edges)
        self.capacities.extend(new_capacities)
        self.tokens.extend(new_tokens)
        
        # Clean up intermediary data
        del balance_edges
        del trust_edges
        del holder_edges
        del outgoing_edges
        del all_edges
        del unique_edges
        gc.collect()

    @staticmethod    
    def _convert_balance(balance_str):
        if pd.isna(balance_str):
            return 0
        try:
            return int(int(balance_str) / 10**15)
        except ValueError:
            print(f"Warning: Unable to convert balance: {balance_str}")
            return 0

    def get_id_for_address(self, address: str) -> str:
        return self.address_to_id.get(address.lower())

    def get_address_for_id(self, id: str) -> str:
        return self.id_to_address.get(id)


class PostgresDataIngestion:
    def __init__(self, db_config: Dict[str, str], queries_dir: str = "queries", chunk_size: int = 100_000):
        self.db_config = db_config
        self.queries_dir = Path(queries_dir)
        self.engine = self._create_engine()
        self.chunk_size = chunk_size
        self._init_data()
    
    def _create_engine(self):
        password = urllib.parse.quote_plus(self.db_config['password'])
        db_url = f"postgresql://{self.db_config['user']}:{password}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        return create_engine(db_url)

    def _read_query(self, filename: str) -> str:
        query_path = self.queries_dir / filename
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")
        with open(query_path, 'r') as f:
            return f.read()

    def _init_data(self):
        try:
            # Read data in chunks
            trust_query = self._read_query("trust_relationships.sql")
            balance_query = self._read_query("account_balances.sql")
            
            # Read full trusts data (usually smaller)
            df_trusts = pd.read_sql_query(trust_query, self.engine)
            
            # Initialize edge lists
            self.edges = []
            self.capacities = []
            self.tokens = []
            
            # Process balances in chunks
            for chunk_df in pd.read_sql_query(balance_query, self.engine, chunksize=self.chunk_size):
                data_ingestion = DataIngestion(df_trusts, chunk_df, self.chunk_size)
                
                # Extend edge data
                self.edges.extend(data_ingestion.edges)
                self.capacities.extend(data_ingestion.capacities)
                self.tokens.extend(data_ingestion.tokens)
                
                # Keep ID mappings from first chunk
                if not hasattr(self, 'address_to_id'):
                    self.address_to_id = data_ingestion.address_to_id
                    self.id_to_address = data_ingestion.id_to_address
                
                del data_ingestion
                gc.collect()
                
        except Exception as e:
            raise Exception(f"Error processing data: {str(e)}")

    def get_id_for_address(self, address: str) -> str:
        return self.address_to_id.get(address.lower())

    def get_address_for_id(self, id: str) -> str:
        return self.id_to_address.get(id)

    @classmethod
    def from_env_vars(cls, queries_dir: str = "queries", chunk_size: int = 100_000):
        required_vars = [
            'POSTGRES_HOST',
            'POSTGRES_PORT',
            'POSTGRES_DB',
            'POSTGRES_USER',
            'POSTGRES_PASSWORD'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        db_config = {
            'host': os.getenv('POSTGRES_HOST'),
            'port': os.getenv('POSTGRES_PORT'),
            'dbname': os.getenv('POSTGRES_DB'),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }
        
        return cls(db_config, queries_dir, chunk_size)