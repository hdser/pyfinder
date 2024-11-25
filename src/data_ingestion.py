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
        """Process edges for a single chunk of data."""
        # Merge trusts and balances
        merged = trust_chunk.merge(
            balance_chunk,
            left_on='trustee',
            right_on='tokenAddress',
            how='inner'
        )
        
        # Filter out self-transfers
        merged = merged[merged['account'] != merged['truster']].copy()
        
        # Convert balances
        merged['balance'] = merged['demurragedTotalBalance'].apply(self._convert_balance)
        merged = merged[merged['balance'] > 0]
        
        # Map to IDs
        merged['account_id'] = merged['account'].map(self.address_to_id)
        merged['truster_id'] = merged['truster'].map(self.address_to_id)
        merged['token_id'] = merged['tokenAddress'].map(self.address_to_id)
        
        # Create intermediate nodes
        merged['intermediate_node'] = merged['account_id'] + '_' + merged['token_id']
        
        # Create edges
        edges1 = merged[['account_id', 'intermediate_node', 'token_id', 'balance']]
        edges2 = merged[['intermediate_node', 'truster_id', 'token_id', 'balance']]
        
        edges1.columns = ['from', 'to', 'token', 'capacity']
        edges2.columns = ['from', 'to', 'token', 'capacity']
        
        all_edges = pd.concat([edges1, edges2]).drop_duplicates()
        
        # Add to main edge lists
        new_edges = list(zip(all_edges['from'], all_edges['to']))
        new_capacities = all_edges['capacity'].tolist()
        new_tokens = all_edges['token'].tolist()
        
        self.edges.extend(new_edges)
        self.capacities.extend(new_capacities)
        self.tokens.extend(new_tokens)
        
        # Clean up
        del merged
        del edges1
        del edges2
        del all_edges
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