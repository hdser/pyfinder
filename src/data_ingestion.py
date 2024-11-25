# src/data_ingestion.py
import pandas as pd
from typing import List, Tuple, Dict
from sqlalchemy import create_engine
import urllib.parse
import os
from pathlib import Path
from dotenv import load_dotenv

class DataIngestion:
    def __init__(self, df_trusts: pd.DataFrame, df_balances: pd.DataFrame):
        """Initialize with trust and balance DataFrames."""
        # Ensure addresses are lowercased for consistency
        df_trusts['truster'] = df_trusts['truster'].str.lower()
        df_trusts['trustee'] = df_trusts['trustee'].str.lower()
        df_balances['account'] = df_balances['account'].str.lower()
        df_balances['tokenAddress'] = df_balances['tokenAddress'].str.lower()

        self.df_trusts = df_trusts
        self.df_balances = df_balances
        self.address_to_id, self.id_to_address = self._create_id_mappings()
        self.edges, self.capacities, self.tokens = self._create_edge_data()

    @classmethod
    def from_csv_files(cls, trusts_file: str, balances_file: str) -> 'DataIngestion':
        """Create instance from CSV files."""
        try:
            df_trusts = pd.read_csv(
                trusts_file, 
                dtype={'truster': 'str', 'trustee': 'str'},
                low_memory=False
            )
            df_balances = pd.read_csv(
                balances_file, 
                dtype={
                    'demurragedTotalBalance': 'float32',
                    'account': 'str',
                    'tokenAddress': 'str'
                }, 
                low_memory=False
            )
            return cls(df_trusts, df_balances)
        except Exception as e:
            raise ValueError(f"Error reading CSV files: {str(e)}")

    def _create_id_mappings(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Create mappings between addresses and internal IDs."""
        unique_addresses = pd.concat([
            self.df_trusts['trustee'], 
            self.df_trusts['truster'], 
            self.df_balances['account'],
            self.df_balances['tokenAddress']
        ]).drop_duplicates().reset_index(drop=True)
        
        address_to_id = {addr: str(idx) for idx, addr in enumerate(unique_addresses)}
        id_to_address = {str(idx): addr for idx, addr in enumerate(unique_addresses)}
        
        return address_to_id, id_to_address

    def _create_edge_data(self) -> Tuple[List[Tuple[str, str]], List[int], List[str]]:
        """Create graph edge data from trust and balance information."""
        # Create unique trust relationships
        df_trusts_unique = self.df_trusts.drop_duplicates()
        
        # Add self-trust relationships
        self_trust = pd.DataFrame({
            'truster': self.df_trusts['truster'].unique(),
            'trustee': self.df_trusts['truster'].unique()
        })
        df_truster_trustees = pd.concat([df_trusts_unique, self_trust]).drop_duplicates()

        # Merge with balances
        df_merged = df_truster_trustees.merge(
            self.df_balances,
            left_on='trustee',
            right_on='tokenAddress',
            how='left'
        )

        # Filter and process edges
        df_filtered = df_merged[df_merged['account'] != df_merged['truster']].copy()
        df_filtered['balance'] = df_filtered['demurragedTotalBalance'].apply(self._convert_balance)
        df_filtered = df_filtered[df_filtered['balance'] > 0]

        # Create node mappings
        df_filtered['account_id'] = df_filtered['account'].map(self.address_to_id)
        df_filtered['truster_id'] = df_filtered['truster'].map(self.address_to_id)
        df_filtered['token_id'] = df_filtered['tokenAddress'].map(self.address_to_id)

        # Create intermediate nodes
        df_filtered['intermediate_node'] = df_filtered['account_id'] + '_' + df_filtered['token_id']

        # Create edges and their properties
        edges1 = df_filtered[['account_id', 'intermediate_node', 'token_id', 'balance']].drop_duplicates()
        edges2 = df_filtered[['intermediate_node', 'truster_id', 'token_id', 'balance']].drop_duplicates()
        
        edges1.columns = ['from', 'to', 'token', 'capacity']
        edges2.columns = ['from', 'to', 'token', 'capacity']
        all_edges = pd.concat([edges1, edges2]).drop_duplicates()

        # Create final lists
        edges = list(zip(all_edges['from'], all_edges['to']))
        capacities = all_edges['capacity'].tolist()
        tokens = all_edges['token'].tolist()

        return edges, capacities, tokens

    @staticmethod    
    def _convert_balance(balance_str) -> int:
        """Convert balance string to integer in mCRC."""
        if pd.isna(balance_str):
            return 0
        try:
            return int(int(balance_str) / 10**15)
        except ValueError:
            print(f"Warning: Unable to convert balance: {balance_str}")
            return 0

    def get_id_for_address(self, address: str) -> str:
        """Get internal ID for a given address."""
        return self.address_to_id.get(address.lower())

    def get_address_for_id(self, id: str) -> str:
        """Get address for a given internal ID."""
        return self.id_to_address.get(id)


class PostgresDataIngestion(DataIngestion):
    def __init__(self, db_config: Dict[str, str], queries_dir: str = "queries"):
        """Initialize PostgreSQL data ingestion."""
        self.db_config = db_config
        self.queries_dir = Path(queries_dir)
        self.engine = self._create_engine()
        df_trusts, df_balances = self._init_data()
        super().__init__(df_trusts, df_balances)

    @classmethod
    def from_env_vars(cls, queries_dir: str = "queries") -> 'PostgresDataIngestion':
        """Create instance from environment variables."""
        load_dotenv()
        
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
        
        return cls(db_config, queries_dir)

    def _create_engine(self):
        """Create SQLAlchemy engine from database configuration."""
        password = urllib.parse.quote_plus(self.db_config['password'])
        db_url = (
            f"postgresql://{self.db_config['user']}:{password}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        )
        return create_engine(db_url)

    def _read_query(self, filename: str) -> str:
        """Read SQL query from file."""
        query_path = self.queries_dir / filename
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")
        
        with open(query_path, 'r') as f:
            return f.read()

    def _init_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Initialize data by fetching from PostgreSQL."""
        try:
            trust_query = self._read_query("trust_relationships.sql")
            balance_query = self._read_query("account_balances.sql")
            
            df_trusts = pd.read_sql_query(trust_query, self.engine)
            df_balances = pd.read_sql_query(balance_query, self.engine)
            
            return df_trusts, df_balances
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error reading SQL queries: {str(e)}")
        except Exception as e:
            raise Exception(f"Error executing SQL queries: {str(e)}")