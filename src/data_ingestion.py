import pandas as pd
from typing import List, Tuple, Dict
from sqlalchemy import create_engine
import urllib.parse
import os
from pathlib import Path

class DataIngestion:
    def __init__(self, df_trusts: pd.DataFrame, df_balances: pd.DataFrame):
        # Ensure addresses are lowercased for consistency
        df_trusts['truster'] = df_trusts['truster'].str.lower()
        df_trusts['trustee'] = df_trusts['trustee'].str.lower()
        df_balances['account'] = df_balances['account'].str.lower()
        df_balances['tokenAddress'] = df_balances['tokenAddress'].str.lower()

        self.df_trusts = df_trusts
        self.df_balances = df_balances
        self.address_to_id, self.id_to_address = self._create_id_mappings()
        self.edges, self.capacities, self.tokens = self._create_edge_data()

    def _create_id_mappings(self) -> Tuple[Dict[str, str], Dict[str, str]]:
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
        # Create a DataFrame of unique truster-trustee pairs
        df_trusts_unique = self.df_trusts.drop_duplicates()
        
        # Add self-trust relationships
        self_trust = pd.DataFrame({'truster': self.df_trusts['truster'].unique(), 'trustee': self.df_trusts['truster'].unique()})
        df_truster_trustees = pd.concat([df_trusts_unique, self_trust]).drop_duplicates()

        # Merge with balances where tokenAddress matches trustee
        df_merged = df_truster_trustees.merge(
            self.df_balances, left_on='trustee', right_on='tokenAddress', how='left'
        )

        # Filter out rows where account is the same as truster
        df_filtered = df_merged[df_merged['account'] != df_merged['truster']].copy()

        # Convert balances and filter out non-positive balances
        df_filtered['balance'] = df_filtered['demurragedTotalBalance'].apply(self._convert_balance)
        df_filtered = df_filtered[df_filtered['balance'] > 0]

        # Map addresses to IDs
        df_filtered['account_id'] = df_filtered['account'].map(self.address_to_id)
        df_filtered['truster_id'] = df_filtered['truster'].map(self.address_to_id)
        df_filtered['token_id'] = df_filtered['tokenAddress'].map(self.address_to_id)

        # Create intermediate nodes
        df_filtered['intermediate_node'] = df_filtered['account_id'] + '_' + df_filtered['token_id']

        # Create edges from account to intermediate node
        edges1 = df_filtered[['account_id', 'intermediate_node', 'token_id', 'balance']].drop_duplicates()
        edges1.columns = ['from', 'to', 'token', 'capacity']

        # Create edges from intermediate node to truster
        edges2 = df_filtered[['intermediate_node', 'truster_id', 'token_id', 'balance']].drop_duplicates()
        edges2.columns = ['from', 'to', 'token', 'capacity']

        # Combine all edges
        all_edges = pd.concat([edges1, edges2]).drop_duplicates()

        # Create the final edge list, capacities, and tokens
        edges = list(zip(all_edges['from'], all_edges['to']))
        capacities = all_edges['capacity'].tolist()
        tokens = all_edges['token'].tolist()


        # Debug: Check for duplicates in the created edges
        edge_set = set()
        duplicate_count = 0
        for (u, v), token in zip(edges, tokens):
            edge_key = (u, v, token)
            if edge_key in edge_set:
                duplicate_count += 1
            else:
                edge_set.add(edge_key)
        
        if duplicate_count > 0:
            print(f"Warning: Found {duplicate_count} duplicate edges")

        return edges, capacities, tokens
    
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
    def __init__(self, db_config: Dict[str, str], queries_dir: str = "queries"):
        """
        Initialize PostgreSQL data ingestion with database configuration.
        
        Args:
            db_config: Dictionary containing database connection parameters:
                - host: database host
                - port: database port
                - dbname: database name
                - user: database user
                - password: database password
            queries_dir: Directory containing SQL query files (default: "queries")
        """
        self.db_config = db_config
        self.queries_dir = Path(queries_dir)
        self.engine = self._create_engine()
        self._init_data()
    
    def _create_engine(self):
        """Create SQLAlchemy engine from database configuration."""
        password = urllib.parse.quote_plus(self.db_config['password'])
        db_url = f"postgresql://{self.db_config['user']}:{password}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        return create_engine(db_url)

    def _read_query(self, filename: str) -> str:
        """Read SQL query from file."""
        query_path = self.queries_dir / filename

        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")
        
        with open(query_path, 'r') as f:
            return f.read()

    def _init_data(self):
        """Initialize the data by fetching from PostgreSQL and processing it."""
        try:
            # Read and execute trust relationships query
            trust_query = self._read_query("trust_relationships.sql")
            df_trusts = pd.read_sql_query(trust_query, self.engine)

            # Read and execute account balances query
            balances_query = self._read_query("account_balances.sql")
            df_balances = pd.read_sql_query(balances_query, self.engine)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error reading SQL queries: {str(e)}")
        except Exception as e:
            raise Exception(f"Error executing SQL queries: {str(e)}")

        self.df_trusts = df_trusts
        self.df_balances = df_balances
        self.address_to_id, self.id_to_address = self._create_id_mappings()
        self.edges, self.capacities, self.tokens = self._create_edge_data()

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
        """Create edge data for the graph from the trust and balance information."""
        # Create a DataFrame of unique truster-trustee pairs
        df_trusts_unique = self.df_trusts.drop_duplicates()
        
        # Add self-trust relationships
        self_trust = pd.DataFrame({
            'truster': self.df_trusts['truster'].unique(), 
            'trustee': self.df_trusts['truster'].unique()
        })
        df_truster_trustees = pd.concat([df_trusts_unique, self_trust]).drop_duplicates()

        # Merge with balances where tokenAddress matches trustee
        df_merged = df_truster_trustees.merge(
            self.df_balances, 
            left_on='trustee', 
            right_on='tokenAddress', 
            how='left'
        )

        # Filter out rows where account is the same as truster
        df_filtered = df_merged[df_merged['account'] != df_merged['truster']].copy()

        # Convert balances and filter out non-positive balances
        df_filtered['balance'] = df_filtered['demurragedTotalBalance'].apply(self._convert_balance)
        df_filtered = df_filtered[df_filtered['balance'] > 0]

        # Map addresses to IDs
        df_filtered['account_id'] = df_filtered['account'].map(self.address_to_id)
        df_filtered['truster_id'] = df_filtered['truster'].map(self.address_to_id)
        df_filtered['token_id'] = df_filtered['tokenAddress'].map(self.address_to_id)

        # Create intermediate nodes
        df_filtered['intermediate_node'] = df_filtered['account_id'] + '_' + df_filtered['token_id']

        # Create edges from account to intermediate node
        edges1 = df_filtered[['account_id', 'intermediate_node', 'token_id', 'balance']].drop_duplicates()
        edges1.columns = ['from', 'to', 'token', 'capacity']

        # Create edges from intermediate node to truster
        edges2 = df_filtered[['intermediate_node', 'truster_id', 'token_id', 'balance']].drop_duplicates()
        edges2.columns = ['from', 'to', 'token', 'capacity']

        # Combine all edges
        all_edges = pd.concat([edges1, edges2]).drop_duplicates()

        # Create the final edge list, capacities, and tokens
        edges = list(zip(all_edges['from'], all_edges['to']))
        capacities = all_edges['capacity'].tolist()
        tokens = all_edges['token'].tolist()

        return edges, capacities, tokens

    @staticmethod    
    def _convert_balance(balance_str):
        """Convert balance string to integer value in mCRC."""
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

    @classmethod
    def from_env_vars(cls, queries_dir: str = "queries"):
        """Create an instance using environment variables for database configuration."""
        import os
        
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