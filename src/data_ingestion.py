import pandas as pd
from typing import List, Tuple, Dict

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
    """
    def _create_edge_data3(self) -> Tuple[List[Tuple[str, str]], List[int], List[str]]:
        # Create a mapping from truster to their unique trustees (including themselves)
        truster_to_trustees = self.df_trusts.groupby('truster')['trustee'].apply(set).to_dict()
        for truster in self.df_trusts['truster'].unique():
            truster_to_trustees.setdefault(truster, set()).add(truster)

        # Convert the mapping into a DataFrame
        df_truster_trustees = pd.DataFrame([
            {'truster': truster, 'unique_trustee': trustee}
            for truster, trustees in truster_to_trustees.items()
            for trustee in trustees
        ])

        # Merge with balances where tokenAddress matches unique_trustee
        df_merged = df_truster_trustees.merge(
            self.df_balances, left_on='unique_trustee', right_on='tokenAddress'
        )

        # Filter out rows where account is the same as truster
        df_filtered = df_merged[df_merged['account'] != df_merged['truster']].copy()

        # Convert balances and filter out non-positive balances
        df_filtered['balance'] = df_filtered['demurragedTotalBalance'].apply(self._convert_balance)
        #df_filtered = df_filtered[df_filtered['balance'] > 0]

        # Map addresses to IDs
        df_filtered['account_id'] = df_filtered['account'].map(self.address_to_id)
        df_filtered['truster_id'] = df_filtered['truster'].map(self.address_to_id)
        df_filtered['token_id'] = df_filtered['tokenAddress'].map(self.address_to_id)

        # Create intermediate nodes
        df_filtered['intermediate_node'] = df_filtered['account_id'] + '_' + df_filtered['token_id']

        # Create edges from account to intermediate node
        edges1 = df_filtered[['account_id', 'intermediate_node', 'token_id', 'balance']]#.drop_duplicates()
        edges1.columns = ['from', 'to', 'token', 'capacity']

        # Create edges from intermediate node to truster
        edges2 = df_filtered[['intermediate_node', 'truster_id', 'token_id', 'balance']]#.drop_duplicates()
        edges2.columns = ['from', 'to', 'token', 'capacity']

        # Combine all edges
        all_edges = pd.concat([edges1, edges2])

        # Check for any remaining duplicates
        duplicates = all_edges[all_edges.duplicated(['from', 'to', 'token','capacity'], keep=False)]
        if not duplicates.empty:
            print("Warning: Duplicates found after edge creation:")
            print(duplicates.to_string())

        # Create the final edge list, capacities, and tokens
        edges = list(zip(all_edges['from'], all_edges['to']))
        capacities = all_edges['capacity'].tolist()
        tokens = all_edges['token'].tolist()

        print(f"Created {len(edges)} unique edges")

        # Debug: Check for duplicates in the created edges
        edge_set = set()
        for (u, v), token in zip(edges, tokens):
            edge_key = (u, v, token)
            if edge_key in edge_set:
                print(f"Duplicate edge found in initial data: {edge_key}")
            else:
                edge_set.add(edge_key)

        return edges, capacities, tokens
    
    def _create_edge_data2(self) -> Tuple[List[Tuple[str, str]], List[int], List[str]]:
        # Create a mapping from truster to their unique trustees (including themselves)
        truster_to_trustees = self.df_trusts.groupby('truster')['trustee'].apply(list).to_dict()
        for truster in self.df_trusts['truster'].unique():
            truster_to_trustees.setdefault(truster, []).append(truster)

        # Convert the mapping into a DataFrame
        df_truster_trustees = pd.DataFrame([
            {'truster': truster, 'unique_trustee': trustee}
            for truster, trustees in truster_to_trustees.items()
            for trustee in trustees
        ])

        # Merge with balances where tokenAddress matches unique_trustee
        df_merged = df_truster_trustees.merge(
            self.df_balances, left_on='unique_trustee', right_on='tokenAddress'
        )

        # Filter out rows where account is the same as truster
        df_filtered = df_merged[df_merged['account'] != df_merged['truster']].copy()

        # Convert balances and filter out non-positive balances
        df_filtered['balance'] = df_filtered['demurragedTotalBalance'].apply(self._convert_balance)
        #df_filtered = df_filtered[df_filtered['balance'] > 0]

        # Map addresses to IDs
        df_filtered['account_id'] = df_filtered['account'].map(self.address_to_id)
        df_filtered['truster_id'] = df_filtered['truster'].map(self.address_to_id)
        df_filtered['token_id'] = df_filtered['tokenAddress'].map(self.address_to_id)
        df_filtered['intermediate_node'] = df_filtered['account_id'] + '_' + df_filtered['token_id']

        # Create edges
        edges = list(zip(df_filtered['account_id'], df_filtered['intermediate_node'])) + \
                list(zip(df_filtered['intermediate_node'], df_filtered['truster_id']))

        # Duplicate capacities and tokens for each edge
        capacities = df_filtered['balance'].tolist() * 2
        tokens = df_filtered['token_id'].tolist() * 2

        # Debug: Check for duplicates in the created edges
        edge_set = set()
        for (u, v), token in zip(edges, tokens):
            edge_key = (u, v, token)
            if edge_key in edge_set:
                print(f"Duplicate edge found in initial data: {edge_key}")
            else:
                edge_set.add(edge_key)

        return edges, capacities, tokens

    """
    @staticmethod    
    def _convert_balance(balance_str):
        if pd.isna(balance_str):
            return 0
        try:
            return int(int(balance_str) / 10**14)
        except ValueError:
            print(f"Warning: Unable to convert balance: {balance_str}")
            return 0
        
    @staticmethod  
    def _convert_balance2(balance_str: str) -> int:
        #return int(balance_str)
        return int(int(balance_str)/10**14)

    def get_id_for_address(self, address: str) -> str:
        return self.address_to_id.get(address.lower())

    def get_address_for_id(self, id: str) -> str:
        return self.id_to_address.get(id)
