import pandas as pd
from decimal import Decimal
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

    def _create_edge_data(self) -> Tuple[List[Tuple[str, str]], List[Decimal], List[str]]:
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
        df_filtered = df_filtered[df_filtered['balance'] > 0]

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

        return edges, capacities, tokens

    @staticmethod
    def _convert_balance(balance_str: str) -> Decimal:
        return Decimal(balance_str)

    def get_id_for_address(self, address: str) -> str:
        return self.address_to_id.get(address.lower())

    def get_address_for_id(self, id: str) -> str:
        return self.id_to_address.get(id)
