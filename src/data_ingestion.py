import pandas as pd
from decimal import Decimal
from typing import List, Tuple, Dict

class DataIngestion:
    def __init__(self, df_trusts: pd.DataFrame, df_balances: pd.DataFrame):
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
        
        address_to_id = {addr.lower(): str(idx) for idx, addr in enumerate(unique_addresses)}
        id_to_address = {str(idx): addr.lower() for idx, addr in enumerate(unique_addresses)}
        
        return address_to_id, id_to_address

    def _create_edge_data(self) -> Tuple[List[Tuple[str, str]], List[float], List[str]]:
        edges = []
        capacities = []
        tokens = []
        unique_trusters = self.df_trusts['truster'].unique()
        i = 0
        for truster in unique_trusters:
            print(len(unique_trusters) - i)
            i += 1
            unique_trustees = list(self.df_trusts[self.df_trusts['truster']==truster]['trustee'].unique())
            unique_trustees.append(truster)

            has_account_tokens = (self.df_balances['tokenAddress'].isin(unique_trustees))
            not_truster = (self.df_balances['account']!=truster)
            df_trusted_token_balances = self.df_balances[has_account_tokens & not_truster]

            for _, (demurragedTotalBalance, account, tokenAddress) in df_trusted_token_balances.iterrows():
                balance = self._convert_balance(demurragedTotalBalance)
                if float(balance) > 0:
                    account_id = self.address_to_id[account.lower()]
                    truster_id = self.address_to_id[truster.lower()]
                    token_id = self.address_to_id[tokenAddress.lower()]

                    intermediate_node = f"{account_id}_{token_id}"

                    edges.append((account_id, intermediate_node))
                    edges.append((intermediate_node, truster_id))
                    capacities.extend([balance, balance])
                    tokens.extend([token_id, token_id])

        return edges, capacities, tokens

    @staticmethod
    def _convert_balance(balance_str: str) -> int:
        return Decimal(balance_str)

    def get_id_for_address(self, address: str) -> str:
        return self.address_to_id.get(address.lower())

    def get_address_for_id(self, id: str) -> str:
        return self.id_to_address.get(id)