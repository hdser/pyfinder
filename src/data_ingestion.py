import pandas as pd
from decimal import Decimal
from typing import List, Tuple

class DataIngestion:
    def __init__(self, df_trusts: pd.DataFrame, df_balances: pd.DataFrame):
        self.df_trusts = df_trusts
        self.df_balances = df_balances
        self.unique_df = self._create_unique_df()
        self.edges, self.capacities, self.tokens = self._create_edge_data()

    def _create_unique_df(self) -> pd.DataFrame:
        unique_addresses = pd.concat([self.df_trusts['trustee'], self.df_trusts['truster'], self.df_balances['account']]).drop_duplicates().reset_index(drop=True)
        unique_df = pd.DataFrame(unique_addresses, columns=['address']).reset_index()
        unique_df.rename(columns={'index': 'unique_id'}, inplace=True)
        return unique_df

    def _create_edge_data(self) -> Tuple[List[Tuple[str, str]], List[float], List[str]]:
        edges = []
        capacities = []
        tokens = []
        unique_trusters = self.df_trusts['truster'].unique()

        for truster in unique_trusters:
            unique_trustees = list(self.df_trusts[self.df_trusts['truster']==truster]['trustee'].unique())
            unique_trustees.append(truster)

            has_account_tokens = (self.df_balances['tokenAddress'].isin(unique_trustees))
            not_truster = (self.df_balances['account']!=truster)
            df_trusted_token_balances = self.df_balances[has_account_tokens & not_truster]

            for _, (demurragedTotalBalance, account, tokenAddress) in df_trusted_token_balances.iterrows():
                balance = self._convert_balance(demurragedTotalBalance)
                if float(balance) > 0:
                    account_id = str(self.unique_df[self.unique_df['address']==account]['unique_id'].values[0])
                    truster_id = str(self.unique_df[self.unique_df['address']==truster]['unique_id'].values[0])
                    token_id = str(self.unique_df[self.unique_df['address']==tokenAddress]['unique_id'].values[0])
                    edges.append((account_id, account_id+'_'+token_id))
                    edges.append((account_id+'_'+token_id, truster_id))
                    capacities.extend([balance, balance])
                    tokens.extend([token_id, token_id])

        return edges, capacities, tokens

    @staticmethod
    def _convert_balance(balance_str: str) -> int:
        return Decimal(balance_str)
#        return int(Decimal(balance_str)/10**18)
