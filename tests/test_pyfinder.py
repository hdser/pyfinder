import unittest
import pandas as pd
import networkx as nx
from src.graph_manager import GraphManager
from src.data_ingestion import DataIngestion
from networkx.algorithms.flow import boykov_kolmogorov
import random
import os
import json
from collections import defaultdict

class TestFlowAlgorithm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the data and create the GraphManager
        trusts_file = 'data/data-trust.csv'
        balances_file = 'data/data-balance.csv'
        cls.graph_manager = GraphManager(trusts_file, balances_file)
        
        # Create DataIngestion object for direct access to trust and balance data
        df_trusts = pd.read_csv(trusts_file)
        df_balances = pd.read_csv(balances_file)
        cls.data_ingestion = DataIngestion(df_trusts, df_balances)

        # Ensure output directory exists
        cls.output_dir = 'test_output'
        os.makedirs(cls.output_dir, exist_ok=True)

    def get_random_addresses(self, n=5):
        # Get all nodes from the graph
        all_nodes = list(self.graph_manager.graph.g_nx.nodes())
        
        # Filter out intermediate nodes (those containing '_')
        real_nodes = [node for node in all_nodes if '_' not in node]
        
        # Convert node IDs to addresses
        addresses = [self.graph_manager.data_ingestion.get_address_for_id(node) for node in real_nodes]
        
        # Remove any None values (in case some IDs don't have corresponding addresses)
        addresses = [addr for addr in addresses if addr is not None]
        
        # Return random sample
        return random.sample(addresses, min(n, len(addresses)))

    def test_multiple_flows(self):
        addresses = self.get_random_addresses(4) 
        test_results = []

        for i in range(len(addresses) - 1):
            source = addresses[i]
            sink = addresses[i + 1]
            
            try:
                result = self.analyze_and_verify_flow(source, sink)
                test_results.append(result)
            except (ValueError, nx.NetworkXError) as e:
                print(f"Skipping pair ({source}, {sink}): {str(e)}")
                continue

        # Write test results to a file
        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)

        self.assertGreater(len(test_results), 0, "No valid flow analyses were performed")


    def analyze_and_verify_flow(self, source, sink):
        result = {
            'source': source,
            'sink': sink,
            'transfers': [],
            'balance_changes': defaultdict(lambda: defaultdict(lambda: {'initial': 0, 'final': 0, 'change': 0})),
            'trust_violations': []
        }

        try:
            # Run the flow analysis
            flow_value, simplified_paths, simplified_edge_flows, original_edge_flows = self.graph_manager.analyze_flow(source, sink, boykov_kolmogorov)
            
            result['flow_value'] = flow_value

            # Initialize balance changes with initial balances
            for (sender, receiver), token_flows in simplified_edge_flows.items():
                for address in [sender, receiver]:
                    address_str = self.graph_manager.data_ingestion.get_address_for_id(address)
                    for token, _ in token_flows.items():
                        token_address = self.graph_manager.data_ingestion.get_address_for_id(token)
                        initial_balance = self.get_balance(address_str, token_address)
                        result['balance_changes'][address_str][token_address]['initial'] = initial_balance
                        result['balance_changes'][address_str][token_address]['final'] = initial_balance

            # Process transfers and update balances
            for (sender, receiver), token_flows in simplified_edge_flows.items():
                sender_address = self.graph_manager.data_ingestion.get_address_for_id(sender)
                receiver_address = self.graph_manager.data_ingestion.get_address_for_id(receiver)
                
                for token, flow in token_flows.items():
                    token_address = self.graph_manager.data_ingestion.get_address_for_id(token)
                    
                    # Record transfer
                    transfer = {
                        'sender': sender_address,
                        'receiver': receiver_address,
                        'token': token_address,
                        'amount': flow
                    }
                    result['transfers'].append(transfer)

                    # Update balances
                    result['balance_changes'][sender_address][token_address]['final'] -= flow
                    result['balance_changes'][sender_address][token_address]['change'] -= flow
                    result['balance_changes'][receiver_address][token_address]['final'] += flow
                    result['balance_changes'][receiver_address][token_address]['change'] += flow

                    # Check trust
                    trust_exists = self.data_ingestion.df_trusts[
                        (self.data_ingestion.df_trusts['truster'] == receiver_address) &
                        (self.data_ingestion.df_trusts['trustee'] == token_address)
                    ].any().any()
                    
                    if not trust_exists:
                        result['trust_violations'].append(f"Receiver {receiver_address} does not trust token {token_address}")

            # Verify that no balance went negative
            for address, tokens in result['balance_changes'].items():
                for token, balances in tokens.items():
                    if balances['final'] < 0:
                        raise ValueError(f"Negative balance for address {address} and token {token}")

            # Remove entries with no changes
            result['balance_changes'] = {
                address: {
                    token: balances for token, balances in tokens.items() if balances['change'] != 0
                }
                for address, tokens in result['balance_changes'].items()
            }
            result['balance_changes'] = {
                address: tokens for address, tokens in result['balance_changes'].items() if tokens
            }

        except Exception as e:
            result['error'] = str(e)

        return result

    def get_balance(self, address, token):
        balance_records = self.data_ingestion.df_balances[
            (self.data_ingestion.df_balances['account'] == address) &
            (self.data_ingestion.df_balances['tokenAddress'] == token)
        ]
        if balance_records.empty:
            return 0
        else:
            balance = balance_records['demurragedTotalBalance'].iloc[0]
            return self.data_ingestion._convert_balance(balance)

if __name__ == '__main__':
    unittest.main()