import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import pytest
import pandas as pd
import numpy as np
from graph_tool.flow import push_relabel_max_flow
from typing import Dict, List, Tuple

from src.graph import NetworkXGraph, GraphToolGraph, ORToolsGraph, NetworkFlowAnalysis
from src.graph_manager import GraphManager
from src.data_ingestion import DataIngestion

class CirclesTestData:
    """
    Test data generator that mimics real Circles UBI token system structure.
    
    In Circles, each user has their own personal token and can trust other users' tokens.
    Key concepts modeled here:
    1. Personal token creation (each account has its own token with same address)
    2. Trust relationships (who accepts whose tokens)
    3. Token balances (how much of each token type accounts hold)
    4. Time-based token issuance (token amounts grow over time)
    """
    
    def create_test_network(self, num_users: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a test network with specified number of users.
        
        The network includes:
        - Multiple users with their own tokens
        - Various trust relationships between users
        - Different token balances for each user
        - Realistic token amounts (in wei, 18 decimals)
        
        Returns:
            Tuple of (trust_df, balance_df) matching real data format
        """
        # Generate user addresses (0xaaa1, 0xaaa2, etc.)
        users = [f"0xaaa{i}" for i in range(1, num_users + 1)]
        
        # Create trust relationships
        trust_relationships = []
        for i, truster in enumerate(users):
            # Each user trusts some others (but not all, to create interesting paths)
            for j, trustee in enumerate(users):
                if i != j and (i + j) % 2 == 0:  # Create selective trust pattern
                    trust_relationships.append({
                        'truster': truster,
                        'trustee': trustee
                    })
        
        # Create token balances
        token_balances = []
        for user in users:
            # Each user has their personal token
            personal_token_balance = {
                'account': user,
                'tokenAddress': user,  # Personal token has same address as user
                'demurragedTotalBalance': 1000 * (10**18)  # 1000 tokens in wei
            }
            token_balances.append(personal_token_balance)
            
            # Users also hold some other users' tokens
            for other_user in users:
                if other_user != user:
                    held_balance = np.random.randint(100, 500) * (10**18)
                    token_balances.append({
                        'account': user,
                        'tokenAddress': other_user,  # Token of other user
                        'demurragedTotalBalance': held_balance
                    })
        
        return (
            pd.DataFrame(trust_relationships),
            pd.DataFrame(token_balances)
        )

    def get_expected_graph_structure(self, trust_df: pd.DataFrame, balance_df: pd.DataFrame) -> Dict:
        """
        Calculate expected graph structure from test data.
        
        Maps out:
        - All vertices (users and intermediate token nodes)
        - All edges (holdings and trust relationships)
        - Expected capacities for each edge
        - Token assignments for each edge
        
        Returns dictionary with complete graph specifications.
        """
        vertices = set()
        edges = []
        
        # Add edges for token holdings (account -> intermediate node)
        for _, row in balance_df.iterrows():
            account = row['account']
            token = row['tokenAddress']
            balance = row['demurragedTotalBalance'] // (10**15)  # Convert to mCRC
            
            # Create intermediate node for this token holding
            intermediate_node = f"{account}_{token}"
            vertices.add(account)
            vertices.add(intermediate_node)
            
            # Add the holding edge
            edges.append({
                'from': account,
                'to': intermediate_node,
                'capacity': balance,
                'token': token
            })
        
        # Add edges for trust relationships (intermediate node -> trusting account)
        for _, row in trust_df.iterrows():
            truster = row['truster']
            trustee = row['trustee']
            
            # For each trust relationship, we need edges from all intermediate
            # nodes holding the trusted token to the trusting account
            trusted_holdings = balance_df[
                balance_df['tokenAddress'] == trustee
            ]
            
            for _, holding in trusted_holdings.iterrows():
                holder = holding['account']
                balance = holding['demurragedTotalBalance'] // (10**15)
                intermediate_node = f"{holder}_{trustee}"
                
                edges.append({
                    'from': intermediate_node,
                    'to': truster,
                    'capacity': balance,
                    'token': trustee
                })
                
                vertices.add(truster)
                vertices.add(intermediate_node)
        
        return {
            'vertices': sorted(list(vertices)),
            'edges': edges
        }

class TestGraphConstruction(unittest.TestCase):
    """
    Test suite for verifying correct graph construction across all implementations.
    
    Tests that each graph implementation:
    1. Creates correct vertices for accounts and intermediate nodes
    2. Creates correct edges for token holdings and trust relationships
    3. Assigns correct capacities based on token balances
    4. Maintains proper token tracking through the graph
    """
    
    def setUp(self):
        """Initialize test environment with realistic Circles data."""
        # Create test network
        self.test_data = CirclesTestData()
        self.trust_df, self.balance_df = self.test_data.create_test_network()
        
        # Process through real DataIngestion
        self.data_ingestion = DataIngestion(self.trust_df, self.balance_df)
        
        # Initialize all implementations
        self.graphs = {
            'NetworkX': NetworkXGraph(
                self.data_ingestion.edges,
                self.data_ingestion.capacities,
                self.data_ingestion.tokens
            ),
            'graph-tool': GraphToolGraph(
                self.data_ingestion.edges,
                self.data_ingestion.capacities,
                self.data_ingestion.tokens
            ),
            'OR-Tools': ORToolsGraph(
                self.data_ingestion.edges,
                self.data_ingestion.capacities,
                self.data_ingestion.tokens
            )
        }
        
        # Get expected graph structure
        self.expected_structure = self.test_data.get_expected_graph_structure(
            self.trust_df, 
            self.balance_df
        )

    def test_vertex_creation(self):
        """
        Test that all implementations create correct vertices.
        
        Verifies:
        - All user accounts are present
        - All intermediate nodes for token holdings exist
        - No unexpected vertices are created
        """
        expected_vertices = set(self.expected_structure['vertices'])
        
        for impl_name, graph in self.graphs.items():
            actual_vertices = set(graph.get_vertices())
            
            # Check all expected vertices exist
            missing_vertices = expected_vertices - actual_vertices
            self.assertEqual(
                len(missing_vertices), 
                0,
                f"{impl_name}: Missing vertices: {missing_vertices}"
            )
            
            # Check no extra vertices exist
            extra_vertices = actual_vertices - expected_vertices
            self.assertEqual(
                len(extra_vertices), 
                0,
                f"{impl_name}: Unexpected vertices: {extra_vertices}"
            )

    def test_edge_creation(self):
        """
        Test that all implementations create correct edges with proper capacities.
        
        Verifies:
        - All token holding edges exist
        - All trust relationship edges exist
        - Edge capacities match token balances
        - Token tracking is maintained
        """
        for impl_name, graph in self.graphs.items():
            for edge in self.expected_structure['edges']:
                # Check edge exists
                self.assertTrue(
                    graph.has_edge(edge['from'], edge['to']),
                    f"{impl_name}: Missing edge {edge['from']} -> {edge['to']}"
                )
                
                # Verify capacity
                capacity = graph.get_edge_capacity(edge['from'], edge['to'])
                self.assertEqual(
                    capacity,
                    edge['capacity'],
                    f"{impl_name}: Wrong capacity for {edge['from']} -> {edge['to']}"
                )
                
                # Verify token assignment
                edge_data = graph.get_edge_data(edge['from'], edge['to'])
                self.assertEqual(
                    edge_data.get('label', edge_data.get('token')),
                    edge['token'],
                    f"{impl_name}: Wrong token for {edge['from']} -> {edge['to']}"
                )

    def test_flow_paths(self):
        """
        Test that all implementations find correct flow paths.
        
        Verifies:
        - Maximum flow computation is consistent
        - All valid paths are found
        - Path capacities are correct
        - Token flows are properly tracked
        """
        # Create flow analysis for each implementation
        analyzers = {
            name: NetworkFlowAnalysis(graph)
            for name, graph in self.graphs.items()
        }
        
        # Test flows between each pair of users
        users = self.trust_df['truster'].unique()
        for source in users:
            for sink in users:
                if source != sink:
                    flows = {}
                    paths = {}
                    
                    # Compute flow with each implementation
                    for impl_name, analyzer in analyzers.items():
                        flows[impl_name], paths[impl_name], _, _ = analyzer.analyze_flow(
                            source, sink
                        )
                    
                    # Verify consistent flow values
                    flow_values = set(flows.values())
                    self.assertEqual(
                        len(flow_values),
                        1,
                        f"Inconsistent flow values for {source}->{sink}: {flows}"
                    )
                    
                    # Verify path validity
                    for impl_name, path_list in paths.items():
                        for path, tokens, amount in path_list:
                            # Check path starts and ends correctly
                            self.assertEqual(path[0], source)
                            self.assertEqual(path[-1], sink)
                            
                            # Verify intermediate nodes match tokens
                            for node, token in zip(path[1:-1], tokens):
                                if '_' in node:
                                    node_token = node.split('_')[1]
                                    self.assertEqual(
                                        node_token,
                                        token,
                                        f"{impl_name}: Token mismatch in path"
                                    )

class TestEndToEndFlow(unittest.TestCase):
    """
    End-to-end testing of the complete flow analysis system.
    
    Tests:
    1. Real data processing pipeline
    2. Graph construction from processed data
    3. Flow computation through the graph
    4. Result interpretation and validation
    """
    
    def setUp(self):
        """Set up test environment with proper DataFrame mocking."""
        # Create test data
        test_data = CirclesTestData()
        self.trust_df, self.balance_df = test_data.create_test_network()
        
        # Set up the mock to properly handle multiple calls
        mock_csv = Mock()
        mock_csv.side_effect = [
            self.trust_df.copy(),  # First call returns trust data
            self.balance_df.copy()  # Second call returns balance data
        ]
        
        # Create the patch
        self.csv_patcher = patch('pandas.read_csv', mock_csv)
        self.csv_patcher.start()
        
        # Initialize managers
        self.manager = GraphManager(
            data_source=("trust.csv", "balance.csv"),
            graph_type="networkx"
        )
        
    def tearDown(self):
        """Clean up mocks after tests."""
        self.csv_patcher.stop()

    def test_flow_analysis(self):
        """
        Test complete flow analysis through each implementation.
        
        Verifies:
        1. Correct flow computation
        2. Path finding
        3. Token tracking
        4. Flow aggregation
        """
        # Test flow between each pair of users
        users = self.trust_df['truster'].unique()
        for source in users:
            for sink in users:
                if source != sink:
                    # Compute expected maximum flow
                    expected_flow = self._calculate_expected_flow(source, sink)
                    
                    # Test each implementation
                    for impl_name, manager in self.managers.items():
                        flow_value, paths, simple_flows, edge_flows = manager.analyze_flow(
                            source, sink
                        )
                        
                        # Verify total flow
                        self.assertEqual(
                            flow_value,
                            expected_flow,
                            f"{impl_name}: Wrong flow value for {source}->{sink}"
                        )
                        
                        # Verify token flows sum correctly
                        self._verify_token_flows(
                            paths, simple_flows, edge_flows,
                            source, sink, impl_name
                        )

    def _calculate_expected_flow(self, source: str, sink: str) -> int:
        """
        Calculate expected maximum flow between two users.
        
        Considers:
        1. Available token balances
        2. Trust relationships
        3. Indirect paths through other users
        """
        # Get direct trust relationship
        direct_trust = self.trust_df[
            (self.trust_df['truster'] == sink) &
            (self.trust_df['trustee'] == source)
        ]
        
        if direct_trust.empty:
            return 0
            
        # Get available token balance
        available_balance = self.balance_df[
            (self.balance_df['account'] == source) &
            (self.balance_df['tokenAddress'] == source)
        ]['demurragedTotalBalance'].iloc[0] // (10**15)
        
        return available_balance

    def _verify_token_flows(self, paths, simple_flows, edge_flows, 
                          source: str, sink: str, impl_name: str):
        """
        Verify that token flows are consistent and valid.
        
        Checks:
        1. Path flows sum to total flow
        2. Each path is valid (follows trust relationships)
        3. Token amounts respect balance constraints
        """
        # Sum flows across all paths
        total_path_flow = sum(amount for _, _, amount in paths)
        
        # Sum flows in simplified representation
        simple_total = sum(
            sum(flows.values())
            for (u, v), flows in simple_flows.items()
            if u == source and v == sink
        )
        
        # Verify consistency
        self.assertEqual(
            total_path_flow,
            simple_total,
            f"{impl_name}: Inconsistent flow summation for {source}->{sink}"
        )
        
        # Verify each path follows trust relationships
        for path, tokens, amount in paths:
            self._verify_path_validity(
                path, tokens, amount,
                source, sink, impl_name
            )

    def _verify_path_validity(self, path: List[str], tokens: List[str], 
                            amount: int, source: str, sink: str, impl_name: str):
        """
        Verify that a single flow path is valid.
        
        Checks:
        1. Path starts at source and ends at sink
        2. All intermediate steps follow trust relationships
        3. Token amounts don't exceed balances
        4. Intermediate nodes properly represent token holdings
        """
        # Verify path endpoints
        self.assertEqual(path[0], source, f"{impl_name}: Path doesn't start at source")
        self.assertEqual(path[-1], sink, f"{impl_name}: Path doesn't end at sink")
        
        # Verify each step in the path
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            if '_' in next_node:
                # This is a token holding edge
                holder, token = next_node.split('_')
                # Verify the holder has sufficient balance
                balance = self.balance_df[
                    (self.balance_df['account'] == holder) &
                    (self.balance_df['tokenAddress'] == token)
                ]['demurragedTotalBalance'].iloc[0] // (10**15)
                
                self.assertLessEqual(
                    amount, 
                    balance,
                    f"{impl_name}: Flow exceeds balance for {holder}'s {token}"
                )
            else:
                # This is a trust relationship edge
                # Find corresponding token from intermediate node
                intermediate = path[i]
                if '_' in intermediate:
                    _, token = intermediate.split('_')
                    # Verify trust relationship exists
                    trust_exists = not self.trust_df[
                        (self.trust_df['truster'] == next_node) &
                        (self.trust_df['trustee'] == token)
                    ].empty
                    
                    self.assertTrue(
                        trust_exists,
                        f"{impl_name}: Missing trust relationship: {next_node} -> {token}"
                    )

class TestPathFinding(unittest.TestCase):
    """
    Test suite specifically for path finding algorithms.
    Tests different network configurations and edge cases.
    """
    
    def setUp(self):
        """Create test networks with specific path configurations."""
        self.test_data = CirclesTestData()
        
        # Create different test scenarios
        self.scenarios = {
            'simple_path': self._create_simple_path_network(),
            'multiple_paths': self._create_multiple_paths_network(),
            'cyclic_paths': self._create_cyclic_network(),
            'no_path': self._create_disconnected_network()
        }

    def _create_simple_path_network(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create network with single clear path through trusted tokens.
        Must include both trust relationships and token holdings.
        """
        trust_data = pd.DataFrame({
            'truster': ['user2', 'user3'],
            'trustee': ['user1', 'user2']
        })
        
        balance_data = pd.DataFrame({
            'account': ['user1', 'user2', 'user2'],
            'tokenAddress': ['user1', 'user2', 'user1'],
            'demurragedTotalBalance': [
                1000 * (10**18),  # user1's own tokens
                1000 * (10**18),  # user2's own tokens
                500 * (10**18)    # user2's holding of user1's tokens
            ]
        })
        
        return trust_data, balance_data

    def _create_multiple_paths_network(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create network with multiple possible paths.
        Include all necessary token holdings for each path.
        """
        trust_data = pd.DataFrame({
            'truster': ['user2', 'user3', 'user2', 'user3'],
            'trustee': ['user1', 'user1', 'user4', 'user4']
        })
        
        balance_data = pd.DataFrame({
            'account': [
                'user1', 'user4',  # Own tokens
                'user2', 'user3',  # Holdings of user1's tokens
                'user2', 'user3'   # Holdings of user4's tokens
            ],
            'tokenAddress': [
                'user1', 'user4',  # Own tokens
                'user1', 'user1',  # user1's tokens held by others
                'user4', 'user4'   # user4's tokens held by others
            ],
            'demurragedTotalBalance': [
                1000 * (10**18),  # user1's balance
                1000 * (10**18),  # user4's balance
                500 * (10**18),   # user2's holding of user1's tokens
                500 * (10**18),   # user3's holding of user1's tokens
                300 * (10**18),   # user2's holding of user4's tokens
                300 * (10**18)    # user3's holding of user4's tokens
            ]
        })
        
        return trust_data, balance_data

    def _create_cyclic_network(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create network with cycles to test cycle handling."""
        trust_data = {
            'truster': ['user2', 'user3', 'user1'],
            'trustee': ['user1', 'user2', 'user3']
        }
        
        balance_data = {
            'account': ['user1', 'user2', 'user3'],
            'tokenAddress': ['user1', 'user2', 'user3'],
            'demurragedTotalBalance': [
                1000 * (10**18),
                1000 * (10**18),
                1000 * (10**18)
            ]
        }
        
        return pd.DataFrame(trust_data), pd.DataFrame(balance_data)

    def _create_disconnected_network(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create network with no valid paths between some nodes."""
        trust_data = {
            'truster': ['user2', 'user4'],
            'trustee': ['user1', 'user3']
        }
        
        balance_data = {
            'account': ['user1', 'user3'],
            'tokenAddress': ['user1', 'user3'],
            'demurragedTotalBalance': [
                1000 * (10**18),
                1000 * (10**18)
            ]
        }
        
        return pd.DataFrame(trust_data), pd.DataFrame(balance_data)

    def _initialize_implementation(self, trust_df: pd.DataFrame, balance_df: pd.DataFrame, impl_type: str) -> NetworkFlowAnalysis:
        """
        Initialize a graph implementation with proper data processing.
        
        The key is to process the DataFrames exactly as the real application does:
        1. Convert balances from wei to mCRC
        2. Create proper intermediate nodes for token holdings
        3. Add edges for both holdings and trust relationships
        """
        # First convert the balance amounts from wei to mCRC
        balance_df = balance_df.copy()
        balance_df['demurragedTotalBalance'] = balance_df['demurragedTotalBalance'].apply(
            lambda x: int(x) // (10**15)  # Convert wei to mCRC
        )
        
        # Create DataIngestion instance with processed data
        data_ingestion = DataIngestion(trust_df, balance_df)
        
        # Create appropriate graph implementation
        if impl_type == 'networkx':
            graph = NetworkXGraph(
                data_ingestion.edges,
                data_ingestion.capacities,
                data_ingestion.tokens
            )
        elif impl_type == 'graph_tool':
            graph = GraphToolGraph(
                data_ingestion.edges,
                data_ingestion.capacities,
                data_ingestion.tokens
            )
        else:  # ortools
            graph = ORToolsGraph(
                data_ingestion.edges,
                data_ingestion.capacities,
                data_ingestion.tokens
            )
            
        return NetworkFlowAnalysis(graph)

    def test_simple_path(self):
        """Test flow finding through simple linear path."""
        trust_df, balance_df = self.scenarios['simple_path']
        
        for impl_type in ['networkx', 'graph_tool', 'ortools']:
            analyzer = self._initialize_implementation(trust_df, balance_df, impl_type)
            
            # Test flow from first to last user
            flow_value, paths, _, _ = analyzer.analyze_flow('user1', 'user3')
            
            # Should find exactly one path
            self.assertEqual(
                len(paths), 
                1,
                f"{impl_type}: Wrong number of paths in simple network"
            )
            
            # Flow should equal minimum capacity along path
            self.assertEqual(
                flow_value,
                500,
                f"{impl_type}: Incorrect flow value in simple path"
            )

    def test_multiple_paths(self):
        """Test finding and utilizing multiple valid paths."""
        trust_df, balance_df = self.scenarios['multiple_paths']
        
        for impl_type in ['networkx', 'graph_tool', 'ortools']:
            analyzer = self._initialize_implementation(trust_df, balance_df, impl_type)
            
            # Test flow between nodes with multiple paths
            flow_value, paths, _, _ = analyzer.analyze_flow('user1', 'user3')
            
            # Should find multiple paths
            self.assertGreater(
                len(paths),
                1,
                f"{impl_type}: Failed to find multiple paths"
            )
            
            # Total flow should sum capacity of all paths
            total_path_flow = sum(amount for _, _, amount in paths)
            self.assertEqual(
                flow_value,
                total_path_flow,
                f"{impl_type}: Flow sum mismatch in multiple paths"
            )

    def test_cyclic_paths(self):
        """Test correct handling of cycles in the trust network."""
        trust_df, balance_df = self.scenarios['cyclic_paths']
        
        for impl_type in ['networkx', 'graph_tool', 'ortools']:
            analyzer = self._initialize_implementation(trust_df, balance_df, impl_type)
            
            # Test flow in cyclic network
            flow_value, paths, _, _ = analyzer.analyze_flow('user1', 'user2')
            
            # Verify no infinite loops
            self.assertIsNotNone(flow_value)
            self.assertIsNotNone(paths)
            
            # Check path validity
            for path, _, _ in paths:
                # No node should appear twice in a path
                seen_nodes = set()
                for node in path:
                    self.assertNotIn(
                        node,
                        seen_nodes,
                        f"{impl_type}: Cycle detected in path"
                    )
                    seen_nodes.add(node)

    def test_no_path(self):
        """Test correct handling when no valid path exists."""
        trust_df, balance_df = self.scenarios['no_path']
        
        for impl_type in ['networkx', 'graph_tool', 'ortools']:
            analyzer = self._initialize_implementation(trust_df, balance_df, impl_type)
            
            # Test flow between disconnected nodes
            flow_value, paths, simple_flows, edge_flows = analyzer.analyze_flow(
                'user1', 
                'user4'
            )
            
            # Should find no flow
            self.assertEqual(
                flow_value,
                0,
                f"{impl_type}: Non-zero flow in disconnected network"
            )
            
            # Should have no paths
            self.assertEqual(
                len(paths),
                0,
                f"{impl_type}: Found paths in disconnected network"
            )
            
            # Should have no flows
            self.assertEqual(
                len(simple_flows),
                0,
                f"{impl_type}: Found flows in disconnected network"
            )

if __name__ == '__main__':
    unittest.main()