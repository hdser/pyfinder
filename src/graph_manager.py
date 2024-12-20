import pandas as pd
from typing import Dict, Union, Tuple, Callable, List, Optional
import random
import time

from src.data_ingestion import DataIngestion, PostgresDataIngestion
from src.graph import GraphCreator, NetworkXGraph, GraphToolGraph, ORToolsGraph, NetworkFlowAnalysis
from src.visualization import Visualization

class GraphManager:
    """
    Manages graph creation, analysis, and visualization for network flow problems.
    
    Coordinates between different components:
    - Data ingestion (CSV or PostgreSQL)
    - Graph implementation (NetworkX, graph-tool, or OR-Tools)
    - Flow analysis
    - Visualization
    """

    def __init__(self, data_source: Union[Tuple[str, str], Tuple[Dict[str, str], str]], 
                 graph_type: str = 'networkx'):
        """Initialize GraphManager with data source and graph implementation."""
        start = time.time()
        self.data_ingestion = self._initialize_data_ingestion(data_source)
        print("Ingestion time: ", time.time()-start)
        
        start = time.time()
        self.graph = GraphCreator.create_graph(
            graph_type, 
            self.data_ingestion.edges, 
            self.data_ingestion.capacities, 
            self.data_ingestion.tokens
        )
        print("Graph Creation time: ", time.time()-start)
        
        self.flow_analysis = NetworkFlowAnalysis(self.graph)
        self.visualization = Visualization()

    def _initialize_data_ingestion(self, data_source):
        """Initialize the appropriate data ingestion based on the data source type."""
        if not isinstance(data_source, tuple) or len(data_source) != 2:
            raise ValueError("data_source must be a tuple with exactly 2 elements")

        if isinstance(data_source[0], dict):
            db_config, queries_dir = data_source
            return PostgresDataIngestion(db_config, queries_dir)
        
        elif isinstance(data_source[0], str) and isinstance(data_source[1], str):
            trusts_file, balances_file = data_source
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
                return DataIngestion(df_trusts, df_balances)
            except Exception as e:
                raise ValueError(f"Error reading CSV files: {str(e)}")
        
        else:
            raise ValueError("Invalid data source format")

    def analyze_flow(self, source: str, sink: str, flow_func=None, cutoff: str = None):
        """Analyze flow between source and sink nodes."""
        source_id = self.data_ingestion.get_id_for_address(source)
        sink_id = self.data_ingestion.get_id_for_address(sink)
        
        if source_id is None or sink_id is None:
            raise ValueError(f"Source address '{source}' or sink address '{sink}' not found in the graph.")
        
        if not self.graph.has_vertex(source_id) or not self.graph.has_vertex(sink_id):
            raise ValueError(f"Source node '{source_id}' or sink node '{sink_id}' not in graph.")
        
        return self.flow_analysis.analyze_flow(source_id, sink_id, flow_func, cutoff)

    def visualize_flow(self, simplified_paths, simplified_edge_flows, original_edge_flows, output_dir: str):
        """Visualize flow paths."""
        self.visualization.ensure_output_directory(output_dir)
        
        self.visualization.plot_flow_paths(
            self.graph,
            simplified_paths, 
            simplified_edge_flows, 
            self.data_ingestion.id_to_address, 
            f"{output_dir}/simplified_flow_paths.png"
        )
        
        self.visualization.plot_full_flow_paths(
            self.graph,
            original_edge_flows, 
            self.data_ingestion.id_to_address, 
            f"{output_dir}/full_flow_paths.png"
        )

    def get_node_info(self):
        """Get information about nodes in the graph."""
        try:
            total_nodes = self.graph.num_vertices()
            total_edges = self.graph.num_edges()
            
            # Sample some nodes for display
            sample_nodes = random.sample(
                list(self.graph.get_vertices()), 
                min(5, self.graph.num_vertices())
            )
            sample_info = []
            
            for node in sample_nodes:
                if '_' in str(node):
                    sample_info.append(f"Intermediate Node: {node}")
                else:
                    address = self.data_ingestion.get_address_for_id(str(node))
                    sample_info.append(f"Node ID: {node}, Address: {address}")
                    
            return (f"Total nodes: {total_nodes}\n"
                    f"Total edges: {total_edges}\n"
                    f"Sample nodes:\n" + 
                    "\n".join(sample_info))
                    
        except Exception as e:
            return f"Error getting node info: {str(e)}"
        

    def analyze_arbitrage(self, source: str, start_token: str, end_token: str,
                         flow_func: Optional[Callable] = None) -> Tuple[int, List, Dict, Dict]:
        """
        Find arbitrage opportunities using max flow.
        
        Args:
            source: Address of source/target node
            start_token: Address of token to start with
            end_token: Address of token to end with
            flow_func: Optional flow algorithm to use
            
        Returns:
            Tuple containing:
            - Total flow value
            - Simplified paths
            - Simplified edge flows
            - Original edge flows
        """
        # Convert addresses to internal IDs
        source_id = self.data_ingestion.get_id_for_address(source)
        start_token_id = self.data_ingestion.get_id_for_address(start_token)
        end_token_id = self.data_ingestion.get_id_for_address(end_token)
        
        if not all([source_id, start_token_id, end_token_id]):
            raise ValueError("Invalid addresses provided")
            
        # Run arbitrage analysis
        results = self.flow_analysis.analyze_arbitrage(
            source_id,
            start_token_id,
            end_token_id,
            flow_func
        )
        
        return results