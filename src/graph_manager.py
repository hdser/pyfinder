import pandas as pd
from typing import Dict, Union, Tuple
import networkx as nx
from .data_ingestion import DataIngestion, PostgresDataIngestion
from .graph import NetworkXGraph, GraphToolGraph, GraphCreator
from .flow_analysis import NetworkFlowAnalysis
from .visualization import Visualization
import random

class GraphManager:
    def __init__(self, data_source: Union[Tuple[str, str], Tuple[Dict[str, str], str]], graph_type: str = 'networkx'):
        """
        Initialize the GraphManager with either CSV files or PostgreSQL connection.
        
        Args:
            data_source: Either:
                - Tuple[str, str]: (trusts_file_path, balances_file_path) for CSV ingestion
                - Tuple[Dict[str, str], str]: (db_config, queries_dir) for PostgreSQL ingestion
            graph_type: Type of graph to create ('networkx' or 'graph_tool')
        """
        self.data_ingestion = self._initialize_data_ingestion(data_source)
        
        self.graph = GraphCreator.create_graph(
            graph_type, 
            self.data_ingestion.edges, 
            self.data_ingestion.capacities, 
            self.data_ingestion.tokens
        )
        
        self.flow_analysis = NetworkFlowAnalysis(self.graph)
        self.visualization = Visualization()

    def _initialize_data_ingestion(self, data_source):
        """
        Initialize the appropriate data ingestion based on the data source type.
        """
        if not isinstance(data_source, tuple):
            raise ValueError("data_source must be a tuple")

        if len(data_source) != 2:
            raise ValueError("data_source must have exactly 2 elements")

        # Check if it's PostgreSQL configuration
        if isinstance(data_source[0], dict):
            db_config, queries_dir = data_source
            return PostgresDataIngestion(db_config, queries_dir)
        
        # Otherwise, assume it's CSV files
        elif isinstance(data_source[0], str) and isinstance(data_source[1], str):
            trusts_file, balances_file = data_source
            try:
                df_trusts = pd.read_csv(trusts_file)
                df_balances = pd.read_csv(balances_file)
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
        if isinstance(self.graph, NetworkXGraph):
            nodes = list(self.graph.g_nx.nodes())
        else:  # GraphToolGraph
            nodes = list(self.graph.id_to_vertex.keys())
        
        sample_nodes = random.sample(nodes, min(5, len(nodes)))
        sample_info = []
        for node in sample_nodes:
            if '_' in str(node):
                sample_info.append(f"Intermediate Node: {node}")
            else:
                address = self.data_ingestion.get_address_for_id(str(node))
                sample_info.append(f"Node ID: {node}, Address: {address}")
        return f"Total nodes: {len(nodes)}\nSample nodes:\n" + "\n".join(sample_info)
