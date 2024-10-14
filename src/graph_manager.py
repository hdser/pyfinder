import pandas as pd
from typing import Callable, Optional
import networkx as nx
from .data_ingestion import DataIngestion
from .graph_creation import NetworkXGraph, GraphToolGraph, GraphCreator
from .flow_analysis import NetworkFlowAnalysis
from .visualization import Visualization

class GraphManager:
    def __init__(self, trusts_file: str, balances_file: str, graph_type: str = 'networkx'):
        df_trusts = pd.read_csv(trusts_file)
        df_balances = pd.read_csv(balances_file)
        
        self.data_ingestion = DataIngestion(df_trusts, df_balances)
        
        self.graph = GraphCreator.create_graph(graph_type, self.data_ingestion.edges, self.data_ingestion.capacities, self.data_ingestion.tokens)
        
        self.flow_analysis = NetworkFlowAnalysis(self.graph)
        self.visualization = Visualization()

    def analyze_flow(self, source: str, sink: str, flow_func: Optional[Callable] = None, cutoff: Optional[float] = None):
        source_id = self.data_ingestion.get_id_for_address(source)
        sink_id = self.data_ingestion.get_id_for_address(sink)
        
        if source_id is None or sink_id is None:
            raise ValueError(f"Source address '{source}' or sink address '{sink}' not found in the graph.")
        
        if not self.graph.has_vertex(source_id) or not self.graph.has_vertex(sink_id):
            raise ValueError(f"Source node '{source_id}' or sink node '{sink_id}' not in graph.")
        
        return self.flow_analysis.analyze_flow(source_id, sink_id, flow_func, cutoff)

    def visualize_flow(self, simplified_paths, simplified_edge_flows, original_edge_flows, output_dir: str):
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