import pandas as pd
from typing import Callable, Optional
from src.data_ingestion import DataIngestion
from src.graph_creation import NetworkXGraph
from src.flow_analysis import NetworkFlowAnalysis
from src.visualization import Visualization

class GraphManager:
    def __init__(self, trusts_file: str, balances_file: str):
        df_trusts = pd.read_csv(trusts_file)
        df_balances = pd.read_csv(balances_file)
        
        self.data_ingestion = DataIngestion(df_trusts, df_balances)
        self.graph = NetworkXGraph(self.data_ingestion.edges, self.data_ingestion.capacities, self.data_ingestion.tokens)
        self.flow_analysis = NetworkFlowAnalysis(df_trusts, df_balances)
        self.visualization = Visualization()

    def analyze_flow(self, source: str, sink: str, flow_func: Optional[Callable] = None, cutoff: Optional[float] = None):
        return self.flow_analysis.analyze_flow(source, sink, flow_func, cutoff)

    def visualize_flow(self, simplified_paths, simplified_edge_flows, original_edge_flows, output_dir: str):
        self.visualization.ensure_output_directory(output_dir)
        self.visualization.plot_full_graph(self.graph.g_nx, f"{output_dir}/full_graph.png")
        self.visualization.plot_flow_paths(self.graph.g_nx, simplified_paths, simplified_edge_flows, f"{output_dir}/simplified_flow_paths.png")
        self.flow_analysis.visualize_full_flow_paths(original_edge_flows, f"{output_dir}/full_flow_paths.png")