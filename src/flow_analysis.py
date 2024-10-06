from typing import List, Tuple, Dict, Callable, Optional
import pandas as pd
import networkx as nx
from decimal import Decimal
from src.data_ingestion import DataIngestion
from src.graph_creation import NetworkXGraph
from src.visualization import Visualization

class NetworkFlowAnalysis:
    def __init__(self, df_trusts: pd.DataFrame, df_balances: pd.DataFrame):
        self.data_ingestion = DataIngestion(df_trusts, df_balances)
        self.graph = NetworkXGraph(self.data_ingestion.edges, self.data_ingestion.capacities, self.data_ingestion.tokens)
        self.visualization = Visualization()

    def analyze_flow(self, source: str, sink: str, flow_func: Optional[Callable] = None, requested_flow: Optional[str] = None):
        flow_value, flow_dict = self.graph.compute_flow(source, sink, flow_func, requested_flow)
        paths, edge_flows = self.graph.flow_decomposition(flow_dict, source, sink)
        
        simplified_graph, simplified_edge_flows = self.simplify_graph(self.graph.g_nx, edge_flows)
        simplified_paths = self.graph.simplified_flow_decomposition(paths)
        
        aggregated_flows = self.aggregate_transfers(simplified_paths, simplified_edge_flows)
        
        return flow_value, simplified_paths, simplified_edge_flows, edge_flows, aggregated_flows
    
    def simplify_graph(self, graph: nx.DiGraph, edge_flows: Dict[Tuple[str, str], float]) -> Tuple[nx.MultiDiGraph, Dict[Tuple[str, str], List[Dict[str, float]]]]:
        simplified_graph = nx.MultiDiGraph()
        simplified_edge_flows = {}

        for (u, v), flow in edge_flows.items():
            token = graph[u][v]['label']
            
            # Remove the '_' from nodes
            real_u = u.split('_')[0]
            real_v = v.split('_')[0]
            
            # Add edge to simplified graph
            if real_u != real_v:  # Avoid self-loops
                simplified_graph.add_edge(real_u, real_v, flow=flow, token=token)
                
                # Update edge flows
                if (real_u, real_v) not in simplified_edge_flows:
                    simplified_edge_flows[(real_u, real_v)] = []
                simplified_edge_flows[(real_u, real_v)].append({'flow': flow, 'token': token})

        return simplified_graph, simplified_edge_flows

    def aggregate_transfers(self, simplified_paths: List[Tuple[List[str], List[str], float]], 
                            simplified_edge_flows: Dict[Tuple[str, str], List[Dict[str, float]]]) -> Dict[Tuple[str, str, str], Decimal]:
        aggregated_flows = {}
        for path, _, _ in simplified_paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if (u, v) in simplified_edge_flows:
                    for flow_data in simplified_edge_flows[(u, v)]:
                        flow = Decimal(flow_data['flow'])
                        token = flow_data['token']
                        key = (u, v, token)
                        if key not in aggregated_flows:
                            aggregated_flows[key] = Decimal('0')
                        aggregated_flows[key] += flow
        return aggregated_flows

    def visualize_full_graph(self):
        self.visualization.plot_full_graph(self.graph.g_nx)

    def visualize_flow_paths(self, paths: List[Tuple[List[str], List[str], float]], edge_flows: Dict[Tuple[str, str], List[Dict[str, float]]]):
        self.visualization.plot_flow_paths(self.graph.g_nx, paths, edge_flows)

    def visualize_full_flow_paths(self, edge_flows: Dict[Tuple[str, str], float], filename: str):
        self.visualization.plot_full_flow_paths(self.graph.g_nx, edge_flows, filename)