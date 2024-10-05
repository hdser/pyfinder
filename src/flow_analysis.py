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
        
        print(f"Flow Value: {flow_value}")
        print("\nFlow Decomposition Paths:")
        for idx, (path, labels, flow) in enumerate(paths, 1):
            path_str = " -> ".join(path)
            labels_str = " -> ".join(labels)
            print(f"Path {idx}: {path_str}")
            print(f"Labels: {labels_str}")
            print(f"Flow: {flow}\n")

        print("\nEdge Flows:")
        for (u, v), flow in edge_flows.items():
            print(f"Edge ({u}, {v}): Flow = {flow}")

        simplified_graph, simplified_edge_flows = self.simplify_graph(self.graph.g_nx, edge_flows)
        simplified_paths = self.graph.simplified_flow_decomposition(paths)
        
        print("\nSimplified Flow Decomposition Paths:")
        for idx, (path, labels, flow) in enumerate(simplified_paths, 1):
            path_str = " -> ".join(path)
            labels_str = " -> ".join(labels)
            print(f"Path {idx}: {path_str}")
            print(f"Labels: {labels_str}")
            print(f"Flow: {flow}\n")

        print("\nSimplified Edge Flows:")
        for (u, v), data_list in simplified_edge_flows.items():
            for data in data_list:
                print(f"Edge ({u}, {v}): Flow = {data['flow']}, Token = {data['token']}")

        return flow_value, simplified_paths, simplified_edge_flows, edge_flows
    
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

    def visualize_full_graph(self):
        self.visualization.plot_full_graph(self.graph.g_nx)

    def visualize_flow_paths(self, paths: List[Tuple[List[str], List[str], float]], edge_flows: Dict[Tuple[str, str], List[Dict[str, float]]]):
        self.visualization.plot_flow_paths(self.graph.g_nx, paths, edge_flows)

    def visualize_full_flow_paths(self, edge_flows: Dict[Tuple[str, str], float], filename: str):
        self.visualization.plot_full_flow_paths(self.graph.g_nx, edge_flows, filename)