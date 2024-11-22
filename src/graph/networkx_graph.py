# src/graph/networkx_graph.py
import networkx as nx
import time
from typing import List, Tuple, Dict, Any, Optional, Iterator, Set, Callable
from collections import defaultdict

from .base import BaseGraph
from .flow.decomposition import decompose_flow, simplify_paths
from .flow.utils import verify_flow_conservation

class NetworkXGraph(BaseGraph):
    def __init__(self, edges: List[Tuple[str, str]], capacities: List[float], tokens: List[str]):
        self.g_nx = self._create_graph(edges, capacities, tokens)

    def _create_graph(self, edges: List[Tuple[str, str]], capacities: List[float], 
                     tokens: List[str]) -> nx.DiGraph:
        """Create NetworkX graph with edge properties."""
        g = nx.DiGraph()
        
        # Add all edges in a single batch with their properties
        edge_data = [
            (u, v, {'capacity': int(capacity), 'label': token}) 
            for (u, v), capacity, token in zip(edges, capacities, tokens)
        ]
        g.add_edges_from(edge_data)
        
        return g

    def compute_flow(self, source: str, sink: str, flow_func: Optional[Callable] = None,
                    requested_flow: Optional[str] = None) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """Compute maximum flow between source and sink nodes."""
        if flow_func is None:
            flow_func = nx.algorithms.flow.preflow_push

        # Early exit if sink has no incoming edges
        if self.g_nx.in_degree(sink) == 0:
            print("Sink has no incoming edges. No flow is possible.")
            return 0, {}

        # Create a copy of the graph for modification
        graph_copy = self.g_nx.copy()
        
        # Process direct edges efficiently
        direct_flow = 0
        direct_flow_dict = {}
        direct_edges = []
        
        # Collect all potential direct edges first
        for node in graph_copy.successors(source):
            if '_' in node and sink in graph_copy.successors(node):
                capacity_source_intermediate = graph_copy[source][node]['capacity']
                capacity_intermediate_sink = graph_copy[node][sink]['capacity']
                direct_edges.append((node, min(capacity_source_intermediate, capacity_intermediate_sink)))

        # Process direct edges if any exist
        if direct_edges:
            direct_edges.sort(key=lambda x: x[1], reverse=True)
            req_flow_int = int(requested_flow) if requested_flow is not None else None
            remaining_flow = req_flow_int if req_flow_int is not None else float('inf')
            
            for intermediate_node, capacity in direct_edges:
                if req_flow_int is not None and remaining_flow <= 0:
                    break
                    
                flow = min(capacity, remaining_flow) if req_flow_int is not None else capacity
                
                if flow > 0:
                    direct_flow_dict.setdefault(source, {})[intermediate_node] = flow
                    direct_flow_dict.setdefault(intermediate_node, {})[sink] = flow
                    direct_flow += flow
                    
                    # Update remaining capacities
                    graph_copy[source][intermediate_node]['capacity'] -= flow
                    graph_copy[intermediate_node][sink]['capacity'] -= flow
                    
                    if req_flow_int is not None:
                        remaining_flow -= flow

            # Early return if direct edges satisfy requested flow
            if req_flow_int is not None and direct_flow >= req_flow_int:
                print(f"Satisfied requested flow of {requested_flow} with direct edges.")
                self._flow_dict = direct_flow_dict  # Cache the flow dictionary
                return direct_flow, direct_flow_dict

        # Calculate remaining requested flow
        remaining_requested_flow = None if requested_flow is None else int(requested_flow) - direct_flow

        try:
            # Remove edges with zero capacity
            zero_capacity_edges = [(u, v) for u, v, d in graph_copy.edges(data=True) 
                                if d['capacity'] <= 0]
            graph_copy.remove_edges_from(zero_capacity_edges)
            
            start = time.time()
            # Compute the maximum flow
            try:
                flow_value, flow_dict = nx.maximum_flow(
                    graph_copy, source, sink,
                    flow_func=flow_func,
                    cutoff=remaining_requested_flow
                )
            except:
                flow_value, flow_dict = nx.maximum_flow(
                    graph_copy, source, sink,
                    flow_func=flow_func
                )
            print(f"Solver Time: {time.time() - start}")

            # Convert values to integers and remove zero flows
            flow_dict = {
                u: {v: int(f) for v, f in flows.items() if f > 0}
                for u, flows in flow_dict.items()
            }
            flow_value = int(flow_value)

            # Combine with direct flows if any
            if direct_flow_dict:
                for u, flows in direct_flow_dict.items():
                    if u not in flow_dict:
                        flow_dict[u] = flows.copy()
                    else:
                        for v, f in flows.items():
                            flow_dict[u][v] = flow_dict[u].get(v, 0) + f

            # Cache the flow dictionary
            self._flow_dict = flow_dict
            return flow_value + direct_flow, flow_dict

        except Exception as e:
            print(f"Error in flow computation: {str(e)}")
            raise

    def flow_decomposition(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str,
                          requested_flow: Optional[int] = None) -> Tuple[List[Tuple[List[str], List[str], int]],
                                                                       Dict[Tuple[str, str], int]]:
        """Decompose flow into paths."""
        # Use cached flow dictionary if available
        flow_dict = getattr(self, '_flow_dict', flow_dict)
        paths, edge_flows = decompose_flow(flow_dict, source, sink, requested_flow)
        
        # Add labels for NetworkX implementation
        labeled_paths = []
        for path, _, flow in paths:
            path_labels = []
            for u, v in zip(path[:-1], path[1:]):
                edge_data = self.get_edge_data(u, v)
                path_labels.append(edge_data.get('label', 'no_label'))
            labeled_paths.append((path, path_labels, flow))
        
        return labeled_paths, edge_flows

    def simplified_flow_decomposition(self, original_paths: List[Tuple[List[str], List[str], int]]) -> List[Tuple[List[str], List[str], int]]:
        """Create simplified paths."""
        return simplify_paths(original_paths)

    # Required BaseGraph interface methods
    def num_vertices(self) -> int:
        return self.g_nx.number_of_nodes()
    
    def num_edges(self) -> int:
        return self.g_nx.number_of_edges()
    
    def get_vertices(self) -> Set[str]:
        return set(self.g_nx.nodes())
    
    def get_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        return [(u, v, d) for u, v, d in self.g_nx.edges(data=True)]
    
    def in_degree(self, vertex_id: str) -> int:
        return self.g_nx.in_degree(vertex_id)
    
    def out_degree(self, vertex_id: str) -> int:
        return self.g_nx.out_degree(vertex_id)
    
    def degree(self, vertex_id: str) -> int:
        return self.g_nx.degree(vertex_id)
    
    def predecessors(self, vertex_id: str) -> Iterator[str]:
        return self.g_nx.predecessors(vertex_id)
    
    def successors(self, vertex_id: str) -> Iterator[str]:
        return self.g_nx.successors(vertex_id)
    
    def has_vertex(self, vertex_id: str) -> bool:
        return vertex_id in self.g_nx
    
    def has_edge(self, u: str, v: str) -> bool:
        return self.g_nx.has_edge(u, v)
    
    def get_edge_data(self, u: str, v: str) -> Dict[str, Any]:
        return self.g_nx.get_edge_data(u, v) or {}
    
    def get_edge_capacity(self, u: str, v: str) -> Optional[int]:
        if self.has_edge(u, v):
            return self.g_nx[u][v].get('capacity')
        return None

    def get_node_outflow_capacity(self, source_id: str) -> int:
        total_capacity = 0
        if source_id not in self.g_nx:
            return total_capacity
        for neighbor in self.g_nx.successors(source_id):
            if '_' in neighbor:
                edge_data = self.g_nx.get_edge_data(source_id, neighbor)
                capacity = edge_data.get('capacity', 0)
                total_capacity += capacity
        return total_capacity

    def get_node_inflow_capacity(self, sink_id: str) -> int:
        total_capacity = 0
        if sink_id not in self.g_nx:
            return total_capacity
        for predecessor in self.g_nx.predecessors(sink_id):
            if '_' in predecessor:
                edge_data = self.g_nx.get_edge_data(predecessor, sink_id)
                capacity = edge_data.get('capacity', 0)
                total_capacity += capacity
        return total_capacity