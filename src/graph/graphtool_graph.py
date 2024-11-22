from graph_tool import Graph
import time
from typing import List, Tuple, Dict, Any, Optional, Iterator, Set, Callable
from collections import defaultdict

from .base import BaseGraph
from .flow.decomposition import decompose_flow, simplify_paths
from .flow.utils import verify_flow_conservation

class GraphToolGraph(BaseGraph):
    def __init__(self, edges: List[Tuple[str, str]], capacities: List[int], tokens: List[str]):
        self.g_gt = self._create_graph(edges, capacities, tokens)
        self._initialize_properties()

    def _initialize_properties(self):
        """Initialize graph properties and mappings."""
        self.vertex_index = self.g_gt.vertex_index
        self.edge_index = self.g_gt.edge_index
        self.capacity = self.g_gt.edge_properties["capacity"]
        self.token = self.g_gt.edge_properties["token"]
        self.vertex_id = self.g_gt.vertex_properties["id"]
        self.id_to_vertex = {self.vertex_id[v]: v for v in self.g_gt.vertices()}
        self.vertex_map = self.id_to_vertex.copy()

    def _create_graph(self, edges: List[Tuple[str, str]], capacities: List[int], 
                     tokens: List[str]) -> Graph:
        """Create a graph-tool graph with properties."""
        # Extract unique vertices
        unique_vertices = set()
        for u_id, v_id in edges:
            unique_vertices.add(u_id)
            unique_vertices.add(v_id)
        vertex_ids = sorted(unique_vertices)

        # Create mapping from vertex ID to index
        vertex_id_to_idx = {vertex_id: idx for idx, vertex_id in enumerate(vertex_ids)}

        # Initialize graph and add vertices
        g = Graph(directed=True)
        g.add_vertex(len(vertex_ids))

        # Create and assign vertex ID property
        v_prop = g.new_vertex_property("string")
        for idx, vertex_id in enumerate(vertex_ids):
            v_prop[g.vertex(idx)] = vertex_id
        g.vertex_properties["id"] = v_prop

        # Prepare edge list with properties
        edge_list = [
            (vertex_id_to_idx[u_id],
             vertex_id_to_idx[v_id],
             capacity,
             token)
            for (u_id, v_id), capacity, token in zip(edges, capacities, tokens)
        ]

        # Create and add edge properties
        e_prop_capacity = g.new_edge_property("int64_t")
        e_prop_token = g.new_edge_property("string")
        g.add_edge_list(
            edge_list,
            eprops=[e_prop_capacity, e_prop_token],
            hashed=False
        )

        # Assign edge properties
        g.edge_properties["capacity"] = e_prop_capacity
        g.edge_properties["token"] = e_prop_token

        return g

    def compute_flow(self, source: str, sink: str, flow_func: Optional[Callable] = None,
                    requested_flow: Optional[str] = None) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """Compute maximum flow between source and sink nodes."""
        s = self.get_vertex(source)
        t = self.get_vertex(sink)

        if s is None or t is None:
            raise ValueError(f"Source node '{source}' or sink node '{sink}' not in graph.")

        # Early exit if sink has no incoming edges
        if t.in_degree() == 0:
            print("Sink has no incoming edges. No flow is possible.")
            return 0, {}

        # Create capacity copy for modifications
        capacity_copy = self.g_gt.new_edge_property("int64_t")
        capacity_copy.a = self.capacity.a.copy()

        # Process direct paths
        direct_flow, direct_flow_dict = self._process_direct_paths(source, sink, s, t, capacity_copy, requested_flow)

        if requested_flow and direct_flow >= int(requested_flow):
            print(f"Satisfied requested flow of {requested_flow} with direct edges.")
            self._flow_dict = direct_flow_dict  # Cache the flow dictionary
            return direct_flow, direct_flow_dict

        remaining_flow = None if requested_flow is None else int(requested_flow) - direct_flow

        # Filter zero capacity edges
        edge_filter = self.g_gt.new_edge_property("bool")
        for e in self.g_gt.edges():
            edge_filter[e] = (capacity_copy[e] > 0)
        self.g_gt.set_edge_filter(edge_filter)

        try:
            # Compute flow
            start = time.time()
            res = flow_func(self.g_gt, s, t, capacity_copy)
            print(f"Solver Time: {time.time() - start}")

            # Compute actual flows
            flow = capacity_copy.copy()
            flow.a = capacity_copy.a - res.a

            # Build flow dictionary
            total_flow, flow_dict = self._build_flow_dict(flow, t, remaining_flow, direct_flow_dict)

            # Cache the flow dictionary
            self._flow_dict = flow_dict
            return total_flow + direct_flow, flow_dict

        finally:
            self.g_gt.clear_filters()

    def _process_direct_paths(self, source: str, sink: str, s, t, capacity_copy, requested_flow: Optional[str]) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """Process direct paths through intermediate nodes."""
        direct_flow = 0
        direct_flow_dict = {}
        direct_edges = []
        
        for e in s.out_edges():
            v = e.target()
            if '_' in self.vertex_id[v]:
                for e2 in v.out_edges():
                    if e2.target() == t:
                        direct_edges.append((
                            self.vertex_id[v],
                            min(int(capacity_copy[e]), int(capacity_copy[e2])),
                            e,
                            e2
                        ))

        if direct_edges:
            direct_edges.sort(key=lambda x: x[1], reverse=True)
            req_flow_int = int(requested_flow) if requested_flow is not None else None
            remaining_flow = req_flow_int if req_flow_int is not None else float('inf')

            for intermediate_node, capacity, e1, e2 in direct_edges:
                if req_flow_int is not None and remaining_flow <= 0:
                    break

                flow = min(capacity, remaining_flow) if req_flow_int is not None else capacity

                if flow > 0:
                    direct_flow_dict.setdefault(source, {})[intermediate_node] = flow
                    direct_flow_dict.setdefault(intermediate_node, {})[sink] = flow
                    direct_flow += flow

                    capacity_copy[e1] -= flow
                    capacity_copy[e2] -= flow

                    if req_flow_int is not None:
                        remaining_flow -= flow

        return direct_flow, direct_flow_dict

    def _build_flow_dict(self, flow, t, remaining_flow: Optional[int],
                        direct_flow_dict: Dict[str, Dict[str, int]]) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """Build flow dictionary from computed flow."""
        total_flow = 0
        flow_dict = {}

        # Process all edges with positive flow
        for e in self.g_gt.edges():
            f = int(flow[e])
            if f > 0:
                u = self.vertex_id[e.source()]
                v = self.vertex_id[e.target()]
                if u not in flow_dict:
                    flow_dict[u] = {}
                flow_dict[u][v] = f
                if e.target() == t:
                    total_flow += f

        if remaining_flow is not None:
            total_flow = min(total_flow, remaining_flow)

        # Combine with direct flows
        if direct_flow_dict:
            for u, flows in direct_flow_dict.items():
                if u not in flow_dict:
                    flow_dict[u] = flows.copy()
                else:
                    for v, f in flows.items():
                        flow_dict[u][v] = flow_dict[u].get(v, 0) + f

        return total_flow, flow_dict

    def flow_decomposition(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str,
                          requested_flow: Optional[int] = None) -> Tuple[List[Tuple[List[str], List[str], int]],
                                                                       Dict[Tuple[str, str], int]]:
        """Decompose flow into paths."""
        # Use cached flow dictionary if available
        flow_dict = getattr(self, '_flow_dict', flow_dict)
        paths, edge_flows = decompose_flow(flow_dict, source, sink, requested_flow)

        # Add graph-tool specific labels
        labeled_paths = []
        for path, _, flow in paths:
            path_labels = []
            for u, v in zip(path[:-1], path[1:]):
                u_vertex = self.get_vertex(u)
                v_vertex = self.get_vertex(v)
                edge = self.g_gt.edge(u_vertex, v_vertex)
                path_labels.append(self.token[edge])
            labeled_paths.append((path, path_labels, flow))

        return labeled_paths, edge_flows

    def simplified_flow_decomposition(self, original_paths: List[Tuple[List[str], List[str], int]]) -> List[Tuple[List[str], List[str], int]]:
        """Create simplified paths."""
        return simplify_paths(original_paths)

    # BaseGraph interface implementation
    def num_vertices(self) -> int:
        return self.g_gt.num_vertices()
    
    def num_edges(self) -> int:
        return self.g_gt.num_edges()
    
    def get_vertices(self) -> Set[str]:
        return {self.vertex_id[v] for v in self.g_gt.vertices()}
    
    def get_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        edges = []
        for e in self.g_gt.edges():
            u = self.vertex_id[e.source()]
            v = self.vertex_id[e.target()]
            data = {
                'capacity': int(self.capacity[e]),
                'label': self.token[e]
            }
            edges.append((u, v, data))
        return edges
    
    def has_vertex(self, vertex_id: str) -> bool:
        return vertex_id in self.id_to_vertex

    def has_edge(self, u: str, v: str) -> bool:
        u_vertex = self.id_to_vertex.get(u)
        v_vertex = self.id_to_vertex.get(v)
        if u_vertex is None or v_vertex is None:
            return False
        return self.g_gt.edge(u_vertex, v_vertex) is not None

    def get_edge_data(self, u: str, v: str) -> Dict:
        u_vertex = self.id_to_vertex.get(u)
        v_vertex = self.id_to_vertex.get(v)
        if u_vertex is None or v_vertex is None:
            return {}
        edge = self.g_gt.edge(u_vertex, v_vertex)
        if edge is None:
            return {}
        return {
            'capacity': int(self.capacity[edge]),
            'label': self.token[edge]
        }

    def get_vertex(self, vertex_id: str):
        return self.id_to_vertex.get(vertex_id)

    def get_edge_capacity(self, u: str, v: str) -> Optional[int]:
        if self.has_edge(u, v):
            u_vertex = self.id_to_vertex[u]
            v_vertex = self.id_to_vertex[v]
            edge = self.g_gt.edge(u_vertex, v_vertex)
            return int(self.capacity[edge])
        return None

    def get_node_outflow_capacity(self, source_id: str) -> int:
        total_capacity = 0
        source_vertex = self.get_vertex(source_id)
        if source_vertex is None:
            return total_capacity
        for edge in source_vertex.out_edges():
            target_vertex = edge.target()
            target_id = self.vertex_id[target_vertex]
            if '_' in target_id:
                capacity = int(self.capacity[edge])
                total_capacity += capacity
        return total_capacity

    def get_node_inflow_capacity(self, sink_id: str) -> int:
        total_capacity = 0
        sink_vertex = self.get_vertex(sink_id)
        if sink_vertex is None:
            return total_capacity
        for edge in sink_vertex.in_edges():
            source_vertex = edge.source()
            source_id = self.vertex_id[source_vertex]
            if '_' in source_id:
                capacity = int(self.capacity[edge])
                total_capacity += capacity
        return total_capacity

    def in_degree(self, vertex_id: str) -> int:
        v = self.id_to_vertex.get(vertex_id)
        return v.in_degree() if v is not None else 0
    
    def out_degree(self, vertex_id: str) -> int:
        v = self.id_to_vertex.get(vertex_id)
        return v.out_degree() if v is not None else 0
    
    def degree(self, vertex_id: str) -> int:
        return self.in_degree(vertex_id) + self.out_degree(vertex_id)
    
    def predecessors(self, vertex_id: str) -> Iterator[str]:
        v = self.id_to_vertex.get(vertex_id)
        if v is not None:
            for u in v.in_neighbors():
                yield self.vertex_id[u]
    
    def successors(self, vertex_id: str) -> Iterator[str]:
        v = self.id_to_vertex.get(vertex_id)
        if v is not None:
            for w in v.out_neighbors():
                yield self.vertex_id[w]