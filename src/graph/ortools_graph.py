from ortools.graph.python import max_flow
import time
from typing import List, Tuple, Dict, Any, Optional, Iterator, Set, Callable
from collections import defaultdict

from .base import BaseGraph
from .flow.decomposition import decompose_flow, simplify_paths
from .flow.utils import verify_flow_conservation

class ORToolsGraph(BaseGraph):
    def __init__(self, edges: List[Tuple[str, str]], capacities: List[float], tokens: List[str]):
        """
        Initialize OR-Tools graph implementation.
        
        Args:
            edges: List of (source, target) node pairs
            capacities: List of edge capacities
            tokens: List of token identifiers for each edge
        """
        # Create mappings and data structures
        self._initialize_data_structures(edges, capacities, tokens)
        
        # Initialize OR-Tools solver
        self.solver = max_flow.SimpleMaxFlow()
        
        # Create adjacency maps for efficient lookups
        self.arc_adjacency = {}  # node_idx -> List[(arc_idx, head_idx, capacity)]
        self.reverse_arc_adjacency = {}  # node_idx -> List[(arc_idx, tail_idx, capacity)]
        
        # Initialize solver with edges
        self._initialize_solver()

    def _initialize_data_structures(self, edges: List[Tuple[str, str]], 
                                 capacities: List[float], tokens: List[str]):
        """Initialize internal data structures."""
        # Create node index mappings
        unique_nodes = set()
        for u, v in edges:
            unique_nodes.add(u)
            unique_nodes.add(v)
        self.node_to_index = {node: idx for idx, node in enumerate(sorted(unique_nodes))}
        self.index_to_node = {idx: node for node, idx in self.node_to_index.items()}

        # Store edge data
        self.edges = []
        self.edge_data = {}  # (u, v) -> {capacity, label}
        self.outgoing_edges = defaultdict(list)  # node -> [(neighbor, capacity, label)]
        self.incoming_edges = defaultdict(list)  # node -> [(neighbor, capacity, label)]

        for (u, v), capacity, token in zip(edges, capacities, tokens):
            self.edges.append((u, v))
            self.edge_data[(u, v)] = {
                'capacity': int(capacity),
                'label': token
            }
            self.outgoing_edges[u].append((v, int(capacity), token))
            self.incoming_edges[v].append((u, int(capacity), token))

    def _initialize_solver(self):
        """Initialize OR-Tools solver with edges."""
        for u, v in self.edges:
            u_idx = self.node_to_index[u]
            v_idx = self.node_to_index[v]
            capacity = self.edge_data[(u, v)]['capacity']
            
            # Add edge to solver
            arc_idx = self.solver.add_arc_with_capacity(u_idx, v_idx, capacity)
            
            # Build forward adjacency
            if u_idx not in self.arc_adjacency:
                self.arc_adjacency[u_idx] = []
            self.arc_adjacency[u_idx].append((arc_idx, v_idx, capacity))
            
            # Build reverse adjacency
            if v_idx not in self.reverse_arc_adjacency:
                self.reverse_arc_adjacency[v_idx] = []
            self.reverse_arc_adjacency[v_idx].append((arc_idx, u_idx, capacity))

    def compute_flow(self, source: str, sink: str, flow_func: Optional[Callable] = None,
                    requested_flow: Optional[str] = None) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """
        Compute maximum flow between source and sink nodes using OR-Tools.
        Note: flow_func parameter is ignored as OR-Tools uses its own algorithm.
        """
        if not self.has_vertex(source) or not self.has_vertex(sink):
            raise ValueError(f"Source node '{source}' or sink node '{sink}' not in graph.")
            
        source_idx = self.node_to_index[source]
        sink_idx = self.node_to_index[sink]
        
        if self.solver.num_arcs() == 0:
            print("No edges in graph. No flow is possible.")
            return 0, {}

        # Process direct paths efficiently using cached adjacency maps
        direct_flow, direct_flow_dict = self._process_direct_paths(
            source, sink, source_idx, sink_idx, requested_flow
        )

        if requested_flow and direct_flow >= int(requested_flow):
            print(f"Satisfied requested flow of {requested_flow} with direct edges.")
            return direct_flow, direct_flow_dict

        remaining_flow = None if requested_flow is None else int(requested_flow) - direct_flow

        # Solve max flow
        start_time = time.time()
        status = self.solver.solve(source_idx, sink_idx)
        print(f"Solver Time: {time.time() - start_time}")

        if status == self.solver.OPTIMAL:
            total_flow, flow_dict = self._build_flow_dict(
                sink_idx, remaining_flow, direct_flow_dict
            )
            # Store the flow dictionary to avoid recomputation
            self._flow_dict = flow_dict
            return total_flow + direct_flow, flow_dict
        else:
            raise RuntimeError("OR-Tools solver failed to find optimal solution")

    def _process_direct_paths(self, source: str, sink: str, source_idx: int, sink_idx: int,
                            requested_flow: Optional[str]) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """Process direct paths through intermediate nodes efficiently."""
        direct_flow = 0
        direct_flow_dict = {}
        max_flow = float('inf') if requested_flow is None else int(requested_flow)
        
        # Use cached adjacency maps
        source_edges = self.arc_adjacency.get(source_idx, [])
        for src_arc_idx, intermediate_idx, source_capacity in source_edges:
            intermediate_node = self.index_to_node[intermediate_idx]
            
            if '_' in intermediate_node:
                sink_edges = self.arc_adjacency.get(intermediate_idx, [])
                for sink_arc_idx, target_idx, sink_capacity in sink_edges:
                    if target_idx == sink_idx:
                        # Calculate flow through this path
                        current_source_cap = self.solver.capacity(src_arc_idx)
                        current_sink_cap = self.solver.capacity(sink_arc_idx)
                        flow = min(current_source_cap, current_sink_cap, max_flow - direct_flow)
                        
                        if flow > 0:
                            # Update capacities in solver
                            self.solver.set_arc_capacity(src_arc_idx, current_source_cap - flow)
                            self.solver.set_arc_capacity(sink_arc_idx, current_sink_cap - flow)
                            
                            # Record the flow
                            direct_flow_dict.setdefault(source, {})[intermediate_node] = flow
                            direct_flow_dict.setdefault(intermediate_node, {})[sink] = flow
                            direct_flow += flow
                            
                            if direct_flow >= max_flow:
                                return direct_flow, direct_flow_dict
                        break

        return direct_flow, direct_flow_dict

    def _build_flow_dict(self, sink_idx: int, remaining_flow: Optional[int],
                        direct_flow_dict: Dict[str, Dict[str, int]]) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """Build flow dictionary from solver results."""
        flow_value = int(self.solver.optimal_flow())
        if remaining_flow is not None:
            flow_value = min(flow_value, remaining_flow)

        flow_dict = {}
        for i in range(self.solver.num_arcs()):
            flow = int(self.solver.flow(i))
            if flow > 0:
                u = self.index_to_node[self.solver.tail(i)]
                v = self.index_to_node[self.solver.head(i)]
                flow_dict.setdefault(u, {})[v] = flow

        # Combine with direct flows
        if direct_flow_dict:
            for u, flows in direct_flow_dict.items():
                if u not in flow_dict:
                    flow_dict[u] = flows.copy()
                else:
                    flow_dict[u].update(flows)

        return flow_value, flow_dict

    def flow_decomposition(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str,
                          requested_flow: Optional[int] = None) -> Tuple[List[Tuple[List[str], List[str], int]],
                                                                       Dict[Tuple[str, str], int]]:
        """Decompose flow into paths."""
        # Use stored flow dictionary to avoid recomputation
        flow_dict = getattr(self, '_flow_dict', flow_dict)
        
        paths, edge_flows = decompose_flow(flow_dict, source, sink, requested_flow)
        
        # Add labels
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

    # BaseGraph interface implementation
    def num_vertices(self) -> int:
        return len(self.node_to_index)
    
    def num_edges(self) -> int:
        return self.solver.num_arcs()
    
    def get_vertices(self) -> Set[str]:
        return set(self.node_to_index.keys())
    
    def get_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        return [(u, v, self.edge_data[(u, v)]) for u, v in self.edges]
    
    def in_degree(self, vertex_id: str) -> int:
        return len(self.incoming_edges[vertex_id])
    
    def out_degree(self, vertex_id: str) -> int:
        return len(self.outgoing_edges[vertex_id])
    
    def degree(self, vertex_id: str) -> int:
        return self.in_degree(vertex_id) + self.out_degree(vertex_id)
    
    def predecessors(self, vertex_id: str) -> Iterator[str]:
        return (u for u, _, _ in self.incoming_edges[vertex_id])
    
    def successors(self, vertex_id: str) -> Iterator[str]:
        return (v for v, _, _ in self.outgoing_edges[vertex_id])
    
    def has_vertex(self, vertex_id: str) -> bool:
        return vertex_id in self.node_to_index

    def has_edge(self, u: str, v: str) -> bool:
        return (u, v) in self.edge_data

    def get_edge_data(self, u: str, v: str) -> Dict:
        return self.edge_data.get((u, v), {})

    def get_vertex(self, vertex_id: str):
        return vertex_id if vertex_id in self.node_to_index else None

    def get_edge_capacity(self, u: str, v: str) -> Optional[int]:
        edge_data = self.edge_data.get((u, v))
        return edge_data['capacity'] if edge_data else None

    def get_node_outflow_capacity(self, source_id: str) -> int:
        total_capacity = 0
        for v, capacity, _ in self.outgoing_edges[source_id]:
            if '_' in v:
                total_capacity += capacity
        return total_capacity

    def get_node_inflow_capacity(self, sink_id: str) -> int:
        total_capacity = 0
        for u, capacity, _ in self.incoming_edges[sink_id]:
            if '_' in u:
                total_capacity += capacity
        return total_capacity