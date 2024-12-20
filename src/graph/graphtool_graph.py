from graph_tool import Graph
import time
from typing import List, Tuple, Dict, Any, Optional, Iterator, Set, Callable
from collections import defaultdict

from .base import BaseGraph
from .flow.decomposition import decompose_flow, simplify_paths
from .flow.utils import verify_flow_conservation
import logging

# Configure logging for the module
logger = logging.getLogger(__name__)

class GraphToolGraph(BaseGraph):
    def __init__(self, edges: List[Tuple[str, str]], capacities: List[int], tokens: List[str]):
        self.logger = logging.getLogger(__name__)
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
        # Create the base graph
        g = Graph(directed=True)
        
        # Create properties
        v_prop = g.new_vertex_property("string")
        e_prop_capacity = g.new_edge_property("int64_t")
        e_prop_token = g.new_edge_property("string")
        
        # Add properties to graph
        g.vertex_properties["id"] = v_prop
        g.edge_properties["capacity"] = e_prop_capacity
        g.edge_properties["token"] = e_prop_token
        
        # Create vertex mapping first
        vertex_map = {}
        for u, v in edges:
            if u not in vertex_map:
                vertex = g.add_vertex()
                vertex_map[u] = vertex
                v_prop[vertex] = u
            if v not in vertex_map:
                vertex = g.add_vertex()
                vertex_map[v] = vertex
                v_prop[vertex] = v
        
        # Add edges with properties
        for (u, v), capacity, token in zip(edges, capacities, tokens):
            e = g.add_edge(vertex_map[u], vertex_map[v])
            e_prop_capacity[e] = int(capacity)  # Ensure integer capacity
            e_prop_token[e] = str(token)  # Ensure string token
        
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
            # First, let's check source capacity
            print("\nSource capacity check:")
            source_capacity = 0
            for e in s.out_edges():
                source_capacity += int(capacity_copy[e])
            print(f"Total source capacity: {source_capacity}")
            
            # Compute flow
            res = flow_func(self.g_gt, s, t, capacity_copy)
        
            # Check flows out of source
            print("\nSource flow check:")
            source_outflow = 0
            for e in s.out_edges():
                flow = int(capacity_copy[e]) - int(res[e])
                source_outflow += flow
                print(f"Edge {self.vertex_id[s]} -> {self.vertex_id[e.target()]}: {flow}")
            print(f"Total flow out of source: {source_outflow}")
            
            # Check flows into sink
            print("\nSink flow check:")
            sink_inflow = 0
            for e in t.in_edges():
                flow = int(capacity_copy[e]) - int(res[e])
                sink_inflow += flow
                print(f"Edge {self.vertex_id[e.source()]} -> {self.vertex_id[t]}: {flow}")
            print(f"Total flow into sink: {sink_inflow}")

            # Let's inspect what we have
            print("\nSample flows into sink before conversion:")
            sink_flow_before = 0
            for e in t.in_edges():
                orig_cap = int(capacity_copy[e])
                residual = int(res[e])
                actual_flow = orig_cap - residual
                sink_flow_before += actual_flow
                print(f"Edge {self.vertex_id[e.source()]} -> sink:")
                print(f"  Original capacity: {orig_cap}")
                print(f"  Residual capacity: {residual}")
                print(f"  Actual flow: {actual_flow}")

            # Convert residual capacities to flows
            res.a = capacity_copy.a - res.a

            print("\nSink flows after conversion:")
            sink_flow_after = 0
            for e in t.in_edges():
                flow = int(res[e])
                sink_flow_after += flow
                print(f"Edge {self.vertex_id[e.source()]} -> sink: {flow}")

            print(f"\nTotal sink flow before: {sink_flow_before}")
            print(f"Total sink flow after: {sink_flow_after}")

            # Build flow dictionary and proceed as before
            total_flow, flow_dict = self._build_flow_dict(
                res,
                t,
                remaining_flow,
                direct_flow_dict
            )

            print(total_flow, direct_flow)
            return total_flow, flow_dict

        finally:
            self.g_gt.clear_filters()

    

    def _process_direct_paths(self, source: str, sink: str, s, t, capacity_copy, requested_flow: Optional[str]) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """Process direct paths through intermediate nodes."""
        direct_flow = 0
        direct_flow_dict = {}
        direct_edges = []
        
        # Store original capacities before modification
        original_capacities = {}
        
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
                        # Store original capacities
                        original_capacities[e] = int(capacity_copy[e])
                        original_capacities[e2] = int(capacity_copy[e2])

        try:
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
            
        finally:
            # Restore any modified capacities if algorithm fails
            for e, cap in original_capacities.items():
                if int(capacity_copy[e]) < 0:  # Check for negative capacities
                    capacity_copy[e] = cap

    
    def _build_flow_dict(self, flow, t, remaining_flow: Optional[int],
                    direct_flow_dict: Dict[str, Dict[str, int]]):
        total_flow = 0
        flow_dict = {}

        # Print a test edge first
        test_edge = None
        for e in t.in_edges():
            test_edge = e
            print(f"\nTest edge {self.vertex_id[e.source()]} -> {self.vertex_id[e.target()]}:")
            print(f"  Flow value: {int(flow[e])}")
            break

        # First process all edges
        for e in self.g_gt.edges():
            f = int(flow[e])
            if f > 0:
                u = self.vertex_id[e.source()]
                v = self.vertex_id[e.target()]
                if u not in flow_dict:
                    flow_dict[u] = {}
                flow_dict[u][v] = f
                if e.target() == t:
                    total_flow += f  # All flows count toward total

        if remaining_flow is not None:
            total_flow = min(total_flow, remaining_flow)

        # No need to add direct flows to total_flow since they're already in the flow values
        if direct_flow_dict:
            for u, flows in direct_flow_dict.items():
                if u not in flow_dict:
                    flow_dict[u] = flows.copy()
                else:
                    for v, f in flows.items():
                        flow_dict[u][v] = flow_dict[u].get(v, 0) + f

        return total_flow, flow_dict


    def _build_flow_dict3(self, flow, t, remaining_flow: Optional[int],
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
        """Compute total outflow capacity from source to intermediate nodes."""
        source_vertex = self.get_vertex(source_id)
        if source_vertex is None:
            return 0
            
        # Map to store max capacity for each holder-token combination
        holder_token_capacities = {}
        
        # Iterate through all outgoing edges
        for e in source_vertex.out_edges():
            target = self.vertex_id[e.target()]
            if '_' in target:
                # Split intermediate node ID into holder and token
                holder, token = target.split('_')
                capacity = int(self.capacity[e])
                
                # Keep only the maximum capacity for each holder-token pair
                key = (holder, token)
                if key not in holder_token_capacities or capacity > holder_token_capacities[key]:
                    holder_token_capacities[key] = capacity
        
        # Sum all maximum capacities
        return sum(holder_token_capacities.values())

    def get_node_inflow_capacity(self, sink_id: str) -> int:
        """Compute total inflow capacity to sink from intermediate nodes."""
        sink_vertex = self.get_vertex(sink_id)
        if sink_vertex is None:
            return 0
            
        # Map to store max capacity for each holder-token combination
        holder_token_capacities = {}
        
        # Iterate through all incoming edges
        for e in sink_vertex.in_edges():
            source = self.vertex_id[e.source()]
            if '_' in source:
                # Split intermediate node ID into holder and token
                holder, token = source.split('_')
                capacity = int(self.capacity[e])
                
                # Keep only the maximum capacity for each holder-token pair
                key = (holder, token)
                if key not in holder_token_capacities or capacity > holder_token_capacities[key]:
                    holder_token_capacities[key] = capacity
        
        # Sum all maximum capacities
        return sum(holder_token_capacities.values())

    def _debug_capacities(self, vertex_id: str, is_source: bool = True):
        """Debug helper to print detailed capacity information."""
        vertex = self.get_vertex(vertex_id)
        if vertex is None:
            print(f"Vertex {vertex_id} not found")
            return
            
        edges = vertex.out_edges() if is_source else vertex.in_edges()
        direction = "outgoing" if is_source else "incoming"
        print(f"\nDebug {direction} edges for vertex {vertex_id}:")
        
        holder_token_capacities = {}
        for e in edges:
            node = self.vertex_id[e.target() if is_source else e.source()]
            if '_' in node:
                holder, token = node.split('_')
                capacity = int(self.capacity[e])
                print(f"Edge to {node}: capacity = {capacity}")
                
                key = (holder, token)
                if key not in holder_token_capacities or capacity > holder_token_capacities[key]:
                    holder_token_capacities[key] = capacity
        
        print("\nHolder-Token Maximums:")
        for (holder, token), capacity in holder_token_capacities.items():
            print(f"Holder: {holder}, Token: {token}, Max Capacity: {capacity}")
        
        total = sum(holder_token_capacities.values())
        print(f"\nTotal Capacity: {total}")

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


    def prepare_arbitrage_graph(self, start_node: str, start_token: str, end_token: str) -> Tuple[str, str]:
        """
        Prepare graph for arbitrage analysis by modifying only outgoing edges from source.
        """
        try:
            # Verify start intermediate node exists
            start_intermediate = f"{start_node}_{start_token}"
            start_vertex = self.get_vertex(start_intermediate)
            if start_vertex is None:
                self.logger.warning(f"No intermediate node found for {start_node} with token {start_token}")
                return None, None

            # Get source vertex
            source_vertex = self.get_vertex(start_node)

            # Store original edge properties
            self._temp_edges = []
            for edge in source_vertex.out_edges():
                self._temp_edges.append((
                    edge,
                    int(self.capacity[edge])
                ))
                self.capacity[edge] = 0

            # Get available capacity from edge to start_intermediate
            available_capacity = 0
            for edge, original_capacity in self._temp_edges:
                if self.vertex_id[edge.target()] == start_intermediate:
                    self.capacity[edge] = original_capacity
                    available_capacity = original_capacity
                    break

            if available_capacity == 0:
                self.logger.warning(f"No edge found from {start_node} to {start_intermediate}")
                self.cleanup_arbitrage_graph()
                return None, None

            # Create virtual sink
            virtual_sink_id = f"virtual_sink_{start_node}_{start_token}_{end_token}"
            virtual_sink = self.g_gt.add_vertex()
            self.vertex_id[virtual_sink] = virtual_sink_id
            self.id_to_vertex[virtual_sink_id] = virtual_sink

            # Find and connect end positions to virtual sink
            edges_added = 0
            for edge in source_vertex.in_edges():
                pred = edge.source()
                pred_id = self.vertex_id[pred]
                if '_' in pred_id:
                    _, token = pred_id.split('_')
                    if token == end_token:
                        # Add edge to virtual sink
                        capacity = min(int(self.capacity[edge]), available_capacity)
                        new_edge = self.g_gt.add_edge(pred, virtual_sink)
                        self.capacity[new_edge] = capacity
                        self.token[new_edge] = end_token
                        edges_added += 1

            if edges_added == 0:
                self.logger.warning("No valid end states found for arbitrage")
                self.cleanup_arbitrage_graph()
                return None, None

            self.logger.info(f"Added {edges_added} edges to virtual sink")
            return start_node, virtual_sink_id

        except Exception as e:
            self.logger.error(f"Error preparing arbitrage graph: {e}")
            self.cleanup_arbitrage_graph()
            raise

    def cleanup_arbitrage_graph(self):
        """Clean up temporary changes made for arbitrage analysis."""
        try:
            # Restore original edge capacities
            if hasattr(self, '_temp_edges'):
                for edge, capacity in self._temp_edges:
                    self.capacity[edge] = capacity
                delattr(self, '_temp_edges')

            # Remove virtual sink and its edges
            for v in list(self.g_gt.vertices()):
                v_id = self.vertex_id[v]
                if str(v_id).startswith('virtual_sink_'):
                    if v_id in self.id_to_vertex:
                        del self.id_to_vertex[v_id]
                    self.g_gt.remove_vertex(v)

        except Exception as e:
            self.logger.error(f"Error cleaning up arbitrage graph: {e}")
            raise

    def interpret_arbitrage_flow(self, flow_dict: Dict[str, Dict[str, int]], 
                            start_node: str, virtual_sink: str) -> Dict[str, Dict[str, int]]:
        """Convert flows through virtual sink back to actual token flows."""
        real_flows = defaultdict(dict)
        
        # First copy all non-virtual flows
        for u, flows in flow_dict.items():
            for v, flow in flows.items():
                if v != virtual_sink and flow > 0:
                    real_flows[u][v] = flow
        
        # Process flows to virtual sink
        for u, flows in flow_dict.items():
            virtual_flow = flows.get(virtual_sink, 0)
            if virtual_flow > 0:
                # Get capacity of edge from u back to start_node
                back_edge_capacity = self.get_edge_capacity(u, start_node)
                if back_edge_capacity is not None:
                    # Add the return flow
                    real_flows[u][start_node] = virtual_flow
        
        return dict(real_flows)