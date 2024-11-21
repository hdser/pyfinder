import networkx as nx
from graph_tool import Graph
from ortools.graph.python import max_flow

from abc import abstractmethod
from typing import Set, Dict, Any, Optional, Iterator, List, Tuple, Callable
import time
from collections import defaultdict, deque

class GraphCreator:
    @staticmethod
    def create_graph(graph_type: str, edges: List[Tuple[str, str]], capacities: List[float], tokens: List[str]):
        if graph_type == 'networkx':
            return NetworkXGraph(edges, capacities, tokens)
        elif graph_type == 'graph_tool':
            return GraphToolGraph(edges, capacities, tokens)
        elif graph_type == 'ortools':
            return ORToolsGraph(edges, capacities, tokens)
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")


class BaseGraph:
    """Abstract base class defining the interface for all graph implementations."""
    
    @abstractmethod
    def num_vertices(self) -> int:
        """Return the total number of vertices in the graph."""
        pass
    
    @abstractmethod
    def num_edges(self) -> int:
        """Return the total number of edges in the graph."""
        pass
    
    @abstractmethod
    def has_vertex(self, vertex_id: str) -> bool:
        """Check if a vertex exists in the graph."""
        pass

    @abstractmethod
    def has_edge(self, u: str, v: str) -> bool:
        """Check if an edge exists between two vertices."""
        pass

    @abstractmethod
    def get_edge_data(self, u: str, v: str) -> Dict[str, Any]:
        """Get edge attributes."""
        pass
    
    @abstractmethod
    def get_vertices(self) -> Set[str]:
        """Return set of all vertex IDs."""
        pass
    
    @abstractmethod
    def get_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Return list of all edges with their data."""
        pass
    
    @abstractmethod
    def in_degree(self, vertex_id: str) -> int:
        """Return number of incoming edges for a vertex."""
        pass
    
    @abstractmethod
    def out_degree(self, vertex_id: str) -> int:
        """Return number of outgoing edges for a vertex."""
        pass
    
    @abstractmethod
    def degree(self, vertex_id: str) -> int:
        """Return total degree (in + out) for a vertex."""
        pass
    
    @abstractmethod
    def predecessors(self, vertex_id: str) -> Iterator[str]:
        """Return iterator over predecessor vertices."""
        pass
    
    @abstractmethod
    def successors(self, vertex_id: str) -> Iterator[str]:
        """Return iterator over successor vertices."""
        pass
    
    @abstractmethod
    def get_edge_capacity(self, u: str, v: str) -> Optional[int]:
        """Get capacity of edge between u and v."""
        pass


class NetworkXGraph(BaseGraph):
    def __init__(self, edges: List[Tuple[str, str]], capacities: List[float], tokens: List[str]):
        self.g_nx = self._create_graph(edges, capacities, tokens)

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
    
    def get_edge_capacity(self, u: str, v: str) -> Optional[int]:
        if self.has_edge(u, v):
            return self.g_nx[u][v].get('capacity')
        return None
    

    def _create_graph(
        self, 
        edges: List[Tuple[str, str]], 
        capacities: List[float], 
        tokens: List[str]
    ) -> nx.DiGraph:
        g = nx.DiGraph()
        
        # Prepare a list of edge tuples with attribute dictionaries
        edge_data = [
            (u, v, {'capacity': int(capacity), 'label': token}) 
            for (u, v), capacity, token in zip(edges, capacities, tokens)
        ]
        
        # Add all edges in a single batch
        g.add_edges_from(edge_data)
        
        return g

    def has_vertex(self, vertex_id: str) -> bool:
        return vertex_id in self.g_nx

    def has_edge(self, u: str, v: str) -> bool:
        return self.g_nx.has_edge(u, v)

    def get_edge_data(self, u: str, v: str) -> Dict:
        return self.g_nx.get_edge_data(u, v) or {}

    def get_vertex(self, vertex_id: str):
        return vertex_id if vertex_id in self.g_nx else None

    def compute_flow(self, source: str, sink: str, flow_func: Optional[Callable] = None, requested_flow: Optional[str] = None) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """Compute maximum flow between source and sink nodes for NetworkX implementation."""
        if flow_func is None:
            flow_func = nx.algorithms.flow.preflow_push

        print(f"Computing flow from {source} to {sink}")
        print(f"Flow function: {flow_func.__name__}")
        
        # Early exit if sink has no incoming edges
        if self.g_nx.in_degree(sink) == 0:
            print("Sink has no incoming edges. No flow is possible.")
            return 0, {}

        # Create a copy of the graph to modify capacities
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

        # Process direct edges if we have any
        if direct_edges:
            # Sort by capacity for optimal processing
            direct_edges.sort(key=lambda x: x[1], reverse=True)
            req_flow_int = int(requested_flow) if requested_flow is not None else None
            remaining_flow = req_flow_int if req_flow_int is not None else float('inf')
            
            # Process each direct edge
            for intermediate_node, capacity in direct_edges:
                if req_flow_int is not None and remaining_flow <= 0:
                    break
                    
                flow = min(capacity, remaining_flow) if req_flow_int is not None else capacity
                
                if flow > 0:
                    # Update direct flow tracking
                    direct_flow += flow
                    if source not in direct_flow_dict:
                        direct_flow_dict[source] = {}
                    if intermediate_node not in direct_flow_dict:
                        direct_flow_dict[intermediate_node] = {}
                        
                    direct_flow_dict[source][intermediate_node] = flow
                    direct_flow_dict[intermediate_node][sink] = flow
                    
                    # Update remaining capacities in the graph
                    graph_copy[source][intermediate_node]['capacity'] -= flow
                    graph_copy[intermediate_node][sink]['capacity'] -= flow
                    
                    if req_flow_int is not None:
                        remaining_flow -= flow

            # Early return if direct edges satisfy the requested flow
            if req_flow_int is not None and direct_flow >= req_flow_int:
                print(f"Satisfied requested flow of {requested_flow} with direct edges.")
                return direct_flow, direct_flow_dict

        # Calculate remaining requested flow
        remaining_requested_flow = None if requested_flow is None else int(requested_flow) - direct_flow

        try:
            # Remove edges with zero capacity
            zero_capacity_edges = [(u, v) for u, v, d in graph_copy.edges(data=True) 
                                if d['capacity'] <= 0]
            graph_copy.remove_edges_from(zero_capacity_edges)
            
            start = time.time()
            # Compute the maximum flow on the modified graph
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
            print("Solver Time: ",time.time() - start)
            # Convert values to integers and remove zero flows
            flow_dict = {
                u: {v: int(f) for v, f in flows.items() if f > 0}
                for u, flows in flow_dict.items()
            }
            flow_value = int(flow_value)

            # Combine with direct flows if we have any
            if direct_flow_dict:
                for u, flows in direct_flow_dict.items():
                    if u not in flow_dict:
                        flow_dict[u] = flows.copy()
                    else:
                        for v, f in flows.items():
                            flow_dict[u][v] = flow_dict[u].get(v, 0) + f

            return flow_value + direct_flow, flow_dict

        except Exception as e:
            print(f"Error in flow computation: {str(e)}")
            raise

    
    def _limit_flow(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str, limit: int) -> Dict[str, Dict[str, int]]:
        limited_flow_dict = defaultdict(lambda: defaultdict(int))
        remaining_flow = limit

        while remaining_flow > 0:
            path = self._find_path(flow_dict, source, sink)
            if not path:
                break

            path_flow = min(min(flow_dict[u][v] for u, v in zip(path[:-1], path[1:])), remaining_flow)
            
            for u, v in zip(path[:-1], path[1:]):
                flow_dict[u][v] -= path_flow
                if flow_dict[u][v] == 0:
                    del flow_dict[u][v]
                flow_dict.setdefault(v, {})[u] = flow_dict[v].get(u, 0) + path_flow
                limited_flow_dict[u][v] += path_flow

            remaining_flow -= path_flow

        return dict(limited_flow_dict)
    

    def _find_flow_path(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str) -> List[str]:
        """Find a path with positive flow from source to sink using DFS with an iterator stack."""
        visited = {source}
        path = [source]
        stack = [(source, iter(flow_dict.get(source, {}).items()))]
        
        while stack:
            current, edges = stack[-1]
            try:
                next_node, flow = next(edges)
                if flow > 0 and next_node not in visited:
                    if next_node == sink:
                        path.append(next_node)
                        return path
                    visited.add(next_node)
                    path.append(next_node)
                    stack.append((next_node, iter(flow_dict.get(next_node, {}).items())))
            except StopIteration:
                stack.pop()
                if path:
                    path.pop()
        
        return []

    def flow_decomposition(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str, 
                        requested_flow: Optional[int] = None) -> Tuple[List[Tuple[List[str], List[str], int]], Dict[Tuple[str, str], int]]:
        """Decompose the flow into paths using optimized path finding."""
        paths = []
        edge_flows = {}
        current_flow = 0
        
        # Build residual flow graph
        residual_flow = {u: dict(flows) for u, flows in flow_dict.items()}
        
        while True:
            # Find a path from source to sink with positive flow
            path = self._find_flow_path(residual_flow, source, sink)
            if not path:
                break
                
            # Find minimum flow along the path
            path_flow = min(residual_flow[u][v] for u, v in zip(path[:-1], path[1:]))
            
            if requested_flow is not None:
                remaining_flow = requested_flow - current_flow
                if remaining_flow <= 0:
                    break
                path_flow = min(path_flow, remaining_flow)
            
            # Extract path labels
            path_labels = []
            for u, v in zip(path[:-1], path[1:]):
                edge_data = self.get_edge_data(u, v)
                path_labels.append(edge_data.get('label', 'no_label'))
                
                # Update edge flows
                edge_flows[(u, v)] = edge_flows.get((u, v), 0) + path_flow
            
            # Update residual graph efficiently
            for u, v in zip(path[:-1], path[1:]):
                residual_flow[u][v] -= path_flow
                if residual_flow[u][v] == 0:
                    del residual_flow[u][v]
                if not residual_flow[u]:
                    del residual_flow[u]
            
            paths.append((path, path_labels, path_flow))
            current_flow += path_flow
            
            if requested_flow is not None and current_flow >= requested_flow:
                break
        
        return paths, edge_flows

    def simplified_flow_decomposition(self, original_paths: List[Tuple[List[str], List[str], int]]) -> List[Tuple[List[str], List[str], int]]:
        """Create simplified paths using a single pass approach."""
        simplified_paths = []
        
        for path, labels, flow in original_paths:
            simplified_path = []
            simplified_labels = []
            current_token = None
            last_real_node = None
            
            for i, (node, label) in enumerate(zip(path, labels)):
                if '_' not in node:
                    if last_real_node is None:
                        # First real node
                        simplified_path.append(node)
                        last_real_node = node
                        current_token = label
                    elif label != current_token:
                        # Token change, add the previous node and update
                        if node != last_real_node:
                            simplified_path.append(last_real_node)
                            simplified_labels.append(current_token)
                            simplified_path.append(node)
                            current_token = label
                        last_real_node = node
            
            # Add the final segment if needed
            if last_real_node and last_real_node != path[-1] and '_' not in path[-1]:
                simplified_path.append(path[-1])
                simplified_labels.append(current_token)
            
            if len(simplified_path) > 1:
                simplified_paths.append((simplified_path, simplified_labels, flow))
        
        return simplified_paths

    def _find_path2(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str) -> List[str]:
        queue = deque([(source, [source])])
        visited = set()

        while queue:
            node, path = queue.popleft()
            if node not in visited:
                if node == sink:
                    return path
                visited.add(node)
                for next_node, flow in flow_dict[node].items():
                    if flow > 0:
                        queue.append((next_node, path + [next_node]))
        return []

    
    def flow_decomposition2(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str, requested_flow: Optional[int] = None) -> Tuple[List[Tuple[List[str], List[str], int]], Dict[Tuple[str, str], int]]:
        paths = []
        edge_flows = defaultdict(int)
        
        remaining_flow = {u: {v: flow for v, flow in flows.items()} for u, flows in flow_dict.items()}
        total_flow = sum(flow_dict[source].values())
        current_flow = 0
        
        if requested_flow is None:
            requested_flow = total_flow

        while current_flow < requested_flow:
            path = self._find_path(remaining_flow, source, sink)
            if not path:
                break
            
            path_flow = min(min(remaining_flow[u][v] for u, v in zip(path[:-1], path[1:])), requested_flow - current_flow)
            
            path_labels = [self.g_nx[u][v].get('label', 'no_label') for u, v in zip(path[:-1], path[1:])]
            
            paths.append((path, path_labels, path_flow))
            
            for u, v in zip(path[:-1], path[1:]):
                remaining_flow[u][v] -= path_flow
                if remaining_flow[u][v] == 0:
                    del remaining_flow[u][v]
                edge_flows[(u, v)] += path_flow

            current_flow += path_flow

            if current_flow >= requested_flow:
                break

        return paths, dict(edge_flows)

    def simplified_flow_decomposition2(self, original_paths: List[Tuple[List[str], List[str], int]]) -> List[Tuple[List[str], List[str], int]]:
        simplified_paths = []
        
        for path, labels, flow in original_paths:
            simplified_path = []
            simplified_labels = []
            current_token = None
            for i, (node, label) in enumerate(zip(path, labels + [None])):
                if '_' not in node:
                    if current_token is None or label != current_token:
                        if simplified_path:
                            simplified_path.append(node)
                            simplified_labels.append(current_token)
                        else:
                            simplified_path = [node]
                        current_token = label
                    elif i == len(path) - 1:
                        simplified_path.append(node)
                        simplified_labels.append(current_token)
            if len(simplified_path) > 1:
                simplified_paths.append((simplified_path, simplified_labels, flow))
        return simplified_paths
    


class GraphToolGraph(BaseGraph):
    def __init__(self, edges: List[Tuple[str, str]], capacities: List[int], tokens: List[str]):
        self.g_gt = self._create_graph(edges, capacities, tokens)
        self.vertex_index = self.g_gt.vertex_index
        self.edge_index = self.g_gt.edge_index
        self.capacity = self.g_gt.edge_properties["capacity"]
        self.token = self.g_gt.edge_properties["token"]
        self.vertex_id = self.g_gt.vertex_properties["id"]
        self.id_to_vertex = {self.vertex_id[v]: v for v in self.g_gt.vertices()}
        self.vertex_map = {self.vertex_id[v]: v for v in self.g_gt.vertices()}

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
    
    def get_edge_capacity(self, u: str, v: str) -> Optional[int]:
        if self.has_edge(u, v):
            u_vertex = self.id_to_vertex[u]
            v_vertex = self.id_to_vertex[v]
            edge = self.g_gt.edge(u_vertex, v_vertex)
            return int(self.capacity[edge])
        return None


    def _create_graph(self, edges: List[Tuple[str, str]], capacities: List[int], tokens: List[str]) -> Graph:
        # Step 1: Extract all unique vertex IDs
        unique_vertices = set()
        for u_id, v_id in edges:
            unique_vertices.add(u_id)
            unique_vertices.add(v_id)
        vertex_ids = sorted(unique_vertices)
        num_vertices = len(vertex_ids)

        # Step 2: Create a mapping from vertex ID to vertex index
        vertex_id_to_idx = {vertex_id: idx for idx, vertex_id in enumerate(vertex_ids)}

        # Step 3: Initialize the graph and add all vertices in bulk
        g = Graph(directed=True)
        g.add_vertex(num_vertices)

        # Step 4: Create and assign vertex properties in bulk
        v_prop = g.new_vertex_property("string")
        for idx, vertex_id in enumerate(vertex_ids):
            v_prop[g.vertex(idx)] = vertex_id
        g.vertex_properties["id"] = v_prop

        # Step 5: Prepare edge list as (source, target, capacity, token)
        edge_list = [
            (
                vertex_id_to_idx[u_id],
                vertex_id_to_idx[v_id],
                capacity,
                token
            )
            for (u_id, v_id), capacity, token in zip(edges, capacities, tokens)
        ]

        # Step 6: Add all edges at once using add_edge_list with properties
        e_prop_capacity = g.new_edge_property("int64_t")
        e_prop_token = g.new_edge_property("string")
        g.add_edge_list(
            edge_list,
            eprops=[e_prop_capacity, e_prop_token],
            hashed=False  # Set to True if edges might contain duplicates and you want to handle them
        )

        # Step 7: Link the properties to the graph
        g.edge_properties["capacity"] = e_prop_capacity
        g.edge_properties["token"] = e_prop_token

        return g


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

    def compute_flow(self, source: str, sink: str, flow_func: Optional[Callable] = None, requested_flow: Optional[str] = None) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """Compute maximum flow between source and sink nodes for graph-tool implementation."""
        
        s = self.get_vertex(source)
        t = self.get_vertex(sink)

        if s is None or t is None:
            raise ValueError(f"Source node '{source}' or sink node '{sink}' not in graph.")

        print(f"Computing flow from {source} to {sink}")
        print(f"Flow function: {flow_func.__name__}")

        # Early exit if sink has no incoming edges
        if t.in_degree() == 0:
            print("Sink has no incoming edges. No flow is possible.")
            return 0, {}

        # Create a copy of the capacity property map
        capacity_copy = self.g_gt.new_edge_property("int64_t")
        capacity_copy.a = self.capacity.a.copy()

        # Process direct edges efficiently
        direct_flow = 0
        direct_flow_dict = {}
        direct_edges = []
        
        # Collect all potential direct edges
        for e in s.out_edges():
            v = e.target()
            if '_' in self.vertex_id[v]:
                for e2 in v.out_edges():
                    if e2.target() == t:
                        capacity_source_intermediate = int(capacity_copy[e])
                        capacity_intermediate_sink = int(capacity_copy[e2])
                        direct_edges.append((
                            self.vertex_id[v],
                            min(capacity_source_intermediate, capacity_intermediate_sink),
                            e,  # Store the actual edges for capacity updates
                            e2
                        ))

        # Process direct edges if we have any
        if direct_edges:
            # Sort by capacity for optimal processing
            direct_edges.sort(key=lambda x: x[1], reverse=True)
            req_flow_int = int(requested_flow) if requested_flow is not None else None
            remaining_flow = req_flow_int if req_flow_int is not None else float('inf')
            
            # Process each direct edge
            for intermediate_node, capacity, e1, e2 in direct_edges:
                if req_flow_int is not None and remaining_flow <= 0:
                    break
                    
                flow = min(capacity, remaining_flow) if req_flow_int is not None else capacity
                
                if flow > 0:
                    direct_flow += flow
                    if source not in direct_flow_dict:
                        direct_flow_dict[source] = {}
                    if intermediate_node not in direct_flow_dict:
                        direct_flow_dict[intermediate_node] = {}
                        
                    direct_flow_dict[source][intermediate_node] = flow
                    direct_flow_dict[intermediate_node][sink] = flow
                    
                    # Update remaining capacities
                    capacity_copy[e1] -= flow
                    capacity_copy[e2] -= flow
                    
                    if req_flow_int is not None:
                        remaining_flow -= flow

            # Early return if direct edges satisfy the requested flow
            if req_flow_int is not None and direct_flow >= req_flow_int:
                print(f"Satisfied requested flow of {requested_flow} with direct edges.")
                return direct_flow, direct_flow_dict

        # Calculate remaining flow
        remaining_requested_flow = None if requested_flow is None else int(requested_flow) - direct_flow

        # Remove edges with zero or negative capacity
        edge_filter = self.g_gt.new_edge_property("bool")
        for e in self.g_gt.edges():
            edge_filter[e] = (capacity_copy[e] > 0)
        
        # Set the edge filter on the graph
        self.g_gt.set_edge_filter(edge_filter)
        
        try:
            # Compute the flow using graph-tool with modified capacities
            start = time.time()
            res = flow_func(self.g_gt, s, t, capacity_copy)
            print("Solver Time: ",time.time() - start)

            # Compute actual flows
            flow = capacity_copy.copy()
            flow.a = capacity_copy.a - res.a

            # Calculate total flow and create flow dictionary
            total_flow = 0
            flow_dict = {}
            start = time.time()
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
            print("flow-dict Time: ",time.time() - start)

            # Apply flow limit if requested
            if remaining_requested_flow is not None:
                total_flow = min(total_flow, remaining_requested_flow)

            # Store the residual graph for later use
            self.residual_capacity = res

            # Combine with direct flows if we have any
            if direct_flow_dict:
                for u, flows in direct_flow_dict.items():
                    if u not in flow_dict:
                        flow_dict[u] = flows.copy()
                    else:
                        for v, f in flows.items():
                            flow_dict[u][v] = flow_dict[u].get(v, 0) + f

            return total_flow + direct_flow, flow_dict

        finally:
            # Clear the edge filter
            self.g_gt.clear_filters()

    
    
    def _find_flow_path(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str) -> List[str]:
        """Find a path with positive flow from source to sink using DFS with an iterator stack."""
        visited = {source}
        path = [source]
        stack = [(source, iter(flow_dict.get(source, {}).items()))]
        
        while stack:
            current, edges = stack[-1]
            try:
                next_node, flow = next(edges)
                if flow > 0 and next_node not in visited:
                    if next_node == sink:
                        path.append(next_node)
                        return path
                    visited.add(next_node)
                    path.append(next_node)
                    stack.append((next_node, iter(flow_dict.get(next_node, {}).items())))
            except StopIteration:
                stack.pop()
                if path:
                    path.pop()
        
        return []

    def flow_decomposition(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str, 
                        requested_flow: Optional[int] = None) -> Tuple[List[Tuple[List[str], List[str], int]], Dict[Tuple[str, str], int]]:
        """Decompose the flow into paths using optimized path finding."""
        paths = []
        edge_flows = {}
        current_flow = 0
        
        # Build residual flow graph
        residual_flow = {u: dict(flows) for u, flows in flow_dict.items()}
        
        while True:
            # Find a path from source to sink with positive flow
            path = self._find_flow_path(residual_flow, source, sink)
            if not path:
                break
                
            # Find minimum flow along the path
            path_flow = min(residual_flow[u][v] for u, v in zip(path[:-1], path[1:]))
            
            if requested_flow is not None:
                remaining_flow = requested_flow - current_flow
                if remaining_flow <= 0:
                    break
                path_flow = min(path_flow, remaining_flow)
            
            # Extract path labels efficiently
            path_labels = []
            for u, v in zip(path[:-1], path[1:]):
                u_vertex = self.get_vertex(u)
                v_vertex = self.get_vertex(v)
                edge = self.g_gt.edge(u_vertex, v_vertex)
                path_labels.append(self.token[edge])
                
                # Update edge flows
                edge_flows[(u, v)] = edge_flows.get((u, v), 0) + path_flow
            
            # Update residual graph efficiently
            for u, v in zip(path[:-1], path[1:]):
                residual_flow[u][v] -= path_flow
                if residual_flow[u][v] == 0:
                    del residual_flow[u][v]
                if not residual_flow[u]:
                    del residual_flow[u]
            
            paths.append((path, path_labels, path_flow))
            current_flow += path_flow
            
            if requested_flow is not None and current_flow >= requested_flow:
                break
        
        return paths, edge_flows

    def simplified_flow_decomposition(self, original_paths: List[Tuple[List[str], List[str], int]]) -> List[Tuple[List[str], List[str], int]]:
        """Create simplified paths using a single pass approach."""
        simplified_paths = []
        
        for path, labels, flow in original_paths:
            simplified_path = []
            simplified_labels = []
            current_token = None
            last_real_node = None
            
            for i, (node, label) in enumerate(zip(path, labels)):
                if '_' not in node:
                    if last_real_node is None:
                        # First real node
                        simplified_path.append(node)
                        last_real_node = node
                        current_token = label
                    elif label != current_token:
                        # Token change, add the previous node and update
                        if node != last_real_node:
                            simplified_path.append(last_real_node)
                            simplified_labels.append(current_token)
                            simplified_path.append(node)
                            current_token = label
                        last_real_node = node
            
            # Add the final segment if needed
            if last_real_node and last_real_node != path[-1] and '_' not in path[-1]:
                simplified_path.append(path[-1])
                simplified_labels.append(current_token)
            
            if len(simplified_path) > 1:
                simplified_paths.append((simplified_path, simplified_labels, flow))
        
        return simplified_paths


    def flow_decomposition2(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str, requested_flow: Optional[int] = None, method: str = 'bfs') -> Tuple[List[Tuple[List[str], List[str], int]], Dict[Tuple[str, str], int]]:
        """
        Decompose the flow into paths using BFS or DFS.

        Parameters:
        - flow_dict: The flow dictionary obtained from compute_flow.
        - source: The source vertex ID.
        - sink: The sink vertex ID.
        - requested_flow: The amount of flow to decompose.
        - method: 'bfs' or 'dfs' to choose the path-finding method.

        Returns:
        - A tuple containing the list of paths and the edge flows.
        """
        s = self.get_vertex(source)
        t = self.get_vertex(sink)

        if s is None or t is None:
            raise ValueError(f"Source node '{source}' or sink node '{sink}' not in graph.")

        total_flow = sum(flow_dict[source].values()) if source in flow_dict else 0
        if requested_flow is None or requested_flow > total_flow:
            requested_flow = total_flow

        paths = []
        edge_flows = defaultdict(int)
        current_flow = 0

        # Build residual flow graph
        residual_flow = {u: dict(v) for u, v in flow_dict.items()}

        while current_flow < requested_flow:
            if method == 'bfs':
                path, path_flow = self._find_flow_path_bfs(residual_flow, source, sink)
            elif method == 'dfs':
                path, path_flow = self._find_flow_path_dfs(residual_flow, source, sink)
            else:
                raise ValueError("Invalid method specified. Choose 'bfs' or 'dfs'.")

            if not path:
                break
            path_flow = min(path_flow, requested_flow - current_flow)
            path_labels = []
            for u, v in zip(path[:-1], path[1:]):
                e = self.g_gt.edge(self.get_vertex(u), self.get_vertex(v))
                path_labels.append(self.token[e])
                # Update residual_flow
                residual_flow[u][v] -= path_flow
                if residual_flow[u][v] == 0:
                    del residual_flow[u][v]
                if not residual_flow[u]:
                    del residual_flow[u]
                edge_flows[(u, v)] += path_flow
            paths.append((path, path_labels, path_flow))
            current_flow += path_flow

        return paths, dict(edge_flows)

    def _find_flow_path_bfs(self, residual_flow: Dict[str, Dict[str, int]], source: str, sink: str) -> Tuple[List[str], int]:
        """
        Find a path using BFS in the residual flow graph.

        Returns:
        - A tuple containing the path and the minimum flow on that path.
        """
        queue = deque()
        queue.append((source, [source], float('inf')))
        visited = {source}

        while queue:
            node, path, flow = queue.popleft()
            if node == sink:
                return path, flow
            for next_node, edge_flow in residual_flow.get(node, {}).items():
                if edge_flow > 0 and next_node not in visited:
                    visited.add(next_node)
                    new_flow = min(flow, edge_flow)
                    queue.append((next_node, path + [next_node], new_flow))
        return [], 0

    def _find_flow_path_dfs(self, residual_flow: Dict[str, Dict[str, int]], source: str, sink: str) -> Tuple[List[str], int]:
        """
        Find a path using DFS in the residual flow graph.

        Returns:
        - A tuple containing the path and the minimum flow on that path.
        """
        stack = [(source, [source], float('inf'))]
        visited = set()

        while stack:
            node, path, flow = stack.pop()
            if node == sink:
                return path, flow
            if node not in visited:
                visited.add(node)
                for next_node, edge_flow in residual_flow.get(node, {}).items():
                    if edge_flow > 0:
                        new_flow = min(flow, edge_flow)
                        stack.append((next_node, path + [next_node], new_flow))
        return [], 0
    
    def simplified_flow_decomposition2(self, original_paths: List[Tuple[List[str], List[str], int]]) -> List[Tuple[List[str], List[str], int]]:
        simplified_paths = []
        for path, labels, flow in original_paths:
            simplified_path = []
            simplified_labels = []
            current_token = None
            for i, (node, label) in enumerate(zip(path, labels + [None])):
                if '_' not in node:
                    if current_token is None or label != current_token:
                        if simplified_path:
                            simplified_path.append(node)
                            simplified_labels.append(current_token)
                        else:
                            simplified_path = [node]
                        current_token = label
                    elif i == len(path) - 1:
                        simplified_path.append(node)
                        simplified_labels.append(current_token)
            if len(simplified_path) > 1:
                simplified_paths.append((simplified_path, simplified_labels, flow))
        return simplified_paths



class ORToolsGraph(BaseGraph):
    def __init__(self, edges: List[Tuple[str, str]], capacities: List[float], tokens: List[str]):
        """
        Initialize OR-Tools graph with edges, capacities and tokens.
        
        Args:
            edges: List of (source, target) node pairs
            capacities: List of edge capacities
            tokens: List of token identifiers for each edge
        """
        # Create a NetworkX graph for storing the graph structure
        self.g_nx = nx.DiGraph()
        for (u, v), capacity, token in zip(edges, capacities, tokens):
            self.g_nx.add_edge(u, v, capacity=int(capacity), label=token)
        
        # Create node index mapping
        self.node_to_index = {node: idx for idx, node in enumerate(self.g_nx.nodes())}
        self.index_to_node = {idx: node for node, idx in self.node_to_index.items()}
        
        # Initialize OR-Tools max flow solver
        self.solver = max_flow.SimpleMaxFlow()
        
        # Create adjacency maps during initialization
        self.arc_adjacency = {}  # node_idx -> List[(arc_idx, head_idx, capacity)]
        self.reverse_arc_adjacency = {}  # node_idx -> List[(arc_idx, tail_idx, capacity)]
        
        # Add all edges to the solver and build adjacency maps
        for u, v, data in self.g_nx.edges(data=True):
            u_idx = self.node_to_index[u]
            v_idx = self.node_to_index[v]
            capacity = data['capacity']
            arc_idx = self.solver.add_arc_with_capacity(u_idx, v_idx, capacity)
            
            # Forward adjacency
            if u_idx not in self.arc_adjacency:
                self.arc_adjacency[u_idx] = []
            self.arc_adjacency[u_idx].append((arc_idx, v_idx, capacity))
            
            # Reverse adjacency
            if v_idx not in self.reverse_arc_adjacency:
                self.reverse_arc_adjacency[v_idx] = []
            self.reverse_arc_adjacency[v_idx].append((arc_idx, u_idx, capacity))

    def num_vertices(self) -> int:
        return len(self.node_to_index)
    
    def num_edges(self) -> int:
        return self.solver.num_arcs()
    
    def get_vertices(self) -> Set[str]:
        return set(self.node_to_index.keys())
    
    def get_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        edges = []
        for i in range(self.solver.num_arcs()):
            u = self.index_to_node[self.solver.tail(i)]
            v = self.index_to_node[self.solver.head(i)]
            data = self.g_nx.get_edge_data(u, v) or {}
            edges.append((u, v, data))
        return edges
    
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
    
    def get_edge_capacity(self, u: str, v: str) -> Optional[int]:
        if self.has_edge(u, v):
            u_idx = self.node_to_index[u]
            v_idx = self.node_to_index[v]
            for i in range(self.solver.num_arcs()):
                if (self.solver.tail(i) == u_idx and 
                    self.solver.head(i) == v_idx):
                    return self.solver.capacity(i)
        return None
    

    def has_vertex(self, vertex_id: str) -> bool:
        """Check if vertex exists in graph."""
        return vertex_id in self.g_nx

    def has_edge(self, u: str, v: str) -> bool:
        """Check if edge exists in graph."""
        return self.g_nx.has_edge(u, v)

    def get_edge_data(self, u: str, v: str) -> Dict:
        """Get edge data."""
        return self.g_nx.get_edge_data(u, v) or {}

    def get_vertex(self, vertex_id: str):
        """Get vertex."""
        return vertex_id if vertex_id in self.g_nx else None
    
    def compute_flow2(self, source: str, sink: str, flow_func: Optional[Callable] = None, 
                requested_flow: Optional[str] = None) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """
        Compute maximum flow between source and sink nodes using OR-Tools.
        
        Args:
            source: Source node ID
            sink: Sink node ID
            flow_func: Ignored (OR-Tools always uses its own algorithm)
            requested_flow: Optional maximum flow to compute
            
        Returns:
            Tuple containing:
            - Total flow value
            - Flow dictionary {node: {neighbor: flow}}
        """
        if not self.has_vertex(source) or not self.has_vertex(sink):
            raise ValueError(f"Source node '{source}' or sink node '{sink}' not in graph.")
            
        # Convert node IDs to indices
        source_idx = self.node_to_index[source]
        sink_idx = self.node_to_index[sink]
        
        # Early exit if sink has no incoming edges
        if self.g_nx.in_degree(sink) == 0:
            print("Sink has no incoming edges. No flow is possible.")
            return 0, {}

        # Store original capacities
        original_capacities = {
            i: self.solver.capacity(i)
            for i in range(self.solver.num_arcs())
        }
        
        # Process direct edges
        direct_flow = 0
        direct_flow_dict = {}
        modified_edges = set()
        
        # Look for direct paths (through intermediate nodes)
        for node in self.g_nx.successors(source):
            if '_' in str(node) and sink in self.g_nx.successors(node):
                # Find edge indices in solver
                source_edge = next((i for i in range(self.solver.num_arcs())
                                if self.solver.tail(i) == self.node_to_index[source]
                                and self.solver.head(i) == self.node_to_index[node]), None)
                sink_edge = next((i for i in range(self.solver.num_arcs())
                                if self.solver.tail(i) == self.node_to_index[node]
                                and self.solver.head(i) == self.node_to_index[sink]), None)
                
                if source_edge is not None and sink_edge is not None:
                    capacity_source_intermediate = original_capacities[source_edge]
                    capacity_intermediate_sink = original_capacities[sink_edge]
                    flow = min(capacity_source_intermediate, capacity_intermediate_sink)
                    
                    if requested_flow:
                        remaining_flow = int(requested_flow) - direct_flow
                        if remaining_flow <= 0:
                            break
                        flow = min(flow, remaining_flow)
                    
                    if flow > 0:
                        # Update flow dictionaries
                        if source not in direct_flow_dict:
                            direct_flow_dict[source] = {}
                        if node not in direct_flow_dict:
                            direct_flow_dict[node] = {}
                        direct_flow_dict[source][node] = flow
                        direct_flow_dict[node][sink] = flow
                        direct_flow += flow
                        
                        # Update capacities in solver
                        self.solver.set_arc_capacity(source_edge, max(0, capacity_source_intermediate - flow))
                        self.solver.set_arc_capacity(sink_edge, max(0, capacity_intermediate_sink - flow))
                        modified_edges.add(source_edge)
                        modified_edges.add(sink_edge)

        # If requested flow is satisfied by direct paths, restore capacities and return
        if requested_flow and direct_flow >= int(requested_flow):
            # Restore original capacities
            for edge_idx in modified_edges:
                self.solver.set_arc_capacity(edge_idx, original_capacities[edge_idx])
            print(f"Satisfied requested flow of {requested_flow} with direct edges.")
            return direct_flow, direct_flow_dict

        # Calculate remaining flow needed
        remaining_requested_flow = None if requested_flow is None else int(requested_flow) - direct_flow

        # Solve max flow with updated capacities
        try:
            status = self.solver.solve(source_idx, sink_idx)
            
            if status == self.solver.OPTIMAL:
                flow_value = int(self.solver.optimal_flow())
                if remaining_requested_flow is not None:
                    flow_value = min(flow_value, remaining_requested_flow)
                
                # Build flow dictionary
                flow_dict = {}
                for i in range(self.solver.num_arcs()):
                    flow = int(self.solver.flow(i))
                    if flow > 0:
                        u = self.index_to_node[self.solver.tail(i)]
                        v = self.index_to_node[self.solver.head(i)]
                        if u not in flow_dict:
                            flow_dict[u] = {}
                        flow_dict[u][v] = flow

                # Combine with direct flows
                if direct_flow_dict:
                    for u, flows in direct_flow_dict.items():
                        if u not in flow_dict:
                            flow_dict[u] = flows.copy()
                        else:
                            for v, f in flows.items():
                                flow_dict[u][v] = flow_dict[u].get(v, 0) + f

                return flow_value + direct_flow, flow_dict
            else:
                raise RuntimeError("OR-Tools solver failed to find optimal solution")
        finally:
            # Always restore original capacities
            for i, capacity in original_capacities.items():
                self.solver.set_arc_capacity(i, capacity)
        
    def compute_flow(self, source: str, sink: str, flow_func: Optional[Callable] = None, 
                requested_flow: Optional[str] = None) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """
        Compute maximum flow between source and sink nodes using OR-Tools.
        """
        if not self.has_vertex(source) or not self.has_vertex(sink):
            raise ValueError(f"Source node '{source}' or sink node '{sink}' not in graph.")
            
        source_idx = self.node_to_index[source]
        sink_idx = self.node_to_index[sink]
        
        if self.solver.num_arcs() == 0:
            print("No edges in graph. No flow is possible.")
            return 0, {}

        # Process direct paths using pre-built adjacency maps
        direct_flow = 0
        direct_flow_dict = {}
        max_flow = float('inf') if requested_flow is None else int(requested_flow)
        
        # Find direct paths efficiently using cached adjacency maps
        source_edges = self.arc_adjacency.get(source_idx, [])
        for src_arc_idx, intermediate_idx, source_capacity in source_edges:
            intermediate_node = self.index_to_node[intermediate_idx]
            
            # Check if this is an intermediate node
            if '_' in intermediate_node:
                # Check for connection to sink
                sink_edges = self.arc_adjacency.get(intermediate_idx, [])
                for sink_arc_idx, target_idx, sink_capacity in sink_edges:
                    if target_idx == sink_idx:
                        # Get current capacities from solver
                        current_source_cap = self.solver.capacity(src_arc_idx)
                        current_sink_cap = self.solver.capacity(sink_arc_idx)
                        
                        # Calculate flow through this path
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
                                print(f"Satisfied requested flow of {requested_flow} with direct edges.")
                                return direct_flow, direct_flow_dict
                        break  # Found connection to sink, no need to check other edges

        # Calculate remaining flow needed
        remaining_requested_flow = None if requested_flow is None else max_flow - direct_flow

        # Solve max flow
        start = time.time()
        status = self.solver.solve(source_idx, sink_idx)
        print("Solver Time: ", time.time() - start)
        
        if status == self.solver.OPTIMAL:
            flow_value = int(self.solver.optimal_flow())
            if remaining_requested_flow is not None:
                flow_value = min(flow_value, remaining_requested_flow)
            
            start = time.time()
            # Build flow dictionary efficiently
            flow_dict = {}
            for i in range(self.solver.num_arcs()):
                flow = int(self.solver.flow(i))
                if flow > 0:
                    u = self.index_to_node[self.solver.tail(i)]
                    v = self.index_to_node[self.solver.head(i)]
                    flow_dict.setdefault(u, {})[v] = flow
            print("flow-dict Time: ", time.time() - start)

            # Combine with direct flows if any
            if direct_flow_dict:
                for u, flows in direct_flow_dict.items():
                    if u not in flow_dict:
                        flow_dict[u] = flows.copy()
                    else:
                        flow_dict[u].update(flows)

            return flow_value + direct_flow, flow_dict
        else:
            raise RuntimeError("OR-Tools solver failed to find optimal solution")

    def flow_decomposition(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str, 
                          requested_flow: Optional[int] = None) -> Tuple[List[Tuple[List[str], List[str], int]], Dict[Tuple[str, str], int]]:
        """
        Decompose the flow into paths from source to sink.
        
        Args:
            flow_dict: Flow dictionary from compute_flow
            source: Source node ID
            sink: Sink node ID
            requested_flow: Optional flow limit
            
        Returns:
            Tuple containing:
            - List of (path, path_labels, flow_value) tuples
            - Dictionary of edge flows
        """
        paths = []
        edge_flows = {}
        current_flow = 0
        
        # Build residual flow graph
        residual_flow = {u: dict(flows) for u, flows in flow_dict.items()}
        
        while True:
            # Find a path from source to sink with positive flow
            path = self._find_flow_path(residual_flow, source, sink)
            if not path:
                break
                
            # Find minimum flow along the path
            path_flow = min(residual_flow[u][v] for u, v in zip(path[:-1], path[1:]))
            
            if requested_flow is not None:
                remaining_flow = requested_flow - current_flow
                if remaining_flow <= 0:
                    break
                path_flow = min(path_flow, remaining_flow)
            
            # Extract path labels
            path_labels = []
            for u, v in zip(path[:-1], path[1:]):
                edge_data = self.get_edge_data(u, v)
                path_labels.append(edge_data.get('label', 'no_label'))
                
                # Update edge flows
                edge_flows[(u, v)] = edge_flows.get((u, v), 0) + path_flow
            
            # Update residual graph
            for u, v in zip(path[:-1], path[1:]):
                residual_flow[u][v] -= path_flow
                if residual_flow[u][v] == 0:
                    del residual_flow[u][v]
                if not residual_flow[u]:
                    del residual_flow[u]
            
            paths.append((path, path_labels, path_flow))
            current_flow += path_flow
            
            if requested_flow is not None and current_flow >= requested_flow:
                break
        
        return paths, edge_flows

    def _find_flow_path(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str) -> List[str]:
        """Find a path with positive flow from source to sink."""
        visited = {source}
        path = [source]
        stack = [(source, iter(flow_dict.get(source, {}).items()))]
        
        while stack:
            current, edges = stack[-1]
            try:
                next_node, flow = next(edges)
                if flow > 0 and next_node not in visited:
                    if next_node == sink:
                        path.append(next_node)
                        return path
                    visited.add(next_node)
                    path.append(next_node)
                    stack.append((next_node, iter(flow_dict.get(next_node, {}).items())))
            except StopIteration:
                stack.pop()
                if path:
                    path.pop()
        
        return []

    def simplified_flow_decomposition(self, original_paths: List[Tuple[List[str], List[str], int]]) -> List[Tuple[List[str], List[str], int]]:
        """Create simplified paths by removing intermediate nodes."""
        simplified_paths = []
        for path, labels, flow in original_paths:
            simplified_path = []
            simplified_labels = []
            current_token = None
            
            for i, (node, label) in enumerate(zip(path, labels + [None])):
                if '_' not in node:
                    if current_token is None or label != current_token:
                        if simplified_path:
                            simplified_path.append(node)
                            simplified_labels.append(current_token)
                        else:
                            simplified_path = [node]
                        current_token = label
                    elif i == len(path) - 1:
                        simplified_path.append(node)
                        simplified_labels.append(current_token)
                        
            if len(simplified_path) > 1:
                simplified_paths.append((simplified_path, simplified_labels, flow))
        
        return simplified_paths