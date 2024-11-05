import networkx as nx
from graph_tool import Graph
from graph_tool.flow import edmonds_karp_max_flow, push_relabel_max_flow, boykov_kolmogorov_max_flow
from graph_tool.search import dfs_search, DFSVisitor
from graph_tool.util import find_edge
from typing import List, Tuple, Dict, Callable, Optional
import time
from collections import defaultdict, deque

class GraphCreator:
    @staticmethod
    def create_graph(graph_type: str, edges: List[Tuple[str, str]], capacities: List[float], tokens: List[str]):
        if graph_type == 'networkx':
            return NetworkXGraph(edges, capacities, tokens)
        elif graph_type == 'graph_tool':
            return GraphToolGraph(edges, capacities, tokens)
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")


class BaseGraph:
    def has_vertex(self, vertex_id: str) -> bool:
        raise NotImplementedError("Subclass must implement abstract method")

    def has_edge(self, u: str, v: str) -> bool:
        raise NotImplementedError("Subclass must implement abstract method")

    def get_edge_data(self, u: str, v: str) -> Dict:
        raise NotImplementedError("Subclass must implement abstract method")


class NetworkXGraph(BaseGraph):
    def __init__(self, edges: List[Tuple[str, str]], capacities: List[float], tokens: List[str]):
        self.g_nx = self._create_graph(edges, capacities, tokens)

    def _create_graph(self, edges: List[Tuple[str, str]], capacities: List[float], tokens: List[str]) -> nx.DiGraph:
        g = nx.DiGraph()
        for (u, v), capacity, token in zip(edges, capacities, tokens):
            g.add_edge(u, v, capacity=int(capacity), label=token)

        import json
        from networkx.readwrite import json_graph
        with open('networkdata1.json', 'w') as outfile1:
            outfile1.write(json.dumps(json_graph.node_link_data(g)))
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

    def _find_path(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str) -> List[str]:
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

    
    def flow_decomposition(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str, requested_flow: Optional[int] = None) -> Tuple[List[Tuple[List[str], List[str], int]], Dict[Tuple[str, str], int]]:
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

    def simplified_flow_decomposition(self, original_paths: List[Tuple[List[str], List[str], int]]) -> List[Tuple[List[str], List[str], int]]:
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
    
    def simplified_flow_decomposition2(self, original_paths: List[Tuple[List[str], List[str], int]]) -> List[Tuple[List[str], List[str], int]]:
        simplified_paths = []
        for path, labels, flow in original_paths:
            simplified_path = [node for node in path if '_' not in node]
            simplified_labels = [label for node, label in zip(path[1:], labels) if '_' not in node]
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

    def _create_graph(self, edges: List[Tuple[str, str]], capacities: List[int], tokens: List[str]) -> Graph:
        g = Graph(directed=True)
        v_prop = g.new_vertex_property("string")
        e_prop_capacity = g.new_edge_property("int64_t")
        e_prop_token = g.new_edge_property("string")

        vertex_map = {}

        for (u_id, v_id), capacity, token in zip(edges, capacities, tokens):
            if u_id not in vertex_map:
                u = g.add_vertex()
                vertex_map[u_id] = u
                v_prop[u] = u_id
            else:
                u = vertex_map[u_id]

            if v_id not in vertex_map:
                v = g.add_vertex()
                vertex_map[v_id] = v
                v_prop[v] = v_id
            else:
                v = vertex_map[v_id]

            e = g.add_edge(u, v)
            e_prop_capacity[e] = int(capacity)
            e_prop_token[e] = token

        g.vertex_properties["id"] = v_prop
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
        if flow_func is None:
            flow_func = push_relabel_max_flow
            
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
            res = flow_func(self.g_gt, s, t, capacity_copy)
            
            # Compute actual flows
            flow = capacity_copy.copy()
            flow.a = capacity_copy.a - res.a

            # Calculate total flow and create flow dictionary
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

    

    def flow_decomposition(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str, requested_flow: Optional[int] = None, method: str = 'bfs') -> Tuple[List[Tuple[List[str], List[str], int]], Dict[Tuple[str, str], int]]:
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
    
    def simplified_flow_decomposition(self, original_paths: List[Tuple[List[str], List[str], int]]) -> List[Tuple[List[str], List[str], int]]:
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

    def simplified_flow_decomposition2(self, original_paths: List[Tuple[List[str], List[str], int]]) -> List[Tuple[List[str], List[str], int]]:
        simplified_paths = []
        for path, labels, flow in original_paths:
            simplified_path = [node for node in path if '_' not in node]
            simplified_labels = [label for node, label in zip(path[1:], labels) if '_' not in node]
            if len(simplified_path) > 1:
                simplified_paths.append((simplified_path, simplified_labels, flow))
        return simplified_paths