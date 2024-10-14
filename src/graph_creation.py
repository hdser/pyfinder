import networkx as nx
from graph_tool import Graph
from graph_tool.flow import edmonds_karp_max_flow, push_relabel_max_flow, boykov_kolmogorov_max_flow
from graph_tool.search import dfs_search, DFSVisitor
from graph_tool.util import find_edge
from typing import List, Tuple, Dict, Callable, Optional
import time
from collections import defaultdict, deque
import heapq

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

import networkx as nx
from typing import List, Tuple, Dict, Callable, Optional
from collections import defaultdict, deque

class NetworkXGraph(BaseGraph):
    def __init__(self, edges: List[Tuple[str, str]], capacities: List[float], tokens: List[str]):
        self.g_nx = self._create_graph(edges, capacities, tokens)

    def _create_graph(self, edges: List[Tuple[str, str]], capacities: List[float], tokens: List[str]) -> nx.DiGraph:
        g = nx.DiGraph()
        for (u, v), capacity, token in zip(edges, capacities, tokens):
            g.add_edge(u, v, capacity=int(capacity), label=token)
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
        if flow_func is None:
            flow_func = nx.algorithms.flow.preflow_push

        print('Started Flow Computation...')
        try:
            requested_flow_int = int(requested_flow) if requested_flow else None
            flow_value, flow_dict = nx.maximum_flow(self.g_nx, source, sink, flow_func=flow_func, cutoff=requested_flow_int)
            flow_value = int(flow_value)
            flow_dict = {u: {v: int(f) for v, f in flows.items()} for u, flows in flow_dict.items()}
        except Exception as e:
            print(f"Error in flow computation: {str(e)}")
            raise

        print('Ended Flow Computation...')

        if requested_flow_int and flow_value > requested_flow_int:
            flow_value = requested_flow_int
            flow_dict = self._limit_flow(flow_dict, source, sink, requested_flow_int)

        return flow_value, flow_dict

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
            simplified_path = [node for node in path if '_' not in node]
            simplified_labels = [label for node, label in zip(path[1:], labels) if '_' not in node]
            if len(simplified_path) > 1:
                simplified_paths.append((simplified_path, simplified_labels, flow))
        return simplified_paths


class PathVisitor(DFSVisitor):
    def __init__(self, residual_capacity, target):
        self.residual_capacity = residual_capacity
        self.target = target
        self.path = []
        self.found = False

    def discover_vertex(self, u):
        self.path.append(u)
        if u == self.target:
            self.found = True
            raise StopIteration()

    def examine_edge(self, e):
        if self.residual_capacity[e] == 0:
            raise ValueError() 


class GraphToolGraph(BaseGraph):
    def __init__(self, edges: List[Tuple[str, str]], capacities: List[int], tokens: List[str]):
        self.g_gt = self._create_graph(edges, capacities, tokens)
        self.vertex_index = self.g_gt.vertex_index
        self.edge_index = self.g_gt.edge_index
        self.capacity = self.g_gt.edge_properties["capacity"]
        self.token = self.g_gt.edge_properties["token"]
        self.vertex_id = self.g_gt.vertex_properties["id"]
        self.id_to_vertex = {self.vertex_id[v]: v for v in self.g_gt.vertices()}

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
        s = self.get_vertex(source)
        t = self.get_vertex(sink)

        if s is None or t is None:
            raise ValueError(f"Source node '{source}' or sink node '{sink}' not in graph.")

        if flow_func is None:
            flow_func = push_relabel_max_flow
        elif flow_func == edmonds_karp_max_flow:
            flow_func = edmonds_karp_max_flow
        elif flow_func == boykov_kolmogorov_max_flow:
            flow_func = boykov_kolmogorov_max_flow

        print(f"Computing flow from {source} to {sink}")
        print(f"Number of vertices: {self.g_gt.num_vertices()}")
        print(f"Number of edges: {self.g_gt.num_edges()}")
        print(f"Flow function: {flow_func.__name__}")

        res = flow_func(self.g_gt, s, t, self.capacity)

        flow = self.capacity.copy()
        flow.a = self.capacity.a - res.a  # Compute the actual flow

        flow_value = int(sum(flow[e] for e in t.in_edges()))  # Ensure integer flow value
        print(f"Total flow value: {flow_value}")

        if requested_flow is not None:
            max_flow = int(requested_flow)
            if flow_value > max_flow:
                flow_value = max_flow
                print(f"Adjusted flow to requested value: {flow_value}")

        # Store the residual graph for later use in flow_decomposition
        self.residual_capacity = res

        # Create flow_dict with integer flows
        flow_dict = defaultdict(lambda: defaultdict(int))
        for e in self.g_gt.edges():
            f = int(flow[e])
            if f > 0:
                u = self.vertex_id[e.source()]
                v = self.vertex_id[e.target()]
                flow_dict[u][v] += f

        return flow_value, dict(flow_dict)


    def flow_decomposition(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str, requested_flow: Optional[int] = None, method: str = 'bfs') -> Tuple[List[Tuple[List[str], List[str], int]], Dict[Tuple[str, str], int]]:
        """
        Decompose the flow into paths using either BFS or DFS.

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

        total_flow = sum(flow_dict[source].values())
        if requested_flow is None or requested_flow > total_flow:
            requested_flow = total_flow

        paths = []
        edge_flows = defaultdict(int)
        current_flow = 0

        # First, process direct edge from source to sink if any
        if sink in flow_dict.get(source, {}):
            direct_flow = flow_dict[source][sink]
            path = [source, sink]
            e = self.g_gt.edge(self.get_vertex(source), self.get_vertex(sink))
            path_labels = [self.token[e]]
            paths.append((path, path_labels, direct_flow))
            edge_flows[(source, sink)] += direct_flow
            # Update flow_dict
            del flow_dict[source][sink]
            if not flow_dict[source]:
                del flow_dict[source]
            current_flow += direct_flow

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
        queue = deque([(source, [source], float('inf'))])
        visited = set()
        while queue:
            node, path, flow = queue.popleft()
            if node == sink:
                return path, flow
            visited.add(node)
            for next_node, edge_flow in residual_flow.get(node, {}).items():
                if edge_flow > 0 and next_node not in visited:
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
            simplified_path = [node for node in path if '_' not in node]
            simplified_labels = [label for node, label in zip(path[1:], labels) if '_' not in node]
            if len(simplified_path) > 1:
                simplified_paths.append((simplified_path, simplified_labels, flow))
        return simplified_paths