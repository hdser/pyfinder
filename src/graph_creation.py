import networkx as nx
from typing import List, Tuple, Dict, Callable, Optional
import json
import time

class NetworkXGraph:
    def __init__(self, edges: List[Tuple[str, str]], capacities: List[float], tokens: List[str]):
        self.g_nx = self._create_graph(edges, capacities, tokens)

    def _create_graph(self, edges: List[Tuple[str, str]], capacities: List[float], tokens: List[str]) -> nx.DiGraph:
        g = nx.DiGraph()
        for (u, v), capacity, token in zip(edges, capacities, tokens):
            g.add_edge(u, v, capacity=capacity, label=token)
        return g


    def compute_flow(self, source: str, sink: str, flow_func: Optional[Callable] = None, requested_flow: Optional[str] = None) -> Tuple[int, Dict[str, Dict[str, int]]]:
        if flow_func is None:
            flow_func = nx.algorithms.flow.preflow_push

        print('Started Flow Computation...')
        try:
            requested_flow_int = int(requested_flow)
            flow_value, flow_dict = nx.maximum_flow(self.g_nx, source, sink, flow_func=flow_func, cutoff=requested_flow_int)
        except:
            flow_value, flow_dict = nx.maximum_flow(self.g_nx, source, sink, flow_func=flow_func)
        #flow_value, flow_dict = nx.maximum_flow(self.g_nx, source, sink, flow_func=flow_func)
        print('Ended Flow Computation...')

       # with open('output/flow_dic.json', 'w') as f:
       #     json.dump(
       #         flow_dict, 
       #         f, 
       #         default=lambda obj: str(obj) if isinstance(obj, int) else TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable"))

        if requested_flow is not None:
            if flow_value > requested_flow_int:
                flow_value = requested_flow_int
                flow_dict = self._limit_flow(flow_dict, source, sink, requested_flow_int)

        return flow_value, flow_dict

    def _limit_flow(self, flow_dict: Dict[str, Dict[str, float]], source: str, sink: str, limit: int) -> Dict[str, Dict[str, int]]:
        limited_flow_dict = {node: {} for node in flow_dict}
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
                if v not in flow_dict or u not in flow_dict[v]:
                    flow_dict[v][u] = path_flow
                else:
                    flow_dict[v][u] += path_flow

                if v not in limited_flow_dict[u]:
                    limited_flow_dict[u][v] = int(0)
                limited_flow_dict[u][v] += int(path_flow)

            remaining_flow -= path_flow

        return limited_flow_dict
    

    def _find_path(self, flow_dict: Dict[str, Dict[str, float]], source: str, sink: str) -> List[str]:
        stack = [(source, [source])]
        visited = set()

        while stack:
            (node, path) = stack.pop()
            if node not in visited:
                if node == sink:
                    return path
                visited.add(node)
                for next_node in flow_dict[node]:
                    if flow_dict[node][next_node] > 0:
                        stack.append((next_node, path + [next_node]))
        return []

    def flow_decomposition(self, flow_dict: Dict[str, Dict[str, int]], source: str, sink: str) -> Tuple[List[Tuple[List[str], List[str], int]], Dict[Tuple[str, str], int]]:
        paths = []
        edge_flows = {}
        
        # Create a deep copy of flow_dict
        remaining_flow = {u: {v: flow for v, flow in flows.items()} for u, flows in flow_dict.items()}
        
        def find_path_iterative(source, sink):
            infinity = 10**38
            stack = [(source, [source], infinity)]
            visited = set()

            while stack:
                node, path, flow = stack.pop()
                
                if node == sink:
                    return path, flow

                if node in visited:
                    continue
                visited.add(node)

                for next_node, flow_value in remaining_flow[node].items():
                    if flow_value > 0 and next_node not in visited:
                        new_flow = min(flow, flow_value)
                        stack.append((next_node, path + [next_node], new_flow))

            return None

        while True:
            result = find_path_iterative(source, sink)
            if not result:
                break
            path, path_flow = result
            
            #print("Found path:", path)
            path_labels = []
            for u, v in zip(path[:-1], path[1:]):
                label = self.g_nx[u][v].get('label', 'no_label')
                path_labels.append(label)
                print(f"Edge {u} -> {v}: label = {label}")
            
            paths.append((path, path_labels, path_flow))
            
            # Update remaining_flow instead of flow_dict
            for u, v in zip(path[:-1], path[1:]):
                remaining_flow[u][v] -= path_flow
                if remaining_flow[u][v] == 0:
                    del remaining_flow[u][v]
                
                # Update edge_flows
                edge_flows[(u, v)] = edge_flows.get((u, v), int(0)) + path_flow

        return paths, edge_flows


    def simplified_flow_decomposition(self, original_paths: List[Tuple[List[str], List[str], int]]) -> List[Tuple[List[str], List[str], int]]:
        simplified_paths = []
        for path, labels, flow in original_paths:
            simplified_path = []
            simplified_labels = []
            for i, node in enumerate(path):
                if '_' not in node:
                    simplified_path.append(node)
                    if i < len(path) - 1 and '_' not in path[i+1]:
                        simplified_labels.append(labels[i])
            
            if len(simplified_path) > 1:
                simplified_paths.append((simplified_path, simplified_labels, flow))

        return simplified_paths