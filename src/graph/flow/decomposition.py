from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from .utils import find_flow_path, update_residual_graph

def decompose_flow(flow_dict: Dict[str, Dict[str, int]], source: str, sink: str,
                  requested_flow: Optional[int] = None) -> Tuple[List[Tuple[List[str], List[str], int]], 
                                                              Dict[Tuple[str, str], int]]:
    """Decompose a flow into paths."""
    paths = []
    edge_flows = {}
    current_flow = 0
    
    # Build residual flow graph
    residual_flow = {u: dict(flows) for u, flows in flow_dict.items()}
    
    while True:
        # Find a path from source to sink with positive flow
        path = find_flow_path(residual_flow, source, sink)
        if not path:
            break
        
        # Calculate path flow
        path_flow = min(residual_flow[u][v] for u, v in zip(path[:-1], path[1:]))
        
        # Apply flow limit if requested
        if requested_flow is not None:
            remaining_flow = requested_flow - current_flow
            if remaining_flow <= 0:
                break
            path_flow = min(path_flow, remaining_flow)
        
        # Update flows
        for u, v in zip(path[:-1], path[1:]):
            edge_flows[(u, v)] = edge_flows.get((u, v), 0) + path_flow
        
        # Update residual graph
        update_residual_graph(residual_flow, path, path_flow)
        
        paths.append((path, [], path_flow))
        current_flow += path_flow
        
        if requested_flow is not None and current_flow >= requested_flow:
            break
    
    return paths, edge_flows

def simplify_paths(original_paths: List[Tuple[List[str], List[str], int]]) -> List[Tuple[List[str], List[str], int]]:
    """Simplify paths by removing intermediate nodes."""
    simplified_paths = []
    
    for path, labels, flow in original_paths:
        simplified_path = []
        simplified_labels = []
        current_token = None
        last_real_node = None
        
        for i, (node, label) in enumerate(zip(path, labels)):
            if '_' not in node:
                if last_real_node is None:
                    simplified_path.append(node)
                    last_real_node = node
                    current_token = label
                elif label != current_token:
                    if node != last_real_node:
                        simplified_path.append(last_real_node)
                        simplified_labels.append(current_token)
                        simplified_path.append(node)
                        current_token = label
                    last_real_node = node
        
        if last_real_node and last_real_node != path[-1] and '_' not in path[-1]:
            simplified_path.append(path[-1])
            simplified_labels.append(current_token)
        
        if len(simplified_path) > 1:
            simplified_paths.append((simplified_path, simplified_labels, flow))
    
    return simplified_paths

__all__ = ['decompose_flow', 'simplify_paths']