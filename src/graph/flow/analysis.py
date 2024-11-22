from typing import List, Tuple, Dict, Callable, Optional
from networkx.algorithms.flow import preflow_push
from graph_tool.flow import push_relabel_max_flow

from ..base import BaseGraph

class NetworkFlowAnalysis:
    """Handle flow analysis for all graph implementations."""
    
    def __init__(self, graph: BaseGraph):
        self.graph = graph

    def analyze_flow(self, source: str, sink: str, flow_func: Optional[Callable] = None, 
                    requested_flow: Optional[str] = None):
        """
        Analyze flow between source and sink nodes.
        
        Args:
            source: Source node ID
            sink: Sink node ID
            flow_func: Flow algorithm to use (optional)
            requested_flow: Maximum flow to compute (optional)
            
        Returns:
            Tuple containing:
            - Flow value
            - Simplified paths
            - Simplified edge flows
            - Original edge flows
        """
        # Get appropriate flow algorithm if none provided
        if flow_func is None:
            flow_func = self._get_default_algorithm()

        # Compute flow only once
        print(f"Computing flow from {source} to {sink}")
        if flow_func:
            print(f"Flow function: {flow_func.__name__}")
            
        flow_value, flow_dict = self.graph.compute_flow(
            source, 
            sink, 
            flow_func, 
            requested_flow
        )

        # Decompose into paths
        paths, edge_flows = self.graph.flow_decomposition(
            flow_dict, 
            source, 
            sink, 
            int(requested_flow) if requested_flow else None
        )
        
        # Create simplified paths and edge flows
        simplified_paths = self.graph.simplified_flow_decomposition(paths)
        simplified_edge_flows = self._simplify_edge_flows(edge_flows)
        
        return flow_value, simplified_paths, simplified_edge_flows, edge_flows

    def _get_default_algorithm(self) -> Optional[Callable]:
        """Get default flow algorithm based on graph implementation."""
        graph_type = self._get_graph_type()
        if graph_type == 'networkx':
            return preflow_push
        elif graph_type == 'graph_tool':
            return push_relabel_max_flow
        return None  # OR-Tools uses its own algorithm

    def _get_graph_type(self) -> str:
        """Determine graph implementation type."""
        module_name = self.graph.__class__.__module__
        if 'networkx' in module_name:
            return 'networkx'
        elif 'graph_tool' in module_name:
            return 'graph_tool'
        return 'ortools'

    def _simplify_edge_flows(self, edge_flows: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Simplify edge flows by combining flows through intermediate nodes."""
        simplified_edge_flows = {}
        for (u, v), flow in edge_flows.items():
            if '_' in u and '_' not in v:
                real_u, token = u.split('_')
                if (real_u, v) not in simplified_edge_flows:
                    simplified_edge_flows[(real_u, v)] = {}
                simplified_edge_flows[(real_u, v)][token] = (
                    simplified_edge_flows[(real_u, v)].get(token, 0) + flow
                )
        return simplified_edge_flows