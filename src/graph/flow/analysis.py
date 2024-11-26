from typing import List, Tuple, Dict, Callable, Optional
from networkx.algorithms.flow import preflow_push
from graph_tool.flow import push_relabel_max_flow
import logging
from collections import defaultdict

from ..base import BaseGraph
from .decomposition import decompose_flow, simplify_paths

# Configure logging for the module
logger = logging.getLogger(__name__)

class NetworkFlowAnalysis:
    """Handle flow analysis for all graph implementations."""
    
    def __init__(self, graph: BaseGraph):
        self.graph = graph
        self.logger = logging.getLogger(__name__)

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

    def _simplify_edge_flows(self, edge_flows: Dict[Tuple[str, str], int]) -> Dict[Tuple[str, str], Dict[str, int]]:
        """Group flows by token for each edge."""
        simplified = {}
        
        for (u, v), flow in edge_flows.items():
            # Skip virtual sink edges
            if v is not None and str(v).startswith('virtual_sink_'):
                continue
                
            # Convert intermediate node flows to token flows
            if '_' in str(u):
                real_u, token = str(u).split('_', 1)
                if (real_u, v) not in simplified:
                    simplified[(real_u, v)] = {}
                simplified[(real_u, v)][token] = (
                    simplified[(real_u, v)].get(token, 0) + flow
                )
                
        return simplified
    

    def analyze_arbitrage(self, start_node: str, start_token: str, end_token: str,  
                     flow_func: Optional[Callable] = None) -> Tuple[int, List, Dict, Dict]:
        """Analyze arbitrage opportunities."""
        try:
            source, virtual_sink = self.graph.prepare_arbitrage_graph(
                start_node, start_token, end_token
            )
            
            if source is None or virtual_sink is None:
                return 0, [], {}, {}

            # Compute max flow
            flow_value, flow_dict = self.graph.compute_flow(
                source, 
                virtual_sink,
                flow_func
            )
            
            if flow_value == 0:
                return 0, [], {}, {}
                
            self.logger.info(f"Found max flow: {flow_value}")
            
            # Find flows to virtual sink
            sink_flows = []
            for end_pos, flows in flow_dict.items():
                if virtual_sink in flows and flows[virtual_sink] > 0:
                    sink_flows.append((end_pos, flows[virtual_sink]))
            
            paths = []
            edge_flows = {}
            
            for end_pos, final_flow in sink_flows:
                # No need to add start_pos as we start from source node
                path = self._find_flow_path(flow_dict, source, None, end_pos, final_flow)
                
                if path:
                    # Complete cycle back to source
                    path.append(source)
                    
                    # Extract tokens from node IDs
                    tokens = []
                    for node in path[1:-1]:  # Skip first and last (source nodes)
                        if '_' in node:
                            _, token = node.split('_')
                            tokens.append(token)
                    
                    # Record flows
                    for i in range(len(path)-1):
                        edge = (path[i], path[i+1])
                        edge_flows[edge] = edge_flows.get(edge, 0) + final_flow
                    
                    paths.append((path, tokens, final_flow))
            
            if not paths:
                return flow_value, [], {}, {}
                
            # Create simplified paths
            simplified_paths = simplify_paths(paths)
            simplified_flows = self._simplify_edge_flows(edge_flows)
            
            return flow_value, simplified_paths, simplified_flows, edge_flows
            
        finally:
            self.graph.cleanup_arbitrage_graph()
            
    def _find_flow_path(self, flow_dict: Dict[str, Dict[str, int]], 
                    source: str, start_pos: str, end_pos: str, 
                    min_flow: int) -> List[str]:
        """Find path with sufficient flow from start to end."""
        # Start with source node only
        path = [source]
        current = source
        visited = {source}
        
        def get_next_node(curr_node: str, visited_nodes: set) -> Optional[str]:
            """Get next valid node in path with sufficient flow."""
            for next_node, flow in flow_dict.get(curr_node, {}).items():
                # Check flow is sufficient and node not already visited
                if flow >= min_flow and next_node not in visited_nodes:
                    return next_node
            return None

        # Build path step by step
        while current != end_pos:
            next_node = get_next_node(current, visited)
            if next_node:
                # Add node to path
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            else:
                # Backtrack if no valid next node
                if len(path) <= 1:
                    return []  # No valid path found
                path.pop()
                current = path[-1]
                    
        return path
