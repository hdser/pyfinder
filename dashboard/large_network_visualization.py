import networkx as nx
import numpy as np
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, 
    HoverTool, 
    Circle,
)
from typing import Dict, Optional, Tuple
import random
from collections import defaultdict

import community
from bokeh.palettes import Category20  # For cluster colors

class LargeNetworkVisualization:
    def __init__(self):
        from bokeh.models import ColumnDataSource
        
        self.node_tooltips = [
            ("Node", "@node"),
            ("Type", "@type"),
            ("Address", "@address"),
            ("Connections", "@connections")
        ]
        self.edge_tooltips = [
            ("From", "@from_node"),
            ("To", "@to_node"),
            ("Token", "@token")
        ]
        
        # Node limits mapping
        self.node_limits = {
            'Very Low': 100,
            'Low': 500,
            'Medium': 1000,
            'High': 2000,
            'Full': None
        }
        
        # Initialize data sources
        self.node_source = ColumnDataSource({
            'x': [], 'y': [], 
            'node': [], 'type': [], 
            'address': [], 'color': [], 
            'size': [], 'connections': [], 
            'alpha': [], 'cluster': []
        })
        
        self.edge_source = ColumnDataSource({
            'xs': [], 'ys': [],
            'from_node': [], 'to_node': [],
            'token': [], 'line_width': [],
            'alpha': []
        })
        
        self.current_node_limit = self.node_limits['Very Low']
        self.layout_cache = {}
        self.current_layout = None
        self.current_visible_nodes = set()
        self.id_to_address = {}
        self.plot = None

    def _filter_connected_nodes(self, G: nx.DiGraph) -> nx.DiGraph:
        """Filter out nodes with no meaningful connections."""
        connected_nodes = []
        
        for node in G.nodes():
            # For intermediate nodes (those with '_'), check if they're part of any valid path
            if '_' in str(node):
                if G.in_degree(node) > 0 and G.out_degree(node) > 0:
                    connected_nodes.append(node)
            else:
                # For regular nodes, check total degree (in + out)
                if G.in_degree(node) + G.out_degree(node) > 0:
                    connected_nodes.append(node)
        
        # Create subgraph with only connected nodes
        return G.subgraph(connected_nodes)

    def set_node_limit(self, level_key: str):
        """Set the current node limit based on the key."""
        if level_key not in self.node_limits:
            raise ValueError(f"Invalid level key: {level_key}")
        self.current_node_limit = self.node_limits[level_key]
        print(f"Node limit set to: {self.current_node_limit}")

    def _create_plot(self, title: str) -> figure:
        """Create a Bokeh plot with appropriate settings."""
        from bokeh.plotting import figure
        
        plot = figure(
            title=title,
            tools="pan,box_zoom,wheel_zoom,reset,save",
            active_scroll=None,
            sizing_mode='stretch_width',
            height=500
        )
        
        # Style the plot
        plot.title.text_font_size = '16px'
        plot.title.align = 'center'
        plot.axis.visible = False
        plot.grid.visible = False
        plot.outline_line_color = None
        plot.background_fill_color = "#ffffff"
        
        return plot

    def _select_important_nodes(self, G: nx.DiGraph, importance_scores: Dict[str, float]) -> set:
        """Select most important nodes based on scores and current limit."""
        # Calculate total degree (in + out) for each node
        degree_dict = {
            node: G.in_degree(node) + G.out_degree(node)
            for node in G.nodes()
        }
        
        # Filter nodes with actual connections
        connected_nodes = [
            node for node, degree in degree_dict.items()
            if degree > 0 or ('_' in str(node) and G.in_degree(node) > 0 and G.out_degree(node) > 0)
        ]
        
        if self.current_node_limit is None or len(connected_nodes) <= self.current_node_limit:
            return set(connected_nodes)

        # Sort nodes by importance, only considering connected nodes
        sorted_nodes = sorted(
            [(node, importance_scores.get(node, 0)) 
            for node in connected_nodes],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top nodes (70%) and random nodes (30%) to reach the limit
        top_count = int(self.current_node_limit * 0.7)
        random_count = self.current_node_limit - top_count
        
        important_nodes = {node for node, _ in sorted_nodes[:top_count]}
        other_nodes = set(connected_nodes) - important_nodes
        
        if other_nodes and random_count > 0:
            random_nodes = set(random.sample(list(other_nodes), min(random_count, len(other_nodes))))
            return important_nodes | random_nodes
        return important_nodes

    def _get_initial_nodes(self, G: nx.DiGraph, percentage: int) -> set:
        """Get initial set of important nodes for first load."""
        # Exclude isolated nodes
        connected_nodes = [node for node, degree in G.degree() if degree > 0]
        total_nodes = len(connected_nodes)
        sample_size = int((percentage / 100.0) * total_nodes)

        # Use degree as a quick importance metric
        degree_dict = {node: G.degree(node) for node in connected_nodes}
        sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Take some high-degree nodes
        top_count = int(sample_size * 0.7)
        top_nodes = set(node for node, _ in sorted_nodes[:top_count])
        
        # Add some random nodes for diversity
        other_nodes = set(connected_nodes) - top_nodes
        random_count = sample_size - top_count
        if other_nodes:
            random_nodes = set(random.sample(list(other_nodes), min(random_count, len(other_nodes))))
            return top_nodes | random_nodes
        return top_nodes

    def _create_subgraph_with_context(self, G: nx.DiGraph, nodes: set, 
                                      max_context_nodes: Optional[int] = None) -> nx.DiGraph:
        """Create a subgraph with selected nodes and their immediate context."""
        context_nodes = set()
        
        # Add immediate neighbors efficiently
        for node in nodes:
            context_nodes.update(G.predecessors(node))
            context_nodes.update(G.successors(node))
        
        # Exclude isolated nodes
        context_nodes = {node for node in context_nodes if G.degree(node) > 0}
        
        # Limit context nodes if specified
        if max_context_nodes and len(context_nodes) > max_context_nodes:
            context_nodes = set(random.sample(list(context_nodes), max_context_nodes))
        
        # Create subgraph
        all_nodes = nodes | context_nodes
        return G.subgraph(all_nodes)

    def update_detail_level(self, level: str):
        """Update the visualization detail level."""
        self.detail_level = level  # Store the detail level
        self.initial_load = False  # Force full load on next update
        self.update_view()

    def get_visible_node_count(self) -> int:
        """Get the number of currently visible nodes."""
        return len(self.current_visible_nodes)

    def _prepare_hover_tooltips(self):
        """Create hover tooltips for nodes and edges."""
        self.node_hover = HoverTool(
            tooltips=self.node_tooltips,
            renderers=['nodes']  # Will be set later
        )
        
        self.edge_hover = HoverTool(
            tooltips=self.edge_tooltips,
            renderers=['edges']  # Will be set later
        )

    def create_initial_view(self, G: nx.DiGraph, id_to_address: Dict[str, str]) -> figure:
        """Create initial network view with clustering, excluding isolated nodes."""
        try:
            # Store the full graph and address mapping
            self.id_to_address = id_to_address
            
            # Filter out isolated nodes
            G_connected = self._filter_connected_nodes(G)
            self.full_graph = G_connected
            
            if len(G_connected) == 0:
                print("No connected nodes found in the graph")
                return self._create_empty_plot("No connected nodes to display")

            # Calculate node importance scores
            degree_dict = dict(G_connected.degree())
            max_degree = max(degree_dict.values()) if degree_dict else 1
            importance_scores = {
                node: deg / max_degree 
                for node, deg in degree_dict.items()
            }
            
            # Select important nodes based on current limit
            important_nodes = self._select_important_nodes(G_connected, importance_scores)
            
            # Create subgraph with selected nodes
            subgraph = G_connected.subgraph(important_nodes)
            
            # Detect communities for clustering
            try:
                import community
                communities = community.best_partition(subgraph.to_undirected())
            except Exception as e:
                print(f"Community detection failed: {str(e)}")
                communities = {node: 0 for node in subgraph.nodes()}
            
            # Calculate clustered layout
            pos = self._calculate_layout(subgraph, communities)
            self.current_layout = pos
            
            # Create the plot
            title = f"Network Overview ({len(subgraph)} connected nodes shown)"
            plot = self._create_plot(title)
            
            # Prepare node and edge data
            node_data = self._prepare_node_data(subgraph, pos, id_to_address, communities)
            edge_data = self._prepare_edge_data(subgraph, pos)
            
            # Add elements to plot
            self._add_graph_elements(plot, node_data, edge_data)
            
            self.current_visible_nodes = set(subgraph.nodes())
            
            return plot
            
        except Exception as e:
            print(f"Error creating initial view: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_empty_plot(f"Error: {str(e)}")

    def _create_empty_plot(self, message: str) -> figure:
        """Create an empty plot with a message."""
        plot = figure(
            title=message,
            tools="reset,save",
            sizing_mode='stretch_width',
            height=500
        )
        plot.title.text_font_size = '16px'
        plot.title.align = 'center'
        plot.axis.visible = False
        plot.grid.visible = False
        plot.outline_line_color = None
        plot.background_fill_color = "#ffffff"
        
        # Add message in the center
        plot.text(
            x=0.5, y=0.5,
            text=[message],
            text_align="center",
            text_baseline="middle",
            text_font_size="14px"
        )
        
        return plot


    def _calculate_layout(self, G: nx.DiGraph, partition: Dict[str, int]) -> Dict[str, Tuple[float, float]]:
        """Calculate layout with increased intra-cluster spacing."""
        try:
            # Use circular layout for very small graphs
            if len(G) <= 10:
                return nx.circular_layout(G, scale=30)  # Increased scale

            # Group nodes by cluster
            cluster_nodes = defaultdict(list)
            for node, cluster in partition.items():
                cluster_nodes[cluster].append(node)

            num_clusters = len(cluster_nodes)
            
            # Calculate optimal grid dimensions
            grid_width = int(np.ceil(np.sqrt(num_clusters * 1.5)))
            grid_height = int(np.ceil(num_clusters / grid_width))
            
            # Increased base spacing
            base_spacing = 50  # Increased from 30
            cluster_spacing_x = base_spacing * (1 + len(G) / 800)  # Adjusted scaling
            cluster_spacing_y = base_spacing * (1 + len(G) / 800)
            
            # Calculate cluster centers with improved spacing
            cluster_centers = {}
            max_cluster_size = max(len(nodes) for nodes in cluster_nodes.values())
            
            for i, cluster in enumerate(cluster_nodes.keys()):
                grid_x = i % grid_width
                grid_y = i // grid_width
                
                # Scale jitter based on cluster size
                cluster_size = len(cluster_nodes[cluster])
                size_factor = cluster_size / max_cluster_size
                jitter = base_spacing * 0.1 * size_factor
                
                x_jitter = random.uniform(-jitter, jitter)
                y_jitter = random.uniform(-jitter, jitter)
                
                x = grid_x * cluster_spacing_x + x_jitter
                y = grid_y * cluster_spacing_y + y_jitter
                
                cluster_centers[cluster] = (x, y)

            # Calculate final node positions with increased spacing
            pos = {}
            for cluster, nodes in cluster_nodes.items():
                if len(nodes) > 1:
                    # Create subgraph for cluster
                    subgraph = G.subgraph(nodes)
                    
                    # Adjust layout based on cluster size
                    if len(nodes) <= 3:
                        cluster_pos = nx.circular_layout(subgraph, scale=4)  # Increased scale
                    else:
                        # Increase spacing in force-directed layout
                        cluster_pos = nx.spring_layout(
                            subgraph,
                            k=4/np.sqrt(len(subgraph)),  # Increased from 2
                            iterations=150,  # More iterations for better spacing
                            scale=4,  # Explicit scale parameter
                            seed=42
                        )

                    # Increased scale factor for intra-cluster spacing
                    scale_factor = np.log2(len(nodes) + 2) * 4  # Doubled from 2
                    
                    # Apply cluster-size-based additional scaling
                    size_factor = 1 + (len(nodes) / max_cluster_size) * 0.5
                    scale_factor *= size_factor
                    
                    # Center and scale the cluster layout
                    center_x, center_y = cluster_centers[cluster]
                    for node, (x, y) in cluster_pos.items():
                        # Add some randomness to prevent overlapping
                        node_jitter = base_spacing * 0.05
                        jitter_x = random.uniform(-node_jitter, node_jitter)
                        jitter_y = random.uniform(-node_jitter, node_jitter)
                        
                        scaled_x = x * scale_factor + center_x + jitter_x
                        scaled_y = y * scale_factor + center_y + jitter_y
                        pos[node] = (scaled_x, scaled_y)
                else:
                    # Single node - place with some offset from center
                    center_x, center_y = cluster_centers[cluster]
                    offset = base_spacing * 0.2
                    pos[nodes[0]] = (
                        center_x + random.uniform(-offset, offset),
                        center_y + random.uniform(-offset, offset)
                    )

            # Post-process positions
            for node in G.nodes():
                if node not in pos:
                    closest_cluster = min(cluster_centers.items(), 
                                    key=lambda x: np.random.rand())[0]
                    center_x, center_y = cluster_centers[closest_cluster]
                    offset = base_spacing * 0.3  # Increased offset
                    pos[node] = (
                        center_x + random.uniform(-offset, offset),
                        center_y + random.uniform(-offset, offset)
                    )

            # Normalize and scale positions
            x_vals = [x for x, y in pos.values()]
            y_vals = [y for x, y in pos.values()]
            x_center = (max(x_vals) + min(x_vals)) / 2
            y_center = (max(y_vals) + min(y_vals)) / 2
            
            # Center the layout and apply final scaling
            pos = {node: (
                    (x - x_center) * 1.2,  # Additional scaling factor
                    (y - y_center) * 1.2
                ) for node, (x, y) in pos.items()}

            return pos

        except Exception as e:
            print(f"Error in layout calculation: {str(e)}")
            # Fallback to basic spring layout with increased spacing
            return nx.spring_layout(G, k=4, scale=30, iterations=150)
        
    def _add_graph_elements(self, plot: figure, node_data: dict, edge_data: dict) -> None:
        """Add nodes and edges to the plot."""
        from bokeh.models import HoverTool, Circle
        
        # Update data sources
        self.node_source.data = node_data
        self.edge_source.data = edge_data
        
        # Add edges
        edges = plot.multi_line(
            xs='xs', ys='ys',
            line_color='gray',
            line_width='line_width',
            line_alpha='alpha',
            source=self.edge_source,
            hover_line_color='#ff7f0e',
            hover_line_alpha=1.0
        )
        
        # Add nodes
        nodes = plot.scatter(
            'x', 'y',
            source=self.node_source,
            size='size',
            fill_color='color',
            line_color='black',
            alpha='alpha',
            hover_fill_color='#ff7f0e',
            hover_alpha=1.0
        )
        
        # Add hover tools
        node_hover = HoverTool(renderers=[nodes], tooltips=self.node_tooltips)
        edge_hover = HoverTool(renderers=[edges], tooltips=self.edge_tooltips)
        plot.add_tools(node_hover, edge_hover)
        
        # Add selection behavior
        nodes.selection_glyph = Circle(
            fill_color='#ff7f0e',
            line_color='black'
        )
        nodes.nonselection_glyph = Circle(
            fill_color='color',
            line_color='black',
            fill_alpha=0.1
        )

        return plot

    def _prepare_node_data(self, G: nx.DiGraph, pos: Dict[str, Tuple[float, float]], 
                      id_to_address: Dict[str, str], communities: Dict[str, int]) -> dict:
        """Prepare node data with clustering information."""
        # Calculate total degrees (in + out)
        degree_dict = {
            node: G.in_degree(node) + G.out_degree(node)
            for node in G.nodes()
        }
        max_degree = max(degree_dict.values()) if degree_dict else 1
        
        node_data = {
            'x': [], 'y': [], 'node': [], 'type': [], 'address': [],
            'color': [], 'size': [], 'connections': [], 'alpha': [],
            'cluster': []
        }
        
        colors = Category20[20]
        
        for node in G.nodes():
            # Skip nodes with no connections
            if degree_dict[node] == 0 and not ('_' in str(node) and G.in_degree(node) > 0 and G.out_degree(node) > 0):
                continue
                
            x, y = pos[node]
            node_data['x'].append(float(x))
            node_data['y'].append(float(y))
            node_data['node'].append(str(node))
            
            degree = degree_dict[node]
            node_data['connections'].append(degree)
            
            # Assign color based on community
            cluster = communities.get(node, 0)
            color = colors[cluster % 20]
            
            if '_' in str(node):
                node_data['type'].append('intermediate')
                node_data['size'].append(5)
                node_data['alpha'].append(0.5)
            else:
                node_data['type'].append('regular')
                size = 5 + (degree / max_degree) * 15
                node_data['size'].append(size)
                node_data['alpha'].append(0.8)
            
            node_data['color'].append(color)
            node_data['cluster'].append(cluster)
            node_data['address'].append(id_to_address.get(str(node), "Unknown"))
        
        return node_data

    def _prepare_edge_data(self, G: nx.DiGraph, pos: Dict[str, Tuple[float, float]]) -> dict:
        """Prepare edge data with improved routing for increased spacing."""
        edge_data = {
            'xs': [], 'ys': [],
            'from_node': [], 'to_node': [],
            'token': [], 'line_width': [], 'alpha': []
        }
        
        # Group edges by endpoints to detect multiedges
        edge_groups = defaultdict(int)
        for u, v in G.edges():
            edge_groups[(u, v)] += 1
        
        for (u, v), count in edge_groups.items():
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                # Calculate base curve parameters
                dx = x1 - x0
                dy = y1 - y0
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Adjust curvature based on distance and edge count
                # Reduced base curvature to account for increased spacing
                curve_strength = min(0.15, dist * 0.08)
                if count > 1:
                    curve_strength *= 1.3
                
                # Calculate control points
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                
                # Add perpendicular offset for curved edges
                nx = -dy / (dist if dist > 0 else 1)
                ny = dx / (dist if dist > 0 else 1)
                
                # Adjust control point placement
                control_x = mid_x + nx * curve_strength
                control_y = mid_y + ny * curve_strength
                
                # Generate curve points with more points for smoother curves
                t = np.linspace(0, 1, 30)
                xs = (1-t)**2 * x0 + 2*(1-t)*t * control_x + t**2 * x1
                ys = (1-t)**2 * y0 + 2*(1-t)*t * control_y + t**2 * y1
                
                edge_data['xs'].append(list(xs))
                edge_data['ys'].append(list(ys))
                edge_data['from_node'].append(str(u))
                edge_data['to_node'].append(str(v))
                
                data = G.get_edge_data(u, v)
                edge_data['token'].append(data.get('label', 'Unknown'))
                edge_data['line_width'].append(1)
                edge_data['alpha'].append(0.6)
        
        return edge_data

    def update_view(self):
        """Update the visualization, excluding isolated nodes."""
        try:
            G = self.full_graph
            id_to_address = self.id_to_address
            if G is None or id_to_address is None:
                print("No graph data available.")
                return

            # We don't need to filter again as self.full_graph is already filtered
            # Calculate importance scores
            degree_dict = dict(G.degree())
            max_degree = max(degree_dict.values()) if degree_dict else 1
            importance_scores = {
                node: deg / max_degree 
                for node, deg in degree_dict.items()
            }
            
            # Select nodes based on current limit
            important_nodes = self._select_important_nodes(G, importance_scores)
            
            # Create subgraph
            subgraph = G.subgraph(important_nodes)
            
            # Detect communities
            try:
                import community
                communities = community.best_partition(subgraph.to_undirected())
            except Exception as e:
                print(f"Community detection failed: {str(e)}")
                communities = {node: 0 for node in subgraph.nodes()}
            
            # Calculate layout
            pos = self._calculate_layout(subgraph, communities)
            self.current_layout = pos
            
            # Update data sources
            node_data = self._prepare_node_data(subgraph, pos, id_to_address, communities)
            edge_data = self._prepare_edge_data(subgraph, pos)
            
            self.node_source.data = node_data
            self.edge_source.data = edge_data
            
            self.current_visible_nodes = set(subgraph.nodes())
            
            if self.plot:
                self.plot.title.text = f"Network Overview ({len(subgraph)} connected nodes shown)"

        except Exception as e:
            print(f"Error updating view: {str(e)}")
            import traceback
            traceback.print_exc()
