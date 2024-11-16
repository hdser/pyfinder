import networkx as nx
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Circle,
    LabelSet,
    LayoutDOM,
    Range1d,
    Label,
    MultiLine,
    Arrow,
    VeeHead,
)
from collections import defaultdict
import numpy as np
from typing import Dict, Tuple, Any, Optional

class InteractiveVisualization:
    def __init__(self):
        self.node_tooltips = [
            ("Node", "@node"),
            ("Type", "@type"),
            ("Address", "@address"),
            ("Connections", "@connections")
        ]
        self.edge_tooltips = [
            ("From", "@from_node"),
            ("To", "@to_node"),
            ("Token", "@token"),
            ("Flow", "@flow{0,0} mCRC")
        ]
        self.max_edges = 1000
        self.max_labels = 50

    def create_bokeh_network(
        self, 
        G: nx.DiGraph,
        edge_flows: Dict[Tuple[str, str], Any],
        id_to_address: Dict[str, str],
        title: str,
        simplified: bool = False
    ) -> LayoutDOM:
        try:
            # Create plot with appropriate size and tools
            plot = figure(
                title=title,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                active_scroll=None,
                sizing_mode='stretch_both',
                min_height=300,
                aspect_ratio=1.5
            )

            # Style the plot
            plot.title.text_font_size = '16px'
            plot.title.align = 'center'
            plot.axis.visible = False
            plot.grid.visible = False
            plot.outline_line_color = None

            # Get source and sink nodes
            source = next(n for n in G.nodes() if G.in_degree(n) == 0)
            sink = next(n for n in G.nodes() if G.out_degree(n) == 0)
            
            # Calculate layout
            pos = self._hierarchical_layout(G, source, sink)

            # Prepare node data
            node_data = self._prepare_node_data(G, pos, id_to_address)
            node_source = ColumnDataSource(node_data)

            # Draw nodes
            nodes = plot.scatter(
                'x', 'y',
                source=node_source,
                size='size',
                fill_color='color',
                line_color='black',
                legend_field='type',
                marker='circle'
            )

            # Add node labels (limited)
            if len(node_data['x']) <= self.max_labels:
                labels = LabelSet(
                    x='x', y='y',
                    text='node',
                    source=node_source,
                    x_offset=5,
                    y_offset=5,
                    text_font_size='8pt',
                    background_fill_color='white',
                    background_fill_alpha=0.7
                )
                plot.add_layout(labels)

            # Prepare edge data based on whether it's simplified or full graph
            if simplified:
                edge_data = self._prepare_multiedge_data(G, pos, edge_flows)
            else:
                if hasattr(G, 'g_gt'):
                    edge_data = self._prepare_graphtool_edge_data(G, pos, edge_flows)
                else:
                    edge_data = self._prepare_edge_data(G, pos, edge_flows)

            # Limit the number of edges if necessary
            if len(edge_data['xs']) > self.max_edges:
                print(f"Limiting visualization to {self.max_edges} edges (out of {len(edge_data['xs'])})")
                for key in edge_data:
                    edge_data[key] = edge_data[key][:self.max_edges]

            edge_source = ColumnDataSource(edge_data)

            # Draw edges
            edges = plot.multi_line(
                xs='xs',
                ys='ys',
                line_color='gray',
                line_width='line_width',
                line_alpha=0.6,
                source=edge_source
            )

            # Add arrows and labels in batches to prevent rendering issues
            if len(edge_data['xs']) <= self.max_edges:
                self._add_arrows_and_labels(plot, edge_data)

            # Add hover tools
            node_hover = HoverTool(renderers=[nodes], tooltips=self.node_tooltips)
            edge_hover = HoverTool(renderers=[edges], tooltips=self.edge_tooltips)
            plot.add_tools(node_hover, edge_hover)

            # Configure legend
            plot.legend.click_policy = "hide"
            plot.legend.location = "top_right"
            plot.legend.background_fill_alpha = 0.7

            return plot

        except Exception as e:
            print(f"Error creating Bokeh network: {str(e)}")
            import traceback
            traceback.print_exc()
            error_plot = figure(title="Error", sizing_mode='stretch_both', min_height=400)
            error_plot.text(0, 0, [f"Error: {str(e)}"], text_color="red")
            return error_plot

    def _calculate_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Calculate layout for the graph."""
        try:
            return nx.spring_layout(
                G,
                k=1/np.sqrt(len(G)),
                iterations=50,
                scale=10
            )
        except Exception as e:
            print(f"Error calculating layout: {str(e)}")
            return nx.circular_layout(G, scale=10)
        
    def _prepare_node_data(self, G: nx.DiGraph, pos: Dict[str, Tuple[float, float]], 
                          id_to_address: Dict[str, str]) -> dict:
        """Prepare node data for visualization."""
        degree_dict = dict(G.degree())
        max_degree = max(degree_dict.values()) if degree_dict else 1
        
        node_data = {
            'x': [], 'y': [], 'node': [], 'type': [], 'address': [],
            'color': [], 'size': [], 'connections': [], 'alpha': []
        }
        
        for node in G.nodes():
            x, y = pos[node]
            node_data['x'].append(float(x))
            node_data['y'].append(float(y))
            node_data['node'].append(str(node))
            
            degree = degree_dict[node]
            node_data['connections'].append(degree)
            
            if '_' in str(node):
                node_data['type'].append('intermediate')
                node_data['color'].append('#ff7f7f')
                node_data['size'].append(5)
                node_data['alpha'].append(0.5)
            else:
                node_data['type'].append('regular')
                node_data['color'].append('#7fbfff')
                size = 5 + (degree / max_degree) * 15
                node_data['size'].append(size)
                node_data['alpha'].append(0.8)
            
            node_data['address'].append(id_to_address.get(str(node), "Unknown"))
        
        return node_data

    def _add_arrows_and_labels(self, plot, edge_data):
        """Add arrows and labels to the plot in batches."""
        batch_size = 20  # Process 20 edges at a time
        
        for i in range(0, len(edge_data['xs']), batch_size):
            batch_end = min(i + batch_size, len(edge_data['xs']))
            
            # Add arrows for this batch
            for j in range(i, batch_end):
                xs = edge_data['xs'][j]
                ys = edge_data['ys'][j]
                line_width = edge_data['line_width'][j]
                
                # Get the last two points of the edge
                x2, x1 = xs[-2:]
                y2, y1 = ys[-2:]
                
                # Calculate the arrow angle
                angle = np.arctan2(y1 - y2, x1 - x2)
                
                arrow_source = ColumnDataSource({
                    'x_start': [xs[-2]],
                    'y_start': [ys[-2]],
                    'x_end': [xs[-1]],
                    'y_end': [ys[-1]]
                })

                plot.add_layout(Arrow(
                    end=VeeHead(size=8, fill_color='gray', line_color='gray'),
                    x_start='x_start',
                    y_start='y_start',
                    x_end='x_end',
                    y_end='y_end',
                    source=arrow_source,
                    line_color='gray',
                    line_alpha=0.6
                ))

            # Add labels for this batch
            for j in range(i, batch_end):
                label = Label(
                    x=edge_data['label_x'][j],
                    y=edge_data['label_y'][j],
                    text=edge_data['label_text'][j],
                    text_font_size='8pt',
                    background_fill_color='white',
                    background_fill_alpha=0.7,
                    text_align='center'
                )
                plot.add_layout(label)

    def _prepare_graphtool_edge_data(self, G, pos, edge_flows):
        """Special edge data preparation for graph-tool graphs."""
        edge_data = {
            'xs': [], 'ys': [],
            'from_node': [], 'to_node': [],
            'token': [], 'flow': [],
            'line_width': [],
            'label_x': [], 'label_y': [],
            'label_text': []
        }

        for (u, v), flow in edge_flows.items():
            if flow > 0:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                # Get token from the original graph
                token = None
                if hasattr(G, 'g_gt'):
                    u_vertex = G.id_to_vertex.get(u)
                    v_vertex = G.id_to_vertex.get(v)
                    if u_vertex is not None and v_vertex is not None:
                        edge = G.g_gt.edge(u_vertex, v_vertex)
                        if edge is not None:
                            token = G.token[edge]

                # Calculate curved path
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                
                # Generate curve points
                t = np.linspace(0, 1, 50)
                xs = (1-t)**2 * x0 + 2*(1-t)*t * mid_x + t**2 * x1
                ys = (1-t)**2 * y0 + 2*(1-t)*t * mid_y + t**2 * y1

                
                label_text = f"Flow: {int(flow):,}\nToken: {token}"

                edge_data['xs'].append(list(xs))
                edge_data['ys'].append(list(ys))
                edge_data['from_node'].append(u)
                edge_data['to_node'].append(v)
                edge_data['token'].append(token if token else "Unknown")
                edge_data['flow'].append(flow)
                edge_data['line_width'].append(1 + np.log1p(float(flow)) * 0.5)
                edge_data['label_x'].append(mid_x)
                edge_data['label_y'].append(mid_y)
                edge_data['label_text'].append(label_text)

        return edge_data
        
    def _hierarchical_layout(self, G: nx.DiGraph, source: str, sink: str,
                           horizontal_spacing: float = 5.0,
                           vertical_spacing: float = 2.0) -> Dict[str, Tuple[float, float]]:
        """Calculate hierarchical layout with proper node positioning."""
        def get_node_rank(node: str) -> int:
            if node == source:
                return 0
            elif node == sink:
                return max_rank
            else:
                paths = list(nx.all_simple_paths(G, source, node))
                return max(len(p) - 1 for p in paths) if paths else len(G)

        # Calculate ranks
        max_rank = len(nx.dag_longest_path(G)) - 1
        ranks = {node: get_node_rank(node) for node in G.nodes()}
        
        # Group nodes by rank
        nodes_by_rank = defaultdict(list)
        for node, rank in ranks.items():
            nodes_by_rank[rank].append(node)

        # Calculate positions
        pos = {}
        for rank, nodes in nodes_by_rank.items():
            x = rank * horizontal_spacing
            total_nodes = len(nodes)
            for i, node in enumerate(sorted(nodes)):
                y = (i - (total_nodes - 1) / 2) * vertical_spacing
                pos[node] = (x, y)

        return pos
    
    def _hierarchical_layout2(self, G: nx.DiGraph, source: str, sink: str,
                        horizontal_spacing: float = 5.0,
                        vertical_spacing: float = 2.0) -> Dict[str, Tuple[float, float]]:
        """
        Calculate hierarchical layout with proper node positioning.
        Uses BFS to assign ranks instead of DAG-specific algorithms.
        """
        from collections import deque, defaultdict
        
        def get_node_rank_bfs(node: str) -> int:
            """Calculate node rank using BFS from source."""
            if node == source:
                return 0
            elif node == sink:
                return max_rank if max_rank > 0 else len(G)
                
            queue = deque([(source, 0)])
            visited = {source}
            min_rank = len(G)  # Default to maximum possible rank
            
            while queue:
                current, rank = queue.popleft()
                if current == node:
                    min_rank = min(min_rank, rank)
                
                for neighbor in G.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, rank + 1))
            
            return min_rank if min_rank < len(G) else len(G) // 2
        
        # First pass: get approximate max rank using BFS
        max_rank = 0
        visited = {source}
        queue = deque([(source, 0)])
        
        while queue:
            node, rank = queue.popleft()
            max_rank = max(max_rank, rank)
            
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, rank + 1))
        
        # Calculate ranks for all nodes
        ranks = {node: get_node_rank_bfs(node) for node in G.nodes()}
        
        # Group nodes by rank
        nodes_by_rank = defaultdict(list)
        for node, rank in ranks.items():
            nodes_by_rank[rank].append(node)
        
        # Calculate positions with balanced distribution
        pos = {}
        max_nodes_per_rank = max(len(nodes) for nodes in nodes_by_rank.values())
        
        for rank, nodes in nodes_by_rank.items():
            x = rank * horizontal_spacing
            total_nodes = len(nodes)
            
            # Adjust vertical spacing based on number of nodes
            effective_spacing = vertical_spacing * (max_nodes_per_rank / max(total_nodes, 1))
            
            for i, node in enumerate(sorted(nodes)):
                y = (i - (total_nodes - 1) / 2) * effective_spacing
                pos[node] = (x, y)
        
        return pos
        
    def _prepare_multiedge_data(self, G: nx.DiGraph, pos: Dict[str, Tuple[float, float]], 
                               edge_flows: Dict[Tuple[str, str], Dict[str, float]]) -> Dict:
        """Prepare edge data for simplified graph with proper curve calculations for multiedges."""
        edge_data = {
            'xs': [], 'ys': [],
            'from_node': [], 'to_node': [],
            'token': [], 'flow': [],
            'line_width': [],
            'label_x': [], 'label_y': [],
            'label_text': []
        }

        for (u, v), token_flows in edge_flows.items():
            num_edges = len(token_flows)
            edge_offsets = np.linspace(-0.3, 0.3, num_edges)

            for (token, flow), offset in zip(token_flows.items(), edge_offsets):
                # Calculate curved path
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                # Control points for quadratic bezier curve
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                
                # Calculate perpendicular offset
                dx = x1 - x0
                dy = y1 - y0
                length = np.sqrt(dx*dx + dy*dy)
                ux = -dy/length
                uy = dx/length
                
                # Apply offset to control point
                control_x = mid_x + ux * offset
                control_y = mid_y + uy * offset

                # Generate curve points
                t = np.linspace(0, 1, 50)
                xs = (1-t)**2 * x0 + 2*(1-t)*t * control_x + t**2 * x1
                ys = (1-t)**2 * y0 + 2*(1-t)*t * control_y + t**2 * y1

                # Calculate label position
                label_x = control_x
                label_y = control_y
                label_text = f"Flow: {int(flow):,}\nToken: {token}"

                edge_data['xs'].append(list(xs))
                edge_data['ys'].append(list(ys))
                edge_data['from_node'].append(u)
                edge_data['to_node'].append(v)
                edge_data['token'].append(token)
                edge_data['flow'].append(flow)
                edge_data['line_width'].append(1 + np.log1p(float(flow)) * 0.5)
                edge_data['label_x'].append(label_x)
                edge_data['label_y'].append(label_y)
                edge_data['label_text'].append(label_text)

        return edge_data
    
    def _prepare_edge_data(self, G: nx.DiGraph, pos: Dict[str, Tuple[float, float]], 
                          edge_flows: Dict[Tuple[str, str], float]) -> Dict:
        """Prepare edge data for full graph."""
        edge_data = {
            'xs': [], 'ys': [],
            'from_node': [], 'to_node': [],
            'token': [], 'flow': [],
            'line_width': [],
            'label_x': [], 'label_y': [],
            'label_text': []
        }

        for (u, v), flow in edge_flows.items():
            if flow > 0:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                # Calculate curved path
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                
                # Get edge data
                if G.has_edge(u, v):
                    edge_data_dict = G.get_edge_data(u, v)
                    token = edge_data_dict.get('label', 'Unknown')
                else:
                    token = 'Unknown'

                # Generate curve points
                t = np.linspace(0, 1, 50)
                xs = (1-t)**2 * x0 + 2*(1-t)*t * mid_x + t**2 * x1
                ys = (1-t)**2 * y0 + 2*(1-t)*t * mid_y + t**2 * y1

                label_text = f"Flow: {int(flow):,}\nToken: {token}"

                edge_data['xs'].append(list(xs))
                edge_data['ys'].append(list(ys))
                edge_data['from_node'].append(u)
                edge_data['to_node'].append(v)
                edge_data['token'].append(token)
                edge_data['flow'].append(flow)
                edge_data['line_width'].append(1 + np.log1p(float(flow)) * 0.5)
                edge_data['label_x'].append(mid_x)
                edge_data['label_y'].append(mid_y)
                edge_data['label_text'].append(label_text)

        return edge_data
    
    def _add_graph_elements(self, plot: figure, node_data: dict, edge_data: dict) -> None:
        """Add nodes and edges to the plot."""
        # Create data sources
        node_source = ColumnDataSource(node_data)
        edge_source = ColumnDataSource(edge_data)
        
        # Add edges
        edges = plot.multi_line(
            xs='xs', ys='ys',
            line_color='gray',
            line_width='line_width',
            line_alpha='alpha',
            source=edge_source,
            hover_line_color='#ff7f0e',
            hover_line_alpha=1.0
        )
        
        # Add nodes
        nodes = plot.scatter(
            'x', 'y',
            source=node_source,
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
        
        # Add node selection behavior
        nodes.selection_glyph = Circle(
            fill_color='#ff7f0e',
            line_color='black'
        )
        nodes.nonselection_glyph = Circle(
            fill_color='color',
            line_color='black',
            fill_alpha=0.1
        )

        # Add labels if not too many nodes
        if len(node_data['x']) <= self.max_labels:
            labels = LabelSet(
                x='x', y='y',
                text='node',
                source=node_source,
                text_font_size='8pt',
                x_offset=5,
                y_offset=5,
                text_alpha=0.7
            )
            plot.add_layout(labels)
    

