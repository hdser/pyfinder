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
from typing import Dict, Tuple, Any

class InteractiveVisualization:
    def __init__(self):
        self.node_tooltips = [
            ("Node", "@node"),
            ("Type", "@type"),
            ("Address", "@address")
        ]
        self.edge_tooltips = [
            ("From", "@from_node"),
            ("To", "@to_node"),
            ("Token", "@token"),
            ("Flow", "@flow{0,0} mCRC")
        ]

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
                active_scroll='wheel_zoom',
                sizing_mode='stretch_both',
                min_height=400,
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
            node_data = {
                'x': [], 'y': [], 
                'node': [], 
                'type': [],
                'address': [],
                'color': [],
                'size': []
            }

            for node in G.nodes():
                x, y = pos[node]
                node_data['x'].append(x)
                node_data['y'].append(y)
                node_data['node'].append(str(node))
                
                if '_' in str(node):
                    node_data['type'].append('intermediate')
                    node_data['color'].append('red')
                    node_data['size'].append(10)
                else:
                    node_data['type'].append('regular')
                    node_data['color'].append('lightblue')
                    node_data['size'].append(15)
                
                address = id_to_address.get(str(node), "Unknown")
                node_data['address'].append(f"{address[:6]}...{address[-4:]}")

            node_source = ColumnDataSource(node_data)

            # Draw nodes using scatter instead of circle
            nodes = plot.scatter(
                'x', 'y',
                source=node_source,
                size='size',
                fill_color='color',
                line_color='black',
                legend_field='type',
                marker='circle'
            )

            # Add node labels
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

            if simplified:
                edge_data = self._prepare_multiedge_data(G, pos, edge_flows, id_to_address)
            else:
                edge_data = self._prepare_edge_data(G, pos, edge_flows, id_to_address)

            edge_source = ColumnDataSource(edge_data)

            # Draw arrows at the end of edges
            arrow_ends = []
            for xs, ys, line_width in zip(edge_data['xs'], edge_data['ys'], edge_data['line_width']):
                # Get the last two points of each edge
                x2, x1 = xs[-2:]
                y2, y1 = ys[-2:]
                
                # Calculate the arrow angle
                angle = np.arctan2(y1 - y2, x1 - x2)
                
                # Create arrow head
                arrow_ends.append({
                    'x_start': [xs[-2]],
                    'y_start': [ys[-2]],
                    'x_end': [xs[-1]],
                    'y_end': [ys[-1]],
                    'angle': [angle],
                    'line_width': [line_width]
                })

            for arrow in arrow_ends:
                arrow_source = ColumnDataSource(arrow)
                plot.add_layout(Arrow(
                    end=VeeHead(
                        size=8,
                        fill_color='gray',  # Set the fill color to gray
                        line_color='gray'   # Set the line color to gray
                    ),
                    x_start='x_start',
                    y_start='y_start',
                    x_end='x_end',
                    y_end='y_end',
                    source=arrow_source,
                    line_color='gray',      # Set the arrow shaft color to gray
                    line_alpha=0.6          # Match the line alpha with the edges
                ))

            # Draw edge curves
            edges = plot.multi_line(
                xs='xs',
                ys='ys',
                line_color='gray',
                line_width='line_width',
                line_alpha=0.6,
                source=edge_source
            )

            # Add edge labels
            for i, (x, y, label) in enumerate(zip(
                edge_data['label_x'],
                edge_data['label_y'],
                edge_data['label_text']
            )):
                edge_label = Label(
                    x=x, y=y,
                    text=label,
                    text_font_size='8pt',
                    background_fill_color='white',
                    background_fill_alpha=0.7,
                    text_align='center'
                )
                plot.add_layout(edge_label)

            # Add hover tools
            node_hover = HoverTool(
                renderers=[nodes],
                tooltips=self.node_tooltips
            )
            edge_hover = HoverTool(
                renderers=[edges],
                tooltips=self.edge_tooltips
            )
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
            error_plot = figure(
                title="Error",
                sizing_mode='stretch_both',
                min_height=400
            )
            error_plot.text(0, 0, [f"Error: {str(e)}"], text_color="red")
            return error_plot
        
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
        
    def _prepare_multiedge_data(self, G: nx.DiGraph, pos: Dict[str, Tuple[float, float]], 
                               edge_flows: Dict[Tuple[str, str], Dict[str, float]],
                               id_to_address: Dict[str, str]) -> Dict:
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
                token_address = id_to_address.get(token, token)
                label_text = f"Flow: {int(flow):,}\nToken: {token_address[:6]}...{token_address[-4:]}"

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
                          edge_flows: Dict[Tuple[str, str], float],
                          id_to_address: Dict[str, str]) -> Dict:
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

                token_address = id_to_address.get(token, token)
                label_text = f"Flow: {int(flow):,}\nToken: {token_address[:6]}...{token_address[-4:]}"

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
    
    

