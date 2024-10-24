import panel as pn
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, 
    HoverTool,
    MultiLine,
    Circle,
    BoxSelectTool,
    TapTool,
    WheelZoomTool,
    PanTool,
    BoxZoomTool,
    ResetTool,
    SaveTool,
    LabelSet,
    LayoutDOM,
    Range1d,
    Arrow,
    NormalHead,
    VeeHead,
    OpenHead,
    Label
)
from bokeh.layouts import column, row
from collections import defaultdict
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class InteractiveVisualization:
    def __init__(self):
        self.tooltips = [
            ("Node", "@node"),
            ("Type", "@type"),
            ("Flow", "@flow"),
            ("Token", "@token")
        ]

    def create_bokeh_network(
        self, 
        G: nx.DiGraph,
        edge_flows: Dict[Tuple[str, str], Any],
        title: str,
        simplified: bool = False
    ) -> LayoutDOM:
        """Create an interactive Bokeh plot for the network."""
        try:
            # Create tools
            wheel_zoom = WheelZoomTool()
            pan = PanTool()
            box_zoom = BoxZoomTool()
            reset = ResetTool()
            save = SaveTool()
            tap = TapTool()

            tools = [pan, box_zoom, wheel_zoom, reset, save, tap]

            # Create plot with proper tools and sizing
            plot = figure(
                title=title,
                tools=tools,
                active_scroll=wheel_zoom,
                sizing_mode='stretch_both',
                min_height=400,
                aspect_ratio=1.5
            )

            # Style the title
            plot.title.text_font_size = '16px'
            plot.title.align = 'center'

            # Get layout positions
            source = next(n for n in G.nodes() if G.in_degree(n) == 0)
            sink = next(n for n in G.nodes() if G.out_degree(n) == 0)
            pos = self._custom_layout(G, source, sink)

            # Calculate bounds with padding
            x_values = [coord[0] for coord in pos.values()]
            y_values = [coord[1] for coord in pos.values()]
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)
            padding = max((x_max - x_min), (y_max - y_min)) * 0.2

            # Set plot ranges
            plot.x_range = Range1d(x_min - padding, x_max + padding)
            plot.y_range = Range1d(y_min - padding, y_max + padding)

            # Create node data source
            node_data = {
                'x': [], 'y': [], 'node': [], 'type': [],
                'color': [], 'size': []
            }

            for node in G.nodes():
                x, y = pos[node]
                node_data['x'].append(x)
                node_data['y'].append(y)
                node_data['node'].append(str(node))
                if '_' in str(node):
                    node_data['type'].append('intermediate')
                    node_data['color'].append('red')
                    node_data['size'].append(15)
                else:
                    node_data['type'].append('regular')
                    node_data['color'].append('lightblue')
                    node_data['size'].append(20)

            node_source = ColumnDataSource(node_data)

            # Draw nodes using scatter
            nodes = plot.scatter(
                'x', 'y',
                source=node_source,
                size='size',
                fill_color='color',
                line_color='black',
                legend_field='type'
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

            # Create edge data source
            edge_data = {
                'x_start': [], 'y_start': [], 'x_end': [], 'y_end': [],
                'flow': [], 'token': [], 'width': [], 'alpha': []
            }

            if simplified:
                for (u, v), token_flows in edge_flows.items():
                    if u in pos and v in pos:
                        for token, flow in token_flows.items():
                            edge_data['x_start'].append(pos[u][0])
                            edge_data['y_start'].append(pos[u][1])
                            edge_data['x_end'].append(pos[v][0])
                            edge_data['y_end'].append(pos[v][1])
                            edge_data['flow'].append(str(flow))
                            edge_data['token'].append(str(token))
                            edge_data['width'].append(1 + np.log1p(float(flow)) * 0.5)
                            edge_data['alpha'].append(0.6)
            else:
                for (u, v), flow in edge_flows.items():
                    if u in pos and v in pos:
                        edge_data['x_start'].append(pos[u][0])
                        edge_data['y_start'].append(pos[u][1])
                        edge_data['x_end'].append(pos[v][0])
                        edge_data['y_end'].append(pos[v][1])
                        edge_data['flow'].append(str(flow))
                        edge_data['token'].append(str(G[u][v].get('label', '')))
                        edge_data['width'].append(1 + np.log1p(float(flow)) * 0.5)
                        edge_data['alpha'].append(0.6)

            edge_source = ColumnDataSource(edge_data)

            # Draw edges with arrows
            for i in range(len(edge_source.data['x_start'])):
                x_start = edge_source.data['x_start'][i]
                y_start = edge_source.data['y_start'][i]
                x_end = edge_source.data['x_end'][i]
                y_end = edge_source.data['y_end'][i]
                flow = edge_source.data['flow'][i]
                token = edge_source.data['token'][i]
                width = edge_source.data['width'][i]
                alpha = edge_source.data['alpha'][i]

                # Create an arrow
                arrow = Arrow(
                    end=NormalHead(size=10, fill_color='gray'),
                    x_start=x_start, y_start=y_start,
                    x_end=x_end, y_end=y_end,
                    line_width=width,
                    line_alpha=alpha,
                    line_color='gray'
                )
                plot.add_layout(arrow)

                # Optional: Add edge labels
                # Calculate midpoint for label
                mid_x = (x_start + x_end) / 2
                mid_y = (y_start + y_end) / 2
                label = Label(
                    x=mid_x, y=mid_y,
                    text=f"Flow: {flow}\nToken: {token}",
                    text_font_size='8pt',
                    text_align='center',
                    text_baseline='middle',
                    background_fill_color='white',
                    background_fill_alpha=0.7
                )
                plot.add_layout(label)

            # Add hover tools
            node_hover = HoverTool(renderers=[nodes], tooltips=[
                ("Node", "@node"),
                ("Type", "@type")
            ])

            plot.add_tools(node_hover)

            # Style the plot
            plot.axis.visible = False
            plot.grid.visible = False
            plot.outline_line_color = None
            plot.toolbar.logo = None

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

    def _custom_layout(self, G: nx.DiGraph, source: str, sink: str, 
                       horizontal_spacing: float = 5.0, 
                       vertical_spacing: float = 2.0) -> Dict[str, Tuple[float, float]]:
        """Custom layout algorithm optimized for flow networks."""
        levels = self._compute_node_levels(G, source)
        pos = {}
        
        # Position source and sink
        max_level = max(levels.values())
        pos[source] = (0, 0)
        pos[sink] = (horizontal_spacing * (max_level + 1), 0)
        
        # Group nodes by level
        nodes_by_level = defaultdict(list)
        for node, level in levels.items():
            if node not in (source, sink):
                nodes_by_level[level].append(node)
                
        # Position nodes at each level
        for level, nodes in nodes_by_level.items():
            x = level * horizontal_spacing
            total_nodes = len(nodes)
            for i, node in enumerate(nodes):
                y = (i - (total_nodes - 1) / 2) * vertical_spacing
                pos[node] = (x, y)
                
        # Handle any nodes not placed by BFS
        remaining_nodes = set(G.nodes()) - set(pos.keys())
        if remaining_nodes:
            x_mid = (max_level + 1) * horizontal_spacing / 2
            for i, node in enumerate(remaining_nodes):
                pos[node] = (x_mid, (i + 1) * vertical_spacing)
                
        return pos

    def _compute_node_levels(self, G: nx.DiGraph, source: str) -> Dict[str, int]:
        """Compute node levels using BFS."""
        levels = {source: 0}
        queue = [(source, 0)]
        visited = {source}
        
        while queue:
            node, level = queue.pop(0)
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    levels[neighbor] = level + 1
                    queue.append((neighbor, level + 1))
        
        # Handle nodes not reachable from source
        remaining_nodes = set(G.nodes()) - set(levels.keys())
        if remaining_nodes:
            max_level = max(levels.values(), default=0)
            middle_level = max_level // 2
            for node in remaining_nodes:
                levels[node] = middle_level
        
        return levels
