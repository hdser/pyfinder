import panel as pn
import param
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from bokeh.plotting import figure
from src.graph import NetworkXGraph, GraphToolGraph, BaseGraph 

from .base_component import BaseComponent
from .interactive_visualization import InteractiveVisualization

class VisualizationComponent(BaseComponent):
    """Component for visualizing network flow analysis results."""
    
    def __init__(self, **params):
        self.interactive_viz = InteractiveVisualization()
        self.full_network_plot = None
        self.graph_manager = None
        super().__init__(**params)

    def _create_components(self):
        """Create visualization components with improved layout."""
        # Create panes for statistics
        self.network_stats = pn.pane.HTML(
            "Network statistics will appear here",
            sizing_mode='stretch_width'
        )
        
        self.metrics_pane = pn.pane.HTML(
            "",
            sizing_mode='stretch_width'
        )
        
        self.histogram_pane = pn.pane.Bokeh(
            sizing_mode='stretch_width',
            margin=(0, 0, 10, 0)
        )
        
        self.histogram_stats_pane = pn.pane.HTML(
            "",
            sizing_mode='stretch_width'
        )

        # Create two-column layout for statistics
        self.stats_top = pn.Row(
            self.network_stats,
            self.histogram_stats_pane,
            sizing_mode='stretch_width'
        )

        # Create layout for metrics and histogram
        self.stats_bottom = pn.Row(
            self.metrics_pane,
            self.histogram_pane,
            sizing_mode='stretch_width'
        )

        # Combine into final layout
        self.stats_layout = pn.Column(
            self.stats_top,
            self.stats_bottom,
            sizing_mode='stretch_width',
            margin=(0, 0, 20, 0)
        )
        
        self.path_stats = pn.pane.Markdown(
            "Path analysis results will appear here",
            styles={'background': '#f8f9fa', 'padding': '15px', 
                   'border-radius': '5px', 'border': '1px solid #dee2e6'},
            sizing_mode='stretch_width'
        )
        
        
        self.network_load_progress = pn.indicators.Progress(
            name='Loading Network',
            value=0,
            max=100,
            visible=False,
            width=200
        )
        
        # Graph visualization panes
        self.full_graph_pane = pn.pane.Bokeh(
            sizing_mode='stretch_width',
            height=500,
            margin=(0, 0, 5, 0)
        )
        
        self.flow_graph_pane = pn.pane.Bokeh(
            sizing_mode='stretch_width',
            height=450,
            margin=(0, 0, 5, 0)
        )
        
        self.simplified_graph_pane = pn.pane.Bokeh(
            sizing_mode='stretch_width',
            height=450,
            margin=(0, 0, 5, 0)
        )
        
        # Transactions box
        self.transactions_box = pn.Column(
            pn.pane.HTML("""
                <div style="height:300px; overflow-y:scroll; 
                        border:1px solid #ddd; padding:10px;">
                    <div id="transactions-content">
                        <p>No transactions to display yet.</p>
                    </div>
                </div>
            """),
            sizing_mode='stretch_width',
            visible=False  # Hide initially
        )
        
        # Create the flow section for use in view()
        self.flow_section = pn.Column(
            pn.pane.Markdown("# Flow Analysis", 
                        css_classes=['section-title']),
            self.path_stats,
            pn.Tabs(
                ("Full Flow Network", pn.Column(
                    self.flow_graph_pane,
                    sizing_mode='stretch_width'
                )),
                ("Simplified Flow Network", pn.Column(
                    self.simplified_graph_pane,
                    sizing_mode='stretch_width'
                )),
                sizing_mode='stretch_width'
            ),
            self.transactions_box,
            sizing_mode='stretch_width',
            css_classes=['section-content'],
            visible=False  # Hide initially
        )

    def _update_path_stats(self, flow_value, simplified_paths, simplified_edge_flows, 
                      original_edge_flows, computation_time, algorithm):
        """Update the path analysis statistics."""
        stats = f"""
        ### Flow Analysis Results
        
        **Flow Information**
        - Total Flow Value: {flow_value:,} mCRC
        - Computation Time: {computation_time:.4f} seconds
        - Algorithm Used: {algorithm}

        **Path Analysis**
        - Number of Distinct Paths: {len(simplified_paths)}
        - Total Transfers: {sum(len(flows) for flows in simplified_edge_flows.values())}
        - Original Edges: {len(original_edge_flows)}
        - Simplified Edge Groups: {len(simplified_edge_flows)}
        """
        
        if hasattr(self, 'path_stats'):
            self.path_stats.object = stats
        
        # Also update the main stats pane
        if hasattr(self, 'stats_pane'):
            self.stats_pane.object = stats

   
    def _initialize_section_contents(self):
        """Initialize the content of each collapsible section."""
        # Network section content
        self.network_content = pn.Column(
            self.network_stats,
            pn.Row(
                self.network_load_progress,
                align='center',
                margin=(10, 10)
            ),
            self.full_graph_pane,
            sizing_mode='stretch_width',
            css_classes=['section-content']
        )
        
        # Flow analysis section content (including tabs)
        self.flow_content = pn.Column(
            self.path_stats,
            self.flow_tabs,
            self.transactions_box,
            sizing_mode='stretch_width',
            css_classes=['section-content']
        )

    def _create_transactions_box(self):
        """Create the transactions display box."""
        """Create the transactions display box."""
        return pn.Column(
            pn.pane.Markdown("# Aggregated Transactions"),
            pn.pane.HTML("""
                <div style="height:300px; overflow-y:scroll; 
                     border:1px solid #ddd; padding:10px;">
                    <div id="transactions-content">
                        <p>No transactions to display yet.</p>
                    </div>
                </div>
            """),
            visible=False
        )

    def _setup_callbacks(self):
        """Set up component callbacks."""
        self._update_detail_level()
    
    def _update_detail_level(self, event=None):
        """Refresh the network visualization with current settings."""
        if not self.graph_manager:
            return
            
        try:
            self.network_load_progress.visible = True
            self.network_load_progress.value = 20
            
            graph = self.graph_manager.graph
            
            self.network_load_progress.value = 40
            
            self.network_load_progress.value = 80
            
            # Clear existing pane
            if hasattr(self, 'full_graph_pane'):
                self.full_graph_pane.object = None
            
            # Create new network section with updated graph
            network_section = pn.Column(
                pn.pane.Markdown("# Network Overview", 
                            css_classes=['section-title']),
                self.network_stats,
                pn.Row(
                    self.network_load_progress,
                    align='center',
                    css_classes=['network-controls']
                ),
                sizing_mode='stretch_width',
                css_classes=['section-content']
            )
            
            # Update the main view
            self._main_view.objects = [network_section, self.flow_section]
            
            self._update_network_stats(graph)
            
            self.network_load_progress.value = 100
            self.network_load_progress.visible = False
            
        except Exception as e:
            print(f"Error refreshing network view: {str(e)}")
            import traceback
            traceback.print_exc()
            self._show_error(f"Error refreshing network view: {str(e)}")
            self.network_load_progress.visible = False

    def _update_network_stats(self, graph: BaseGraph):
        """Update network statistics with consistent boxed layout and histogram."""
        try:
            # Calculate all metrics first
            all_nodes = graph.get_vertices()
            real_nodes = {v for v in all_nodes if '_' not in v}
            intermediate_nodes = {v for v in all_nodes if '_' in v}
            connected_nodes = {v for v in real_nodes if graph.degree(v) > 0}

            # Format overview stats
            overview_stats = {
                "Total Real Nodes": f"{len(real_nodes):,}",
                "Total Intermediate Nodes": f"{len(intermediate_nodes):,}",
                "Total Edges": f"{graph.num_edges():,}",
                "Connected Nodes": f"{len(connected_nodes):,}"
            }
            overview_md = self._format_stats_box("Network Overview", overview_stats)
            self.network_stats.object = overview_md

            # Calculate detailed network metrics
            metrics = {
                "Edge-Vertex Ratio": f"{graph.num_edges() / len(real_nodes):.4f}" if len(real_nodes) > 0 else "0",
                "Graph Density": f"{self._calculate_density(graph, connected_nodes):.6f}",
                "Average Total Degree": f"{sum(graph.degree(v) for v in connected_nodes) / len(connected_nodes):.4f}",
                "Average In-Degree": f"{sum(graph.in_degree(v) for v in connected_nodes) / len(connected_nodes):.4f}",
                "Average Out-Degree": f"{sum(graph.out_degree(v) for v in connected_nodes) / len(connected_nodes):.4f}",
                "Maximum In-Degree": f"{max(graph.in_degree(v) for v in connected_nodes):,}",
                "Maximum Out-Degree": f"{max(graph.out_degree(v) for v in connected_nodes):,}",
                "Median Degree": f"{sorted(graph.degree(v) for v in connected_nodes)[len(connected_nodes) // 2]:,}"
            }
            metrics_md = self._format_stats_box("Network Metrics", metrics)
            self.metrics_pane.object = metrics_md

            # Calculate connection counts for intermediate nodes
            connection_counts = [graph.degree(node) for node in intermediate_nodes]

            # Create histogram
            import numpy as np
            from bokeh.plotting import figure
            from bokeh.models import (
                ColumnDataSource, 
                HoverTool, 
                BoxAnnotation,
                LinearAxis, 
                Grid
            )

            # Calculate histogram data
            hist, edges = np.histogram(connection_counts, bins=50)
            
            # Create source data
            source = ColumnDataSource(data=dict(
                count=hist,
                left=edges[:-1],
                right=edges[1:],
                center=(edges[:-1] + edges[1:]) / 2,
            ))

            # Create figure
            p = figure(
                title="Distribution of Intermediate Node Connections",
                width=400,
                height=300,
                toolbar_location=None,
                tools="hover",
                sizing_mode='stretch_width'
            )

            # Style the plot
            p.title.text_font_size = '14px'
            p.title.text_font_style = 'normal'
            p.title.text_color = '#333'
            p.title.align = 'center'
            
            # Add hover tool
            hover = HoverTool(tooltips=[
                ("Range", "@left{0,0} - @right{0,0}"),
                ("Count", "@count{0,0}")
            ])
            p.add_tools(hover)

            # Create histogram bars
            p.quad(
                bottom=0,
                top='count',
                left='left',
                right='right',
                source=source,
                fill_color='#1a73e8',
                line_color='white',
                alpha=0.7,
                hover_fill_color='#1a73e8',
                hover_fill_alpha=1.0
            )

            # Style axes
            p.xaxis.axis_label = 'Number of Connections'
            p.yaxis.axis_label = 'Frequency'
            p.xgrid.grid_line_color = None
            p.ygrid.grid_line_color = '#e0e0e0'
            p.ygrid.grid_line_alpha = 0.5
            p.background_fill_color = "#f8f9fa"
            p.border_fill_color = "white"
            p.outline_line_color = None
            
            # Customize axis appearance
            p.axis.axis_line_color = None
            p.axis.major_tick_line_color = None
            p.axis.minor_tick_line_color = None
            p.axis.axis_label_text_font_size = '12px'
            p.axis.axis_label_text_font_style = 'normal'
            p.axis.axis_label_text_color = '#666666'

            # Add mean line
            mean_value = np.mean(connection_counts)
            mean_box = BoxAnnotation(
                left=mean_value, 
                right=mean_value,
                fill_color='red', 
                fill_alpha=0.4,
                line_color='red', 
                line_width=2
            )
            p.add_layout(mean_box)

            # Update histogram pane
            self.histogram_pane.object = p

            # Calculate connection statistics
            connection_stats = {
                "Average Connections": f"{np.mean(connection_counts):.2f}",
                "Median Connections": f"{np.median(connection_counts):.0f}",
                "Maximum Connections": f"{np.max(connection_counts):,}",
                "Minimum Connections": f"{np.min(connection_counts):,}",
                "Standard Deviation": f"{np.std(connection_counts):.2f}",
                "Total Intermediate Nodes": f"{len(intermediate_nodes):,}"
            }
            stats_md = self._format_stats_box("Intermediate Node Statistics", connection_stats)
            self.histogram_stats_pane.object = stats_md

        except Exception as e:
            print(f"Error updating network stats: {str(e)}")
            import traceback
            traceback.print_exc()
            self._show_error(f"Error updating network stats: {str(e)}")

    def _format_stats_box(self, title: str, stats: Dict[str, str]) -> str:
        """Format statistics in a consistent boxed layout."""
        md = f"""
        <div style="
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 10px;
        ">
            <div style="
                font-size: 16px;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
                padding-bottom: 8px;
                border-bottom: 2px solid #f0f0f0;
            ">{title}</div>
            <div style="
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
            ">
        """
        
        for label, value in stats.items():
            md += f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    padding: 5px 10px;
                    background: #f8f9fa;
                    border-radius: 4px;
                ">
                    <span style="color: #666;">{label}:</span>
                    <span style="
                        font-family: monospace;
                        color: #1a73e8;
                        font-weight: 500;
                    ">{value}</span>
                </div>
            """
        
        md += """
            </div>
        </div>
        """
        return md

    def _calculate_density(self, graph: BaseGraph, nodes: Set[str]) -> float:
        """Calculate graph density using only BaseGraph interface methods."""
        if len(nodes) <= 1:
            return 0.0
        possible_edges = len(nodes) * (len(nodes) - 1)
        actual_edges = sum(1 for u, v, _ in graph.get_edges() 
                        if u in nodes and v in nodes)
        return actual_edges / possible_edges if possible_edges > 0 else 0

    def _calculate_degree_metrics(self, graph: BaseGraph, nodes: Set[str]) -> Dict[str, float]:
        """Calculate degree-related metrics using only BaseGraph interface methods."""
        if not nodes:
            return {}
            
        in_degrees = [graph.in_degree(v) for v in nodes]
        out_degrees = [graph.out_degree(v) for v in nodes]
        total_degrees = [graph.degree(v) for v in nodes]
        
        return {
            "Average Degree": sum(total_degrees) / len(nodes),
            "Max In-Degree": max(in_degrees),
            "Max Out-Degree": max(out_degrees),
            "Min In-Degree": min(in_degrees),
            "Min Out-Degree": min(out_degrees),
            "Median Degree": sorted(total_degrees)[len(total_degrees) // 2],
            "Degree Variance": self._calculate_variance(total_degrees)
        }

    def _calculate_connectivity_metrics(self, graph: BaseGraph, nodes: Set[str]) -> Dict[str, Union[float, int]]:
        """Calculate connectivity metrics using only BaseGraph interface methods."""
        metrics = {}
        
        # Calculate number of source and sink nodes
        sources = sum(1 for v in nodes if graph.in_degree(v) == 0 and graph.out_degree(v) > 0)
        sinks = sum(1 for v in nodes if graph.out_degree(v) == 0 and graph.in_degree(v) > 0)
        
        # Calculate reciprocity (bi-directional edges)
        reciprocal_edges = 0
        total_edges = 0
        for u in nodes:
            for v in nodes:
                if graph.has_edge(u, v):
                    total_edges += 1
                    if graph.has_edge(v, u):
                        reciprocal_edges += 1
        
        reciprocity = reciprocal_edges / total_edges if total_edges > 0 else 0
        
        metrics.update({
            "Source Nodes": sources,
            "Sink Nodes": sinks,
            "Reciprocity": reciprocity,
            "Bi-directional Edge Ratio": reciprocity / 2  # Divide by 2 to avoid counting twice
        })
        
        return metrics

    def _calculate_flow_metrics(self, graph: BaseGraph, nodes: Set[str]) -> Dict[str, Union[float, int]]:
        """Calculate flow and capacity related metrics."""
        metrics = {}
        
        # Calculate total and average capacity
        total_capacity = 0
        capacity_count = 0
        max_capacity = float('-inf')
        min_capacity = float('inf')
        
        for u, v, _ in graph.get_edges():
            if u in nodes and v in nodes:
                capacity = graph.get_edge_capacity(u, v)
                if capacity is not None:
                    total_capacity += capacity
                    capacity_count += 1
                    max_capacity = max(max_capacity, capacity)
                    min_capacity = min(min_capacity, capacity)
        
        if capacity_count > 0:
            metrics.update({
                "Total Edge Capacity": total_capacity,
                "Average Edge Capacity": total_capacity / capacity_count,
                "Maximum Edge Capacity": max_capacity,
                "Minimum Edge Capacity": min_capacity
            })
        
        return metrics

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        return squared_diff_sum / len(values)

    def _create_metrics_table(self, title: str, metrics: Dict[str, Any]) -> str:
        """Create an HTML table for a set of metrics."""
        if not metrics:
            return ""
            
        html = f"""
        <div class="metrics-section">
            <div class="metrics-title">{title}</div>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """
        
        for metric, value in metrics.items():
            formatted_value = (
                f"{value:,.4f}" if isinstance(value, float) 
                else f"{value:,}" if isinstance(value, int)
                else str(value)
            )
            html += f"""
                <tr>
                    <td>{metric}</td>
                    <td class="metric-value">{formatted_value}</td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        """
        return html

    def _calculate_path_metrics(self, graph: BaseGraph, connected_nodes: Set[str]) -> Dict[str, float]:
        """Calculate path-related metrics for the network."""
        try:
            nx_graph = graph.to_networkx()
            subgraph = nx_graph.subgraph(connected_nodes)
            
            # Sample a subset of nodes for path calculations if graph is large
            sample_size = min(100, len(connected_nodes))
            sample_nodes = random.sample(list(connected_nodes), sample_size)
            
            path_lengths = []
            for u in sample_nodes:
                for v in sample_nodes:
                    if u != v:
                        try:
                            path_length = nx.shortest_path_length(subgraph, u, v)
                            path_lengths.append(path_length)
                        except nx.NetworkXNoPath:
                            continue
            
            if path_lengths:
                return {
                    "Average Path Length": sum(path_lengths) / len(path_lengths),
                    "Max Path Length (Diameter)": max(path_lengths),
                    "Min Path Length": min(path_lengths)
                }
        except Exception as e:
            print(f"Error calculating path metrics: {str(e)}")
        
        return {}

    def initialize_graph(self, graph_manager):
        """Initialize with a graph manager."""
        try:
            self.graph_manager = graph_manager
            self.network_load_progress.visible = True
            self.network_load_progress.value = 20
            
            graph = graph_manager.graph
            self.network_load_progress.value = 40
            
            self.network_load_progress.value = 80
            
            # Update the graph pane
            self.full_graph_pane.param.trigger('object')
            
            self._update_network_stats(graph)
            
            self.network_load_progress.value = 100
            self.network_load_progress.visible = False
            
        except Exception as e:
            print(f"Error initializing network view: {str(e)}")
            import traceback
            traceback.print_exc()
            self._show_error(f"Error initializing network view: {str(e)}")
            self.network_load_progress.visible = False

    
    def update_view(self, results, computation_time, graph_manager, algorithm):
        """Update all visualization elements with new analysis results."""
        if results is None:
            self._show_no_results()
            return

        try:
            flow_value, simplified_paths, simplified_edge_flows, original_edge_flows = results

            # Make flow section visible
            self.flow_section.visible = True
            self.transactions_box.visible = True

            # Update path analysis statistics
            self._update_path_stats(
                flow_value=flow_value,
                simplified_paths=simplified_paths,
                simplified_edge_flows=simplified_edge_flows,
                original_edge_flows=original_edge_flows,
                computation_time=computation_time,
                algorithm=algorithm
            )

            # Update flow visualizations
            if graph_manager:
                # Create full flow visualization
                flow_subgraph = self._create_flow_subgraph(graph_manager.graph, original_edge_flows)
                full_plot = self.interactive_viz.create_bokeh_network(
                    flow_subgraph,
                    original_edge_flows,
                    graph_manager.data_ingestion.id_to_address,
                    "Full Flow Network",
                    simplified=False
                )
                self.flow_graph_pane.object = full_plot

                # Create simplified flow visualization
                simplified_graph = nx.DiGraph()
                for (u, v), token_flows in simplified_edge_flows.items():
                    for token, flow in token_flows.items():
                        simplified_graph.add_edge(u, v, label=token, flow=flow)
                
                simplified_plot = self.interactive_viz.create_bokeh_network(
                    simplified_graph,
                    simplified_edge_flows,
                    graph_manager.data_ingestion.id_to_address,
                    "Simplified Flow Network",
                    simplified=True
                )
                self.simplified_graph_pane.object = simplified_plot

            # Update transactions
            self._update_transactions(simplified_edge_flows, graph_manager)

        except Exception as e:
            print(f"Error updating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
            self._show_error(str(e))


    def _create_flow_subgraph(self, graph, edge_flows):
        """Create a subgraph containing only the edges and nodes involved in the flow."""
        # Create a new NetworkX DiGraph for visualization regardless of input graph type
        flow_subgraph = nx.DiGraph()
        
        # Add edges and their data from the flow
        for (u, v), flow in edge_flows.items():
            if flow > 0:
                # Get edge data based on graph type
                if hasattr(graph, 'g_nx'):  # NetworkX graph
                    if graph.has_edge(u, v):
                        edge_data = graph.get_edge_data(u, v)
                        token = edge_data.get('label')
                    else:
                        token = None
                else:  # graph-tool graph
                    if graph.has_edge(u, v):
                        edge_data = graph.get_edge_data(u, v)
                        token = edge_data.get('label')
                    else:
                        token = None
                        
                # Add edge to subgraph
                flow_subgraph.add_edge(u, v, label=token)

        return flow_subgraph

    def _update_stats(self, flow_value, simplified_paths, simplified_edge_flows, 
                      original_edge_flows, computation_time, algorithm):
        """Update the statistics panel."""
        stats = f"""
        # Analysis Results

        ## Flow Information
        - Total Flow Value: {flow_value:,} mCRC
        - Computation Time: {computation_time:.4f} seconds
        - Library Used: {algorithm}
        - Algorithm Used: {algorithm}

        ## Path Analysis
        - Number of Distinct Paths: {len(simplified_paths)}
        - Total Transfers: {sum(len(flows) for flows in simplified_edge_flows.values())}

        ## Graph Details
        - Original Edges: {len(original_edge_flows)}
        - Simplified Edge Groups: {len(simplified_edge_flows)}
        """
        self.stats_pane.object = stats

    def _update_transactions(self, simplified_edge_flows, graph_manager):
        """Update the transactions display."""
        transactions = "<ul style='list-style-type: none; padding-left: 0;'>"
        
        # Sort transfers by flow value
        sorted_transfers = []
        for (u, v), token_flows in simplified_edge_flows.items():
            for token, flow in token_flows.items():
                sorted_transfers.append((u, v, token, flow))
        sorted_transfers.sort(key=lambda x: x[3], reverse=True)
        
        # Create HTML for each transfer
        for u, v, token, flow in sorted_transfers:
            u_address = graph_manager.data_ingestion.get_address_for_id(u)
            v_address = graph_manager.data_ingestion.get_address_for_id(v)
            token_address = graph_manager.data_ingestion.get_address_for_id(token)
            
            transactions += f"""
            <li style='margin-bottom: 10px; padding: 10px; 
                       background-color: rgba(0,123,255,0.05); 
                       border-radius: 5px;'>
                <div style='display: grid; grid-template-columns: 100px auto;'>
                    <div><strong>From:</strong></div>
                    <div>{u_address} (id: {u})</div>
                    
                    <div><strong>To:</strong></div>
                    <div>{v_address}  (id: {v})</div>
                    
                    <div><strong>Flow:</strong></div>
                    <div>{flow:} mCRC</div>
                    <div><strong>Token:</strong></div>
                    <div>{token_address}  (id: {token})</div>
                   
                </div>
            </li>"""
        
        transactions += "</ul>"
        
        self.transactions_box[-1].object = f"""
            <div style="height:300px; overflow-y:scroll; 
                       border:1px solid #ddd; padding:10px; 
                       background-color: white; border-radius: 5px;">
                <div id="transactions-content">
                    <h3 style="margin-top: 0;">Transfers</h3>
                    {transactions}
                </div>
            </div>
        """
        self.transactions_box.visible = True

    def _show_no_results(self):
        """Reset visualization when no results are available."""
        self.stats_pane.object = """
        # No Results Available

        Please run an analysis to see results here.

        To get started:
        1. Configure your data source
        2. Initialize the graph
        3. Set source and sink addresses
        4. Click "Run Analysis"
        """
        
        self.transactions_box[-1].object = """
            <div style="height:300px; overflow-y:scroll; 
                       border:1px solid #ddd; padding:10px; 
                       background-color: #f8f9fa; border-radius: 5px;">
                <div id="transactions-content" style="text-align: center; 
                                                    padding-top: 100px;">
                    <h3>No Transactions Available</h3>
                    <p>Run an analysis to see transaction details.</p>
                </div>
            </div>
        """
        self.transactions_box.visible = True
        
        # Clear graph panes
        self.full_graph_pane.object = None
        self.simplified_graph_pane.object = None

    def _show_error(self, error_message: str):
        """Show error state in visualization components."""
        try:
            if hasattr(self, 'network_stats'):
                self.network_stats.object = f"""
                ## Error in Visualization

                {error_message}

                Please try refreshing the view or adjusting the settings.
                """
            
            # Clear any existing plots without triggering updates
            if hasattr(self, 'full_graph_pane'):
                self.full_graph_pane.object = None
                
        except Exception as e:
            print(f"Error showing error message: {str(e)}")

    def view(self):
        """Return the component's view with updated controls."""
        # Network Overview Section with improved layout
        network_section = pn.Column(
            pn.pane.Markdown("# Network Overview", 
                        css_classes=['section-title']),
            #self.network_stats,
            self.stats_layout,
            sizing_mode='stretch_width',
            css_classes=['section-content'],
            margin=(0, 0, 20, 0)  # Add bottom margin
        )
        
        # Store main view for later updates
        self._main_view = pn.Column(
            network_section,
            self.flow_section,
            sizing_mode='stretch_width'
        )
        
        return self._main_view
        