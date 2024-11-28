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


pn.config.raw_css.append("""
.section-title {
    background: #f8f9fa;
    padding: 15px 20px;
    margin: 0 0 20px 0;
    border-radius: 8px;
    border-left: 5px solid #007bff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.section-title h1 {
    margin: 0;
    color: #2c3e50;
    font-size: 24px;
    font-weight: 500;
}

/* Alternative style for subsections if needed */
.subsection-title {
    background: #e9ecef;
    padding: 12px 20px;
    margin: 0 0 15px 0;
    border-radius: 6px;
    border-left: 4px solid #6c757d;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.subsection-title h1 {
    margin: 0;
    color: #495057;
    font-size: 20px;
    font-weight: 500;
}
""")

class VisualizationComponent(BaseComponent):
    """Component for visualizing network flow analysis results."""
    
    def __init__(self, **params):
        self.interactive_viz = InteractiveVisualization()
        self.full_network_plot = None
        self.graph_manager = None
        super().__init__(**params)

    def _create_components(self):
        """Create visualization components with improved layout."""
        # Create graph visualization panes first
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

        # Progress indicator
        self.network_load_progress = pn.indicators.Progress(
            name='Loading Network',
            value=0,
            max=100,
            visible=False,
            width=200
        )

        # Flow info panes
        self.flow_info_pane = pn.pane.HTML(
            "",
            sizing_mode='stretch_width'
        )
        
        self.path_analysis_pane = pn.pane.HTML(
            "",
            sizing_mode='stretch_width'
        )

        # Create layout for path analysis
        self.path_stats_top = pn.Row(
            self.flow_info_pane,
            sizing_mode='stretch_width'
        )

        self.path_stats_bottom = pn.Row(
            self.path_analysis_pane,
            sizing_mode='stretch_width'
        )

        # Combine into final path stats layout
        self.path_stats_layout = pn.Column(
            self.path_stats_top,
            self.path_stats_bottom,
            sizing_mode='stretch_width',
            margin=(0, 0, 20, 0)
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
            pn.pane.Markdown(
                "# Flow Analysis", 
                css_classes=['section-title'],
                margin=(0, 0, 20, 0)
            ),
            self.path_stats_layout,  # Use path_stats_layout instead of path_stats
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

        self.capacity_metrics_pane = pn.pane.HTML(
            "",
            sizing_mode='stretch_width'
        )
        
        # Update stats layout to include capacity metrics
        self.stats_layout = pn.Column(
            self.stats_top,
            self.metrics_pane,
            self.capacity_metrics_pane,
            self.stats_bottom,
            sizing_mode='stretch_width',
            margin=(0, 0, 20, 0)
        )

    def _update_path_stats(self, source: int, sink: int, flow_value: int, simplified_paths: list, 
                      simplified_edge_flows: dict, original_edge_flows: dict, 
                      computation_time: float, algorithm: str, requested_flow: Optional[str] = None):
        """Update the path analysis statistics."""
        try:
            # Calculate metrics
            source_id = self.graph_manager.data_ingestion.get_id_for_address(source)
            sink_id = self.graph_manager.data_ingestion.get_id_for_address(sink)
            source_outflow = self.graph_manager.graph.get_node_outflow_capacity(source_id)
            sink_inflow = self.graph_manager.graph.get_node_inflow_capacity(sink_id)
            theoretical_max = min(source_outflow, sink_inflow)

            #self.graph_manager.graph._debug_capacities(source_id, is_source=True) 
            #self.graph_manager.graph._debug_capacities(sink_id, is_source=True) 
            
            # Calculate actual flow by summing flows only at sink edges
            actual_flow = 0
            for (u, v), flows in simplified_edge_flows.items():
                if v == sink_id:  # Only count flows that reach the sink
                    actual_flow += sum(flows.values())
            
            # Calculate flow gap using actual flow
            if requested_flow is not None and requested_flow.strip():
                target_flow = int(requested_flow)
                flow_gap = target_flow - actual_flow
                flow_gap_percentage = (flow_gap / target_flow * 100) if target_flow > 0 else 0
                comparison_text = "Requested Flow"
            else:
                target_flow = theoretical_max
                flow_gap = theoretical_max - actual_flow
                flow_gap_percentage = (flow_gap / theoretical_max * 100) if theoretical_max > 0 else 0
                comparison_text = "Theoretical Maximum"

            # Format flow information using actual flow
            flow_info = {
                "Achieved Flow": f"{actual_flow:,} mCRC",
                comparison_text: f"{target_flow:,} mCRC",
                "Flow Gap": f"{flow_gap:,} mCRC ({flow_gap_percentage:.2f}%)",
                "Flow Type": "Maximum Flow" if requested_flow is None else "Requested Flow",
                "Computation Time": f"{computation_time:.4f} seconds",
                "Algorithm": algorithm,
                "Source Max Outflow": f"{source_outflow:,} mCRC",
                "Sink Max Inflow": f"{sink_inflow:,} mCRC"
            }

            # Format path analysis metrics
            path_flows = [path[2] for path in simplified_paths]
            path_info = {
                "Distinct Paths": f"{len(simplified_paths):,}",
                "Total Transfers": f"{sum(len(flows) for flows in simplified_edge_flows.values()):,}",
                "Original Edges": f"{len(original_edge_flows):,}",
                "Simplified Groups": f"{len(simplified_edge_flows):,}",
                "Average Flow per Path": f"{actual_flow / len(simplified_paths):,.2f} mCRC" if simplified_paths else "0",
                "Min Flow Path": f"{min(path_flows):,.2f} mCRC" if path_flows else "0",
                "Max Flow Path": f"{max(path_flows):,.2f} mCRC" if path_flows else "0"
            }

            # Update displays using the existing _format_stats_box method
            self.flow_info_pane.object = self._format_stats_box("Flow Information", flow_info)
            self.path_analysis_pane.object = self._format_stats_box("Path Analysis", path_info)

        except Exception as e:
            print(f"Error updating path stats: {str(e)}")
            import traceback
            traceback.print_exc()
            self._show_error(f"Error updating path stats: {str(e)}")

   
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
                pn.pane.Markdown(
                    "# Network Stats", 
                    css_classes=['section-title'],
                    margin=(0, 0, 20, 0)
                ),
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
        """Update network statistics with consistent boxed layout and detailed metrics."""
        try:
            # Calculate all metrics first
            all_nodes = graph.get_vertices()
            real_nodes = {v for v in all_nodes if '_' not in v}
            intermediate_nodes = {v for v in all_nodes if '_' in v}

            # Calculate source/sink nodes
            isolated_nodes = {v for v in real_nodes if graph.degree(v) == 0}

            # Format overview stats
            overview_stats = {
                "Total Real Nodes": f"{len(real_nodes):,}",
                "Total Intermediate Nodes": f"{len(intermediate_nodes):,}",
                "Total Edges": f"{graph.num_edges():,}",
                "Isolated Nodes": f"{len(isolated_nodes):,}"
            }
            overview_md = self._format_stats_box("Network Overview", overview_stats)
            self.network_stats.object = overview_md

            # Calculate detailed network metrics
            out_degrees = [graph.out_degree(v) for v in all_nodes]
            in_degrees = [graph.in_degree(v) for v in all_nodes]
            total_degrees = [graph.degree(v) for v in all_nodes]

            metrics = {
                "Edge-Vertex Ratio": f"{graph.num_edges() / len(all_nodes):.4f}" if len(all_nodes) > 0 else "0",
                "Average Total Degree": f"{np.mean(total_degrees):.4f}",
                "Median Total Degree": f"{np.median(total_degrees):.4f}",
                "Maximum Total Degree": f"{max(total_degrees):,}",
                "Average Out-Degree": f"{np.mean(out_degrees):.4f}",
                "Median Out-Degree": f"{np.median(out_degrees):.4f}",
                "Maximum Out-Degree": f"{max(out_degrees):,}",
                "Average In-Degree": f"{np.mean(in_degrees):.4f}",
                "Median In-Degree": f"{np.median(in_degrees):.4f}",
                "Maximum In-Degree": f"{max(in_degrees):,}"
            }
            metrics_md = self._format_stats_box("Network Metrics", metrics)
            self.metrics_pane.object = metrics_md

            # Calculate capacity metrics
            capacity_metrics = self._calculate_capacity_metrics(graph)
            capacity_md = self._format_stats_box("Capacity Metrics (mCRC)", capacity_metrics)
            self.capacity_metrics_pane.object = capacity_md

            # Create and update histogram
            self._create_connection_histogram(graph, intermediate_nodes)

        except Exception as e:
            print(f"Error updating network stats: {str(e)}")
            import traceback
            traceback.print_exc()
            self._show_error(f"Error updating network stats: {str(e)}")

    def _calculate_capacity_metrics(self, graph: BaseGraph) -> dict:
        """Calculate capacity-related metrics for the network."""
        try:
            edge_capacities = []
            real_capacities = {}

            # Collect capacities
            for u, v, data in graph.get_edges():
                capacity = data.get('capacity', 0)
                edge_capacities.append(capacity)
                
                # Track capacities through intermediate nodes
                if '_' in v:
                    real_capacities[v] = real_capacities.get(v, 0) + capacity
                

            # Calculate metrics
            metrics = {
                "Total Network Capacity": f"{sum(list(real_capacities.values())):,}",
                "Average Node Capacity": f"{np.mean(list(real_capacities.values())):,.2f}",
                "Median Node Capacity": f"{np.median(list(real_capacities.values())):,.2f}",
                "Maximum Node Capacity": f"{max(real_capacities.values()):,}",
                "Minimum Node Capacity": f"{min(real_capacities.values()):,}",
            }
            return metrics
        except Exception as e:
            print(f"Error calculating capacity metrics: {str(e)}")
            return {"Error": "Failed to calculate capacity metrics"}

    def _create_connection_histogram(self, graph: BaseGraph, intermediate_nodes: set):
        """Create histogram visualization for intermediate node connections."""
        try:
            import numpy as np
            from bokeh.plotting import figure
            from bokeh.models import (
                ColumnDataSource, 
                HoverTool, 
                BoxAnnotation,
                LinearAxis, 
                Grid
            )

            # Calculate connection counts
            connection_counts = [graph.degree(node) for node in intermediate_nodes]

            # Calculate histogram data
            hist, edges = np.histogram(connection_counts, bins=150)
            
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

            # Style axes and plot
            self._style_histogram_plot(p)

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

            # Calculate and update connection statistics
            connection_stats = {
                "Average Connections": f"{np.mean(connection_counts):.0f}",
                "Median Connections": f"{np.median(connection_counts):.0f}",
                "Maximum Connections": f"{np.max(connection_counts):,}",
                "Minimum Connections": f"{np.min(connection_counts):,}"
            }
            stats_md = self._format_stats_box("Intermediate Node Statistics", connection_stats)
            self.histogram_stats_pane.object = stats_md

        except Exception as e:
            print(f"Error creating histogram: {str(e)}")
            import traceback
            traceback.print_exc()

    def _style_histogram_plot(self, p: figure):
        """Apply consistent styling to histogram plot."""
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


    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        return squared_diff_sum / len(values)


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

    
    def update_view(self, results=None, computation_time=0, graph_manager=None, algorithm=None):
        """Update all visualization elements with new analysis results."""
        if results is None:
            self._show_no_results()
            return

        try:
            flow_value, simplified_paths, simplified_edge_flows, original_edge_flows = results

            # Make flow section visible
            self.flow_section.visible = True
            self.transactions_box.visible = True

            # Store any requested flow value from the current state
            current_requested_flow = None
            if hasattr(self, 'analysis') and hasattr(self.analysis, 'requested_flow_mCRC'):
                current_requested_flow = self.analysis.requested_flow_mCRC if self.analysis.requested_flow_mCRC.strip() else None

            # Update path analysis statistics only once
            if hasattr(self, 'source') and hasattr(self, 'sink'):
                self._update_path_stats(
                    source=self.source,
                    sink=self.sink,
                    flow_value=flow_value,
                    simplified_paths=simplified_paths,
                    simplified_edge_flows=simplified_edge_flows,
                    original_edge_flows=original_edge_flows,
                    computation_time=computation_time,
                    algorithm=algorithm,
                    requested_flow=current_requested_flow
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
        tf_json = []
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

        #    tf_json.append({
        #        "from": u_address,
        #        "to": v_address,
        #        "token": token_address,
        #        "flow": flow * 10**15
        #    })

        #import json
        #with open('transfer_data.json', 'w') as f:
        #    json.dump(tf_json, f)
        
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
            pn.pane.Markdown("# Network Stats", 
                        css_classes=['section-title']),
            self.stats_layout,
            sizing_mode='stretch_width',
            css_classes=['section-content'],
            margin=(0, 0, 20, 0)  
        )
        
        # Store main view for later updates
        self._main_view = pn.Column(
            network_section,
            self.flow_section,
            sizing_mode='stretch_width'
        )
        
        return self._main_view
        