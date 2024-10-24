import io
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict
import panel as pn
import os
import pandas as pd
from .base import DataSourceComponent
from .csv_component import CSVDataSourceComponent
from .postgres_component import (
    PostgresManualComponent,
    PostgresEnvComponent
)
from .analysis_component import AnalysisComponent 
from .visualization_component import VisualizationComponent
from src.graph_manager import GraphManager
from src.graph import NetworkXGraph, GraphToolGraph
import time

class NetworkFlowDashboard:
    def __init__(self):
        # Initialize components
        self.data_sources = {
            "CSV": CSVDataSourceComponent(),
            "PostgreSQL (Manual)": PostgresManualComponent(),
            "PostgreSQL (ENV)": PostgresEnvComponent()
        }
        self.analysis = AnalysisComponent()
        self.visualization = VisualizationComponent()
        self.graph_manager = None
        
        # Initialize display components
        self._init_display_components()
        
        # Create the data source tabs
        self._create_data_source_tabs()
        # Set up component callbacks
        self._setup_callbacks()

    def _init_display_components(self):
        """Initialize all display components."""
        # Status displays
        self.status_indicator = pn.pane.Markdown(
            "Status: Not initialized", 
            styles={'color': 'gray'}
        )
        
        # Results and information displays
        self.stats_pane = pn.pane.Markdown(
            "Configure data source and initialize graph to begin.",
            styles={'background': '#f8f9fa', 'padding': '10px'}
        )
        
        # Transaction display
        self.transactions_box = pn.Column(
            pn.pane.Markdown("# Aggregated Transactions"),
            pn.pane.HTML("""
                <div style="height:300px; width:600px; overflow-y:scroll; 
                     border:1px solid #ddd; padding:10px;">
                    <div id="transactions-content">
                        <p>No transactions to display yet.</p>
                    </div>
                </div>
            """),
            visible=False,
            styles={'background': '#f8f9fa', 'padding': '10px'}
        )
        
        # Graph visualization panes
        self.full_graph_pane = pn.pane.PNG(
            None,
            sizing_mode='scale_both'
        )
        self.simplified_graph_pane = pn.pane.PNG(
            None,
            sizing_mode='scale_both'
        )

    def _create_data_source_tabs(self):
        """Create tabs for different data sources."""
        self.tab_names = list(self.data_sources.keys())
        tab_contents = [(name, component.view()) 
                       for name, component in self.data_sources.items()]
        
        self.data_source_tabs = pn.Tabs(
            *tab_contents,
            styles={'width': '100%'}
        )

    def _setup_callbacks(self):
        """Set up callbacks for components and tabs."""
        # Analysis component callbacks
        if hasattr(self.analysis, 'init_button'):
            self.analysis.init_button.on_click(self._initialize_graph_manager)
        
        if hasattr(self.analysis, 'run_button'):
            self.analysis.run_button.on_click(self._run_analysis)
        
        # Tab change callback
        if hasattr(self.data_source_tabs, 'param'):
            self.data_source_tabs.param.watch(self._handle_tab_change, 'active')

    def _handle_tab_change(self, event):
        """Handle data source tab changes."""
        # Reset graph manager when data source changes
        self.graph_manager = None
        if hasattr(self.analysis, 'update_status'):
            self.analysis.update_status("Data source changed. Please initialize graph.", 'warning')

    def _get_active_data_source(self):
        """Get the currently active data source component."""
        try:
            active_index = self.data_source_tabs.active
            active_name = self.tab_names[active_index]
            return self.data_sources[active_name]
        except Exception as e:
            print(f"Error getting active data source: {str(e)}")
            return self.data_sources["CSV"]  # Default to CSV if there's an error

    def _initialize_graph_manager(self, event):
        """Initialize the graph manager with the current data source configuration."""
        try:
            self.analysis.init_status.object = "Initializing graph..."
            self.analysis.init_status.styles = {'color': 'blue'}
            self.stats_pane.object = "Loading graph... Please wait."
            
            # Get active data source component
            active_source = self._get_active_data_source()
            
            # Get configuration (file paths)
            config = active_source.get_configuration()
            if not config:
                raise ValueError("No valid configuration available")
            
            # Initialize graph manager
            self.graph_manager = GraphManager(
                config,
                'networkx' if self.analysis.graph_library == 'NetworkX' else 'graph_tool'
            )
            
            # Update status and info
            if isinstance(self.graph_manager.graph, NetworkXGraph):
                num_nodes = self.graph_manager.graph.g_nx.number_of_nodes()
                num_edges = self.graph_manager.graph.g_nx.number_of_edges()
            else:  # GraphToolGraph
                num_nodes = self.graph_manager.graph.g_gt.num_vertices()
                num_edges = self.graph_manager.graph.g_gt.num_edges()
            
            # Update status indicators
            self.analysis.init_status.object = "Graph initialized successfully"
            self.analysis.init_status.styles = {'color': 'green'}
            self.stats_pane.object = f"""
            # Graph Information
            - Number of nodes: {num_nodes:,}
            - Number of edges: {num_edges:,}
            - Graph library: {self.analysis.graph_library}
            
            Graph is ready for analysis.
            """
            
            # Enable analysis inputs
            self.analysis.enable_analysis_inputs(True)
            
        except Exception as e:
            error_msg = f"Initialization Error: {str(e)}"
            print(f"Detailed error: {str(e)}")  # Debug print
            self.analysis.init_status.object = error_msg
            self.analysis.init_status.styles = {'color': 'red'}
            self.stats_pane.object = f"Error initializing graph: {str(e)}"
            self.graph_manager = None
            self.analysis.enable_analysis_inputs(False)

    def _update_graph_info(self):
        """Update graph information display."""
        if not self.graph_manager:
            self.stats_pane.object = """
            # Graph Status
            No graph initialized. Please configure data source and initialize graph.
            """
            return
            
        try:
            # Get graph size information
            if isinstance(self.graph_manager.graph, NetworkXGraph):
                num_nodes = self.graph_manager.graph.g_nx.number_of_nodes()
                num_edges = self.graph_manager.graph.g_nx.number_of_edges()
            else:  # GraphToolGraph
                num_nodes = self.graph_manager.graph.g_gt.num_vertices()
                num_edges = self.graph_manager.graph.g_gt.num_edges()
            
            # Update stats display
            self.stats_pane.object = f"""
            # Graph Information
            - Number of nodes: {num_nodes:,}
            - Number of edges: {num_edges:,}
            - Graph library: {self.analysis.graph_library}
            
            Graph is ready for analysis.
            """
            
        except Exception as e:
            self.stats_pane.object = f"""
            # Error Getting Graph Information
            An error occurred while retrieving graph information: {str(e)}
            """

    def _run_analysis(self, event):
        """Run the flow analysis with current configuration."""
        if not self.graph_manager:
            self.analysis.update_status("Please initialize the graph first", 'warning')
            return

        try:
            # Update status
            self.analysis.update_status("Running analysis...", 'progress')
            
            # Get analysis parameters
            source = self.analysis.source
            sink = self.analysis.sink
            flow_func = self.analysis.get_algorithm_func()
            cutoff = self.analysis.requested_flow_mCRC or None

            # Validate addresses
            if not (source and sink):
                raise ValueError("Please provide both source and sink addresses")

            # Run analysis with timing
            start_time = time.time()
            self.results = self.graph_manager.analyze_flow(  # Store results in self.results
                source=source,
                sink=sink,
                flow_func=flow_func,
                cutoff=cutoff
            )
            computation_time = time.time() - start_time

            # Update visualization using the dashboard's method
            self.update_results_view(computation_time)

            self.analysis.update_status("Analysis completed successfully", 'success')

        except Exception as e:
            error_msg = f"Analysis Error: {str(e)}"
            print(f"Detailed error: {str(e)}")  # Debug print
            self.analysis.update_status(error_msg, 'error')
            # Clear any previous results
            self.results = None
            self.update_results_view(0)  # Update view to show error state

    def _create_sidebar(self):
        """Create the dashboard sidebar with proper spacing."""
        return pn.Column(
            pn.Accordion(
                ("Data Source Configuration", pn.Column(
                    self.data_source_tabs,
                    sizing_mode='stretch_width',
                    margin=(0, 10)
                )),
                active=[0],
                sizing_mode='stretch_width',
                margin=(5, 0)
            ),
            self.analysis.view(),
            margin=(10, 10, 20, 10),
            styles={'background': '#f8f9fa'},  # Correct way to set background
            min_height=800,
            sizing_mode='stretch_width',
            scroll=True
        )

    def update_results_view(self, computation_time):
        """Update the UI with analysis results"""
        if self.results is None:
            self.stats_pane.object = "No results yet. Click 'Run Analysis' to start."
            self.transactions_box[-1].object = """
                <div style="height:300px; width:600px; overflow-y:scroll; border:1px solid #ddd; padding:10px;">
                    <div id="transactions-content">No transactions to display.</div>
                </div>
            """
            self.full_graph_pane.object = None
            self.simplified_graph_pane.object = None
            self.transactions_box.visible = False
            return

        flow_value, simplified_paths, simplified_edge_flows, original_edge_flows = self.results

        # Update statistics - get algorithm from analysis component
        stats = f"""
        # Results
        - Algorithm: {self.analysis.algorithm}
        - Requested Flow: {self.analysis.requested_flow_mCRC if self.analysis.requested_flow_mCRC else 'Max Flow'}
        - Achieved Flow Value: {flow_value}
        - Computation Time: {computation_time:.4f} seconds
        - Number of Simplified Paths: {len(simplified_paths)}
        - Number of Original Edges: {len(original_edge_flows)}
        - Number of Simplified Transfers: {sum(len(flows) for flows in simplified_edge_flows.values())}
        """
        self.stats_pane.object = stats
        
        # Format transactions view
        transactions = "<ul style='list-style-type: none; padding-left: 0;'>"
        for (u, v), token_flows in simplified_edge_flows.items():
            u_address = self.graph_manager.data_ingestion.get_address_for_id(u)
            v_address = self.graph_manager.data_ingestion.get_address_for_id(v)
            for token, flow in token_flows.items():
                token_address = self.graph_manager.data_ingestion.get_address_for_id(token)
                transactions += f"""
                <li style='margin-bottom: 10px; padding: 5px; border-bottom: 1px solid #eee;'>
                    <div><strong>From:</strong> {u_address[:6]}...{u_address[-4:]}</div>
                    <div><strong>To:</strong> {v_address[:6]}...{v_address[-4:]}</div>
                    <div><strong>Flow:</strong> {flow}</div>
                    <div><strong>Token:</strong> {token_address[:6]}...{token_address[-4:]}</div>
                </li>"""
        transactions += "</ul>"
        
        self.transactions_box[-1].object = f"""
            <div style="height:300px; width:600px; overflow-y:scroll; border:1px solid #ddd; padding:10px;">
                <div id="transactions-content">{transactions}</div>
            </div>
        """
        self.transactions_box.visible = True
        
        try:
            # Update visualizations using the dashboard's own methods
            self.full_graph_pane.object = self.create_path_graph(
                self.graph_manager.graph.g_nx if hasattr(self.graph_manager.graph, 'g_nx') 
                else self.graph_manager.graph,
                original_edge_flows,
                "Full Graph"
            )

            self.simplified_graph_pane.object = self.create_aggregated_graph(
                simplified_edge_flows,
                "Simplified Transfers Graph"
            )

        except Exception as e:
            error_msg = f"Error creating visualizations: {str(e)}"
            self.status_indicator.object = error_msg
            self.status_indicator.styles = {'color': 'red'}
            print(f"Visualization error: {str(e)}")
            import traceback
            traceback.print_exc()

    def create_path_graph(self, G, edge_flows, title):
        """Create visualization of the full path graph"""
        plt.figure(figsize=(8, 5))
        
        # Create a subgraph with only the nodes and edges in the flow
        subgraph = nx.DiGraph()
        for (u, v) in edge_flows.keys():
            subgraph.add_edge(u, v)

        try:
            # Identify source and sink
            source = next(node for node in subgraph.nodes() if subgraph.in_degree(node) == 0)
            sink = next(node for node in subgraph.nodes() if subgraph.out_degree(node) == 0)

            pos = self.visualization.custom_flow_layout(subgraph, source, sink)
            
            # Draw nodes
            noncross_nodes = [node for node in subgraph.nodes() if '_' not in str(node)]
            nx.draw_networkx_nodes(subgraph, pos, nodelist=noncross_nodes, 
                                node_color='lightblue', node_shape='o', node_size=200)
            
            cross_nodes = [node for node in subgraph.nodes() if '_' in str(node)]
            nx.draw_networkx_nodes(subgraph, pos, nodelist=cross_nodes, 
                                node_color='red', node_shape='P', node_size=100)
            
            nx.draw_networkx_labels(subgraph, pos, font_size=6, font_weight='bold')

            # Draw edges
            nx.draw_networkx_edges(subgraph, pos, edge_color='gray', 
                                arrows=True, arrowsize=10, connectionstyle="arc3,rad=0.1")

            # Prepare edge labels
            edge_labels = {}
            for (u, v), flow in edge_flows.items():
                edge_data = G.get_edge_data(u, v)
                label = f"Flow: {flow}\nToken: {edge_data['label']}"
                edge_labels[(u, v)] = label
            
            nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=4)

            plt.title(title, fontsize=12)
            plt.axis('off')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            return buf
                
        except Exception as e:
            plt.close()
            print(f"Error creating path graph: {str(e)}")
            return None

    def create_aggregated_graph(self, simplified_edge_flows, title):
        """Create visualization of the simplified graph"""
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        
        try:
            # Create a graph with simplified flows
            G = nx.MultiDiGraph()
            for (u, v), token_flows in simplified_edge_flows.items():
                for token, flow in token_flows.items():
                    G.add_edge(u, v, flow=flow, token=token)

            # Identify source and sink
            source = next(node for node in G.nodes() if G.in_degree(node) == 0)
            sink = next(node for node in G.nodes() if G.out_degree(node) == 0)

            # Generate positions for nodes
            pos = self.visualization.custom_flow_layout(G, source, sink)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                node_shape='o', node_size=200, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=6, font_weight='bold', ax=ax)

            # Organize edges between the same nodes
            edges_between_nodes = defaultdict(list)
            for u, v, k in G.edges(keys=True):
                edges_between_nodes[(u, v)].append(k)

            # Draw edges with different curvatures
            for (u, v), keys in edges_between_nodes.items():
                num_edges = len(keys)
                rad_list = [0.15] if num_edges == 1 else np.linspace(-0.3, 0.3, num_edges)
                
                for k, rad in zip(keys, rad_list):
                    edge_data = G[u][v][k]
                    label = f"Flow: {edge_data['flow']}\nToken: {edge_data['token']}"

                    x1, y1 = pos[u]
                    x2, y2 = pos[v]

                    arrow = mpatches.FancyArrowPatch(
                        (x1, y1), (x2, y2),
                        connectionstyle=f"arc3,rad={rad}",
                        arrowstyle='-|>',
                        mutation_scale=20,
                        color='gray',
                        linewidth=0.5,
                        zorder=1
                    )
                    ax.add_patch(arrow)

                    # Calculate and add label
                    dx = x2 - x1
                    dy = y2 - y1
                    angle = np.arctan2(dy, dx)
                    offset = np.array([-np.sin(angle), np.cos(angle)]) * rad * 0.5
                    midpoint = np.array([(x1 + x2) / 2, (y1 + y2) / 2]) + offset

                    plt.text(midpoint[0], midpoint[1], label, 
                        fontsize=4, ha='center', va='center', zorder=2)

            plt.title(title, fontsize=12)
            plt.axis('off')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            return buf
                
        except Exception as e:
            plt.close()
            print(f"Error creating aggregated graph: {str(e)}")
            return None


    def view(self):
        """Create and return the dashboard view."""
        # Create visualization tabs
        graph_tabs = pn.Tabs(
            ("Full Graph", self.full_graph_pane),
            ("Aggregated Graph", self.simplified_graph_pane),
            styles={'background': 'white'},
            sizing_mode='stretch_width'
        )

        # Create main content with visualization
        main_content = pn.Column(
            pn.pane.Markdown("## Network Flow Analysis"),
            self.stats_pane,
            self.transactions_box,
            pn.pane.Markdown("## Flow Paths"),
            graph_tabs,
            margin=(10, 10, 10, 10),
            sizing_mode='stretch_width'
        )

        # Create template
        template = pn.template.MaterialTemplate(
            title="pyFinder Flow Analysis Dashboard",
            header_background="#007BFF",
            header_color="#ffffff",
            sidebar=self._create_sidebar(),
            main=main_content,
            sidebar_width=500
        )

        return template


def create_dashboard():
    """Create and return a new dashboard instance."""
    dashboard = NetworkFlowDashboard()
    return dashboard.view()

if __name__ == "__main__":
    # This is handled in run.py
    pass