import panel as pn
import param
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from bokeh.plotting import figure
from src.graph import NetworkXGraph, GraphToolGraph 

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
        """Create visualization components with fixed node limits."""
        # Stats components
        self.network_stats = pn.pane.Markdown(
            "Network statistics will appear here",
            styles={'background': '#f8f9fa', 'padding': '15px', 
                   'border-radius': '5px', 'border': '1px solid #dee2e6'},
            sizing_mode='stretch_width'
        )
        
        self.path_stats = pn.pane.Markdown(
            "Path analysis results will appear here",
            styles={'background': '#f8f9fa', 'padding': '15px', 
                   'border-radius': '5px', 'border': '1px solid #dee2e6'},
            sizing_mode='stretch_width'
        )
        
       

        self.refresh_button = pn.widgets.Button(
            name="↻ Refresh View",
            button_type="default",
            width=100
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
            
            # Convert graph if using graph-tool
            if isinstance(self.graph_manager.graph, GraphToolGraph):
                # Create NetworkX graph from graph-tool
                G = nx.DiGraph()
                
                # Add nodes
                for v in self.graph_manager.graph.g_gt.vertices():
                    node_id = str(self.graph_manager.graph.vertex_id[v])
                    G.add_node(node_id)
                
                # Add edges with their properties
                for e in self.graph_manager.graph.g_gt.edges():
                    source_id = str(self.graph_manager.graph.vertex_id[e.source()])
                    target_id = str(self.graph_manager.graph.vertex_id[e.target()])
                    edge_props = {
                        'label': self.graph_manager.graph.token[e],
                        'capacity': self.graph_manager.graph.capacity[e]
                    }
                    G.add_edge(source_id, target_id, **edge_props)
                    
                nx_graph = G
            else:
                nx_graph = self.graph_manager.graph.g_nx
            
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
            
            self._update_network_stats(nx_graph)
            
            self.network_load_progress.value = 100
            self.network_load_progress.visible = False
            
        except Exception as e:
            print(f"Error refreshing network view: {str(e)}")
            import traceback
            traceback.print_exc()
            self._show_error(f"Error refreshing network view: {str(e)}")
            self.network_load_progress.visible = False

    def _update_network_stats(self, G: nx.DiGraph):
        """Update network statistics display."""
        try:
            # Exclude nodes with no connections
            connected_nodes = [node for node, degree in G.degree() if degree > 0]
            total_conn_nodes = len(connected_nodes)
            total_edges = G.number_of_edges()
            total_nodes = G.number_of_nodes()
            
            if not total_nodes:
                stats = "No connected nodes found in the network"
            else:
                # Calculate network metrics
                avg_degree = 2 * total_edges / total_nodes
                density = nx.density(G.subgraph(connected_nodes))
                
                stats = f"""
                ## Network Statistics
                
                ### Overview
                - Total Nodes: {total_nodes:,}
                - Total Connected Nodes: {total_conn_nodes:,}
                - Total Edges: {total_edges:,}
                    
                ### Metrics
                - Average Degree: {avg_degree:.2f}
                - Network Density: {density:.4f}
                """
            
            self.network_stats.object = stats
            
        except Exception as e:
            print(f"Error updating network stats: {str(e)}")
            import traceback
            traceback.print_exc()
            self._show_error(f"Error updating network stats: {str(e)}")

    def initialize_graph(self, graph_manager):
        """Initialize with a graph manager."""
        try:
            self.graph_manager = graph_manager
            self.network_load_progress.visible = True
            self.network_load_progress.value = 20
            
            # Convert graph-tool graph to NetworkX if needed
            if isinstance(graph_manager.graph, GraphToolGraph):
                # Create NetworkX graph from graph-tool
                G = nx.DiGraph()
                g_gt = graph_manager.graph.g_gt
                vertex_id = graph_manager.graph.vertex_id
                token = graph_manager.graph.token
                capacity = graph_manager.graph.capacity
                
                # Add nodes
                for v in g_gt.vertices():
                    node_id = str(vertex_id[v])
                    G.add_node(node_id)
                
                # Add edges with their properties
                for e in g_gt.edges():
                    source_id = str(vertex_id[e.source()])
                    target_id = str(vertex_id[e.target()])
                    edge_props = {
                        'label': token[e],
                        'capacity': capacity[e]
                    }
                    G.add_edge(source_id, target_id, **edge_props)
                
                nx_graph = G
            else:
                nx_graph = graph_manager.graph.g_nx
            
            self.network_load_progress.value = 40
            
            
            self.network_load_progress.value = 80
            
            # Update the graph pane
            #self.full_graph_pane.object = new_plot
            self.full_graph_pane.param.trigger('object')
            
            self._update_network_stats(nx_graph)
            
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
        self.path_stats.object = stats

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
        # Network Overview Section with simplified controls
        network_section = pn.Column(
            pn.pane.Markdown("# Network Overview", 
                        css_classes=['section-title']),
            self.network_stats,
            pn.Row(
                self.network_load_progress,
                align='center',
                css_classes=['network-controls']
            ),
            self.full_graph_pane,
            sizing_mode='stretch_width',
            css_classes=['section-content']
        )
        
        # Store main view for later updates
        self._main_view = pn.Column(
            network_section,
            self.flow_section,
            sizing_mode='stretch_width'
        )
        
        return self._main_view