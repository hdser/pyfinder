import panel as pn
import param
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

from .base_component import BaseComponent
from .interactive_visualization import InteractiveVisualization

class VisualizationComponent(BaseComponent):
    """Component for visualizing network flow analysis results."""
    
    def __init__(self, **params):
        self.interactive_viz = InteractiveVisualization()
        super().__init__(**params)

    def _create_components(self):
        """Create visualization components."""
        # Statistics panel
        self.stats_pane = pn.pane.Markdown(
            "Results will appear here after analysis.",
            sizing_mode='stretch_width'
        )
        
        # Graph visualization panes
        self.full_graph_pane = pn.pane.Bokeh(
            sizing_mode='stretch_both',
            min_height=500,
            margin=(10, 10)
        )
        
        self.simplified_graph_pane = pn.pane.Bokeh(
            sizing_mode='stretch_both',
            min_height=500,
            margin=(10, 10)
        )
        
        # Transactions display
        self.transactions_box = self._create_transactions_box()

    def _create_transactions_box(self):
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
        pass  # No callbacks needed for now

    def update_view(self, results, computation_time, graph_manager, algorithm):
        """Update all visualization elements with new analysis results."""
        if results is None:
            self._show_no_results()
            return

        try:
            flow_value, simplified_paths, simplified_edge_flows, original_edge_flows = results
            
            # Update statistics
            self._update_stats(
                flow_value=flow_value,
                simplified_paths=simplified_paths,
                simplified_edge_flows=simplified_edge_flows,
                original_edge_flows=original_edge_flows,
                computation_time=computation_time,
                algorithm=algorithm
            )

            # Update transactions
            self._update_transactions(simplified_edge_flows, graph_manager)

            try:
                print("Creating full graph visualization...")
                # Create subgraph with only the nodes and edges in the flow
                flow_subgraph = self._create_flow_subgraph(graph_manager.graph.g_nx, original_edge_flows)

                # Create full graph plot using the flow subgraph
                full_plot = self.interactive_viz.create_bokeh_network(
                    flow_subgraph,
                    original_edge_flows,
                    "Flow Paths",
                    simplified=False
                )
                full_plot.sizing_mode = 'stretch_both'
                self.full_graph_pane.object = full_plot
                print("Full graph plot set")

                print("Creating simplified graph...")
                # Create simplified graph
                simplified_graph = nx.DiGraph()
                for (u, v), token_flows in simplified_edge_flows.items():
                    for token, flow in token_flows.items():
                        simplified_graph.add_edge(u, v, label=token)

                # Create simplified plot
                simplified_plot = self.interactive_viz.create_bokeh_network(
                    simplified_graph,
                    simplified_edge_flows,
                    "Simplified Flow Paths",
                    simplified=True
                )
                simplified_plot.sizing_mode = 'stretch_both'
                self.simplified_graph_pane.object = simplified_plot
                print("Simplified plot set")

            except Exception as e:
                print(f"Error creating visualization: {str(e)}")
                import traceback
                traceback.print_exc()
                self._show_error(f"Visualization error: {str(e)}")

        except Exception as e:
            print(f"Error updating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
            self._show_error(str(e))

    def _create_flow_subgraph(self, G_full, edge_flows):
        """Create a subgraph containing only the edges and nodes involved in the flow."""
        edges_in_flow = list(edge_flows.keys())
        flow_subgraph = G_full.edge_subgraph(edges_in_flow).copy()
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
                    <div>{u_address[:6]}...{u_address[-4:]}</div>
                    <div><strong>To:</strong></div>
                    <div>{v_address[:6]}...{v_address[-4:]}</div>
                    <div><strong>Flow:</strong></div>
                    <div>{flow:,} mCRC</div>
                    <div><strong>Token:</strong></div>
                    <div>{token_address[:6]}...{token_address[-4:]}</div>
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
        """Show error state in all visualization components."""
        self.stats_pane.object = f"""
        # Error in Analysis

        An error occurred while processing the results:

        ```
        {error_message}
        ```

        Please try again or contact support if the issue persists.
        """
        
        self.transactions_box[-1].object = """
            <div style="height:300px; overflow-y:scroll; 
                       border:1px solid #ddd; padding:10px; 
                       background-color: #fff0f0; border-radius: 5px;">
                <div id="transactions-content" style="text-align: center; 
                                                    padding-top: 100px;">
                    <h3 style="color: red;">Error Processing Transactions</h3>
                    <p>Please try again or contact support.</p>
                </div>
            </div>
        """
        
        # Clear graph panes
        self.full_graph_pane.object = None
        self.simplified_graph_pane.object = None

    def view(self):
        """Return the component's view."""
        # Create graph containers with proper sizing
        full_graph_container = pn.Column(
            self.full_graph_pane,
            sizing_mode='stretch_both',
            min_height=600
        )
        
        simplified_graph_container = pn.Column(
            self.simplified_graph_pane,
            sizing_mode='stretch_both',
            min_height=600
        )

        # Create tabs
        graph_tabs = pn.Tabs(
            ("Flow Network", full_graph_container),
            ("Simplified Flow Network", simplified_graph_container),
            sizing_mode='stretch_both'
        )

        # Main layout
        return pn.Column(
            pn.pane.Markdown("## Network Flow Analysis"),
            self.stats_pane,
            self.transactions_box,
            pn.pane.Markdown("## Flow Paths"),
            graph_tabs,
            sizing_mode='stretch_both',
            min_width=800
        )
