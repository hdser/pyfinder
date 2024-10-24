# dashboard/visualization/visualization_component.py
import panel as pn
import param
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import io
from dashboard.base_component import BaseComponent
from src.visualization import Visualization as SrcVisualization

class VisualizationComponent(BaseComponent):
    """Component for visualizing network flow analysis results."""
    
    def __init__(self, **params):
        self.src_visualization = SrcVisualization()
        super().__init__(**params)

    def _create_components(self):
        """Create visualization components."""
        # Statistics panel
        self.stats_pane = pn.pane.Markdown(
            "Results will appear here after analysis.",
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
        
        # Transactions display
        self.transactions_box = self._create_transactions_box()
        
    def _create_transactions_box(self):
        """Create the transactions display box."""
        return pn.Column(
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

    def _setup_callbacks(self):
        """Set up any necessary callbacks."""
        pass  # No callbacks needed for now

    def update_view(self, results, computation_time, graph_manager,algorithm):
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

            # Create and update visualizations
            self._update_visualizations(
                graph_manager.graph,
                original_edge_flows,
                simplified_edge_flows
            )

        except Exception as e:
            print(f"Error updating visualizations: {str(e)}")
            self._show_error(str(e))

    def _update_visualizations(self, graph, original_edge_flows, simplified_edge_flows):
        """Update both graph visualizations."""
        try:
            # Create full graph visualization
            full_graph = self.src_visualization.create_path_graph(
                graph,
                original_edge_flows,
                "Full Flow Paths"
            )
            if full_graph:
                self.full_graph_pane.object = full_graph

            # Create simplified graph visualization
            simplified_graph = self.src_visualization.create_aggregated_graph(
                simplified_edge_flows,
                "Simplified Transfers Graph"
            )
            if simplified_graph:
                self.simplified_graph_pane.object = simplified_graph

        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            self._show_visualization_error(str(e))

    def _show_visualization_error(self, error_msg):
        """Show error message in visualization panes."""
        error_img = self._create_error_image(error_msg)
        if error_img:
            self.full_graph_pane.object = error_img
            self.simplified_graph_pane.object = error_img

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
        """Update the transactions view."""
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
            <div style="height:300px; width:600px; overflow-y:scroll; 
                       border:1px solid #ddd; padding:10px; 
                       background-color: white; border-radius: 5px;">
                <div id="transactions-content">
                    <h3 style="margin-top: 0;">Transfers</h3>
                    {transactions}
                </div>
            </div>
        """
        self.transactions_box.visible = True

    def _update_graph_visualizations(self, graph, simplified_edge_flows, original_edge_flows):
        """Update both graph visualizations."""
        try:
            # Full graph visualization
            full_graph_buf = self.src_visualization.create_path_graph(
                graph,
                original_edge_flows,
                "Full Flow Paths"
            )
            if full_graph_buf:
                self.full_graph_pane.object = full_graph_buf
            
            # Simplified graph visualization
            simplified_graph_buf = self.src_visualization.create_aggregated_graph(
                simplified_edge_flows,
                "Simplified Transfers Graph"
            )
            if simplified_graph_buf:
                self.simplified_graph_pane.object = simplified_graph_buf
                
        except Exception as e:
            error_message = f"Error updating visualizations: {str(e)}"
            self.full_graph_pane.object = self._create_error_image(error_message)
            self.simplified_graph_pane.object = self._create_error_image(error_message)

    def _create_error_image(self, error_message: str) -> bytes:
        """Create an error image with the given message."""
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, error_message,
                horizontalalignment='center',
                verticalalignment='center',
                wrap=True,
                fontsize=12,
                color='red')
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf

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
            <div style="height:300px; width:600px; overflow-y:scroll; 
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
        
        # Create empty state images for graph panes
        empty_state = self._create_error_image("No graph data available")
        self.full_graph_pane.object = empty_state
        self.simplified_graph_pane.object = empty_state

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
            <div style="height:300px; width:600px; overflow-y:scroll; 
                       border:1px solid #ddd; padding:10px; 
                       background-color: #fff0f0; border-radius: 5px;">
                <div id="transactions-content" style="text-align: center; 
                                                    padding-top: 100px;">
                    <h3 style="color: red;">Error Processing Transactions</h3>
                    <p>Please try again or contact support.</p>
                </div>
            </div>
        """
        
        error_image = self._create_error_image(error_message)
        self.full_graph_pane.object = error_image
        self.simplified_graph_pane.object = error_image

    def view(self):
        """Return the component's view."""
        return pn.Column(
            pn.pane.Markdown("## Network Flow Analysis"),
            self.stats_pane,
            self.transactions_box,
            pn.pane.Markdown("## Flow Paths"),
            pn.Tabs(
                ("Full Graph", self.full_graph_pane),
                ("Aggregated Graph", self.simplified_graph_pane)
            ),
            width=800,
            sizing_mode='stretch_width'
        )
    
    @staticmethod
    def custom_flow_layout(G, source, sink, horizontal_spacing=10, vertical_spacing=2):
        """Create custom layout for flow network visualization."""
        def bfs_levels(G, source):
            levels = {source: 0}
            visited = {source}
            current_level = [source]
            level = 1
            
            while current_level:
                next_level = []
                for node in current_level:
                    for neighbor in G.neighbors(node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            levels[neighbor] = level
                            next_level.append(neighbor)
                current_level = next_level
                level += 1
                
            # Handle any remaining nodes (disconnected from source)
            remaining_nodes = set(G.nodes()) - visited
            if remaining_nodes:
                max_level = max(levels.values(), default=0)
                for node in remaining_nodes:
                    if node == sink:
                        levels[node] = max_level + 1
                    else:
                        levels[node] = max_level // 2  # Place disconnected nodes in the middle
                        
            return levels

        try:
            pos = {}
            levels = bfs_levels(G, source)
            max_level = max(levels.values())

            # Position source and sink
            pos[source] = (0, 0)
            pos[sink] = (horizontal_spacing * max_level, 0)

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
                    # Center nodes vertically and space them evenly
                    y = (i - (total_nodes - 1) / 2) * vertical_spacing
                    pos[node] = (x, y)

            # Verify all nodes have positions
            if not all(node in pos for node in G.nodes()):
                missing_nodes = [node for node in G.nodes() if node not in pos]
                print(f"Warning: Missing positions for nodes: {missing_nodes}")
                # Assign default positions to any missing nodes
                x_max = max(x for x, y in pos.values())
                for i, node in enumerate(missing_nodes):
                    pos[node] = (x_max/2, (i + 1) * vertical_spacing)

            return pos

        except Exception as e:
            print(f"Error in custom_flow_layout: {str(e)}")
            # Fallback to a simple circular layout if something goes wrong
            return nx.circular_layout(G)