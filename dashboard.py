import panel as pn
import param
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import io
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from src.graph_manager import GraphManager
from src.visualization import Visualization
from src.graph import NetworkXGraph, GraphToolGraph
from networkx.algorithms.flow import (
    edmonds_karp,
    preflow_push,
    shortest_augmenting_path,
    boykov_kolmogorov,
    dinitz,
)
from graph_tool.flow import (
    edmonds_karp_max_flow as gt_edmonds_karp,
    push_relabel_max_flow as gt_push_relabel,
    boykov_kolmogorov_max_flow as gt_boykov_kolmogorov,
)

pn.extension()

class NetworkFlowDashboard(param.Parameterized):
    # Hidden data source parameter (for internal use)
    data_source_type = param.ObjectSelector(default="CSV", objects=["CSV", "PostgreSQL (Manual)", "PostgreSQL (ENV)"])
    
    # CSV parameters
    csv_trusts_file = param.String(default="")
    csv_balances_file = param.String(default="")
    
    # PostgreSQL parameters
    pg_host = param.String(default="localhost")
    pg_port = param.String(default="5432")
    pg_dbname = param.String(default="")
    pg_user = param.String(default="")
    pg_password = param.String(default="")
    pg_queries_dir = param.String(default="queries")
    
    # Analysis parameters
    source = param.String(default="0x3fb47823a7c66553fb6560b75966ef71f5ccf1d0")
    sink = param.String(default="0xe98f0672a8e31b408124f975749905f8003a2e04")
    requested_flow_mCRC = param.String(default="")
    algorithm = param.ObjectSelector(default="Default (Preflow Push)", objects=[
        "Default (Preflow Push)",
        "Edmonds-Karp",
        "Shortest Augmenting Path",
        "Boykov-Kolmogorov",
        "Dinitz"
    ])
    graph_library = param.ObjectSelector(default="NetworkX", objects=["NetworkX", "graph-tool"])
    
    def __init__(self, **params):
        super().__init__(**params)
        self.graph_manager = None
        self.visualization = Visualization()
        self.results = None
        
        # Initialize display components
        self.stats_pane = pn.pane.Markdown("Results will appear here after analysis.")
        self.transactions_pane = pn.pane.Markdown("Simplified transactions will appear here after analysis.")
        self.full_graph_pane = pn.pane.PNG(None)
        self.simplified_graph_pane = pn.pane.PNG(None)
        self.status_indicator = pn.pane.Markdown("Status: Not initialized", styles={'color': 'gray'})
        
        # File upload widgets
        self.trusts_file_input = pn.widgets.FileInput(
            name='Trust Relationships File (CSV)',
            accept='.csv',
            height=45,
            width=300
        )
        self.balances_file_input = pn.widgets.FileInput(
            name='Account Balances File (CSV)',
            accept='.csv',
            height=45,
            width=300
        )
        
        # Load configuration buttons (one per tab)
        self.csv_load_button = pn.widgets.Button(
            name="Load CSV Configuration",
            button_type="primary",
            width=200
        )
        self.pg_manual_load_button = pn.widgets.Button(
            name="Load PostgreSQL Configuration",
            button_type="primary",
            width=200
        )
        self.pg_env_load_button = pn.widgets.Button(
            name="Load Environment Configuration",
            button_type="primary",
            width=200
        )
        
        # Analysis panel (initially hidden)
        self.analysis_panel = pn.Column(visible=False)
        
        # Set up file upload watchers
        self.trusts_file_input.param.watch(self.handle_trusts_upload, 'value')
        self.balances_file_input.param.watch(self.handle_balances_upload, 'value')
        
        # Set up tab change watcher
        self.data_source_tabs = self.create_data_source_tabs()
        self.data_source_tabs.param.watch(self.handle_tab_change, 'active')
        
        # Set up load button watchers
        self.csv_load_button.on_click(lambda event: self.load_configuration("CSV"))
        self.pg_manual_load_button.on_click(lambda event: self.load_configuration("PostgreSQL (Manual)"))
        self.pg_env_load_button.on_click(lambda event: self.load_configuration("PostgreSQL (ENV)"))
        
        # Create transactions display
        self.transactions_box = pn.Column(
            pn.pane.Markdown("# Aggregated Transactions"),
            pn.pane.HTML("""
                <div style="height:300px; width:600px; overflow-y:scroll; border:1px solid #ddd; padding:10px;">
                    <div id="transactions-content"></div>
                </div>
            """),
            visible=False
        )

    def create_data_source_tabs(self):
        """Create and configure the data source tabs"""
        # CSV configuration tab
        csv_config = pn.Column(
            pn.pane.Markdown("### CSV File Upload"),
            pn.Column(
                self.trusts_file_input,
                pn.layout.Spacer(height=10),
                self.balances_file_input,
                sizing_mode='stretch_width'
            ),
            pn.layout.Spacer(height=20),
            pn.Row(
                self.csv_load_button,
                align='center'
            ),
            name="CSV"
        )

        # PostgreSQL Manual configuration tab
        pg_manual_config = pn.Column(
            pn.pane.Markdown("### PostgreSQL Connection Details"),
            pn.Param(self.param.pg_host, widgets={'pg_host': 
                pn.widgets.TextInput(name="Host")}),
            pn.Param(self.param.pg_port, widgets={'pg_port': 
                pn.widgets.TextInput(name="Port")}),
            pn.Param(self.param.pg_dbname, widgets={'pg_dbname': 
                pn.widgets.TextInput(name="Database Name")}),
            pn.Param(self.param.pg_user, widgets={'pg_user': 
                pn.widgets.TextInput(name="Username")}),
            pn.Param(self.param.pg_password, widgets={'pg_password': 
                pn.widgets.PasswordInput(name="Password")}),
            pn.Param(self.param.pg_queries_dir, widgets={'pg_queries_dir': 
                pn.widgets.TextInput(name="Queries Directory")}),
            pn.layout.Spacer(height=20),
            pn.Row(
                self.pg_manual_load_button,
                align='center'
            ),
            name="PostgreSQL (Manual)"
        )

        # PostgreSQL ENV configuration tab
        pg_env_config = pn.Column(
            pn.pane.Markdown("""
            ### Environment Variables Configuration
            
            Make sure the following environment variables are set:
            - POSTGRES_HOST
            - POSTGRES_PORT
            - POSTGRES_DB
            - POSTGRES_USER
            - POSTGRES_PASSWORD
            """),
            pn.layout.Spacer(height=20),
            pn.Row(
                self.pg_env_load_button,
                align='center'
            ),
            name="PostgreSQL (ENV)"
        )

        # Create tabs and set initial state
        tabs = pn.Tabs(
            csv_config, 
            pg_manual_config, 
            pg_env_config,
        )
        
        return tabs

    def handle_tab_change(self, event):
        """Handle tab change events"""
        # Map tab index to data source type
        tab_to_source = {
            0: "CSV",
            1: "PostgreSQL (Manual)",
            2: "PostgreSQL (ENV)"
        }
        
        # Update data source type
        self.data_source_type = tab_to_source[event.new]
        
        # Reset analysis panel visibility
        self.analysis_panel.visible = False
        
        # Update status
        self.status_indicator.object = f"Status: Selected {self.data_source_type} configuration"
        self.status_indicator.styles = {'color': 'gray'}
        
        # Check environment variables if ENV tab selected
        if event.new == 2:
            self.update_env_status()

    def load_configuration(self, source_type: str):
        """Handle configuration loading for the selected data source"""
        try:
            if source_type == "CSV":
                if not self.csv_trusts_file or not self.csv_balances_file:
                    raise ValueError("Please upload both trust relationships and account balances files")
            
            elif source_type == "PostgreSQL (Manual)":
                if not all([self.pg_host, self.pg_port, self.pg_dbname, self.pg_user, self.pg_password]):
                    raise ValueError("Please fill in all PostgreSQL connection details")
            
            elif source_type == "PostgreSQL (ENV)":
                self._verify_env_variables()
            
            # Update data source type
            self.data_source_type = source_type
            
            # Show analysis panel
            self.analysis_panel.visible = True
            
            # Update status
            self.status_indicator.object = f"Status: {source_type} configuration loaded successfully"
            self.status_indicator.styles = {'color': 'green'}
            
        except Exception as e:
            self.status_indicator.object = f"Status: Configuration Error - {str(e)}"
            self.status_indicator.styles = {'color': 'red'}
            self.analysis_panel.visible = False

    def _verify_env_variables(self):
        """Verify that all required environment variables are set"""
        load_dotenv()
        required_vars = ['POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def handle_trusts_upload(self, event):
        """Handle trust relationships file upload"""
        if event.new:
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            filename = os.path.join('uploads', 'trusts.csv')
            with open(filename, 'wb') as f:
                f.write(event.new)
            self.csv_trusts_file = filename
            self.status_indicator.object = "Status: Trust relationships file uploaded"
            self.status_indicator.styles = {'color': 'green'}

    def handle_balances_upload(self, event):
        """Handle account balances file upload"""
        if event.new:
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            filename = os.path.join('uploads', 'balances.csv')
            with open(filename, 'wb') as f:
                f.write(event.new)
            self.csv_balances_file = filename
            self.status_indicator.object = "Status: Account balances file uploaded"
            self.status_indicator.styles = {'color': 'green'}

    def update_env_status(self):
        """Check and update environment variables status"""
        try:
            self._verify_env_variables()
            status = """
            ### Environment Variables Configuration
            
            ✅ All required environment variables are set:
            - POSTGRES_HOST
            - POSTGRES_PORT
            - POSTGRES_DB
            - POSTGRES_USER
            - POSTGRES_PASSWORD
            
            Ready to load configuration!
            """
            status_color = 'green'
        except ValueError as e:
            status = f"""
            ### Environment Variables Configuration
            
            ⚠️ Configuration Error:
            {str(e)}
            
            Please set these variables in your environment or .env file.
            """
            status_color = 'red'
        
        # Update the environment tab content
        self.data_source_tabs[2][0].object = status
        self.data_source_tabs[2][0].styles = {'color': status_color}

    def _get_data_source(self):
        """Get configured data source based on selected type"""
        if self.data_source_type == "CSV":
            if not self.csv_trusts_file or not self.csv_balances_file:
                raise ValueError("Please load CSV configuration first")
            if not os.path.exists(self.csv_trusts_file) or not os.path.exists(self.csv_balances_file):
                raise FileNotFoundError("One or both CSV files not found")
            return (self.csv_trusts_file, self.csv_balances_file)
        
        elif self.data_source_type == "PostgreSQL (Manual)":
            db_config = {
                'host': self.pg_host,
                'port': self.pg_port,
                'dbname': self.pg_dbname,
                'user': self.pg_user,
                'password': self.pg_password
            }
            return (db_config, self.pg_queries_dir)
        
        else:  # PostgreSQL (ENV)
            load_dotenv()
            self._verify_env_variables()
            db_config = {
                'host': os.getenv('POSTGRES_HOST'),
                'port': os.getenv('POSTGRES_PORT'),
                'dbname': os.getenv('POSTGRES_DB'),
                'user': os.getenv('POSTGRES_USER'),
                'password': os.getenv('POSTGRES_PASSWORD')
            }
            return (db_config, self.pg_queries_dir)

    def update_algorithm_list(self, event):
        """Update algorithm list based on selected graph library"""
        self.param.algorithm.objects = self.get_algorithm_list()
        self.algorithm = self.param.algorithm.objects[0]

    def get_algorithm_list(self):
        """Get list of algorithms based on selected graph library"""
        if self.graph_library == 'NetworkX':
            return [
                "Default (Preflow Push)",
                "Edmonds-Karp",
                "Shortest Augmenting Path",
                "Boykov-Kolmogorov",
                "Dinitz"
            ]
        else:  # graph-tool
            return [
                "Default (Push-Relabel)",
                "Edmonds-Karp",
                "Boykov-Kolmogorov"
            ]

    def get_algorithm_func(self):
        """Get the algorithm function based on selected algorithm"""
        if self.graph_library == 'NetworkX':
            algorithm_map = {
                "Default (Preflow Push)": preflow_push,
                "Edmonds-Karp": edmonds_karp,
                "Shortest Augmenting Path": shortest_augmenting_path,
                "Boykov-Kolmogorov": boykov_kolmogorov,
                "Dinitz": dinitz
            }
        else:  # graph-tool
            algorithm_map = {
                "Default (Push-Relabel)": gt_push_relabel,
                "Edmonds-Karp": gt_edmonds_karp,
                "Boykov-Kolmogorov": gt_boykov_kolmogorov
            }
        return algorithm_map[self.algorithm]

    def view(self):
        """Create and return the dashboard view"""
        # Analysis configuration
        init_button = pn.widgets.Button(name="Initialize Graph", button_type="primary")
        run_button = pn.widgets.Button(name="Run Analysis", button_type="primary")
        
        analysis_config = pn.Column(
            pn.pane.Markdown("## Analysis Configuration"),
            pn.Param(self.param.graph_library, widgets={'graph_library': pn.widgets.Select}),
            *[pn.Param(self.param[name], widgets={name: widget}) 
              for name, widget in {
                  'source': pn.widgets.TextInput,
                  'sink': pn.widgets.TextInput,
                  'requested_flow_mCRC': pn.widgets.TextInput,
                  'algorithm': pn.widgets.Select,
              }.items()]
        )
        
        self.analysis_panel.extend([
            pn.layout.Divider(),
            analysis_config,
            pn.Row(init_button, margin=(10, 0)),
            pn.Row(run_button, margin=(10, 0)),
            self.status_indicator
        ])

        # Bind analysis buttons events
        init_button.on_click(self.initialize_graph_manager)
        run_button.on_click(self.run_analysis)

        # Create visualization tabs
        graph_tabs = pn.Tabs(
            ("Full Graph", self.full_graph_pane),
            ("Aggregated Graph", self.simplified_graph_pane)
        )

        # Data source panel (shows tabs immediately)
        data_source_panel = pn.Column(
            pn.pane.Markdown("## Data Source Configuration"),
            self.data_source_tabs,
        )

        # Combine all controls in sidebar
        controls = pn.Column(
            data_source_panel,
            self.analysis_panel,
            width=400
        )

        # Main content panel
        main_content = pn.Column(
            pn.pane.Markdown("## Network Flow Analysis"),
            self.stats_pane,
            self.transactions_box,
            pn.pane.Markdown("## Flow Paths"),
            graph_tabs,
            width=800
        )

        # Create template
        template = pn.template.MaterialTemplate(
            title="pyFinder Flow Analysis Dashboard",
            sidebar=[controls],
            main=[main_content],
            header_background="#007BFF",
        )
        
        return template

    def initialize_graph_manager(self, event):
        """Initialize the graph manager with selected configuration"""
        try:
            self.status_indicator.object = "Status: Initializing..."
            self.status_indicator.styles = {'color': 'blue'}
            
            data_source = self._get_data_source()
            graph_library = 'networkx' if self.graph_library == 'NetworkX' else 'graph_tool'
            
            self.graph_manager = GraphManager(data_source, graph_library)
            
            if isinstance(self.graph_manager.graph, NetworkXGraph):
                num_nodes = self.graph_manager.graph.g_nx.number_of_nodes()
                num_edges = self.graph_manager.graph.g_nx.number_of_edges()
            else:  # GraphToolGraph
                num_nodes = self.graph_manager.graph.g_gt.num_vertices()
                num_edges = self.graph_manager.graph.g_gt.num_edges()
            
            self.status_indicator.object = "Status: Initialized successfully"
            self.status_indicator.styles = {'color': 'green'}
            
            self.stats_pane.object = f"""
            # Graph Information
            - Number of nodes: {num_nodes}
            - Number of edges: {num_edges}
            - Data source: {self.data_source_type}
            - Graph library: {self.graph_library}
            """
            
        except Exception as e:
            self.status_indicator.object = f"Status: Error - {str(e)}"
            self.status_indicator.styles = {'color': 'red'}
            self.stats_pane.object = f"Error initializing graph: {str(e)}"



    def run_analysis(self, event):
        """Run the flow analysis"""
        if self.graph_manager is None:
            self.stats_pane.object = "Please initialize the graph manager first."
            return

        self.status_indicator.object = "Status: Running analysis..."
        self.status_indicator.styles = {'color': 'blue'}
        
        requested_flow_mCRC = None if not self.requested_flow_mCRC else self.requested_flow_mCRC
        algorithm_func = self.get_algorithm_func()
        
        start_time = time.time()
        try:
            self.results = self.graph_manager.analyze_flow(
                self.source, 
                self.sink, 
                flow_func=algorithm_func, 
                cutoff=requested_flow_mCRC
            )
            self.status_indicator.object = "Status: Analysis completed successfully"
            self.status_indicator.styles = {'color': 'green'}
        except Exception as e:
            self.status_indicator.object = f"Status: Error during analysis - {str(e)}"
            self.status_indicator.styles = {'color': 'red'}
            self.stats_pane.object = f"An error occurred during analysis: {str(e)}"
            return

        end_time = time.time()
        computation_time = end_time - start_time
        
        self.update_results_view(computation_time)

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

        # Update statistics
        stats = f"""
        # Results
        - Algorithm: {self.algorithm}
        - Requested Flow: {self.requested_flow_mCRC if self.requested_flow_mCRC else 'Max Flow'}
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
        
        # Update visualizations
        self.full_graph_pane.object = self.create_path_graph(self.graph_manager.graph, original_edge_flows, "Full Graph")
        self.simplified_graph_pane.object = self.create_aggregated_graph(simplified_edge_flows, "Simplified Transfers Graph")

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
            self.status_indicator.object = f"Error creating path graph: {str(e)}"
            self.status_indicator.styles = {'color': 'red'}
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
            self.status_indicator.object = f"Error creating aggregated graph: {str(e)}"
            self.status_indicator.styles = {'color': 'red'}
            return None


def create_dashboard():
    """Create and return a new dashboard instance"""
    dashboard = NetworkFlowDashboard()
    app = dashboard.view()
    return app






# Only create and show the dashboard when running directly
if __name__ == "__main__":
    app = create_dashboard()
    app.show(port=5006)
