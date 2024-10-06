import panel as pn
import param
import networkx as nx
import matplotlib.pyplot as plt
import io
from src.graph_manager import GraphManager
from src.visualization import Visualization
from decimal import Decimal
from networkx.algorithms.flow import (
    edmonds_karp,
    preflow_push,
    shortest_augmenting_path,
    boykov_kolmogorov,
    dinitz,
)

pn.extension()

class NetworkFlowDashboard(param.Parameterized):
    source = param.String(default="9")
    sink = param.String(default="318")
    requested_flow = param.String(default="")
    algorithm = param.ObjectSelector(default="Default (Preflow Push)", objects=[
        "Default (Preflow Push)",
        "Edmonds-Karp",
        "Shortest Augmenting Path",
        "Boykov-Kolmogorov",
        "Dinitz"
    ])
    
    def __init__(self, **params):
        super().__init__(**params)
        self.graph_manager = GraphManager('data/circles_public_V_CrcV2_TrustRelations.csv', 'data/circles_public_V_CrcncesByAccountAndToken.csv')
        self.visualization = Visualization()
        self.results = None
        self.stats_pane = pn.pane.Markdown("Results will appear here after analysis.")
        self.transactions_pane = pn.pane.Markdown("Simplified transactions will appear here after analysis.")
        self.full_graph_pane = pn.pane.PNG(None)
        self.simplified_graph_pane = pn.pane.PNG(None)
        self.transactions_box = pn.Column(
            pn.pane.Markdown("# Aggregated Transactions"),
            pn.pane.HTML("""
                <div style="height:300px; width:600px; overflow-y:scroll; border:1px solid #ddd; padding:10px;">
                    <div id="transactions-content"></div>
                </div>
            """),
            visible=False
        )

    def run_analysis(self, event):
        requested_flow = None if not self.requested_flow else self.requested_flow
        algorithm_func = self.get_algorithm_func()
        
        self.results = self.graph_manager.analyze_flow(self.source, self.sink, flow_func=algorithm_func, cutoff=requested_flow)
        self.update_results_view()

    def get_algorithm_func(self):
        algorithm_map = {
            "Default (Preflow Push)": preflow_push,
            "Edmonds-Karp": edmonds_karp,
            "Shortest Augmenting Path": shortest_augmenting_path,
            "Boykov-Kolmogorov": boykov_kolmogorov,
            "Dinitz": dinitz
        }
        return algorithm_map[self.algorithm]

    def update_results_view(self):
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

        flow_value, simplified_paths, simplified_edge_flows, original_edge_flows, aggregated_flows = self.results

        stats = f"""
        # Results
        - Algorithm: {self.algorithm}
        - Requested Flow: {self.requested_flow if self.requested_flow else 'Max Flow'}
        - Achieved Flow Value: {flow_value}
        - Number of Simplified Paths: {len(simplified_paths)}
        - Number of Original Edges: {len(original_edge_flows)}
        - Number of Aggregated Transfers: {len(aggregated_flows)}
        """

        self.stats_pane.object = stats
        
        # Format aggregated transactions
        transactions = ""
        for (u, v, token), flow in aggregated_flows.items():
            transactions += f"<li>{u} --> {v} (Flow: {flow}, Token: {token})</li>"
        transactions += "</ul>"
        
        self.transactions_box[-1].object = f"""
            <div style="height:300px; width:600px; overflow-y:scroll; border:1px solid #ddd; padding:10px;">
                <div id="transactions-content">{transactions}</div>
            </div>
        """
        self.transactions_box.visible = True
        
        self.full_graph_pane.object = self.create_path_graph(self.graph_manager.graph.g_nx, original_edge_flows, "Full Graph")
        self.simplified_graph_pane.object = self.create_aggregated_graph(aggregated_flows, "Aggregated Transfers Graph")

    def create_path_graph(self, G, edge_flows, title):
        plt.figure(figsize=(10, 7))
        
        # Create a subgraph with only the nodes and edges in the flow
        subgraph = nx.DiGraph()
        for (u, v) in edge_flows.keys():
            subgraph.add_edge(u, v)

        # Identify source and sink
        source = next(node for node in subgraph.nodes() if subgraph.in_degree(node) == 0)
        sink = next(node for node in subgraph.nodes() if subgraph.out_degree(node) == 0)

        pos = self.visualization.custom_flow_layout(subgraph, source, sink)
        
        # Draw nodes
        noncross_nodes = [node for node in subgraph.nodes() if '_' not in node]
        nx.draw_networkx_nodes(subgraph, pos, nodelist=noncross_nodes, node_color='lightblue', node_shape='o', node_size=500)
        
        cross_nodes = [node for node in subgraph.nodes() if '_' in node]
        nx.draw_networkx_nodes(subgraph, pos, nodelist=cross_nodes, node_color='red', node_shape='P', node_size=300)
        
        nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight='bold')

        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")

        # Prepare edge labels
        edge_labels = {}
        for (u, v), flow in edge_flows.items():
            label = f"Flow: {flow}\nToken: {G[u][v]['label']}"
            edge_labels[(u, v)] = label
        
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=6)

        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf

    def create_aggregated_graph(self, aggregated_flows, title):
        plt.figure(figsize=(10, 7))
        
        # Create a graph with aggregated flows
        G = nx.MultiDiGraph()
        for (u, v, token), flow in aggregated_flows.items():
            G.add_edge(u, v, flow=flow, token=token)

        # Identify source and sink
        source = next(node for node in G.nodes() if G.in_degree(node) == 0)
        sink = next(node for node in G.nodes() if G.out_degree(node) == 0)

        pos = self.visualization.custom_flow_layout(G, source, sink)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_shape='o', node_size=500)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")

        # Prepare edge labels
        edge_labels = {}
        for (u, v, k) in G.edges(keys=True):
            edge_data = G[u][v][k]
            label = f"Flow: {edge_data['flow']}\nToken: {edge_data['token']}"
            edge_labels[(u, v, k)] = label
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf

    def view(self):
        run_button = pn.widgets.Button(name="Run Analysis", button_type="primary")
        
        controls = pn.Column(
            pn.pane.Markdown("## Controls"),
            *[pn.Param(self.param[name], widgets={name: widget}) 
              for name, widget in {
                  'source': pn.widgets.TextInput,
                  'sink': pn.widgets.TextInput,
                  'requested_flow': pn.widgets.TextInput,
                  'algorithm': pn.widgets.Select,
              }.items()],
            run_button
        )
        
        run_button.on_click(self.run_analysis)

        graph_tabs = pn.Tabs(
            ("Full Graph", self.full_graph_pane),
            ("Aggregated Graph", self.simplified_graph_pane)
        )

        template = pn.template.MaterialTemplate(
            title="pyFinder Flow Analysis Dashboard",
            sidebar=[controls],
            main=[
                pn.Row(
                    pn.Column(
                        pn.pane.Markdown("## Network Flow Analysis"),
                        self.stats_pane,
                        self.transactions_box,
                        pn.pane.Markdown("## Flow Paths"),
                        graph_tabs
                    )
                )
            ],
        )
        return template

dashboard = NetworkFlowDashboard()
app = dashboard.view()

if __name__ == "__main__":
    app.show(port=5006)