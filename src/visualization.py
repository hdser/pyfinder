import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import os

class Visualization:
    @staticmethod
    def plot_full_graph(g: nx.DiGraph, filename: str = 'full_graph.png'):
        # Step 1: Use Fruchterman-Reingold layout for initial positioning
        pos = nx.spring_layout(g, k=.5, iterations=50)
        
        # Step 2: Apply logarithmic scaling to spread out nodes
        pos = {node: (np.sign(x) * np.log(1 + abs(x)), np.sign(y) * np.log(1 + abs(y))) 
            for node, (x, y) in pos.items()}
        
        # Step 3: Normalize positions
        pos_array = np.array(list(pos.values()))
        min_pos, max_pos = pos_array.min(), pos_array.max()
        pos = {node: ((x - min_pos) / (max_pos - min_pos), (y - min_pos) / (max_pos - min_pos)) 
            for node, (x, y) in pos.items()}
        
        # Step 4: Scale positions to desired range (e.g., 0 to 2000)
        scale = 100
        pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}

        plt.figure(figsize=(12, 8))
        
        noncross_nodes = [node for node in g.nodes() if '_' not in node]
        nx.draw_networkx_nodes(g, pos, nodelist=noncross_nodes, node_color='lightblue', node_shape='o', node_size=20)
        
        cross_nodes = [node for node in g.nodes() if '_' in node]
        nx.draw_networkx_nodes(g, pos, nodelist=cross_nodes, node_color='red', node_shape='P', node_size=10)
        
        nx.draw_networkx_edges(g, pos, edge_color='gray', arrows=True)
        nx.draw_networkx_labels(g, pos)
        
        edge_labels = nx.get_edge_attributes(g, 'label')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)

        plt.axis('off')
        plt.tight_layout()
        
        # Save the plot to a file
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to free up memory
        print(f"Full graph saved to {filename}")

    @staticmethod
    def custom_flow_layout(G, source, sink):
        def bfs_levels(G, source):
            levels = {source: 0}
            queue = [(source, 0)]
            while queue:
                node, level = queue.pop(0)
                for neighbor in G.neighbors(node):
                    if neighbor not in levels:
                        levels[neighbor] = level + 1
                        queue.append((neighbor, level + 1))
            return levels

        pos = {}
        levels = bfs_levels(G, source)
        max_level = max(levels.values())

        # Position source and sink
        pos[source] = (0, 0)
        pos[sink] = (1, 0)

        # Group nodes by level
        nodes_by_level = {}
        for node, level in levels.items():
            if node not in (source, sink):
                nodes_by_level.setdefault(level, []).append(node)

        # Position nodes at each level
        for level, nodes in nodes_by_level.items():
            x = level / (max_level + 1)  # Distribute levels evenly
            for i, node in enumerate(nodes):
                y = (i - (len(nodes) - 1) / 2) / max(len(nodes), 1)  # Center vertically
                pos[node] = (x, y)

        return pos

    @staticmethod
    def plot_flow_paths(g: nx.DiGraph, paths: List[Tuple[List[str], List[str], float]], edge_flows: Dict[Tuple[str, str], List[Dict[str, float]]], filename: str = 'flow_paths.png'):
        # Create a new graph with only the nodes and edges in the simplified paths
        subgraph = nx.MultiDiGraph()
        for path, _, _ in paths:
            nx.add_path(subgraph, path)

        source = paths[0][0][0]  # Assume the first node of the first path is the source
        sink = paths[0][0][-1]  # Assume the last node of the first path is the sink

        pos = Visualization.custom_flow_layout(subgraph, source, sink)
        plt.figure(figsize=(12, 8))
        
        nx.draw_networkx_nodes(subgraph, pos, node_color='lightblue', node_size=300)
        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight='bold')
        
        edge_labels = {}
        for (u, v), data_list in edge_flows.items():
            if subgraph.has_edge(u, v):
                label = "\n".join([f"Flow: {data['flow']}\nToken: {data['token']}" for data in data_list])
                edge_labels[(u, v)] = label
        
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=6)

        plt.title("Simplified Flow Paths", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Simplified flow paths graph saved to {filename}")

    @staticmethod
    def plot_full_flow_paths(g: nx.DiGraph, edge_flows: Dict[Tuple[str, str], float], filename: str):
        flow_graph = nx.DiGraph()
        for (u, v), flow in edge_flows.items():
            flow_graph.add_edge(u, v, flow=flow, token=g[u][v]['label'])

        # Identify source and sink
        source = [node for node in flow_graph.nodes() if flow_graph.in_degree(node) == 0][0]
        sink = [node for node in flow_graph.nodes() if flow_graph.out_degree(node) == 0][0]

        pos = Visualization.custom_flow_layout(flow_graph, source, sink)
        plt.figure(figsize=(12, 8))
        
        noncross_nodes = [node for node in flow_graph.nodes() if '_' not in node]
        nx.draw_networkx_nodes(flow_graph, pos, nodelist=noncross_nodes, node_color='lightblue', node_shape='o', node_size=50)
        
        cross_nodes = [node for node in flow_graph.nodes() if '_' in node]
        nx.draw_networkx_nodes(flow_graph, pos, nodelist=cross_nodes, node_color='red', node_shape='P', node_size=50)
        
        #nx.draw_networkx_nodes(flow_graph, pos, node_color='lightblue', node_size=300)
        nx.draw_networkx_edges(flow_graph, pos, edge_color='gray', arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_labels(flow_graph, pos, font_size=8, font_weight='bold')
        
        edge_labels = {(u, v): f"Flow: {flow}\nToken: {flow_graph[u][v]['token']}" 
                       for (u, v), flow in edge_flows.items()}
        
        nx.draw_networkx_edge_labels(flow_graph, pos, edge_labels=edge_labels, font_size=6)

        plt.title("Full Flow Paths (including '_' nodes)", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Full flow paths graph saved to {filename}")

    @staticmethod
    def ensure_output_directory(directory: str):
        """Ensure the output directory exists."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created output directory: {directory}")