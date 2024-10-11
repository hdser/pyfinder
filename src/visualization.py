import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Dict, Optional
import os
from collections import defaultdict

class Visualization:
    @staticmethod
    def add_id_address_mapping(fig, ax, id_to_address: Dict[str, str], nodes: List[str], edge_data):
        # Prepare the text for the mapping
        mapping_text = "ID to Address Mapping:\n\n"
        added_ids = set()

        for node in nodes:
            if '_' not in node and node not in added_ids:  # Exclude intermediate nodes
                address = id_to_address.get(node, "Not found")
                mapping_text += f"ID {node}: {address[:6]}...{address[-4:]}\n"
                added_ids.add(node)

        # Add token IDs that are present in the edge_data and not already in the node list
        for edge in edge_data:
            if len(edge) == 4:
                u, v, key, data = edge  # For MultiDiGraph edges with keys
            elif len(edge) == 3:
                u, v, data = edge
            elif len(edge) == 2:
                (u, v), data = edge
            else:
                continue  # Skip this edge if the data format is unexpected

            # Ensure data is a dictionary before accessing 'label' or 'token'
            if isinstance(data, dict):
                token = data.get('label') or data.get('token')
                if token and token not in added_ids:
                    address = id_to_address.get(token, "Not found")
                    mapping_text += f"ID {token}: {address[:6]}...{address[-4:]}\n"
                    added_ids.add(token)
            else:
                continue  # Skip if data is not a dictionary

        # Add the text to the axes
        ax.text(0, 1, mapping_text, verticalalignment='top', fontsize=8, fontfamily='monospace')
        ax.axis('off')

    @staticmethod
    def plot_full_graph(g: nx.DiGraph, id_to_address: Dict[str, str], filename: str = 'full_graph.png'):
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
        
        # Step 4: Scale positions to desired range
        scale = 100
        pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}

        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax_graph = fig.add_subplot(gs[0])
        ax_text = fig.add_subplot(gs[1])
        
        noncross_nodes = [node for node in g.nodes() if '_' not in node]
        nx.draw_networkx_nodes(g, pos, ax=ax_graph, nodelist=noncross_nodes, node_color='lightblue', node_shape='o', node_size=20)
        
        cross_nodes = [node for node in g.nodes() if '_' in node]
        nx.draw_networkx_nodes(g, pos, ax=ax_graph, nodelist=cross_nodes, node_color='red', node_shape='P', node_size=10)
        
        nx.draw_networkx_edges(g, pos, ax=ax_graph, edge_color='gray', arrows=True)
        
        # Use unique IDs for node labels
        labels = {node: node for node in g.nodes()}
        nx.draw_networkx_labels(g, pos, labels, ax=ax_graph, font_size=8)
        
        edge_labels = nx.get_edge_attributes(g, 'label')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, ax=ax_graph)

        ax_graph.set_title("Full Graph", fontsize=16)
        ax_graph.axis('off')

        #Visualization.add_id_address_mapping(fig, ax_text, id_to_address, g.nodes(), g.edges(data=True))

        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Full graph saved to {filename}")

    @staticmethod
    def custom_flow_layout(G, source, sink, horizontal_spacing=10, vertical_spacing=2):
        def bfs_levels(G, source):
            levels = {source: 0}
            queue = [(source, 0)]
            while queue:
                node, level = queue.pop(0)
                for neighbor in G.neighbors(node):
                    if neighbor not in levels:
                        levels[neighbor] = level + 1
                        queue.append((neighbor, level + 1))
                if len(levels) == len(G):
                    break
            return levels

        pos = {}
        levels = bfs_levels(G, source)
        max_level = max(levels.values())

        # Position source and sink
        pos[source] = (0, 0)
        pos[sink] = (horizontal_spacing * (max_level + 1), 0)

        # Group nodes by level
        nodes_by_level = {}
        for node, level in levels.items():
            if node not in (source, sink):
                nodes_by_level.setdefault(level, []).append(node)

        # Position nodes at each level
        for level, nodes in nodes_by_level.items():
            x = level * horizontal_spacing
            for i, node in enumerate(nodes):
                y = (i - (len(nodes) - 1) / 2) * vertical_spacing
                pos[node] = (x, y)

        return pos

    @staticmethod
    def plot_flow_paths(g: nx.DiGraph, paths: List[Tuple[List[str], List[str], float]],
                        edge_flows: Dict[Tuple[str, str], List[Dict[str, float]]],
                        id_to_address: Dict[str, str], filename: str = 'flow_paths.png'):
        if not paths:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No flow paths found", horizontalalignment='center', verticalalignment='center', fontsize=16)
            plt.axis('off')
            plt.savefig(filename, dpi=600, bbox_inches='tight')
            plt.close()
            print(f"No flow paths graph saved to {filename}")
            return

        flow_graph = nx.MultiDiGraph()

        for (u, v), token_flows in edge_flows.items():
            for token, flow in token_flows.items():
                flow_graph.add_edge(u, v, flow=flow, token=token)

        source = paths[0][0][0]
        sink = paths[0][0][-1]

        pos = Visualization.custom_flow_layout(flow_graph, source, sink)
        fig, ax = plt.subplots(figsize=(20, 10))

        nx.draw_networkx_nodes(flow_graph, pos, ax=ax, node_color='lightblue', node_shape='o', node_size=500)
        nx.draw_networkx_labels(flow_graph, pos, font_size=8, font_weight='bold', ax=ax)

        edges_between_nodes = defaultdict(list)
        for u, v, k in flow_graph.edges(keys=True):
            edges_between_nodes[(u, v)].append(k)

        for (u, v), keys in edges_between_nodes.items():
            num_edges = len(keys)
            if num_edges == 1:
                rad_list = [0.0]  # No curvature needed for a single edge
            else:
                # Assign curvature values ranging from -0.3 to 0.3
                rad_list = np.linspace(-0.3, 0.3, num_edges)
            
            for k, rad in zip(keys, rad_list):
                edge_data = flow_graph[u][v][k]
                label = f"Flow: {edge_data['flow']:.2f}\nToken: {edge_data['token']}"

                x1, y1 = pos[u]
                x2, y2 = pos[v]

                # Create a curved arrow between the nodes
                arrow = mpatches.FancyArrowPatch(
                    (x1, y1), (x2, y2),
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle='-|>',
                    mutation_scale=20,
                    color='gray',
                    linewidth=1,
                    zorder=1  # Ensure edges are drawn below nodes
                )
                ax.add_patch(arrow)

                # Calculate label position along the edge
                dx = x2 - x1
                dy = y2 - y1
                angle = np.arctan2(dy, dx)
                offset = np.array([-np.sin(angle), np.cos(angle)]) * rad * 0.5
                midpoint = np.array([(x1 + x2) / 2, (y1 + y2) / 2]) + offset

                # Add the label at the calculated position
                ax.text(midpoint[0], midpoint[1], label, fontsize=6, ha='center', va='center', 
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7), zorder=2)

        ax.set_title("Simplified Flow Paths", fontsize=16)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Simplified flow paths graph saved to {filename}")

    @staticmethod
    def plot_full_flow_paths(g: nx.DiGraph, edge_flows: Dict[Tuple[str, str], float], 
                             id_to_address: Dict[str, str], filename: str):
        flow_graph = nx.DiGraph()
        for (u, v), flow in edge_flows.items():
            if flow > 0:
                if g.has_edge(u, v):
                    edge_data = g.get_edge_data(u, v)
                    token = edge_data.get('label') if isinstance(edge_data, dict) else None
                else:
                    print(f"Warning: Edge ({u}, {v}) not found in graph g.")
                    token = None
                flow_graph.add_edge(u, v, flow=flow, token=token)

        if not flow_graph.nodes():
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No flow paths found", horizontalalignment='center', verticalalignment='center', fontsize=16)
            plt.axis('off')
            plt.savefig(filename, dpi=600, bbox_inches='tight')
            plt.close()
            print(f"No flow paths graph saved to {filename}")
            return

        sources = [node for node in flow_graph.nodes() if flow_graph.in_degree(node) == 0]
        sinks = [node for node in flow_graph.nodes() if flow_graph.out_degree(node) == 0]

        if not sources or not sinks:
            print("Error: No source or sink found in the flow graph.")
            return

        source = sources[0]
        sink = sinks[0]

        pos = Visualization.custom_flow_layout(flow_graph, source, sink, horizontal_spacing=10, vertical_spacing=4)
        #fig, (ax_graph, ax_text) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
        fig, (ax_graph) = plt.subplots(1, 1, figsize=(20, 10))
        
        noncross_nodes = [node for node in flow_graph.nodes() if '_' not in node]
        nx.draw_networkx_nodes(flow_graph, pos, ax=ax_graph, nodelist=noncross_nodes, node_color='lightblue', node_shape='o', node_size=300)
        
        cross_nodes = [node for node in flow_graph.nodes() if '_' in node]
        nx.draw_networkx_nodes(flow_graph, pos, ax=ax_graph, nodelist=cross_nodes, node_color='red', node_shape='P', node_size=200)
        
        nx.draw_networkx_edges(flow_graph, pos, ax=ax_graph, edge_color='gray', arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")
        
        labels = {node: node for node in flow_graph.nodes()}
        nx.draw_networkx_labels(flow_graph, pos, labels, ax=ax_graph, font_size=8, font_weight='bold')
        
        edge_labels = {}
        for u, v, data in flow_graph.edges(data=True):
            label = f"Flow: {data.get('flow', '')}"
            if 'token' in data:
                label += f"\nID: {data['token']}"
            edge_labels[(u, v)] = label.strip()
        
        nx.draw_networkx_edge_labels(flow_graph, pos, edge_labels=edge_labels, ax=ax_graph, font_size=6)

        ax_graph.set_title("Full Flow Paths (including 'auxiliar' nodes)", fontsize=16)
        ax_graph.axis('off')

        #Visualization.add_id_address_mapping(fig, ax_text, id_to_address, flow_graph.nodes(), flow_graph.edges(data=True))

        plt.tight_layout()
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Full flow paths graph saved to {filename}")

    @staticmethod
    def ensure_output_directory(directory: str):
        """Ensure the output directory exists."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created output directory: {directory}")
