# src/visualization.py
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Dict, Union
import os
from collections import defaultdict
from .graph import BaseGraph

class Visualization:
    """Handle visualization of graph and flow analysis results."""

    @staticmethod
    def add_id_address_mapping(fig, ax, id_to_address: Dict[str, str], nodes: List[str], edge_data):
        """Add mapping between IDs and addresses to the visualization."""
        mapping_text = "ID to Address Mapping:\n\n"
        added_ids = set()

        for node in nodes:
            if '_' not in node and node not in added_ids:
                address = id_to_address.get(node, "Not found")
                mapping_text += f"ID {node}: {address[:6]}...{address[-4:]}\n"
                added_ids.add(node)

        for edge in edge_data:
            if isinstance(edge, tuple) and len(edge) >= 2:
                u, v = edge[:2]
                data = edge[2] if len(edge) > 2 else {}
            else:
                continue

            if isinstance(data, dict):
                token = data.get('label') or data.get('token')
                if token and token not in added_ids:
                    address = id_to_address.get(token, "Not found")
                    mapping_text += f"ID {token}: {address[:6]}...{address[-4:]}\n"
                    added_ids.add(token)

        ax.text(0, 1, mapping_text, verticalalignment='top', fontsize=8, fontfamily='monospace')
        ax.axis('off')

    @staticmethod
    def custom_flow_layout(G, source, sink, horizontal_spacing=10, vertical_spacing=2):
        """Create a custom hierarchical layout for the flow graph."""
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

        pos[source] = (0, 0)
        pos[sink] = (horizontal_spacing * (max_level + 1), 0)

        nodes_by_level = defaultdict(list)
        for node, level in levels.items():
            if node not in (source, sink):
                nodes_by_level[level].append(node)

        for level, nodes in nodes_by_level.items():
            x = level * horizontal_spacing
            for i, node in enumerate(nodes):
                y = (i - (len(nodes) - 1) / 2) * vertical_spacing
                pos[node] = (x, y)

        return pos

    def plot_flow_paths(self, graph: BaseGraph, simplified_paths: List, 
                       simplified_edge_flows: Dict, id_to_address: Dict[str, str], 
                       filename: str = 'flow_paths.png'):
        """Plot simplified flow paths."""
        if not simplified_paths:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No flow paths found", horizontalalignment='center', 
                    verticalalignment='center', fontsize=16)
            plt.axis('off')
            plt.savefig(filename, dpi=600, bbox_inches='tight')
            plt.close()
            print(f"No flow paths graph saved to {filename}")
            return

        flow_graph = nx.MultiDiGraph()

        for (u, v), token_flows in simplified_edge_flows.items():
            for token, flow in token_flows.items():
                flow_graph.add_edge(u, v, flow=flow, token=token)

        source = simplified_paths[0][0][0]
        sink = simplified_paths[0][0][-1]

        pos = self.custom_flow_layout(flow_graph, source, sink)
        fig, ax = plt.subplots(figsize=(20, 10))

        nx.draw_networkx_nodes(flow_graph, pos, ax=ax, node_color='lightblue', 
                             node_shape='o', node_size=500)
        nx.draw_networkx_labels(flow_graph, pos, font_size=8, font_weight='bold', ax=ax)

        edges_between_nodes = defaultdict(list)
        for u, v, k in flow_graph.edges(keys=True):
            edges_between_nodes[(u, v)].append(k)

        for (u, v), keys in edges_between_nodes.items():
            num_edges = len(keys)
            rad_list = np.linspace(-0.3, 0.3, num_edges) if num_edges > 1 else [0.0]
            
            for k, rad in zip(keys, rad_list):
                edge_data = flow_graph[u][v][k]
                label = f"Flow: {edge_data['flow']}\nToken: {edge_data['token']}"

                x1, y1 = pos[u]
                x2, y2 = pos[v]

                arrow = mpatches.FancyArrowPatch(
                    (x1, y1), (x2, y2),
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle='-|>',
                    mutation_scale=20,
                    color='gray',
                    linewidth=1,
                    zorder=1
                )
                ax.add_patch(arrow)

                dx = x2 - x1
                dy = y2 - y1
                angle = np.arctan2(dy, dx)
                offset = np.array([-np.sin(angle), np.cos(angle)]) * rad * 0.5
                midpoint = np.array([(x1 + x2) / 2, (y1 + y2) / 2]) + offset

                ax.text(midpoint[0], midpoint[1], label, fontsize=6, ha='center', va='center', 
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7), zorder=2)

        ax.set_title("Simplified Flow Paths", fontsize=16)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Simplified flow paths graph saved to {filename}")

    def plot_full_flow_paths(self, graph: BaseGraph, edge_flows: Dict,
                           id_to_address: Dict[str, str], filename: str):
        """Plot full flow paths including intermediate nodes."""
        flow_graph = nx.DiGraph()
        for (u, v), flow in edge_flows.items():
            if flow > 0:
                edge_data = {}
                if graph.has_edge(u, v):
                    edge_data = graph.get_edge_data(u, v)
                    token = edge_data.get('label')
                else:
                    print(f"Warning: Edge ({u}, {v}) not found in graph.")
                    token = None
                flow_graph.add_edge(u, v, flow=flow, token=token)

        if not flow_graph.nodes():
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No flow paths found", horizontalalignment='center', 
                    verticalalignment='center', fontsize=16)
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

        pos = self.custom_flow_layout(flow_graph, source, sink, horizontal_spacing=10, vertical_spacing=4)
        fig, ax_graph = plt.subplots(1, 1, figsize=(20, 10))
        
        noncross_nodes = [node for node in flow_graph.nodes() if '_' not in node]
        nx.draw_networkx_nodes(flow_graph, pos, ax=ax_graph, nodelist=noncross_nodes, 
                             node_color='lightblue', node_shape='o', node_size=300)
        
        cross_nodes = [node for node in flow_graph.nodes() if '_' in node]
        nx.draw_networkx_nodes(flow_graph, pos, ax=ax_graph, nodelist=cross_nodes, 
                             node_color='red', node_shape='P', node_size=200)
        
        nx.draw_networkx_edges(flow_graph, pos, ax=ax_graph, edge_color='gray', 
                             arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")
        
        labels = {node: node for node in flow_graph.nodes()}
        nx.draw_networkx_labels(flow_graph, pos, labels, ax=ax_graph, 
                              font_size=8, font_weight='bold')
        
        edge_labels = {}
        for u, v, data in flow_graph.edges(data=True):
            label = f"Flow: {data.get('flow', '')}"
            if 'token' in data:
                label += f"\nID: {data['token']}"
            edge_labels[(u, v)] = label.strip()
        
        nx.draw_networkx_edge_labels(flow_graph, pos, edge_labels=edge_labels, 
                                   ax=ax_graph, font_size=6)

        ax_graph.set_title("Full Flow Paths (including 'auxiliary' nodes)", fontsize=16)
        ax_graph.axis('off')

        plt.tight_layout()
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Full flow paths graph saved to {filename}")

    @staticmethod
    def ensure_output_directory(directory: str):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created output directory: {directory}")