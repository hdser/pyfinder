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

    def plot_flow_paths(self, graph: BaseGraph, paths: List[Tuple[List[str], List[str], int]], 
                   simplified_flows: Dict, id_to_address: Dict[str, str], 
                   filename: str,
                   source_address: str):
        """
        Plot simplified flow paths showing token transformations.
        """
        viz_graph = nx.DiGraph()
        
        # Find source node ID by matching the address
        address_to_id = {addr.lower(): id_ for id_, addr in id_to_address.items()}
        source_node = address_to_id.get(source_address.lower())
        
        if not source_node:
            raise ValueError(f"Source address {source_address} not found in address mapping")
                    
        print(f"Identified source node: {source_node} ({id_to_address.get(source_node, 'Unknown')})")
                    
        # Process simplified flows directly
        for (u, v), token_dict in simplified_flows.items():
            if u not in viz_graph:
                viz_graph.add_node(u)
            if v not in viz_graph:
                viz_graph.add_node(v)
                        
            # Add edge with token information
            for token, amount in token_dict.items():
                if not viz_graph.has_edge(u, v):
                    viz_graph.add_edge(u, v, tokens={}, total_flow=0)
                viz_graph[u][v]['tokens'][token] = amount
                viz_graph[u][v]['total_flow'] += amount
        
        # Create hierarchical layout with source at the top
        pos = self._create_hierarchical_layout(viz_graph, source_node)
        
        plt.figure(figsize=(15, 12))
        
        # Draw nodes
        nx.draw_networkx_nodes(
            viz_graph, pos,
            node_color=['red' if n == source_node else 'lightblue' for n in viz_graph.nodes()],
            node_size=[1500 if n == source_node else 1000 for n in viz_graph.nodes()],
            alpha=0.9
        )
        
        # Draw edges with curvature to minimize overlaps
        edge_styles = []
        for u, v in viz_graph.edges():
            if u == source_node:
                edge_styles.append(('lightcoral', 'dashed', 'arc3,rad=0.2'))
            elif v == source_node:
                edge_styles.append(('lightgreen', 'dashed', 'arc3,rad=0.2'))
            else:
                edge_styles.append(('gray', 'solid', 'arc3,rad=0.0'))
        
        for (u, v), (color, style, connectionstyle) in zip(viz_graph.edges(), edge_styles):
            nx.draw_networkx_edges(
                viz_graph, pos,
                edgelist=[(u, v)],
                width=1 + np.log1p(viz_graph[u][v]['total_flow']) * 0.5,
                edge_color=color,
                style=style,
                arrows=True,
                arrowsize=20,
                alpha=0.6,
                connectionstyle=connectionstyle
            )
        
        # Create node labels
        labels = {}
        for node in viz_graph.nodes():
            addr = id_to_address.get(str(node), str(node))
            if len(addr) > 10:
                labels[node] = f"{addr[:6]}...{addr[-4:]}"
            else:
                labels[node] = addr
        
        nx.draw_networkx_labels(
            viz_graph, pos,
            labels,
            font_size=10,
            font_weight='bold'
        )
        
        # Create edge labels showing token flows
        edge_labels = {}
        for u, v, data in viz_graph.edges(data=True):
            label_lines = []
            for token, amount in data['tokens'].items():
                label_lines.append(f"Token {token}")
                label_lines.append(f"{amount:,} mCRC")
            edge_labels[(u, v)] = '\n'.join(label_lines)
        
        nx.draw_networkx_edge_labels(
            viz_graph, pos,
            edge_labels,
            font_size=8,
            rotate=False
        )
        
        # Add title with proper source node
        source_addr = id_to_address.get(str(source_node), str(source_node))
        if len(source_addr) > 10:
            source_addr = f"{source_addr[:6]}...{source_addr[-4:]}"
        plt.title(f"Arbitrage Flow Paths\nSource: {source_addr}", fontsize=16, pad=20)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='lightcoral', linestyle='--', lw=2, label='Outgoing Flows (Start token)'),
            plt.Line2D([0], [0], color='lightgreen', linestyle='--', lw=2, label='Incoming Flows (End token)'),
            plt.Line2D([0], [0], color='gray', linestyle='-', lw=2, label='Intermediate Flows'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                    markersize=15, label='Source Account'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                    markersize=15, label='Intermediate Account')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


    def plot_flow_paths2(self, graph: BaseGraph, paths: List[Tuple[List[str], List[str], int]], 
                   simplified_flows: Dict, id_to_address: Dict[str, str], 
                   filename: str,
                   source_address: str):
        """
        Plot simplified flow paths showing token transformations.
        """
        viz_graph = nx.DiGraph()
        
        # Find source node ID by matching the address
        address_to_id = {addr.lower(): id_ for id_, addr in id_to_address.items()}
        source_node = address_to_id.get(source_address.lower())
        
        if not source_node:
            raise ValueError(f"Source address {source_address} not found in address mapping")
                    
        print(f"Identified source node: {source_node} ({id_to_address.get(source_node, 'Unknown')})")
                    
        # Process simplified flows directly
        for (u, v), token_dict in simplified_flows.items():
            if u not in viz_graph:
                viz_graph.add_node(u)
            if v not in viz_graph:
                viz_graph.add_node(v)
                        
            # Add edge with token information
            for token, amount in token_dict.items():
                if not viz_graph.has_edge(u, v):
                    viz_graph.add_edge(u, v, tokens={}, total_flow=0)
                viz_graph[u][v]['tokens'][token] = amount
                viz_graph[u][v]['total_flow'] += amount
                
        # Create layout with source above the circle
        pos = self._create_circular_layout(viz_graph, source_node)
        
        plt.figure(figsize=(15, 12))
        
        # Draw nodes
        nx.draw_networkx_nodes(
            viz_graph, pos,
            node_color=['red' if n == source_node else 'lightblue' for n in viz_graph.nodes()],
            node_size=[1500 if n == source_node else 1000 for n in viz_graph.nodes()],
            alpha=0.7
        )
        
        # Draw edges
        outgoing_edges = []
        incoming_edges = []
        intermediate_edges = []
        
        for u, v, data in viz_graph.edges(data=True):
            if u == source_node:
                outgoing_edges.append((u, v))
            elif v == source_node:
                incoming_edges.append((u, v))
            else:
                intermediate_edges.append((u, v))
        
        # Draw outgoing edges with curvature
        if outgoing_edges:
            nx.draw_networkx_edges(
                viz_graph, pos,
                edgelist=outgoing_edges,
                width=[1 + np.log1p(viz_graph[u][v]['total_flow']) * 0.5 for u, v in outgoing_edges],
                edge_color='lightcoral',
                style='dashed',
                arrows=True,
                arrowsize=20,
                alpha=0.6,
                connectionstyle='arc3, rad=-0.3'  # Adjust curvature as needed
            )
        
        # Draw incoming edges with curvature
        if incoming_edges:
            nx.draw_networkx_edges(
                viz_graph, pos,
                edgelist=incoming_edges,
                width=[1 + np.log1p(viz_graph[u][v]['total_flow']) * 0.5 for u, v in incoming_edges],
                edge_color='lightgreen',
                style='dashed',
                arrows=True,
                arrowsize=20,
                alpha=0.6,
                connectionstyle='arc3, rad=0.3'  # Adjust curvature as needed
            )
        
        # Draw intermediate edges without curvature
        if intermediate_edges:
            nx.draw_networkx_edges(
                viz_graph, pos,
                edgelist=intermediate_edges,
                width=[1 + np.log1p(viz_graph[u][v]['total_flow']) * 0.5 for u, v in intermediate_edges],
                edge_color='gray',
                style='solid',
                arrows=True,
                arrowsize=20,
                alpha=0.6
                # No curvature for intermediate edges
            )
        
        # Create node labels
        labels = {}
        for node in viz_graph.nodes():
            addr = id_to_address.get(str(node), str(node))
            if len(addr) > 10:
                labels[node] = f"{addr[:6]}...{addr[-4:]}"
            else:
                labels[node] = addr
        
        nx.draw_networkx_labels(
            viz_graph, pos,
            labels,
            font_size=10,
            font_weight='bold'
        )
        
        # Create edge labels showing token flows
        edge_labels = {}
        for u, v, data in viz_graph.edges(data=True):
            label_lines = []
            for token, amount in data['tokens'].items():
                label_lines.append(f"Token {token}")
                label_lines.append(f"{amount:,} mCRC")
            edge_labels[(u, v)] = '\n'.join(label_lines)
        
        nx.draw_networkx_edge_labels(
            viz_graph, pos,
            edge_labels,
            font_size=8,
            rotate=False
        )
        
        # Add title with proper source node
        source_addr = id_to_address.get(str(source_node), str(source_node))
        if len(source_addr) > 10:
            source_addr = f"{source_addr[:6]}...{source_addr[-4:]}"
        plt.title(f"Arbitrage Flow Paths\nSource: {source_addr}", fontsize=16, pad=20)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='lightcoral', linestyle='--', lw=2, label='Outgoing Flows (Start token)'),
            plt.Line2D([0], [0], color='lightgreen', linestyle='--', lw=2, label='Incoming Flows (End token)'),
            plt.Line2D([0], [0], color='gray', linestyle='-', lw=2, label='Intermediate Flows'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                    markersize=15, label='Source Account'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                    markersize=15, label='Intermediate Account')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


    def _create_circular_layout(self, G: nx.DiGraph, source_node: str) -> Dict[str, Tuple[float, float]]:
        """Create a circular layout with the source node at the top."""
        pos = {}
        nodes = list(G.nodes())
        
        # Remove source node from list for circular arrangement
        remaining_nodes = [n for n in nodes if n != source_node]
        num_nodes = len(remaining_nodes)
        
        # Arrange other nodes in a circle
        r = 3  # radius of circle
        for i, node in enumerate(remaining_nodes):
            theta = 2 * np.pi * i / num_nodes  # angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            pos[node] = (x, y)
        
        # Place source node above the circle
        pos[source_node] = (0, r + 1)
        
        return pos

    
    def plot_full_flow_paths(self, graph: BaseGraph, edge_flows: Dict,
                           id_to_address: Dict[str, str], filename: str):
        """
        Plot full flow paths including intermediate nodes.
        Shows complete path including token-holding positions.
        """
        flow_graph = nx.DiGraph()
        
        # Add edges with their flows
        for (u, v), flow in edge_flows.items():
            if flow > 0:
                edge_data = graph.get_edge_data(u, v) if graph.has_edge(u, v) else {}
                token = edge_data.get('label', 'unknown')
                flow_graph.add_edge(u, v, flow=flow, token=token)
        
        if not flow_graph.nodes():
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No flow paths found", 
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    fontsize=16)
            plt.axis('off')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # Find source and sink
        sources = [node for node in flow_graph.nodes() if flow_graph.in_degree(node) == 0]
        sinks = [node for node in flow_graph.nodes() if flow_graph.out_degree(node) == 0]
        
        if not sources or not sinks:
            print("Error: No source or sink found in the flow graph.")
            return
        
        source = sources[0]
        sink = sinks[0]
        
        # Create layout
        pos = self._hierarchical_layout(flow_graph, source, sink)
        
        plt.figure(figsize=(20, 10))
        
        # Draw different node types
        real_nodes = [n for n in flow_graph.nodes() if '_' not in str(n)]
        intermediate_nodes = [n for n in flow_graph.nodes() if '_' in str(n)]
        
        nx.draw_networkx_nodes(
            flow_graph, pos,
            nodelist=real_nodes,
            node_color='lightblue',
            node_size=1000,
            alpha=0.7
        )
        
        nx.draw_networkx_nodes(
            flow_graph, pos,
            nodelist=intermediate_nodes,
            node_color='lightgreen',
            node_size=800,
            node_shape='s',
            alpha=0.7
        )
        
        # Draw edges
        edge_widths = []
        for u, v in flow_graph.edges():
            flow = flow_graph[u][v]['flow']
            edge_widths.append(1 + np.log1p(flow) * 0.5)
        
        nx.draw_networkx_edges(
            flow_graph, pos,
            width=edge_widths,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            alpha=0.6
        )
        
        # Create labels
        node_labels = {}
        for node in flow_graph.nodes():
            if '_' in str(node):
                _, token = str(node).split('_')
                node_labels[node] = f"Token {token}"
            else:
                addr = id_to_address.get(str(node), str(node))
                node_labels[node] = f"{addr[:6]}...{addr[-4:]}" if len(addr) > 10 else addr
        
        nx.draw_networkx_labels(
            flow_graph, pos,
            node_labels,
            font_size=8
        )
        
        # Edge labels
        edge_labels = {}
        for u, v in flow_graph.edges():
            data = flow_graph.get_edge_data(u, v)
            flow = data['flow']
            token = data.get('token', 'unknown')
            edge_labels[(u, v)] = f"Flow: {flow:,}\nToken: {token}"
        
        nx.draw_networkx_edge_labels(
            flow_graph, pos,
            edge_labels,
            font_size=6
        )
        
        plt.title("Complete Flow Paths with Intermediate Nodes", fontsize=16)
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='lightblue', markersize=15,
                      label='Accounts'),
            plt.Line2D([0], [0], marker='s', color='w',
                      markerfacecolor='lightgreen', markersize=15,
                      label='Token Positions')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_hierarchical_layout(self, G: nx.DiGraph, source_node: str) -> Dict[str, Tuple[float, float]]:
        """Create a hierarchical layout with the source node at the top."""
        # Compute the shortest paths from the source node to all other nodes
        lengths = nx.single_source_shortest_path_length(G, source_node)
        
        # Group nodes by their distance from the source node
        nodes_by_level = defaultdict(list)
        for node, level in lengths.items():
            nodes_by_level[level].append(node)
        
        # Sort the levels to ensure consistent ordering
        sorted_levels = sorted(nodes_by_level.items())
        
        pos = {}
        y_gap = 2.0  # Vertical gap between layers
        x_gap = 2.0  # Horizontal gap between nodes in the same layer
        
        # Position nodes level by level
        for level, nodes in sorted_levels:
            y = -level * y_gap  # Negative y to place nodes below the source
            num_nodes = len(nodes)
            x_offset = -((num_nodes - 1) * x_gap) / 2  # Center the nodes
            for i, node in enumerate(sorted(nodes)):
                x = x_offset + i * x_gap
                pos[node] = (x, y)
        
        return pos


    def _hierarchical_layout(self, G: nx.DiGraph, source: str, sink: str,
                           horizontal_spacing: float = 5.0,
                           vertical_spacing: float = 2.0) -> Dict[str, Tuple[float, float]]:
        """Create a hierarchical layout for the flow graph."""
        def get_node_rank(node: str) -> int:
            if node == source:
                return 0
            elif node == sink:
                return max_rank
            else:
                paths = list(nx.all_simple_paths(G, source, node))
                return max(len(p) - 1 for p in paths) if paths else len(G)

        # Calculate ranks
        max_rank = len(nx.dag_longest_path(G)) - 1
        ranks = {node: get_node_rank(node) for node in G.nodes()}
        
        # Group nodes by rank
        nodes_by_rank = defaultdict(list)
        for node, rank in ranks.items():
            nodes_by_rank[rank].append(node)
            
        # Calculate positions
        pos = {}
        for rank, nodes in nodes_by_rank.items():
            x = rank * horizontal_spacing
            for i, node in enumerate(sorted(nodes)):
                y = (i - (len(nodes) - 1) / 2) * vertical_spacing
                pos[node] = (x, y)
        
        return pos

    @staticmethod
    def ensure_output_directory(directory: str):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created output directory: {directory}")