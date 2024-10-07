from src.graph_manager import GraphManager
import networkx as nx
from networkx.algorithms.flow import (
    edmonds_karp, 
    preflow_push, 
    shortest_augmenting_path,
    boykov_kolmogorov,
    dinitz,
)
import time
import os
from typing import List, Tuple, Callable, Optional
import pandas as pd
import random
from decimal import Decimal

def get_node_info(graph: nx.DiGraph):
    nodes = list(graph.nodes())
    return f"Total nodes: {len(nodes)}\nSample nodes: {random.sample(nodes, min(5, len(nodes)))}"

def run_mode(graph_manager: GraphManager):
    algorithms = {
        '1': ('Default (Preflow Push)', preflow_push),
        '2': ('Edmonds-Karp', edmonds_karp),
        '3': ('Shortest Augmenting Path', shortest_augmenting_path),
        '4': ('Boykov-Kolmogorov', boykov_kolmogorov),
        '5': ('Dinitz', dinitz),
    }

    while True:
        print("\nChoose an algorithm:")
        for key, (name, _) in algorithms.items():
            print(f"{key}. {name}")
        print("q. Quit")

        choice = input("Enter your choice: ")

        if choice.lower() == 'q':
            break

        if choice not in algorithms:
            print("Invalid choice. Please try again.")
            continue

        algo_name, algo_func = algorithms[choice]

        print(f"\nNode information:\n{graph_manager.get_node_info()}")

        while True:
            print("\nEnter Ethereum addresses for source and sink nodes.")
            print("Addresses should be in the format: 0x1234...abcd (42 characters total)")
            source = input("Enter source node (Ethereum address): ").strip().lower()
            sink = input("Enter sink node (Ethereum address): ").strip().lower()

            if len(source) != 42 or not source.startswith('0x') or len(sink) != 42 or not sink.startswith('0x'):
                print("Error: Invalid address format. Please ensure addresses are 42 characters long and start with '0x'.")
                continue

            try:
                source_id = graph_manager.data_ingestion.get_id_for_address(source)
                sink_id = graph_manager.data_ingestion.get_id_for_address(sink)
                if source_id is None or sink_id is None:
                    raise ValueError(f"Source address '{source}' or sink address '{sink}' not found in the graph.")
                break
            except ValueError as e:
                print(f"Error: {str(e)}")
                continue

        print("\nGraph information:")
        print(f"Number of nodes in graph: {graph_manager.graph.g_nx.number_of_nodes()}")
        print(f"Number of edges in graph: {graph_manager.graph.g_nx.number_of_edges()}")
        print("Sample of node IDs in graph:", list(graph_manager.graph.g_nx.nodes())[:10])
        
        print("\nDetailed node information:")
        for node in list(graph_manager.graph.g_nx.nodes())[:10]:
            if '_' in node:
                print(f"Intermediate Node ID: {node}")
            else:
                address = graph_manager.data_ingestion.get_address_for_id(node)
                print(f"Node ID: {node}, Address: {address}")

        requested_flow = input("Enter requested flow value (press Enter for max flow): ")
        requested_flow = requested_flow if requested_flow else None

        try:
            start_time = time.time()
            flow_value, simplified_paths, simplified_edge_flows, original_edge_flows, aggregated_flows = graph_manager.analyze_flow(source, sink, algo_func, requested_flow)
            end_time = time.time()

            print(f"\nAlgorithm: {algo_name}")
            print(f"Execution time: {end_time - start_time:.4f} seconds")
            print(f"Flow from {source} to {sink}: {flow_value}")
            if requested_flow is not None:
                print(f"Requested flow: {requested_flow}")
                if Decimal(flow_value) < Decimal(requested_flow):
                    print("Note: Achieved flow is less than requested flow.")

            print("\nAggregated Transfers:")
            for (u, v, token), flow in aggregated_flows.items():
                u_address = graph_manager.data_ingestion.get_address_for_id(u)
                v_address = graph_manager.data_ingestion.get_address_for_id(v)
                token_address = graph_manager.data_ingestion.get_address_for_id(token)
                print(f"{u_address} --> {v_address} (Flow: {flow}, Token: {token_address})")

            output_dir = 'output'
            graph_manager.visualize_flow(simplified_paths, simplified_edge_flows, original_edge_flows, output_dir)
            print(f"\nVisualization saved in the '{output_dir}' directory.")
        except nx.NetworkXError as e:
            print(f"Error occurred: {str(e)}")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")

def benchmark_mode(graph_manager: GraphManager):
    algorithms = [
        ('Default (Preflow Push)', preflow_push),
        ('Edmonds-Karp', edmonds_karp),
        ('Shortest Augmenting Path', shortest_augmenting_path),
        ('Boykov-Kolmogorov', boykov_kolmogorov),
        ('Dinitz', dinitz),
    ]

    results = []

    print(f"\nNode information:\n{get_node_info(graph_manager.graph.g_nx)}")

    requested_flow = input("Enter requested flow value for all pairs (press Enter for max flow): ")
    requested_flow = requested_flow if requested_flow else None

    # Ask the user for the number of random source-sink pairs
    num_pairs_input = input("Enter the number of random source-sink pairs to test: ")
    try:
        num_pairs = int(num_pairs_input)
    except ValueError:
        print("Invalid number entered. Defaulting to 5 pairs.")
        num_pairs = 5

    # Get list of nodes (excluding intermediate nodes)
    nodes = [node for node in graph_manager.graph.g_nx.nodes() if '_' not in node]

    # Generate random source-sink pairs
    source_sink_pairs = []
    while len(source_sink_pairs) < num_pairs:
        source_id = random.choice(nodes)
        sink_id = random.choice(nodes)
        if source_id != sink_id:
            source_sink_pairs.append((source_id, sink_id))

    for algo_name, algo_func in algorithms:
        for source_id, sink_id in source_sink_pairs:
            try:
                source_address = graph_manager.data_ingestion.get_address_for_id(source_id)
                sink_address = graph_manager.data_ingestion.get_address_for_id(sink_id)
            except KeyError:
                print(f"Skipping pair ({source_id}, {sink_id}): One or both IDs not found in mapping.")
                continue

            if source_id not in graph_manager.graph.g_nx or sink_id not in graph_manager.graph.g_nx:
                print(f"Skipping pair ({source_id}, {sink_id}): One or both nodes not in graph.")
                continue

            try:
                start_time = time.time()
                # Pass Ethereum addresses to analyze_flow
                flow_value, _, _, _, aggregated_flows = graph_manager.analyze_flow(source_address, sink_address, algo_func, requested_flow)
                end_time = time.time()
                execution_time = end_time - start_time

                results.append({
                    'Algorithm': algo_name,
                    'Source': source_address,
                    'Sink': sink_address,
                    'Flow Value': str(flow_value),
                    'Requested Flow': requested_flow if requested_flow is not None else 'Max',
                    'Execution Time': execution_time,
                    'Num Aggregated Transfers': len(aggregated_flows)
                })
            except nx.NetworkXError as e:
                print(f"Error occurred for {algo_name} with source {source_address} and sink {sink_address}: {str(e)}")
            except Exception as e:
                print(f"An unexpected error occurred for {algo_name} with source {source_address} and sink {sink_address}: {str(e)}")

    if results:
        df_results = pd.DataFrame(results)
        print("\nBenchmark Results:")
        print(df_results)

        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        csv_file = os.path.join(output_dir, 'benchmark_results.csv')
        df_results.to_csv(csv_file, index=False)
        print(f"\nBenchmark results saved to {csv_file}")
    else:
        print("No valid results to display. Please check your source-sink pairs and graph structure.")


def main():
    trusts_file = 'data/circles_public_V_CrcV2_TrustRelations.csv'
    balances_file = 'data/circles_public_V_CrcncesByAccountAndToken.csv'
    graph_manager = GraphManager(trusts_file, balances_file)

    print("Choose a mode:")
    print("1. Run Mode")
    print("2. Benchmark Mode")
    mode = input("Enter your choice (1 or 2): ")

    if mode == '1':
        run_mode(graph_manager)
    elif mode == '2':
        benchmark_mode(graph_manager)
    else:
        print("Invalid mode selection. Exiting.")

if __name__ == "__main__":
    main()
