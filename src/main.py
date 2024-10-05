from src.graph_manager import GraphManager
import networkx as nx
from networkx.algorithms.flow import (
    edmonds_karp, 
    preflow_push, 
    shortest_augmenting_path,
    maximum_flow_value, 
    minimum_cut,
    minimum_cut_value,
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

        print(f"\nNode information:\n{get_node_info(graph_manager.graph.g_nx)}")
        
        source = input("Enter source node: ")
        sink = input("Enter sink node: ")

        if source not in graph_manager.graph.g_nx or sink not in graph_manager.graph.g_nx:
            print(f"Error: Source node '{source}' or sink node '{sink}' not in graph.")
            continue

        requested_flow = input("Enter requested flow value (press Enter for max flow): ")
        requested_flow = requested_flow if requested_flow else None

        try:
            start_time = time.time()
            flow_value, simplified_paths, simplified_edge_flows, original_edge_flows = graph_manager.analyze_flow(source, sink, algo_func, requested_flow)
            end_time = time.time()

            print(f"\nAlgorithm: {algo_name}")
            print(f"Execution time: {end_time - start_time:.4f} seconds")
            print(f"Flow from {source} to {sink}: {flow_value}")
            if requested_flow is not None:
                print(f"Requested flow: {requested_flow}")
                if Decimal(flow_value) < Decimal(requested_flow):
                    print("Note: Achieved flow is less than requested flow.")

            output_dir = 'output'
            graph_manager.visualize_flow(simplified_paths, simplified_edge_flows, original_edge_flows, output_dir)
            print(f"\nVisualization saved in the '{output_dir}' directory.")
        except nx.NetworkXError as e:
            print(f"Error occurred: {str(e)}")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")

def benchmark_mode(graph_manager: GraphManager, source_sink_pairs: List[Tuple[str, str]]):
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

    for algo_name, algo_func in algorithms:
        for source, sink in source_sink_pairs:
            if source not in graph_manager.graph.g_nx or sink not in graph_manager.graph.g_nx:
                print(f"Skipping pair ({source}, {sink}): One or both nodes not in graph.")
                continue
            
            try:
                start_time = time.time()
                if algo_func in [maximum_flow_value, minimum_cut_value]:
                    flow_value = algo_func(graph_manager.graph.g_nx, source, sink)
                elif algo_func == minimum_cut:
                    cut_value, partition = algo_func(graph_manager.graph.g_nx, source, sink)
                    flow_value = cut_value
                else:
                    flow_value, _, _, _ = graph_manager.analyze_flow(source, sink, algo_func, requested_flow)
                end_time = time.time()
                execution_time = end_time - start_time

                results.append({
                    'Algorithm': algo_name,
                    'Source': source,
                    'Sink': sink,
                    'Flow Value': str(flow_value),
                    'Requested Flow': requested_flow if requested_flow is not None else 'Max',
                    'Execution Time': execution_time
                })
            except nx.NetworkXError as e:
                print(f"Error occurred for {algo_name} with source {source} and sink {sink}: {str(e)}")
            except Exception as e:
                print(f"An unexpected error occurred for {algo_name} with source {source} and sink {sink}: {str(e)}")

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
        # You can modify this list of source-sink pairs as needed
        source_sink_pairs = [
            ('9', '1'),
            ('9', '318'),
            ('9', '10'),
        ]
        benchmark_mode(graph_manager, source_sink_pairs)
    else:
        print("Invalid mode selection. Exiting.")

if __name__ == "__main__":
    main()