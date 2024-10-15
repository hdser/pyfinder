from src.graph_manager import GraphManager
from src.graph_creation import NetworkXGraph, GraphToolGraph
import networkx as nx
from networkx.algorithms.flow import (
    edmonds_karp, 
    preflow_push, 
    shortest_augmenting_path,
    boykov_kolmogorov,
    dinitz,
)
from graph_tool.flow import edmonds_karp_max_flow as gt_edmonds_karp, push_relabel_max_flow as gt_push_relabel, boykov_kolmogorov_max_flow as gt_boykov_kolmogorov
import time
import os
from typing import List, Tuple, Callable, Optional
import pandas as pd
import random
import signal
import os
from datetime import datetime

def timeout_handler(signum, frame):
    raise TimeoutError("Flow computation timed out")

def write_transaction_info(flow_value, execution_time, simplified_paths, simplified_edge_flows, id_to_address, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/flow_results_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("Flow Computation Results\n")
        f.write("------------------------\n")
        f.write(f"Total Flow: {flow_value}\n")
        f.write(f"Computation Time: {execution_time:.6f}s\n")
        f.write(f"Number of Transfers: {len(simplified_edge_flows)}\n\n")
        f.write(f"Distinct Paths: {len(simplified_paths)}\n\n")
        
        f.write("Detailed Transfers:\n")
        f.write("-------------------\n")
        for (u, v), token_flows in simplified_edge_flows.items():
            for token, flow in token_flows.items():
                f.write(f"From: {id_to_address[u]}\n")
                f.write(f"To: {id_to_address[v]}\n")
                f.write(f"Token: {id_to_address[token]}\n")
                f.write(f"Amount: {flow}\n")
                f.write("-------------------\n")
    
    print(f"Transaction information has been written to {filename}")

def run_mode(graph_manager: GraphManager):
    algorithms = {
        '1': ('Default (Preflow Push)', preflow_push if isinstance(graph_manager.graph, NetworkXGraph) else gt_push_relabel),
        '2': ('Edmonds-Karp', edmonds_karp if isinstance(graph_manager.graph, NetworkXGraph) else gt_edmonds_karp),
        '3': ('Shortest Augmenting Path', shortest_augmenting_path if isinstance(graph_manager.graph, NetworkXGraph) else None),
        '4': ('Boykov-Kolmogorov', boykov_kolmogorov if isinstance(graph_manager.graph, NetworkXGraph) else gt_boykov_kolmogorov),
        '5': ('Dinitz', dinitz if isinstance(graph_manager.graph, NetworkXGraph) else None),
    }

    while True:
        print("\nChoose an algorithm:")
        for key, (name, func) in algorithms.items():
            if func is not None:
                print(f"{key}. {name}")
        print("q. Quit")

        choice = input("Enter your choice: ")

        if choice.lower() == 'q':
            break

        if choice not in algorithms or algorithms[choice][1] is None:
            print("Invalid choice. Please try again.")
            continue

        algo_name, algo_func = algorithms[choice]

        while True:
            print("\nEnter Ethereum addresses for source and sink nodes.")
            print("Addresses should be in the format: 0x1234...abcd (42 characters total)")
            source = input("Enter source node (Ethereum address): ").strip().lower()
            sink = input("Enter sink node (Ethereum address): ").strip().lower()

            if len(source) != 42 or not source.startswith('0x') or len(sink) != 42 or not sink.startswith('0x'):
                print("Error: Invalid address format. Please ensure addresses are 42 characters long and start with '0x'.")
                continue

            break

        requested_flow = input("Enter requested flow value (press Enter for max flow) [mCRC]: ")
        requested_flow = requested_flow if requested_flow else None

        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # Set timeout to 5 minutes

            start_time = time.time()
            flow_value, simplified_paths, simplified_edge_flows, original_edge_flows = graph_manager.analyze_flow(source, sink, algo_func, requested_flow)
            end_time = time.time()
            execution_time = end_time - start_time

            signal.alarm(0)  # Cancel the alarm

            print(f"\nAlgorithm: {algo_name}")
            print(f"Execution time: {execution_time:.4f} seconds")
            print(f"Flow from {source} to {sink}: {flow_value} mCRC")
            if requested_flow is not None:
                print(f"Requested flow [mCRC]: {requested_flow}")
                if int(flow_value) < int(requested_flow):
                    print("Note: Achieved flow is less than requested flow.")

            output_dir = 'output'
            os.makedirs(output_dir, exist_ok=True)
            graph_manager.visualize_flow(simplified_paths, simplified_edge_flows, original_edge_flows, output_dir)
            print(f"\nVisualization saved in the '{output_dir}' directory.")

            # Write transaction information to file
            write_transaction_info(flow_value, execution_time, simplified_paths, simplified_edge_flows, 
                                   graph_manager.data_ingestion.id_to_address, output_dir)

        except TimeoutError:
            print("Flow computation timed out after 5 minutes.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Error details:")
            import traceback
            traceback.print_exc()

def benchmark_mode(graph_manager: GraphManager):
    algorithms = [
        ('Default (Preflow Push)', preflow_push if isinstance(graph_manager.graph, NetworkXGraph) else gt_push_relabel),
        ('Edmonds-Karp', edmonds_karp if isinstance(graph_manager.graph, NetworkXGraph) else gt_edmonds_karp),
        ('Shortest Augmenting Path', shortest_augmenting_path if isinstance(graph_manager.graph, NetworkXGraph) else None),
        ('Boykov-Kolmogorov', boykov_kolmogorov if isinstance(graph_manager.graph, NetworkXGraph) else gt_boykov_kolmogorov),
        ('Dinitz', dinitz if isinstance(graph_manager.graph, NetworkXGraph) else None),
    ]

    results = []

    print(f"\nNode information:\n{graph_manager.get_node_info()}")

    requested_flow = input("Enter requested flow value for all pairs (press Enter for max flow) [mCRC]: ")
    requested_flow = requested_flow if requested_flow else None

    num_pairs_input = input("Enter the number of random source-sink pairs to test: ")
    try:
        num_pairs = int(num_pairs_input)
    except ValueError:
        print("Invalid number entered. Defaulting to 5 pairs.")
        num_pairs = 5

    if isinstance(graph_manager.graph, NetworkXGraph):
        nodes = [node for node in graph_manager.graph.g_nx.nodes() if '_' not in str(node)]
    else:  # GraphToolGraph
        nodes = [str(graph_manager.graph.vertex_id[v]) for v in graph_manager.graph.g_gt.vertices() if '_' not in str(graph_manager.graph.vertex_id[v])]

    source_sink_pairs = []
    while len(source_sink_pairs) < num_pairs:
        source_id = random.choice(nodes)
        sink_id = random.choice(nodes)
        if source_id != sink_id:
            source_sink_pairs.append((source_id, sink_id))

    for algo_name, algo_func in algorithms:
        if algo_func is None:
            continue
        for source_id, sink_id in source_sink_pairs:
            try:
                source_address = graph_manager.data_ingestion.get_address_for_id(source_id)
                sink_address = graph_manager.data_ingestion.get_address_for_id(sink_id)
            except KeyError:
                print(f"Skipping pair ({source_id}, {sink_id}): One or both IDs not found in mapping.")
                continue

            if not graph_manager.graph.has_vertex(source_id) or not graph_manager.graph.has_vertex(sink_id):
                print(f"Skipping pair ({source_id}, {sink_id}): One or both nodes not in graph.")
                continue

            try:
                start_time = time.time()
                flow_value, simplified_paths, _, _ = graph_manager.analyze_flow(source_address, sink_address, algo_func, requested_flow)
                end_time = time.time()
                execution_time = end_time - start_time

                results.append({
                    'Algorithm': algo_name,
                    'Source': source_address,
                    'Sink': sink_address,
                    'Flow Value': str(flow_value),
                    'Requested Flow': requested_flow if requested_flow is not None else 'Max',
                    'Execution Time': execution_time,
                    'Num Transfers': len(simplified_paths)
                })
            except Exception as e:
                print(f"An error occurred for {algo_name} with source {source_address} and sink {sink_address}: {str(e)}")

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
    trusts_file = 'data/data-trust.csv'
    balances_file = 'data/data-balance.csv'
    
    print("Choose graph library:")
    print("1. NetworkX")
    print("2. graph-tool")
    graph_library_choice = input("Enter your choice (1 or 2): ")
    
    graph_library = 'networkx' if graph_library_choice == '1' else 'graph_tool'
    graph_manager = GraphManager(trusts_file, balances_file, graph_library)

    print("\nGraph information:")
    if isinstance(graph_manager.graph, NetworkXGraph):
        print(f"Number of nodes in graph: {graph_manager.graph.g_nx.number_of_nodes()}")
        print(f"Number of edges in graph: {graph_manager.graph.g_nx.number_of_edges()}")
    else:  # GraphToolGraph
        print(f"Number of nodes in graph: {graph_manager.graph.g_gt.num_vertices()}")
        print(f"Number of edges in graph: {graph_manager.graph.g_gt.num_edges()}")

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