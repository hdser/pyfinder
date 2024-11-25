from typing import Dict, List, Tuple, Optional, Any
import argparse
import time
import random
import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
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
    boykov_kolmogorov_max_flow as gt_boykov_kolmogorov
)

from src.graph_manager import GraphManager 

def load_postgres_config() -> Dict[str, str]:
    """Load PostgreSQL configuration from environment variables."""
    load_dotenv()
    
    required_vars = [
        'POSTGRES_HOST',
        'POSTGRES_PORT',
        'POSTGRES_DB',
        'POSTGRES_USER',
        'POSTGRES_PASSWORD'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return {
        'host': os.getenv('POSTGRES_HOST'),
        'port': os.getenv('POSTGRES_PORT'),
        'dbname': os.getenv('POSTGRES_DB'),
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD')
    }

def get_data_source() -> Tuple[Any, Any]:
    """Get data source configuration from user input."""
    print("\nChoose data source:")
    print("1. Local CSV files")
    print("2. PostgreSQL (from .env)")
    
    choice = input("Enter your choice (1-2): ")
    queries_dir = "queries"  # Default queries directory

    if choice == "1":
        trusts_file = input("Enter path to trusts CSV file [default: data/data-trust.csv]: ").strip()
        balances_file = input("Enter path to balances CSV file [default: data/data-balance.csv]: ").strip()
        
        # Use default paths if none provided
        if not trusts_file:
            trusts_file = 'data/data-trust.csv'
        if not balances_file:
            balances_file = 'data/data-balance.csv'
        
        if not os.path.exists(trusts_file) or not os.path.exists(balances_file):
            raise FileNotFoundError("One or both CSV files not found")
        
        return (trusts_file, balances_file)
    
    elif choice == "2":
        return (load_postgres_config(), queries_dir)
    
    else:
        raise ValueError("Invalid choice")

def write_results(flow_value: int, execution_time: float, simplified_paths: List, 
                 simplified_edge_flows: Dict, graph_manager: GraphManager, output_dir: str):
    """Write analysis results to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/flow_results_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("Flow Computation Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Flow: {flow_value:,} mCRC\n")
        f.write(f"Computation Time: {execution_time:.6f}s\n")
        f.write(f"Number of Distinct Paths: {len(simplified_paths)}\n")
        f.write(f"Number of Transfers: {len(simplified_edge_flows)}\n\n")
        
        f.write("Detailed Transfers:\n")
        f.write("-" * 50 + "\n")
        for (u, v), token_flows in simplified_edge_flows.items():
            for token, flow in token_flows.items():
                f.write(f"From: {graph_manager.data_ingestion.get_address_for_id(u)}\n")
                f.write(f"To: {graph_manager.data_ingestion.get_address_for_id(v)}\n")
                f.write(f"Token: {graph_manager.data_ingestion.get_address_for_id(token)}\n")
                f.write(f"Amount: {flow:,} mCRC\n")
                f.write("-" * 50 + "\n")
    
    print(f"\nResults written to {filename}")

def run_analysis(graph_manager: GraphManager, algorithm_name: str, algorithm_func, 
                source: str, sink: str, requested_flow: Optional[str] = None):
    """Run flow analysis with given parameters."""
    try:
        start_time = time.time()
        flow_value, simplified_paths, simplified_edge_flows, original_edge_flows = graph_manager.analyze_flow(
            source=source,
            sink=sink,
            flow_func=algorithm_func,
            cutoff=requested_flow
        )
        execution_time = time.time() - start_time

        print(f"\nAlgorithm: {algorithm_name}")
        print(f"Execution time: {execution_time:.4f} seconds")
        print(f"Flow value: {flow_value:,} mCRC")
        if requested_flow:
            print(f"Requested flow: {requested_flow:,} mCRC")
            if flow_value < int(requested_flow):
                print("Note: Achieved flow is less than requested flow")

        # Create output directory
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)

        # Save visualizations
        graph_manager.visualize_flow(
            simplified_paths,
            simplified_edge_flows,
            original_edge_flows,
            output_dir
        )
        print(f"\nVisualizations saved in {output_dir}/")

        # Write detailed results
        write_results(
            flow_value,
            execution_time,
            simplified_paths,
            simplified_edge_flows,
            graph_manager,
            output_dir
        )

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

def get_algorithms(graph_type: str) -> Dict[str, Any]:
    """Get available algorithms for the selected graph type."""
    if graph_type == 'networkx':
        return {
            '1': ('Preflow Push', preflow_push),
            '2': ('Edmonds-Karp', edmonds_karp),
            '3': ('Shortest Augmenting Path', shortest_augmenting_path),
            '4': ('Boykov-Kolmogorov', boykov_kolmogorov),
            '5': ('Dinitz', dinitz),
        }
    elif graph_type == 'graph_tool':
        return {
            '1': ('Push-Relabel', gt_push_relabel),
            '2': ('Edmonds-Karp', gt_edmonds_karp),
            '3': ('Boykov-Kolmogorov', gt_boykov_kolmogorov),
        }
    else:  # OR-Tools
        return {
            '1': ('OR-Tools Max Flow', None),
        }

def main():
    try:
        # Get data source
        data_source = get_data_source()

        # Choose graph implementation
        print("\nChoose graph implementation:")
        print("1. NetworkX")
        print("2. graph-tool")
        print("3. OR-Tools")
        graph_choice = input("Enter choice (1-3): ")

        graph_type_map = {
            '1': 'networkx',
            '2': 'graph_tool',
            '3': 'ortools'
        }
        graph_type = graph_type_map.get(graph_choice, 'networkx')

        # Initialize GraphManager
        print("\nInitializing graph...")
        graph_manager = GraphManager(data_source, graph_type)
        print("\nGraph information:")
        print(graph_manager.get_node_info())

        # Get available algorithms
        algorithms = get_algorithms(graph_type)

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

            # Get source and sink addresses
            while True:
                print("\nEnter Ethereum addresses (42 characters starting with 0x):")
                source = input("Source address: ").strip().lower()
                sink = input("Sink address: ").strip().lower()

                if all(len(addr) == 42 and addr.startswith('0x') 
                      for addr in [source, sink]):
                    break
                print("Invalid address format. Please try again.")

            # Get requested flow
            requested_flow = input("\nEnter requested flow [mCRC] (press Enter for max flow): ")
            requested_flow = requested_flow if requested_flow.strip() else None

            # Run analysis
            algorithm_name, algorithm_func = algorithms[choice]
            run_analysis(
                graph_manager,
                algorithm_name,
                algorithm_func,
                source,
                sink,
                requested_flow
            )

    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())