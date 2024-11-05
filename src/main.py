from src.graph_manager import GraphManager
from src.graph import NetworkXGraph, GraphToolGraph
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
from typing import List, Tuple, Callable, Optional, Dict, Union
import pandas as pd
import random
import signal
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

def load_postgres_from_env() -> Dict[str, str]:
    """Load PostgreSQL configuration from environment variables."""
    required_vars = [
        'POSTGRES_HOST',
        'POSTGRES_PORT',
        'POSTGRES_DB',
        'POSTGRES_USER',
        'POSTGRES_PASSWORD'
    ]
    
    # Check if all required variables are present
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

def get_data_source() -> Union[tuple, Dict[str, str], Tuple[Dict[str, str], str]]:
    """Get data source configuration from user input."""
    print("\nChoose data source:")
    print("1. Local CSV files")
    print("2. PostgreSQL database (manual configuration)")
    print("3. PostgreSQL database (from .env file)")
    print("4. PostgreSQL database (from saved configuration)")
    
    choice = input("Enter your choice (1-4): ")

    if choice == "1":
        trusts_file = input("Enter path to trusts CSV file [default: data/data-trust.csv]: ").strip()
        balances_file = input("Enter path to balances CSV file [default: data/data-balance.csv]: ").strip()
        
        # Use default paths if none provided
        if not trusts_file:
            trusts_file = 'data/data-trust.csv'
        if not balances_file:
            balances_file = 'data/data-balance.csv'
            
        # Verify files exist
        if not os.path.exists(trusts_file) or not os.path.exists(balances_file):
            raise FileNotFoundError("One or both CSV files not found")
            
        return (trusts_file, balances_file)
    
    # For PostgreSQL options, also get queries directory
   # queries_dir = input("\nEnter path to SQL queries directory [default: queries]: ").strip() or "queries"
   # if not os.path.exists(queries_dir):
   #     raise FileNotFoundError(f"Queries directory not found: {queries_dir}")
    queries_dir = "queries"

    if choice == "2":
        print("\nEnter PostgreSQL connection details:")
        db_config = {
            'host': input("Host [localhost]: ").strip() or 'localhost',
            'port': input("Port [5432]: ").strip() or '5432',
            'dbname': input("Database name: ").strip(),
            'user': input("Username: ").strip(),
            'password': input("Password: ").strip()
        }
        
        # Ask if user wants to save configuration
        save_config = input("\nSave this configuration for future use? (y/n): ").lower().strip()
        if save_config == 'y':
            save_db_config(db_config)
            print("Configuration saved.")
        
        return (db_config, queries_dir)
    
    elif choice == "3":
        # Load .env file
       # env_path = input("Enter path to .env file [default: .env]: ").strip() or '.env'
       # if not os.path.exists(env_path):
       #     raise FileNotFoundError(f"Environment file not found: {env_path}")
        env_path='.env'
        # Load the environment variables
        load_dotenv(env_path)
        
        try:
            return (load_postgres_from_env(), queries_dir)
        except ValueError as e:
            raise ValueError(f"Error loading from .env file: {str(e)}")
    
    elif choice == "4":
        config = load_db_config()
        if not config:
            raise ValueError("No saved configuration found. Please use another option.")
        print("Using saved PostgreSQL configuration.")
        return (config, queries_dir)
    
    else:
        raise ValueError("Invalid choice")

def load_db_config() -> Optional[Dict[str, str]]:
    """Load database configuration from file if it exists."""
    config_file = Path.home() / '.pyfinder' / 'config.ini'
    if config_file.exists():
        config = configparser.ConfigParser()
        config.read(config_file)
        if 'postgresql' in config:
            return dict(config['postgresql'])
    return None

def save_db_config(db_config: Dict[str, str]):
    """Save database configuration to file."""
    config = configparser.ConfigParser()
    config['postgresql'] = db_config
    
    config_dir = Path.home() / '.pyfinder'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config_dir / 'config.ini', 'w') as configfile:
        config.write(configfile)

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
    """Run mode with proper handling of OR-Tools implementation."""
    
    # Define algorithms based on implementation type
    if isinstance(graph_manager.graph, NetworkXGraph):
        algorithms = {
            '1': ('Preflow Push', preflow_push),
            '2': ('Edmonds-Karp', edmonds_karp),
            '3': ('Shortest Augmenting Path', shortest_augmenting_path),
            '4': ('Boykov-Kolmogorov', boykov_kolmogorov),
            '5': ('Dinitz', dinitz),
        }
    elif isinstance(graph_manager.graph, GraphToolGraph):
        algorithms = {
            '1': ('Push-Relabel', gt_push_relabel),
            '2': ('Edmonds-Karp', gt_edmonds_karp),
            '3': ('Boykov-Kolmogorov', gt_boykov_kolmogorov),
        }
    else:  # OR-Tools
        algorithms = {
            '1': ('OR-Tools Max Flow', None),  # OR-Tools uses its own algorithm
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
            flow_value, simplified_paths, simplified_edge_flows, original_edge_flows = graph_manager.analyze_flow(
                source=source,
                sink=sink,
                flow_func=algo_func,  # Will be None for OR-Tools
                cutoff=requested_flow
            )
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

            # Create output directory if it doesn't exist
            output_dir = 'output'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save visualization
            graph_manager.visualize_flow(simplified_paths, simplified_edge_flows, original_edge_flows, output_dir)
            print(f"\nVisualization saved in the '{output_dir}' directory.")

            # Write detailed transaction info
            write_transaction_info(
                flow_value, 
                execution_time, 
                simplified_paths, 
                simplified_edge_flows, 
                graph_manager.data_ingestion.id_to_address, 
                output_dir
            )

        except TimeoutError:
            print("Flow computation timed out after 5 minutes.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Error details:")
            import traceback
            traceback.print_exc()

def benchmark_mode(graph_manager: GraphManager):
    # Define algorithms based on implementation type
    if isinstance(graph_manager.graph, NetworkXGraph):
        algorithms = [
           # ('Preflow Push', preflow_push),
            ('Edmonds-Karp', edmonds_karp),
            ('Shortest Augmenting Path', shortest_augmenting_path),
            ('Boykov-Kolmogorov', boykov_kolmogorov),
            ('Dinitz', dinitz)
        ]
    elif isinstance(graph_manager.graph, GraphToolGraph):
        algorithms = [
           # ('Push-Relabel', gt_push_relabel),
            ('Edmonds-Karp', gt_edmonds_karp),
            ('Boykov-Kolmogorov', gt_boykov_kolmogorov)
        ]
    else:  # OR-Tools
        algorithms = [('OR-Tools Max Flow', None)]

    print(f"\nNode information:\n{graph_manager.get_node_info()}")

    requested_flow = input("Enter requested flow value for all pairs (press Enter for max flow) [mCRC]: ")
    requested_flow = requested_flow if requested_flow else None

    num_pairs_input = input("Enter the number of random source-sink pairs to test: ")
    try:
        num_pairs = int(num_pairs_input)
    except ValueError:
        print("Invalid number entered. Defaulting to 5 pairs.")
        num_pairs = 5

    # Generate source-sink pairs
    if isinstance(graph_manager.graph, NetworkXGraph):
        nodes = [node for node in graph_manager.graph.g_nx.nodes() if '_' not in str(node)]
    elif isinstance(graph_manager.graph, GraphToolGraph):
        nodes = [str(graph_manager.graph.vertex_id[v]) for v in graph_manager.graph.g_gt.vertices() 
                if '_' not in str(graph_manager.graph.vertex_id[v])]
    else:  # OR-Tools
        nodes = [node for node in graph_manager.graph.node_to_index.keys() if '_' not in str(node)]

    source_sink_pairs = []
    while len(source_sink_pairs) < num_pairs:
        source_id = random.choice(nodes)
        sink_id = random.choice(nodes)
        if source_id != sink_id:
            source_sink_pairs.append((source_id, sink_id))

    # Create output directory and files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f'benchmark_results_{timestamp}.csv')
    log_file = os.path.join(output_dir, f'benchmark_log_{timestamp}.txt')

    # Initialize results file with headers
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Timestamp,Algorithm,Implementation,Source ID,Source Address,Sink ID,Sink Address,"
                "Flow Value,Requested Flow,Execution Time,Num Paths,Num Transfers\n")

    # Initialize log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Benchmark started at: {timestamp}\n")
        f.write(f"Implementation: {type(graph_manager.graph).__name__}\n")
        f.write(f"Number of pairs: {num_pairs}\n")
        f.write(f"Requested flow: {requested_flow if requested_flow else 'Max'}\n\n")

    results = []
    total_tests = len(algorithms) * len(source_sink_pairs)
    completed = 0

    for algo_name, algo_func in algorithms:
        print(f"\nTesting algorithm: {algo_name}")
        
        for source_id, sink_id in source_sink_pairs:
            try:
                source_address = graph_manager.data_ingestion.get_address_for_id(source_id)
                sink_address = graph_manager.data_ingestion.get_address_for_id(sink_id)
                
                print(f"\nTest {completed + 1}/{total_tests}:")
                print(f"Algorithm: {algo_name}")
                print(f"Source: {source_address[:10]}...")
                print(f"Sink: {sink_address[:10]}...")
                
                start_time = time.time()
                flow_value, simplified_paths, simplified_edge_flows, _ = graph_manager.analyze_flow(
                    source_address, 
                    sink_address, 
                    algo_func, 
                    requested_flow
                )
                execution_time = time.time() - start_time

                # Create and store result
                result = {
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Algorithm': algo_name,
                    'Implementation': type(graph_manager.graph).__name__,
                    'Source ID': source_id,
                    'Source Address': source_address,
                    'Sink ID': sink_id,
                    'Sink Address': sink_address,
                    'Flow Value': str(flow_value),
                    'Requested Flow': requested_flow if requested_flow else 'Max',
                    'Execution Time': execution_time,
                    'Num Paths': len(simplified_paths),
                    'Num Transfers': len(simplified_edge_flows)
                }
                
                results.append(result)
                
                # Write result immediately to CSV
                with open(results_file, 'a', encoding='utf-8') as f:
                    f.write(f"{result['Timestamp']},{result['Algorithm']},{result['Implementation']},"
                           f"{result['Source ID']},{result['Source Address']},{result['Sink ID']},"
                           f"{result['Sink Address']},{result['Flow Value']},{result['Requested Flow']},"
                           f"{result['Execution Time']},{result['Num Paths']},{result['Num Transfers']}\n")
                
                # Write to log file
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\nTest #{completed + 1}\n")
                    f.write(f"Algorithm: {algo_name}\n")
                    f.write(f"Source: {source_address}\n")
                    f.write(f"Sink: {sink_address}\n")
                    f.write(f"Flow Value: {flow_value}\n")
                    f.write(f"Execution Time: {execution_time:.4f}s\n")
                    f.write(f"Number of Paths: {len(simplified_paths)}\n")
                    f.write(f"Number of Transfers: {len(simplified_edge_flows)}\n")
                    f.write("-" * 50 + "\n")

                print(f"Result: Flow = {flow_value}, Time = {execution_time:.4f}s")
                print(f"Results written to {results_file}")
                
            except Exception as e:
                error_msg = f"Error with {algo_name} ({source_id}→{sink_id}): {str(e)}"
                print(error_msg)
                
                # Log error
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\nERROR in test #{completed + 1}\n")
                    f.write(f"{error_msg}\n")
                    f.write(f"Stack trace:\n{traceback.format_exc()}\n")
                    f.write("-" * 50 + "\n")
            
            completed += 1

    # Write final summary
    if results:
        df_results = pd.DataFrame(results)
        df_results['Flow Value'] = pd.to_numeric(df_results['Flow Value'], errors='coerce')
        
        # Create summary with better formatting
        execution_stats = df_results.groupby(['Implementation', 'Algorithm'])['Execution Time'].agg([
            ('Number of Tests', 'count'),
            ('Mean Time (s)', 'mean'),
            ('Std Dev (s)', 'std'),
            ('Min Time (s)', 'min'),
            ('Max Time (s)', 'max')
        ]).round(4)
        
        flow_stats = df_results.groupby(['Implementation', 'Algorithm'])['Flow Value'].agg([
            ('Mean Flow', 'mean'),
            ('Min Flow', 'min'),
            ('Max Flow', 'max')
        ]).round(2)
        
        # Combine the statistics
        summary = pd.concat([execution_stats, flow_stats], axis=1)
        
        # Write summary to log
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\nFinal Summary:\n")
            f.write("=" * 50 + "\n")
            f.write("\nExecution Time Statistics:\n")
            f.write(execution_stats.to_string())
            f.write("\n\nFlow Statistics:\n")
            f.write(flow_stats.to_string())
                
        # Save summary to CSV with better formatting
        summary_file = os.path.join(output_dir, f'benchmark_summary_{timestamp}.csv')
        
        # Format for CSV
        summary.to_csv(summary_file)
        
        # Display in console with better formatting
        print("\nExecution Time Statistics:")
        print(execution_stats)
        print("\nFlow Statistics:")
        print(flow_stats)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
        print(f"Log file saved to: {log_file}")
    else:
        print("No valid results to display.")

def compare_libraries_mode(networkx_graph_manager: GraphManager, graphtool_graph_manager: GraphManager):
    algorithms = {
        'NetworkX': [
            ('Preflow Push', preflow_push),
            ('Edmonds-Karp', edmonds_karp),
            ('Shortest Augmenting Path', shortest_augmenting_path),
            ('Boykov-Kolmogorov', boykov_kolmogorov),
            ('Dinitz', dinitz),
        ],
        'graph-tool': [
            ('Push-Relabel', gt_push_relabel),
            ('Edmonds-Karp', gt_edmonds_karp),
            ('Boykov-Kolmogorov', gt_boykov_kolmogorov),
        ]
    }

    results = []

    num_pairs_input = input("Enter the number of random source-sink pairs to test: ")
    try:
        num_pairs = int(num_pairs_input)
    except ValueError:
        print("Invalid number entered. Defaulting to 5 pairs.")
        num_pairs = 5

    # Use NetworkX graph for node selection (assuming both graphs have the same nodes)
    nodes = [node for node in networkx_graph_manager.graph.g_nx.nodes() if '_' not in str(node)]

    source_sink_pairs = []
    while len(source_sink_pairs) < num_pairs:
        source_id = random.choice(nodes)
        sink_id = random.choice(nodes)
        if source_id != sink_id:
            source_sink_pairs.append((source_id, sink_id))

    for source_id, sink_id in source_sink_pairs:
        source_address = networkx_graph_manager.data_ingestion.get_address_for_id(source_id)
        sink_address = networkx_graph_manager.data_ingestion.get_address_for_id(sink_id)

        if not networkx_graph_manager.graph.has_vertex(source_id) or not networkx_graph_manager.graph.has_vertex(sink_id):
            print(f"Skipping pair ({source_id}, {sink_id}): One or both nodes not in graph.")
            continue

        for _ in range(3):  # Run 3 times for each pair
            requested_flow = random.randint(1000, 1000000)

            for library, algos in algorithms.items():
                graph_manager = networkx_graph_manager if library == 'NetworkX' else graphtool_graph_manager

                for algo_name, algo_func in algos:
                    try:
                        start_time = time.time()
                        flow_value, simplified_paths, simplified_edge_flows, _ = graph_manager.analyze_flow(source_address, sink_address, algo_func, str(requested_flow))
                        end_time = time.time()
                        execution_time = end_time - start_time

                        results.append({
                            'Library': library,
                            'Algorithm': algo_name,
                            'Source': source_address,
                            'Sink': sink_address,
                            'Flow Value': str(flow_value),
                            'Requested Flow': str(requested_flow),
                            'Execution Time': execution_time,
                            'Num Paths': len(simplified_paths),
                            'Num Transfers': len(simplified_edge_flows)
                        })
                    except Exception as e:
                        print(f"An error occurred for {library} {algo_name} with source {source_address} and sink {sink_address}: {str(e)}")

    if results:
        df_results = pd.DataFrame(results)
        print("\nComparison Results:")
        print(df_results)

        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        csv_file = os.path.join(output_dir, 'library_comparison_results.csv')
        df_results.to_csv(csv_file, index=False)
        print(f"\nComparison results saved to {csv_file}")
    else:
        print("No valid results to display. Please check your source-sink pairs and graph structure.")

def main():
    try:
        data_source = get_data_source()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {str(e)}")
        return

    print("\nInitializing graph...")
    
    print("Choose graph library:")
    print("1. NetworkX")
    print("2. graph-tool")
    print("3. OR-Tools")
    graph_library_choice = input("Enter your choice (1-3): ")
    
    library_map = {
        '1': 'networkx',
        '2': 'graph_tool',
        '3': 'ortools'
    }
    
    if graph_library_choice not in library_map:
        print("Invalid choice. Using NetworkX as default.")
        graph_library = 'networkx'
    else:
        graph_library = library_map[graph_library_choice]
    
    try:
        # If data_source is a dictionary, it's PostgreSQL config
        if isinstance(data_source, tuple) and isinstance(data_source[0], dict):
            db_config, queries_dir = data_source
            graph_manager = GraphManager((db_config, queries_dir), graph_library)
        else:
            # Otherwise it's CSV files
            graph_manager = GraphManager(data_source, graph_library)

        print("\nGraph information:")
        if isinstance(graph_manager.graph, NetworkXGraph):
            print(f"Number of nodes in graph: {graph_manager.graph.g_nx.number_of_nodes()}")
            print(f"Number of edges in graph: {graph_manager.graph.g_nx.number_of_edges()}")
        elif isinstance(graph_manager.graph, GraphToolGraph):
            print(f"Number of nodes in graph: {graph_manager.graph.g_gt.num_vertices()}")
            print(f"Number of edges in graph: {graph_manager.graph.g_gt.num_edges()}")
        else:  # ORToolsGraph
            print(f"Number of nodes in graph: {len(graph_manager.graph.node_to_index)}")
            print(f"Number of edges in graph: {graph_manager.graph.solver.num_arcs()}")

        mode = input("\nChoose mode (1: Run, 2: Benchmark, 3: Compare Libraries): ")

        if mode == '1':
            run_mode(graph_manager)
        elif mode == '2':
            benchmark_mode(graph_manager)
        elif mode == '3':
            # Create managers for each implementation
            implementations = {
                'networkx': GraphManager(data_source, 'networkx'),
                'graph_tool': GraphManager(data_source, 'graph_tool'),
                'ortools': GraphManager(data_source, 'ortools')
            }
            compare_implementations_mode(implementations)
        else:
            print("Invalid mode selection. Exiting.")
            
    except Exception as e:
        print(f"Error initializing graph manager: {str(e)}")
        import traceback
        traceback.print_exc()
        return


def compare_implementations_mode(implementations: Dict[str, GraphManager]):
    """Compare performance across all implementations with immediate result writing."""
    algorithms = {
        'networkx': [
            ('Preflow Push', preflow_push),
            ('Edmonds-Karp', edmonds_karp),
            ('Shortest Augmenting Path', shortest_augmenting_path),
            ('Boykov-Kolmogorov', boykov_kolmogorov),
            ('Dinitz', dinitz),
        ],
        'graph_tool': [
            ('Push-Relabel', gt_push_relabel),
            ('Edmonds-Karp', gt_edmonds_karp),
            ('Boykov-Kolmogorov', gt_boykov_kolmogorov),
        ],
        'ortools': [
            ('OR-Tools Max Flow', None),
        ]
    }

    num_pairs_input = input("Enter the number of random source-sink pairs to test: ")
    try:
        num_pairs = int(num_pairs_input)
    except ValueError:
        print("Invalid number entered. Defaulting to 5 pairs.")
        num_pairs = 5

    # Create output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, f'comparison_results_{timestamp}.csv')
    log_file = os.path.join(output_dir, f'comparison_log_{timestamp}.txt')

    # Write headers
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Timestamp,Implementation,Algorithm,Source,Sink,Flow Value,"
                "Requested Flow,Execution Time,Num Paths,Num Transfers\n")

    # Initialize log
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Comparison started at: {timestamp}\n")
        f.write(f"Number of pairs: {num_pairs}\n")
        f.write(f"Implementations: {', '.join(implementations.keys())}\n\n")

    # Select node pairs
    networkx_manager = implementations['networkx']
    nodes = [node for node in networkx_manager.graph.g_nx.nodes() if '_' not in str(node)]
    
    source_sink_pairs = []
    while len(source_sink_pairs) < num_pairs:
        source_id = random.choice(nodes)
        sink_id = random.choice(nodes)
        if source_id != sink_id:
            source_sink_pairs.append((source_id, sink_id))

    total_tests = sum(len(algos) for algos in algorithms.values()) * len(source_sink_pairs) * 3
    completed = 0
    
    results = []
    current_summary = pd.DataFrame()

    for source_id, sink_id in source_sink_pairs:
        source_address = networkx_manager.data_ingestion.get_address_for_id(source_id)
        sink_address = networkx_manager.data_ingestion.get_address_for_id(sink_id)

        print(f"\nTesting pair {len(results) // (len(algorithms) * 3) + 1}/{num_pairs}:")
        print(f"Source: {source_address[:10]}...")
        print(f"Sink: {sink_address[:10]}...")

        for trial in range(3):
            requested_flow = random.randint(1000, 1000000)
            
            for impl_name, graph_manager in implementations.items():
                for algo_name, algo_func in algorithms[impl_name]:
                    try:
                        print(f"\nTest {completed + 1}/{total_tests}:")
                        print(f"Implementation: {impl_name}")
                        print(f"Algorithm: {algo_name}")
                        print(f"Trial: {trial + 1}/3")

                        start_time = time.time()
                        flow_value, simplified_paths, simplified_edge_flows, _ = graph_manager.analyze_flow(
                            source_address, 
                            sink_address, 
                            algo_func, 
                            str(requested_flow)
                        )
                        execution_time = time.time() - start_time

                        result = {
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Implementation': impl_name,
                            'Algorithm': algo_name,
                            'Source': source_address,
                            'Sink': sink_address,
                            'Flow Value': str(flow_value),
                            'Requested Flow': str(requested_flow),
                            'Execution Time': execution_time,
                            'Num Paths': len(simplified_paths),
                            'Num Transfers': len(simplified_edge_flows)
                        }
                        
                        results.append(result)
                        
                        # Write result immediately
                        with open(results_file, 'a', encoding='utf-8') as f:
                            f.write(f"{result['Timestamp']},{result['Implementation']},"
                                   f"{result['Algorithm']},{result['Source']},"
                                   f"{result['Sink']},{result['Flow Value']},"
                                   f"{result['Requested Flow']},{result['Execution Time']},"
                                   f"{result['Num Paths']},{result['Num Transfers']}\n")
                        
                        # Write to log
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"\nTest #{completed + 1}\n")
                            f.write(f"Implementation: {impl_name}\n")
                            f.write(f"Algorithm: {algo_name}\n")
                            f.write(f"Trial: {trial + 1}\n")
                            f.write(f"Flow Value: {flow_value}\n")
                            f.write(f"Execution Time: {execution_time:.4f}s\n")
                            f.write("-" * 50 + "\n")

                        print(f"Result: Flow = {flow_value}, Time = {execution_time:.4f}s")
                        
                        # Update and display running summary
                        if len(results) > 0:
                            df = pd.DataFrame(results)
                            current_summary = df.groupby(['Implementation', 'Algorithm']).agg({
                                'Execution Time': ['count', 'mean', 'std', 'min', 'max']
                            }).round(4)
                            
                            print("\nRunning Summary:")
                            print(current_summary)

                    except Exception as e:
                        error_msg = f"Error with {impl_name} {algo_name}: {str(e)}"
                        print(error_msg)
                        
                        # Log error
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"\nERROR in test #{completed + 1}\n")
                            f.write(f"{error_msg}\n")
                            f.write(f"Stack trace:\n{traceback.format_exc()}\n")
                            f.write("-" * 50 + "\n")
                    
                    completed += 1
                    print(f"Progress: {completed}/{total_tests} tests completed")

    # Write final results and summaries
    if results:
        df_results = pd.DataFrame(results)
        df_results['Flow Value'] = pd.to_numeric(df_results['Flow Value'], errors='coerce')
        
        # Create execution time statistics
        execution_stats = df_results.groupby(['Implementation', 'Algorithm'])['Execution Time'].agg([
            ('Number of Tests', 'count'),
            ('Mean Time (s)', 'mean'),
            ('Std Dev (s)', 'std'),
            ('Min Time (s)', 'min'),
            ('Max Time (s)', 'max')
        ]).round(4)
        
        # Create flow statistics
        flow_stats = df_results.groupby(['Implementation', 'Algorithm'])['Flow Value'].agg([
            ('Mean Flow', 'mean'),
            ('Min Flow', 'min'),
            ('Max Flow', 'max')
        ]).round(2)
        
        # Create path statistics
        path_stats = df_results.groupby(['Implementation', 'Algorithm']).agg({
            'Num Paths': ['mean', 'min', 'max'],
            'Num Transfers': ['mean', 'min', 'max']
        }).round(2)
        
        # Write detailed summary to log
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\nFinal Summary:\n")
            f.write("=" * 50 + "\n")
            f.write("\nExecution Time Statistics:\n")
            f.write(execution_stats.to_string())
            f.write("\n\nFlow Statistics:\n")
            f.write(flow_stats.to_string())
            f.write("\n\nPath Statistics:\n")
            f.write(path_stats.to_string())
            f.write("\n\nPerformance Rankings:\n")
            f.write("-" * 50 + "\n")
            
            # Add rankings
            rankings = df_results.groupby(['Implementation', 'Algorithm'])['Execution Time'].mean().sort_values()
            for i, (idx, val) in enumerate(rankings.items(), 1):
                impl, algo = idx
                f.write(f"{i}. {impl} - {algo}: {val:.4f}s\n")
        
        # Save individual statistics files
        stats_dir = os.path.join(output_dir, f'comparison_stats_{timestamp}')
        os.makedirs(stats_dir, exist_ok=True)
        
        execution_stats.to_csv(os.path.join(stats_dir, 'execution_stats.csv'))
        flow_stats.to_csv(os.path.join(stats_dir, 'flow_stats.csv'))
        path_stats.to_csv(os.path.join(stats_dir, 'path_stats.csv'))
        
        # Display results
        print("\nExecution Time Statistics:")
        print(execution_stats)
        print("\nFlow Statistics:")
        print(flow_stats)
        print("\nTop Performers (by average execution time):")
        print(rankings.head())
        
        print(f"\nFiles saved:")
        print(f"- Detailed results: {results_file}")
        print(f"- Statistics: {stats_dir}/")
        print(f"- Log file: {log_file}")
        
        #return {
        #    'results': results_file,
        #    'stats_dir': stats_dir,
        #    'log': log_file
        #}
    else:
        print("No valid results to display.")
        #return None

if __name__ == "__main__":
    main()