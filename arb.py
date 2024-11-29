import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from pathlib import Path
import time
from collections import defaultdict
import logging
from argparse import ArgumentParser

from src.graph_manager import GraphManager
from src.data_ingestion import DataIngestion
from src.visualization import Visualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class ArbitrageAnalyzer:
    def __init__(self, trust_file: str, balance_file: str, output_dir: str, implementation: str = None):
        self.trust_file = Path(trust_file)
        self.balance_file = Path(balance_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.implementation = implementation
        
        # Initialize single graph manager
        self.manager = None
        if implementation:
            self.initialize_graph_manager(implementation)
    
    def initialize_graph_manager(self, implementation: str):
        """Initialize graph manager for specified implementation."""
        try:
            logger.info(f"Initializing {implementation} implementation...")
            start_time = time.time()
            
            self.manager = GraphManager(
                data_source=(str(self.trust_file), str(self.balance_file)),
                graph_type=implementation
            )
            
            duration = time.time() - start_time
            logger.info(f"Successfully created {implementation} implementation in {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error creating {implementation} implementation: {str(e)}")
            raise

    def print_network_stats(self):
        """Print network statistics for current implementation."""
        if not self.manager:
            logger.error("No graph implementation available")
            return
            
        graph = self.manager.graph
        logger.info(f"\nNetwork statistics ({self.implementation} implementation):")
        
        # Basic counts
        real_nodes = {v for v in graph.get_vertices() if '_' not in v}
        intermediate_nodes = {v for v in graph.get_vertices() if '_' in v}
        
        logger.info(f"Total nodes: {graph.num_vertices():,}")
        logger.info(f"Real nodes: {len(real_nodes):,}")
        logger.info(f"Intermediate nodes: {len(intermediate_nodes):,}")
        logger.info(f"Total edges: {graph.num_edges():,}")
        
        # Degree statistics
        degrees = [graph.degree(v) for v in real_nodes]
        logger.info("\nDegree statistics (real nodes):")
        logger.info(f"Average degree: {np.mean(degrees):.2f}")
        logger.info(f"Median degree: {np.median(degrees):.2f}")
        logger.info(f"Max degree: {max(degrees):,}")
        logger.info(f"Min degree: {min(degrees):,}")

    def analyze_arbitrage(self, source: str, start_token: str, end_token: str):
        """Analyze arbitrage opportunities using current implementation."""
        if not self.manager:
            logger.error("No graph implementation available")
            return
            
        logger.info(f"\nAnalyzing arbitrage using {self.implementation}...")
        logger.info(f"Source: {source}")
        logger.info(f"Start token: {start_token}")
        logger.info(f"End token: {end_token}")
        
        try:
            start_time = time.time()
            
            # Run arbitrage analysis
            flow_value, paths, simplified_flows, original_flows = self.manager.analyze_arbitrage(
                source=source,
                start_token=start_token,
                end_token=end_token
            )
            computation_time = time.time() - start_time
            
            # Log results
            logger.info(f"Analysis completed in {computation_time:.2f} seconds")
            logger.info(f"Total arbitrage flow: {flow_value:,} mCRC")
            
            if not paths:
                logger.info("No arbitrage paths found")
                return None
                
            logger.info(f"\nFound {len(paths)} distinct arbitrage paths:")
            
            results = {
                'computation_time': computation_time,
                'flow_value': flow_value,
                'paths': []
            }
            
            for i, (path, tokens, amount) in enumerate(paths, 1):
                path_info = {
                    'path': path,
                    'tokens': tokens,
                    'amount': amount,
                    'path_length': len(path),
                    'unique_tokens': len(set(tokens))
                }
                results['paths'].append(path_info)
                
                logger.info(f"\nPath {i}:")
                logger.info(f"Nodes: {' -> '.join(path)}")
                logger.info(f"Tokens: {' -> '.join(tokens)}")
                logger.info(f"Amount: {amount:,} mCRC")
                logger.info(f"Path length: {path_info['path_length']}")
                logger.info(f"Unique tokens: {path_info['unique_tokens']}")
            
            # Save visualizations with source address
            self._save_visualizations(paths, simplified_flows, source)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing arbitrage: {str(e)}")
            return None

    def _save_visualizations(self, paths: List[Tuple], flows: Dict, source_address: str):
        """Save visualizations of arbitrage paths."""
        try:
            # Save path visualizations
            vis_file = self.output_dir / 'arbitrage_paths.png'
            self.manager.visualization.plot_flow_paths(
                self.manager.graph,
                paths,
                flows,
                self.manager.data_ingestion.id_to_address,
                str(vis_file),
                source_address
            )
            logger.info(f"Saved path visualization to {vis_file}")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")


def configure_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    file_handler = logging.FileHandler('arbitrage_analysis.log')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)

def main():
    configure_logging()
        
    parser = ArgumentParser(description='Analyze arbitrage opportunities in Circles UBI network')
    parser.add_argument('--trust-file', type=str, required=True,
                       help='Path to trust relationships CSV file')
    parser.add_argument('--balance-file', type=str, required=True,
                       help='Path to account balances CSV file')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Directory for output files')
    parser.add_argument('--implementation', type=str, 
                       choices=['networkx', 'graph_tool', 'ortools'],
                       required=True,
                       help='Implementation to use')
    parser.add_argument('--source', type=str,
                       help='Source address for arbitrage analysis')
    parser.add_argument('--start-token', type=str,
                       help='Starting token address')
    parser.add_argument('--end-token', type=str,
                       help='Target token address')
    
    args = parser.parse_args()
    
    # Initialize analyzer with specific implementation
    analyzer = ArbitrageAnalyzer(
        args.trust_file,
        args.balance_file,
        args.output_dir,
        args.implementation
    )
    
    # Print network statistics
    #analyzer.print_network_stats()
    
    # Run arbitrage analysis if addresses provided
    if all([args.source, args.start_token, args.end_token]):
        analyzer.analyze_arbitrage(
            args.source,
            args.start_token,
            args.end_token
        )

if __name__ == '__main__':
    main()