from typing import List, Tuple, Dict, Optional
import panel as pn
from .visualization_component import VisualizationComponent
from .spark_network_visualization import SparkVisualizationComponent

class SparkEnhancedVisualization(VisualizationComponent):
    """Enhanced visualization component with Spark capabilities."""
    
    def __init__(self, **params):
        super().__init__(**params)
        self.spark_viz = None
        self._create_spark_components()
        
    def _create_spark_components(self):
        """Create Spark-specific UI components."""
        self.use_spark = pn.widgets.Checkbox(
            name="Use Spark for Large Networks",
            value=False
        )
        
        self.spark_status = pn.pane.Markdown(
            "Spark visualization disabled",
            styles={'color': 'gray'}
        )
        
        # Add Spark components to network controls
        if hasattr(self, 'network_content'):
            self.network_content.insert(
                2,  # Insert after existing controls
                pn.Row(
                    self.use_spark,
                    self.spark_status,
                    align='center',
                    margin=(10, 10)
                )
            )
    
    def initialize_graph(self, graph_manager):
        """Initialize with a graph manager, potentially using Spark for large graphs."""
        super().initialize_graph(graph_manager)
        
        # Initialize Spark visualization if graph is large
        if isinstance(graph_manager.graph, NetworkXGraph):
            num_nodes = graph_manager.graph.g_nx.number_of_nodes()
        else:  # GraphToolGraph
            num_nodes = graph_manager.graph.g_gt.num_vertices()
            
        if num_nodes > 1000:
            self.use_spark.value = True
            self._initialize_spark_visualization()
    
    def _initialize_spark_visualization(self):
        """Initialize Spark visualization component."""
        try:
            if self.spark_viz is None:
                self.spark_viz = SparkVisualizationComponent()
                self.spark_status.object = "Spark visualization ready"
                self.spark_status.styles = {'color': 'green'}
        except Exception as e:
            self.spark_status.object = f"Error initializing Spark: {str(e)}"
            self.spark_status.styles = {'color': 'red'}
            self.use_spark.value = False
    
    def update_view(self, results, computation_time, graph_manager, algorithm):
        """Update visualization, potentially using Spark for large networks."""
        if not self.use_spark.value or self.spark_viz is None:
            # Use regular visualization
            super().update_view(results, computation_time, graph_manager, algorithm)
            return
            
        try:
            # Extract edges for Spark processing
            if isinstance(graph_manager.graph, NetworkXGraph):
                edges = list(graph_manager.graph.g_nx.edges())
            else:  # GraphToolGraph
                edges = [(str(graph_manager.graph.vertex_id[e.source()]), 
                         str(graph_manager.graph.vertex_id[e.target()]))
                        for e in graph_manager.graph.g_gt.edges()]
                        
            # Create node attributes dictionary
            node_attrs = {}
            if results is not None:
                flow_value, simplified_paths, simplified_edge_flows, _ = results
                
                # Add flow information to node attributes
                for path in simplified_paths:
                    for node in path[0]:  # path nodes
                        if node not in node_attrs:
                            node_attrs[node] = {'flow_value': flow_value}
            
            # Update Spark visualization
            self.spark_viz.update_visualization(edges, node_attrs)
            
            # Update statistics
            self._update_path_stats(
                flow_value=results[0] if results else 0,
                simplified_paths=results[1] if results else [],
                simplified_edge_flows=results[2] if results else {},
                original_edge_flows=results[3] if results else {},
                computation_time=computation_time,
                algorithm=algorithm
            )
            
        except Exception as e:
            self.spark_status.object = f"Error updating Spark visualization: {str(e)}"
            self.spark_status.styles = {'color': 'red'}
            # Fallback to regular visualization
            super().update_view(results, computation_time, graph_manager, algorithm)
    
    def cleanup(self):
        """Clean up resources including Spark."""
        if self.spark_viz is not None:
            self.spark_viz.cleanup()
