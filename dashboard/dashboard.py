import panel as pn
import time
from typing import Optional, Dict, Any

from .csv_component import CSVDataSourceComponent
from .postgres_component import PostgresManualComponent, PostgresEnvComponent
from .analysis_component import AnalysisComponent 
from .visualization_component import VisualizationComponent
from .spark_network_visualization import SparkVisualizationComponent
from src.graph_manager import GraphManager
from src.graph import NetworkXGraph, GraphToolGraph

class NetworkFlowDashboard:
    def __init__(self, enable_spark: bool = True):
        # Initialize components
        self.data_sources = {
            "CSV": CSVDataSourceComponent(),
            "PostgreSQL (Manual)": PostgresManualComponent(),
            "PostgreSQL (ENV)": PostgresEnvComponent()
        }
        self.analysis = AnalysisComponent()
        
        # Use either Spark or regular visualization
        self.visualization = (SparkVisualizationComponent() if enable_spark 
                            else VisualizationComponent())
                            
        self.graph_manager = None

        # Create the data source tabs
        self._create_data_source_tabs()
        # Set up component callbacks
        self._setup_callbacks()

    def _initialize_spark(self):
        """Initialize Spark if not already initialized."""
        if not self.spark_initialized:
            try:
                from pyspark.sql import SparkSession
                
                # Create Spark session with GraphFrames
                self.spark = SparkSession.builder \
                    .appName("NetworkVisualization") \
                    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
                    .getOrCreate()
                    
                self.spark_initialized = True
                print("Spark initialized successfully")
                
            except Exception as e:
                print(f"Warning: Could not initialize Spark: {str(e)}")
                self.spark_initialized = False

    def _create_data_source_tabs(self):
        """Create tabs for different data sources."""
        self.tab_names = list(self.data_sources.keys())
        tab_contents = [(name, component.view()) 
                       for name, component in self.data_sources.items()]
        
        self.data_source_tabs = pn.Tabs(
            *tab_contents,
            sizing_mode='stretch_width'
        )

    def _setup_callbacks(self):
        """Set up callbacks for components and tabs."""
        # Analysis component callbacks
        if hasattr(self.analysis, 'init_button'):
            self.analysis.init_button.on_click(self._initialize_graph_manager)
        
        if hasattr(self.analysis, 'run_button'):
            self.analysis.run_button.on_click(self._run_analysis)
        
        # Tab change callback
        if hasattr(self.data_source_tabs, 'param'):
            self.data_source_tabs.param.watch(self._handle_tab_change, 'active')

    def _handle_tab_change(self, event):
        """Handle data source tab changes."""
        # Reset graph manager when data source changes
        self.graph_manager = None
        if hasattr(self.analysis, 'update_status'):
            self.analysis.update_status("Data source changed. Please initialize graph.", 'warning')

    def _get_active_data_source(self):
        """Get the currently active data source component."""
        try:
            active_index = self.data_source_tabs.active
            active_name = self.tab_names[active_index]
            return self.data_sources[active_name]
        except Exception as e:
            print(f"Error getting active data source: {str(e)}")
            return self.data_sources["CSV"]  # Default to CSV if there's an error


    def _initialize_graph_manager(self, event):
        """Initialize the graph manager with the current data source configuration."""
        try:
            self.analysis.init_status.object = "Initializing graph..."
            self.analysis.init_status.styles = {'color': 'blue'}
            
            active_source = self._get_active_data_source()
            config = active_source.get_configuration()
            
            if not config:
                raise ValueError("No valid configuration available")
            
            self.graph_manager = GraphManager(
                config,
                'networkx' if self.analysis.graph_library == 'NetworkX' else 'graph_tool'
            )
            
            # Initialize visualization with the graph manager
            self.visualization.initialize_graph(self.graph_manager)
            
            self.analysis.init_status.object = "Graph initialized successfully"
            self.analysis.init_status.styles = {'color': 'green'}
            self.analysis.enable_analysis_inputs(True)
            
        except Exception as e:
            error_msg = f"Initialization Error: {str(e)}"
            print(f"Detailed error: {str(e)}")
            self.analysis.init_status.object = error_msg
            self.analysis.init_status.styles = {'color': 'red'}
            self.graph_manager = None
            self.analysis.enable_analysis_inputs(False)


    def _run_analysis(self, event):
        """Run the flow analysis with current configuration."""
        if not self.graph_manager:
            self.analysis.update_status("Please initialize the graph first", 'warning')
            return

        try:
            # Update status
            self.analysis.update_status("Running analysis...", 'progress')
            
            # Get analysis parameters
            source = self.analysis.source
            sink = self.analysis.sink
            flow_func = self.analysis.get_algorithm_func()
            cutoff = self.analysis.requested_flow_mCRC or None

            # Validate addresses
            if not (source and sink):
                raise ValueError("Please provide both source and sink addresses")

            # Run analysis with timing
            start_time = time.time()
            self.results = self.graph_manager.analyze_flow(
                source=source,
                sink=sink,
                flow_func=flow_func,
                cutoff=cutoff
            )
            computation_time = time.time() - start_time

            # Update visualization with Spark support if applicable
            self.update_results_view(computation_time)
            self.analysis.update_status("Analysis completed successfully", 'success')

        except Exception as e:
            error_msg = f"Analysis Error: {str(e)}"
            print(f"Detailed error: {str(e)}")
            self.analysis.update_status(error_msg, 'error')
            self.results = None
            self.update_results_view(0)

    def _create_sidebar(self):
        """Create the dashboard sidebar."""
        return pn.Column(
            pn.Accordion(
                ("Data Source Configuration", pn.Column(
                    self.data_source_tabs,
                    sizing_mode='stretch_width',
                    margin=(0, 10)
                )),
                active=[0],
                sizing_mode='stretch_width',
                margin=(5, 0)
            ),
            self.analysis.view(),
            margin=(10, 10, 20, 10),
            min_height=800,
            sizing_mode='stretch_width',
            scroll=True
        )


    def update_results_view(self, computation_time: float):
        """Update the visualization and results view."""
        if self.results is None:
            self.visualization.update_view(None, computation_time, None, None)
            return

        try:
            # Update visualization component with the results
            self.visualization.update_view(
                results=self.results,
                computation_time=computation_time,
                graph_manager=self.graph_manager,
                algorithm=self.analysis.algorithm
            )

        except Exception as e:
            error_msg = f"Error updating visualizations: {str(e)}"
            print(error_msg)
            print(f"Detailed error: {str(e)}")
            self.analysis.update_status(error_msg, 'error')
            self.visualization.update_view(None, 0, None, None)

    def view(self):
        """Create and return the dashboard view."""
        # Create main content with visualization
        main_content = pn.Column(
            self.visualization.view(),
            margin=(10, 10, 10, 10),
            sizing_mode='stretch_both'
        )

        # Create template with styling
        template = pn.template.MaterialTemplate(
            title="pyFinder: Network Flow Analysis Dashboard",
            header_background="#007BFF",
            header_color="#ffffff",
            sidebar=self._create_sidebar(),
            main=main_content,
            sidebar_width=500
        )

        return template
    
    def cleanup(self):
        """Clean up resources including Spark."""
        if hasattr(self, 'visualization'):
            self.visualization.cleanup()
        if hasattr(self, 'spark') and self.spark_initialized:
            self.spark.stop()

def create_dashboard(enable_spark: bool = True):
    """Create and return a new dashboard instance with optional Spark support."""
    dashboard = NetworkFlowDashboard(enable_spark=enable_spark)
    return dashboard.view()

if __name__ == "__main__":
    # This is handled in run.py
    pass
