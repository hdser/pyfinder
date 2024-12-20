import panel as pn
import time

from .csv_component import CSVDataSourceComponent
from .postgres_component import (
    PostgresManualComponent,
    PostgresEnvComponent
)
from .analysis_component import AnalysisComponent 
from .visualization_component import VisualizationComponent
from src.graph_manager import GraphManager
from src.graph import NetworkXGraph, GraphToolGraph, ORToolsGraph

# Move the CSS definition here, at the module level
pn.config.raw_css.append("""
.network-section {
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    padding: 15px;
    margin-bottom: 20px;
}

.paths-section {
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    padding: 15px;
    margin-bottom: 20px;
}

.transactions-section {
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    padding: 15px;
    margin-bottom: 20px;
}

.section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 15px;
}

.stats-panel {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 5px;
    border: 1px solid #dee2e6;
    margin-bottom: 15px;
}

.control-panel {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px;
    background: #f8f9fa;
    border-radius: 5px;
    margin-bottom: 15px;
}
""")

class NetworkFlowDashboard:
    def __init__(self):
        # Initialize components
        self.data_sources = {
            "CSV": CSVDataSourceComponent(),
            "PostgreSQL (Manual)": PostgresManualComponent(),
            "PostgreSQL (ENV)": PostgresEnvComponent()
        }

        self.analysis = AnalysisComponent()
        self.visualization = VisualizationComponent()
        self.graph_manager = None
        
        # Create the data source tabs
        self._create_data_source_tabs()
        # Set up component callbacks
        self._setup_callbacks()

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
            
            # Map graph library selection to implementation
            graph_type_map = {
                'NetworkX': 'networkx',
                'graph-tool': 'graph_tool',
                'OR-Tools': 'ortools'
            }
            
            graph_type = graph_type_map.get(self.analysis.graph_library)
            if not graph_type:
                raise ValueError(f"Unsupported graph library: {self.analysis.graph_library}")
            
            start = time.time()
            # Initialize graph manager with selected implementation
            self.graph_manager = GraphManager(config, graph_type)
            print("Initialize graph manager time: ",time.time()-start)

            start = time.time()
            # Initialize visualization
            self.visualization.initialize_graph(self.graph_manager)
            print("Initialize visualization time: ",time.time()-start)
            
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
            # Update status through analysis component
            self.analysis.update_status("Running analysis...", 'progress')
            
            # Get analysis parameters
            source = self.analysis.source
            sink = self.analysis.sink
            flow_func = self.analysis.get_algorithm_func()
            
            # Empty string means max flow (no cutoff)
            requested_flow = self.analysis.requested_flow_mCRC if str(self.analysis.requested_flow_mCRC).strip().isdigit() else None

            # Validate addresses
            if not (source and sink):
                raise ValueError("Please provide both source and sink addresses")

            # Run analysis with timing
            start_time = time.time()
            
            # When requested_flow is None, compute_flow will calculate maximum flow
            results = self.graph_manager.analyze_flow(
                source=source,
                sink=sink,
                flow_func=flow_func,
                cutoff=requested_flow  # None means compute max flow
            )
            computation_time = time.time() - start_time

            # Update visualization with results
            flow_value, simplified_paths, simplified_edge_flows, original_edge_flows = results
            
            # Update path stats directly - pass None for requested_flow to indicate max flow calculation
            self.visualization._update_path_stats(
                source=source,
                sink=sink,
                flow_value=flow_value,
                simplified_paths=simplified_paths,
                simplified_edge_flows=simplified_edge_flows,
                original_edge_flows=original_edge_flows,
                computation_time=computation_time,
                algorithm=self.analysis.algorithm,
                requested_flow=requested_flow
            )

            # Update visualization view
            self.visualization.update_view(
                results=results,
                computation_time=computation_time,
                graph_manager=self.graph_manager,
                algorithm=self.analysis.algorithm
            )

            # Update status with success message
            if requested_flow is None:
                self.analysis.update_status(f"Maximum flow analysis completed successfully", 'success')
            else:
                self.analysis.update_status(f"Flow analysis completed successfully", 'success')

        except Exception as e:
            error_msg = f"Analysis Error: {str(e)}"
            print(f"Detailed error: {str(e)}")
            self.analysis.update_status(error_msg, 'error')
            self.visualization.update_view()

    def _create_sidebar(self):
        """Create the dashboard sidebar with proper spacing."""
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

    def update_results_view(self, computation_time):
        """Update the UI with analysis results"""
        if self.results is None:
            self.visualization.update_view(None, 0, None, None)
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

        # Create template with updated styling
        template = pn.template.MaterialTemplate(
            title="pyFinder: Path Finder Dashboard",
            header_background="#007BFF",
            header_color="#ffffff",
            sidebar=self._create_sidebar(),
            main=main_content,
            sidebar_width=500,
            css_files=[
                'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
            ]
        )

        return template

def create_dashboard():
    """Return a function that creates a new dashboard instance per session."""
    def dashboard_view():
        dashboard = NetworkFlowDashboard()
        return dashboard.view()
    return dashboard_view


if __name__ == "__main__":
    # This is handled in run.py
    pass
