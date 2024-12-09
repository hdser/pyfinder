import panel as pn
from typing import Optional
import time
import os
from dotenv import load_dotenv


from .analysis_component import AnalysisComponent 
from .visualization_component import VisualizationComponent
from src.graph import NetworkXGraph, GraphToolGraph, ORToolsGraph
from src.graph_manager import GraphManager

class AutoConfigDashboard:
    def __init__(self):
        # Initialize components
        self.analysis = AnalysisComponent()
        self.visualization = VisualizationComponent()
        self.graph_manager = None
        
        # Set up component callbacks
        self._setup_callbacks()

    def _setup_callbacks(self):
        """Set up callbacks for components."""
        if hasattr(self.analysis, 'init_button'):
            self.analysis.init_button.on_click(self._initialize_graph_manager)
        
        if hasattr(self.analysis, 'run_button'):
            self.analysis.run_button.on_click(self._run_analysis)

    def _load_postgres_config(self):
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

    def _initialize_graph_manager(self, event):
        """Initialize the graph manager with PostgreSQL configuration."""
        try:
            self.analysis.init_status.object = "Initializing graph..."
            self.analysis.init_status.styles = {'color': 'blue'}
            
            # Get PostgreSQL configuration from .env
            db_config = self._load_postgres_config()
            queries_dir = "queries"  # Default queries directory
            
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
            self.graph_manager = GraphManager((db_config, queries_dir), graph_type)
            print("Initialize graph manager time: ", time.time()-start)

            start = time.time()
            # Initialize visualization
            self.visualization.initialize_graph(self.graph_manager)
            print("Initialize visualization time: ", time.time()-start)
            
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
            requested_flow = self.analysis.requested_flow_mCRC if str(self.analysis.requested_flow_mCRC).strip().isdigit() else None

            # Validate addresses
            if not (source and sink):
                raise ValueError("Please provide both source and sink addresses")

            # Run analysis with timing
            start_time = time.time()
            results = self.graph_manager.analyze_flow(
                source=source,
                sink=sink,
                flow_func=flow_func,
                cutoff=requested_flow
            )
            computation_time = time.time() - start_time

            # Update visualization
            flow_value, simplified_paths, simplified_edge_flows, original_edge_flows = results
            
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

            self.visualization.update_view(
                results=results,
                computation_time=computation_time,
                graph_manager=self.graph_manager,
                algorithm=self.analysis.algorithm
            )

            # Update status
            if requested_flow is None:
                self.analysis.update_status(f"Maximum flow analysis completed successfully", 'success')
            else:
                self.analysis.update_status(f"Flow analysis completed successfully", 'success')

        except Exception as e:
            error_msg = f"Analysis Error: {str(e)}"
            print(f"Detailed error: {str(e)}")
            self.analysis.update_status(error_msg, 'error')
            self.visualization.update_view()

    def view(self):
        """Create and return the dashboard view."""
        # Create main content with visualization
        main_content = pn.Column(
            self.visualization.view(),
            margin=(10, 10, 10, 10),
            sizing_mode='stretch_both'
        )

        # Create sidebar with just analysis section
        sidebar = pn.Column(
            self.analysis.view(),
            margin=(10, 10, 20, 10),
            min_height=800,
            sizing_mode='stretch_width',
            scroll=True
        )

        # Create template
        template = pn.template.MaterialTemplate(
            title="pyFinder: Path Finder Dashboard",
            header_background="#007BFF",
            header_color="#ffffff",
            sidebar=sidebar,
            main=main_content,
            sidebar_width=500,
            css_files=[
                'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
            ]
        )

        return template

def create_mini_dashboard():
    """Return a function that creates a new dashboard instance per session."""
    def dashboard_view():
        dashboard = AutoConfigDashboard()
        return dashboard.view()
    return dashboard_view