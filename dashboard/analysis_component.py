import panel as pn
import param
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
    boykov_kolmogorov_max_flow as gt_boykov_kolmogorov,
)
from .base_component import BaseComponent

class AnalysisComponent(BaseComponent):
    # Graph library selection
    graph_library = param.Selector(default="NetworkX", objects=["NetworkX", "graph-tool", "OR-Tools"])

    # Analysis parameters (initially disabled)
    source = param.String(default="0x3fb47823a7c66553fb6560b75966ef71f5ccf1d0")
    sink = param.String(default="0xe98f0672a8e31b408124f975749905f8003a2e04")
    requested_flow_mCRC = param.String(default="")
    algorithm = param.Selector(default="Preflow Push", objects=[
        "Preflow Push",
        "Edmonds-Karp",
        "Shortest Augmenting Path",
        "Boykov-Kolmogorov",
        "Dinitz"
    ])

    def __init__(self, **params):
        super().__init__(**params)
        self.param.watch(self._update_algorithm_list, 'graph_library')

    def _create_components(self):
        """Create UI components."""
        # Library selection
        self.library_select = pn.widgets.Select(
            name="Graph Library",
            options=["NetworkX", "graph-tool", "OR-Tools"],
            value=self.graph_library,
            width=200
        )

        # Status indicators
        self.init_status = pn.pane.Markdown(
            "Select library and initialize graph",
            styles={'color': 'gray', 'font-style': 'italic'}
        )

        # Create implementation status BEFORE using it
        self.implementation_status = pn.pane.Markdown(
            "",  # Initial empty status
            styles={'color': 'gray', 'font-style': 'italic'}
        )

        self.compute_status = pn.pane.Markdown(
            "Configure analysis parameters",
            styles={'color': 'gray', 'font-style': 'italic'}
        )

        # Initialize button
        self.init_button = pn.widgets.Button(
            name="Initialize Graph",
            button_type="primary",
            sizing_mode="fixed",
            width=200,
            height=40
        )

        # Analysis inputs (initially disabled)
        self.source_input = pn.widgets.TextInput(
            name="Source Address",
            value=self.source,
            disabled=True
        )
        
        self.sink_input = pn.widgets.TextInput(
            name="Sink Address",
            value=self.sink,
            disabled=True
        )
        
        self.flow_input = pn.widgets.TextInput(
            name="Requested Flow (mCRC)",
            value=self.requested_flow_mCRC,
            placeholder="Leave empty for max flow",
            disabled=True
        )

        # Update algorithm options based on library
        algorithms = self.get_algorithm_list()
        self.algorithm_select = pn.widgets.Select(
            name="Algorithm",
            options=algorithms,
            value=algorithms[0],
            width=200,
            disabled=True
        )

        self.run_button = pn.widgets.Button(
            name="Run Analysis",
            button_type="primary",
            width=200,
            height=40,
            disabled=True
        )


       # self._update_implementation_status(self.graph_library)

    def _setup_callbacks(self):
        """Set up component callbacks."""
        # Link widgets to parameters
        self.library_select.link(self, value='graph_library')
        self.source_input.link(self, value='source')
        self.sink_input.link(self, value='sink')
        self.flow_input.link(self, value='requested_flow_mCRC')
        self.algorithm_select.link(self, value='algorithm')

       # self.library_select.param.watch(
       #     lambda event: self._update_implementation_status(event.new), 
       #     'value'
       # )

    def _update_implementation_status(self, library: str):
        """Update the implementation-specific status message."""
        status_messages = {
            'NetworkX': "Using NetworkX implementation (Python-based, good for prototyping)",
            'graph-tool': "Using graph-tool implementation (C++/Python, high performance)",
            'OR-Tools': "Using Google OR-Tools implementation (Industrial-grade solver)"
        }
        
        status_colors = {
            'NetworkX': '#1f77b4',  # Blue
            'graph-tool': '#2ca02c',  # Green
            'OR-Tools': '#ff7f0e'   # Orange
        }
        
        self.implementation_status.object = status_messages.get(library, "")
        self.implementation_status.styles = {
            'color': status_colors.get(library, 'gray'),
            'font-style': 'normal',
            'font-weight': 'bold'
        }

    def enable_analysis_inputs(self, enable=True):
        """Enable or disable analysis inputs."""
        self.source_input.disabled = not enable
        self.sink_input.disabled = not enable
        self.flow_input.disabled = not enable
        self.algorithm_select.disabled = not enable
        self.run_button.disabled = not enable

    def _update_algorithm_list(self, event):
        """Update available algorithms based on selected graph library."""
        algorithms = self.get_algorithm_list()
        self.param.algorithm.objects = algorithms
        self.algorithm = algorithms[0]
        if hasattr(self, 'algorithm_select'):
            self.algorithm_select.options = algorithms
            self.algorithm_select.value = algorithms[0]

        #if hasattr(self, 'implementation_status'):
        #    self._update_implementation_status(event.new)


    def get_algorithm_list(self):
        """Get list of algorithms based on selected graph library."""
        if self.graph_library == 'NetworkX':
            return [
                "Preflow Push",
                "Edmonds-Karp",
                "Shortest Augmenting Path",
                "Boykov-Kolmogorov",
                "Dinitz"
            ]
        elif self.graph_library == 'graph-tool':
            return [
               # "Push-Relabel",
                "Edmonds-Karp",
                "Boykov-Kolmogorov"
            ]
        else:  # OR-Tools
            return ["OR-Tools Max Flow"]  # OR-Tools uses its own algorithm

    def get_algorithm_func(self):
        """Get the algorithm function based on selected algorithm."""
        if self.graph_library == 'NetworkX':
            algorithm_map = {
                "Preflow Push": preflow_push,
                "Edmonds-Karp": edmonds_karp,
                "Shortest Augmenting Path": shortest_augmenting_path,
                "Boykov-Kolmogorov": boykov_kolmogorov,
                "Dinitz": dinitz
            }
            return algorithm_map[self.algorithm]
        elif self.graph_library == 'graph-tool':
            algorithm_map = {
              #  "Push-Relabel": gt_push_relabel,
                "Edmonds-Karp": gt_edmonds_karp,
                "Boykov-Kolmogorov": gt_boykov_kolmogorov
            }
            return algorithm_map[self.algorithm]
        else:  # OR-Tools
            return None 

    def update_status(self, message: str, status_type: str = 'info'):
        """Update the compute status message."""
        color_map = {
            'info': 'gray',
            'success': 'green',
            'error': 'red',
            'warning': 'orange',
            'progress': 'blue'
        }
        self.compute_status.object = message
        self.compute_status.styles = {
            'color': color_map.get(status_type, 'gray'),
            'font-style': 'italic' if status_type == 'info' else 'normal'
        }

    def view(self):
        """Return the component's view with proper spacing."""
        return pn.Accordion(
            ("Analysis Configuration", pn.Column(
                pn.Row(
                    self.library_select,
                    self.init_button,
                    align='center',
                    margin=(0, 10)
                ),
                self.init_status,
                self.implementation_status,  
                pn.layout.Divider(),
                self.source_input,
                self.sink_input,
                self.flow_input,
                self.algorithm_select,
                pn.Row(
                    self.run_button,
                    align='center'
                ),
                self.compute_status,
                margin=(5, 10),
                sizing_mode='stretch_width'
            )),
            active=[0],
            sizing_mode='stretch_width',
            margin=(5, 0)
        )