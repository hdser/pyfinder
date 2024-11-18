import panel as pn
import param
import os
from pathlib import Path
import dask.dataframe as dd
import pandas as pd
from typing import Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import filedialog

class CSVDataSourceComponent(param.Parameterized):
    selected_directory = param.String(default="")
    csv_trusts_file = param.String(default="")
    csv_balances_file = param.String(default="")
    validation_state = param.Dict(default={})
    is_loading = param.Boolean(default=False)
    
    def __init__(self, **params):
        super().__init__(**params)
        self.executor = ThreadPoolExecutor(max_workers=2)
        # Initialize tkinter root window for file dialog
        self.tk_root = None
        self._create_components()
        self._setup_callbacks()
    
    def _create_components(self):
        """Create all UI components."""
        self.status = pn.pane.Markdown(
            "Please select directory containing CSV files",
            styles={'color': 'gray', 'font-weight': 'bold'}
        )
        
        self.directory_input = pn.widgets.TextInput(
            name='Data Directory',
            placeholder='Enter path to directory containing CSV files',
            sizing_mode='stretch_width'
        )
        
        self.browse_button = pn.widgets.Button(
            name="Browse Directory",
            button_type="default",
            sizing_mode='fixed',
            width=150,
            height=40
        )
        
        self.default_dir_button = pn.widgets.Button(
            name="Use Default Directory",
            button_type="default",
            sizing_mode='fixed',
            width=150,
            height=40
        )
        
        self.load_button = pn.widgets.Button(
            name="Load Configuration",
            button_type="primary",
            disabled=True,
            sizing_mode='fixed',
            width=150,
            height=40
        )
        
        self.trusts_info = pn.pane.Markdown("### Trust File\nNot selected")
        self.balances_info = pn.pane.Markdown("### Balance File\nNot selected")

    def _setup_callbacks(self):
        """Set up all component callbacks."""
        self.directory_input.param.watch(self._handle_directory_change, 'value')
        self.browse_button.on_click(self._browse_directory)
        self.default_dir_button.on_click(self._use_default_directory)
        self.load_button.on_click(self._handle_load)

    def _initialize_tk_root(self):
        """Initialize the tkinter root window if not already initialized."""
        if self.tk_root is None:
            self.tk_root = tk.Tk()
            self.tk_root.withdraw()  # Hide the main window

    def _browse_directory(self, event):
        """Handle directory browse button click using tkinter's file dialog."""
        try:
            self._initialize_tk_root()
            directory = filedialog.askdirectory(
                initialdir=os.getcwd(),
                title="Select Data Directory"
            )
            if directory:  # Only update if a directory was selected
                self.directory_input.value = directory
                self._check_directory(directory)
        except Exception as e:
            self.status.object = f"Error browsing directory: {str(e)}"
            self.status.styles = {'color': 'red'}

    def _handle_directory_change(self, event):
        """Handle directory input changes."""
        if event.new:
            self._check_directory(event.new)

    def _check_directory(self, directory: str):
        """Check if directory contains required CSV files."""
        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                raise ValueError("Directory does not exist")
            if not directory_path.is_dir():
                raise ValueError("Selected path is not a directory")
            
            trust_file = directory_path / "data-trust.csv"
            balance_file = directory_path / "data-balance.csv"
            
            if not trust_file.exists() or not balance_file.exists():
                raise ValueError("Directory must contain 'data-trust.csv' and 'data-balance.csv'")
            
            self.csv_trusts_file = str(trust_file)
            self.csv_balances_file = str(balance_file)
            
            self._update_file_info()
            
            # Start validation
            self.executor.submit(self._validate_file, str(trust_file), 'trusts')
            self.executor.submit(self._validate_file, str(balance_file), 'balances')
            
            self.status.object = "Files found and being validated"
            self.status.styles = {'color': '#007bff'}
            
        except Exception as e:
            self.status.object = f"Error: {str(e)}"
            self.status.styles = {'color': 'red'}
            self.csv_trusts_file = ""
            self.csv_balances_file = ""
            self.validation_state = {}
            self._update_file_info()

    def _use_default_directory(self, event=None):
        """Use default data directory."""
        potential_locations = [
            "data",
            "../data",
            "../../data",
            os.path.join(os.path.dirname(__file__), "../../../data")
        ]
        
        for location in potential_locations:
            if os.path.exists(location):
                trust_path = os.path.join(location, "data-trust.csv")
                balance_path = os.path.join(location, "data-balance.csv")
                if os.path.exists(trust_path) and os.path.exists(balance_path):
                    location = os.path.abspath(location)
                    self.directory_input.value = location
                    self._check_directory(location)
                    return
        
        self.status.object = "Default data directory not found"
        self.status.styles = {'color': 'red'}

    def _validate_file(self, filepath: str, validation_key: str) -> bool:
        """Validate CSV file format and content."""
        try:
            # Quick format check
            pd.read_csv(filepath, nrows=5)
            
            # Verify with dask
            ddf = dd.read_csv(filepath, blocksize="64MB")
            ddf.head(1, compute=True)
            
            self.validation_state[validation_key] = True
            self._update_file_info()
            
            if all(self.validation_state.values()):
                self.status.object = "All files validated successfully"
                self.status.styles = {'color': 'green'}
            
            return True
            
        except Exception as e:
            print(f"Validation error for {validation_key}: {str(e)}")
            self.validation_state[validation_key] = False
            self._update_file_info()
            self.status.object = f"Validation failed: {str(e)}"
            self.status.styles = {'color': 'red'}
            return False

    def _update_file_info(self):
        """Update file information displays."""
        def format_file_info(filepath: str, validation_key: str) -> str:
            if not filepath:
                return "### Not selected"
            
            file = Path(filepath)
            if not file.exists():
                return "### File not found"
            
            size_mb = file.stat().st_size / (1024 * 1024)
            is_valid = self.validation_state.get(validation_key, False)
            status = "✅ Valid" if is_valid else "⚠️ Validating..."
            
            return f"""### Selected File
- Name: {file.name}
- Size: {size_mb:.2f} MB
- Status: {status}"""
        
        self.trusts_info.object = format_file_info(self.csv_trusts_file, 'trusts')
        self.balances_info.object = format_file_info(self.csv_balances_file, 'balances')
        
        # Update button state
        files_selected = bool(self.csv_trusts_file and self.csv_balances_file)
        files_valid = all(self.validation_state.values()) if self.validation_state else False
        self.load_button.disabled = not (files_selected and files_valid)

    def _handle_load(self, event):
        """Handle load button click."""
        try:
            if self.validate_configuration():
                self.status.object = "Configuration validated and ready to use"
                self.status.styles = {'color': 'green'}
            else:
                self.status.object = "Configuration validation failed"
                self.status.styles = {'color': 'red'}
        except Exception as e:
            self.status.object = f"Error validating configuration: {str(e)}"
            self.status.styles = {'color': 'red'}

    def validate_configuration(self) -> bool:
        """Validate the current configuration."""
        return bool(self.csv_trusts_file and 
                   self.csv_balances_file and 
                   all(self.validation_state.values()))

    def get_configuration(self) -> Optional[Tuple[str, str]]:
        """Get the current configuration."""
        if not self.validate_configuration():
            return None
        return (self.csv_trusts_file, self.csv_balances_file)

    def view(self):
        """Return the component's view."""
        return pn.Column(
            pn.Row(
                pn.Column(
                    "## Data Source Configuration",
                    self.status,
                    margin=(0, 0, 20, 0),
                    sizing_mode='stretch_width'
                ),
                sizing_mode='stretch_width'
            ),
            pn.Row(
                pn.Column(
                    "### Select Directory",
                    self.directory_input,
                    sizing_mode='stretch_width'
                ),
                pn.Column(
                    self.browse_button,
                    self.default_dir_button,
                    margin=(25, 0, 0, 10)
                )
            ),
            pn.Row(
                pn.Column(
                    self.trusts_info,
                    sizing_mode='stretch_width'
                ),
                pn.Column(
                    self.balances_info,
                    sizing_mode='stretch_width'
                ),
                sizing_mode='stretch_width'
            ),
            pn.Row(
                self.load_button,
                align='center',
                margin=(10, 0)
            ),
            sizing_mode='stretch_width',
            margin=(10, 10)
        )