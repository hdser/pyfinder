import panel as pn
import param
import os
import logging
from pathlib import Path
import dask.dataframe as dd
import pandas as pd
from typing import Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor

class CSVDataSourceComponent(param.Parameterized):
    selected_directory = param.String(default="")
    csv_trusts_file = param.String(default="")
    csv_balances_file = param.String(default="")
    validation_state = param.Dict(default={})
    is_loading = param.Boolean(default=False)
    
    def __init__(self, **params):
        super().__init__(**params)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.logger = logging.getLogger(__name__)
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

    def _browse_directory(self, event):
        """Handle directory browse button click."""
        try:
            # Try to use tkinter if available
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            directory = filedialog.askdirectory(
                initialdir=os.getcwd(),
                title="Select Data Directory"
            )
            if directory:  # Only update if a directory was selected
                self.directory_input.value = directory
                self._check_directory(directory)
        except Exception as e:
            self.logger.warning(f"GUI file browser not available: {str(e)}")
            # Fallback to default directory if tkinter is not available
            self._use_default_directory()
            self.status.object = "GUI file browser not available. Using default directory or enter path manually."
            self.status.styles = {'color': 'orange'}

    def _handle_directory_change(self, event):
        """Handle directory input changes."""
        if event.new:
            self._check_directory(event.new)

    def _find_data_directory(self) -> Optional[Path]:
        """
        Find the data directory by checking multiple possible locations.
        Prioritizes Docker environment paths.
        """
        # Check if running in Docker
        in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
        self.logger.info(f"Running in Docker: {in_docker}")

        # Prioritized locations for data directory
        potential_locations = []
        
        if in_docker:
            potential_locations.append(Path("/app/data"))
        
        potential_locations.extend([
            Path.cwd() / "data",
            Path.cwd().parent / "data",
            Path(__file__).parent.parent.parent / "data",
            Path.home() / "data"
        ])

        for location in potential_locations:
            self.logger.info(f"Checking data directory: {location}")
            if location.exists() and location.is_dir():
                trust_file = location / "data-trust.csv"
                balance_file = location / "data-balance.csv"
                
                if trust_file.exists() and balance_file.exists():
                    self.logger.info(f"Found valid data directory: {location}")
                    return location
                else:
                    self.logger.warning(f"Directory exists but missing required files: {location}")
            else:
                self.logger.debug(f"Directory not found: {location}")
        
        return None

    def _use_default_directory(self, event=None):
        """Use default data directory with improved logging and error handling."""
        try:
            data_dir = self._find_data_directory()
            if data_dir:
                self.logger.info(f"Using default data directory: {data_dir}")
                self.directory_input.value = str(data_dir)
                self._check_directory(str(data_dir))
            else:
                error_msg = (
                    "Default data directory not found. Please ensure data files exist in one of:\n"
                    "- /app/data (Docker)\n"
                    "- ./data\n"
                    "- ../data\n"
                    "- Project root /data"
                )
                self.logger.error(error_msg)
                self.status.object = error_msg
                self.status.styles = {'color': 'red'}
        except Exception as e:
            self.logger.exception("Error using default directory")
            self.status.object = f"Error accessing default directory: {str(e)}"
            self.status.styles = {'color': 'red'}

    def _check_directory(self, directory: str):
        """Check if directory contains required CSV files with improved error handling."""
        try:
            directory_path = Path(directory)
            self.logger.info(f"Checking directory: {directory_path}")
            
            if not directory_path.exists():
                raise ValueError(f"Directory does not exist: {directory_path}")
            if not directory_path.is_dir():
                raise ValueError(f"Not a directory: {directory_path}")
            
            trust_file = directory_path / "data-trust.csv"
            balance_file = directory_path / "data-balance.csv"
            
            missing_files = []
            if not trust_file.exists():
                missing_files.append("data-trust.csv")
            if not balance_file.exists():
                missing_files.append("data-balance.csv")
                
            if missing_files:
                raise ValueError(f"Missing required files: {', '.join(missing_files)}")
            
            # Check file permissions
            try:
                with open(trust_file) as f:
                    f.read(1)
                with open(balance_file) as f:
                    f.read(1)
            except PermissionError:
                raise ValueError("Permission denied accessing data files")
            
            self.csv_trusts_file = str(trust_file)
            self.csv_balances_file = str(balance_file)
            
            self._update_file_info()
            
            # Start validation
            self.executor.submit(self._validate_file, str(trust_file), 'trusts')
            self.executor.submit(self._validate_file, str(balance_file), 'balances')
            
            self.status.object = "Files found and being validated"
            self.status.styles = {'color': '#007bff'}
            
        except Exception as e:
            self.logger.exception(f"Error checking directory: {directory}")
            self.status.object = f"Error: {str(e)}"
            self.status.styles = {'color': 'red'}
            self.csv_trusts_file = ""
            self.csv_balances_file = ""
            self.validation_state = {}
            self._update_file_info()

    def _validate_file(self, filepath: str, validation_key: str) -> bool:
        """Validate CSV file format and content with improved error handling."""
        try:
            self.logger.info(f"Validating {validation_key} file: {filepath}")
            
            # Quick format check
            pd.read_csv(filepath, nrows=5)
            
            # Verify with dask
            ddf = dd.read_csv(filepath, blocksize="64MB")
            ddf.head(1, compute=True)
            
            self.validation_state[validation_key] = True
            self._update_file_info()
            
            if all(self.validation_state.values()):
                self.logger.info("All files validated successfully")
                self.status.object = "All files validated successfully"
                self.status.styles = {'color': 'green'}
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Validation error for {validation_key}")
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
                self.logger.info("Configuration validated successfully")
                self.status.object = "Configuration validated and ready to use"
                self.status.styles = {'color': 'green'}
            else:
                self.logger.warning("Configuration validation failed")
                self.status.object = "Configuration validation failed"
                self.status.styles = {'color': 'red'}
        except Exception as e:
            self.logger.exception("Error validating configuration")
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