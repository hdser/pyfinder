import panel as pn
import param
import os
import io
import pandas as pd
from dashboard.data_source.base import DataSourceComponent

class CSVDataSourceComponent(DataSourceComponent):
    # Class parameters
    csv_trusts_file = param.String(default="")
    csv_balances_file = param.String(default="")
    
    def _create_components(self):
        """Create UI components."""
        self.status = pn.pane.Markdown(
            "Please select CSV files",
            styles={'color': 'gray'}
        )
        
        # Path displays
        self.trusts_path_display = pn.pane.Markdown("Trust File: Not selected")
        self.balances_path_display = pn.pane.Markdown("Balance File: Not selected")
        
        # File inputs for manual upload
        self.trusts_file_input = pn.widgets.FileInput(
            name='Trust Relationships File (CSV)',
            accept='.csv'
        )
        
        self.balances_file_input = pn.widgets.FileInput(
            name='Account Balances File (CSV)',
            accept='.csv'
        )
        
        # Buttons
        self.default_files_button = pn.widgets.Button(
            name="Use Default Files",
            button_type="default"
        )
        
        self.load_button = pn.widgets.Button(
            name="Load Configuration",
            button_type="primary",
            disabled=True
        )

    def _setup_callbacks(self):
        """Set up component callbacks."""
        self.trusts_file_input.param.watch(self._handle_trusts_upload, 'value')
        self.balances_file_input.param.watch(self._handle_balances_upload, 'value')
        self.default_files_button.on_click(self._use_default_files)
        self.load_button.on_click(self._handle_load)

    def _update_displays(self):
        """Update file path displays and load button state."""
        self.trusts_path_display.object = f"Trust File: {os.path.basename(self.csv_trusts_file) if self.csv_trusts_file else 'Not selected'}"
        self.balances_path_display.object = f"Balance File: {os.path.basename(self.csv_balances_file) if self.csv_balances_file else 'Not selected'}"
        self.load_button.disabled = not (self.csv_trusts_file and self.csv_balances_file)

    def _handle_trusts_upload(self, event):
        """Handle trust relationships file upload."""
        if event.new:
            try:
                # Create uploads directory if it doesn't exist
                if not os.path.exists('uploads'):
                    os.makedirs('uploads')
                
                # Save uploaded file
                filename = os.path.join('uploads', 'trusts.csv')
                with open(filename, 'wb') as f:
                    f.write(event.new)
                
                self.csv_trusts_file = filename
                self._update_displays()
                self.status.object = "Trust relationships file uploaded"
                self.status.styles = {'color': 'green'}
            except Exception as e:
                self.status.object = f"Error saving trust file: {str(e)}"
                self.status.styles = {'color': 'red'}

    def _handle_balances_upload(self, event):
        """Handle account balances file upload."""
        if event.new:
            try:
                # Create uploads directory if it doesn't exist
                if not os.path.exists('uploads'):
                    os.makedirs('uploads')
                
                # Save uploaded file
                filename = os.path.join('uploads', 'balances.csv')
                with open(filename, 'wb') as f:
                    f.write(event.new)
                
                self.csv_balances_file = filename
                self._update_displays()
                self.status.object = "Account balances file uploaded"
                self.status.styles = {'color': 'green'}
            except Exception as e:
                self.status.object = f"Error saving balances file: {str(e)}"
                self.status.styles = {'color': 'red'}

    def _use_default_files(self, event):
        """Use default CSV files from data directory."""
        try:
            # Check several possible locations for the data files
            potential_locations = [
                "data",
                "../data",
                "../../data",
                os.path.join(os.path.dirname(__file__), "../../../data")
            ]
            
            found = False
            for location in potential_locations:
                trust_path = os.path.join(location, "data-trust.csv")
                balance_path = os.path.join(location, "data-balance.csv")
                
                if os.path.exists(trust_path) and os.path.exists(balance_path):
                    self.csv_trusts_file = os.path.abspath(trust_path)
                    self.csv_balances_file = os.path.abspath(balance_path)
                    found = True
                    break
            
            if found:
                self._update_displays()
                self.status.object = "Default files loaded successfully"
                self.status.styles = {'color': 'green'}
            else:
                raise FileNotFoundError("Default CSV files not found in any expected location")
                
        except Exception as e:
            self.status.object = f"Error loading default files: {str(e)}"
            self.status.styles = {'color': 'red'}
            self.csv_trusts_file = ""
            self.csv_balances_file = ""
            self._update_displays()

    def _handle_load(self, event):
        """Handle the load configuration button click."""
        try:
            self.validate_configuration()
            self.status.object = "Configuration validated successfully"
            self.status.styles = {'color': 'green'}
        except Exception as e:
            self.status.object = f"Configuration Error: {str(e)}"
            self.status.styles = {'color': 'red'}

    def validate_configuration(self):
        """Validate the current configuration."""
        if not self.csv_trusts_file or not self.csv_balances_file:
            raise ValueError("Both CSV files must be selected")
        
        if not os.path.exists(self.csv_trusts_file):
            raise FileNotFoundError(f"Trust file not found: {self.csv_trusts_file}")
        
        if not os.path.exists(self.csv_balances_file):
            raise FileNotFoundError(f"Balance file not found: {self.csv_balances_file}")

    def get_configuration(self):
        """Get the current configuration."""
        if not self.csv_trusts_file or not self.csv_balances_file:
            return None
        return (self.csv_trusts_file, self.csv_balances_file)

    def view(self):
        """Return the component's view with proper spacing."""
        return pn.Column(
            self.status,
            pn.pane.Markdown("### Choose Trusts File"),
            self.trusts_file_input,
            pn.pane.Markdown("### Choose Balances File"),
            self.balances_file_input,
            pn.Row(
                pn.Column(
                    f"Trust: {os.path.basename(self.csv_trusts_file) if self.csv_trusts_file else 'Not selected'}",
                    f"Balance: {os.path.basename(self.csv_balances_file) if self.csv_balances_file else 'Not selected'}",
                    styles={'font-family': 'monospace'}
                ),
                margin=(10, 0)
            ),
            pn.Row(
                self.default_files_button,
                self.load_button,
                align='center',
                margin=(10, 0)
            ),
            margin=(10, 10),
            sizing_mode='stretch_width'
        )