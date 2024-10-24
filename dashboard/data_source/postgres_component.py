import panel as pn
import param
from dotenv import load_dotenv
import os
from dashboard.data_source.base import DataSourceComponent

class PostgresManualComponent(DataSourceComponent):
    pg_host = param.String(default="localhost")
    pg_port = param.String(default="5432")
    pg_dbname = param.String(default="")
    pg_user = param.String(default="")
    pg_password = param.String(default="")
    pg_queries_dir = param.String(default="queries")
    
    def _create_components(self):
        """Create the PostgreSQL configuration components."""
        # Input fields
        self.host_input = pn.widgets.TextInput(
            name="Host",
            value=self.pg_host,
            placeholder="localhost"
        )
        self.port_input = pn.widgets.TextInput(
            name="Port",
            value=self.pg_port,
            placeholder="5432"
        )
        self.dbname_input = pn.widgets.TextInput(
            name="Database Name",
            value=self.pg_dbname,
            placeholder="database"
        )
        self.user_input = pn.widgets.TextInput(
            name="Username",
            value=self.pg_user,
            placeholder="user"
        )
        self.password_input = pn.widgets.PasswordInput(
            name="Password",
            value=self.pg_password,
            placeholder="password"
        )
        self.queries_dir_input = pn.widgets.TextInput(
            name="Queries Directory",
            value=self.pg_queries_dir,
            placeholder="queries"
        )
        
        # Load button and status indicator
        self.load_button = pn.widgets.Button(
            name="Load PostgreSQL Configuration",
            button_type="primary",
            width=200
        )
        self.status = pn.pane.Markdown(
            "Status: Not configured",
            styles={'color': 'gray'}
        )

    def _setup_callbacks(self):
        """Set up event callbacks for the components."""
        self.load_button.on_click(self._handle_load)
        self.host_input.param.watch(self._update_param, 'value')
        self.port_input.param.watch(self._update_param, 'value')
        self.dbname_input.param.watch(self._update_param, 'value')
        self.user_input.param.watch(self._update_param, 'value')
        self.password_input.param.watch(self._update_param, 'value')
        self.queries_dir_input.param.watch(self._update_param, 'value')

    def _update_param(self, event):
        """Update the corresponding parameter when input changes."""
        param_name = event.obj.name.lower().replace(' ', '_')
        if hasattr(self, f'pg_{param_name}'):
            setattr(self, f'pg_{param_name}', event.new)

    def _handle_load(self, event):
        """Handle the load button click event."""
        try:
            self.validate_configuration()
            self.status.object = "Status: Configuration loaded successfully"
            self.status.styles = {'color': 'green'}
        except ValueError as e:
            self.status.object = f"Status: Error - {str(e)}"
            self.status.styles = {'color': 'red'}

    def validate_configuration(self):
        """Validate the PostgreSQL configuration."""
        required_fields = ['pg_host', 'pg_port', 'pg_dbname', 'pg_user', 'pg_password']
        missing_fields = [
            field.replace('pg_', '') for field in required_fields 
            if not getattr(self, field)
        ]
        
        if missing_fields:
            raise ValueError(
                f"Missing required fields: {', '.join(missing_fields)}"
            )
        
        if not os.path.exists(self.pg_queries_dir):
            raise ValueError(
                f"Queries directory not found: {self.pg_queries_dir}"
            )

    def get_configuration(self):
        """Get the current configuration."""
        return {
            'host': self.pg_host,
            'port': self.pg_port,
            'dbname': self.pg_dbname,
            'user': self.pg_user,
            'password': self.pg_password
        }, self.pg_queries_dir

    def view(self):
        """Return the component's view."""
        return pn.Column(
            pn.pane.Markdown("### PostgreSQL Configuration"),
            self.host_input,
            self.port_input,
            self.dbname_input,
            self.user_input,
            self.password_input,
            self.queries_dir_input,
            pn.layout.Spacer(height=20),
            pn.Row(self.load_button, align='center'),
            self.status,
            sizing_mode='stretch_width'
        )

class PostgresEnvComponent(DataSourceComponent):
    def _create_components(self):
        """Create the environment-based PostgreSQL configuration components."""
        self.load_button = pn.widgets.Button(
            name="Load Environment Configuration",
            button_type="primary",
            width=200
        )
        self.status = pn.pane.Markdown(
            "Status: Not configured",
            styles={'color': 'gray'}
        )
        self._check_env_variables()

    def _setup_callbacks(self):
        """Set up event callbacks for the components."""
        self.load_button.on_click(self._handle_load)

    def _handle_load(self, event):
        """Handle the load button click event."""
        try:
            self.validate_configuration()
            self.status.object = "Status: Configuration loaded successfully"
            self.status.styles = {'color': 'green'}
        except ValueError as e:
            self.status.object = f"Status: Error - {str(e)}"
            self.status.styles = {'color': 'red'}

    def _check_env_variables(self):
        """Check if all required environment variables are set."""
        try:
            self.validate_configuration()
            self.status.object = "✅ All required environment variables are set"
            self.status.styles = {'color': 'green'}
        except ValueError as e:
            self.status.object = f"⚠️ Configuration Error: {str(e)}"
            self.status.styles = {'color': 'red'}

    def validate_configuration(self):
        """Validate the environment configuration."""
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
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    def get_configuration(self):
        """Get the configuration from environment variables."""
        load_dotenv()
        return {
            'host': os.getenv('POSTGRES_HOST'),
            'port': os.getenv('POSTGRES_PORT'),
            'dbname': os.getenv('POSTGRES_DB'),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }, 'queries'

    def view(self):
        """Return the component's view."""
        return pn.Column(
            pn.pane.Markdown("""
            ### Environment Configuration
            
            The following environment variables are required:
            - POSTGRES_HOST
            - POSTGRES_PORT
            - POSTGRES_DB
            - POSTGRES_USER
            - POSTGRES_PASSWORD
            """),
            pn.layout.Spacer(height=20),
            pn.Row(self.load_button, align='center'),
            self.status,
            sizing_mode='stretch_width'
        )