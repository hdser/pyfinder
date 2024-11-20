import panel as pn
from dashboard import create_dashboard
import os
from pathlib import Path

def setup_environment():
    """Setup environment variables based on whether we're in production or development"""
    # Determine if we're running in production (Digital Ocean)
    is_production = os.getenv('DIGITAL_OCEAN_APP', '').lower() == 'true'
    
    if is_production:
        # Production settings
        os.environ.setdefault('PYTHONPATH', '/app')
        os.environ.setdefault('PYTHONUNBUFFERED', '1')
        base_dir = Path('/app')
    else:
        # Development settings
        base_dir = Path(__file__).parent
        if not 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] = str(base_dir)
    
    return is_production, base_dir

def main():
    # Setup environment
    is_production, base_dir = setup_environment()
    
    # Get the app URL from environment variable or use default for local development
    app_url = os.getenv('APP_URL', None)
    
    # Set up allowed origins
    allowed_origins = ['localhost:5006', '0.0.0.0:5006', '127.0.0.1:5006']
    
    # If we have an app URL (production), add it to allowed origins
    if app_url:
        # Remove 'https://' if present and any trailing slashes
        app_url = app_url.replace('https://', '').rstrip('/')
        allowed_origins.append(app_url)
    
    # Initialize Panel
    pn.extension(sizing_mode="stretch_width")
    
    # Print debugging information
    print(f"Running in {'production' if is_production else 'development'} mode")
    print(f"Base directory: {base_dir}")
    print(f"Python path: {os.getenv('PYTHONPATH')}")
    print(f"Allowed WebSocket origins: {allowed_origins}")
    
    # Serve the dashboard with the correct websocket settings
    pn.serve(
        create_dashboard(),
        port=5006,
        address="0.0.0.0",
        allow_websocket_origin=allowed_origins,
        show=False
    )

if __name__ == "__main__":
    main()