import panel as pn
from dashboard import create_dashboard
import os
from pathlib import Path

def get_allowed_origins():
    """Get allowed origins for websocket connections"""
    # Base allowed origins for local development
    allowed_origins = ['localhost:5006', '0.0.0.0:5006', '127.0.0.1:5006']
    
    # Get the Digital Ocean app URL (it includes https://)
    app_url = os.getenv('APP_URL', '')
    if app_url:
        # Clean the URL to just the domain
        clean_url = app_url.replace('https://', '').replace('http://', '').rstrip('/')
        allowed_origins.append(clean_url)
        print(f"Adding Digital Ocean URL to allowed origins: {clean_url}")
    
    # Get the raw host (sometimes provided by Digital Ocean)
    raw_host = os.getenv('HOST', '')
    if raw_host and raw_host not in allowed_origins:
        allowed_origins.append(raw_host)
    
    return allowed_origins

def setup_environment():
    """Setup environment variables based on whether we're in production or development"""
    # Check if we're running on Digital Ocean
    is_production = bool(os.getenv('APP_URL', ''))
    
    if is_production:
        base_dir = Path('/app')
        os.environ.setdefault('PYTHONPATH', '/app')
        print("Running in production mode on Digital Ocean")
    else:
        base_dir = Path(__file__).parent
        os.environ.setdefault('PYTHONPATH', str(Path(__file__).parent))
        print("Running in development mode")
    
    return is_production, base_dir

def main():
    # Setup environment
    is_production, base_dir = setup_environment()
    
    # Get allowed origins
    allowed_origins = get_allowed_origins()
    
    # Initialize Panel
    pn.extension(sizing_mode="stretch_width")
    
    # Print configuration for debugging
    print(f"Environment Configuration:")
    print(f"- Base directory: {base_dir}")
    print(f"- Python path: {os.getenv('PYTHONPATH')}")
    print(f"- App URL: {os.getenv('APP_URL', 'Not set')}")
    print(f"- Host: {os.getenv('HOST', 'Not set')}")
    print(f"- Allowed WebSocket origins: {allowed_origins}")
    
    # Configure server
    server_kwargs = {
        'port': 5006,
        'address': '0.0.0.0',
        'allow_websocket_origin': allowed_origins,
        'show': False
    }
    
    # Add extra server configurations for production
    if is_production:
        server_kwargs.update({
            'check_unused_sessions': 1000,  # milliseconds
            'keep_alive_milliseconds': 1000,
            'num_procs': 1,
            'websocket_max_message_size': 20*1024*1024  # 20MB
        })
    
    print(f"\nStarting server with configuration:")
    for key, value in server_kwargs.items():
        print(f"- {key}: {value}")
        
    # Serve the dashboard
    pn.serve(create_dashboard(), **server_kwargs)

if __name__ == "__main__":
    main()