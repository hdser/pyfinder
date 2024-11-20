import panel as pn
from dashboard import create_dashboard
import os

def print_env_info():
    """Print environment information for debugging"""
    print("\nEnvironment Variables:")
    env_vars = {
        'BOKEH_ALLOW_WS_ORIGIN': 'Allowed WebSocket Origins',
        'APP_DOMAIN': 'App Domain',
        'APP_URL': 'App URL',
        'PUBLIC_URL': 'Public URL',
        'PYTHONPATH': 'Python Path'
    }
    
    for var, description in env_vars.items():
        value = os.getenv(var, 'Not set')
        print(f"- {description} ({var}): {value}")

def get_allowed_origins():
    """Get allowed origins for websocket connections"""
    # Get origins from environment variable
    do_origin = os.getenv('BOKEH_ALLOW_WS_ORIGIN', '')
    allowed_origins = do_origin.split(',') if do_origin else []
    
    # Always include local development origins
    local_origins = ['localhost:5006', '0.0.0.0:5006', '127.0.0.1:5006']
    allowed_origins.extend(local_origins)
    
    # Remove any empty strings and duplicates
    allowed_origins = list(filter(None, set(allowed_origins)))
    
    return allowed_origins

def main():
    # Print environment information
    print_env_info()
    
    # Initialize Panel
    pn.extension(sizing_mode="stretch_width")
    
    # Get allowed origins
    allowed_origins = get_allowed_origins()
    print(f"\nConfiguration:")
    print(f"- Allowed WebSocket origins: {allowed_origins}")
    
    # Serve the dashboard
    server_kwargs = {
        'port': 5006,
        'address': '0.0.0.0',
        'allow_websocket_origin': allowed_origins,
        'show': False
    }
    
    # Add production settings if we're on Digital Ocean
    if os.getenv('APP_DOMAIN'):
        server_kwargs.update({
            'check_unused_sessions': 1000,
            'keep_alive_milliseconds': 1000,
            'num_procs': 1,
            'websocket_max_message_size': 20*1024*1024
        })
    
    print("\nServer Configuration:")
    for key, value in server_kwargs.items():
        print(f"- {key}: {value}")
    
    # Serve the dashboard
    pn.serve(create_dashboard(), **server_kwargs)

if __name__ == "__main__":
    main()