import panel as pn
from dashboard import create_dashboard
import os
from urllib.parse import urlparse

def print_env_info():
    """Print all environment variables for debugging."""
    print("\nEnvironment Variables:")
    for key, value in os.environ.items():
        print(f"{key}: {value}")

def get_allowed_origins():
    """Get allowed origins for websocket connections."""
    allowed_origins = []

    # Get origins from BOKEH_ALLOW_WS_ORIGIN environment variable
    bokeh_origins = os.getenv('BOKEH_ALLOW_WS_ORIGIN')
    if bokeh_origins:
        allowed_origins.extend(bokeh_origins.split(','))

    # Parse APP_URL to get the domain
    app_url = os.getenv('APP_URL')
    if app_url:
        parsed_url = urlparse(app_url)
        app_domain = parsed_url.netloc
        if app_domain:
            allowed_origins.append(app_domain)

    # Always include local development origins
    local_origins = ['localhost:5006', '0.0.0.0:5006', '127.0.0.1:5006']
    allowed_origins.extend(local_origins)

    # Remove duplicates
    allowed_origins = list(set(allowed_origins))

    return allowed_origins

def main():
    # Print environment variables
    print_env_info()

    # Initialize Panel
    pn.extension(sizing_mode="stretch_width")

    # Get allowed origins
    allowed_origins = get_allowed_origins()
    print(f"\nConfiguration:")
    print(f"- Allowed WebSocket origins: {allowed_origins}")

    # Server configuration
    server_kwargs = {
        'port': int(os.getenv('PORT', '5006')),
        'address': os.getenv('HOST', '0.0.0.0'),
        'allow_websocket_origin': allowed_origins,
        'show': False
    }

    # Production settings for DigitalOcean
    if os.getenv('APP_URL'):
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
