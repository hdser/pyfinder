import panel as pn
from dashboard import create_dashboard
import os
from urllib.parse import urlparse

def get_allowed_origins():
    """Get allowed origins for websocket connections."""
    allowed_origins = set()
    
    # Get origins from BOKEH_ALLOW_WS_ORIGIN environment variable
    bokeh_origins = os.getenv('BOKEH_ALLOW_WS_ORIGIN', '')
    if bokeh_origins:
        allowed_origins.update(origin.strip() for origin in bokeh_origins.split(','))

    # Parse APP_URL if available
    app_url = os.getenv('APP_URL')
    if app_url:
        parsed_url = urlparse(app_url)
        app_domain = parsed_url.netloc or parsed_url.path.strip('/')
        if app_domain:
            allowed_origins.add(app_domain)
            # Also add the full URL
            allowed_origins.add(app_url.rstrip('/'))

    # Always include development origins
    dev_origins = {'localhost:5006', '0.0.0.0:5006', '127.0.0.1:5006'}
    allowed_origins.update(dev_origins)

    return list(allowed_origins)

def main():
    # Initialize Panel
    pn.extension(sizing_mode="stretch_width")

    # Get configuration
    port = int(os.getenv('PORT', '5006'))
    host = os.getenv('HOST', '0.0.0.0')
    allowed_origins = get_allowed_origins()
    
    print(f"\nServer Configuration:")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Allowed Origins: {allowed_origins}")

    # Server configuration
    server_kwargs = {
        'port': port,
        'address': host,
        'allow_websocket_origin': allowed_origins,
        'show': False,
        'websocket_max_message_size': int(os.getenv('BOKEH_WEBSOCKET_MAX_MESSAGE_SIZE', 20*1024*1024)),
        'check_unused_sessions': 1000,
        'keep_alive_milliseconds': 1000,
        'num_procs': 1
    }

    # Serve the dashboard
    pn.serve(create_dashboard(), **server_kwargs)

if __name__ == "__main__":
    main()