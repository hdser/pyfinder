import panel as pn
from dashboard import create_dashboard
import os
from urllib.parse import urlparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_allowed_origins():
    """Get allowed origins for websocket connections."""
    allowed_origins = set()
    
    # Get Digital Ocean app domain
    app_domain = os.getenv('APP_DOMAIN')
    if app_domain:
        # Add both domain and full HTTPS URL
        allowed_origins.add(app_domain)
        allowed_origins.add(f"https://{app_domain}")
        logger.info(f"Added Digital Ocean domain: {app_domain}")
    
    # Get origins from BOKEH_ALLOW_WS_ORIGIN environment variable
    bokeh_origins = os.getenv('BOKEH_ALLOW_WS_ORIGIN', '')
    if bokeh_origins:
        for origin in bokeh_origins.split(','):
            origin = origin.strip()
            if origin:
                allowed_origins.add(origin)
                # Also add with https:// if not present
                if not origin.startswith(('http://', 'https://')):
                    allowed_origins.add(f"https://{origin}")
        logger.info(f"Added Bokeh origins: {bokeh_origins}")

    # Always include development origins
    dev_origins = {'localhost:5006', '0.0.0.0:5006', '127.0.0.1:5006'}
    allowed_origins.update(dev_origins)
    logger.info(f"Added development origins: {dev_origins}")

    # Remove any empty strings
    allowed_origins = {origin for origin in allowed_origins if origin}
    
    # Log all origins
    logger.info(f"Final allowed origins: {allowed_origins}")
    
    return list(allowed_origins)

def main():
    # Initialize Panel
    pn.extension(sizing_mode="stretch_width")

    # Get configuration
    port = int(os.getenv('PORT', '5006'))
    host = os.getenv('HOST', '0.0.0.0')
    allowed_origins = get_allowed_origins()
    
    logger.info("Server Configuration:")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Allowed Origins: {allowed_origins}")

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

    try:
        # Set Bokeh environment variable as backup
        os.environ['BOKEH_ALLOW_WS_ORIGIN'] = ','.join(allowed_origins)
        
        # Serve the dashboard
        logger.info("Starting Panel server...")
        pn.serve(create_dashboard(), **server_kwargs)
        
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()