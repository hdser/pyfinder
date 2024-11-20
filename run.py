import panel as pn
from dashboard import create_dashboard
import os
from urllib.parse import urlparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def get_allowed_origins():
    """Get allowed origins for websocket connections."""
    allowed_origins = set()
    
    # Always include Digital Ocean app domain
    allowed_origins.add('pyfinder-app-6r72q.ondigitalocean.app')
    
    # Get other origins from environment variables
    if os.getenv('APP_DOMAIN'):
        allowed_origins.add(os.getenv('APP_DOMAIN'))
    
    if os.getenv('BOKEH_ALLOW_WS_ORIGIN'):
        origins = os.getenv('BOKEH_ALLOW_WS_ORIGIN').split(',')
        allowed_origins.update(origin.strip() for origin in origins if origin.strip())
    
    # Development origins
    dev_origins = {'localhost:5006', '0.0.0.0:5006', '127.0.0.1:5006'}
    allowed_origins.update(dev_origins)
    
    # Clean origins (remove any protocol, path, etc)
    cleaned_origins = set()
    for origin in allowed_origins:
        if '://' in origin:
            origin = origin.split('://', 1)[1]
        origin = origin.split('/')[0]  # Remove any path
        cleaned_origins.add(origin)
    
    # Log all origins
    logger.info("Allowed Origins:")
    for origin in sorted(cleaned_origins):
        logger.info(f"  - {origin}")
        
    return list(cleaned_origins)

def main():
    # Initialize Panel
    pn.extension(sizing_mode="stretch_width")
    
    # Server configuration
    port = int(os.getenv('PORT', '5006'))
    host = os.getenv('HOST', '0.0.0.0')
    allowed_origins = get_allowed_origins()
    
    logger.info("Server Configuration:")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    
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
        # Set Bokeh environment variable
        os.environ['BOKEH_ALLOW_WS_ORIGIN'] = ','.join(allowed_origins)
        
        # Start server
        logger.info("Starting Panel server...")
        pn.serve(create_dashboard(), **server_kwargs)
        
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise

if __name__ == "__main__":
    main()