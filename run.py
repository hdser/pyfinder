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
    
    # Get Digital Ocean app domain
    app_domain = os.getenv('APP_DOMAIN', '')
    if app_domain:
        allowed_origins.add(app_domain)
        logger.info(f"Added app domain: {app_domain}")
    
    bokeh_origins = os.getenv('BOKEH_ALLOW_WS_ORIGIN', '')
    if bokeh_origins:
        allowed_origins.update(origin.strip() for origin in bokeh_origins.split(','))
    
    # Development origins
    dev_origins = {'localhost:5006', '0.0.0.0:5006', '127.0.0.1:5006'}
    allowed_origins.update(dev_origins)
    
    # Clean origins - remove any protocol/path
    cleaned_origins = set()
    for origin in allowed_origins:
        if origin:
            # Remove protocol if present
            if '://' in origin:
                origin = origin.split('://', 1)[1]
            # Remove path if present
            origin = origin.split('/', 1)[0]
            # Remove port 80/443 if present
            for port in [':80', ':443']:
                if origin.endswith(port):
                    origin = origin[:-len(port)]
            cleaned_origins.add(origin)
    
    logger.info("Allowed Origins:")
    for origin in sorted(cleaned_origins):
        logger.info(f"  - {origin}")
    
    return list(cleaned_origins)

def main():
    # Initialize Panel
    pn.extension(sizing_mode="stretch_width")
    
    # Get configuration
    port = int(os.getenv('PORT', '5006'))
    host = os.getenv('HOST', '0.0.0.0')
    allowed_origins = get_allowed_origins()
    
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
        # Log environment and configuration
        logger.info(f"Starting server on {host}:{port}")
        logger.info(f"Data directory: {os.path.abspath('/app/data')}")
        if os.path.exists('/app/data'):
            files = os.listdir('/app/data')
            logger.info(f"Files in data directory: {files}")
        
        # Serve the dashboard
        pn.serve(create_dashboard(), **server_kwargs)
        
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()