import panel as pn
from dashboard import create_dashboard
import os
from urllib.parse import urlparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def print_env_info():
    """Print relevant environment variables for debugging."""
    debug_vars = [
        'APP_DOMAIN', 'BOKEH_ALLOW_WS_ORIGIN', 'APP_URL', 
        'HOST', 'PORT', 'PUBLIC_URL', 'HOSTNAME'
    ]
    
    logger.info("Environment Variables:")
    for var in debug_vars:
        value = os.getenv(var)
        logger.info(f"{var}: {value}")

def clean_origin(origin: str) -> str:
    """Clean origin string to remove protocol and path."""
    if not origin:
        return origin
    # Remove any protocol
    if '://' in origin:
        origin = origin.split('://', 1)[1]
    # Remove any path
    origin = origin.split('/', 1)[0]
    # Remove any port if it's standard http/https port
    if origin.endswith(':80') or origin.endswith(':443'):
        origin = origin.rsplit(':', 1)[0]
    return origin

def get_allowed_origins():
    """Get allowed origins for websocket connections."""
    allowed_origins = set()
    
    # Print environment info for debugging
    print_env_info()
    
    # Get Digital Ocean app domain
    app_domain = os.getenv('APP_DOMAIN', '')
    app_url = os.getenv('APP_URL', '')
    
    if app_domain:
        allowed_origins.add(clean_origin(app_domain))
        logger.info(f"Added app domain: {app_domain}")
    
    if app_url:
        parsed = urlparse(app_url)
        if parsed.netloc:
            allowed_origins.add(parsed.netloc)
        logger.info(f"Added app URL: {app_url}")
    
    # Get origins from BOKEH_ALLOW_WS_ORIGIN environment variable
    bokeh_origins = os.getenv('BOKEH_ALLOW_WS_ORIGIN', '')
    if bokeh_origins:
        for origin in bokeh_origins.split(','):
            origin = origin.strip()
            if origin:
                allowed_origins.add(clean_origin(origin))
        logger.info(f"Added Bokeh origins: {bokeh_origins}")

    # Always include development origins
    dev_origins = {'localhost:5006', '0.0.0.0:5006', '127.0.0.1:5006'}
    allowed_origins.update(dev_origins)
    logger.info(f"Added development origins: {dev_origins}")

    # Remove any empty strings
    allowed_origins = {origin for origin in allowed_origins if origin}
    
    # Log final configuration
    logger.info("Final allowed origins:")
    for origin in sorted(allowed_origins):
        logger.info(f"  - {origin}")
    
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
        
        # Log the complete server configuration
        logger.info("Complete server configuration:")
        for key, value in server_kwargs.items():
            logger.info(f"  {key}: {value}")
        
        # Serve the dashboard
        logger.info("Starting Panel server...")
        pn.serve(create_dashboard(), **server_kwargs)
        
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()