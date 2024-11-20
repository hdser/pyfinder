import panel as pn
from dashboard import create_dashboard
import os

def main():
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
    
    # Print allowed origins for debugging
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
