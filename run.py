# run.py

import panel as pn
from dashboard import create_dashboard
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Network Flow Analysis Dashboard')
    parser.add_argument('--no-spark', action='store_true',
                        help='Disable Spark support for visualization')
    args = parser.parse_args()

    # Initialize Panel
    pn.extension('bokeh', sizing_mode="stretch_width")
    
    # Create and serve the dashboard
    app = create_dashboard(enable_spark=not args.no_spark)
    
    try:
        # Serve the app on port 5006
        pn.serve(app, port=5006, show=True)
    finally:
        # Ensure cleanup happens when the server stops
        if hasattr(app, 'cleanup'):
            app.cleanup()

if __name__ == "__main__":
    main()
