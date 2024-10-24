import panel as pn
from dashboard import create_dashboard

def main():
    # Initialize Panel
    pn.extension(sizing_mode="stretch_width")
    
    # Create and serve the dashboard
    app = create_dashboard()
    app.show(port=5006)

if __name__ == "__main__":
    main()