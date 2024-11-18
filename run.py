import panel as pn
from dashboard import create_dashboard

def main():
    # Initialize Panel
    pn.extension(sizing_mode="stretch_width")
    
    # Serve the dashboard using panel.serve
    pn.serve(create_dashboard(), port=5006)

if __name__ == "__main__":
    main()