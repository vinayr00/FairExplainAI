import sys
from pathlib import Path

# Set PYTHONPATH programmatically
root_dir = Path(__file__).parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Redirect to the main streamlit dashboard
if __name__ == "__main__":
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        import importlib
        stcli = importlib.import_module("streamlit.cli")
    
    # Run the dashboard app entry point with production configuration
    sys.argv = [
        "streamlit", 
        "run", 
        str(root_dir / "src" / "dashboard" / "app.py"),
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    sys.exit(stcli.main())
