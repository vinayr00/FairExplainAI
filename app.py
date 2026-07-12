import sys
from pathlib import Path

# Set PYTHONPATH programmatically
root_dir = Path(__file__).parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Detect if running inside streamlit
try:
    from streamlit.runtime import exists as runtime_exists
except ImportError:
    runtime_exists = lambda: False

if runtime_exists():
    # If running inside a Streamlit server context, run the dashboard directly
    from src.dashboard.app import main
    if __name__ == "__main__":
        main()
else:
    # If running via standard Python, start the Streamlit CLI and launch the server
    if __name__ == "__main__":
        try:
            import streamlit.web.cli as stcli
        except ImportError:
            import importlib
            stcli = importlib.import_module("streamlit.cli")
        
        # Configure Streamlit production server settings
        sys.argv = [
            "streamlit", 
            "run", 
            str(root_dir / "app.py"),
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ]
        sys.exit(stcli.main())
