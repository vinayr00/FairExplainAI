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
    # Ensure processed datasets are built before starting
    from pathlib import Path
    
    processed_dir = root_dir / "data" / "processed"
    required_files = [
        "compas_baseline.csv", "compas_extended.csv",
        "compas_baseline_raw.csv", "compas_extended_raw.csv",
        "adult_baseline.csv", "adult_extended.csv",
        "adult_baseline_raw.csv", "adult_extended_raw.csv"
    ]
    missing = [f for f in required_files if not (processed_dir / f).exists()]
    if missing:
        print("Processed dataset files are missing. Building datasets on startup...")
        try:
            import configs.config as cfg
            from src.preprocessing.build_dataset import build_datasets
            
            # Build compas
            cfg.DATASET = "compas"
            cfg.DATASET_PATH = cfg.DATASET_CONFIG["compas"]["dataset_path"]
            cfg.TARGET_COLUMN = cfg.DATASET_CONFIG["compas"]["target"]
            cfg.PROTECTED_ATTRIBUTES = cfg.DATASET_CONFIG["compas"]["protected"]
            cfg.SENSITIVE_ATTRIBUTE = cfg.DATASET_CONFIG["compas"]["sensitive"]
            build_datasets()
            
            # Build adult
            cfg.DATASET = "adult"
            cfg.DATASET_PATH = cfg.DATASET_CONFIG["adult"]["dataset_path"]
            cfg.TARGET_COLUMN = cfg.DATASET_CONFIG["adult"]["target"]
            cfg.PROTECTED_ATTRIBUTES = cfg.DATASET_CONFIG["adult"]["protected"]
            cfg.SENSITIVE_ATTRIBUTE = cfg.DATASET_CONFIG["adult"]["sensitive"]
            build_datasets()
            
            # Restore default config settings
            cfg.DATASET = "compas"
            cfg.DATASET_PATH = cfg.DATASET_CONFIG["compas"]["dataset_path"]
            cfg.TARGET_COLUMN = cfg.DATASET_CONFIG["compas"]["target"]
            cfg.PROTECTED_ATTRIBUTES = cfg.DATASET_CONFIG["compas"]["protected"]
            cfg.SENSITIVE_ATTRIBUTE = cfg.DATASET_CONFIG["compas"]["sensitive"]
            
            print("Successfully preprocessed both datasets on startup.")
        except Exception as e:
            print(f"Error building datasets on startup: {e}")

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
