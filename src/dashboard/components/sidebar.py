import streamlit as st
from pathlib import Path

def ensure_datasets_built():
    """Checks if processed dataset files exist, and builds them if missing."""
    if "datasets_verified" in st.session_state and st.session_state["datasets_verified"]:
        return
        
    from configs.config import BASE_DIR
    processed_dir = Path(BASE_DIR) / "data" / "processed"
    required_files = [
        "compas_baseline.csv", "compas_extended.csv",
        "compas_baseline_raw.csv", "compas_extended_raw.csv",
        "adult_baseline.csv", "adult_extended.csv",
        "adult_baseline_raw.csv", "adult_extended_raw.csv"
    ]
    missing = [f for f in required_files if not (processed_dir / f).exists()]
    if missing:
        # Display a nice status message in the sidebar while building
        with st.sidebar.status("🔧 Preparing datasets on first launch...", expanded=True) as status:
            try:
                import configs.config as cfg
                from src.preprocessing.build_dataset import build_datasets
                
                status.write("Processing COMPAS dataset...")
                # Build compas
                cfg.DATASET = "compas"
                cfg.DATASET_PATH = cfg.DATASET_CONFIG["compas"]["dataset_path"]
                cfg.TARGET_COLUMN = cfg.DATASET_CONFIG["compas"]["target"]
                cfg.PROTECTED_ATTRIBUTES = cfg.DATASET_CONFIG["compas"]["protected"]
                cfg.SENSITIVE_ATTRIBUTE = cfg.DATASET_CONFIG["compas"]["sensitive"]
                build_datasets()
                
                status.write("Processing Adult Income dataset...")
                # Build adult
                cfg.DATASET = "adult"
                cfg.DATASET_PATH = cfg.DATASET_CONFIG["adult"]["dataset_path"]
                cfg.TARGET_COLUMN = cfg.DATASET_CONFIG["adult"]["target"]
                cfg.PROTECTED_ATTRIBUTES = cfg.DATASET_CONFIG["adult"]["protected"]
                cfg.SENSITIVE_ATTRIBUTE = cfg.DATASET_CONFIG["adult"]["sensitive"]
                build_datasets()
                
                # Restore default config settings based on current session selection
                active = st.session_state.get("selected_dataset", "compas")
                cfg.DATASET = active
                cfg.DATASET_PATH = cfg.DATASET_CONFIG[active]["dataset_path"]
                cfg.TARGET_COLUMN = cfg.DATASET_CONFIG[active]["target"]
                cfg.PROTECTED_ATTRIBUTES = cfg.DATASET_CONFIG[active]["protected"]
                cfg.SENSITIVE_ATTRIBUTE = cfg.DATASET_CONFIG[active]["sensitive"]
                
                status.update(label="✅ Datasets prepared successfully!", state="complete")
            except Exception as e:
                status.update(label=f"❌ Error preparing datasets: {e}", state="error")
                raise e
                
    st.session_state["datasets_verified"] = True

def render_sidebar():
    """Renders the standard sidebar for dataset selection and navigation info.
    Uses decoupled session state to persist selection across page changes.
    """
    # Ensure datasets exist before pages try to load them
    ensure_datasets_built()
    
    st.sidebar.title("Configuration")
    
    # Initialize the selected_dataset in session_state if not present
    if "selected_dataset" not in st.session_state:
        st.session_state["selected_dataset"] = "compas"
        
    dataset_options = ["compas", "adult"]
    try:
        current_idx = dataset_options.index(st.session_state["selected_dataset"])
    except ValueError:
        current_idx = 0
        
    # Decouple the widget key from "selected_dataset" to prevent pruning by Streamlit
    selected_dataset = st.sidebar.selectbox(
        "Select Active Dataset",
        dataset_options,
        index=current_idx,
        key="dataset_selectbox_widget"
    )
    
    # If the user changed the selection, update persistent state and trigger rerun
    if selected_dataset != st.session_state["selected_dataset"]:
        st.session_state["selected_dataset"] = selected_dataset
        st.rerun()
        
    st.sidebar.title("Navigation Info")
    st.sidebar.info("Use the sidebar pages to explore: \n- Dataset & EDA\n- Interactive Model Training\n- Fairness Mitigation\n- SHAP & Counterfactual Explanations\n- Model Comparisons & Policy Recommendations")
    
    return st.session_state["selected_dataset"]
