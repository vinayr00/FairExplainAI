import streamlit as st

def render_sidebar():
    """Renders the standard sidebar for dataset selection and navigation info.
    Uses decoupled session state to persist selection across page changes.
    """
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
