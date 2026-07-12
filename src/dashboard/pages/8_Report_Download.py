import streamlit as st
from pathlib import Path
from configs.config import REPORTS_DIR

def main():
    st.set_page_config(page_title="Download Reports - FairExplainAI", layout="wide")
    
    from src.dashboard.components.sidebar import render_sidebar
    dataset = render_sidebar()
    
    st.title("📥 Export & Download Reports")
    st.markdown(f"""
    Retrieve and download pre-generated summary reports of dataset profiles, model comparison lists, 
    and fairness metrics evaluation for the **{dataset.upper()}** dataset.
    """)

    from configs.config import BASE_DIR
    reports_dir = Path(BASE_DIR) / "reports" / dataset
    md_path = reports_dir / "summary_report.md"
    html_path = reports_dir / "summary_report.html"
    json_path = reports_dir / "summary_report.json"

    if not md_path.exists() or not html_path.exists():
        st.error(f"Pre-generated reports not found for {dataset.upper()}. Please run the training pipeline first using main.py.")
        return

    # Load contents
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    with open(json_path, "r") as f:
        json_content = f.read()

    # Layout for downloads
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📄 Download Markdown Report",
            data=md_content,
            file_name="FairExplainAI_Summary_Report.md",
            mime="text/markdown"
        )
        
    with col2:
        st.download_button(
            label="🌐 Download HTML Report",
            data=html_content,
            file_name="FairExplainAI_Summary_Report.html",
            mime="text/html"
        )
        
    with col3:
        st.download_button(
            label="⚙️ Download JSON Metrics",
            data=json_content,
            file_name="FairExplainAI_Summary_Metrics.json",
            mime="application/json"
        )

    # Preview report in UI
    st.subheader("Report Preview")
    
    preview_type = st.radio("Select format to preview", ["HTML Rendered", "Markdown Source", "JSON Data"])
    
    if preview_type == "HTML Rendered":
        st.components.v1.html(html_content, height=800, scrolling=True)
    elif preview_type == "Markdown Source":
        st.code(md_content, language="markdown")
    elif preview_type == "JSON Data":
        st.json(json_content)

if __name__ == "__main__":
    main()
