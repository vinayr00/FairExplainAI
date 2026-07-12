"""Report generator module for fairness and model explanations."""
import json
from pathlib import Path

def generate_json_report(metrics_dict, output_path):
    """Saves explainability/fairness metrics dictionary to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
        
    return output_path

def generate_markdown_summary(summary_data: dict) -> str:
    """Generates a complete research-grade Markdown report from execution summary data."""
    md = []
    md.append("# FairExplainAI Research Project Summary Report")
    md.append("This report summarizes the experimental evaluation of the baseline model against leakage-aware models with fairness mitigations.\n")
    
    # 1. Dataset Overview
    if "dataset" in summary_data:
        ds = summary_data["dataset"]
        md.append("## 1. Dataset & Feature Engineering")
        md.append(f"- **Raw Shape**: {ds.get('raw_shape', 'N/A')}")
        md.append(f"- **Baseline Processed Shape**: {ds.get('baseline_shape', 'N/A')}")
        md.append(f"- **Extended Processed Shape**: {ds.get('extended_shape', 'N/A')}")
        md.append("- **Scientific Justification**: Dropped target leakage variables (like `is_recid`, `decile_score`) and administrative variables to avoid target-feedback circularity and administrative bias.")
        md.append("\n")
        
    # 2. Model Evaluation
    if "models" in summary_data:
        md.append("## 2. Model Performance Evaluation")
        md.append("| Model Name | Accuracy | F1-Score | ROC-AUC | Training Time (s) | Inference Time (s) |")
        md.append("|---|---|---|---|---|---|")
        for m in summary_data["models"]:
            md.append(f"| {m['model_name']} | {m.get('accuracy', 0.0):.4f} | {m.get('f1_score', 0.0):.4f} | {m.get('roc_auc', 0.0):.4f} | {m.get('training_time', 0.0):.4f} | {m.get('inference_time', 0.0):.4f} |")
        md.append("\n")
        
    # 3. Fairness Mitigation Comparison
    if "fairness_comparison" in summary_data:
        md.append("## 3. Bias Assessment & Fairness Mitigation")
        md.append("| Model Status | Demographic Parity Diff | Equalized Odds Diff | Equal Opportunity Diff | Disparate Impact Ratio |")
        md.append("|---|---|---|---|---|")
        for f in summary_data["fairness_comparison"]:
            md.append(f"| {f['model_status']} | {f.get('demographic_parity_difference', 0.0):.4f} | {f.get('equalized_odds_difference', 0.0):.4f} | {f.get('equal_opportunity_difference', 0.0):.4f} | {f.get('disparate_impact_ratio', 0.0):.4f} |")
        md.append("\n")
        
    # 4. Recommendation Engine Output
    if "recommendation" in summary_data:
        rec = summary_data["recommendation"]
        md.append("## 4. Multi-Criteria Recommendation Engine")
        md.append(f"- **Recommended Model**: **{rec.get('model_name', 'N/A')}**")
        md.append(f"- **Overall Utility Score**: {rec.get('total_score', 0.0):.4f}")
        md.append(f"- **Justification**: {rec.get('justification', '')}")
        md.append("\n")
        
    return "\n".join(md)

def generate_html_summary(summary_data: dict) -> str:
    """Generates an HTML report from summary data for display in Streamlit or downloading."""
    md_content = generate_markdown_summary(summary_data)
    # Simple markdown-to-html conversion replacement
    # Using basic replacements for headings and tables to create an elegant HTML
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #34495e; padding-bottom: 10px; }}
            h2 {{ color: #2980b9; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; color: #2c3e50; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            ul {{ padding-left: 20px; }}
            li {{ margin-bottom: 8px; }}
            .highlight {{ background-color: #e8f4f8; padding: 15px; border-left: 5px solid #2980b9; margin: 20px 0; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <h1>⚖️ FairExplainAI Experimental Report</h1>
        <p>This document details the scientific evaluation and bias mitigation results for the COMPAS recidivism prediction dataset.</p>
        
        <h2>1. Preprocessing and Feature Selection Summary</h2>
        <div class="highlight">
            <p><strong>Baseline Dataset Shape:</strong> {summary_data.get('dataset', {}).get('baseline_shape', 'N/A')}</p>
            <p><strong>Extended Dataset Shape:</strong> {summary_data.get('dataset', {}).get('extended_shape', 'N/A')}</p>
        </div>
        <p>Our extended preprocessing removes identifiers, timestamps, and target leakage parameters (specifically <code>decile_score</code>, which represents a black-box model output) to prevent statistical confounding and circular reasoning in model estimation.</p>
        
        <h2>2. Classifier Performance Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Model Name</th>
                    <th>Accuracy</th>
                    <th>F1-Score</th>
                    <th>ROC-AUC</th>
                    <th>Train Time (s)</th>
                    <th>Inference Time (s)</th>
                </tr>
            </thead>
            <tbody>
    """
    for m in summary_data.get("models", []):
        html += f"""
                <tr>
                    <td><strong>{m['model_name']}</strong></td>
                    <td>{m.get('accuracy', 0.0):.4f}</td>
                    <td>{m.get('f1_score', 0.0):.4f}</td>
                    <td>{m.get('roc_auc', 0.0):.4f}</td>
                    <td>{m.get('training_time', 0.0):.4f}</td>
                    <td>{m.get('inference_time', 0.0):.4f}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
        
        <h2>3. Fairness Mitigation Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Model Mitigation Status</th>
                    <th>Demographic Parity Diff</th>
                    <th>Equalized Odds Diff</th>
                    <th>Equal Opportunity Diff</th>
                    <th>Disparate Impact Ratio</th>
                </tr>
            </thead>
            <tbody>
    """
    for f in summary_data.get("fairness_comparison", []):
        html += f"""
                <tr>
                    <td><strong>{f['model_status']}</strong></td>
                    <td>{f.get('demographic_parity_difference', 0.0):.4f}</td>
                    <td>{f.get('equalized_odds_difference', 0.0):.4f}</td>
                    <td>{f.get('equal_opportunity_difference', 0.0):.4f}</td>
                    <td>{f.get('disparate_impact_ratio', 0.0):.4f}</td>
                </tr>
        """
        
    rec = summary_data.get("recommendation", {})
    html += f"""
            </tbody>
        </table>
        
        <h2>4. Multi-Criteria Policy Recommendation</h2>
        <div class="highlight" style="border-left-color: #27ae60; background-color: #ebf7ee;">
            <p><strong>Top Recommended Model:</strong> {rec.get('model_name', 'N/A')} (Score: {rec.get('total_score', 0.0):.4f})</p>
            <p><strong>Justification:</strong> {rec.get('justification', '')}</p>
        </div>
    </body>
    </html>
    """
    return html

