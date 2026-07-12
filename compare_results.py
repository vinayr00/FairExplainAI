import pandas as pd
from pathlib import Path

def main():
    print("========================================================================")
    print("COMPILING MULTI-DATASET COMPARISON REPORT")
    print("========================================================================")

    compas_path = Path("results/compas/comparison/model_comparison.csv")
    adult_path = Path("results/adult/comparison/model_comparison.csv")

    if not compas_path.exists():
        print(f"[-] WARNING: COMPAS results not found at: {compas_path}. Please run COMPAS pipeline first.")
    if not adult_path.exists():
        print(f"[-] WARNING: Adult results not found at: {adult_path}. Please run Adult pipeline first.")

    dfs = []
    if compas_path.exists():
        df_compas = pd.read_csv(compas_path)
        df_compas["Dataset"] = "COMPAS"
        dfs.append(df_compas)
    if adult_path.exists():
        df_adult = pd.read_csv(adult_path)
        df_adult["Dataset"] = "Adult Income"
        dfs.append(df_adult)

    if not dfs:
        print("[-] ERROR: No results found for either dataset. Cannot generate comparison report.")
        return

    # Combine results
    df_combined = pd.concat(dfs, axis=0, ignore_index=True)

    # Reorder columns to put Dataset and Rank at the front
    cols = ["Dataset", "rank", "model_name", "mitigation_status", "accuracy", "f1_score", "roc_auc", 
            "demographic_parity_difference", "equalized_odds_difference", "disparate_impact_ratio",
            "training_time", "inference_time", "total_score"]
    
    # Filter existing columns
    existing_cols = [c for c in cols if c in df_combined.columns]
    df_report = df_combined[existing_cols].copy()

    # Sort primarily by Dataset and rank
    df_report = df_report.sort_values(by=["Dataset", "rank"]).reset_index(drop=True)

    # Save comparison report
    output_path = Path("results/comparison_report.csv")
    df_report.to_csv(output_path, index=False)
    print(f"[+] Multi-dataset comparison report successfully saved to: {output_path}")

    # Display a beautiful summary table
    print("\n--- PUBLICATION-QUALITY COMPARISON TABLE ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_report.to_string(index=False))
    print("========================================================================")

if __name__ == "__main__":
    main()
