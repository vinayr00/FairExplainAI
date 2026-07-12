"""Model comparison helpers."""
import pandas as pd

def compare_metrics(baseline_metrics: dict, improved_metrics: dict) -> pd.DataFrame:
    """Generates comparison DataFrame between baseline and improved model metrics."""
    records = []
    all_keys = set(baseline_metrics.keys()).union(set(improved_metrics.keys()))
    
    for key in all_keys:
        val_base = baseline_metrics.get(key, None)
        val_imp = improved_metrics.get(key, None)
        diff = val_imp - val_base if (val_base is not None and val_imp is not None) else None
        
        records.append({
            "metric": key,
            "baseline": val_base,
            "improved": val_imp,
            "difference": diff
        })
        
    return pd.DataFrame(records)
