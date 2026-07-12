"""
recommendation_engine.py
------------------------
Utility to compare models based on performance, fairness, and efficiency,
and recommend the best model using user-defined weights.
"""

import logging
from typing import Dict, Any, List
import pandas as pd
from configs.config import RECOMMENDATION_PRIORITIES, RECOMMENDATION_SUBWEIGHTS

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Ranks and recommends models based on a multi-criteria utility function."""

    def __init__(
        self,
        priorities: Dict[str, float] = None,
        subweights: Dict[str, float] = None
    ):
        self.priorities = priorities if priorities is not None else RECOMMENDATION_PRIORITIES
        self.subweights = subweights if subweights is not None else RECOMMENDATION_SUBWEIGHTS

    def rank_models(self, models_metrics: List[Dict[str, Any]]) -> pd.DataFrame:
        """Calculates a utility score for each model and returns a ranked DataFrame.
        
        Each metrics dict in models_metrics must contain:
            - model_name
            - accuracy
            - f1_score
            - roc_auc (optional, defaults to accuracy if not present)
            - demographic_parity_difference
            - equalized_odds_difference
            - training_time
            - inference_time
        """
        if not models_metrics:
            logger.warning("Empty models metrics list passed to RecommendationEngine.")
            return pd.DataFrame()

        df = pd.DataFrame(models_metrics)
        
        # Fallbacks for missing metrics
        if "roc_auc" not in df.columns:
            df["roc_auc"] = df["accuracy"]
        if "f1_score" not in df.columns:
            df["f1_score"] = df["accuracy"]

        # Ensure numeric values
        numeric_cols = [
            "accuracy", "f1_score", "roc_auc", 
            "demographic_parity_difference", "equalized_odds_difference",
            "training_time", "inference_time"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col]).fillna(0.0)

        # -----------------------------
        # 1. Performance Score Calculation
        # -----------------------------
        perf_acc_w = self.subweights.get("accuracy_weight", 0.5)
        perf_f1_w = self.subweights.get("f1_weight", 0.3)
        perf_auc_w = self.subweights.get("roc_auc_weight", 0.2)
        
        df["performance_score"] = (
            df["accuracy"] * perf_acc_w +
            df["f1_score"] * perf_f1_w +
            df["roc_auc"] * perf_auc_w
        )

        # -----------------------------
        # 2. Fairness Score Calculation
        # -----------------------------
        # Lower difference is better, so we use (1 - difference) as the utility score
        fair_dp_w = self.subweights.get("demographic_parity_weight", 0.5)
        fair_eo_w = self.subweights.get("equalized_odds_weight", 0.5)
        
        df["fairness_score"] = (
            (1.0 - df["demographic_parity_difference"].clip(0.0, 1.0)) * fair_dp_w +
            (1.0 - df["equalized_odds_difference"].clip(0.0, 1.0)) * fair_eo_w
        )

        # -----------------------------
        # 3. Efficiency Score Calculation
        # -----------------------------
        # Normalize training and inference time so that min time -> 1.0 and max time -> 0.0
        # If there is only 1 model, or max time == min time, default to 1.0
        for col in ["training_time", "inference_time"]:
            col_min = df[col].min()
            col_max = df[col].max()
            col_range = col_max - col_min
            
            if col_range > 0:
                df[f"{col}_score"] = 1.0 - (df[col] - col_min) / col_range
            else:
                df[f"{col}_score"] = 1.0

        eff_train_w = self.subweights.get("training_time_weight", 0.5)
        eff_infer_w = self.subweights.get("inference_time_weight", 0.5)
        
        df["efficiency_score"] = (
            df["training_time_score"] * eff_train_w +
            df["inference_time_score"] * eff_infer_w
        )

        # -----------------------------
        # 4. Total Utility Score
        # -----------------------------
        total_perf_w = self.priorities.get("accuracy_weight", 0.40)
        total_fair_w = self.priorities.get("fairness_weight", 0.40)
        total_eff_w = self.priorities.get("efficiency_weight", 0.20)
        
        df["total_score"] = (
            df["performance_score"] * total_perf_w +
            df["fairness_score"] * total_fair_w +
            df["efficiency_score"] * total_eff_w
        )

        # Rank and return
        df_ranked = df.sort_values(by="total_score", ascending=False).reset_index(drop=True)
        df_ranked["rank"] = df_ranked.index + 1
        
        return df_ranked

    def recommend_best(self, models_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Returns the single best recommended model and its reason."""
        df_ranked = self.rank_models(models_metrics)
        if df_ranked.empty:
            return {}
            
        best = df_ranked.iloc[0]
        recommendation = {
            "model_name": best["model_name"],
            "total_score": float(best["total_score"]),
            "performance_score": float(best["performance_score"]),
            "fairness_score": float(best["fairness_score"]),
            "efficiency_score": float(best["efficiency_score"]),
            "rank": 1,
            "justification": (
                f"The model '{best['model_name']}' is recommended as the optimal choice. "
                f"It achieved a combined utility score of {best['total_score']:.4f} (Accuracy: {best['accuracy']:.4f}, "
                f"Demographic Parity Difference: {best['demographic_parity_difference']:.4f}, "
                f"Equalized Odds Difference: {best['equalized_odds_difference']:.4f}, "
                f"Training Time: {best['training_time']:.4f}s)."
            )
        }
        return recommendation
