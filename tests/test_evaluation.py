"""Unit tests for evaluation and recommendation engine."""
import pandas as pd
from src.evaluation.recommendation_engine import RecommendationEngine

def test_recommendation_engine():
    """Verifies that the RecommendationEngine ranks models correctly based on metrics."""
    metrics_list = [
        {
            "model_name": "ModelA",
            "accuracy": 0.85,
            "f1_score": 0.84,
            "roc_auc": 0.88,
            "demographic_parity_difference": 0.25,
            "equalized_odds_difference": 0.30,
            "training_time": 5.0,
            "inference_time": 0.1
        },
        {
            "model_name": "ModelB",
            "accuracy": 0.82,
            "f1_score": 0.81,
            "roc_auc": 0.85,
            "demographic_parity_difference": 0.05,
            "equalized_odds_difference": 0.08,
            "training_time": 1.0,
            "inference_time": 0.02
        }
    ]
    
    engine = RecommendationEngine(
        priorities={"accuracy_weight": 0.4, "fairness_weight": 0.4, "efficiency_weight": 0.2}
    )
    
    df_ranked = engine.rank_models(metrics_list)
    assert not df_ranked.empty
    assert len(df_ranked) == 2
    
    # Model B should rank higher because it is extremely fair and efficient, despite slightly lower accuracy
    # Model B DP diff is 0.05 vs Model A 0.25. Let's check which is ranked #1.
    assert df_ranked.iloc[0]["model_name"] == "ModelB"
    
    recommendation = engine.recommend_best(metrics_list)
    assert recommendation["model_name"] == "ModelB"
    assert "justification" in recommendation
