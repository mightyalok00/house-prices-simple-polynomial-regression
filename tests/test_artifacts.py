import json

import joblib
import pandas as pd

from src.modeling import MODEL_FEATURES, RAW_FEATURES


def test_saved_model_matches_metrics_and_schema():
    metrics = json.loads(open("outputs/metrics.json", encoding="utf-8").read())
    artifact = joblib.load("models/selected_polynomial_model.joblib")
    assert artifact["degree"] == metrics["best_degree"] == 1
    assert artifact["raw_features"] == RAW_FEATURES
    assert artifact["model_features"] == MODEL_FEATURES


def test_repeated_cv_outputs_are_complete_and_paired():
    metrics = json.loads(open("outputs/metrics.json", encoding="utf-8").read())
    folds = pd.read_csv("outputs/repeated_cv_fold_results.csv")
    assert len(folds) == 50
    assert set(folds["degree"]) == {1, 2}
    assert not folds.duplicated(["degree", "repeat", "fold"]).any()
    paired = metrics["paired_degree_comparison"]
    assert paired["paired_splits"] == 25
    assert paired["difference_95CI"][0] > 0
