import numpy as np
import pandas as pd
import pytest

from src.modeling import (
    MODEL_FEATURES,
    RAW_FEATURES,
    bootstrap_metric_intervals,
    fit_model,
    predict_prices,
    prepare_features,
    select_best_degree,
)


def test_prepare_features_has_stable_schema():
    train = pd.read_csv("train.csv", nrows=10)
    features = prepare_features(train)
    assert list(features.columns) == MODEL_FEATURES
    assert len(features) == 10


def test_prepare_features_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required feature columns"):
        prepare_features(pd.DataFrame({RAW_FEATURES[0]: [1]}))


def test_degree_one_model_returns_finite_nonnegative_prices():
    train = pd.read_csv("train.csv", nrows=100)
    fitted = fit_model(prepare_features(train), train["SalePrice"], degree=1)
    predictions = predict_prices(fitted, prepare_features(train.head(5)))
    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()


def test_selection_and_bootstrap_are_reproducible():
    results = [{"degree": 1, "CV_RMSE": 10.0}, {"degree": 2, "CV_RMSE": 12.0}]
    assert select_best_degree(results) == 1
    actual = pd.Series([100.0, 120.0, 140.0, 160.0])
    predicted = np.array([105.0, 115.0, 150.0, 155.0])
    first = bootstrap_metric_intervals(actual, predicted, n_resamples=100)
    second = bootstrap_metric_intervals(actual, predicted, n_resamples=100)
    assert first == second
