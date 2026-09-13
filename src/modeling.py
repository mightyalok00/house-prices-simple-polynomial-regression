"""Shared data preparation and polynomial-regression helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

TARGET = "SalePrice"
RAW_FEATURES = [
    "OverallQual",
    "OverallCond",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotRmsAbvGrd",
    "Fireplaces",
    "BedroomAbvGr",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "HalfBath",
    "YearBuilt",
    "YearRemodAdd",
    "YrSold",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
]
MODEL_FEATURES = [
    "OverallQual",
    "OverallCond",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotRmsAbvGrd",
    "Fireplaces",
    "BedroomAbvGr",
    "TotalSF",
    "HouseAge",
    "RemodelAge",
    "TotalBathrooms",
    "TotalPorchSF",
    "HasGarage",
    "HasBasement",
    "HasFireplace",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact numeric feature set with transparent domain features."""
    missing = [column for column in RAW_FEATURES if column not in df.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))

    features = df[RAW_FEATURES].copy()
    features["TotalSF"] = (
        features["TotalBsmtSF"]
        + features["1stFlrSF"]
        + features["2ndFlrSF"]
    )
    features["HouseAge"] = (features["YrSold"] - features["YearBuilt"]).clip(
        lower=0
    )
    features["RemodelAge"] = (
        features["YrSold"] - features["YearRemodAdd"]
    ).clip(lower=0)
    features["TotalBathrooms"] = features["FullBath"] + 0.5 * features["HalfBath"]
    features["TotalPorchSF"] = features[
        ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    ].sum(axis=1)
    features["HasGarage"] = (features["GarageArea"].fillna(0) > 0).astype(int)
    features["HasBasement"] = (features["TotalBsmtSF"].fillna(0) > 0).astype(int)
    features["HasFireplace"] = (features["Fireplaces"].fillna(0) > 0).astype(int)
    return features[MODEL_FEATURES]


def outlier_mask(df: pd.DataFrame) -> pd.Series:
    """Flag unusually large, low-priced houses using a documented Kaggle rule."""
    return (df["GrLivArea"] > 4_000) & (df[TARGET] < 300_000)


def fit_model(X: pd.DataFrame, y: pd.Series, degree: int = 2):
    """Fit log-target polynomial regression without an sklearn Pipeline."""
    imputer = SimpleImputer(strategy="median")
    polynomial = PolynomialFeatures(degree=degree, include_bias=False)
    scaler = StandardScaler()

    clean = imputer.fit_transform(X)
    expanded = polynomial.fit_transform(clean)
    scaled = scaler.fit_transform(expanded)

    model = LinearRegression()
    model.fit(scaled, np.log1p(y))
    return imputer, polynomial, scaler, model


def predict_prices(fitted, X: pd.DataFrame) -> np.ndarray:
    """Transform features and return non-negative prices on the dollar scale."""
    imputer, polynomial, scaler, model = fitted
    clean = imputer.transform(X)
    expanded = polynomial.transform(clean)
    scaled = scaler.transform(expanded)
    return np.maximum(np.expm1(model.predict(scaled)), 0.0)


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Calculate easy-to-interpret metrics on the original dollar scale."""
    return {
        "MAE": round(float(mean_absolute_error(actual, predicted)), 2),
        "RMSE": round(float(mean_squared_error(actual, predicted) ** 0.5), 2),
        "R2": round(float(r2_score(actual, predicted)), 4),
    }


def cross_validate_degrees(
    X: pd.DataFrame, y: pd.Series, degrees: tuple[int, ...] = (1, 2), n_splits: int = 5
) -> list[dict[str, float | int]]:
    """Compare polynomial degrees with leakage-safe manual cross-validation."""
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results: list[dict[str, float | int]] = []
    for degree in degrees:
        fold_metrics = []
        for train_index, valid_index in splitter.split(X):
            fitted = fit_model(X.iloc[train_index], y.iloc[train_index], degree=degree)
            predicted = predict_prices(fitted, X.iloc[valid_index])
            fold_metrics.append(regression_metrics(y.iloc[valid_index], predicted))
        rmse_values = np.array([m["RMSE"] for m in fold_metrics], dtype=float)
        rmse_margin = 1.96 * float(rmse_values.std(ddof=1)) / np.sqrt(n_splits)
        results.append(
            {
                "degree": degree,
                "CV_MAE": round(float(np.mean([m["MAE"] for m in fold_metrics])), 2),
                "CV_RMSE": round(
                    float(rmse_values.mean()), 2
                ),
                "CV_RMSE_std": round(float(rmse_values.std(ddof=1)), 2),
                "CV_RMSE_95CI_low": round(float(rmse_values.mean() - rmse_margin), 2),
                "CV_RMSE_95CI_high": round(float(rmse_values.mean() + rmse_margin), 2),
                "CV_R2": round(float(np.mean([m["R2"] for m in fold_metrics])), 4),
                "folds": n_splits,
            }
        )
    return results


def bootstrap_metric_intervals(
    actual: pd.Series,
    predicted: np.ndarray,
    n_resamples: int = 2_000,
    random_state: int = 42,
) -> dict[str, list[float]]:
    """Return reproducible 95% bootstrap intervals for holdout MAE and RMSE."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if len(actual_values) != len(predicted_values) or len(actual_values) == 0:
        raise ValueError("Actual and predicted values must have the same non-zero length.")
    rng = np.random.default_rng(random_state)
    mae_samples = np.empty(n_resamples)
    rmse_samples = np.empty(n_resamples)
    for index in range(n_resamples):
        sample = rng.integers(0, len(actual_values), len(actual_values))
        errors = actual_values[sample] - predicted_values[sample]
        mae_samples[index] = np.mean(np.abs(errors))
        rmse_samples[index] = np.sqrt(np.mean(errors**2))
    return {
        "MAE_95CI": [round(float(v), 2) for v in np.percentile(mae_samples, [2.5, 97.5])],
        "RMSE_95CI": [round(float(v), 2) for v in np.percentile(rmse_samples, [2.5, 97.5])],
    }


def select_best_degree(
    results: list[dict[str, float | int]], metric: str = "CV_RMSE"
) -> int:
    """Select the degree with the lowest cross-validated error metric."""
    if not results:
        raise ValueError("At least one degree result is required.")
    if metric not in results[0]:
        raise ValueError(f"Unknown selection metric: {metric}")
    return int(min(results, key=lambda result: float(result[metric]))["degree"])
