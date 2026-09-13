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
        results.append(
            {
                "degree": degree,
                "CV_MAE": round(float(np.mean([m["MAE"] for m in fold_metrics])), 2),
                "CV_RMSE": round(
                    float(np.mean([m["RMSE"] for m in fold_metrics])), 2
                ),
                "CV_R2": round(float(np.mean([m["R2"] for m in fold_metrics])), 4),
                "folds": n_splits,
            }
        )
    return results
