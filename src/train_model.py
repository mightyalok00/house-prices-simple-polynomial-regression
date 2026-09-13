"""Beginner-friendly degree-2 polynomial regression for Kaggle House Prices.

This intentionally uses a small numeric feature set and plain LinearRegression.
It avoids advanced models, cross-validation, hyperparameter tuning, and complex feature engineering.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

FEATURES = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
TARGET = "SalePrice"

def main():
    required = [DATA / "train.csv", DATA / "test.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Kaggle file(s): " + ", ".join(missing))

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    X = train[FEATURES]
    y = train[TARGET]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Simple missing-value treatment: learn training medians, then reuse them.
    imputer = SimpleImputer(strategy="median")
    X_train_clean = imputer.fit_transform(X_train)
    X_valid_clean = imputer.transform(X_valid)

    # Degree 2 adds squared terms and pairwise interactions.
    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = polynomial.fit_transform(X_train_clean)
    X_valid_poly = polynomial.transform(X_valid_clean)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    valid_pred = model.predict(X_valid_poly)

    metrics = {
        "MAE": round(float(mean_absolute_error(y_valid, valid_pred)), 2),
        "RMSE": round(float(mean_squared_error(y_valid, valid_pred) ** 0.5), 2),
        "R2": round(float(r2_score(y_valid, valid_pred)), 4),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "original_features": len(FEATURES),
        "polynomial_features": int(X_train_poly.shape[1]),
        "degree": 2,
        "random_state": 42,
    }
    (OUTPUTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))

    # Refit the same simple method on all training rows for Kaggle test predictions.
    full_imputer = SimpleImputer(strategy="median")
    X_full = full_imputer.fit_transform(X)
    X_test = full_imputer.transform(test[FEATURES])
    full_polynomial = PolynomialFeatures(degree=2, include_bias=False)
    X_full_poly = full_polynomial.fit_transform(X_full)
    X_test_poly = full_polynomial.transform(X_test)
    final_model = LinearRegression()
    final_model.fit(X_full_poly, y)
    test_pred = np.maximum(final_model.predict(X_test_poly), 0)

    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": test_pred})
    submission.to_csv(OUTPUTS / "polynomial_submission.csv", index=False)
    print(f"Saved: {OUTPUTS / 'polynomial_submission.csv'}")

if __name__ == "__main__":
    main()
