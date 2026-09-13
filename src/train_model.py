"""Train and evaluate the improved degree-2 polynomial regression model."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from modeling import (
    MODEL_FEATURES,
    TARGET,
    cross_validate_degrees,
    fit_model,
    outlier_mask,
    predict_prices,
    prepare_features,
    regression_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
OUTPUTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def save_diagnostics(actual: pd.Series, predicted, row_ids: pd.Series) -> None:
    """Save residual data, largest errors, and diagnostic figures."""
    residuals = pd.DataFrame(
        {"Id": row_ids.to_numpy(), "Actual": actual.to_numpy(), "Predicted": predicted}
    )
    residuals["Residual"] = residuals["Actual"] - residuals["Predicted"]
    residuals["AbsoluteError"] = residuals["Residual"].abs()
    residuals.to_csv(OUTPUTS / "validation_residuals.csv", index=False)
    residuals.nlargest(20, "AbsoluteError").to_csv(
        OUTPUTS / "largest_validation_errors.csv", index=False
    )

    plt.figure(figsize=(8, 5))
    plt.scatter(residuals["Predicted"], residuals["Residual"], alpha=0.6)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Predicted SalePrice")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Residuals vs predicted price")
    plt.tight_layout()
    plt.savefig(FIGURES / "residuals_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(residuals["Actual"], residuals["Predicted"], alpha=0.6)
    low = min(residuals["Actual"].min(), residuals["Predicted"].min())
    high = max(residuals["Actual"].max(), residuals["Predicted"].max())
    plt.plot([low, high], [low, high], "--", color="black")
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("Actual vs predicted price")
    plt.tight_layout()
    plt.savefig(FIGURES / "actual_vs_predicted.png", dpi=150)
    plt.close()


def main() -> None:
    required = [ROOT / "train.csv", ROOT / "test.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Kaggle file(s): " + ", ".join(missing))

    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    flagged_outliers = outlier_mask(train)
    modeling_train = train.loc[~flagged_outliers].reset_index(drop=True)

    X = prepare_features(modeling_train)
    y = modeling_train[TARGET]
    row_ids = modeling_train["Id"]
    X_train, X_valid, y_train, y_valid, _, id_valid = train_test_split(
        X, y, row_ids, test_size=0.20, random_state=42
    )

    fitted = fit_model(X_train, y_train, degree=2)
    valid_pred = predict_prices(fitted, X_valid)
    holdout_metrics = regression_metrics(y_valid, valid_pred)
    save_diagnostics(y_valid, valid_pred, id_valid)

    degree_comparison = cross_validate_degrees(X, y, degrees=(1, 2), n_splits=5)
    pd.DataFrame(degree_comparison).to_csv(
        OUTPUTS / "degree_comparison.csv", index=False
    )

    metrics = {
        **holdout_metrics,
        "target_transform": "log1p",
        "validation_method": "80/20 holdout",
        "cross_validation_folds": 5,
        "degree_comparison": degree_comparison,
        "train_rows_after_outlier_removal": int(len(modeling_train)),
        "validation_rows": int(len(X_valid)),
        "removed_outliers": int(flagged_outliers.sum()),
        "raw_features_used": 20,
        "model_features": len(MODEL_FEATURES),
        "polynomial_features": int(fitted[1].n_output_features_),
        "final_degree": 2,
        "random_state": 42,
    }
    (OUTPUTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))

    final_fitted = fit_model(X, y, degree=2)
    test_pred = predict_prices(final_fitted, prepare_features(test))
    submission = pd.DataFrame({"Id": test["Id"], TARGET: test_pred})
    submission.to_csv(OUTPUTS / "polynomial_submission.csv", index=False)
    print(f"Saved: {OUTPUTS / 'polynomial_submission.csv'}")


if __name__ == "__main__":
    main()
