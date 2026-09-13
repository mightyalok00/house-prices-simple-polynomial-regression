"""Train degrees 1 and 2, then select the better polynomial regression model."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modeling import (
    MODEL_FEATURES,
    RAW_FEATURES,
    TARGET,
    bootstrap_metric_intervals,
    cross_validate_degrees,
    fit_model,
    outlier_mask,
    predict_prices,
    prepare_features,
    regression_metrics,
    select_best_degree,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = ROOT / "models"
OUTPUTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)


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

    degree_comparison = cross_validate_degrees(X, y, degrees=(1, 2), n_splits=5)
    best_degree = select_best_degree(degree_comparison, metric="CV_RMSE")
    pd.DataFrame(degree_comparison).to_csv(
        OUTPUTS / "degree_comparison.csv", index=False
    )

    holdout_comparison = []
    holdout_predictions = {}
    for degree in (1, 2):
        fitted = fit_model(X_train, y_train, degree=degree)
        predicted = predict_prices(fitted, X_valid)
        holdout_predictions[degree] = predicted
        holdout_comparison.append(
            {
                "degree": degree,
                **regression_metrics(y_valid, predicted),
                "polynomial_features": int(fitted[1].n_output_features_),
            }
        )
    pd.DataFrame(holdout_comparison).to_csv(
        OUTPUTS / "holdout_degree_comparison.csv", index=False
    )
    selected_holdout = next(
        result for result in holdout_comparison if result["degree"] == best_degree
    )
    save_diagnostics(y_valid, holdout_predictions[best_degree], id_valid)
    holdout_intervals = bootstrap_metric_intervals(
        y_valid, holdout_predictions[best_degree]
    )

    metrics = {
        "MAE": selected_holdout["MAE"],
        "RMSE": selected_holdout["RMSE"],
        "R2": selected_holdout["R2"],
        "target_transform": "log1p",
        "validation_method": "80/20 holdout",
        "cross_validation_folds": 5,
        "degree_comparison": degree_comparison,
        "holdout_degree_comparison": holdout_comparison,
        "selection_metric": "CV_RMSE",
        "holdout_bootstrap_intervals": holdout_intervals,
        "bootstrap_resamples": 2_000,
        "train_rows_after_outlier_removal": int(len(modeling_train)),
        "validation_rows": int(len(X_valid)),
        "removed_outliers": int(flagged_outliers.sum()),
        "raw_features_used": 20,
        "model_features": len(MODEL_FEATURES),
        "polynomial_features": selected_holdout["polynomial_features"],
        "best_degree": best_degree,
        "final_degree": best_degree,
        "random_state": 42,
    }
    (OUTPUTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))

    test_features = prepare_features(test)
    submissions = {}
    fitted_models = {}
    for degree in (1, 2):
        fitted = fit_model(X, y, degree=degree)
        fitted_models[degree] = fitted
        test_pred = predict_prices(fitted, test_features)
        submission = pd.DataFrame({"Id": test["Id"], TARGET: test_pred})
        submission.to_csv(OUTPUTS / f"degree_{degree}_submission.csv", index=False)
        submissions[degree] = submission

    submission = submissions[best_degree]
    submission.to_csv(OUTPUTS / "polynomial_submission.csv", index=False)

    selected_model = fitted_models[best_degree]
    feature_names = selected_model[1].get_feature_names_out(MODEL_FEATURES)
    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "standardized_log_price_coefficient": selected_model[3].coef_,
            "absolute_coefficient": np.abs(selected_model[3].coef_),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    coefficients.to_csv(OUTPUTS / "selected_model_coefficients.csv", index=False)

    artifact = {
        "degree": best_degree,
        "raw_features": RAW_FEATURES,
        "model_features": MODEL_FEATURES,
        "fitted_model": selected_model,
    }
    model_path = MODELS / "selected_polynomial_model.joblib"
    temporary_path = model_path.with_suffix(".joblib.tmp")
    joblib.dump(artifact, temporary_path)
    temporary_path.replace(model_path)
    print(f"Selected degree {best_degree} using lowest five-fold CV RMSE.")
    print(f"Saved selected submission: {OUTPUTS / 'polynomial_submission.csv'}")


if __name__ == "__main__":
    main()
