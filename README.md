# House Price Prediction with Polynomial Regression

[![Live Streamlit App](https://img.shields.io/badge/Live_App-Open_in_Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://house-prices-simple-polynomial-regression.streamlit.app/)
[![Quality checks](https://github.com/mightyalok00/house-prices-simple-polynomial-regression/actions/workflows/quality.yml/badge.svg)](https://github.com/mightyalok00/house-prices-simple-polynomial-regression/actions/workflows/quality.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end data science project that predicts residential sale prices using a deliberately focused polynomial regression workflow. The project compares polynomial degrees 1 and 2 under identical validation conditions, selects the better-generalizing model, and serves predictions through a public Streamlit application.

## Live application

**[Launch the House Price Prediction app](https://house-prices-simple-polynomial-regression.streamlit.app/)**

The app provides seven intuitive property inputs—including bedrooms above ground—an estimated sale price, and an emoji-based price filter for exploring comparable homes. It loads the validated model artifact at startup and does not retrain during user interaction.

> **Production model:** Polynomial regression **degree 1**, selected using the lowest RMSE across 25 matched repeated cross-validation splits.

## Project objective

The goal is to demonstrate a transparent and reproducible regression workflow rather than maximize leaderboard performance with advanced models. The repository covers:

- domain-informed feature engineering;
- documented outlier handling;
- leakage-safe preprocessing;
- matched evaluation of polynomial degrees 1 and 2;
- residual, uncertainty, and coefficient analysis;
- reproducible model and submission artifacts; and
- a tested Streamlit prediction interface.

No Random Forest, XGBoost, LightGBM, CatBoost, neural network, stacking, or automated model-selection framework is used.

## Modeling approach

The workflow uses 20 raw numeric columns to create 16 interpretable model features covering property quality, usable area, age, bathrooms, porches, garages, basements, and fireplaces.

The estimator pipeline contains:

1. median imputation learned only from the current training fold;
2. `PolynomialFeatures` with degree 1 or degree 2;
3. standardization for numerical stability;
4. `LinearRegression`; and
5. a `log1p(SalePrice)` target transformed back into dollars for evaluation and prediction.

Two unusually large, low-priced observations are removed using the documented rule `GrLivArea > 4000 and SalePrice < 300000`.

Degree 1 produces **16 terms**. Degree 2 expands the same inputs to **152 terms**, including squared features and pairwise interactions.

## Model evaluation

Both degrees are evaluated on the same rows, features, preprocessing steps, log target, and random splits. Model selection uses five-times repeated five-fold cross-validation, producing 25 matched validation results per degree.

### Repeated cross-validation

| Degree | MAE | RMSE | R² |
|---:|---:|---:|---:|
| **1** | **$17,685.81** | **$25,773.41** | **0.8930** |
| 2 | $18,238.90 | $27,335.27 | 0.8795 |

### Independent 80/20 holdout

| Degree | MAE | RMSE | R² |
|---:|---:|---:|---:|
| 1 | $17,957.65 | $24,450.04 | 0.8918 |
| **2** | **$17,464.30** | **$24,112.02** | **0.8947** |

## Selection decision

**Degree 1 is the selected model.** Although degree 2 performs slightly better on the single holdout split, degree 1 provides stronger and more stable evidence across repeated validation:

- degree 1 wins **84%** of the 25 paired splits;
- its average RMSE is **$1,561.86 lower**;
- the approximate 95% confidence interval for that improvement is **$972.61 to $2,151.10**; and
- it uses 16 terms instead of 152, reducing unnecessary complexity.

The decision therefore favors the more accurate and parsimonious model under the primary selection methodology. The illustrated [degree comparison report](docs/degree_1_vs_degree_2_comparison.pdf) provides the full evidence and rationale.

## Model interpretation

The strongest standardized associations in the selected degree-1 log-price model are:

| Feature | Standardized coefficient |
|---|---:|
| `TotalSF` | +0.1461 |
| `HouseAge` | -0.1061 |
| `OverallQual` | +0.0915 |
| `OverallCond` | +0.0641 |
| `GrLivArea` | +0.0495 |

These coefficients describe associations within the fitted model, not causal effects. Correlated housing features may share or redistribute coefficient weight.

## Repository structure

```text
├── data/                   # Kaggle source data and field descriptions
├── docs/                   # Technical reports and project documentation
├── models/                 # Validated selected-model artifact
├── notebooks/              # Executable analysis notebook
├── outputs/
│   ├── comparisons/        # Degree and repeated-CV results
│   ├── diagnostics/        # Residuals, coefficients, errors, and figures
│   ├── submissions/        # Degree-specific and selected submissions
│   └── metrics.json        # Machine-readable evaluation summary
├── src/                    # Feature engineering, modeling, and training code
├── tests/                  # Model, artifact, and Streamlit tests
├── streamlit_app.py        # Public application entry point
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Notebook, plotting, and testing dependencies
└── LICENSE                 # MIT license for the project code
```

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the development dependencies and reproduce the model artifacts:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m src.train_model
```

Launch the app:

```bash
python -m streamlit run streamlit_app.py
```

Run the automated verification suite:

```bash
python -m pytest -q
```

The project requires no API keys or secrets. GitHub Actions runs compilation and automated tests after every push and pull request.

## Outputs and documentation

- [Deep project analysis](docs/DEEP_PROJECT_ANALYSIS.md)
- [Degree 1 versus degree 2 report](docs/degree_1_vs_degree_2_comparison.pdf)
- [Project questions and answers](docs/PROJECT_QUESTIONS.md)
- [Data dictionary guide](docs/DATA_DICTIONARY_GUIDE.md)
- [Selected metrics](outputs/metrics.json)
- [Selected Kaggle-format submission](outputs/submissions/polynomial_submission.csv)

## Limitations

This is an educational portfolio project, not a professional property valuation service or a competition-winning system. It uses numeric features only, evaluates two polynomial degrees, and applies a manually defined outlier rule. Polynomial models can extrapolate poorly beyond the training distribution. Reported results are local validation metrics, not a Kaggle leaderboard score.

## Dataset attribution

The data originates from Kaggle's **House Prices: Advanced Regression Techniques** competition. Use and redistribution remain subject to Kaggle's competition and dataset terms.

## License

The project code is available under the [MIT License](LICENSE). The included Kaggle data remains subject to Kaggle's competition and dataset terms.
