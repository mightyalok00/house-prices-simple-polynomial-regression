# House Prices — Polynomial Degree Comparison and Selection

A clean, beginner-friendly machine-learning portfolio project using the Kaggle **House Prices: Advanced Regression Techniques** dataset. It trains polynomial regression at degrees 1 and 2, compares both fairly, and selects the better-generalizing degree without introducing advanced models.

## Project goal
Predict `SalePrice` with carefully selected numeric and domain-derived housing features while demonstrating a correct regression workflow: inspect the data, document outlier handling, prevent leakage, compare polynomial degrees, evaluate residuals, and create a Kaggle-format submission.

## Why this project is intentionally simple
This repository is built to prove understanding rather than hide the workflow behind advanced algorithms. It uses:

- `PolynomialFeatures(degree=1)` and `PolynomialFeatures(degree=2)`
- `LinearRegression`
- one reproducible 80/20 holdout plus five-fold cross-validation
- median imputation learned from the training data
- standardized polynomial terms for numerical stability
- a `log1p(SalePrice)` target that is converted back to dollars
- 16 interpretable model features derived from 20 raw numeric columns

It does **not** use Random Forest, XGBoost, LightGBM, CatBoost, neural networks, automated feature selection, hyperparameter tuning, or stacked models.

## Data used

| Feature group | Examples |
|---|---|
| Quality and condition | `OverallQual`, `OverallCond` |
| Size and rooms | `GrLivArea`, `TotalSF`, `TotRmsAbvGrd`, `BedroomAbvGr` |
| Garage and amenities | `GarageCars`, `GarageArea`, `Fireplaces` |
| Age | `HouseAge`, `RemodelAge` |
| Engineered totals | `TotalBathrooms`, `TotalPorchSF` |
| Presence indicators | `HasGarage`, `HasBasement`, `HasFireplace` |

Degree 1 keeps **16 linear terms**. Degree 2 expands the same inputs into **152 terms**, including squares and pairwise interactions. Two unusually large, low-priced training observations are removed by the documented rule `GrLivArea > 4000 and SalePrice < 300000`.

## Degree 1 versus degree 2

Both degrees use identical rows, features, preprocessing, log target, and five cross-validation folds.

| Degree | 5-fold CV MAE | 5-fold CV RMSE | 5-fold CV R² |
|---:|---:|---:|---:|
| **1** | **$17,661.62** | **$25,784.67** | **0.8925** |
| 2 | $18,089.54 | $27,150.27 | 0.8799 |

The independent 80/20 holdout gives a useful secondary check:

| Degree | Holdout MAE | Holdout RMSE | Holdout R² |
|---:|---:|---:|---:|
| 1 | $17,957.65 | $24,450.04 | 0.8918 |
| **2** | **$17,464.30** | **$24,112.02** | **0.8947** |

## Which degree is best?

**Degree 1 is selected.** Degree 2 is slightly better on one holdout split, but degree 1 has the lower average error and higher R² across five folds. Cross-validation is the primary selection evidence because it tests performance across multiple train/validation partitions and is less dependent on one lucky split.

The result also shows that the 152-term degree-2 model adds complexity without improving average generalization. Degree 1 is therefore the more accurate and parsimonious choice for this feature set.

The comparison also reports the standard deviation and approximate 95% confidence interval of fold RMSE. The selected holdout metrics include reproducible bootstrap confidence intervals, so the evaluation communicates uncertainty instead of presenting point estimates alone.

For Degree 1, the strongest standardized log-price coefficients are `TotalSF` (+0.1461), `HouseAge` (-0.1061), `OverallQual` (+0.0915), `OverallCond` (+0.0641), and `GrLivArea` (+0.0495). These are associations within the fitted model, not causal effects; correlated housing features can share or redistribute coefficient weight.

See the illustrated [degree 1 vs degree 2 comparison report](docs/degree_1_vs_degree_2_comparison.pdf) for the full evidence and decision rationale.

## Selected-model result

Using degree 1 on the reproducible 80/20 holdout:

| Metric | Result |
|---|---:|
| MAE | $17,957.65 |
| RMSE | $24,450.04 |
| R² | 0.8918 |
| Training rows after documented outlier removal | 1458 |
| Validation rows | 292 |

These are local validation results, **not a Kaggle leaderboard score**. The selected model and `polynomial_submission.csv` now use degree 1.

## Repository structure

```text
house-prices-simple-polynomial-regression/
├── train.csv
├── test.csv
├── sample_submission.csv
├── data_description.txt
├── docs/
│   ├── DEEP_PROJECT_ANALYSIS.md
│   ├── degree_1_vs_degree_2_comparison.pdf
│   ├── PROJECT_QUESTIONS.md
│   ├── DATA_DICTIONARY_GUIDE.md
├── house_price_poly_regression_description_questions.docx
├── notebooks/
│   └── House_Prices_Simple_Polynomial_Regression.ipynb
├── outputs/
│   ├── figures/
│   ├── degree_comparison.csv
│   ├── holdout_degree_comparison.csv
│   ├── degree_1_submission.csv
│   ├── degree_2_submission.csv
│   ├── largest_validation_errors.csv
│   ├── metrics.json
│   ├── polynomial_submission.csv
│   ├── selected_model_coefficients.csv
│   └── validation_residuals.csv
├── models/
│   └── selected_polynomial_model.joblib
├── tests/
│   ├── test_app.py
│   └── test_modeling.py
├── src/
│   ├── modeling.py
│   └── train_model.py
├── app.py
├── .env.example
├── .gitignore
├── environment.yml
├── requirements.txt
└── README.md
```

## Setup in Windows / VS Code

```powershell
cd D:\House_Prices_Simple_Polynomial_Regression_Cleaned\house-prices-advanced-regression-techniques
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional local environment file:

```powershell
copy .env.example .env
```

No secret API key is needed for this project.

## Run the project

```powershell
python src\train_model.py
```

This trains both degrees, selects the lowest five-fold CV RMSE, and recreates both submissions, the selected submission, metrics, residual diagnostics, and figures in `outputs/`.

It also saves the validated selected model to `models/selected_polynomial_model.joblib`. The Streamlit app loads this artifact rather than retraining during startup. Standardized coefficients are exported to `outputs/selected_model_coefficients.csv` for transparent Degree 1 interpretation.

Run the automated verification suite with:

```powershell
python -m pytest -q
```

GitHub Actions repeats compilation and tests after every push and pull request.

For a step-by-step learning version, open:

```text
notebooks/House_Prices_Simple_Polynomial_Regression.ipynb
```

## Polynomial regression in plain English
A normal linear model tries to fit a straight relationship. Polynomial regression keeps linear regression as the estimator but first creates extra columns such as `OverallQual²`, `GrLivArea²`, and interactions such as `OverallQual × GrLivArea`. This lets the fitted surface bend while remaining easy to inspect and explain.

## Data-leakage protection
Each holdout or cross-validation fold fits its median imputer, polynomial transformer, and scaler on training rows only, then applies them to validation rows. This prevents validation data from influencing preprocessing.

## Portfolio talking points
In an interview, explain why the target is log-transformed, how the domain features summarize usable space and age, and why five-fold results are more dependable than one holdout. Be candid that degree 2 wins the single holdout narrowly, but degree 1 wins the broader cross-validation comparison and is therefore selected.

## Limitations
This is a learning project, not a competition-winning solution. It still ignores categorical variables, tests only degrees 1 and 2, and uses a manually chosen outlier rule. Polynomial terms can also extrapolate poorly outside the training range. These limitations are documented rather than hidden.

## Deep analysis
See [`docs/DEEP_PROJECT_ANALYSIS.md`](docs/DEEP_PROJECT_ANALYSIS.md) for the detailed data-science reasoning, assumptions, model interpretation, limitations, interview explanation, and improvement roadmap.

## Dataset attribution
The raw files come from Kaggle's House Prices competition. Respect Kaggle's dataset/competition terms when redistributing or publishing data.


## 🖥️ Run the Streamlit app

After installing the requirements, launch the interactive prediction app:

```powershell
python -m streamlit run app.py
```

The app reads the selected degree from `outputs/metrics.json` and trains that polynomial model. It includes the preserved **emoji price filter** (`🌱`, `🏡`, `✨`, `👑`) for exploring comparable training homes. The filter changes only the displayed examples; it does not change or retrain the model.
