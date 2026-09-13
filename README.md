# House Prices — Simple Polynomial Regression

A clean, beginner-friendly machine-learning portfolio project using the Kaggle **House Prices: Advanced Regression Techniques** dataset, but intentionally applying a **simple degree-2 polynomial regression** model rather than advanced ensembles.

## Project goal
Predict `SalePrice` using a small set of easy-to-explain numeric housing features while demonstrating a correct regression workflow: understand the data, split it, handle missing values safely, create polynomial terms, train a linear regression model, evaluate it, and create a Kaggle-format submission.

## Why this project is intentionally simple
This repository is built to prove understanding rather than hide the workflow behind advanced algorithms. It uses:

- `PolynomialFeatures(degree=2)`
- `LinearRegression`
- one reproducible 80/20 train-validation split
- median imputation learned from the training data
- six interpretable numeric features

It does **not** use Random Forest, XGBoost, LightGBM, CatBoost, neural networks, automated feature selection, hyperparameter tuning, or stacked models.

## Selected features

| Feature | Why it is useful |
|---|---|
| `OverallQual` | Overall material and finish quality |
| `GrLivArea` | Above-ground living area |
| `GarageCars` | Garage vehicle capacity |
| `TotalBsmtSF` | Total basement area |
| `FullBath` | Number of full bathrooms |
| `YearBuilt` | Approximate property age/newness |

Degree 2 expands these 6 inputs into **27 polynomial terms**, including squared terms and pairwise interactions.

## Verified validation result
Using `random_state=42` and an 80/20 split:

| Metric | Result |
|---|---:|
| MAE | $21,267.85 |
| RMSE | $32,248.79 |
| R² | 0.8644 |
| Training rows | 1168 |
| Validation rows | 292 |

These are local validation results, **not a Kaggle leaderboard score**. The purpose is to show a transparent polynomial-regression baseline.

## Repository structure

```text
house-prices-advanced-regression-techniques/
├── train.csv
├── test.csv
├── sample_submission.csv
├── data_description.txt
├── docs/
│   ├── DEEP_PROJECT_ANALYSIS.md
│   ├── PROJECT_QUESTIONS.md
│   ├── DATA_DICTIONARY_GUIDE.md
├── house_price_poly_regression_description_questions.docx
├── notebooks/
│   └── House_Prices_Simple_Polynomial_Regression.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics.json
│   └── polynomial_submission.csv
├── src/
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

This recreates `outputs/metrics.json` and `outputs/polynomial_submission.csv`.

For a step-by-step learning version, open:

```text
notebooks/House_Prices_Simple_Polynomial_Regression.ipynb
```

## Polynomial regression in plain English
A normal linear model tries to fit a straight relationship. Polynomial regression keeps linear regression as the estimator but first creates extra columns such as `OverallQual²`, `GrLivArea²`, and interactions such as `OverallQual × GrLivArea`. This lets the fitted surface bend while remaining easy to inspect and explain.

## Data-leakage protection
The validation workflow fits the median imputer and polynomial transformer on the training split only, then applies those learned transformations to the validation split. This is important because the validation data should not influence training-time preprocessing.

## Portfolio talking points
In an interview, explain that you deliberately chose a small feature set and degree 2 because the goal was to demonstrate polynomial regression clearly. Mention the difference between MAE, RMSE, and R², why an 80/20 split is used, why preprocessing is learned from training data only, and why a high-degree polynomial can overfit.

## Limitations
This is a learning baseline, not a competition-winning solution. It ignores many categorical variables, uses only one validation split, does not tune the degree, and can be influenced by outliers. Those limitations are documented intentionally rather than hidden.

## Deep analysis
See [`docs/DEEP_PROJECT_ANALYSIS.md`](docs/DEEP_PROJECT_ANALYSIS.md) for the detailed data-science reasoning, assumptions, model interpretation, limitations, interview explanation, and improvement roadmap.

## Dataset attribution
The raw files come from Kaggle's House Prices competition. Respect Kaggle's dataset/competition terms when redistributing or publishing data.


## 🖥️ Run the Streamlit app

After installing the requirements, launch the interactive prediction app:

```powershell
python -m streamlit run app.py
```

The app uses the same beginner-friendly degree-2 polynomial regression approach as the training script. It includes an **emoji price filter** (`🌱`, `🏡`, `✨`, `👑`) for exploring comparable training homes. The filter changes only the displayed examples; it does not change or retrain the model.
