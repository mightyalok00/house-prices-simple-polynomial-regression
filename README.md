# House Prices — Simple Polynomial Regression

A clean, beginner-friendly machine-learning portfolio project using the Kaggle **House Prices: Advanced Regression Techniques** dataset, but intentionally applying a **simple degree-2 polynomial regression** model rather than advanced ensembles.

## Project goal
Predict `SalePrice` with carefully selected numeric and domain-derived housing features while demonstrating a correct regression workflow: inspect the data, document outlier handling, prevent leakage, compare polynomial degrees, evaluate residuals, and create a Kaggle-format submission.

## Why this project is intentionally simple
This repository is built to prove understanding rather than hide the workflow behind advanced algorithms. It uses:

- `PolynomialFeatures(degree=2)`
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

Degree 2 expands the 16 model features into **152 polynomial terms**, including squared terms and pairwise interactions. Two unusually large, low-priced training observations are removed by the documented rule `GrLivArea > 4000 and SalePrice < 300000`.

## Verified validation result
Using `random_state=42` and an 80/20 split:

| Metric | Result |
|---|---:|
| MAE | $17,464.30 |
| RMSE | $24,112.02 |
| R² | 0.8947 |
| Training rows after documented outlier removal | 1458 |
| Validation rows | 292 |

Five-fold cross-validation is also reported for degrees 1 and 2. It honestly shows that degree 1 currently generalizes slightly better, while degree 2 remains the final model because this project specifically demonstrates squared and interaction terms. These are local results, **not a Kaggle leaderboard score**.

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
│   ├── degree_comparison.csv
│   ├── largest_validation_errors.csv
│   ├── metrics.json
│   ├── polynomial_submission.csv
│   └── validation_residuals.csv
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

This recreates the metrics, degree comparison, residual diagnostics, figures, and Kaggle submission in `outputs/`.

For a step-by-step learning version, open:

```text
notebooks/House_Prices_Simple_Polynomial_Regression.ipynb
```

## Polynomial regression in plain English
A normal linear model tries to fit a straight relationship. Polynomial regression keeps linear regression as the estimator but first creates extra columns such as `OverallQual²`, `GrLivArea²`, and interactions such as `OverallQual × GrLivArea`. This lets the fitted surface bend while remaining easy to inspect and explain.

## Data-leakage protection
Each holdout or cross-validation fold fits its median imputer, polynomial transformer, and scaler on training rows only, then applies them to validation rows. This prevents validation data from influencing preprocessing.

## Portfolio talking points
In an interview, explain why the target is log-transformed, how the domain features summarize usable space and age, why preprocessing is learned from training data only, and why five-fold results are more dependable than one holdout. Be candid that degree 1 currently wins cross-validation even though degree 2 is retained for this educational polynomial project.

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

The app uses the same beginner-friendly degree-2 polynomial regression approach as the training script. It includes an **emoji price filter** (`🌱`, `🏡`, `✨`, `👑`) for exploring comparable training homes. The filter changes only the displayed examples; it does not change or retrain the model.
