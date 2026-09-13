# Deep Project Analysis — House Prices Simple Polynomial Regression

## 1. Executive summary
This project predicts residential sale price with a deliberately simple polynomial regression baseline. The source training table contains **1,460 rows and 81 columns**. The target is `SalePrice`. Instead of attempting to use every available variable, the project chooses six numeric features that are understandable to a recruiter, interviewer, and beginner data scientist.

The degree-2 transformation expands six raw features into **27 model inputs**. On the held-out 20% validation split, the current reproducible result is **MAE $21,267.85**, **RMSE $32,248.79**, and **R² 0.8644**.

## 2. Business problem
A property-pricing workflow needs a reasonable estimate of sale price from known house attributes. This project asks: can a small collection of structural and quality variables explain a useful portion of sale-price variation without relying on advanced models?

## 3. Target variable
`SalePrice` is a continuous numeric target, so this is a supervised regression problem. Its mean in the training data is approximately **$180,921**, with a median of **$163,000**. Because expensive properties can create large errors, RMSE is expected to be higher than MAE.

## 4. Feature rationale
- **OverallQual:** quality is strongly related to buyer willingness to pay.
- **GrLivArea:** more usable above-ground space generally increases value.
- **GarageCars:** a simple capacity measure representing garage utility.
- **TotalBsmtSF:** captures basement size and extra usable/storage space.
- **FullBath:** represents household convenience and floor-plan utility.
- **YearBuilt:** approximates age, construction era, and often condition/modernity.

The project intentionally avoids dozens of categorical variables so the polynomial-regression mechanism remains understandable.

## 5. Why degree 2
Degree 1 is ordinary multiple linear regression. Degree 2 adds both squared effects and interactions. For example, the impact of living area can change as houses become very large, and the value of extra space can interact with overall quality. Degree 2 gives the model limited curvature without the extreme instability that can come from unnecessarily high degrees.

## 6. Data preparation
The training data is split **80% training / 20% validation** with `random_state=42`. Median imputation is fit only on the training split and reused on validation. This avoids leaking validation information into preprocessing. In the Kaggle test file, `GarageCars` and `TotalBsmtSF` each contain one missing value, so median imputation also prevents prediction failure.

## 7. Polynomial expansion
With six raw features and `degree=2`, `include_bias=False`, scikit-learn creates 27 transformed features. They include the original six terms, six squared terms, and pairwise interactions. The estimator itself remains ordinary least-squares `LinearRegression`.

## 8. Evaluation
### MAE: $21,267.85
Mean Absolute Error is the average absolute dollar difference between predicted and actual prices. It is easy to communicate because it remains in dollars.

### RMSE: $32,248.79
Root Mean Squared Error penalizes large misses more strongly than MAE. The gap between RMSE and MAE indicates that some validation observations have substantially larger errors.

### R²: 0.8644
R² indicates that this simple model explains roughly **86.6%** of validation-set variance under this particular split. It should not be interpreted as prediction accuracy in percentage terms.

## 9. Interpretation caution
Polynomial coefficients are harder to explain one-by-one because each original feature appears in multiple terms and the features are on different scales. For a beginner portfolio, the most defensible interpretation is directional and structural: the transformation allows nonlinear and interaction effects, while the validation metrics measure whether that flexibility improves predictive usefulness.

## 10. Data structures demonstrated
This project also demonstrates basic Python/data-science structures:
- **List:** the selected feature names.
- **Dictionary:** the metrics object saved to JSON.
- **DataFrame:** raw train/test data and the final submission.
- **NumPy arrays:** transformed matrices produced by preprocessing.
- **Path objects:** safe, platform-independent project paths in Python.

## 11. Error analysis ideas
A strong manual follow-up is to create a validation DataFrame with actual price, predicted price, absolute error, and percentage error; then sort from largest error to smallest. Large residuals often reveal unusual luxury houses, extreme living-area values, or properties whose value depends on categorical variables omitted from this simple feature set.

## 12. Strengths
- Easy to reproduce.
- Easy to explain in an interview.
- No target leakage in preprocessing.
- Clear separation of raw data, source code, notebook, documentation, and outputs.
- Uses a real regression dataset and produces a correctly formatted test submission.
- Keeps the modeling method aligned with the stated polynomial-regression learning objective.

## 13. Limitations
- Uses only six of the available predictors.
- Ignores categorical variables such as neighborhood and exterior quality labels.
- A single train-validation split is less stable than cross-validation.
- Ordinary least squares can be sensitive to outliers and correlated polynomial terms.
- The degree is fixed at 2 rather than selected empirically.
- The Kaggle competition typically rewards stronger models and log-scale treatment; those techniques are outside this project's simple-regression scope.

## 14. Why not use advanced models here?
Using XGBoost or Random Forest would probably improve predictive performance, but it would weaken the educational objective. This repository is meant to show that the author understands how polynomial regression works, how preprocessing is separated between train and validation, and how regression metrics should be interpreted.

## 15. Interview explanation
A concise interview explanation:

> I used six understandable numeric housing features and a degree-2 polynomial transformation. I split the training data 80/20, learned median replacements only from the training part, transformed the features into squared and interaction terms, and fitted ordinary linear regression. On the held-out split I obtained an R² of 0.8644, MAE of about $21,268, and RMSE of about $32,249. I kept the method intentionally simple because the goal was to demonstrate polynomial regression rather than competition-level ensemble modeling.

## 16. Improvement roadmap without changing the current project's purpose
If this project is later extended, improvements should be clearly separated from the baseline: compare degree 1 vs degree 2, examine residuals, test log-transformed `SalePrice`, add a few carefully chosen features, then introduce cross-validation. Advanced ensemble models should belong in a separate branch or follow-up project so the simple polynomial baseline remains intact.

## 17. Final conclusion
The project demonstrates a correct and transparent polynomial-regression workflow. The validation score is useful for a simple baseline, while the documented limitations show awareness that strong real-world prediction requires broader features and more rigorous validation. For GitHub and entry-level data-science interviews, clarity and reproducibility are the main strengths of this version.
