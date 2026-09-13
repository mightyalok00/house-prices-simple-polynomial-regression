# Deep Project Analysis — House Prices Simple Polynomial Regression

## 1. Executive summary
This project predicts residential sale price with a deliberately simple polynomial regression model. The source training table contains **1,460 rows and 81 columns**. The target is `SalePrice`. The project uses 20 raw numeric columns to create 16 understandable model features, including total area, house age, remodel age, total bathrooms, porch area, and simple amenity indicators.

The degree-2 transformation expands 16 model features into **152 model inputs**. With a log-transformed target and two documented outliers removed, the held-out 20% validation result is **MAE $17,464.30**, **RMSE $24,112.02**, and **R² 0.8947**. Five-fold cross-validation also compares degrees 1 and 2; degree 1 currently generalizes slightly better, while degree 2 is retained to demonstrate nonlinear and interaction terms.

## 2. Business problem
A property-pricing workflow needs a reasonable estimate of sale price from known house attributes. This project asks: can a small collection of structural and quality variables explain a useful portion of sale-price variation without relying on advanced models?

## 3. Target variable
`SalePrice` is a continuous numeric target, so this is a supervised regression problem. Its mean in the training data is approximately **$180,921**, with a median of **$163,000**. Because expensive properties can create large errors, RMSE is expected to be higher than MAE.

## 4. Feature rationale
- **OverallQual:** quality is strongly related to buyer willingness to pay.
- **GrLivArea:** more usable above-ground space generally increases value.
- **GarageCars:** a simple capacity measure representing garage utility.
- **TotalSF:** combines basement and floor areas into a broad size measure.
- **TotalBathrooms:** represents household convenience while giving half baths appropriate weight.
- **HouseAge / RemodelAge:** express age relative to the sale year rather than as raw calendar dates.
- **HasGarage / HasBasement / HasFireplace:** distinguish absence from a small numeric value.

The project intentionally avoids dozens of categorical variables so the polynomial-regression mechanism remains understandable.

## 5. Why degree 2
Degree 1 is ordinary multiple linear regression. Degree 2 adds both squared effects and interactions. For example, the impact of living area can change as houses become very large, and the value of extra space can interact with overall quality. Degree 2 gives the model limited curvature without the extreme instability that can come from unnecessarily high degrees.

## 6. Data preparation
Two unusually large, low-priced observations are removed using the explicit rule `GrLivArea > 4000 and SalePrice < 300000`. The remaining data is split **80% training / 20% validation** with `random_state=42`. Median imputation, polynomial expansion, and scaling are fit only on training rows and reused on validation. `log1p(SalePrice)` reduces target skew; predictions are converted back to dollars with `expm1`. The same leakage-safe procedure is repeated independently in every cross-validation fold.

## 7. Polynomial expansion
With 16 model features and `degree=2`, `include_bias=False`, scikit-learn creates 152 transformed features. They include the original terms, squared terms, and pairwise interactions. The transformed columns are standardized for numerical stability, and the estimator remains ordinary least-squares `LinearRegression`.

## 8. Evaluation
### MAE: $17,464.30
Mean Absolute Error is the average absolute dollar difference between predicted and actual prices. It is easy to communicate because it remains in dollars.

### RMSE: $24,112.02
Root Mean Squared Error penalizes large misses more strongly than MAE. The gap between RMSE and MAE indicates that some validation observations have substantially larger errors.

### R²: 0.8947
R² indicates that this simple model explains roughly **89.5%** of validation-set variance under this particular split. It should not be interpreted as prediction accuracy in percentage terms.

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
The training script saves every validation residual, the 20 largest absolute errors, an actual-versus-predicted chart, and a residual-versus-predicted chart. Large residuals often reveal luxury houses or properties whose value depends on categorical variables omitted from this feature set.

## 12. Strengths
- Easy to reproduce.
- Easy to explain in an interview.
- No target leakage in preprocessing.
- Clear separation of raw data, source code, notebook, documentation, and outputs.
- Uses a real regression dataset and produces a correctly formatted test submission.
- Keeps the modeling method aligned with the stated polynomial-regression learning objective.

## 13. Limitations
- Uses only selected numeric predictors and does not encode categorical variables.
- Ignores categorical variables such as neighborhood and exterior quality labels.
- Ordinary least squares can be sensitive to outliers and correlated polynomial terms.
- Only degrees 1 and 2 are compared, and degree 1 currently has the stronger average cross-validation result.
- The final degree remains fixed at 2 for the educational objective rather than selected strictly by cross-validation.
- The manually chosen outlier rule may not generalize to another housing dataset.

## 14. Why not use advanced models here?
Using XGBoost or Random Forest would probably improve predictive performance, but it would weaken the educational objective. This repository is meant to show that the author understands how polynomial regression works, how preprocessing is separated between train and validation, and how regression metrics should be interpreted.

## 15. Interview explanation
A concise interview explanation:

> I used 16 understandable numeric and domain-derived housing features with a degree-2 polynomial transformation. I removed two observations using a documented outlier rule, trained on `log1p(SalePrice)`, and learned imputation and scaling only from training rows. On the held-out split I obtained an R² of 0.8947, MAE of about $17,464, and RMSE of about $24,112. Five-fold comparison showed that degree 1 currently generalizes slightly better, but I retained degree 2 because the goal is to demonstrate polynomial interactions rather than competition-level ensemble modeling.

## 16. Improvement roadmap without changing the current project's purpose
Future improvements could repeat cross-validation with alternative outlier rules, add confidence or prediction intervals, and investigate the largest residuals in more detail. Advanced ensemble models should belong in a separate branch or follow-up project so this repository remains focused on polynomial regression.

## 17. Final conclusion
The project demonstrates a correct and transparent polynomial-regression workflow. The validation score is useful for a simple baseline, while the documented limitations show awareness that strong real-world prediction requires broader features and more rigorous validation. For GitHub and entry-level data-science interviews, clarity and reproducibility are the main strengths of this version.
