# Deep Project Analysis — Polynomial Degree 1 vs Degree 2

## 1. Executive summary
This project predicts residential sale price with a deliberately simple polynomial regression model. The source training table contains **1,460 rows and 81 columns**. The target is `SalePrice`. The project uses 20 raw numeric columns to create 16 understandable model features, including total area, house age, remodel age, total bathrooms, porch area, and simple amenity indicators.

The project trains both degree 1 (16 terms) and degree 2 (152 terms). Five-fold cross-validation selects **degree 1** because it has lower average RMSE (**$25,784.67 vs $27,150.27**) and higher average R² (**0.8925 vs 0.8799**). Degree 2 is narrowly better on the single 80/20 holdout, which demonstrates why the broader cross-validation result is a more dependable selection basis.

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

## 5. Why compare degrees 1 and 2
Degree 1 is polynomial regression with only first-order terms and is equivalent to multiple linear regression. Degree 2 adds squared effects and pairwise interactions. The comparison tests whether that extra curvature improves unseen-data performance rather than assuming that a more complex model must be better.

## 6. Data preparation
Two unusually large, low-priced observations are removed using the explicit rule `GrLivArea > 4000 and SalePrice < 300000`. The remaining data is split **80% training / 20% validation** with `random_state=42`. Median imputation, polynomial expansion, and scaling are fit only on training rows and reused on validation. `log1p(SalePrice)` reduces target skew; predictions are converted back to dollars with `expm1`. The same leakage-safe procedure is repeated independently in every cross-validation fold.

## 7. Polynomial expansion
With 16 model features, degree 1 produces 16 terms and degree 2 produces 152 terms with `include_bias=False`. Degree 2 includes the original terms, squared terms, and pairwise interactions. Both versions use identical median imputation, standardization, and log-target treatment; the estimator remains ordinary least-squares `LinearRegression`.

## 8. Evaluation

| Degree | 5-fold CV MAE | 5-fold CV RMSE | 5-fold CV R² | Holdout MAE | Holdout RMSE | Holdout R² |
|---:|---:|---:|---:|---:|---:|---:|
| **1** | **$17,661.62** | **$25,784.67** | **0.8925** | $17,957.65 | $24,450.04 | 0.8918 |
| 2 | $18,089.54 | $27,150.27 | 0.8799 | **$17,464.30** | **$24,112.02** | **0.8947** |

MAE is the average absolute dollar error. RMSE penalizes large misses more strongly. R² measures explained variance and is not a percentage accuracy score.

## 9. Which degree is best?
Degree 1 is the best choice for this dataset and feature set because it wins the primary five-fold comparison. Degree 2 wins only the secondary holdout by a small margin. Selecting degree 1 reduces the transformed feature count from 152 to 16, lowers average validation error, and avoids unnecessary variance. The code therefore trains both degrees, selects the lowest cross-validated RMSE, and uses degree 1 for the final submission and Streamlit prediction model.

## 10. Interpretation caution
Polynomial coefficients are harder to explain one-by-one because each original feature appears in multiple terms and the features are on different scales. For a beginner portfolio, the most defensible interpretation is directional and structural: the transformation allows nonlinear and interaction effects, while the validation metrics measure whether that flexibility improves predictive usefulness.

## 11. Data structures demonstrated
This project also demonstrates basic Python/data-science structures:
- **List:** the selected feature names.
- **Dictionary:** the metrics object saved to JSON.
- **DataFrame:** raw train/test data and the final submission.
- **NumPy arrays:** transformed matrices produced by preprocessing.
- **Path objects:** safe, platform-independent project paths in Python.

## 12. Error analysis
The training script saves every validation residual, the 20 largest absolute errors, an actual-versus-predicted chart, and a residual-versus-predicted chart. Large residuals often reveal luxury houses or properties whose value depends on categorical variables omitted from this feature set.

## 13. Strengths
- Easy to reproduce.
- Easy to explain in an interview.
- No target leakage in preprocessing.
- Clear separation of raw data, source code, notebook, documentation, and outputs.
- Uses a real regression dataset and produces a correctly formatted test submission.
- Keeps the modeling method aligned with the stated polynomial-regression learning objective.

## 14. Limitations
- Uses only selected numeric predictors and does not encode categorical variables.
- Ignores categorical variables such as neighborhood and exterior quality labels.
- Ordinary least squares can be sensitive to outliers and correlated polynomial terms.
- Only degrees 1 and 2 are compared; higher degrees are intentionally excluded to control complexity.
- The manually chosen outlier rule may not generalize to another housing dataset.

## 15. Why not use advanced models here?
Using XGBoost or Random Forest would probably improve predictive performance, but it would weaken the educational objective. This repository is meant to show that the author understands how polynomial regression works, how preprocessing is separated between train and validation, and how regression metrics should be interpreted.

## 16. Interview explanation
A concise interview explanation:

> I trained polynomial regression at degrees 1 and 2 using the same 16 features, preprocessing, log target, and five folds. Degree 2 was slightly better on one holdout, but degree 1 had the lower five-fold CV RMSE, $25,785 versus $27,150, and higher CV R². I therefore selected degree 1 because repeated validation is more reliable than one split and the simpler model generalizes better.

## 17. Improvement roadmap without changing the current project's purpose
Future improvements could repeat cross-validation with alternative outlier rules, add confidence or prediction intervals, and investigate the largest residuals in more detail. Advanced ensemble models should belong in a separate branch or follow-up project so this repository remains focused on polynomial regression.

## 18. Final conclusion
The project demonstrates a correct and transparent polynomial-regression selection workflow. Degree 1 is the evidence-based winner for the current data because it generalizes better across five folds. The documented disagreement with the single holdout is a strength: it shows why model choice should rely on repeated validation rather than the most favorable isolated score.
