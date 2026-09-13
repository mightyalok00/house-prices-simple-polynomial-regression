# Project Objective and Questions

## Objective
Build and compare degree-1 and degree-2 polynomial regression models that predict `SalePrice` from interpretable numeric house attributes, then select the better degree using five-times repeated five-fold cross-validated RMSE.

## 12 portfolio questions
1. What are the dataset dimensions, target variable, and basic data types?
2. What are the mean, median, minimum, maximum, and spread of `SalePrice`?
3. Which numeric and domain-derived features will be selected for the simple model, and why?
4. Are there missing values in the selected training and test features?
5. How can a Python list be used to manage the selected feature names cleanly?
6. What does an 80/20 train-validation split accomplish, and why use `random_state=42`?
7. What terms are created by degree 1 versus degree 2?
8. Why does degree 1 use 16 terms while degree 2 uses 152?
9. How is plain `LinearRegression` fitted to the polynomial feature matrix?
10. What do MAE, RMSE, and R² say about validation performance?
11. How can a dictionary store model metrics and be exported as JSON?
12. Why is degree 1 selected after 25 paired CV splits even though degree 2 is slightly better on one holdout split?
