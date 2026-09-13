# Project Objective and Questions

## Objective
Build and explain a simple degree-2 polynomial regression model that predicts `SalePrice` from a small, interpretable set of numeric house attributes, while demonstrating core Python data structures and a clean train/validation workflow.

## 12 portfolio questions
1. What are the dataset dimensions, target variable, and basic data types?
2. What are the mean, median, minimum, maximum, and spread of `SalePrice`?
3. Which numeric and domain-derived features will be selected for the simple model, and why?
4. Are there missing values in the selected training and test features?
5. How can a Python list be used to manage the selected feature names cleanly?
6. What does an 80/20 train-validation split accomplish, and why use `random_state=42`?
7. What new terms are created by a degree-2 polynomial transformation?
8. How many model features exist before and after polynomial expansion?
9. How is plain `LinearRegression` fitted to the polynomial feature matrix?
10. What do MAE, RMSE, and R² say about validation performance?
11. How can a dictionary store model metrics and be exported as JSON?
12. How can the trained method be refit on all training rows to produce a Kaggle-format `Id,SalePrice` submission?
