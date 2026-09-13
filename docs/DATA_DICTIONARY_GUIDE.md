# Data Dictionary Guide

The authoritative variable descriptions are in the project-root `data_description.txt`. For this simple model, focus on:

| Column | Role | Meaning |
|---|---|---|
| `SalePrice` | Target | Sale price in dollars |
| `OverallQual` | Feature | Overall material and finish quality |
| `GrLivArea` | Feature | Above-ground living area in square feet |
| `GarageCars` | Feature | Garage capacity in cars |
| `TotalBsmtSF` | Feature | Total basement area in square feet |
| `FullBath` | Feature | Full bathrooms above grade |
| `YearBuilt` | Feature | Original construction year |
| `Id` | Identifier | Row/property identifier used in submissions |

Do not treat `Id` as a predictive housing characteristic in this baseline.
