"""Streamlit app for the beginner House Prices polynomial regression project."""
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"

FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
]
TARGET = "SalePrice"

st.set_page_config(
    page_title="House Price Polynomial Regression",
    page_icon="🏠",
    layout="wide",
)


@st.cache_data
def load_training_data() -> pd.DataFrame:
    """Load the original Kaggle training data."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TRAIN_PATH}. Keep train.csv in the project root."
        )
    return pd.read_csv(TRAIN_PATH)


@st.cache_resource
def train_simple_model(train_df: pd.DataFrame):
    """Fit the exact same simple degree-2 polynomial regression used by the project."""
    X = train_df[FEATURES]
    y = train_df[TARGET]

    imputer = SimpleImputer(strategy="median")
    X_clean = imputer.fit_transform(X)

    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = polynomial.fit_transform(X_clean)

    model = LinearRegression()
    model.fit(X_poly, y)
    return imputer, polynomial, model


def apply_emoji_price_filter(df: pd.DataFrame, option: str) -> pd.DataFrame:
    """Filter training examples by an easy-to-read emoji price band."""
    if option == "🌈 All homes":
        return df
    if option == "🌱 Entry — up to $150K":
        return df[df[TARGET] <= 150_000]
    if option == "🏡 Mid-range — $150K to $250K":
        return df[(df[TARGET] > 150_000) & (df[TARGET] <= 250_000)]
    if option == "✨ Premium — $250K to $400K":
        return df[(df[TARGET] > 250_000) & (df[TARGET] <= 400_000)]
    if option == "👑 Luxury — above $400K":
        return df[df[TARGET] > 400_000]
    return df


try:
    train_df = load_training_data()
except Exception as exc:
    st.error(f"Could not load the training data: {exc}")
    st.stop()

imputer, polynomial, model = train_simple_model(train_df)

st.title("🏠 House Price Prediction — Simple Polynomial Regression")
st.caption(
    "Beginner-friendly Streamlit demo using only 6 numeric features, degree-2 "
    "PolynomialFeatures, median imputation, and LinearRegression."
)

with st.sidebar:
    st.header("🎛️ App Controls")
    emoji_filter = st.selectbox(
        "🏷️ Emoji price filter",
        [
            "🌈 All homes",
            "🌱 Entry — up to $150K",
            "🏡 Mid-range — $150K to $250K",
            "✨ Premium — $250K to $400K",
            "👑 Luxury — above $400K",
        ],
        help="This filters the training-data examples shown in the explorer. It does not alter the trained model.",
    )
    show_explorer = st.checkbox("🔎 Show comparable homes", value=True)

st.subheader("🧮 Enter House Features")
col1, col2, col3 = st.columns(3)

with col1:
    overall_qual = st.slider(
        "⭐ Overall Quality (1–10)", 1, 10, int(train_df["OverallQual"].median())
    )
    gr_liv_area = st.number_input(
        "📐 Above-ground living area (sq ft)",
        min_value=100,
        max_value=10000,
        value=int(train_df["GrLivArea"].median()),
        step=50,
    )

with col2:
    garage_cars = st.slider(
        "🚗 Garage capacity (cars)", 0, 5, int(train_df["GarageCars"].median())
    )
    total_bsmt_sf = st.number_input(
        "🧱 Basement area (sq ft)",
        min_value=0,
        max_value=7000,
        value=int(train_df["TotalBsmtSF"].median()),
        step=50,
    )

with col3:
    full_bath = st.slider(
        "🛁 Full bathrooms", 0, 5, int(train_df["FullBath"].median())
    )
    year_built = st.number_input(
        "📅 Year built",
        min_value=1800,
        max_value=2030,
        value=int(train_df["YearBuilt"].median()),
        step=1,
    )

input_df = pd.DataFrame(
    [[overall_qual, gr_liv_area, garage_cars, total_bsmt_sf, full_bath, year_built]],
    columns=FEATURES,
)

if st.button("🚀 Predict Sale Price", type="primary", width="stretch"):
    clean_input = imputer.transform(input_df)
    poly_input = polynomial.transform(clean_input)
    prediction = float(model.predict(poly_input)[0])
    prediction = max(prediction, 0.0)

    st.success(f"### 💰 Estimated Sale Price: ${prediction:,.0f}")
    st.info(
        "This is an educational estimate from a simple polynomial regression model, "
        "not a professional real-estate valuation."
    )

st.divider()

if show_explorer:
    st.subheader("🔎 Comparable Training Homes")
    filtered = apply_emoji_price_filter(train_df, emoji_filter)

    m1, m2, m3 = st.columns(3)
    m1.metric("🏘️ Homes shown", f"{len(filtered):,}")
    if len(filtered):
        m2.metric("💵 Median price", f"${filtered[TARGET].median():,.0f}")
        m3.metric("📈 Average price", f"${filtered[TARGET].mean():,.0f}")
    else:
        m2.metric("💵 Median price", "—")
        m3.metric("📈 Average price", "—")

    display_columns = ["Id", TARGET] + FEATURES
    st.dataframe(
        filtered[display_columns].sort_values(TARGET, ascending=False).head(100),
        width="stretch",
        hide_index=True,
    )
    st.caption("The emoji filter affects this explorer only; the regression model remains unchanged.")

with st.expander("🧠 How this model works"):
    st.markdown(
        """
1. Selects six easy-to-understand numeric house features.
2. Replaces missing feature values with the median from the training data.
3. Expands the six inputs into degree-2 polynomial and interaction terms.
4. Fits a standard `LinearRegression` model.
5. Uses the fitted model to estimate `SalePrice`.

This project intentionally avoids advanced models and hyperparameter tuning.
"""
    )
