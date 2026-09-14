"""Streamlit app for the beginner House Prices polynomial regression project."""
from pathlib import Path
import json

import pandas as pd
import joblib
import streamlit as st

from src.modeling import (
    RAW_FEATURES,
    TARGET,
    predict_prices,
    prepare_features,
)

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "train.csv"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
MODEL_PATH = ROOT / "models" / "selected_polynomial_model.joblib"

APP_INPUT_FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "BedroomAbvGr",
    "YearBuilt",
]

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
            f"Missing {TRAIN_PATH}. Keep the training data in data/train.csv."
        )
    return pd.read_csv(TRAIN_PATH)


def load_selected_degree() -> int:
    """Read the degree selected by the reproducible training comparison."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "Missing outputs/metrics.json. Run python -m src.train_model first."
        )
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    degree = int(metrics["best_degree"])
    if degree not in (1, 2):
        raise ValueError("The selected polynomial degree must be 1 or 2.")
    return degree


@st.cache_resource
def load_selected_model(degree: int):
    """Load and validate the versioned model produced by the training script."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing models/selected_polynomial_model.joblib. "
            "Run python -m src.train_model first."
        )
    artifact = joblib.load(MODEL_PATH)
    if artifact.get("degree") != degree:
        raise ValueError("Saved model degree does not match outputs/metrics.json.")
    if artifact.get("model_features") != list(prepare_features(pd.read_csv(TRAIN_PATH, nrows=1)).columns):
        raise ValueError("Saved model feature schema is incompatible with this app.")
    return artifact["fitted_model"]


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
    selected_degree = load_selected_degree()
except Exception as exc:
    st.error(f"Could not load the training data: {exc}")
    st.stop()

try:
    fitted_model = load_selected_model(selected_degree)
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

st.title("🏠 House Price Prediction — Selected Polynomial Regression")
st.caption(
    f"Degree {selected_degree} was selected using the lowest repeated-CV RMSE. "
    "The model uses domain features, median imputation, scaling, and a log-price target."
)
st.info(f"Active trained model: polynomial regression degree {selected_degree}")

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
    bedroom_abv_gr = st.slider(
        "🛏️ Bedrooms above ground",
        0,
        8,
        int(train_df["BedroomAbvGr"].median()),
        key="bedroom_above_ground",
    )
    year_built = st.number_input(
        "📅 Year built",
        min_value=1800,
        max_value=2030,
        value=int(train_df["YearBuilt"].median()),
        step=1,
    )

input_values = train_df[RAW_FEATURES].median(numeric_only=True).to_dict()
input_values.update(
    {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "FullBath": full_bath,
        "BedroomAbvGr": bedroom_abv_gr,
        "YearBuilt": year_built,
    }
)
input_df = pd.DataFrame([input_values], columns=RAW_FEATURES)

if st.button("🚀 Predict Sale Price", type="primary", width="stretch"):
    prediction = float(
        predict_prices(fitted_model, prepare_features(input_df))[0]
    )

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

    display_columns = ["Id", TARGET] + APP_INPUT_FEATURES
    st.dataframe(
        filtered[display_columns].sort_values(TARGET, ascending=False).head(100),
        width="stretch",
        hide_index=True,
    )
    st.caption("The emoji filter affects this explorer only; the regression model remains unchanged.")

with st.expander("🧠 How this model works"):
    st.markdown(
        """
1. Uses 16 understandable numeric and domain-derived house features.
2. Creates transparent domain features such as total area, house age, and total bathrooms.
3. Replaces missing values with training medians and scales the polynomial terms.
4. Trains both degrees 1 and 2 across 25 matched repeated-CV splits.
5. Fits the selected `LinearRegression` model to `log1p(SalePrice)` and converts predictions back to dollars.

This project intentionally avoids advanced models and hyperparameter tuning.
"""
    )
