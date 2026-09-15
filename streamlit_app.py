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

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .app-hero {
        padding: 1.25rem 1.4rem;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 18px;
        margin-bottom: 1.2rem;
    }

    .app-hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: clamp(1.8rem, 4vw, 2.6rem);
        line-height: 1.15;
    }

    .app-hero p {
        margin: 0;
        opacity: 0.78;
        font-size: 0.98rem;
    }

    .section-card {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 16px;
        padding: 1.1rem 1.15rem 0.35rem 1.15rem;
        margin: 0.45rem 0 1rem 0;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 14px;
        padding: 0.85rem 1rem;
        min-height: 110px;
    }

    div[data-testid="stButton"] > button {
        min-height: 3rem;
        border-radius: 12px;
        font-weight: 700;
    }

    .prediction-card {
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 18px;
        padding: 1.15rem 1.25rem;
        text-align: center;
        margin-top: 0.8rem;
    }

    .prediction-card .label {
        font-size: 0.95rem;
        opacity: 0.75;
        margin-bottom: 0.2rem;
    }

    .prediction-card .value {
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 800;
        line-height: 1.1;
        margin: 0.15rem 0 0.35rem 0;
    }

    .prediction-card .note {
        font-size: 0.88rem;
        opacity: 0.7;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        .app-hero,
        .section-card,
        .prediction-card {
            border-radius: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
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
    if artifact.get("model_features") != list(
        prepare_features(pd.read_csv(TRAIN_PATH, nrows=1)).columns
    ):
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

st.markdown(
    f"""
    <div class="app-hero">
        <h1>🏠 House Price Prediction</h1>
        <p>
            Simple Polynomial Regression · Degree {selected_degree} selected using repeated-CV RMSE ·
            Educational portfolio project
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🎛️ App Controls")
    st.caption(f"Active model: Polynomial Regression · Degree {selected_degree}")
    st.divider()

    emoji_filter = st.selectbox(
        "🏷️ Comparable-home price band",
        [
            "🌈 All homes",
            "🌱 Entry — up to $150K",
            "🏡 Mid-range — $150K to $250K",
            "✨ Premium — $250K to $400K",
            "👑 Luxury — above $400K",
        ],
        help=(
            "This only filters the training examples shown below. "
            "It does not change the trained model."
        ),
    )
    show_explorer = st.toggle("🔎 Show comparable homes", value=True)

st.subheader("🧮 Enter House Features")
st.caption("Adjust the property details below, then generate an estimated sale price.")

with st.form("prediction_form", border=True):
    left, right = st.columns(2, gap="large")

    with left:
        overall_qual = st.slider(
            "⭐ Overall Quality",
            min_value=1,
            max_value=10,
            value=int(train_df["OverallQual"].median()),
            help="Overall material and finish quality, where 10 is best.",
        )
        gr_liv_area = st.number_input(
            "📐 Above-ground living area (sq ft)",
            min_value=100,
            max_value=10000,
            value=int(train_df["GrLivArea"].median()),
            step=50,
        )
        total_bsmt_sf = st.number_input(
            "🧱 Basement area (sq ft)",
            min_value=0,
            max_value=7000,
            value=int(train_df["TotalBsmtSF"].median()),
            step=50,
        )
        year_built = st.number_input(
            "📅 Year built",
            min_value=1800,
            max_value=2030,
            value=int(train_df["YearBuilt"].median()),
            step=1,
        )

    with right:
        garage_cars = st.slider(
            "🚗 Garage capacity (cars)",
            min_value=0,
            max_value=5,
            value=int(train_df["GarageCars"].median()),
        )
        full_bath = st.slider(
            "🛁 Full bathrooms",
            min_value=0,
            max_value=5,
            value=int(train_df["FullBath"].median()),
        )
        bedroom_abv_gr = st.slider(
            "🛏️ Bedrooms above ground",
            min_value=0,
            max_value=8,
            value=int(train_df["BedroomAbvGr"].median()),
            key="bedroom_above_ground",
        )
        st.info(
            "Tip: Overall quality and living area are typically among the strongest "
            "signals in this simple regression project."
        )

    submitted = st.form_submit_button(
        "🚀 Predict Sale Price",
        type="primary",
        width="stretch",
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

if submitted:
    prediction = float(predict_prices(fitted_model, prepare_features(input_df))[0])
    st.success("Estimated Sale Price")
    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="label">Estimated Sale Price</div>
            <div class="value">${prediction:,.0f}</div>
            <div class="note">
                Educational estimate from a simple polynomial regression model —
                not a professional real-estate valuation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

if show_explorer:
    st.subheader("🔎 Comparable Training Homes")
    st.caption(
        "Explore the highest-priced training examples inside the selected price band."
    )

    filtered = apply_emoji_price_filter(train_df, emoji_filter)

    m1, m2, m3 = st.columns(3, gap="medium")
    m1.metric("🏘️ Homes shown", f"{len(filtered):,}")
    if len(filtered):
        m2.metric("💵 Median price", f"${filtered[TARGET].median():,.0f}")
        m3.metric("📈 Average price", f"${filtered[TARGET].mean():,.0f}")
    else:
        m2.metric("💵 Median price", "—")
        m3.metric("📈 Average price", "—")

    display_columns = ["Id", TARGET] + APP_INPUT_FEATURES
    display_df = (
        filtered[display_columns]
        .sort_values(TARGET, ascending=False)
        .head(100)
        .rename(
            columns={
                "SalePrice": "Sale Price",
                "OverallQual": "Overall Quality",
                "GrLivArea": "Living Area",
                "GarageCars": "Garage Cars",
                "TotalBsmtSF": "Basement Area",
                "FullBath": "Full Baths",
                "BedroomAbvGr": "Bedrooms",
                "YearBuilt": "Year Built",
            }
        )
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sale Price": st.column_config.NumberColumn(format="$%d"),
            "Living Area": st.column_config.NumberColumn(format="%d sq ft"),
            "Basement Area": st.column_config.NumberColumn(format="%d sq ft"),
        },
    )
    st.caption(
        "The sidebar price filter affects this explorer only; the regression model remains unchanged."
    )

with st.expander("🧠 How this model works"):
    st.markdown(
        """
1. Uses 16 understandable numeric and domain-derived house features.
2. Creates transparent domain features such as total area, house age, and total bathrooms.
3. Replaces missing values with training medians and scales the polynomial terms.
4. Trains polynomial degrees 1 and 2 across 25 matched repeated-CV splits.
5. Fits the selected `LinearRegression` model to `log1p(SalePrice)` and converts predictions back to dollars.

This project intentionally avoids advanced models and hyperparameter tuning so the regression workflow stays clear and explainable.
"""
    )
