import calendar
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path("france_real_estate_price_model.joblib")

# These metrics are shown if the saved model does not already contain metrics.
# They correspond to the final training result used for this project.
DEFAULT_MODEL_METRICS = {
    "rows_used": 869761,
    "mae": 73090.51,
    "median_absolute_error": 44534.34,
    "rmse": 122376.68,
    "r2": 0.6448,
}


def install_sklearn_compatibility_shims():
    # The included model was saved with Scikit-learn 1.6.1.
    # Some newer Scikit-learn versions removed this private helper class.
    # Defining it as a list-like object lets joblib unpickle the old model.
    try:
        import sklearn.compose._column_transformer as column_transformer
    except Exception:
        return

    if hasattr(column_transformer, "_RemainderColsList"):
        return

    class _RemainderColsList(list):
        def __init__(self, columns=None, *args, **kwargs):
            if columns is None:
                columns = []
            super().__init__(columns)

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            elif state is not None:
                try:
                    self.clear()
                    self.extend(state)
                except TypeError:
                    pass

    column_transformer._RemainderColsList = _RemainderColsList


st.set_page_config(
    page_title="France Real Estate Price Predictor",
    page_icon="house",
    layout="centered",
)


@st.cache_resource
def load_model_bundle(model_path: Path):
    # Streamlit caches the model so it is not loaded again after every click.
    install_sklearn_compatibility_shims()
    return joblib.load(model_path)


def format_price(value):
    return f"{value:,.0f} EUR".replace(",", " ")


def normalize_postal_code(code_postal):
    # Postal codes are categorical data, so we keep them as 5-character strings.
    digits = "".join(ch for ch in str(code_postal) if ch.isdigit())
    return digits.zfill(5)[:5]


def build_property_dataframe(
    surface,
    rooms,
    property_type,
    code_postal,
    terrain,
    month,
    top_cp,
):
    
    cp = normalize_postal_code(code_postal)
    department = cp[:2]

    # Rare postal codes are grouped by department to avoid unknown categories.
    cp_model = cp if cp in top_cp else f"OTHER_{department}"

    return pd.DataFrame(
        [
            {
                "Surface reelle bati": surface,
                "Nombre pieces principales": rooms,
                "surface_log": np.log1p(surface),
                "terrain_log": np.log1p(min(max(terrain, 0), 10000)),
                "rooms_per_m2": rooms / surface,
                "month": month,
                "Type local": property_type,
                "departement": department,
                "cp_model": cp_model,
            }
        ]
    )


def evaluate_listing_price(listing_price, predicted_price, threshold_percent):
    # Positive difference means the listing is more expensive than the prediction.
    difference_percent = (listing_price - predicted_price) / predicted_price * 100

    if difference_percent > threshold_percent:
        status = "Overvalued"
        status_help = "The listing price is higher than the model estimate."
    elif difference_percent < -threshold_percent:
        status = "Undervalued"
        status_help = "The listing price is lower than the model estimate."
    else:
        status = "Fair price"
        status_help = "The listing price is close to the model estimate."

    return difference_percent, status, status_help


st.title("France Real Estate Price Predictor")
st.caption("Estimate residential property prices in France using a DVF-trained model.")

if not MODEL_PATH.exists():
  
    st.error("Model file not found.")
    st.write(
        "Place `france_real_estate_price_model.joblib` in the same folder as "
        "`app.py`, then restart the app."
    )
    st.stop()

try:
    bundle = load_model_bundle(MODEL_PATH)
except Exception as exc:
    st.error("Could not load the model file.")
    if "_RemainderColsList" in str(exc):
        st.warning(
            "This usually means that Scikit-learn is not the same version as the "
            "one used to save the model. Reinstall the dependencies with "
            "`python -m pip install --force-reinstall -r requirements.txt`."
        )
    st.exception(exc)
    st.stop()

if isinstance(bundle, dict):
    # The training script saves a dictionary containing the model and metadata.
    model = bundle.get("model")
    top_cp = set(bundle.get("top_cp", []))
    metrics = bundle.get("metrics", DEFAULT_MODEL_METRICS)
else:
    # This fallback supports a file that contains only the model object.
    model = bundle
    top_cp = set()
    metrics = DEFAULT_MODEL_METRICS

if model is None:
    st.error("The model bundle does not contain a `model` key.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    threshold_percent = st.slider(
        "Fair price threshold (%)",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
    )
    st.header("Model")
    st.metric("R2 score", f"{metrics.get('r2', 0):.4f}")
    st.metric("MAE", format_price(metrics.get("mae", 0)))
    st.caption(f"Rows used: {metrics.get('rows_used', 0):,}".replace(",", " "))

st.subheader("Property details")

col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox("Property type", ["Appartement", "Maison"])
    surface = st.number_input(
        "Living surface (m2)",
        min_value=10.0,
        max_value=400.0,
        value=65.0,
        step=1.0,
    )
    rooms = st.number_input(
        "Main rooms",
        min_value=1,
        max_value=12,
        value=3,
        step=1,
    )

with col2:
    code_postal = st.text_input("Postal code", value="75015", max_chars=5)
    terrain = st.number_input(
        "Land surface (m2)",
        min_value=0.0,
        max_value=10000.0,
        value=0.0,
        step=10.0,
    )
    month_name = st.selectbox(
        "Transaction month",
        list(calendar.month_name)[1:],
        index=5,
    )

listing_price = st.number_input(
    "Listing price (EUR)",
    min_value=20000.0,
    max_value=2500000.0,
    value=280000.0,
    step=5000.0,
)

if st.button("Predict price", type="primary", use_container_width=True):
    month = list(calendar.month_name).index(month_name)

    # Convert user inputs into a one-row dataframe accepted by the model.
    property_df = build_property_dataframe(
        surface=surface,
        rooms=rooms,
        property_type=property_type,
        code_postal=code_postal,
        terrain=terrain,
        month=month,
        top_cp=top_cp,
    )

    try:
        # The model returns an estimated market price in euros.
        predicted_price = float(model.predict(property_df)[0])
    except Exception as exc:
        st.error("Prediction failed. Check that the app features match the trained model.")
        st.exception(exc)
        st.stop()

    predicted_price = max(predicted_price, 0)

    # Compare the user's listing price with the model estimate.
    difference_percent, status, status_help = evaluate_listing_price(
        listing_price=listing_price,
        predicted_price=predicted_price,
        threshold_percent=threshold_percent,
    )

    predicted_m2 = predicted_price / surface
    listing_m2 = listing_price / surface

    st.subheader("Prediction result")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Estimated price", format_price(predicted_price))
    metric_col2.metric("Listing price", format_price(listing_price))
    metric_col3.metric("Difference", f"{difference_percent:+.1f}%")

    st.info(f"Status: {status}. {status_help}")

    st.write(
        pd.DataFrame(
            [
                {
                    "Predicted price per m2": format_price(predicted_m2),
                    "Listing price per m2": format_price(listing_m2),
                    "Postal code used": normalize_postal_code(code_postal),
                    "Type": property_type,
                }
            ]
        )
    )

    with st.expander("Model input sent to prediction pipeline"):
        st.dataframe(property_df, use_container_width=True)

st.caption(
    "This app provides an estimate, not a certified valuation. Real prices can depend "
    "on condition, floor, view, energy rating, neighborhood micro-location, and market timing."
)
