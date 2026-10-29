"""
Farm Financial Planner
Run: streamlit run app.py

Land units: acres and guntas
Yield units: quintals per acre
Financial values: Indian rupees (₹)
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Farm Financial Planner",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# 2. BRIGHT GREEN FRONTEND THEME
# --------------------------------------------------

st.markdown("""
<style>
    :root {
        --green: #218838;
        --dark-green: #155724;
        --mint: #edf8eb;
        --light-green: #dff2dc;
        --white: #ffffff;
        --text: #203b28;
        --muted: #637968;
        --border: #d5e8d3;
    }

    .stApp {
        background: linear-gradient(
            180deg, #eff9ec 0%, #ffffff 45%, #f5fbf3 100%
        );
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(239, 249, 236, 0.9);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg, #dff2dc 0%, #f0f9ed 100%
        );
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--dark-green);
    }

    h1, h2, h3 {
        color: var(--dark-green);
        letter-spacing: -0.4px;
    }

    p, label {
        color: var(--text);
    }

    .hero-card {
        padding: 30px;
        border-radius: 24px;
        background: linear-gradient(
            120deg, #155724 0%, #218838 55%, #70c879 100%
        );
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 28px rgba(33, 136, 56, 0.16);
    }

    .hero-card .eyebrow {
        color: #dff8dc;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero-card h1 {
        color: white;
        font-size: 38px;
        font-weight: 800;
        margin: 8px 0;
    }

    .hero-card p {
        color: #f0fff0;
        font-size: 16px;
        line-height: 1.6;
    }

    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
    }

    .hero-tag {
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.28);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--border);
        border-left: 5px solid #48a64c;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 5px 18px rgba(29, 91, 39, 0.06);
    }

    [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-weight: 650;
    }

    [data-testid="stMetricValue"] {
        color: var(--dark-green) !important;
        font-weight: 800;
    }

    div[data-testid="stTabs"] button[role="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 12px 16px;
        font-weight: 700;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--dark-green) !important;
        background: #e5f5e2;
        border-bottom-color: var(--green) !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(120deg, #218838, #40a950);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #155724;
        color: white;
        border: none;
    }

    input, textarea, [data-baseweb="select"] > div {
        border-radius: 10px !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: var(--border) !important;
    }

    @media (max-width: 700px) {
        .hero-card {
            padding: 20px;
        }

        .hero-card h1 {
            font-size: 28px;
        }
    }
</style>

<div class="hero-card">
    <div class="eyebrow">
        🌱 Smart Agriculture • Better Planning
    </div>
    <h1>🌾 Farm Financial Planner</h1>
    <p>
        Plan with confidence. Estimate crop yield, cultivation costs,
        revenue and profitability using your farm's land area.
    </p>
    <div class="hero-tags">
        <span class="hero-tag">🌾 Crop Insights</span>
        <span class="hero-tag">📊 Financial Estimates</span>
        <span class="hero-tag">🍃 Sustainable Farming</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Prototype: cultivation costs and selling prices are sample "
    "assumptions, not verified current market rates. Yield estimates "
    "may differ from actual harvests."
)


# --------------------------------------------------
# 3. CROP DATA
# Sample costs: ₹ per acre
# Sample prices: ₹ per quintal
# --------------------------------------------------

CROPS = {
    "Rice": {
        "costs": {
            "Seeds": 3000,
            "Fertilizers & pesticides": 8000,
            "Labour": 12000,
            "Machinery & irrigation": 8000,
            "Other expenses": 4000,
        },
        "sample_yield": 22,
        "price_per_quintal": 2300,
    },
    "Cotton": {
        "costs": {
            "Seeds": 3500,
            "Fertilizers & pesticides": 11000,
            "Labour": 17000,
            "Machinery & irrigation": 9000,
            "Other expenses": 4500,
        },
        "sample_yield": 8,
        "price_per_quintal": 7000,
    },
    "Maize": {
        "costs": {
            "Seeds": 2200,
            "Fertilizers & pesticides": 6500,
            "Labour": 8000,
            "Machinery & irrigation": 6000,
            "Other expenses": 2300,
        },
        "sample_yield": 20,
        "price_per_quintal": 2200,
    },
    "Groundnut": {
        "costs": {
            "Seeds": 4500,
            "Fertilizers & pesticides": 7000,
            "Labour": 11000,
            "Machinery & irrigation": 6500,
            "Other expenses": 3000,
        },
        "sample_yield": 8,
        "price_per_quintal": 6000,
    },
}

LOW_RISK_ROI = 50
MEDIUM_RISK_ROI = 20

GUNTAS_PER_ACRE = 40

# Valid only if the model predicts tonnes per hectare.
TONNE_HA_TO_QUINTAL_ACRE = 4.0468564224


# --------------------------------------------------
# 4. LOAD TRAINED MODEL
# Keep yield_model.joblib beside app.py.
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "yield_model.joblib"

model = None
model_bundle = {}
model_load_error = None

if MODEL_PATH.exists():
    try:
        model_bundle = joblib.load(MODEL_PATH)

        if isinstance(model_bundle, dict) and "model" in model_bundle:
            model = model_bundle["model"]
        else:
            model = model_bundle

    except Exception as exc:
        model_load_error = str(exc)


# --------------------------------------------------
# 5. HELPER FUNCTIONS
# --------------------------------------------------

def rupees(value):
    if value is None:
        return "N/A"
    return f"₹{value:,.0f}"


def percent(value):
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def risk_label(roi):
    if roi is None:
        return "Not available"
    if roi >= LOW_RISK_ROI:
        return "Low"
    if roi >= MEDIUM_RISK_ROI:
        return "Medium"
    return "High"


def land_to_acres(acres, guntas):
    return acres + (guntas / GUNTAS_PER_ACRE)


def land_display(area_acres):
    whole_acres = int(area_acres)
    remaining_guntas = round(
        (area_acres - whole_acres) * GUNTAS_PER_ACRE
    )

    if remaining_guntas == GUNTAS_PER_ACRE:
        whole_acres += 1
        remaining_guntas = 0

    return f"{whole_acres} acres, {remaining_guntas} guntas"


def get_model_features():
    if isinstance(model_bundle, dict):
        features = model_bundle.get("features")
        if isinstance(features, (list, tuple)) and features:
            return list(features)

    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return ["crop", "state", "district", "season", "year"]


def predict_yield_per_acre(crop_name, state, district, season, year):
    if model is None:
        return None, "Model unavailable"

    try:
        features = get_model_features()

        input_values = {
            "crop": crop_name,
            "state": state,
            "district": district,
            "season": season,
            "year": year,
        }

        row = {}

        for feature in features:
            key = str(feature).strip().lower()

            if key not in input_values:
                raise ValueError(
                    f"Unknown model feature '{feature}'. "
                    "Check the features saved with the model."
                )

            row[feature] = input_values[key]

        input_df = pd.DataFrame([row], columns=features)
        prediction = float(model.predict(input_df)[0])

        if prediction <= 0:
            raise ValueError(
                "The model returned a non-positive yield. "
                "Check the training target and units."
            )

        yield_per_acre = prediction * TONNE_HA_TO_QUINTAL_ACRE

        return yield_per_acre, None

    except Exception as exc:
        return None, str(exc)


def calculate(crop_data, area_acres, yield_per_acre, price_change_pct):
    cost_per_acre = sum(crop_data["costs"].values())
    total_cost = cost_per_acre * area_acres

    total_yield = yield_per_acre * area_acres

    price = crop_data["price_per_quintal"] * (
        1 + price_change_pct / 100
    )

    revenue = total_yield * price
    profit = revenue - total_cost

    roi = (
        profit / total_cost * 100
        if total_cost > 0
        else None
    )

    break_even = (
        total_cost / total_yield
        if total_yield > 0
        else None
    )

    return {
        "cost_per_acre": cost_per_acre,
        "total_cost": total_cost,
        "yield_per_acre": yield_per_acre,
        "total_yield": total_yield,
        "price": price,
        "revenue": revenue,
        "profit": profit,
        "roi": roi,
        "break_even": break_even,
        "risk": risk_label(roi),
    }


# --------------------------------------------------
# 6. SIDEBAR INPUTS
# --------------------------------------------------

st.sidebar.title("🌱 Your Farm")
st.sidebar.caption("Enter your farm details below.")

crop_name = st.sidebar.selectbox(
    "Select crop",
    list(CROPS.keys())
)

st.sidebar.subheader("📏 Land Measurement")

col_acres, col_guntas = st.sidebar.columns(2)

with col_acres:
    acres = st.number_input(
        "Acres",
        min_value=0,
        max_value=10000,
        value=2,
        step=1
    )

with col_guntas:
    guntas = st.number_input(
        "Guntas",
        min_value=0,
        max_value=39,
        value=0,
        step=1
    )

area = land_to_acres(acres, guntas)

st.sidebar.success(f"Total land: {area:.4f} acres")

st.sidebar.subheader("📍 Location and Season")

state = st.sidebar.text_input(
    "State",
    value="Telangana"
).strip()

district_options = []

if isinstance(model_bundle, dict):
    saved_districts = model_bundle.get("districts", [])

    if isinstance(saved_districts, (list, tuple)):
        district_options = [
            str(item) for item in saved_districts
            if str(item).strip()
        ]

if district_options:
    district = st.sidebar.selectbox(
        "District",
        district_options
    )
else:
    district = st.sidebar.text_input(
        "District",
        value=""
    ).strip()

season = st.sidebar.selectbox(
    "Growing season",
    ["Kharif", "Rabi", "Summer"]
)

default_year = 2026

if isinstance(model_bundle, dict):
    saved_year = model_bundle.get("latest_year")

    if saved_year is not None:
        try:
            default_year = int(saved_year)
        except (TypeError, ValueError):
            pass

year = st.sidebar.number_input(
    "Crop year",
    min_value=2000,
    max_value=2100,
    value=default_year,
    step=1
)

st.sidebar.subheader("💰 Price Scenario")

price_change = st.sidebar.slider(
    "Change in selling price (%)",
    min_value=-30,
    max_value=30,
    value=0,
    step=5
)

st.sidebar.caption(
    "Adjust the slider to explore how price changes affect profitability."
)


# --------------------------------------------------
# 7. VALIDATE INPUTS AND PREDICT YIELD
# --------------------------------------------------

if area <= 0:
    st.error("Enter a land area greater than zero.")
    st.stop()

if not state or not district:
    st.info("Enter your state and district to continue.")
    st.stop()

if model_load_error:
    st.warning(
        "The saved model could not be loaded. "
        f"Details: {model_load_error}"
    )

yield_prediction, prediction_error = predict_yield_per_acre(
    crop_name,
    state,
    district,
    season,
    int(year)
)

if yield_prediction is None:
    st.warning(
        "The trained model is unavailable for these inputs. "
        "Sample yield values will be used for demonstration."
    )

    if prediction_error and prediction_error != "Model unavailable":
        with st.expander("Model diagnostic"):
            st.code(prediction_error)

    selected_yield = CROPS[crop_name]["sample_yield"]
    using_model = False
else:
    selected_yield = yield_prediction
    using_model = True


# --------------------------------------------------
# 8. CALCULATE ALL CROP RESULTS
# --------------------------------------------------

all_results = {}

for name, data in CROPS.items():

    if name == crop_name:
        predicted_yield = selected_yield
    else:
        other_prediction, _ = predict_yield_per_acre(
            name,
            state,
            district,
            season,
            int(year)
        )

        if other_prediction is not None:
            predicted_yield = other_prediction
        else:
            predicted_yield = data["sample_yield"]

    all_results[name] = calculate(
        data,
        area,
        predicted_yield,
        price_change
    )


# --------------------------------------------------
# 9. FARM SUMMARY
# --------------------------------------------------

st.subheader("📍 Farm Summary")

summary1, summary2, summary3 = st.columns(3)

summary1.metric("Land area", land_display(area))
summary2.metric("Land in acres", f"{area:.4f}")
summary3.metric(
    "Land in hectares",
    f"{area * 0.4046856422:.4f}"
)

if using_model:
    st.success("Yield estimate generated using the trained model.")
else:
    st.info(
        "Sample yield is being used. Add a compatible "
        "yield_model.joblib beside app.py to enable model predictions."
    )


# --------------------------------------------------
# 10. DASHBOARD TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🌾 Selected Crop",
    "📊 Compare Crops",
    "ℹ️ Scope and Limitations"
])


# --------------------------------------------------
# TAB 1: SELECTED CROP
# --------------------------------------------------

with tab1:
    r = all_results[crop_name]

    st.subheader(f"{crop_name} — {land_display(area)}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Cultivation Cost", rupees(r["total_cost"]))
    col2.metric(
        "Expected Yield",
        f'{r["total_yield"]:,.2f} quintals'
    )
    col3.metric("Gross Revenue", rupees(r["revenue"]))

    col4, col5, col6 = st.columns(3)

    col4.metric("Net Profit / Loss", rupees(r["profit"]))
    col5.metric("Return on Investment", percent(r["roi"]))

    break_even_text = (
        "N/A"
        if r["break_even"] is None
        else f'₹{r["break_even"]:,.2f}/quintal'
    )

    col6.metric("Break-Even Price", break_even_text)

    st.divider()

    st.subheader("🌱 Yield and Price Details")

    detail1, detail2, detail3 = st.columns(3)

    detail1.metric(
        "Yield per Acre",
        f'{r["yield_per_acre"]:,.2f} quintals'
    )
    detail2.metric(
        "Selling Price",
        f'₹{r["price"]:,.0f}/quintal'
    )
    detail3.metric(
        "Cost per Acre",
        rupees(r["cost_per_acre"])
    )

    st.caption(
        f"Price scenario: {price_change:+d}%. "
        "Break-even price is the estimated selling price needed "
        "to cover cultivation costs."
    )

    st.subheader("💵 Cultivation Cost Breakdown")

    cost_table = pd.DataFrame({
        "Expense": list(CROPS[crop_name]["costs"].keys()),
        "Cost per Acre (₹)": list(CROPS[crop_name]["costs"].values())
    })

    cost_table["Total Cost (₹)"] = (
        cost_table["Cost per Acre (₹)"] * area
    )

    st.dataframe(
        cost_table,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("⚠️ Financial Risk Indicator")

    if r["risk"] == "Low":
        st.success(f"Low indicator — ROI is {percent(r['roi'])}.")
    elif r["risk"] == "Medium":
        st.warning(f"Medium indicator — ROI is {percent(r['roi'])}.")
    else:
        st.error(f"High indicator — ROI is {percent(r['roi'])}.")

    st.caption(
        "Prototype rule: ROI ≥ 50% is Low; 20% to below 50% is "
        "Medium; below 20% is High. This is a simple financial "
        "indicator, not a validated agricultural risk prediction."
    )


# --------------------------------------------------
# TAB 2: COMPARE CROPS
# --------------------------------------------------

with tab2:
    st.subheader("🌾 Crop-wise Financial Comparison")

    comparison_rows = []

    for name, result in all_results.items():
        comparison_rows.append({
            "Crop": name,
            "Yield (quintals/acre)": round(
                result["yield_per_acre"], 2
            ),
            "Total Yield (quintals)": round(
                result["total_yield"], 2
            ),
            "Cost per Acre (₹)": round(
                result["cost_per_acre"], 2
            ),
            "Total Cost (₹)": round(
                result["total_cost"], 2
            ),
            "Revenue (₹)": round(
                result["revenue"], 2
            ),
            "Profit / Loss (₹)": round(
                result["profit"], 2
            ),
            "ROI (%)": (
                round(result["roi"], 2)
                if result["roi"] is not None
                else None
            ),
            "Risk Indicator": result["risk"],
        })

    comparison_df = pd.DataFrame(comparison_rows)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📈 Estimated Profit by Crop")

    profit_df = comparison_df.set_index("Crop")["Profit / Loss (₹)"]

    st.bar_chart(profit_df, color="#2E8B3C")

    st.caption(
        "All crops use the same land area and price-change setting. "
        "Sample yields are used wherever the trained model cannot "
        "produce a prediction. Costs and prices are sample assumptions."
    )


# --------------------------------------------------
# TAB 3: SCOPE AND LIMITATIONS
# --------------------------------------------------

with tab3:
    st.subheader("🌿 How the Planner Works")

    st.markdown("""
    **Land Measurement**
    - 1 acre = 40 guntas.
    - Land is converted into decimal acres.
    - 1 acre is approximately 0.404686 hectares.

    **Yield Prediction**
    - The saved model estimates yield from crop, state, district,
      season and year, when those inputs match its trained features.
    - The conversion assumes the model predicts tonnes per hectare.
    - Sample yields are used if the model is unavailable or incompatible.

    **Financial Calculations**
    - Total cost = cost per acre × land area.
    - Total yield = yield per acre × land area.
    - Revenue = total yield × selling price per quintal.
    - Profit / loss = revenue − cultivation cost.
    - ROI = profit / cultivation cost × 100.
    - Break-even price = total cost / total yield.

    **Important Limitations**
    - Cost and selling-price figures are sample values, not live data.
    - Actual yield depends on soil, rainfall, irrigation, pests,
      seed variety and farming practices.
    - Market prices vary by location, quality and date.
    - The financial risk indicator is a simple rule, not a
      validated machine-learning risk model.
    """)

    st.subheader("🔧 Model Information")

    if model is not None:
        st.success("A model file was loaded.")
       

        if isinstance(model_bundle, dict):
            if model_bundle.get("features"):
                st.write(
                    "Saved input features:",
                    model_bundle["features"]
                )

            if model_bundle.get("metrics"):
                st.write("Training metrics:")
                st.json(model_bundle["metrics"])
    else:
        st.warning(
            "No compatible trained model is loaded. "
            "Check that yield_model.joblib exists beside app.py."
        )


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "🌿 Farm Financial Planner | Prototype for planning and "
    "comparison. Verify local cultivation costs, yield assumptions "
    "and market prices before making farming decisions."
)