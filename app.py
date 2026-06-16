import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Poverty Rate Prediction",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS (UI STYLING)
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f2027, #000000 70%);
    color: white;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #ff4ecd, #4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 30px;
}

.card {
    background: rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 0 30px rgba(79,172,254,0.18);
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #ff4ecd;
    margin-bottom: 10px;
}

.stButton>button {
    background: linear-gradient(90deg, #ff4ecd, #4facfe);
    color: white;
    border-radius: 30px;
    padding: 14px 35px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.08);
}

.gauge {
    font-size: 56px;
    font-weight: 900;
    text-align: center;
    animation: pulse 1.6s infinite;
}

@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}

.low { color: #00ff9c; }
.moderate { color: #ffd166; }
.high { color: #ff7a18; }
.very-high { color: #ff1744; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA & MODELS
# --------------------------------------------------
df = pd.read_excel("state_economy_predictor_for_poverty_dataset.xlsx")

linear_model = joblib.load("linear_model.pkl")
poly_model = joblib.load("poly_model.pkl")
poly = joblib.load("poly_transformer.pkl")
feature_names = joblib.load("feature_names.pkl")

states = sorted(df["state"].unique())
categories = sorted(df["category"].unique())

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown("<div class='main-title'>State Economy Based Poverty Rate Prediction System</div>", unsafe_allow_html=True)

# --------------------------------------------------
# INPUT UI
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Basic Information</div>", unsafe_allow_html=True)

    state = st.selectbox("Select State", states)
    category = st.selectbox("Select Category", categories)
    model_choice = st.selectbox("Regression Model", ["Linear Regression", "Polynomial Regression"])
    year = st.slider("Year", 2010, 2030, 2021)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Economic Indicators</div>", unsafe_allow_html=True)

    population = st.slider("Population", 10_000, 20_000_000, 3_000_000, step=10_000)
    literacy = st.slider("Literacy Rate (%)", 0.0, 100.0, 90.0)
    unemployment = st.slider("Unemployment Rate (%)", 0.0, 50.0, 6.0)
    labour = st.slider("Labour Participation Rate (%)", 0.0, 100.0, 75.0)
    healthcare = st.slider("Healthcare Index", 0.0, 100.0, 60.0)
    inflation = st.slider("Inflation Rate (%)", 0.0, 20.0, 4.5)
    gdp = st.slider("GDP Growth Rate (%)", -10.0, 20.0, 6.0)
    income = st.slider("Per Capita Income (₹)", 10_000, 500_000, 150_000)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# PREPARE INPUT DATA
# --------------------------------------------------
input_dict = {
    "year": year,
    "population": population,
    "literacy_rate": literacy,
    "unemployment_rate": unemployment,
    "labour_participation_rate": labour,
    "healthcare_index": healthcare,
    "inflation_rate": inflation,
    "gdp_growth_rate": gdp,
    "per_capita_income": income,
    f"state_{state}": 1,
    f"category_{category}": 1
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Predict Poverty Rate"):

    if model_choice == "Linear Regression":
        prediction = linear_model.predict(input_df)[0]
    else:
        prediction = poly_model.predict(poly.transform(input_df))[0]

    # Clamp output
    prediction = max(0, min(prediction, 100))

    # Poverty level styling
    if prediction < 20:
        label, cls = "Low Poverty", "low"
    elif prediction < 40:
        label, cls = "Moderate Poverty", "moderate"
    elif prediction < 60:
        label, cls = "High Poverty", "high"
    else:
        label, cls = "Very High Poverty", "very-high"

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Predicted Poverty Rate</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='gauge {cls}'>{prediction:.2f}%</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;font-size:22px;' class='{cls}'>{label}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    💡 <b>Insight:</b><br>
    Higher unemployment, low per-capita income, rural classification, and weak GDP growth
    significantly increase the predicted poverty rate according to historical data trends.
    </div>
    """, unsafe_allow_html=True)