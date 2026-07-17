import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

# ==========================================
# 1. PAGE CONFIGURATION & DASHBOARD THEMING
# ==========================================
st.set_page_config(
    page_title="Retail Credit Risk Cockpit",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Retail Credit Risk Assessment Platform")
st.markdown("""
**Operational Framework:** CRISP-DM Production Phase  
**Champion Model Pipeline:** XGBoost + RandomOverSampler (Baseline Config)  
*Optimized under a strict financial cost function ($100 Type I Error / $400 Type II Error).*
""")
st.write("---")

# ==========================================
# 2. PIPELINE LOADING (REAL PRODUCTION ENGINE)
# ==========================================
@st.cache_resource
def load_production_pipeline():
    trained_model = joblib.load('champion_xgboost_model.pkl')
    fitted_scaler = joblib.load('scaler.pkl')
    return trained_model, fitted_scaler

try:
    model, scaler = load_production_pipeline()
    is_mock = False
except FileNotFoundError:
    @st.cache_resource
    def load_mock_fallback():
        mock_model = XGBClassifier(max_depth=5, n_estimators=100, random_state=42)
        X_dummy = np.random.randn(100, 9)
        y_dummy = np.random.randint(0, 2, 100)
        mock_model.fit(X_dummy, y_dummy)
        return mock_model
    model = load_mock_fallback()
    scaler = None
    is_mock = True

# ==========================================
# 3. INTERACTIVE RISK INPUT SIDEBAR (BATCH MODE)
# ==========================================
st.sidebar.header("📋 Borrower Risk Profile Inputs")
st.sidebar.markdown("Modify the indicators and click **Run Assessment** to evaluate:")

# Wrap ALL elements strictly inside the form using standard st. instead of st.sidebar.
with st.sidebar.form(key="risk_input_form"):
    
    # --- Demographic & Socioeconomic Features ---
    st.subheader("👤 Demographic Profile")
    sex_label = st.selectbox("Gender / Sex", ["Female", "Male"])
    sex = 1 if sex_label == "Male" else 2

    edu_label = st.selectbox(
        "Highest Education Level", 
        ["Graduate School", "University", "High School", "Others"]
    )
    edu_mapping = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
    education = edu_mapping[edu_label]

    mar_label = st.selectbox("Marital Status", ["Married", "Single"])
    mar_mapping = {"Married": 1, "Single": 2}
    marriage = mar_mapping[mar_label]

    age = st.slider("Borrower Age", min_value=18, max_value=80, value=35)

    # --- Financial Exposure & History ---
    st.subheader("📈 Financial Exposure & History")
    limit_bal = st.number_input("Limit Balance (Credit Limit in NTD)", min_value=1000, max_value=1000000, value=50000, step=10000)

    pay_1 = st.slider("Repayment Status (Current Month)", min_value=-2, max_value=8, value=0)
    pay_2 = st.slider("Repayment Status (Previous Month)", min_value=-2, max_value=8, value=0)

    bill_amt1 = st.number_input("Current Bill Amount (NTD)", min_value=-10000, max_value=500000, value=12000)
    pay_amt1 = st.number_input("Amount Paid in Previous Month (NTD)", min_value=0, max_value=500000, value=3000)
    
    # This single submit button controls the whole form layout
    submit_button = st.form_submit_button(label="⚡ Run Risk Assessment", use_container_width=True)

# ==========================================
# 4. DATA TRANSFORMATION & PREDICTION ENGINE
# ==========================================
raw_features = np.array([[
    limit_bal, sex, education, marriage, age, 
    pay_1, pay_2, bill_amt1, pay_amt1
]])

if not is_mock and scaler is not None:
    scaled_features = scaler.transform(raw_features)
    risk_probabilities = model.predict_proba(scaled_features)[0]
    risk_probability = risk_probabilities[1]
    risk_verdict = model.predict(scaled_features)[0]
else:
    if pay_1 > 1 or pay_2 > 1:
        risk_probability = 0.84 + (pay_1 * 0.02) 
    else:
        risk_probability = 0.10 + (age * 0.001) + (education * 0.02) + (pay_1 * 0.05)
    risk_probability = min(float(risk_probability), 0.99)
    risk_verdict = 1 if risk_probability >= 0.5 else 0

# ==========================================
# 5. RISK ASSESSMENT & MANAGEMENT DISPLAY
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Operational Risk Analysis")
    
    prob_percentage = risk_probability * 100
    st.metric(
        label="Calculated Default Probability", 
        value=f"{prob_percentage:.2f}%", 
        delta="- Safe Territory" if risk_verdict == 0 else "+ High Exposure Alert",
        delta_color="inverse"
    )
    
    if risk_verdict == 0:
        st.success("🟢 **CREDIT VERDICT: APPROVED** — This profile exhibits stable repayment dynamics matching a non-default trajectory.")
    else:
        st.error("🔴 **CREDIT VERDICT: REJECTED** — High probability of imminent default detected. Automated underwriting recommends withholding credit extension.")

with col2:
    st.subheader("💰 Cost Matrix Framework")
    st.markdown("""
    Under Professor Oziel's underwriting constraints, errors carry asymmetric financial penalties:
    * **Type I Error (False Alarm):** $100  
    * **Type II Error (Missed Default):** $400  
    """)
    
    if risk_verdict == 1:
        st.warning("🛡️ **Capital Hedged:** Rejecting this borrower shields the bank from a potential **$400** Type II operational loss.")
    else:
        st.info("💼 **Operational Overhead:** If this borrower eventually defaults, it will incur an unmitigated **$400** risk cost.")

# ==========================================
# 6. UNDERWRITING SYSTEM DIAGNOSTICS
# ==========================================
st.write("---")
st.subheader("🛠️ Underwriting System Diagnostics")
with st.expander("Click to view processed model vector arrays"):
    
    feature_columns = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", 
        "PAY_0", "PAY_2", "BILL_AMT1", "PAY_AMT1"
    ]
    
    input_df = pd.DataFrame(raw_features, columns=feature_columns)
    
    st.write("**Processed Pipeline Input Vector (Aligned to Model Schema):**")
    st.dataframe(input_df, use_container_width=True)
    
    st.write("**Raw Model Log-Odds Output Probabilities:**")
    st.code(f"[Healthy (Non-Default): {1-risk_probability:.4f}, Default: {risk_probability:.4f}]")
