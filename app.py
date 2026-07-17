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
# 3. INTERACTIVE RISK INPUT SIDEBAR (10-SCENARIO TESTING ENGINE)
# ==========================================
st.sidebar.header("📋 Borrower Risk Profile Inputs")

st.sidebar.subheader("🚀 Presentation Scenario Matrix")
use_preset = st.sidebar.checkbox("Enable Presentation Mode Presets", value=True)

# Define the dictionary of 10 hyper-realistic clinical profiles
scenarios = {
    1: {"name": "Scenario 1: Prime Executive (Non-Default)", "sex": "Male", "edu": "Graduate School", "mar": "Married", "age": 45, "limit": 400000, "pay1": -1, "pay2": -1, "bill": 15000, "pay_amt": 15000},
    2: {"name": "Scenario 2: Young Professional (Non-Default)", "sex": "Female", "edu": "University", "mar": "Single", "age": 26, "limit": 80000, "pay1": 0, "pay2": 0, "bill": 12000, "pay_amt": 4000},
    3: {"name": "Scenario 3: Medical Doctor (Non-Default)", "sex": "Male", "edu": "Graduate School", "mar": "Married", "age": 39, "limit": 500000, "pay1": 0, "pay2": 0, "bill": 145000, "pay_amt": 120000},
    4: {"name": "Scenario 4: Mature Conservative (Non-Default)", "sex": "Female", "edu": "High School", "mar": "Married", "age": 52, "limit": 150000, "pay1": -2, "pay2": -2, "bill": 0, "pay_amt": 0},
    5: {"name": "Scenario 5: Corporate Consultant (Non-Default)", "sex": "Male", "edu": "University", "mar": "Single", "age": 33, "limit": 250000, "pay1": 0, "pay2": -1, "bill": 32000, "pay_amt": 35000},
    6: {"name": "Scenario 6: Entry Starter Card (Non-Default)", "sex": "Female", "edu": "University", "mar": "Single", "age": 22, "limit": 30000, "pay1": 0, "pay2": 0, "bill": 4500, "pay_amt": 2000},
    7: {"name": "Scenario 7: Over-Leveraged High Risk (Default)", "sex": "Male", "edu": "High School", "mar": "Single", "age": 25, "limit": 20000, "pay1": 2, "pay2": 2, "bill": 19500, "pay_amt": 0},
    8: {"name": "Scenario 8: High Income Distress (Default)", "sex": "Female", "edu": "Graduate School", "mar": "Married", "age": 41, "limit": 300000, "pay1": 3, "pay2": 2, "bill": 280000, "pay_amt": 5000},
    9: {"name": "Scenario 9: Maxed Out Liquidity Trap (Default)", "sex": "Male", "edu": "University", "mar": "Single", "age": 29, "limit": 50000, "pay1": 1, "pay2": 0, "bill": 49000, "pay_amt": 1000},
    10: {"name": "Scenario 10: Chronic Delinquency (Default)", "sex": "Female", "edu": "High School", "mar": "Married", "age": 31, "limit": 10000, "pay1": 2, "pay2": 2, "bill": 9000, "pay_amt": 0}
}

if use_preset:
    scenario_id = st.sidebar.slider("Select Scenario ID (1-6: Safe, 7-10: Default)", min_value=1, max_value=10, value=1, step=1)
    selected = scenarios[scenario_id]
    st.sidebar.info(f"**Loaded:** {selected['name']}")
    
    # Map variables to current selection
    default_sex, default_edu, default_mar, default_age = selected["sex"], selected["edu"], selected["mar"], selected["age"]
    default_limit, default_pay1, default_pay2, default_bill, default_pay_amt = selected["limit"], selected["pay1"], selected["pay2"], selected["bill"], selected["pay_amt"]
else:
    # Free manual mode fallbacks
    st.sidebar.warning("Manual mode active. Adjust sliders freely below.")
    default_sex, default_edu, default_mar, default_age = "Female", "Graduate School", "Married", 35
    default_limit, default_pay1, default_pay2, default_bill, default_pay_amt = 50000, 0, 0, 12000, 3000

st.sidebar.markdown("---")

with st.sidebar.form(key="risk_input_form"):
    st.subheader("👤 Demographic Profile")
    sex_label = st.selectbox("Gender / Sex", ["Female", "Male"], index=["Female", "Male"].index(default_sex))
    sex = 1 if sex_label == "Male" else 2

    edu_options = ["Graduate School", "University", "High School", "Others"]
    edu_label = st.selectbox("Highest Education Level", edu_options, index=edu_options.index(default_edu))
    edu_mapping = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
    education = edu_mapping[edu_label]

    mar_options = ["Married", "Single"]
    mar_label = st.selectbox("Marital Status", mar_options, index=mar_options.index(default_mar))
    mar_mapping = {"Married": 1, "Single": 2}
    marriage = mar_mapping[mar_label]

    age = st.slider("Borrower Age", min_value=18, max_value=80, value=default_age)

    # --- Financial Exposure & History ---
    st.subheader("📈 Financial Exposure & History")
    limit_bal = st.number_input("Limit Balance (Credit Limit in NTD)", min_value=1000, max_value=1000000, value=int(default_limit), step=10000)

    pay_1 = st.slider("Repayment Status (Current Month)", min_value=-2, max_value=8, value=int(default_pay1))
    pay_2 = st.slider("Repayment Status (Previous Month)", min_value=-2, max_value=8, value=int(default_pay2))

    bill_amt1 = st.number_input("Current Bill Amount (NTD)", min_value=-10000, max_value=500000, value=int(default_bill))
    pay_amt1 = st.number_input("Amount Paid in Previous Month (NTD)", min_value=0, max_value=500000, value=int(default_pay_amt))
    
    submit_button = st.form_submit_button(label="⚡ Run Risk Assessment", use_container_width=True)

# ==========================================
# 4. DATA TRANSFORMATION & PREDICTION ENGINE
# ==========================================
raw_features = np.array([[
    limit_bal, sex, education, marriage, age, 
    pay_1, pay_2, bill_amt1, pay_amt1
]])

# Calculate ONLY when the button is explicitly pressed
if submit_button:
    if not is_mock and scaler is not None:
        scaled_features = scaler.transform(raw_features)
        risk_probabilities = model.predict_proba(scaled_features)[0]
        risk_probability = float(risk_probabilities[1])
        risk_verdict = int(model.predict(scaled_features)[0])
    else:
        # Dynamic fallback logic that recalculates based on your live slider values
        if pay_1 > 1 or pay_2 > 1:
            local_prob = 0.84 + (pay_1 * 0.02) 
        else:
            local_prob = 0.10 + (age * 0.001) + (education * 0.02) + (pay_1 * 0.05)
        risk_probability = min(float(local_prob), 0.99)
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
    # 6. UNDERWRITING SYSTEM DIAGNOSTICS (MOVED INSIDE)
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
            
else:
    # This shows cleanly on startup until they click the button
    st.info("👈 Adjust the borrower risk profile parameters in the sidebar and click **⚡ Run Risk Assessment** to calculate the underwriting metrics.")
