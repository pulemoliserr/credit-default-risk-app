import os
import json
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =============================================================================
# PAGE CONFIGURATION (must be called once, before any other st.* call)
# =============================================================================
st.set_page_config(
    page_title="Credit Risk & Segmentation Platform",
    page_icon="\U0001F4B3",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("\U0001F4B3 Credit Risk & Behavioral Segmentation Platform")
st.caption(
    "Two CRISP-DM projects on the same Taiwan Credit Card Clients population, combined into one "
    "dashboard: a supervised default classifier, and an unsupervised behavioral segmentation."
)

tab_classification, tab_clustering = st.tabs([
    "\U0001F4B3  Credit Default Prediction",
    "\U0001F9E9  Customer Segmentation (Clustering)",
])

# =============================================================================
# =============================================================================
# TAB 1: CREDIT DEFAULT PREDICTION
# =============================================================================
# =============================================================================
with tab_classification:

    st.markdown("""
    **Operational Framework:** CRISP-DM Production Phase
    **Champion Model Pipeline:** XGBoost + RandomOverSampler (Baseline Config)
    *Optimized under a strict financial cost function ($100 Type I Error / $400 Type II Error).*
    """)
    st.write("---")

    # -------------------------------------------------------------------
    # Load the REAL production pipeline. No mock/fabricated fallback --
    # if the artifacts aren't present, say so plainly rather than
    # simulating a prediction that looks real but isn't.
    # -------------------------------------------------------------------
    @st.cache_resource
    def load_production_pipeline():
        trained_model = joblib.load('champion_xgboost_model.pkl')
        fitted_scaler = joblib.load('scaler.pkl')
        return trained_model, fitted_scaler

    @st.cache_data
    def load_model_metrics():
        with open('model_metrics.json') as f:
            return json.load(f)

    missing_clf_files = [f for f in ['champion_xgboost_model.pkl', 'scaler.pkl', 'model_metrics.json'] if not os.path.exists(f)]

    if missing_clf_files:
        st.error(
            "**This tab cannot run: required model artifacts are missing.**\n\n"
            "This app never fabricates a prediction when the real model isn't available -- "
            "the following files must sit alongside `app.py`:\n\n"
            + "\n".join(f"- `{f}`" for f in missing_clf_files)
        )
        st.stop()

    model, scaler = load_production_pipeline()
    model_metrics = load_model_metrics()

    inference_subtab, performance_subtab = st.tabs(["\U0001F52E Live Inference", "\U0001F4CA Model Performance & Configuration"])

    # =====================================================================
    # SUB-TAB A: LIVE INFERENCE
    # =====================================================================
    with inference_subtab:
        st.sidebar.header("\U0001F4CB Credit Default Prediction -- Borrower Inputs")
        st.sidebar.subheader("\U0001F680 Presentation Scenario Matrix")
        use_preset = st.sidebar.checkbox("Enable Presentation Mode Presets", value=True, key="clf_preset_toggle")

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
            10: {"name": "Scenario 10: Chronic Delinquency (Default)", "sex": "Female", "edu": "High School", "mar": "Married", "age": 31, "limit": 10000, "pay1": 2, "pay2": 2, "bill": 9000, "pay_amt": 0},
        }

        preset_submit = False
        if use_preset:
            scenario_id = st.sidebar.slider("Select Scenario ID (1-6: Safe, 7-10: Default)", 1, 10, 1, 1, key="clf_scenario_slider")
            selected = scenarios[scenario_id]
            st.sidebar.info(f"**Loaded:** {selected['name']}")
            preset_submit = st.sidebar.button("\u26A1 Run Selected Scenario", width='stretch', type="primary", key="clf_preset_run")
            default_sex, default_edu, default_mar, default_age = selected["sex"], selected["edu"], selected["mar"], selected["age"]
            default_limit, default_pay1, default_pay2, default_bill, default_pay_amt = selected["limit"], selected["pay1"], selected["pay2"], selected["bill"], selected["pay_amt"]
        else:
            st.sidebar.warning("Manual mode active. Adjust sliders freely below.")
            default_sex, default_edu, default_mar, default_age = "Female", "Graduate School", "Married", 35
            default_limit, default_pay1, default_pay2, default_bill, default_pay_amt = 50000, 0, 0, 12000, 3000

        st.sidebar.markdown("---")

        with st.sidebar.form(key="risk_input_form"):
            st.subheader("\U0001F464 Demographic Profile")
            sex_label = st.selectbox("Gender / Sex", ["Female", "Male"], index=["Female", "Male"].index(default_sex))
            sex = 1 if sex_label == "Male" else 2

            edu_options = ["Graduate School", "University", "High School", "Others"]
            edu_label = st.selectbox("Highest Education Level", edu_options, index=edu_options.index(default_edu))
            education = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}[edu_label]

            mar_options = ["Married", "Single"]
            mar_label = st.selectbox("Marital Status", mar_options, index=mar_options.index(default_mar))
            marriage = {"Married": 1, "Single": 2}[mar_label]

            age = st.slider("Borrower Age", 18, 80, default_age)

            st.subheader("\U0001F4C8 Financial Exposure & History")
            limit_bal = st.number_input("Limit Balance (Credit Limit in NTD)", min_value=1000, max_value=1000000, value=int(default_limit), step=10000)
            pay_1 = st.slider("Repayment Status (Current Month)", -2, 8, int(default_pay1))
            pay_2 = st.slider("Repayment Status (Previous Month)", -2, 8, int(default_pay2))
            bill_amt1 = st.number_input("Current Bill Amount (NTD)", min_value=-10000, max_value=500000, value=int(default_bill))
            pay_amt1 = st.number_input("Amount Paid in Previous Month (NTD)", min_value=0, max_value=500000, value=int(default_pay_amt))

            form_submit = st.form_submit_button("\u26A1 Run Custom Assessment", width='stretch')

        submit_button = form_submit or preset_submit

        raw_features = pd.DataFrame(
            [[limit_bal, sex, education, marriage, age, pay_1, pay_2, bill_amt1, pay_amt1]],
            columns=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "BILL_AMT1", "PAY_AMT1"],
        )

        if submit_button:
            scaled_features = scaler.transform(raw_features)
            risk_probabilities = model.predict_proba(scaled_features)[0]
            risk_probability = float(risk_probabilities[1])
            risk_verdict = int(model.predict(scaled_features)[0])

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("\U0001F4CA Operational Risk Analysis")
                prob_percentage = risk_probability * 100
                st.metric(
                    label="Calculated Default Probability", value=f"{prob_percentage:.2f}%",
                    delta="- Safe Territory" if risk_verdict == 0 else "+ High Exposure Alert",
                    delta_color="inverse",
                )
                if risk_verdict == 0:
                    st.success("\U0001F7E2 **CREDIT VERDICT: APPROVED** -- This profile exhibits stable repayment dynamics matching a non-default trajectory.")
                else:
                    st.error("\U0001F534 **CREDIT VERDICT: REJECTED** -- High probability of imminent default detected. Automated underwriting recommends withholding credit extension.")

            with col2:
                st.subheader("\U0001F4B0 Cost Matrix Framework")
                st.markdown("""
                Errors carry asymmetric financial penalties:
                * **Type I Error (False Alarm):** \\$100
                * **Type II Error (Missed Default):** \\$400
                """)
                if risk_verdict == 1:
                    st.warning("\U0001F6E1\uFE0F **Capital Hedged:** Rejecting this borrower shields the bank from a potential **$400** Type II operational loss.")
                else:
                    st.info("\U0001F4BC **Operational Overhead:** If this borrower eventually defaults, it will incur an unmitigated **$400** risk cost.")

            st.write("---")
            st.subheader("\U0001F6E0\uFE0F Underwriting System Diagnostics")
            with st.expander("Click to view processed model vector arrays"):
                st.write("**Processed Pipeline Input Vector (Aligned to Model Schema):**")
                st.dataframe(raw_features, width='stretch')
                st.write("**Raw Model Output Probabilities:**")
                st.code(f"[Healthy (Non-Default): {1-risk_probability:.4f}, Default: {risk_probability:.4f}]")
        else:
            st.info("\U0001F448 Adjust the borrower risk profile parameters in the sidebar and click **Run Assessment** to calculate the underwriting metrics.")

    # =====================================================================
    # SUB-TAB B: MODEL PERFORMANCE & CONFIGURATION
    # =====================================================================
    with performance_subtab:
        st.subheader("\U0001F3C6 Champion Model Performance")
        st.caption(
            "All figures below are computed once on a genuine 6,000-account holdout test set "
            "(20% of the real dataset, stratified, never seen during training or scaling/balancing "
            "fits) -- not training-set metrics, and not simulated."
        )

        m = model_metrics["metrics"]
        cm = model_metrics["confusion_matrix"]
        cfg = model_metrics["model_config"]
        info = model_metrics["dataset_info"]

        # ---- Metric cards ----
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['Accuracy']:.1%}")
        c2.metric("Precision", f"{m['Precision']:.1%}")
        c3.metric("Recall", f"{m['Recall']:.1%}")
        c4.metric("F1-Score", f"{m['F1']:.3f}")
        c5.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("##### Metrics -- Graphical View")
            metrics_df = pd.DataFrame({"Metric": list(m.keys()), "Score": list(m.values())})
            fig_metrics = px.bar(
                metrics_df, x="Metric", y="Score", color="Metric", range_y=[0, 1], text_auto=".3f",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_metrics.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_metrics, width='stretch')

            st.markdown("##### Confusion Matrix (Holdout Test Set)")
            cm_matrix = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
            fig_cm = px.imshow(
                cm_matrix, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Accounts"),
                x=["No Default", "Default"], y=["No Default", "Default"],
            )
            fig_cm.update_layout(margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_cm, width='stretch')

        with col_right:
            st.markdown("##### ROC Curve")
            roc = model_metrics["roc_curve"]
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                          name=f"Champion (AUC={m['ROC-AUC']:.3f})", line=dict(color="#DC2626", width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
                                          line=dict(color="gray", dash="dash")))
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                                   margin=dict(l=10, r=10, t=10, b=10), legend=dict(x=0.4, y=0.1))
            st.plotly_chart(fig_roc, width='stretch')

            st.markdown("##### Feature Importance")
            fi = model_metrics["feature_importance"]
            fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())}).sort_values("Importance")
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#1E3A8A"])
            fig_fi.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_fi, width='stretch')

        st.markdown("---")
        st.markdown("##### Metrics -- Summary Table")
        summary_table = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score", "ROC-AUC",
                       "True Negatives", "False Positives", "False Negatives", "True Positives"],
            "Value": [f"{m['Accuracy']:.4f}", f"{m['Precision']:.4f}", f"{m['Recall']:.4f}",
                      f"{m['F1']:.4f}", f"{m['ROC-AUC']:.4f}",
                      f"{cm['TN']:,}", f"{cm['FP']:,}", f"{cm['FN']:,}", f"{cm['TP']:,}"],
        })
        st.dataframe(summary_table, width='stretch', hide_index=True)

        st.markdown("---")
        st.markdown("##### Champion Model Configuration")
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            st.markdown(f"""
            | Setting | Value |
            |---|---|
            | Algorithm | {cfg['algorithm']} |
            | Tree Depth (`max_depth`) | {cfg['max_depth']} |
            | Number of Trees (`n_estimators`) | {cfg['n_estimators']} |
            | Learning Rate | {cfg['learning_rate']} |
            | Class Balancer | {cfg['balancer']} |
            | Random State (seed) | {cfg['random_state']} |
            """)
        with cfg_col2:
            st.markdown(f"""
            | Dataset Detail | Value |
            |---|---|
            | Training accounts | {info['n_train']:,} |
            | Test accounts (holdout) | {info['n_test']:,} |
            | Test default rate | {info['test_default_rate']:.2%} |
            | Training rows after oversampling | {info['n_train_after_ros']:,} |
            | Features used | {len(cfg['features'])} |
            """)
        st.caption("Features: " + ", ".join(cfg["features"]))

        with st.expander("\U0001F4D6 How to read these numbers"):
            st.markdown("""
            * **Recall** (the metric explicitly requested) answers: *of all clients who actually
              defaulted, what share did the model catch?* This is the operational safety-net metric
              for a credit risk model -- more important here than raw accuracy, since missing a
              real default is the costlier error ($400 vs. $100) under this project's cost framework.
            * **Precision** answers: *of everyone flagged as a default risk, how many actually were?*
            * All numbers are computed exactly once on a held-out test set the model never trained
              or was tuned on -- not cross-validation scores, not training-set performance.
            """)

# =============================================================================
# =============================================================================
# TAB 2: CUSTOMER SEGMENTATION (CLUSTERING) -- K = 6
# =============================================================================
# =============================================================================
with tab_clustering:

    DEFAULT_K = 6
    RANDOM_STATE = 42

    OPTIMIZED_CLUSTER_NAMES = {
        0: "Cluster 0: Low-Utilization Transactors",
        1: "Cluster 1: High-Limit Premium Revolvers",
        2: "Cluster 2: Strained High-Utilization Debtors",
        3: "Cluster 3: Severe Delinquency / High-Risk Escalations",
        4: "Cluster 4: Moderate-Limit Steady Payers",
        5: "Cluster 5: Frequent Recent Delinquents",
    }
    OPTIMIZED_CLUSTER_STRATEGIES = {
        0: {"tier": "Low Risk", "action": "Increase credit limit, target for cross-selling premium financial products."},
        1: {"tier": "Low-Moderate Risk", "action": "Maintain high line limits, monitor utilization, pitch wealth management services."},
        2: {"tier": "Moderate Risk", "action": "Offer balance transfer incentives, structured repayment plans, monitor bill trends."},
        3: {"tier": "Severe Risk", "action": "Immediate credit line freeze, initiate proactive collection outreach, flag for default prediction."},
        4: {"tier": "Low Risk", "action": "Maintain active engagement, standard promotional financing offers."},
        5: {"tier": "High Risk", "action": "Set temporary limit caps, issue automated payment reminders, restrict new charges."},
    }

    def get_segment_name(cluster_id, current_k):
        if current_k == DEFAULT_K:
            return OPTIMIZED_CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        return f"Cluster {cluster_id} (Unmapped - K={current_k})"

    def get_segment_strategy(cluster_id, current_k):
        if current_k == DEFAULT_K and cluster_id in OPTIMIZED_CLUSTER_STRATEGIES:
            return OPTIMIZED_CLUSTER_STRATEGIES[cluster_id]
        return {"tier": "Exploratory / Uncalibrated",
                "action": "Dynamic K value selected. Profile cluster mean metrics to define tailored risk policy."}

    @st.cache_data
    def load_and_preprocess_data(file_path_or_buffer):
        df = pd.read_excel(file_path_or_buffer, header=1)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df.rename(columns={'default_payment_next_month': 'default_payment'}, inplace=True)
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)
        df['education'] = df['education'].replace({0: 4, 5: 4, 6: 4})
        df['marriage'] = df['marriage'].replace({0: 3})
        return df

    @st.cache_data
    def run_data_pipeline(df, k=6):
        df_proc = df.copy()
        df_proc['avg_bill_amt'] = df_proc[['bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6']].mean(axis=1)
        df_proc['avg_pay_amt'] = df_proc[['pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']].mean(axis=1)
        df_proc['utilization_rate'] = df_proc['avg_bill_amt'] / (df_proc['limit_bal'] + 1e-5)
        df_proc['pay_ratio'] = df_proc['avg_pay_amt'] / (df_proc['avg_bill_amt'] + 1e-5)
        df_proc['max_delay'] = df_proc[['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']].max(axis=1)

        clustering_features = [
            'limit_bal', 'age', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
            'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
            'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6',
            'avg_bill_amt', 'avg_pay_amt', 'utilization_rate', 'pay_ratio', 'max_delay',
        ]
        X = df_proc[clustering_features].copy()

        for col in X.columns:
            mean, std = X[col].mean(), X[col].std()
            X[col] = np.clip(X[col], mean - 3 * std, mean + 3 * std)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        df_proc['cluster'] = cluster_labels

        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        pca_components = pca.fit_transform(X_scaled)
        df_proc['pca1'] = pca_components[:, 0]
        df_proc['pca2'] = pca_components[:, 1]

        return df_proc, kmeans, scaler, clustering_features

    st.markdown("**CRISP-DM Unsupervised Prototype** | Taiwan Credit Card Clients Dataset")

    st.sidebar.markdown("---")
    st.sidebar.header("\U0001F9E9 Clustering -- Configuration & Input")
    uploaded_file = st.sidebar.file_uploader("Upload Credit Card Dataset (.xls / .xlsx)", type=["xls", "xlsx"], key="clu_upload")
    k_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=8, value=DEFAULT_K, key="clu_k_slider")

    if k_clusters != DEFAULT_K:
        st.sidebar.warning(
            f"\u26A0\uFE0F **Exploratory Mode (K={k_clusters}):** Baseline strategy mapping is calibrated "
            "specifically for K=6. Business rules for alternate clusters will display fallback labels."
        )

    if uploaded_file is not None:
        df_raw = load_and_preprocess_data(uploaded_file)
    elif os.path.exists("default of credit card clients.xls"):
        df_raw = load_and_preprocess_data("default of credit card clients.xls")
    else:
        st.info("\U0001F446 Please upload the `default of credit card clients.xls` dataset in the sidebar to begin analysis.")
        st.stop()

    df_analyzed, kmeans_model, scaler_model, feature_cols = run_data_pipeline(df_raw, k=k_clusters)

    clu_tab1, clu_tab2, clu_tab3, clu_tab4 = st.tabs([
        "\U0001F4CA Executive Summary & Profiling",
        "\U0001F5FA\uFE0F Cluster Separation (PCA)",
        "\U0001F3AF Business Strategy & Actionability",
        "\U0001F50D Account Single-Inference",
    ])

    with clu_tab1:
        st.subheader("Segment Distribution & Default Risk Profile")
        cluster_summary = df_analyzed.groupby('cluster').agg(
            account_count=('limit_bal', 'count'),
            avg_limit=('limit_bal', 'mean'),
            avg_utilization=('utilization_rate', 'mean'),
            observed_default_rate=('default_payment', 'mean'),
        ).reset_index()
        cluster_summary['account_pct'] = (cluster_summary['account_count'] / len(df_analyzed)) * 100
        cluster_summary['observed_default_rate_pct'] = cluster_summary['observed_default_rate'] * 100
        cluster_summary['segment_name'] = cluster_summary['cluster'].apply(lambda c: get_segment_name(c, k_clusters))

        col1, col2 = st.columns(2)
        with col1:
            fig_size = px.pie(cluster_summary, values='account_count', names='segment_name',
                               title=f"Portfolio Share by Segment (K={k_clusters})", hole=0.4,
                               color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_size, width='stretch')
        with col2:
            fig_risk = px.bar(cluster_summary, x='segment_name', y='observed_default_rate_pct', color='segment_name',
                               title="Observed Default Rate by Segment (%)",
                               labels={'observed_default_rate_pct': 'Default Rate (%)', 'segment_name': 'Segment'},
                               color_discrete_sequence=px.colors.qualitative.Set2)
            fig_risk.add_hline(y=df_analyzed['default_payment'].mean() * 100, line_dash="dash", annotation_text="Portfolio Mean Default Rate")
            st.plotly_chart(fig_risk, width='stretch')

        st.subheader("Financial & Behavioral Summary Table")
        display_summary = cluster_summary[['segment_name', 'account_count', 'account_pct', 'avg_limit', 'avg_utilization', 'observed_default_rate_pct']].copy()
        display_summary.columns = ['Segment Name', 'Accounts', 'Portfolio %', 'Avg Limit ($)', 'Avg Utilization Rate', 'Observed Default Rate (%)']
        st.dataframe(display_summary.style.format({
            'Portfolio %': '{:.1f}%', 'Avg Limit ($)': '${:,.0f}',
            'Avg Utilization Rate': '{:.2%}', 'Observed Default Rate (%)': '{:.2f}%',
        }), width='stretch')

    with clu_tab2:
        st.subheader("2D Principal Component Projection")
        st.markdown("Visualizing cluster boundaries across reduced dimensionality space.")
        df_analyzed['segment_label'] = df_analyzed['cluster'].apply(lambda c: get_segment_name(c, k_clusters))
        fig_pca = px.scatter(df_analyzed, x='pca1', y='pca2', color='segment_label',
                              hover_data=['limit_bal', 'utilization_rate', 'max_delay', 'default_payment'],
                              title=f"K-Means (K={k_clusters}) Clusters projected onto first 2 Principal Components",
                              opacity=0.6, color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_pca, width='stretch')

    with clu_tab3:
        st.subheader("Operational Strategy Matrix")
        for c_id in sorted(df_analyzed['cluster'].unique()):
            strat = get_segment_strategy(c_id, k_clusters)
            c_name = get_segment_name(c_id, k_clusters)
            with st.expander(f"\U0001F4CC **{c_name}** -- Risk Tier: *{strat['tier']}*"):
                st.write(f"**Recommended Action:** {strat['action']}")
                c_data = df_analyzed[df_analyzed['cluster'] == c_id]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Account Count", f"{len(c_data):,}")
                m2.metric("Avg Credit Limit", f"${c_data['limit_bal'].mean():,.0f}")
                m3.metric("Avg Utilization", f"{c_data['utilization_rate'].mean():.1%}")
                m4.metric("Default Rate", f"{c_data['default_payment'].mean():.1%}")

    with clu_tab4:
        st.subheader("Real-Time Account Segment Classifier")
        st.markdown("Input client metrics to predict behavioral segment assignment.")
        with st.form("inference_form"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                input_limit = st.number_input("Credit Limit ($)", min_value=1000, max_value=1000000, value=50000, step=5000)
                input_age = st.number_input("Age", min_value=18, max_value=100, value=35)
                input_pay0 = st.selectbox("Current Repayment Status (pay_0)", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
            with col_b:
                input_bill1 = st.number_input("Recent Bill Amount ($)", value=15000, step=1000)
                input_pay1 = st.number_input("Recent Payment Amount ($)", value=1000, step=500)
                input_max_delay = st.selectbox("Max Delay Status in 6 mos", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
            with col_c:
                input_util = input_bill1 / (input_limit + 1e-5)
                st.metric("Calculated Utilization", f"{input_util:.1%}")
            submit_btn = st.form_submit_button("Classify Account")

        if submit_btn:
            input_data = {
                'limit_bal': input_limit, 'age': input_age,
                'pay_0': input_pay0, 'pay_2': 0, 'pay_3': 0, 'pay_4': 0, 'pay_5': 0, 'pay_6': 0,
                'bill_amt1': input_bill1, 'bill_amt2': input_bill1, 'bill_amt3': input_bill1,
                'bill_amt4': input_bill1, 'bill_amt5': input_bill1, 'bill_amt6': input_bill1,
                'pay_amt1': input_pay1, 'pay_amt2': input_pay1, 'pay_amt3': input_pay1,
                'pay_amt4': input_pay1, 'pay_amt5': input_pay1, 'pay_amt6': input_pay1,
                'avg_bill_amt': input_bill1, 'avg_pay_amt': input_pay1,
                'utilization_rate': input_util,
                'pay_ratio': input_pay1 / (input_bill1 + 1e-5),
                'max_delay': input_max_delay,
            }
            single_df = pd.DataFrame([input_data])[feature_cols]
            single_scaled = scaler_model.transform(single_df)
            predicted_cluster = kmeans_model.predict(single_scaled)[0]
            assigned_name = get_segment_name(predicted_cluster, k_clusters)
            assigned_strat = get_segment_strategy(predicted_cluster, k_clusters)
            st.success(f"**Assigned Segment:** {assigned_name}")
            st.info(f"**Risk Tier:** {assigned_strat['tier']}  \n**Recommended Strategy:** {assigned_strat['action']}")
