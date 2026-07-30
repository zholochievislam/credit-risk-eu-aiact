import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
from src.modeling import load_final_model
from src.fairness import load_thresholds
from src.oversight import process_new_applicant
from src.explainability import shap_explainer, get_explanation, get_waterfall_figure, FEATURE_MAPPING

st.set_page_config(page_title="Credit Risk Scoring Engine", layout="wide")
st.title("Credit Risk Scoring Engine")
st.caption("EU AI Act Compliant — Explainable & Fair Credit Decisioning")


@st.cache_resource
def load_resources():
    model, preprocessor = load_final_model()
    thresholds = load_thresholds()
    return model, preprocessor, thresholds

final_model, final_preprocessor, group_thresholds = load_resources()

# --- SIDEBAR: ввод данных заёмщика ---
with st.sidebar:
    st.header("Applicant Information")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income ($)", min_value=1, value=50000, step=1000)
    emp_length = st.number_input("Employment Length (years)", min_value=0.0, value=5.0, step=0.5)
    loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=0, value=10000, step=500)

    loan_percent_income = loan_amnt / income
    st.caption(f"Calculated Loan-to-Income Ratio: **{loan_percent_income:.1%}**")

    cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    loan_intent = st.selectbox("Loan Purpose", [
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
        "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
    ])
    default_on_file = st.selectbox("Historical Default on File", ["N", "Y"])

    evaluate_clicked = st.button("Evaluate Application", type="primary")

applicant_data = pd.DataFrame([{
    "person_age": age,
    "person_income": income,
    "person_emp_length": emp_length,
    "loan_amnt": loan_amnt,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cred_hist_length,
    "person_home_ownership": home_ownership,
    "loan_intent": loan_intent,
    "cb_person_default_on_file": default_on_file
}])

tab_result, tab_explain, tab_compliance = st.tabs([
    "📊 Scoring Result", "🔍 Explainability", "⚖️ Compliance & Governance"
])

# --- ВКЛАДКА 1: Scoring Result ---
with tab_result:
    if evaluate_clicked:
        risk_proba, decision = process_new_applicant(
            applicant_data, final_model, final_preprocessor,
            group_thresholds, home_ownership
        )
        st.session_state["risk_proba"] = risk_proba
        st.session_state["decision"] = decision

    if "risk_proba" in st.session_state:
        st.subheader("Scoring Result")
        st.metric("Default Risk Probability", f"{st.session_state['risk_proba']:.1%}")

        decision = st.session_state["decision"]
        if decision == "AUTO_APPROVE":
            st.success(f"✅ {decision} — Application automatically approved")
        elif decision == "AUTO_REJECT":
            st.error(f"❌ {decision} — Application automatically rejected")
        else:
            st.warning(f"⚠️ {decision} — Routed to loan officer for manual review")
    else:
        st.info("Fill in the applicant's details in the sidebar and click **Evaluate Application**.")

# --- ВКЛАДКА 2: Explainability ---
with tab_explain:
    if "risk_proba" in st.session_state:
        st.subheader("Explainability (EU AI Act — Article 13)")

        applicant_transformed = final_preprocessor.transform(applicant_data)
        clean_feature_names = [
            name.split('__')[-1] for name in final_preprocessor.get_feature_names_out()
        ]

        explainer, shap_values = shap_explainer(final_model, applicant_transformed)

        explanation_text = get_explanation(
            shap_values=shap_values, sample_index=0,
            feature_names=clean_feature_names, X_display=applicant_data,
            reason_map=FEATURE_MAPPING, top_n=4
        )
        st.text(explanation_text)

        fig = get_waterfall_figure(shap_values, clean_feature_names, sample_index=0)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Evaluate an application first to see its explanation.")

# --- ВКЛАДКА 3: Compliance & Governance Dashboard ---
with tab_compliance:
    st.subheader("Decision Audit Trail (EU AI Act — Article 12)")

    try:
        log_df = pd.read_json("logs/decisions_log.jsonl", lines=True)
        st.dataframe(log_df, use_container_width=True)

        st.divider()
        st.subheader("Decision Distribution")
        st.bar_chart(log_df["decision"].value_counts())
    except (FileNotFoundError, ValueError):
        st.info("No decisions logged yet. Evaluate an application to populate the audit trail.")

    st.divider()
    st.subheader("Fairness Audit Summary (EU AI Act — Article 9)")

    try:
        with open("models/model_metrics.json", "r") as f:
            metrics = json.load(f)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Before Mitigation**")
            before = metrics["fairness_audit"]["person_home_ownership_before"]
            st.metric("Disparate Impact Ratio (DIR)", f"{before['DIR']:.2f}",
                      help="Target: ≥ 0.80")
            st.metric("Equal Opportunity Diff (EOD)", f"{before['EOD']:.2f}",
                      help="Target: ≤ 0.10")

        with col2:
            st.markdown("**After Mitigation**")
            after = metrics["fairness_audit"]["person_home_ownership_after"]
            st.metric("Disparate Impact Ratio (DIR)", f"{after['DIR']:.2f}",
                      delta=f"{after['DIR'] - before['DIR']:+.2f}",
                      help="Target: ≥ 0.80")
            st.metric("Equal Opportunity Diff (EOD)", f"{after['EOD']:.2f}",
                      delta=f"{after['EOD'] - before['EOD']:+.2f}", delta_color="inverse",
                      help="Target: ≤ 0.10")

        st.caption(
            "Attribute audited: **person_home_ownership**. Mitigation applied via "
            "per-group decision thresholds, at a measured cost of "
            f"{metrics['mitigation_tradeoff']['accuracy_before'] - metrics['mitigation_tradeoff']['accuracy_after']:.1%} "
            "overall accuracy."
        )

    except (FileNotFoundError, KeyError) as e:
        st.info("No fairness audit results found. Run the Day 5 fairness notebook first.")