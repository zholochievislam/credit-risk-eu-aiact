import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from src.modeling import load_final_model
from src.fairness import load_thresholds
from src.oversight import process_new_applicant
from src.explainability import shap_explainer, get_explanation, get_waterfall_figure, FEATURE_MAPPING

st.title("Credit Risk Scoring Engine")
st.caption("EU AI Act Compliant: Explainable & Fair Credit Decisioning")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #14181F;
}
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E5EA;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    color: #14181F;
}
[data-testid="stMetricLabel"] {
    color: #5B6472;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #0E7C86;
    border-bottom-color: #0E7C86;
}
button[kind="primary"] {
    background-color: #0E7C86;
    border: none;
    border-radius: 6px;
    font-weight: 500;
}
button[kind="primary"]:hover {
    background-color: #0B646C;
}
[data-testid="stDataFrame"], [data-testid="stExpander"] {
    border: 1px solid #E2E5EA;
    border-radius: 8px;
}
[data-testid="stAlert"] {
    border-radius: 6px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_PATH = os.path.join(_APP_DIR, "logs", "decisions_log.jsonl")


@st.cache_resource
def load_resources():
    model, preprocessor = load_final_model()
    thresholds = load_thresholds()
    return model, preprocessor, thresholds

final_model, final_preprocessor, group_thresholds = load_resources()

def render_status_chip(label, status_type):
    colors = {
        "positive": ("#EAF5EF", "#1F7A4D"),
        "negative": ("#FBEAE9", "#B3261E"),
        "neutral": ("#FDF3E3", "#B7791F"),
    }
    bg, fg = colors[status_type]
    st.markdown(
        f"""<div style="display:inline-flex; align-items:center; gap:8px;
            background-color:{bg}; color:{fg}; padding:6px 14px;
            border-radius:999px; font-family:'IBM Plex Mono', monospace;
            font-size:0.85rem; font-weight:600; letter-spacing:0.02em;">
            <span style="width:6px; height:6px; border-radius:50%; background-color:{fg};"></span>
            {label}</div>""",
        unsafe_allow_html=True
    )


#  SIDEBAR
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

# Scoring Result
with tab_result:
    if evaluate_clicked:
        risk_proba, decision = process_new_applicant(
            applicant_data, final_model, final_preprocessor,
            group_thresholds, home_ownership
        )
        st.session_state["risk_proba"] = risk_proba
        st.session_state["decision"] = decision
        st.session_state["eval_home_ownership"] = home_ownership
        st.session_state["eval_group_threshold"] = group_thresholds[home_ownership]

    if "risk_proba" in st.session_state:
        st.subheader("Scoring Result")
        st.metric("Default Risk Probability", f"{st.session_state['risk_proba']:.1%}")

        decision = st.session_state["decision"]
        group_threshold = st.session_state["eval_group_threshold"]
        margin = 0.10
        lower_bound = group_threshold - margin
        upper_bound = group_threshold + margin

        if decision == "AUTO_APPROVE":
            render_status_chip("AUTO APPROVE", "positive")
        elif decision == "AUTO_REJECT":
            render_status_chip("AUTO REJECT", "negative")
        else:
            render_status_chip("MANUAL REVIEW", "neutral")

        with st.expander("Why this decision? (Decision band explained)"):
            action = "sent it to a loan officer for manual review" if decision == "MANUAL_REVIEW" else "made an automatic decision"
            st.markdown(
                f"Applicants are grouped by home ownership type, and each group has its own approval "
                f"threshold. This applicant falls under {st.session_state['eval_home_ownership']}, "
                f"where the threshold is {group_threshold:.1%}. A margin of 10 percentage points is "
                f"applied on either side: applications below {lower_bound:.1%} are approved automatically, "
                f"applications above {upper_bound:.1%} are declined automatically, and anything in between "
                f"is sent to a loan officer for manual review. This applicant's calculated risk is "
                f"{st.session_state['risk_proba']:.1%}, which is why the system {action}."
            )
    else:
        st.info("Fill in the applicant's details in the sidebar and click **Evaluate Application**.")

# Explainability
with tab_explain:
    if "risk_proba" in st.session_state:
        st.subheader("Explainability (EU AI Act, Article 13)")

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

        with st.expander("How to read this explanation"):
            st.markdown(
                "The paragraph above translates the model's internal calculation into plain terms: "
                "it starts from the average risk across all applicants, then explains what pushed "
                "this specific applicant's risk up or down. The chart below shows the same reasoning "
                "in the model's own mathematical terms rather than as percentages. At the top, f(x) "
                "is this applicant's final score. At the bottom, E[f(X)] is the average score across "
                "all applicants in the training data, the same baseline referenced in the paragraph "
                "above. Blue bars lower the score from that baseline, red bars raise it, and they are "
                "stacked from the baseline on the left to the applicant's final score on the right. "
                "Both f(x) and E[f(X)] are shown on a mathematical scale called log odds rather than "
                "as percentages: converting between the two is exactly what the plain language "
                "summary above already does."
            )

        fig = get_waterfall_figure(shap_values, clean_feature_names, sample_index=0)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Evaluate an application first to see its explanation.")

# Compliance & Governance Dashboard
with tab_compliance:
    st.subheader("Decision Audit Trail (EU AI Act — Article 12)")

    try:
        log_df = pd.read_json(LOGS_PATH, lines=True)
        st.dataframe(log_df, use_container_width=True)

        st.divider()
        st.subheader("Decision Distribution")
        st.bar_chart(log_df["decision"].value_counts())
    except (FileNotFoundError, ValueError):
        st.info("No decisions logged yet. Evaluate an application to populate the audit trail.")

    st.divider()
    st.subheader("Fairness Audit Summary (EU AI Act, Article 9)")

    with st.expander("What do DIR and EOD mean, and why before/after?"):
        st.markdown(
            "Two measurements are tracked here. The first, Disparate Impact Ratio, compares how often "
            "applicants are approved across different housing situations. A value of 1.0 would mean "
            "equal approval rates, and anything below 0.80 is generally considered a warning sign. The "
            "second, Equal Opportunity Difference, measures something different: how reliably the model "
            "catches applicants who actually default, regardless of their housing situation. A value of "
            "0 would mean the model is equally accurate for everyone; values above 0.10 are generally "
            "considered meaningfully unequal.\n\n"
            "An earlier version of this model was noticeably less reliable at flagging real defaults "
            "among mortgage holders than among renters or homeowners. The figures below show the result "
            "after that was corrected by calibrating a separate approval threshold for each housing "
            "category, improving both measurements at a small cost to overall accuracy."
        )

    try:
        with open("models/model_metrics.json", "r") as f:
            metrics = json.load(f)

        audit = metrics["fairness_audit"]

        st.markdown("**Audited Attributes — Current Status**")

        attribute_rows = [
            ("Age Group", audit["age_group"]),
            ("Loan Intent", audit["loan_intent"]),
            ("Home Ownership (pre-mitigation)", audit["person_home_ownership_before"]),
            ("Home Ownership (post-mitigation)", audit["person_home_ownership_after"]),
        ]

        summary_df = pd.DataFrame([
            {
                "Attribute": name,
                "DIR": row["DIR"],
                "EOD": row["EOD"],
                "DIR Status": "✅ PASS" if row["DIR"] >= 0.80 else "⚠️ FLAG",
                "EOD Status": "✅ PASS" if row["EOD"] <= 0.10 else "⚠️ FLAG",
            }
            for name, row in attribute_rows
        ])

        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.caption("Thresholds: DIR ≥ 0.80 (Four-Fifths Rule), EOD ≤ 0.10 (commonly cited fairness benchmark).")

        st.divider()
        st.markdown("**Mitigation Detail — `person_home_ownership`**")

        col1, col2 = st.columns(2)
        before = audit["person_home_ownership_before"]
        after = audit["person_home_ownership_after"]

        with col1:
            st.markdown("*Before Mitigation*")
            st.metric("Disparate Impact Ratio (DIR)", f"{before['DIR']:.2f}")
            st.metric("Equal Opportunity Diff (EOD)", f"{before['EOD']:.2f}")

        with col2:
            st.markdown("*After Mitigation*")
            st.metric("Disparate Impact Ratio (DIR)", f"{after['DIR']:.2f}",
                       delta=f"{after['DIR'] - before['DIR']:+.2f}")
            st.metric("Equal Opportunity Diff (EOD)", f"{after['EOD']:.2f}",
                       delta=f"{after['EOD'] - before['EOD']:+.2f}", delta_color="inverse")

        tradeoff = metrics["mitigation_tradeoff"]
        acc_cost = tradeoff["accuracy_before"] - tradeoff["accuracy_after"]
        st.caption(
            f"Mitigation applied via per-group decision thresholds, at a measured cost of "
            f"{acc_cost:.1%} overall accuracy ({tradeoff['accuracy_before']:.1%} → {tradeoff['accuracy_after']:.1%})."
        )

    except (FileNotFoundError, KeyError):
        st.info("No fairness audit results found. Run the Day 5 fairness notebook first.")