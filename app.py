import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Graduation Risk Dashboard", layout="wide")


@st.cache_resource
def build_data_and_models():
    np.random.seed(1)
    N = 2000
    entry_mode = np.random.choice(["UTME", "Direct Entry"], size=N, p=[0.75, 0.25])
    attendance_rate = np.clip(np.random.beta(a=6, b=2, size=N), 0, 1)
    study_hours = np.clip(np.random.normal(loc=15, scale=6, size=N), 0, 40)
    entry_cgpa = np.clip(np.random.normal(loc=3.6, scale=0.6, size=N), 1.0, 5.0)
    financial_stress = np.random.binomial(1, 0.35, size=N)

    df = pd.DataFrame({
        "entry_mode": entry_mode, "attendance_rate": attendance_rate,
        "study_hours_per_week": study_hours, "entry_cgpa": entry_cgpa,
        "financial_stress": financial_stress,
    })

    def standardize(x):
        return (x - x.mean()) / x.std(), x.mean(), x.std()

    z_attendance, m_att, s_att = standardize(df["attendance_rate"])
    z_entry_cgpa, m_cgpa, s_cgpa = standardize(df["entry_cgpa"])
    z_study_hours, m_study, s_study = standardize(df["study_hours_per_week"])
    z_entry_mode = (df["entry_mode"] == "Direct Entry").astype(int)

    np.random.seed(202)
    log_odds = (0.9 + 1.1*z_attendance + 0.8*z_entry_cgpa + 0.4*z_study_hours
                - 0.5*df["financial_stress"] - 0.2*z_entry_mode)
    p_graduate = 1 / (1 + np.exp(-log_odds))
    df["graduated"] = np.random.binomial(1, p_graduate)

    np.random.seed(303)
    noise = np.random.normal(0, 0.25, size=N)
    final_cgpa = np.clip(
        df["entry_cgpa"] + 0.35*z_attendance + 0.15*z_study_hours
        - 0.20*df["financial_stress"] + noise, 1.0, 5.0
    )
    df["final_cgpa"] = final_cgpa

    def classify(cgpa):
        if cgpa >= 4.50: return "First Class"
        elif cgpa >= 3.50: return "Second Class Upper"
        elif cgpa >= 2.40: return "Second Class Lower"
        else: return "Third Class"

    df["class_of_degree"] = df["final_cgpa"].apply(classify)
    df.loc[df["graduated"] == 0, "class_of_degree"] = "N/A - Did Not Graduate"

    df["z_attendance"], df["z_entry_cgpa"], df["z_study_hours"], df["z_entry_mode"] = (
        z_attendance, z_entry_cgpa, z_study_hours, z_entry_mode
    )

    feature_cols = ["z_attendance", "z_entry_cgpa", "z_study_hours", "financial_stress", "z_entry_mode"]
    X = df[feature_cols].values
    y = df["graduated"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logit = LogisticRegression(penalty=None, max_iter=1000).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=505).fit(X_train, y_train)
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=606).fit(X_train, y_train)

    grad_df = df[df["graduated"] == 1].copy()
    Xc = grad_df[feature_cols].values
    yc = grad_df["class_of_degree"].values
    multi_logit = LogisticRegression(penalty=None, max_iter=2000)
    multi_logit.fit(Xc, yc)

    standardization_params = {
        "attendance": (m_att, s_att), "entry_cgpa": (m_cgpa, s_cgpa), "study_hours": (m_study, s_study)
    }

    return logit, rf, gbm, multi_logit, standardization_params, feature_cols


logit_model, rf_model, gbm_model, multi_logit_model, std_params, feature_cols = build_data_and_models()

st.title("🎓 Student Graduation Risk Dashboard")
st.caption("Simulated model for demonstration - trained on synthetic Nigerian-university-style data")

st.sidebar.header("Enter Student Variables")
entry_mode_input = st.sidebar.selectbox("Entry Mode", ["UTME", "Direct Entry"])
attendance_input = st.sidebar.slider("Attendance Rate", 0.0, 1.0, 0.75, 0.01)
study_hours_input = st.sidebar.slider("Study Hours per Week", 0, 40, 15)
entry_cgpa_input = st.sidebar.slider("Entry/Current CGPA", 1.0, 5.0, 3.6, 0.01)
financial_stress_input = st.sidebar.selectbox("Financial Stress", ["No", "Yes"])

m_att, s_att = std_params["attendance"]
m_cgpa, s_cgpa = std_params["entry_cgpa"]
m_study, s_study = std_params["study_hours"]

z_att = (attendance_input - m_att) / s_att
z_cgpa = (entry_cgpa_input - m_cgpa) / s_cgpa
z_study = (study_hours_input - m_study) / s_study
z_mode = 1 if entry_mode_input == "Direct Entry" else 0
fin_stress = 1 if financial_stress_input == "Yes" else 0

X_input = np.array([[z_att, z_cgpa, z_study, fin_stress, z_mode]])

col1, col2, col3 = st.columns(3)
logit_p = logit_model.predict_proba(X_input)[0, 1]
rf_p = rf_model.predict_proba(X_input)[0, 1]
gbm_p = gbm_model.predict_proba(X_input)[0, 1]

col1.metric("Logistic Regression", f"{logit_p:.1%}")
col2.metric("Random Forest", f"{rf_p:.1%}")
col3.metric("Gradient Boosting", f"{gbm_p:.1%}")

st.subheader("Graduation Probability (average across models)")
avg_p = np.mean([logit_p, rf_p, gbm_p])
st.progress(avg_p)
st.write(f"**{avg_p:.1%}** average predicted probability of graduating")

if avg_p < 0.4:
    st.error("HIGH RISK - recommend early intervention")
elif avg_p < 0.65:
    st.warning("MODERATE RISK - monitor closely")
else:
    st.success("LOW RISK - on track")

st.subheader("Odds Ratio Interpretation (Logistic Regression)")
coefs = pd.Series(logit_model.coef_[0], index=feature_cols)
odds_ratios = np.exp(coefs)
for feat, orat in odds_ratios.items():
    direction = "increases" if orat > 1 else "decreases"
    st.write(f"- **{feat}**: each 1-SD increase {direction} odds of graduating by a factor of **{orat:.2f}**")

st.subheader("Class of Degree Probability (if graduating)")
class_probs = multi_logit_model.predict_proba(X_input)[0]
class_prob_df = pd.DataFrame({
    "Class": multi_logit_model.classes_,
    "Probability": class_probs
}).sort_values("Probability", ascending=False)
st.bar_chart(class_prob_df.set_index("Class"))

st.caption("This dashboard uses a SIMULATED dataset for demonstration purposes, "
           "not real student records. Built as a portfolio project applying statistical "
           "machine learning to an education-risk use cases.")