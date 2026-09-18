"""
Customer Analytics Dashboard
------------------------------
Interactive Streamlit dashboard for exploring the customer dataset and
running live churn predictions with the trained model.

Run:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocessing import ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customer_data.csv"
MODELS_DIR = BASE_DIR / "models"

# ---- Color palette (shared with notebook / evaluation plots) --------------------
PRIMARY = "#2E5EAA"
SECONDARY = "#E8734A"
SUCCESS = "#2FA37C"
NEUTRAL = "#6B7280"
BG = "#F7F8FA"
CHURN_COLOR_MAP = {"Retained": PRIMARY, "Churned": SECONDARY}

st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Minimal custom styling ------------------------------------------------------
st.markdown(
    f"""
    <style>
    .main {{ background-color: {BG}; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.9rem; font-weight: 700; color: {PRIMARY}; }}
    div[data-testid="stMetricLabel"] {{ font-size: 0.85rem; color: {NEUTRAL}; }}
    h1, h2, h3 {{ color: #1F2937; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["churn_label"] = df["churned"].map({0: "Retained", 1: "Churned"})
    return df


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    with open(MODELS_DIR / "best_model_name.txt") as f:
        model_name = f.read().strip()
    comparison_df = pd.read_csv(MODELS_DIR / "model_comparison.csv")
    importance_df = pd.read_csv(MODELS_DIR / "feature_importance.csv")
    return model, preprocessor, model_name, comparison_df, importance_df


df = load_data()
model, preprocessor, model_name, comparison_df, importance_df = load_model_artifacts()

# ============================================================================
# SIDEBAR — filters + navigation
# ============================================================================
st.sidebar.title("📊 Customer Analytics")
st.sidebar.caption("DevelopersHub Corporation — Data Science Internship Project")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Customer Explorer", "Churn Prediction", "Model Performance"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
st.sidebar.caption("Applied to Overview & Customer Explorer pages")

selected_regions = st.sidebar.multiselect(
    "Region", options=sorted(df["region"].unique()), default=sorted(df["region"].unique())
)
selected_tiers = st.sidebar.multiselect(
    "Subscription Tier",
    options=sorted(df["subscription_tier"].unique()),
    default=sorted(df["subscription_tier"].unique()),
)
age_range = st.sidebar.slider(
    "Age Range", int(df["age"].min()), int(df["age"].max()),
    (int(df["age"].min()), int(df["age"].max())),
)

filtered_df = df[
    df["region"].isin(selected_regions)
    & df["subscription_tier"].isin(selected_tiers)
    & df["age"].between(*age_range)
]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered_df):,}** of {len(df):,} customers")

# ============================================================================
# PAGE: Overview
# ============================================================================
if page == "Overview":
    st.title("Customer Analytics Overview")
    st.caption("Synthetic customer dataset — purchase history, demographics & engagement metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(filtered_df):,}")
    col2.metric("Churn Rate", f"{filtered_df['churned'].mean():.1%}")
    col3.metric("Avg. Order Value", f"${filtered_df['avg_order_value'].mean():,.2f}")
    col4.metric("Avg. Tenure", f"{filtered_df['tenure_months'].mean():.1f} mo")

    st.markdown("---")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Churn Distribution")
        churn_counts = filtered_df["churn_label"].value_counts().reset_index()
        churn_counts.columns = ["Status", "Count"]
        fig = px.pie(
            churn_counts, names="Status", values="Count", hole=0.45,
            color="Status", color_discrete_map=CHURN_COLOR_MAP,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=340)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Churn Rate by Subscription Tier")
        tier_churn = (
            filtered_df.groupby("subscription_tier")["churned"].mean().reset_index()
            .sort_values("churned", ascending=False)
        )
        tier_churn["churned"] = tier_churn["churned"] * 100
        fig = px.bar(
            tier_churn, x="subscription_tier", y="churned",
            color_discrete_sequence=[SECONDARY], text_auto=".1f",
        )
        fig.update_layout(
            yaxis_title="Churn Rate (%)", xaxis_title="",
            margin=dict(t=10, b=10, l=10, r=10), height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns([1, 1])

    with c3:
        st.subheader("Age Distribution")
        fig = px.histogram(
            filtered_df, x="age", nbins=30, color_discrete_sequence=[PRIMARY],
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Customers by Region")
        region_counts = filtered_df["region"].value_counts().reset_index()
        region_counts.columns = ["Region", "Count"]
        fig = px.bar(
            region_counts, x="Region", y="Count", color_discrete_sequence=[SUCCESS],
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Engagement vs. Spend (colored by churn status)")
    sample_df = filtered_df.sample(min(2000, len(filtered_df)), random_state=42)
    fig = px.scatter(
        sample_df, x="days_since_last_purchase", y="total_spent_last_year",
        color="churn_label", color_discrete_map=CHURN_COLOR_MAP,
        opacity=0.55, labels={
            "days_since_last_purchase": "Days Since Last Purchase",
            "total_spent_last_year": "Total Spent (Last Year, $)",
            "churn_label": "Status",
        },
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Customers with a longer gap since their last purchase cluster toward higher churn — "
        "consistent with recency being the strongest single churn predictor in this dataset."
    )

# ============================================================================
# PAGE: Customer Explorer
# ============================================================================
elif page == "Customer Explorer":
    st.title("Customer Explorer")
    st.caption("Browse and search individual customer records")

    search_id = st.text_input("Search by Customer ID (optional)", placeholder="e.g. 100042")
    display_df = filtered_df.copy()
    if search_id.strip():
        try:
            display_df = display_df[display_df["customer_id"] == int(search_id.strip())]
        except ValueError:
            st.warning("Customer ID must be numeric.")

    st.dataframe(
        display_df[
            [
                "customer_id", "age", "gender", "region", "subscription_tier",
                "tenure_months", "num_purchases_last_year", "total_spent_last_year",
                "days_since_last_purchase", "email_open_rate", "churn_label",
            ]
        ].rename(columns={"churn_label": "status"}),
        use_container_width=True,
        height=460,
    )
    st.caption(f"{len(display_df):,} records shown")

    st.markdown("---")
    st.subheader("Compare a Numeric Feature Across Churn Status")
    numeric_choice = st.selectbox("Feature", NUMERIC_FEATURES, index=NUMERIC_FEATURES.index("days_since_last_purchase"))
    fig = px.box(
        filtered_df, x="churn_label", y=numeric_choice, color="churn_label",
        color_discrete_map=CHURN_COLOR_MAP,
        labels={"churn_label": "Status"},
    )
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=420)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: Churn Prediction
# ============================================================================
elif page == "Churn Prediction":
    st.title("Live Churn Prediction")
    st.caption(f"Powered by the trained **{model_name}** classifier")

    st.markdown("Enter a customer's profile to predict their churn risk in real time.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demographics**")
            age = st.slider("Age", 18, 80, 35)
            gender = st.selectbox("Gender", sorted(df["gender"].unique()))
            region = st.selectbox("Region", sorted(df["region"].unique()))
            annual_income = st.number_input("Annual Income ($)", 15000, 220000, 55000, step=1000)

        with c2:
            st.markdown("**Account & Purchases**")
            tenure_months = st.slider("Tenure (months)", 1, 120, 18)
            subscription_tier = st.selectbox("Subscription Tier", ["Basic", "Standard", "Premium"])
            num_purchases_last_year = st.slider("Purchases Last Year", 0, 50, 6)
            avg_order_value = st.number_input("Avg. Order Value ($)", 5.0, 900.0, 50.0, step=5.0)
            days_since_last_purchase = st.slider("Days Since Last Purchase", 0, 400, 30)
            returns_last_year = st.slider("Returns Last Year", 0, 10, 0)

        with c3:
            st.markdown("**Engagement**")
            email_open_rate = st.slider("Email Open Rate", 0.0, 1.0, 0.4, step=0.01)
            app_sessions_last_month = st.slider("App Sessions (Last Month)", 0, 60, 8)
            avg_session_duration_min = st.number_input("Avg. Session Duration (min)", 0.5, 60.0, 8.0, step=0.5)
            support_tickets_last_year = st.slider("Support Tickets Last Year", 0, 15, 1)
            referral_count = st.slider("Referral Count", 0, 15, 0)
            marketing_opt_in = st.checkbox("Opted in to marketing emails", value=True)

        submitted = st.form_submit_button("Predict Churn Risk", type="primary", use_container_width=True)

    if submitted:
        total_spent_last_year = num_purchases_last_year * avg_order_value
        loyalty_points = max(total_spent_last_year * 0.5 + app_sessions_last_month * 8, 0)

        input_row = pd.DataFrame([{
            "age": age,
            "annual_income": annual_income,
            "tenure_months": tenure_months,
            "num_purchases_last_year": num_purchases_last_year,
            "avg_order_value": avg_order_value,
            "total_spent_last_year": total_spent_last_year,
            "days_since_last_purchase": days_since_last_purchase,
            "returns_last_year": returns_last_year,
            "email_open_rate": email_open_rate,
            "app_sessions_last_month": app_sessions_last_month,
            "avg_session_duration_min": avg_session_duration_min,
            "support_tickets_last_year": support_tickets_last_year,
            "loyalty_points": loyalty_points,
            "referral_count": referral_count,
            "gender": gender,
            "region": region,
            "subscription_tier": subscription_tier,
            "marketing_opt_in": int(marketing_opt_in),
        }])[ALL_FEATURES]

        input_t = preprocessor.transform(input_row)
        churn_proba = model.predict_proba(input_t)[0, 1]
        churn_pred = int(churn_proba >= 0.5)

        st.markdown("---")
        r1, r2 = st.columns([1, 2])

        with r1:
            st.metric("Predicted Churn Probability", f"{churn_proba:.1%}")
            if churn_pred == 1:
                st.error("⚠️ High churn risk")
            else:
                st.success("✅ Low churn risk")

        with r2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_proba * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": SECONDARY if churn_pred else PRIMARY},
                    "steps": [
                        {"range": [0, 40], "color": "#E8F0FE"},
                        {"range": [40, 70], "color": "#FDE8DE"},
                        {"range": [70, 100], "color": "#FBD5C4"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig.update_layout(height=260, margin=dict(t=20, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "This prediction is generated by a model trained on synthetic data for demonstration "
            "purposes and should not be used for real business decisions."
        )

# ============================================================================
# PAGE: Model Performance
# ============================================================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.caption("Comparison of the three classification models evaluated for this project")

    st.subheader("Model Comparison")
    display_cols = ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    styled = comparison_df[display_cols].copy()
    for c in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        styled[c] = (styled[c] * 100).round(1).astype(str) + "%"
    st.dataframe(styled.rename(columns={
        "model": "Model", "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1_score": "F1 Score", "roc_auc": "ROC-AUC",
    }), use_container_width=True, hide_index=True)

    best_row = comparison_df.iloc[0]
    st.info(f"**Best model: {best_row['model']}** — {best_row['accuracy']:.1%} test accuracy, {best_row['roc_auc']:.3f} ROC-AUC")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Accuracy by Model")
        fig = px.bar(
            comparison_df.sort_values("accuracy"), x="accuracy", y="model", orientation="h",
            color_discrete_sequence=[PRIMARY], text_auto=".1%",
        )
        fig.update_layout(
            xaxis_title="Accuracy", yaxis_title="", margin=dict(t=10, b=10, l=10, r=10), height=320,
            xaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("ROC-AUC by Model")
        fig = px.bar(
            comparison_df.sort_values("roc_auc"), x="roc_auc", y="model", orientation="h",
            color_discrete_sequence=[SUCCESS], text_auto=".3f",
        )
        fig.update_layout(
            xaxis_title="ROC-AUC", yaxis_title="", margin=dict(t=10, b=10, l=10, r=10), height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader(f"Feature Importance — {best_row['model']}")
    top_importance = importance_df.head(12).sort_values("importance")
    fig = px.bar(
        top_importance, x="importance", y="feature", orientation="h",
        color_discrete_sequence=[PRIMARY],
    )
    fig.update_layout(
        xaxis_title="Importance", yaxis_title="", margin=dict(t=10, b=10, l=10, r=10), height=440,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "`days_since_last_purchase` dominates as the strongest churn predictor, followed by "
        "subscription tier, support tickets, and marketing opt-in status."
    )

st.markdown("---")
st.caption(
    "Built as part of a Data Science & Analytics Internship at DevelopersHub Corporation. "
    "Dataset is synthetically generated for demonstration purposes."
)
