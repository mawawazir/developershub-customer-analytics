"""
generate_data.py
-----------------
Generates a synthetic customer analytics dataset for the DevelopersHub
Customer Analytics project.

The dataset simulates 10,000+ customers with demographic, purchase-history,
and engagement features, plus a binary target (`churned`) that depends on
those features in a realistic, learnable way (with injected noise so the
classification task isn't trivial).

Run:
    python src/generate_data.py
Output:
    data/customer_data.csv
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 10500


def generate_customer_data(n=N_CUSTOMERS, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- Demographics -----------------------------------------------
    customer_id = np.arange(100000, 100000 + n)

    age = rng.normal(loc=39, scale=13, size=n).clip(18, 80).round().astype(int)

    gender = rng.choice(
        ["Female", "Male", "Non-binary"], size=n, p=[0.49, 0.47, 0.04]
    )

    regions = ["North", "South", "East", "West", "Central"]
    region = rng.choice(regions, size=n, p=[0.22, 0.24, 0.19, 0.20, 0.15])

    income_base = 28000 + (age - 18) * 950
    annual_income = (income_base + rng.normal(0, 14000, size=n)).clip(15000, 220000).round(-2)

    # ---- Account / tenure ---------------------------------------------
    tenure_months = rng.gamma(shape=2.2, scale=14, size=n).clip(1, 120).round().astype(int)

    subscription_tier = rng.choice(
        ["Basic", "Standard", "Premium"], size=n, p=[0.45, 0.38, 0.17]
    )

    # ---- Purchase history ----------------------------------------------
    tier_multiplier = pd.Series(subscription_tier).map(
        {"Basic": 0.7, "Standard": 1.0, "Premium": 1.6}
    ).values

    num_purchases_last_year = (
        rng.poisson(lam=6 * tier_multiplier, size=n) + rng.integers(0, 3, size=n)
    ).clip(0, 80)

    avg_order_value = (
        rng.gamma(shape=2.5, scale=22, size=n) * tier_multiplier
    ).clip(5, 900).round(2)

    total_spent_last_year = (num_purchases_last_year * avg_order_value).round(2)

    days_since_last_purchase = rng.exponential(scale=45, size=n).clip(0, 730).round().astype(int)

    returns_last_year = rng.binomial(
        n=np.maximum(num_purchases_last_year, 1), p=0.06
    )

    # ---- Engagement metrics ---------------------------------------------
    email_open_rate = rng.beta(2.2, 3.2, size=n).round(3)  # 0-1
    app_sessions_last_month = rng.negative_binomial(4, 0.35, size=n).clip(0, 90)
    avg_session_duration_min = rng.gamma(shape=2.0, scale=4.5, size=n).clip(0.5, 60).round(2)
    support_tickets_last_year = rng.poisson(lam=1.1, size=n).clip(0, 20)

    loyalty_points = (
        total_spent_last_year * 0.5 + app_sessions_last_month * 8 + rng.normal(0, 150, size=n)
    ).clip(0, None).round().astype(int)

    referral_count = rng.poisson(lam=0.6, size=n).clip(0, 15)

    marketing_opt_in = rng.choice([1, 0], size=n, p=[0.62, 0.38])

    # ---- Target: churned ------------------------------------------------
    # Build a latent "churn score" from a realistic combination of signals.
    # Includes genuine non-linear structure (a tenure threshold effect, and
    # a returns x tenure interaction) so tree-based models (XGBoost, Random
    # Forest) have real structure to exploit beyond a linear combination -
    # matching how these models tend to outperform logistic regression in
    # practice on this kind of tabular behavioral data.
    z = (
        0.028 * days_since_last_purchase
        - 0.055 * num_purchases_last_year
        - 0.60 * email_open_rate
        - 0.045 * app_sessions_last_month
        - 0.020 * tenure_months
        + 0.35 * returns_last_year
        + 0.18 * support_tickets_last_year
        - 0.000006 * total_spent_last_year
        - 0.45 * marketing_opt_in
        - pd.Series(subscription_tier).map({"Basic": -0.1, "Standard": 0.15, "Premium": 0.55}).values
    )

    # Non-linear structure #1: customers in their first 6 months behave
    # differently - an early "honeymoon risk window" where low engagement
    # is a much stronger churn signal than it is for established customers.
    new_customer = tenure_months < 6
    z = z + new_customer * (0.5 - email_open_rate) * 3.6

    # Non-linear structure #2: a return matters much more for a brand-new
    # customer (early bad experience) than for a long-tenured one who has
    # weathered issues before. This is a genuine interaction, not a main
    # effect, so it specifically rewards models that can split on it.
    z = z + returns_last_year * np.where(tenure_months < 12, 0.9, 0.05)

    # Non-linear structure #3: a support ticket combined with low recent
    # purchase activity signals real frustration; a ticket alone (from an
    # otherwise active customer) is often just a resolved question.
    low_activity = num_purchases_last_year < 4
    z = z + support_tickets_last_year * np.where(low_activity, 0.5, 0.05)

    z = (z - z.mean()) / z.std()  # standardize
    churn_prob = 1 / (1 + np.exp(-(z * 4.1 - 0.15)))
    # small amount of label noise keeps the task learnable but not trivial,
    # tuned so a tree-based model lands in the ~85% accuracy range
    noise = rng.normal(0, 0.032, size=n)
    churn_prob_noisy = np.clip(churn_prob + noise, 0.01, 0.99)
    churned = rng.binomial(1, churn_prob_noisy)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "gender": gender,
        "region": region,
        "annual_income": annual_income,
        "tenure_months": tenure_months,
        "subscription_tier": subscription_tier,
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
        "marketing_opt_in": marketing_opt_in,
        "churned": churned,
    })

    # Inject a small amount of realistic missingness (real-world data is messy)
    for col, frac in [("annual_income", 0.03), ("email_open_rate", 0.02), ("avg_session_duration_min", 0.015)]:
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_customer_data()
    out_path = "data/customer_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} rows -> {out_path}")
    print(f"Churn rate: {df['churned'].mean():.2%}")
    print(f"Missing values:\n{df.isna().sum()[df.isna().sum() > 0]}")
