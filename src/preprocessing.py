"""
preprocessing.py
------------------
Shared data-loading and preprocessing utilities used by the EDA notebook,
training script, evaluation script, and Streamlit dashboard so all four
stay consistent with each other.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "customer_data.csv"

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "tenure_months",
    "num_purchases_last_year",
    "avg_order_value",
    "total_spent_last_year",
    "days_since_last_purchase",
    "returns_last_year",
    "email_open_rate",
    "app_sessions_last_month",
    "avg_session_duration_min",
    "support_tickets_last_year",
    "loyalty_points",
    "referral_count",
]
CATEGORICAL_FEATURES = ["gender", "region", "subscription_tier"]
BINARY_FEATURES = ["marketing_opt_in"]
TARGET = "churned"
ID_COLUMN = "customer_id"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the raw customer dataset from disk."""
    df = pd.read_csv(path)
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - median-imputes + scales numeric features
      - most-frequent-imputes + one-hot-encodes categorical features
      - passes through the binary flag as-is
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
    ])
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Retrieve output feature names from a fitted ColumnTransformer."""
    num_names = NUMERIC_FEATURES
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    bin_names = BINARY_FEATURES
    return num_names + cat_names + bin_names


def train_test_split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
