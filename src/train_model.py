"""
train_model.py
----------------
Trains and compares three classification models for customer churn
prediction (Logistic Regression, Random Forest, XGBoost), selects the
best performer, and saves it (along with the fitted preprocessor and a
metrics summary) to the models/ directory.

Run:
    python src/train_model.py
Outputs:
    models/best_model.joblib
    models/preprocessor.joblib
    models/model_comparison.csv
    models/feature_importance.csv
"""

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from preprocessing import (
    load_data,
    build_preprocessor,
    get_feature_names,
    train_test_split_data,
)

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def evaluate_model(model, X_test_t, y_test, name: str) -> dict:
    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main():
    print("Loading data...")
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    print(f"Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")

    print("Fitting preprocessor...")
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    fitted_models = {}

    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        model.fit(X_train_t, y_train)
        elapsed = time.time() - t0

        cv_scores = cross_val_score(model, X_train_t, y_train, cv=5, scoring="accuracy", n_jobs=-1)

        metrics = evaluate_model(model, X_test_t, y_test, name)
        metrics["cv_accuracy_mean"] = cv_scores.mean()
        metrics["cv_accuracy_std"] = cv_scores.std()
        metrics["train_time_sec"] = round(elapsed, 2)

        results.append(metrics)
        fitted_models[name] = model

        print(
            f"  accuracy={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1_score']:.4f}  "
            f"roc_auc={metrics['roc_auc']:.4f}  "
            f"cv_acc={metrics['cv_accuracy_mean']:.4f}±{metrics['cv_accuracy_std']:.4f}  "
            f"({elapsed:.1f}s)"
        )

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name} (accuracy={results_df.iloc[0]['accuracy']:.4f})")

    # Feature importance (for tree-based models) or coefficients (for logistic regression)
    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_
    else:
        importance = np.abs(best_model.coef_[0])
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # Save artifacts
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    results_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    importance_df.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    with open(MODELS_DIR / "best_model_name.txt", "w") as f:
        f.write(best_name)

    print(f"\nSaved best model ({best_name}) and artifacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
