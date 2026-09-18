"""
evaluate_model.py
--------------------
Loads the saved best model + preprocessor, re-evaluates on the held-out
test set, and generates diagnostic visualizations (confusion matrix,
ROC curve, precision-recall curve, feature importance chart) saved to
reports/figures/.

Run:
    python src/evaluate_model.py
Outputs:
    reports/figures/confusion_matrix.png
    reports/figures/roc_curve.png
    reports/figures/precision_recall_curve.png
    reports/figures/feature_importance.png
    reports/model_evaluation_report.txt
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from preprocessing import load_data, train_test_split_data

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Brand-ish palette used consistently across notebook + dashboard + evaluation plots
PALETTE = {
    "primary": "#2E5EAA",
    "secondary": "#E8734A",
    "success": "#2FA37C",
    "neutral": "#6B7280",
    "bg": "#F7F8FA",
}


def main():
    print("Loading model, preprocessor, and test data...")
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    with open(MODELS_DIR / "best_model_name.txt") as f:
        model_name = f.read().strip()

    df = load_data()
    _, X_test, _, y_test = train_test_split_data(df)
    X_test_t = preprocessor.transform(X_test)

    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    # ---- Text report -----------------------------------------------------
    report = classification_report(y_test, y_pred, target_names=["Retained", "Churned"])
    report_path = BASE_DIR / "reports" / "model_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Best Model: {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"\n{report}")
    print(f"Saved text report -> {report_path}")

    # ---- Confusion matrix --------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"],
        annot_kws={"size": 14, "weight": "bold"}, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # ---- ROC curve --------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2.5, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=1.2, linestyle="--", label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=13, weight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", bbox_inches="tight")
    plt.close()

    # ---- Precision-Recall curve ---------------------------------------------
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=PALETTE["secondary"], lw=2.5, label=f"AP = {ap_score:.3f}")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=13, weight="bold")
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png", bbox_inches="tight")
    plt.close()

    # ---- Feature importance ------------------------------------------------
    importance_df = pd.read_csv(MODELS_DIR / "feature_importance.csv").head(12)
    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1],
        color=PALETTE["primary"],
    )
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Top 12 Feature Importances — {model_name}", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close()

    print(f"\nSaved 4 diagnostic plots -> {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
