"""
main.py
Full GBM pipeline for disease diagnosis using the Kaggle Breast Cancer dataset.
"""

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

from preprocessing import build_preprocessor
from model_gbm import build_gbm, cross_val_check, evaluate_model, save_model, save_metrics


def main(random_state=42):
    # === Setup directories ===
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # === Load raw data ===
    df = pd.read_csv("data/raw/data.csv")
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    # === Save cleaned dataset ===
    processed_path = "data/processed/breast_cancer_clean.csv"
    df.to_csv(processed_path, index=False)
    print(f"✅ Cleaned dataset saved to {processed_path}")
    print(f"Shape: {df.shape[0]} samples × {df.shape[1]} columns")

    # === Basic EDA ===
    summary = df.describe().T
    summary.to_csv("outputs/data_summary.csv")
    print("📊 Summary stats saved to outputs/data_summary.csv")

    plt.figure(figsize=(6, 4))
    sns.histplot(df["radius_mean"], bins=30, kde=True, color="royalblue")
    plt.title("Distribution of radius_mean")
    plt.savefig("figures/hist_radius_mean.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(5, 3))
    sns.countplot(x="diagnosis", data=df, palette="Set2")
    plt.title("Class Distribution (Benign=0, Malignant=1)")
    plt.savefig("figures/class_balance.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr.iloc[:10, :10], cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap (First 10 Features)")
    plt.tight_layout()
    plt.savefig("figures/corr_heatmap.png")
    plt.show()
    plt.close()

    # === Split data ===
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    print("🔹 Train:", X_train.shape, " Test:", X_test.shape)

    # === Build pipeline ===
    numeric_features = list(X.columns)
    preprocessor = build_preprocessor(numeric_features=numeric_features)
    gbm = build_gbm(random_state=random_state)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("gbm", gbm)
    ])

    # === Cross-validation ===
    cv_results = cross_val_check(gbm, X_train, y_train, folds=5)
    print("📈 Cross-validation:", json.dumps(cv_results, indent=2))

    # === Fit model ===
    pipeline.fit(X_train, y_train)

    # === Evaluate ===
    metrics, y_prob, y_pred = evaluate_model(pipeline, X_test, y_test)
    metrics.update(cv_results)
    print("✅ Test Metrics:", json.dumps(metrics, indent=2))

    save_model(pipeline, "outputs/gbm_model.joblib")
    save_metrics(metrics, "outputs/metrics.json")

    # === ROC and Confusion Matrix ===
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve (Test Set)")
    plt.savefig("figures/roc_curve.png")
    plt.show()
    plt.close()

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Benign", "Malignant"])
    plt.title("Confusion Matrix")
    plt.savefig("figures/confusion_matrix.png")
    plt.show()
    plt.close()

    # === Feature importance ===
    gbm_model = pipeline.named_steps["gbm"]
    importances = gbm_model.feature_importances_
    feat_imp = pd.DataFrame({"feature": numeric_features, "importance": importances})
    feat_imp = feat_imp.sort_values("importance", ascending=False)
    feat_imp.to_csv("outputs/feature_importances.csv", index=False)

    top10 = feat_imp.head(10)
    plt.figure(figsize=(6, 4))
    sns.barplot(x="importance", y="feature", data=top10, palette="crest")
    plt.title("Top 10 Important Features")
    plt.tight_layout()
    plt.savefig("figures/top10_features.png")
    plt.show()
    plt.close()

    print("🎉 All results saved to 'outputs/' and 'figures/'.")


if __name__ == "__main__":
    main()
