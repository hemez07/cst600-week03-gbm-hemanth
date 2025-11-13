"""
model_gbm.py
Functions to create, train, and evaluate a Gradient Boosting Classifier model.
"""

import json
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from joblib import dump


def build_gbm(random_state=42):
    """Create a Gradient Boosting model with tuned hyperparameters."""
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=random_state,
        n_iter_no_change=10,
        validation_fraction=0.1,
        tol=1e-4
    )
    return model


def cross_val_check(estimator, X_train, y_train, folds=5):
    """Perform cross-validation to check model stability."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    acc = cross_val_score(estimator, X_train, y_train, cv=skf, scoring="accuracy")
    roc = cross_val_score(estimator, X_train, y_train, cv=skf, scoring="roc_auc")

    return {
        "cv_accuracy_mean": float(acc.mean()),
        "cv_accuracy_std": float(acc.std()),
        "cv_roc_auc_mean": float(roc.mean()),
        "cv_roc_auc_std": float(roc.std())
    }


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate model on hold-out test set."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics, y_prob, y_pred


def save_model(model, path="outputs/gbm_model.joblib"):
    dump(model, path)
    return path


def save_metrics(metrics, path="outputs/metrics.json"):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
