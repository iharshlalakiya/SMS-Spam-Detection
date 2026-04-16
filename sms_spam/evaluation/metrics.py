"""
Evaluation Module for SMS Spam Detection

Provides metric calculation, confusion-matrix plotting, ROC curve plotting,
and model-comparison helpers.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score,
)


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate evaluation metrics for a binary classifier.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    y_pred_proba : array-like, optional
        Probability estimates for the positive class.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1_score``
        (and ``roc_auc`` when *y_pred_proba* is provided).
    """
    f1 = f1_score(y_true, y_pred, zero_division=0)
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1_score":  f1,
        "f1":        f1,   # backward-compatible alias
    }
    if y_pred_proba is not None:
        proba = y_pred_proba[:, 1] if len(np.array(y_pred_proba).shape) > 1 else y_pred_proba
        metrics["roc_auc"] = roc_auc_score(y_true, proba)
    return metrics


def print_metrics(metrics, model_name="Model"):
    """Print metrics in a formatted table."""
    print(f"\n{'='*50}")
    print(f"Results for {model_name}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"{metric.upper():15s}: {value:.4f}")
    print(f"{'='*50}")


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot a confusion matrix heatmap and optionally save it."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.show()
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, model_name, save_path=None):
    """Plot a ROC curve and optionally save it."""
    proba = y_pred_proba[:, 1] if len(np.array(y_pred_proba).shape) > 1 else y_pred_proba
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curve saved to {save_path}")
    plt.show()
    plt.close()


def compare_models(results_dict, save_path=None):
    """
    Bar charts for Accuracy, Precision, Recall, and F1-Score across models.

    Parameters
    ----------
    results_dict : dict
        ``{model_name: metrics_dict}``
    save_path : str, optional
        Path to save the figure.
    """
    df = pd.DataFrame(results_dict).T
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    available = [m for m in metrics if m in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = sns.color_palette("husl", len(df))

    for idx, metric in enumerate(available):
        ax = axes[idx]
        bars = ax.bar(df.index, df[metric], color=colors)
        ax.set_title(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        for bar, val in zip(bars, df[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison chart saved to {save_path}")
    plt.show()
    plt.close()


def create_comparison_table(results_dict):
    """Return a formatted DataFrame of model metrics."""
    df = pd.DataFrame(results_dict).T.round(4)
    if "training_time" in df.columns:
        df["training_time"] = df["training_time"].apply(lambda x: f"{x:.2f}s")
    df.index.name = "Model"
    return df


def plot_all_roc_curves(results_dict, y_true, save_path=None):
    """Plot ROC curves for multiple models on one figure."""
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", len(results_dict))

    for (model_name, data), color in zip(results_dict.items(), colors):
        if "y_pred_proba" in data:
            proba = data["y_pred_proba"]
            proba = proba[:, 1] if len(np.array(proba).shape) > 1 else proba
            fpr, tpr, _ = roc_curve(y_true, proba)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f"{model_name} (AUC = {auc(fpr, tpr):.4f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curves saved to {save_path}")
    plt.show()
    plt.close()


def save_results_to_csv(results_dict, filepath):
    """Save a comparison table to CSV."""
    df = create_comparison_table(results_dict)
    df.to_csv(filepath)
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
