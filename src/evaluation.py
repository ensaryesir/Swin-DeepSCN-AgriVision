"""
evaluation.py
=============
Module 5 – Evaluation, Metrics & Visualisation
================================================

This module provides a unified evaluation interface:
    * Computes Accuracy, Precision, Recall, F1-Score (macro + per-class).
    * Prints a formatted results table to the terminal.
    * Generates and saves styled Confusion Matrix figures using seaborn.
    * Produces a side-by-side comparison of DeepSCN vs. MLP metrics.

All sklearn metrics use macro averaging by default:
    Macro = arithmetic mean over per-class values.
    This treats all classes equally regardless of support, which is
    appropriate for the (approximately balanced) PlantVillage subsets.
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe in WSL headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src import config


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str = "Model",
    train_time: float = 0.0,
) -> Dict[str, float]:
    """
    Compute and print classification metrics for a single model.

    Parameters
    ----------
    y_true      : (N,) ground-truth integer labels
    y_pred      : (N,) predicted integer labels
    class_names : list of class name strings (in label-index order)
    model_name  : display name used in the printout
    train_time  : training wall-clock time in seconds

    Returns
    -------
    metrics : dict with keys:
        'accuracy', 'precision', 'recall', 'f1', 'train_time'
    """
    acc  = accuracy_score(y_true, y_pred)

    # Macro averaging: compute metric for each class, then take unweighted mean.
    # zero_division=0 avoids warnings when a class has no predicted samples.
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true,    y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true,         y_pred, average="macro", zero_division=0)

    # Full per-class report
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    # ---- Formatted terminal output ----
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  RESULTS — {model_name}")
    print(sep)
    print(f"  Total Training Time : {train_time:>10.4f} s")
    print(f"  Accuracy            : {acc:>10.4f}")
    print(f"  Precision (macro)   : {prec:>10.4f}")
    print(f"  Recall    (macro)   : {rec:>10.4f}")
    print(f"  F1-Score  (macro)   : {f1:>10.4f}")
    print(f"\n  Per-Class Report:\n")
    print(report)
    print(sep)

    return {
        "accuracy"   : float(acc),
        "precision"  : float(prec),
        "recall"     : float(rec),
        "f1"         : float(f1),
        "train_time" : float(train_time),
    }


# ---------------------------------------------------------------------------
# Confusion matrix visualisation
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: str,
    normalise: bool = True,
) -> None:
    """
    Render and save a styled confusion matrix heat-map.

    The confusion matrix C is defined as:
        C[i, j] = number of samples with true label i predicted as label j.

    When normalised=True, each row is divided by its sum (true class total),
    so the diagonal shows per-class recall (true positive rate).

    Parameters
    ----------
    y_true      : (N,) ground-truth labels
    y_pred      : (N,) predicted labels
    class_names : list of class name strings
    model_name  : title string for the figure
    save_path   : absolute path to save the .png file
    normalise   : if True, show row-normalised (recall) matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalise:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt      = ".2f"
        vmin, vmax = 0.0, 1.0
        cbar_label = "Row-normalised (Recall)"
    else:
        cm_display = cm
        fmt      = "d"
        vmin, vmax = None, None
        cbar_label = "Sample Count"

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=vmin, vmax=vmax,
        linewidths=0.5,
        linecolor="lightgrey",
        annot_kws={"size": 11},
        cbar_kws={"label": cbar_label},
    )

    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, pad=14)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluation] Saved confusion matrix → {save_path}")


# ---------------------------------------------------------------------------
# Side-by-side comparison figure
# ---------------------------------------------------------------------------

def plot_comparison(
    deepscn_metrics: Dict[str, float],
    mlp_metrics: Dict[str, float],
    save_path: Optional[str] = None,
) -> None:
    """
    Generate a grouped bar chart comparing DeepSCN vs. MLP on all metrics.

    Parameters
    ----------
    deepscn_metrics : dict returned by compute_metrics for DeepSCN
    mlp_metrics     : dict returned by compute_metrics for MLP
    save_path       : if provided, save the figure here
    """
    metric_keys   = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    deepscn_vals = [deepscn_metrics[k] for k in metric_keys]
    mlp_vals     = [mlp_metrics[k]     for k in metric_keys]

    x     = np.arange(len(metric_labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: classification metrics ----
    ax = axes[0]
    bars1 = ax.bar(x - width / 2, deepscn_vals, width, label="DeepSCN",
                   color="#3A86FF", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, mlp_vals,     width, label="MLP (Baseline)",
                   color="#FF6B6B", alpha=0.85, edgecolor="white")

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Classification Metrics: DeepSCN vs. MLP", fontsize=12)
    ax.legend(fontsize=10)
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ---- Right: training time comparison (log scale) ----
    ax2 = axes[1]
    models       = ["DeepSCN", "MLP (Baseline)"]
    times        = [deepscn_metrics["train_time"], mlp_metrics["train_time"]]
    colors       = ["#3A86FF", "#FF6B6B"]
    bars3 = ax2.bar(models, times, color=colors, alpha=0.85, edgecolor="white",
                    width=0.4)
    ax2.set_yscale("log")   # log scale makes the speedup visually obvious
    ax2.set_ylabel("Training Time (seconds) — log scale", fontsize=11)
    ax2.set_title("Training Time Comparison", fontsize=12)
    ax2.bar_label(bars3, fmt="%.2f s", padding=3, fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # Annotate speedup factor
    speedup = mlp_metrics["train_time"] / max(deepscn_metrics["train_time"], 1e-9)
    ax2.text(
        0, deepscn_metrics["train_time"] * 1.5,
        f"~{speedup:.1f}× faster",
        ha="center", va="bottom", fontsize=11,
        color="#3A86FF", fontweight="bold"
    )

    plt.suptitle("Swin-DeepSCN vs. MLP Baseline — Plant Disease Classification",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Evaluation] Saved comparison chart → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 4, size=200)
    y_pred = rng.integers(0, 4, size=200)
    classes = ["Healthy", "Early_blight", "Late_blight", "Pepper_healthy"]

    m = compute_metrics(y_true, y_pred, classes, "TestModel", train_time=1.23)
    plot_confusion_matrix(y_true, y_pred, classes, "TestModel",
                          save_path="./outputs/test_cm.png")
