"""
main.py
=======
Swin-DeepSCN-AgriVision — End-to-End Plant Disease Classification Pipeline
==========================================================================

Pipeline Overview
-----------------

  ┌─────────────────────────────────────────────────────┐
  │ 1.  Data Loading (data_loader.py)                   │
  │     PlantVillage ──► FilteredImageFolder            │
  │                  ──► Stratified Train/Test split    │
  │                  ──► Augmented DataLoaders          │
  └───────────────────────────┬─────────────────────────┘
                              │  (B, 3, 224, 224) batches
  ┌───────────────────────────▼─────────────────────────┐
  │ 2.  Feature Extraction (feature_extractor.py)        │
  │     Pre-trained Swin-B (frozen, no_grad)             │
  │     → L2-normalised 1024-d embeddings                │
  │     → Cached to disk (./cache/)                      │
  └──────┬────────────────────────────────────┬──────────┘
         │  (N_train, 1024)                   │  (N_test, 1024)
  ┌──────▼────────────────┐   ┌───────────────▼──────────┐
  │ 3.  DeepSCN (deepscn) │   │ 4.  MLP Baseline          │
  │   Frozen random W_hid │   │   Adam + CrossEntropy     │
  │   Pseudoinverse W_out │   │   Backprop 30 epochs      │
  │   (one matrix invert) │   │   (iterative gradient)    │
  └──────┬────────────────┘   └───────────────┬──────────┘
         │                                    │
  ┌──────▼────────────────────────────────────▼──────────┐
  │ 5.  Evaluation (evaluation.py)                        │
  │     Accuracy / Precision / Recall / F1-Score          │
  │     Training-time comparison                          │
  │     Confusion matrices saved as PNG                   │
  └──────────────────────────────────────────────────────┘
         │
  ┌──────▼───────────────────────┐
  │ 6.  XAI – Grad-CAM++         │
  │     (xai_gradcam.py)         │
  │     Attention overlay saved  │
  └──────────────────────────────┘

Usage
-----
    python main.py [--force-extract] [--xai <path_to_image>]

Arguments
---------
    --force-extract   : re-run Swin feature extraction even if cache exists
    --xai <img_path>  : path to a leaf image for Grad-CAM visualisation
    --no-xai          : skip the XAI step entirely

Author: Generated for METU "Mathematical Models in Image & Video Processing"
"""

import argparse
import os
import sys
import time

import numpy as np

from src import config
from src.data_loader        import get_dataloaders
from src.feature_extractor  import extract_and_cache
from src.deepscn            import DeepSCN
from src.baseline_mlp       import MLPTrainer
from src.evaluation         import (
    compute_metrics,
    plot_comparison,
    plot_confusion_matrix,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swin-DeepSCN Plant Disease Classification Pipeline"
    )
    parser.add_argument(
        "--force-extract", action="store_true",
        help="Recompute Swin embeddings even if cache files exist."
    )
    parser.add_argument(
        "--xai", type=str, default=None, metavar="IMAGE_PATH",
        help="Path to a leaf image for Grad-CAM visualisation."
    )
    parser.add_argument(
        "--no-xai", action="store_true",
        help="Skip the XAI / Grad-CAM step."
    )
    parser.add_argument(
        "--deepscn-nodes", type=int, default=config.DEEPSCN_MAX_NODES,
        help=f"Max hidden nodes for DeepSCN (default: {config.DEEPSCN_MAX_NODES})."
    )
    parser.add_argument(
        "--mlp-epochs", type=int, default=config.MLP_EPOCHS,
        help=f"Training epochs for MLP (default: {config.MLP_EPOCHS})."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("\n" + "=" * 66)
    print("  Swin-DeepSCN-AgriVision — Plant Disease Classification")
    print("=" * 66)
    print(f"  Device          : {config.DEVICE}")
    print(f"  Dataset root    : {config.DATASET_ROOT}")
    print(f"  Selected classes: {config.SELECTED_CLASSES}")
    print(f"  Output dir      : {config.OUTPUT_DIR}")
    print("=" * 66 + "\n")

    # =========================================================================
    # STEP 1 — Data Loading
    # =========================================================================
    print("─" * 40)
    print("STEP 1 — Loading dataset")
    print("─" * 40)

    if not os.path.isdir(config.DATASET_ROOT):
        print(
            f"\n[ERROR] Dataset root not found: '{config.DATASET_ROOT}'\n"
            f"  Please adjust DATASET_ROOT in config.py to point to your\n"
            f"  local PlantVillage directory.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    train_loader, test_loader, class_names = get_dataloaders()

    # =========================================================================
    # STEP 2 — Swin Transformer Feature Extraction
    # =========================================================================
    print("\n" + "─" * 40)
    print("STEP 2 — Swin Transformer Feature Extraction")
    print("─" * 40)

    X_train, y_train, X_test, y_test = extract_and_cache(
        train_loader, test_loader,
        force_recompute=args.force_extract,
    )

    print(f"\n  X_train : {X_train.shape}  (N_train × embed_dim)")
    print(f"  X_test  : {X_test.shape}   (N_test  × embed_dim)")

    n_classes  = len(class_names)
    embed_dim  = X_train.shape[1]

    # =========================================================================
    # STEP 3 — DeepSCN Training
    # =========================================================================
    print("\n" + "─" * 40)
    print("STEP 3 — DeepSCN Training (Pseudoinverse)")
    print("─" * 40)

    deepscn = DeepSCN(
        max_nodes    = args.deepscn_nodes,
        n_candidates = config.DEEPSCN_CANDIDATES,
        tolerance    = config.DEEPSCN_TOLERANCE,
        ridge_lambda = config.DEEPSCN_LAMBDA,
        weight_range = config.DEEPSCN_WEIGHT_RANGE,
        activation   = config.DEEPSCN_ACTIVATION,
        verbose      = True,
    )
    deepscn.fit(X_train, y_train)

    # =========================================================================
    # STEP 4 — MLP Baseline Training
    # =========================================================================
    print("\n" + "─" * 40)
    print("STEP 4 — MLP Baseline Training (Backpropagation)")
    print("─" * 40)

    mlp_trainer = MLPTrainer(
        input_dim  = embed_dim,
        n_classes  = n_classes,
        epochs     = args.mlp_epochs,
    )
    mlp_trainer.fit(X_train, y_train)

    # =========================================================================
    # STEP 5 — Evaluation
    # =========================================================================
    print("\n" + "─" * 40)
    print("STEP 5 — Evaluation")
    print("─" * 40)

    # ---- DeepSCN on test set ----
    y_pred_deepscn = deepscn.predict(X_test)
    deepscn_metrics = compute_metrics(
        y_true     = y_test,
        y_pred     = y_pred_deepscn,
        class_names= class_names,
        model_name = "DeepSCN (Pseudoinverse)",
        train_time = deepscn.train_time_,
    )

    plot_confusion_matrix(
        y_true     = y_test,
        y_pred     = y_pred_deepscn,
        class_names= class_names,
        model_name = "DeepSCN",
        save_path  = config.CONFUSION_MATRIX_DEEPSCN,
        normalise  = True,
    )

    # ---- MLP on test set ----
    y_pred_mlp = mlp_trainer.predict(X_test)
    mlp_metrics = compute_metrics(
        y_true     = y_test,
        y_pred     = y_pred_mlp,
        class_names= class_names,
        model_name = "MLP Baseline (Adam)",
        train_time = mlp_trainer.train_time_,
    )

    plot_confusion_matrix(
        y_true     = y_test,
        y_pred     = y_pred_mlp,
        class_names= class_names,
        model_name = "MLP Baseline",
        save_path  = config.CONFUSION_MATRIX_MLP,
        normalise  = True,
    )

    # ---- Comparison chart ----
    comparison_path = os.path.join(config.OUTPUT_DIR, "model_comparison.png")
    plot_comparison(deepscn_metrics, mlp_metrics, save_path=comparison_path)

    # ---- Summary table ----
    print("\n" + "=" * 66)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 66)
    header = f"{'Metric':<22} {'DeepSCN (Pseudoinverse)':>22} {'MLP (Adam)':>14}"
    print(header)
    print("-" * 66)
    for key, label in [
        ("train_time", "Training Time (s)"),
        ("accuracy",   "Accuracy"),
        ("precision",  "Precision (macro)"),
        ("recall",     "Recall (macro)"),
        ("f1",         "F1-Score (macro)"),
    ]:
        d_val = deepscn_metrics[key]
        m_val = mlp_metrics[key]
        print(f"  {label:<20} {d_val:>22.4f} {m_val:>14.4f}")
    print("-" * 66)

    speedup = mlp_metrics["train_time"] / max(deepscn_metrics["train_time"], 1e-9)
    print(f"\n  DeepSCN was {speedup:.1f}× faster to train than MLP.\n")
    print(f"  Hidden nodes used: {deepscn.n_nodes_}")
    print(f"  Output files saved to: {config.OUTPUT_DIR}/")
    print("=" * 66)

    # =========================================================================
    # STEP 6 — XAI / Grad-CAM (optional)
    # =========================================================================
    if not args.no_xai:
        print("\n" + "─" * 40)
        print("STEP 6 — XAI: Grad-CAM++ Attention Map")
        print("─" * 40)

        xai_image_path = args.xai

        # If no explicit image was given, find the first test image automatically
        if xai_image_path is None:
            xai_image_path = _find_sample_image(config.DATASET_ROOT,
                                                 config.SELECTED_CLASSES)

        if xai_image_path and os.path.isfile(xai_image_path):
            try:
                from src.xai_gradcam import generate_gradcam
                generate_gradcam(
                    image_path  = xai_image_path,
                    class_names = class_names,
                    save_path   = config.GRADCAM_OUTPUT,
                )
            except Exception as exc:
                print(f"[XAI] Grad-CAM skipped due to error: {exc}")
                print("       Install 'grad-cam' package: pip install grad-cam")
        else:
            print("[XAI] No valid image path provided. "
                  "Use --xai <path> or check DATASET_ROOT.")
            print("       Skipping Grad-CAM step.")

    print("\n[Pipeline] All done.\n")


# ---------------------------------------------------------------------------
# Helper: auto-find a sample image for XAI
# ---------------------------------------------------------------------------

def _find_sample_image(dataset_root: str, selected_classes: list) -> str:
    """
    Finds the first .jpg image in the first available class folder.

    Returns an absolute path string, or None if nothing is found.
    """
    for cls in selected_classes:
        cls_dir = os.path.join(dataset_root, cls)
        if os.path.isdir(cls_dir):
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    return os.path.join(cls_dir, fname)
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
