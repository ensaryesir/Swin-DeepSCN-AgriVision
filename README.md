# 🌿 Swin-DeepSCN-AgriVision

**Plant Disease Classification via Hierarchical Feature Extraction + Stochastic Configuration Networks**

A complete, end-to-end Python pipeline for the course  
*"Mathematical Models in Image and Video Processing"* — MSc, 2025–2026.

---

## 📐 Architecture Overview

```
PlantVillage Dataset
       │
       ▼
┌──────────────────────────────────────────┐
│   1. Data Loader  (data_loader.py)        │
│   FilteredImageFolder + Stratified Split  │
│   Augmentation: Flip, Jitter, Rotation    │
└───────────────────┬──────────────────────┘
                    │  (B × 3 × 224 × 224)
                    ▼
┌──────────────────────────────────────────┐
│  2. Swin Transformer  (feature_extractor) │
│  Pre-trained backbone, frozen weights     │
│  Shifted Window Self-Attention (SW-MSA)   │
│  Output: L2-normalised 1024-d embeddings  │
│  Cached to ./cache/ for fast re-runs      │
└──────────┬─────────────────────┬──────────┘
           │                     │
    ┌──────▼──────┐      ┌───────▼──────┐
    │ 3. DeepSCN  │      │4. MLP Baseline│
    │  Frozen rand│      │ Adam Optimizer│
    │  hidden W   │      │ CrossEntropy  │
    │  Pseudoinv  │      │ 30 epochs     │
    │  output W   │      └───────┬──────┘
    └──────┬──────┘              │
           └──────────┬──────────┘
                      ▼
          ┌───────────────────────┐
          │    5. Evaluation       │
          │ Accuracy/Prec/Rec/F1  │
          │ Confusion Matrices    │
          │ Time Comparison       │
          └───────────┬───────────┘
                      ▼
          ┌───────────────────────┐
          │  6. XAI — Grad-CAM++  │
          │  Disease region maps  │
          └───────────────────────┘
```

---

## 🔬 Mathematical Core — DeepSCN

The **Stochastic Configuration Network** (Wang & Li, 2017) trains in two distinct phases:

### Hidden Layer — Random, Frozen Weights

Hidden weights **W** and biases **b** are drawn from `Uniform[-r, r]` and **never updated**:

```
H = σ(X · W + b)     (shape: N × L)
```

where `σ` is ReLU/Tanh/Sigmoid and `L` is the number of hidden nodes.

### Output Layer — Analytical Pseudoinverse

The output weights **β** solve the regularised least-squares problem in **one single matrix operation**:

```
β* = (Hᵀ H + λI)⁻¹ Hᵀ Y       (Ridge regression / Tikhonov)
```

This guarantees global optimality and eliminates gradient descent entirely.

### Incremental Node Construction

Nodes are added one at a time; a node is accepted only if it reduces the residual `‖Y − Hβ‖`:

```
For k = 1 → max_nodes:
    Sample L random candidates {wˡ, bˡ}
    Accept best candidate (corr w/ residual)
    Append to H, recompute β via pseudoinverse
    If no residual improvement → STOP
```

### Complexity Comparison

| Method | Training Cost |
|--------|--------------|
| Standard MLP | O(N · d · L · epochs) — iterative SGD |
| DeepSCN | O(K · L³) — K sequential matrix inversions |
| Speedup | Typically **10–100×** faster training |

---

## 📦 Project Structure

```
Swin-DeepSCN-AgriVision/
├── requirements.txt          # WSL/Linux dependencies
├── src/                      # Source code modules
│   ├── config.py             # Central configuration (paths, hyperparams)
│   ├── data_loader.py        # FilteredImageFolder + DataLoaders
│   ├── feature_extractor.py  # Swin Transformer backbone + caching
│   ├── deepscn.py            # DeepSCN from scratch (Math core)
│   ├── baseline_mlp.py       # MLP with backpropagation baseline
│   ├── evaluation.py         # Metrics, confusion matrices, comparison plots
│   └── xai_gradcam.py        # Grad-CAM++ explainability
├── main.py                   # End-to-end pipeline orchestrator
├── cache/                    # Auto-created: .npy embedding cache
└── outputs/                  # Auto-created: plots, confusion matrices
```

---

## 🚀 Setup & Usage

### 1. Install dependencies (WSL 2 / Linux)

```bash
# Create a virtual environment
python3 -m venv venv && source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# For PyTorch with CUDA 11.8 (replace with your CUDA version):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Configure the dataset path

Edit `src/config.py` and set `DATASET_ROOT` to your local PlantVillage path:

```python
# src/config.py
DATASET_ROOT = os.path.expanduser("~/data/PlantVillage")

SELECTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Pepper__bell___healthy",
]
```

> **Note**: Class folder names must match exactly what's on disk.  
> Run `ls ~/data/PlantVillage/` to verify the exact folder names.

### 3. Run the full pipeline

```bash
# Standard run (uses cached embeddings if available)
python main.py

# Force re-extraction of Swin embeddings
python main.py --force-extract

# Run with a custom leaf image for Grad-CAM
python main.py --xai ~/data/PlantVillage/Tomato___healthy/image001.jpg

# Skip the XAI step
python main.py --no-xai

# Tune DeepSCN and MLP parameters
python main.py --deepscn-nodes 512 --mlp-epochs 50
```

### 4. Run individual modules

```bash
# Test data loading only
python -m src.data_loader

# Test feature extraction only
python -m src.feature_extractor

# Test DeepSCN on synthetic data
python -m src.deepscn

# Test MLP on synthetic data
python -m src.baseline_mlp

# Generate Grad-CAM for a single image
python -m src.xai_gradcam ~/data/PlantVillage/Tomato___Early_blight/image001.jpg
```

---

## ⚙️ Key Hyperparameters

All parameters are in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SWIN_MODEL_NAME` | `swin_base_patch4_window7_224` | timm model ID |
| `DEEPSCN_MAX_NODES` | 1024 | Max hidden nodes |
| `DEEPSCN_CANDIDATES` | 100 | Random candidates per step |
| `DEEPSCN_LAMBDA` | 1e-4 | Ridge regularisation λ |
| `DEEPSCN_TOLERANCE` | 1e-6 | Residual stopping threshold |
| `MLP_EPOCHS` | 30 | Backprop training epochs |
| `MLP_LR` | 1e-3 | Adam learning rate |
| `TEST_SPLIT` | 0.20 | Test set fraction |

---

## 📊 Expected Outputs

| File | Description |
|------|-------------|
| `outputs/confusion_matrix_deepscn.png` | DeepSCN confusion matrix (normalised) |
| `outputs/confusion_matrix_mlp.png` | MLP confusion matrix (normalised) |
| `outputs/model_comparison.png` | Side-by-side metrics + training time chart |
| `outputs/gradcam_attention.png` | Grad-CAM++ disease region attention |
| `cache/train_X_*.npy` | Cached Swin train embeddings |
| `cache/test_X_*.npy` | Cached Swin test embeddings |

---

## 📚 References

1. **Swin Transformer**: Liu, Z. et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*. ICCV 2021. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)

2. **Stochastic Configuration Networks**: Wang, D., & Li, M. (2017). *Stochastic Configuration Networks: Fundamentals and Algorithms*. IEEE TNNLS, 28(12), 3066–3080. [DOI:10.1109/TNNLS.2017.2729236](https://doi.org/10.1109/TNNLS.2017.2729236)

3. **Grad-CAM**: Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)

4. **PlantVillage Dataset**: Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health*. [arXiv:1511.08060](https://arxiv.org/abs/1511.08060)
