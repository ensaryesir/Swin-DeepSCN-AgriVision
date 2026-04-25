"""
config.py
=========
Central configuration file for the Swin-DeepSCN-AgriVision pipeline.

All hyper-parameters, paths, and toggles live here so that every other
module can import from a single source of truth without hard-coding values.
"""

import os

# ---------------------------------------------------------------------------
# 1.  DATASET CONFIGURATION
# ---------------------------------------------------------------------------

# Root directory of the PlantVillage dataset (flat class-folder structure).
# Adjust this to wherever you extracted the dataset inside WSL.
# Expected layout:
#   DATASET_ROOT/
#     Tomato___healthy/           <-- named exactly as in the original dataset
#     Tomato___Early_blight/
#     Tomato___Late_blight/
#     Pepper__bell___healthy/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, "data", "PlantVillage")

# Subset of class folder-names to use.  Keeping 4 classes keeps GPU/RAM
# requirements manageable on a WSL 2 machine with modest resources.
SELECTED_CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

# Fraction of data reserved for the test split (stratified).
TEST_SPLIT    = 0.20
RANDOM_SEED   = 42

# ---------------------------------------------------------------------------
# 2.  IMAGE PREPROCESSING CONFIGURATION
# ---------------------------------------------------------------------------

# Swin Transformer (swin_base_patch4_window7_224) expects 224×224 RGB images.
IMAGE_SIZE    = 224
# ImageNet statistics — used because the Swin backbone is ImageNet-pretrained.
NORM_MEAN     = [0.485, 0.456, 0.406]
NORM_STD      = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# 3.  FEATURE EXTRACTION CONFIGURATION
# ---------------------------------------------------------------------------

# timm model name.  Change to 'swin_tiny_patch4_window7_224' for a
# lighter-weight option that still produces the same embedding dimension.
SWIN_MODEL_NAME   = "swin_base_patch4_window7_224"
# Embedding dimensionality produced by the Swin backbone (post global pool).
# swin_base → 1024-d,  swin_tiny / swin_small → 768-d.
# Set to None to auto-detect from model metadata at runtime.
SWIN_EMBED_DIM    = None    # None → auto-detected in feature_extractor.py

FEATURE_BATCH_SIZE = 32     # batch size during feature extraction (no grad)

# Directory where extracted .npy embedding arrays will be cached to disk
# so re-runs skip the expensive forward pass.
# Directory where extracted .npy embedding arrays will be cached to disk
# so re-runs skip the expensive forward pass.

# ---------------------------------------------------------------------------
# 4.  DeepSCN CONFIGURATION
# ---------------------------------------------------------------------------

# Maximum number of hidden nodes the incremental SCN builder can reach.
DEEPSCN_MAX_NODES  = 512
# Number of new candidate nodes sampled at each construction step.
DEEPSCN_CANDIDATES = 10
# Tolerance threshold T in the SCN inequality constraint:
#   The new node is accepted only if it reduces the residual error.
DEEPSCN_TOLERANCE  = 1e-6
# Regularisation coefficient λ for the regularised pseudo-inverse:
#   W_out = (H^T H + λI)^{-1} H^T Y
#   λ prevents ill-conditioned solutions when H columns are near-collinear.
DEEPSCN_LAMBDA     = 1e-4
# Range for the random hidden-layer weight initialisation U[-r, r].
DEEPSCN_WEIGHT_RANGE = (-1.0, 1.0)
# Activation function for the hidden layer.  'relu' | 'tanh' | 'sigmoid'.
DEEPSCN_ACTIVATION = "relu"

# ---------------------------------------------------------------------------
# 5.  BASELINE MLP CONFIGURATION
# ---------------------------------------------------------------------------

MLP_HIDDEN_DIMS  = [512, 256]   # neurons per hidden layer
MLP_DROPOUT      = 0.3
MLP_LR           = 1e-3
MLP_EPOCHS       = 30
MLP_BATCH_SIZE   = 64
MLP_WEIGHT_DECAY = 1e-4

# ---------------------------------------------------------------------------
# 6.  OUTPUT / ARTEFACT PATHS
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FEATURE_CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

CONFUSION_MATRIX_DEEPSCN = os.path.join(OUTPUT_DIR, "confusion_matrix_deepscn.png")
CONFUSION_MATRIX_MLP      = os.path.join(OUTPUT_DIR, "confusion_matrix_mlp.png")
GRADCAM_OUTPUT            = os.path.join(OUTPUT_DIR, "gradcam_attention.png")

# ---------------------------------------------------------------------------
# 7.  HARDWARE
# ---------------------------------------------------------------------------

# 'cuda' | 'cpu'.  Feature extraction benefits greatly from a GPU.
# DeepSCN itself runs in NumPy on CPU (matrix inversion is not GPU-bound).
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
