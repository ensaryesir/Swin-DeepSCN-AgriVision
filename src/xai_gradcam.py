"""
xai_gradcam.py
==============
Module 6 – Explainable AI: Grad-CAM Attention Maps for Swin Transformer
=========================================================================

What is Grad-CAM?
-----------------
Gradient-weighted Class Activation Mapping (Grad-CAM; Selvaraju et al., 2017)
highlights which spatial regions of an input image most strongly influenced
a model's prediction for a specific class.

Given a convolutional (or attention-based) layer L producing feature maps A:

    1. Compute the gradient of the class score S_c w.r.t. each feature map A_k:
            ∂S_c / ∂A_k^{i,j}    (where i,j index spatial positions)

    2. Global average pool the gradients to obtain importance weights:
            α_k^c = (1 / Z) ΣΣ_{i,j} (∂S_c / ∂A_k^{i,j})

    3. Create the weighted feature combination and apply ReLU:
            L_CAM = ReLU(Σ_k α_k^c · A_k)

    4. Upsample L_CAM to the input image size and overlay it as a heatmap.

ReLU is applied because we are only interested in features that have a
*positive* influence on the class score.

Swin Transformer Adaptation
----------------------------
Swin Transformer uses window-based self-attention instead of convolution.
The pytorch-grad-cam library handles this via the 'SwinT' wrapper, which
targets the last layer-norm + attention output as the "convolutional" layer.

We use the `GradCAMPlusPlus` variant (Chattopadhay et al., 2018) which
produces smoother and more localised maps than vanilla Grad-CAM by using
2nd-order gradient information.

Bug-Fix Log (v2)
-----------------
1. TARGET LAYER: Changed from `norm1` (pre-attention) to `model.norm`
   (final LayerNorm after all stages). `norm1` captures features BEFORE
   the shifted-window attention — its activations do not reflect the
   attention-modulated spatial patterns, causing edge/corner artefacts
   from the window-partition padding. `model.norm` outputs post-attention,
   post-MLP features that encode where the model actually "looked".

2. GRADIENT FLOW: The backbone must have `requires_grad = True` during
   Grad-CAM so that gradients flow from the class score back through the
   target layer. Previously, while parameters were not explicitly frozen,
   no explicit enablement was done either. Now we explicitly enable grads
   on ALL parameters for the Grad-CAM pass (weights are never *updated*
   since no optimizer step is called).

3. RESHAPE TRANSFORM: Made auto-detecting — reads actual spatial dims
   from the tensor instead of relying on hardcoded height=7, width=7.
   This makes the code robust to different Swin variants (tiny/small/base).

4. UPSAMPLING: Applied Gaussian smoothing (σ=1.0) after bilinear
   upsampling to soften the blocky 7×7 → 224×224 interpolation artefacts
   that cause sharp grid-like edges in the heatmap.

5. IMAGE PREPROCESSING & L2 NORM: Verified correct; no changes needed.
"""

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

# pytorch-grad-cam (installed as 'grad-cam' package)
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src import config


# ---------------------------------------------------------------------------
# Helper: reshape transform for timm Swin models
# ---------------------------------------------------------------------------

def _swin_reshape_transform(tensor: torch.Tensor,
                             height: int = 7,
                             width: int = 7) -> torch.Tensor:
    """
    Reshape feature blocks to spatial (B, C, H, W) layout for Grad-CAM.

    Timm Swin layers output tensors in two possible formats depending
    on the layer and timm version:
      - 4D: (B, H, W, C) — from norm1, model.norm, block output
      - 3D: (B, H*W, C)  — from norm2 (after window-reverse + reshape)

    FIX: Auto-detect spatial dimensions instead of relying on hardcoded
    height/width.  For 3D tensors, we compute H = W = √(seq_len) which
    is always valid for Swin (square spatial grids at each stage).
    """
    if tensor.dim() == 4:
        # Already (B, H, W, C) — just permute to PyTorch (B, C, H, W)
        return tensor.permute(0, 3, 1, 2)
    elif tensor.dim() == 3:
        # (B, seq_len, C) — infer spatial dimensions
        B, seq_len, C = tensor.shape
        h = w = int(seq_len ** 0.5)
        if h * w != seq_len:
            # Non-square fallback: use provided height/width hints
            h, w = height, width
        result = tensor.reshape(B, h, w, C)
        return result.permute(0, 3, 1, 2)
    else:
        raise ValueError(
            f"[XAI] Unexpected tensor dim={tensor.dim()}, shape={tensor.shape}"
        )


# ---------------------------------------------------------------------------
# Helper: Gaussian smoothing for heatmap post-processing
# ---------------------------------------------------------------------------

def _smooth_cam(cam: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to the upsampled CAM heatmap to reduce
    the blocky 7×7 → 224×224 interpolation artefacts.

    Uses scipy if available, otherwise falls back to a simple box blur.
    """
    try:
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(cam, sigma=sigma)
    except ImportError:
        # Simple 3×3 box blur as fallback
        from numpy.lib.stride_tricks import sliding_window_view
        pad = 1
        padded = np.pad(cam, pad, mode="reflect")
        windows = sliding_window_view(padded, (3, 3))
        smoothed = windows.mean(axis=(-2, -1))

    # Re-normalise to [0, 1]
    vmin, vmax = smoothed.min(), smoothed.max()
    if vmax - vmin > 1e-8:
        smoothed = (smoothed - vmin) / (vmax - vmin)
    return smoothed.astype(np.float32)


# ---------------------------------------------------------------------------
# Helper: find the best target layer for Grad-CAM
# ---------------------------------------------------------------------------

def _find_target_layer(model: nn.Module) -> list:
    """
    Select the optimal target layer for Grad-CAM on a timm Swin model.

    Priority order (best → fallback):
      1. model.norm       — Final LayerNorm applied to the 7×7 spatial
                            feature map AFTER all 4 stages.  This captures
                            post-attention, post-MLP features: the most
                            refined spatial representation before global
                            average pooling.  Output: (B, 7, 7, 1024).

      2. model.layers[-1].blocks[-1].norm2
                          — Post-attention LayerNorm inside the last block.
                            Output: (B, 49, 1024) — 3D but reshape-able.

      3. model.layers[-1].blocks[-1].norm1
                          — Pre-attention LayerNorm.  WORST choice because
                            activations do NOT contain attention information.
                            The shifted-window partition/reverse can inject
                            edge/corner artefacts into these features.

    Why `model.norm` is the best choice:
      The Swin forward pass is:
        x = patch_embed(img)
        for stage in self.layers:
            x = stage(x)            # each stage: blocks + patch_merge
        x = self.norm(x)            # ← TARGET: final spatial features
        x = self.head(x)            # global_pool → fc → class scores

      At `model.norm`, the tensor is still spatial (B, 7, 7, 1024) and
      contains ALL attention-modulated information from all 4 stages.
      Gradients from the class score flow back through the short path
      head → norm, giving strong, clean spatial signals.
    """
    # Priority 1: model.norm (final LayerNorm — best for Swin)
    if hasattr(model, "norm") and isinstance(model.norm, nn.LayerNorm):
        layer = model.norm
        print(f"[XAI] Target layer: model.norm (final LayerNorm, post-attention)")
        return [layer]

    # Priority 2: norm2 in last block (post-attention, pre-MLP)
    try:
        layer = model.layers[-1].blocks[-1].norm2
        print(f"[XAI] Target layer: layers[-1].blocks[-1].norm2 (post-attention)")
        return [layer]
    except (AttributeError, IndexError):
        pass

    # Priority 3: norm1 in last block (pre-attention — least ideal)
    try:
        layer = model.layers[-1].blocks[-1].norm1
        print(f"[XAI] Target layer: layers[-1].blocks[-1].norm1 (pre-attention, fallback)")
        return [layer]
    except (AttributeError, IndexError):
        pass

    raise RuntimeError(
        "[XAI] Could not find a suitable target layer in the model. "
        "Please verify the timm model architecture."
    )


# ---------------------------------------------------------------------------
# Main XAI function
# ---------------------------------------------------------------------------

def generate_gradcam(
    image_path: str,
    target_class: Optional[int] = None,
    model_name: str = config.SWIN_MODEL_NAME,
    save_path: str = config.GRADCAM_OUTPUT,
    class_names: Optional[list] = None,
    trained_head_state: Optional[dict] = None,
) -> np.ndarray:
    """
    Generate Grad-CAM++ attention map for a single leaf image.

    Parameters
    ----------
    image_path         : absolute path to the input image file
    target_class       : class index to explain (None → uses predicted class)
    model_name         : timm model name (should match the extractor model)
    save_path          : where to save the overlay image
    class_names        : optional list of class names for the figure title
    trained_head_state : optional state_dict from a trained classification head.
                         If provided, replaces the random head with trained weights
                         so that Grad-CAM explains a model that actually learned the
                         PlantVillage classes. Without this, the head is randomly
                         initialised and predictions are meaningless.

    Returns
    -------
    cam_image : (H, W, 3) uint8 numpy array — overlay of heatmap + image
    """
    device = config.DEVICE

    # ---- Step 0: Seed for deterministic Grad-CAM output ----
    # When called standalone (not via main.py), seeds may not be set yet.
    # Re-seeding is idempotent and guarantees the same heatmap every run.
    config.set_seed(config.RANDOM_SEED)

    # ---- Step 1: Load model WITH its classification head ----
    n_classes = len(config.SELECTED_CLASSES)
    print(f"[XAI] Loading model '{model_name}' for Grad-CAM ...")
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=n_classes,
    ).to(device)

    # ---- Step 1b: Replace the random head with trained weights ----
    # The critical fix: without trained weights the classification head
    # is randomly initialised, producing meaningless predictions.
    if trained_head_state is not None:
        # Build a small MLP head matching baseline_mlp architecture
        from src.baseline_mlp import MLP
        embed_dim = model.head.fc.in_features
        trained_mlp = MLP(
            input_dim=embed_dim,
            n_classes=n_classes,
        )
        trained_mlp.load_state_dict(trained_head_state)
        trained_mlp.to(device)
        trained_mlp.eval()

        # Wrapper that applies L2 normalization before the MLP.
        # This is CRITICAL: during feature extraction (feature_extractor.py),
        # embeddings are L2-normalised before being fed to the MLP.
        # Without this, the MLP receives unnormalised activations and
        # produces meaningless predictions.
        class _L2NormAndMLP(nn.Module):
            def __init__(self, mlp_sequential):
                super().__init__()
                self.net = mlp_sequential
            def forward(self, x):
                x = torch.nn.functional.normalize(x, p=2, dim=-1)
                return self.net(x)

        # Replace ONLY the final FC layer inside ClassifierHead.
        # This preserves global_pool + flatten so the tensor arriving
        # at our wrapper is correctly shaped as (B, 1024) — not 4D.
        model.head.fc = _L2NormAndMLP(trained_mlp.net)
        print("[XAI] Loaded TRAINED classification head (MLP weights + L2 norm)")
    else:
        print("[XAI] WARNING: Using untrained random head — predictions may be inaccurate!")

    # ---- Step 1c: FIX #4 — Ensure gradient flow through backbone ----
    # Grad-CAM needs gradients to flow from the class score back through
    # the target layer.  We explicitly enable requires_grad on ALL
    # parameters.  This does NOT mean we update them (no optimizer.step)
    # — it only allows ∂score/∂activation to be computed.
    for param in model.parameters():
        param.requires_grad = True

    model.eval()

    # ---- Step 2: FIX #1 — Select the best target layer ----
    # OLD (BUGGY):  target_layer = [model.layers[-1].blocks[-1].norm1]
    #   norm1 captures features BEFORE attention → edge/corner artefacts.
    # NEW:  model.norm = final LayerNorm AFTER all stages, post-attention.
    target_layer = _find_target_layer(model)

    # ---- Step 3: Prepare the image ----
    # FIX #3: Exactly matches the test-time transform in data_loader.py.
    # The transform chain is: Resize → ToTensor → Normalize(ImageNet stats)
    # No augmentation (flip/jitter/rotation) — identical to test pipeline.
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])

    pil_image  = Image.open(image_path).convert("RGB")
    pil_resized = pil_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))

    img_float = np.array(pil_resized, dtype=np.float32) / 255.0

    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # ---- Step 4: Run model to get predicted class ----
    with torch.no_grad():
        logits = model(input_tensor)
        pred_class = logits.argmax(dim=1).item()

    class_idx = target_class if target_class is not None else pred_class
    class_label = (
        class_names[class_idx] if class_names else f"Class {class_idx}"
    )
    pred_label = (
        class_names[pred_class] if class_names else f"Class {pred_class}"
    )
    print(f"[XAI] Predicted class: {pred_class} ({pred_label})")
    print(f"[XAI] Explaining class: {class_idx} ({class_label})")

    # ---- Step 5: Generate Grad-CAM++ heatmap ----
    cam = GradCAMPlusPlus(
        model=model,
        target_layers=target_layer,
        reshape_transform=_swin_reshape_transform,
    )
    targets = [ClassifierOutputTarget(class_idx)]

    # grayscale_cam shape: (1, H, W) — values ∈ [0, 1]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]   # (H, W)

    # ---- Step 5b: FIX #5 — Smooth the upsampled heatmap ----
    # The raw CAM is upsampled from 7×7 → 224×224 via bilinear interpolation
    # inside pytorch-grad-cam.  This produces blocky artefacts at grid cell
    # boundaries.  A light Gaussian blur removes them without destroying
    # the spatial signal.
    grayscale_cam = _smooth_cam(grayscale_cam, sigma=1.5)

    # ---- Step 6: Overlay heatmap on original image ----
    cam_image = show_cam_on_image(
        img_float,          # (H, W, 3) float [0, 1]
        grayscale_cam,      # (H, W)    float [0, 1]
        use_rgb=True,
        colormap=4,         # cv2.COLORMAP_JET  (0=Autumn, 4=Jet)
        image_weight=0.5,   # 50% original + 50% heatmap
    )

    # ---- Step 7: Save and display side-by-side ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(pil_resized)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("Grad-CAM++ Heatmap", fontsize=12)
    axes[1].axis("off")
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="jet"),
        ax=axes[1], fraction=0.046, pad=0.04
    )

    axes[2].imshow(cam_image)
    axes[2].set_title(f"Overlay — Explaining: {class_label}", fontsize=12)
    axes[2].axis("off")

    plt.suptitle(
        "Swin Transformer — Grad-CAM++ Disease-Region Attention",
        fontsize=13, y=1.01
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[XAI] Saved Grad-CAM overlay → {save_path}")
    return cam_image


# ---------------------------------------------------------------------------
# Multi-image attention grid (optional convenience helper)
# ---------------------------------------------------------------------------

def generate_multi_gradcam(
    image_paths: list,
    class_names: list,
    model_name: str = config.SWIN_MODEL_NAME,
    save_path: str = None,
) -> None:
    """
    Generate a grid of Grad-CAM overlays for multiple sample images
    (one per class).

    Parameters
    ----------
    image_paths : list of image file paths (one representative per class)
    class_names : matching list of class name strings
    model_name  : timm model name
    save_path   : if provided, save grid to this path
    """
    n = len(image_paths)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))

    for i, (img_path, cls_name) in enumerate(zip(image_paths, class_names)):
        pil_image   = Image.open(img_path).convert("RGB")
        pil_resized = pil_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        img_float   = np.array(pil_resized, dtype=np.float32) / 255.0

        # Re-use the main function to compute the overlay
        cam_image = generate_gradcam(
            image_path=img_path,
            class_names=class_names,
            save_path=os.path.join(
                config.OUTPUT_DIR, f"gradcam_{cls_name}.png"
            ),
        )

        axes[0, i].imshow(pil_resized)
        axes[0, i].set_title(f"Original\n{cls_name}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(cam_image)
        axes[1, i].set_title("Grad-CAM++", fontsize=9)
        axes[1, i].axis("off")

    plt.suptitle("Per-Class Disease Attention Maps", fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[XAI] Saved multi-image grid → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Standalone test (requires a real image path)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python xai_gradcam.py <path_to_leaf_image.jpg>")
        sys.exit(1)
    generate_gradcam(
        image_path=sys.argv[1],
        class_names=config.SELECTED_CLASSES,
        save_path=config.GRADCAM_OUTPUT,
    )
