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

def _swin_reshape_transform(tensor: torch.Tensor, height: int = 7,
                             width: int = 7) -> torch.Tensor:
    """
    Reshape feature blocks to spatial (B, C, H, W) layout for Grad-CAM.
    
    Timm recent versions return a 4D tensor (B, H, W, C) directly for Swin stages.
    Older versions returned a flat sequence (B, H*W, C).
    """
    if tensor.dim() == 4:
        # If already (B, H, W, C), just permute channels to PyTorch format (B, C, H, W)
        return tensor.permute(0, 3, 1, 2)
    else:
        # Fallback for old (B, Sequence, C) format
        result = tensor.reshape(
            tensor.size(0),   # batch
            height,           # spatial H
            width,            # spatial W
            tensor.size(2),   # channels / embed_dim
        )
        return result.permute(0, 3, 1, 2)


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

    model.eval()

    # ---- Step 2: Identify the target layer ----
    try:
        target_layer = [model.layers[-1].blocks[-1].norm1]
    except AttributeError:
        target_layer = [model.norm]

    # ---- Step 3: Prepare the image ----
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
    print(f"[XAI] Predicted class: {pred_class} ({class_label})")
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
