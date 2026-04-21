"""
feature_extractor.py
====================
Module 2 – Hierarchical Feature Extraction via Swin Transformer
================================================================

Theory — Swin Transformer (Liu et al. 2021)
--------------------------------------------
The Swin Transformer introduces two key innovations over the original ViT:

  1. **Hierarchical feature maps** — using 4 stages that progressively
     halve the spatial resolution (patch merging), producing multi-scale
     representations analogous to a CNN's feature pyramid.

  2. **Shifted Window Self-Attention (SW-MSA)** — attention is computed
     locally within non-overlapping windows of size M×M (default 7×7).
     On alternate layers, the windows are shifted by (M/2, M/2) to allow
     cross-window connections.  Complexity is O(n·M²) rather than O(n²).

For our pipeline we use the Swin-B model pre-trained on ImageNet-22k and
fine-tuned on ImageNet-1k (provided by timm).  We discard the final linear
classification head and replace it with a global average pooling operation,
yielding a fixed-length 1024-d embedding per image.

These embeddings are then:
  - normalised (L2) for numerical stability in the DeepSCN pseudoinverse
  - cached to disk so repeated experiments skip the expensive forward pass

No gradients are computed (torch.no_grad()) because the backbone is entirely
frozen; we treat it as a deterministic, fixed feature function f: R^{H×W×3}
→ R^{1024}.
"""

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import timm
from tqdm import tqdm

from src import config


# ---------------------------------------------------------------------------
# Swin Feature Extractor class
# ---------------------------------------------------------------------------

class SwinFeatureExtractor(nn.Module):
    """
    Wraps a pre-trained timm Swin Transformer, strips its classification head,
    and exposes a single forward pass that returns a pooled embedding vector.

    Architecture outline (Swin-Base):
      Input  : (B, 3, 224, 224)
      Patch Embed (4×4, stride 4)  → (B, 56×56, 128)
      Stage 1 (2 blocks, window=7) → (B, 56×56, 128)
      Patch Merge                  → (B, 28×28, 256)
      Stage 2 (2 blocks)           → (B, 28×28, 256)
      Patch Merge                  → (B, 14×14, 512)
      Stage 3 (18 blocks)          → (B, 14×14, 512)
      Patch Merge                  → (B,  7× 7, 1024)
      Stage 4 (2 blocks)           → (B,  7× 7, 1024)
      Global Average Pool          → (B, 1024)
      [Classification head removed]

    Parameters
    ----------
    model_name : timm model identifier (default from config.SWIN_MODEL_NAME)
    pretrained : whether to download ImageNet weights (default True)
    device     : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model_name: str = config.SWIN_MODEL_NAME,
        pretrained: bool = True,
        device: str = config.DEVICE,
    ):
        super().__init__()
        self.device = device

        print(f"[FeatureExtractor] Loading '{model_name}' ...")
        # timm's create_model with num_classes=0 automatically removes the
        # final classification linear layer, leaving global_pool → embedding.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,      # remove FC head → output is embedding
            global_pool="avg",  # global average pool over spatial tokens
        )

        # Freeze ALL backbone parameters — we are using it purely as a
        # fixed feature function, not fine-tuning it.
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.to(self.device)
        self.backbone.eval()

        # Auto-detect embedding dimensionality by running a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE,
                                device=self.device)
            embed_dim = self.backbone(dummy).shape[-1]

        self.embed_dim = embed_dim
        print(f"[FeatureExtractor] Embedding dim = {self.embed_dim}")

    @torch.no_grad()
    def extract(self, loader: torch.utils.data.DataLoader) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Iterate over a DataLoader and extract embeddings for every image.

        Parameters
        ----------
        loader : DataLoader yielding (images, labels) batches

        Returns
        -------
        embeddings : np.ndarray of shape (N, embed_dim)   — float32
        labels     : np.ndarray of shape (N,)             — int64
        """
        all_embeds = []
        all_labels = []

        for images, labels in tqdm(loader, desc="  Extracting features"):
            images = images.to(self.device)

            # Forward pass through frozen Swin backbone.
            # Shape: (batch, embed_dim) — global-average-pooled token output
            embeds = self.backbone(images)

            # L2-normalise each embedding vector so that cosine distance
            # becomes equivalent to Euclidean distance; this stabilises the
            # pseudoinverse computation in DeepSCN.
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=-1)

            all_embeds.append(embeds.cpu().numpy())
            all_labels.append(labels.numpy())

        embeddings = np.vstack(all_embeds).astype(np.float32)
        labels_arr = np.concatenate(all_labels).astype(np.int64)

        return embeddings, labels_arr


# ---------------------------------------------------------------------------
# Public helper — extract and cache
# ---------------------------------------------------------------------------

def extract_and_cache(
    train_loader,
    test_loader,
    cache_dir: str = config.FEATURE_CACHE_DIR,
    model_name: str = config.SWIN_MODEL_NAME,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Swin embeddings for train+test, caching results to disk.

    If the cache files already exist (and force_recompute is False) the
    embeddings are loaded from disk — skipping the GPU forward pass entirely.

    Parameters
    ----------
    train_loader    : DataLoader for the training split
    test_loader     : DataLoader for the test split
    cache_dir       : directory to read/write .npy cache files
    model_name      : timm model identifier (used to version the cache)
    force_recompute : if True, always re-run extraction even if cache exists

    Returns
    -------
    X_train, y_train, X_test, y_test : numpy arrays
    """
    # Use model name as part of cache file names to avoid mixing caches
    tag = model_name.replace("/", "_")
    train_X_path = os.path.join(cache_dir, f"train_X_{tag}.npy")
    train_y_path = os.path.join(cache_dir, f"train_y_{tag}.npy")
    test_X_path  = os.path.join(cache_dir, f"test_X_{tag}.npy")
    test_y_path  = os.path.join(cache_dir, f"test_y_{tag}.npy")

    cache_exists = all(
        os.path.exists(p)
        for p in [train_X_path, train_y_path, test_X_path, test_y_path]
    )

    if cache_exists and not force_recompute:
        print("[FeatureExtractor] Loading embeddings from disk cache ...")
        X_train = np.load(train_X_path)
        y_train = np.load(train_y_path)
        X_test  = np.load(test_X_path)
        y_test  = np.load(test_y_path)
        print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
        return X_train, y_train, X_test, y_test

    extractor = SwinFeatureExtractor(
        model_name=model_name,
        pretrained=True,
        device=config.DEVICE,
    )

    print("\n[FeatureExtractor] Extracting TRAIN embeddings ...")
    X_train, y_train = extractor.extract(train_loader)

    print("\n[FeatureExtractor] Extracting TEST embeddings ...")
    X_test, y_test = extractor.extract(test_loader)

    # ---- Persist to disk ----
    os.makedirs(cache_dir, exist_ok=True)
    np.save(train_X_path, X_train)
    np.save(train_y_path, y_train)
    np.save(test_X_path,  X_test)
    np.save(test_y_path,  y_test)
    print(f"[FeatureExtractor] Saved embeddings to '{cache_dir}'")

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_loader import get_dataloaders

    train_loader, test_loader, classes = get_dataloaders()
    X_train, y_train, X_test, y_test = extract_and_cache(
        train_loader, test_loader, force_recompute=False
    )
    print(f"\nX_train shape : {X_train.shape}")
    print(f"y_train sample: {y_train[:10]}")
