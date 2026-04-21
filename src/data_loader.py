"""
data_loader.py
==============
Module 1 – Data Loading & Preprocessing
========================================

Responsibilities
----------------
* Discover the PlantVillage directory tree and filter to SELECTED_CLASSES.
* Build stratified train/test splits with reproducible seeding.
* Return PyTorch DataLoader objects with proper transforms applied.

PlantVillage directory convention
----------------------------------
The standard Kaggle release uses the following layout:

    <DATASET_ROOT>/
        Tomato___healthy/
            image0001.jpg
            ...
        Tomato___Early_blight/
            ...

This module works with torchvision.datasets.ImageFolder which expects
exactly that flat class-folder structure.

Mathematical note on normalisation
------------------------------------
We subtract the ImageNet per-channel mean μ and divide by standard
deviation σ (both in [0,1] scale after ToTensor):

    x_norm = (x - μ) / σ

This ensures the input distribution is centred ~N(0,1) per channel,
which aligns with the training distribution of the pre-trained Swin
Transformer and accelerates convergence.
"""

import os
import sys
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from src import config


# ---------------------------------------------------------------------------
# Helper: build a class-filtered ImageFolder
# ---------------------------------------------------------------------------

class FilteredImageFolder(datasets.ImageFolder):
    """
    A thin subclass of ImageFolder that restricts loading to a user-defined
    list of class folder names.

    Parameters
    ----------
    root   : str  — path to the dataset root (parent of class folders)
    classes: list — list of class folder names to keep

    All other ImageFolder constructor arguments are forwarded unchanged.
    """

    def __init__(self, root: str, selected_classes: List[str], **kwargs):
        # Store the whitelist BEFORE calling super().__init__ because
        # ImageFolder.find_classes() is called inside __init__.
        self._selected_classes = set(selected_classes)
        super().__init__(root=root, **kwargs)

    # Override the class-discovery method so we only surface the classes
    # that appear in SELECTED_CLASSES.
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Returns only the subset of class directories that exist on disk
        AND are in the user-provided whitelist.
        """
        all_classes = sorted(
            entry.name
            for entry in os.scandir(directory)
            if entry.is_dir()
        )
        # Keep only whitelisted classes that are actually present in the root
        classes = [c for c in all_classes if c in self._selected_classes]

        missing = self._selected_classes - set(all_classes)
        if missing:
            print(
                f"[WARNING] The following SELECTED_CLASSES were NOT found "
                f"on disk and will be skipped: {missing}",
                file=sys.stderr,
            )

        if not classes:
            raise FileNotFoundError(
                f"None of SELECTED_CLASSES {self._selected_classes} were "
                f"found under '{directory}'.  "
                f"Please check DATASET_ROOT in config.py."
            )

        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return classes, class_to_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_transforms(train: bool) -> transforms.Compose:
    """
    Construct the image transform pipeline.

    Training transforms apply light augmentations to regularise the feature
    extractor implicitly (although we freeze the Swin backbone, augmentation
    still creates diversity in the embedding cache for potential future use).

    Parameters
    ----------
    train : bool — whether to include augmentation-specific transforms

    Returns
    -------
    torchvision.transforms.Compose
    """
    base = [
        # Step 1 — resize to the Swin Transformer's expected input resolution.
        #   Swin-B/224 uses a 4×4 patch stride, so token count = (224/4)^2 = 3136.
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    ]

    if train:
        base += [
            # Random horizontal flip: p=0.5 (leaves are bilaterally symmetric)
            transforms.RandomHorizontalFlip(p=0.5),
            # Colour jitter to simulate lighting variation in field images
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            # Random rotation ±15°: leaves can appear at any orientation
            transforms.RandomRotation(degrees=15),
        ]

    base += [
        # Convert PIL Image (H×W×C, uint8 [0,255]) →  Tensor (C×H×W, float [0,1])
        transforms.ToTensor(),
        # Channel-wise Z-score normalisation using ImageNet statistics
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ]

    return transforms.Compose(base)


def get_dataloaders(
    dataset_root: str = config.DATASET_ROOT,
    selected_classes: List[str] = config.SELECTED_CLASSES,
    test_split: float = config.TEST_SPLIT,
    random_seed: int = config.RANDOM_SEED,
    feature_batch_size: int = config.FEATURE_BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Build stratified train/test DataLoaders for the PlantVillage subset.

    Strategy
    --------
    1. Load the full filtered dataset with *no* augmentation to get indices.
    2. Perform a stratified split on (index, label) pairs using scikit-learn's
       train_test_split (which guarantees the same class proportions in both
       halves).
    3. Wrap the train Subset with augmentation transforms and the test Subset
       with base transforms.

    Note: torchvision Subset does not support per-Subset transforms directly;
    we handle this by creating two separate FilteredImageFolder instances
    (one with train transforms, one with test transforms) and applying the
    same index splits.

    Parameters
    ----------
    dataset_root      : root of PlantVillage dataset
    selected_classes  : list of class folder names to use
    test_split        : fraction of data to hold out for testing
    random_seed       : RNG seed for reproducibility
    feature_batch_size: batch size for the DataLoaders

    Returns
    -------
    train_loader : DataLoader (augmented)
    test_loader  : DataLoader (no augmentation)
    class_names  : list of class name strings in label-index order
    """
    print(f"\n[DataLoader] Loading dataset from: {dataset_root}")
    print(f"[DataLoader] Selected classes: {selected_classes}")

    # ---- Step 1: discover all sample indices without any augmentation ----
    full_dataset = FilteredImageFolder(
        root=dataset_root,
        selected_classes=selected_classes,
        transform=build_transforms(train=False),   # neutral transform
    )

    class_names = full_dataset.classes
    targets     = [label for _, label in full_dataset.samples]
    all_indices = list(range(len(full_dataset)))

    print(f"[DataLoader] Total samples found: {len(full_dataset)}")
    for idx, name in enumerate(class_names):
        count = targets.count(idx)
        print(f"             Class {idx} '{name}': {count} images")

    # ---- Step 2: stratified split ----
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_split,
        random_state=random_seed,
        stratify=targets,   # preserves class proportions
    )

    print(f"[DataLoader] Train samples: {len(train_indices)}, "
          f"Test samples: {len(test_indices)}")

    # ---- Step 3: two separate datasets with appropriate transforms ----
    train_dataset = FilteredImageFolder(
        root=dataset_root,
        selected_classes=selected_classes,
        transform=build_transforms(train=True),
    )
    test_dataset = FilteredImageFolder(
        root=dataset_root,
        selected_classes=selected_classes,
        transform=build_transforms(train=False),
    )

    train_subset = Subset(train_dataset, train_indices)
    test_subset  = Subset(test_dataset,  test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=feature_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(config.DEVICE == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=feature_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(config.DEVICE == "cuda"),
        drop_last=False,
    )

    return train_loader, test_loader, class_names


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_loader, test_loader, classes = get_dataloaders()
    print(f"\nClasses : {classes}")
    images, labels = next(iter(train_loader))
    print(f"Batch shape : {images.shape}")   # (B, 3, 224, 224)
    print(f"Label sample: {labels[:8]}")
