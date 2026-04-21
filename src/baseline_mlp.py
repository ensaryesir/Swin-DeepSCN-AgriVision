"""
baseline_mlp.py
===============
Module 4 – Baseline Multi-Layer Perceptron (MLP) Classifier
============================================================

This module implements a standard fully-connected neural network that
is trained with gradient-based optimisation (Adam + CrossEntropyLoss) on
the SAME Swin Transformer embeddings used by DeepSCN.

The purpose is to provide a scientifically fair baseline:
    - Same input features (frozen Swin embeddings)
    - Different training algorithm: backpropagation vs. pseudoinverse
    - We compare accuracy, precision, recall, F1-Score AND training time.

Expected result per the SCN literature:
    SCN/DeepSCN should train ~10–100× faster than the MLP while achieving
    competitive (or superior) classification accuracy, because:
        • No iterative epoch loop
        • No gradient computation through hidden layers
        • Training reduces to a single O(L³) matrix inversion
"""

import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import config


# ---------------------------------------------------------------------------
# MLP Architecture
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Standard fully-connected feed-forward network.

    Architecture
    ------------
        InputLayer(input_dim)
        [Linear → BatchNorm → ReLU → Dropout] × len(hidden_dims)
        Linear(hidden_dims[-1], n_classes)

    All hidden weights are initialised with Kaiming uniform (He initialisation)
    which is the PyTorch default for ReLU activations:

        W ~ Uniform[-√(6 / fan_in), √(6 / fan_in)]

    This initialisation ensures that the variance of activations is preserved
    through a ReLU network at initialisation (He et al., 2015).

    Parameters
    ----------
    input_dim   : dimensionality of the input feature vector
    hidden_dims : list of integers — neurons per hidden layer
    n_classes   : number of output classes
    dropout     : dropout probability applied after each hidden activation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = config.MLP_HIDDEN_DIMS,
        n_classes: int = 4,
        dropout: float = config.MLP_DROPOUT,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for out_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            in_dim = out_dim

        # Final classification layer
        layers.append(nn.Linear(in_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, input_dim) float32 tensor

        Returns
        -------
        logits : (B, n_classes) — raw, unnormalised class scores
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Trainer / Evaluator
# ---------------------------------------------------------------------------

class MLPTrainer:
    """
    Trains and evaluates an MLP classifier on pre-computed embeddings.

    Parameters
    ----------
    input_dim    : dimension of the input embedding
    n_classes    : number of output classes
    hidden_dims  : list of hidden-layer widths
    dropout      : dropout probability
    lr           : Adam learning rate
    epochs       : number of training epochs
    batch_size   : mini-batch size
    weight_decay : L2 regularisation coefficient for Adam
    device       : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: List[int]  = config.MLP_HIDDEN_DIMS,
        dropout: float          = config.MLP_DROPOUT,
        lr: float               = config.MLP_LR,
        epochs: int             = config.MLP_EPOCHS,
        batch_size: int         = config.MLP_BATCH_SIZE,
        weight_decay: float     = config.MLP_WEIGHT_DECAY,
        device: str             = config.DEVICE,
    ):
        self.epochs    = epochs
        self.device    = device
        self.batch_size = batch_size

        self.model = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            n_classes=n_classes,
            dropout=dropout,
        ).to(device)

        # Adam: adaptive moment estimation — adjusts per-parameter learning
        # rates using estimates of first and second moments of gradients.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # CrossEntropyLoss = LogSoftmax + NLLLoss
        # Numerically equivalent to: -Σ y_c * log(softmax(z)_c)
        self.criterion = nn.CrossEntropyLoss()

        # Learning rate cosine annealing — gradually reduces LR over epochs
        # which helps escape local minima and converge to a flatter minimum.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        self.train_time_: float = 0.0
        self.train_loss_history_: List[float] = []
        self.train_acc_history_: List[float] = []

    def _make_loader(self, X: np.ndarray, y: np.ndarray,
                     shuffle: bool) -> DataLoader:
        """Wrap numpy arrays in a TensorDataset and DataLoader."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        ds  = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(self.device == "cuda"))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "MLPTrainer":
        """
        Train the MLP using mini-batch stochastic gradient descent (Adam).

        Parameters
        ----------
        X_train : (N, d) float32 embedding matrix
        y_train : (N,)   integer label vector

        Returns
        -------
        self
        """
        loader = self._make_loader(X_train, y_train, shuffle=True)
        self.model.train()

        print(f"\n[MLP] Training for {self.epochs} epochs ...")
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            correct    = 0
            total      = 0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # --- Forward pass ---
                logits = self.model(X_batch)

                # CrossEntropyLoss:
                #   L = -Σ_i log( exp(z_{y_i}) / Σ_c exp(z_c) )
                loss = self.criterion(logits, y_batch)

                # --- Backward pass (compute ∇L w.r.t. all parameters) ---
                self.optimizer.zero_grad()
                loss.backward()

                # --- Parameter update (Adam step) ---
                self.optimizer.step()

                # Accumulate diagnostics
                epoch_loss += loss.item() * X_batch.size(0)
                preds       = logits.argmax(dim=1)
                correct    += (preds == y_batch).sum().item()
                total      += X_batch.size(0)

            self.scheduler.step()

            avg_loss = epoch_loss / total
            accuracy = correct / total
            self.train_loss_history_.append(avg_loss)
            self.train_acc_history_.append(accuracy)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch [{epoch:3d}/{self.epochs}] "
                      f"Loss: {avg_loss:.4f}  "
                      f"Acc: {accuracy:.4f}  "
                      f"LR: {self.scheduler.get_last_lr()[0]:.6f}")

        self.train_time_ = time.time() - t0
        print(f"[MLP] Training complete | Time = {self.train_time_:.4f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return class predictions for embedding matrix X.

        Parameters
        ----------
        X : (N, d) float32

        Returns
        -------
        y_pred : (N,) int64
        """
        self.model.eval()
        loader = self._make_loader(X, np.zeros(len(X), dtype=np.int64),
                                   shuffle=False)
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                logits  = self.model(X_batch)
                preds   = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
        return np.concatenate(all_preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return softmax probabilities.

        Parameters
        ----------
        X : (N, d)

        Returns
        -------
        proba : (N, C) float32
        """
        self.model.eval()
        loader = self._make_loader(X, np.zeros(len(X), dtype=np.int64),
                                   shuffle=False)
        all_proba = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                logits  = self.model(X_batch)
                proba   = torch.softmax(logits, dim=1).cpu().numpy()
                all_proba.append(proba)
        return np.concatenate(all_proba)


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    X, y = make_classification(
        n_samples=600, n_features=128, n_classes=4,
        n_informative=30, random_state=42
    )
    X = X.astype(np.float32)

    trainer = MLPTrainer(input_dim=128, n_classes=4, epochs=15)
    trainer.fit(X, y)
    preds = trainer.predict(X)
    print(f"Train accuracy: {accuracy_score(y, preds):.4f}")
    print(f"Train time    : {trainer.train_time_:.4f}s")
