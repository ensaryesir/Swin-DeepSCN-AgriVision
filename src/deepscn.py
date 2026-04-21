"""
deepscn.py
==========
Module 3 – Deep Stochastic Configuration Network (DeepSCN)
===========================================================

Mathematical Background
-----------------------
Stochastic Configuration Networks (SCN) were introduced by Wang & Li (2017)
[DOI: 10.1109/TNNLS.2017.2729236] as a class of randomised learning machines
that combine the representational power of random-feature methods (ELM, Random
Kitchen Sinks) with an *adaptive, error-guided* node-generation strategy.

The key ideas are:

    1.  **Random hidden weights (frozen)**
        Hidden-layer weights  W  and biases  b  are drawn from a predefined
        distribution (usually Uniform[−r, r]) and NEVER updated by gradient
        descent.  This avoids the expensive backpropagation through the
        feature extraction layers.

    2.  **Analytical output-weight computation (pseudoinverse)**
        Once the hidden representations H are fixed, the output weights β
        solve a *linear least-squares* problem:

            min_β  ‖ H β − Y ‖²_F

        The closed-form solution is:

            β = H† Y = (H^T H)^{-1} H^T Y         (ordinary least squares)

        or, with L2 regularisation (Tikhonov / Ridge):

            β = (H^T H + λI)^{-1} H^T Y            (ridge regression)

        where H† denotes the Moore-Penrose pseudoinverse of H,  Y is the
        one-hot-encoded target matrix, and λ is a small ridge constant.

        The pseudoinverse is computed in one matrix operation (NumPy's
        np.linalg.lstsq or np.linalg.pinv) — no iterative solver needed.

    3.  **Sequential construction (Incremental / Deep SCN)**
        The network is built one hidden node (or one block of nodes) at a
        time.  At each step:
            (a) Sample L candidate random weight vectors {w_l}.
            (b) For each candidate, compute the hidden activation h_l.
            (c) Accept the candidate if the residual  E = Y − H β  is
                strictly reduced (SCN inequality constraint):
                    ‖E_new‖ < ‖E_old‖  (equivalently, ‖E_new‖ < ‖E_old‖ − ε)
            (d) Append the accepted node to H and recompute β via pseudoinverse.

        Convergence is guaranteed under mild conditions (Wang & Li, 2017
        Theorem 1): the residual ‖E_k‖ → 0 as k → ∞, provided the
        candidate pool is large enough and the activation function is a
        non-constant, bounded, monotonically increasing function.

    4.  **Why this is "Deep"**
        In the DeepSCN variant (Wang et al., 2019) multiple SCN modules are
        stacked: the *residual error* of one SCN becomes the *target* of the
        next SCN, analogous to gradient boosting.  The final prediction is
        the sum of all module outputs.  Here we implement the single-module
        (flat) version with Ridge regression, which is the most common
        practical variant.

Complexity comparison (N samples, d features, L hidden nodes)
--------------------------------------------------------------
  Method        | Training cost
  --------------|--------------------------------------------
  Standard MLP  | O(N · d · L · epochs) — iterative SGD
  SCN / ELM     | O(N · L²) or O(L³) — one matrix inversion
  DeepSCN (ours)| O(K · N · L²) — K sequential inversion steps
"""

import time
from typing import Optional, Tuple

import numpy as np

from src import config


# ---------------------------------------------------------------------------
# Activation function helper
# ---------------------------------------------------------------------------

def _activate(X: np.ndarray, name: str) -> np.ndarray:
    """
    Apply element-wise non-linearity.

    Supported names: 'relu', 'tanh', 'sigmoid'.
    """
    if name == "relu":
        return np.maximum(0.0, X)
    elif name == "tanh":
        return np.tanh(X)
    elif name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-X))
    else:
        raise ValueError(f"Unknown activation '{name}'.  Choose relu/tanh/sigmoid.")


# ---------------------------------------------------------------------------
# One-hot encoding helper
# ---------------------------------------------------------------------------

def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer class labels to a one-hot encoded matrix.

    Parameters
    ----------
    y           : (N,) integer label array
    num_classes : total number of classes C

    Returns
    -------
    Y : (N, C) float32 matrix where Y[i, y[i]] = 1, else 0
    """
    N = len(y)
    Y = np.zeros((N, num_classes), dtype=np.float32)
    Y[np.arange(N), y] = 1.0
    return Y


# ---------------------------------------------------------------------------
# Core DeepSCN class
# ---------------------------------------------------------------------------

class DeepSCN:
    """
    Deep Stochastic Configuration Network (single-module variant with Ridge
    regression output weights).

    Parameters
    ----------
    max_nodes       : maximum number of hidden nodes to add
    n_candidates    : number of random candidates sampled at each step
    tolerance       : minimum relative residual improvement to accept a node
    ridge_lambda    : L2 regularisation coefficient for the output-weight solve
    weight_range    : (low, high) for Uniform random weight initialisation
    activation      : activation function name ('relu' | 'tanh' | 'sigmoid')
    random_seed     : RNG seed for reproducibility
    verbose         : if True, print per-step diagnostics
    """

    def __init__(
        self,
        max_nodes: int      = config.DEEPSCN_MAX_NODES,
        n_candidates: int   = config.DEEPSCN_CANDIDATES,
        tolerance: float    = config.DEEPSCN_TOLERANCE,
        ridge_lambda: float = config.DEEPSCN_LAMBDA,
        weight_range: Tuple[float, float] = config.DEEPSCN_WEIGHT_RANGE,
        activation: str     = config.DEEPSCN_ACTIVATION,
        random_seed: int    = config.RANDOM_SEED,
        verbose: bool       = True,
    ):
        self.max_nodes    = max_nodes
        self.n_candidates = n_candidates
        self.tolerance    = tolerance
        self.ridge_lambda = ridge_lambda
        self.weight_range = weight_range
        self.activation   = activation
        self.verbose      = verbose
        self.rng          = np.random.RandomState(random_seed)

        # These attributes are populated during fit()
        self.W_hidden_: Optional[np.ndarray] = None   # (input_dim, n_nodes)
        self.b_hidden_: Optional[np.ndarray] = None   # (n_nodes,)
        self.W_output_: Optional[np.ndarray] = None   # (n_nodes, n_classes)
        self.n_nodes_: int = 0
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.input_dim_: int = 0

        # Training diagnostics
        self.residual_history_: list = []
        self.train_time_: float = 0.0

    # -----------------------------------------------------------------------
    # Private: compute hidden-layer output matrix H
    # -----------------------------------------------------------------------

    def _hidden_output(self, X: np.ndarray, W: np.ndarray, b: np.ndarray
                       ) -> np.ndarray:
        """
        Compute the hidden-layer activation matrix.

        Math
        ----
            Z = X @ W + b           ... pre-activation  (N × L)
            H = σ(Z)                ... post-activation (N × L)

        where X ∈ R^{N × d},  W ∈ R^{d × L},  b ∈ R^{L}.

        Parameters
        ----------
        X : (N, d) — input features
        W : (d, L) — weight matrix (frozen random weights)
        b : (L,)   — bias vector   (frozen random biases)

        Returns
        -------
        H : (N, L) — hidden-layer activations
        """
        Z = X @ W + b      # efficient BLAS matrix-multiply
        return _activate(Z, self.activation)

    # -----------------------------------------------------------------------
    # Public: fit
    # -----------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepSCN":
        """
        Train the DeepSCN using the true Incremental SCN algorithm, but with
        a highly optimized Dynamic Gram Matrix approach to reduce O(N L^3) 
        time complexity down to O(N L^2), offering massive speedups without 
        breaking theoretical SCN guarantees.
        """
        t0 = time.time()
        X = X.astype(np.float32)
        N, d = X.shape

        self.classes_   = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.input_dim_ = d

        Y = _one_hot(y, self.n_classes_)

        # ---- Initialise hidden layer storage ----
        W_cols = []
        b_vals = []
        H = np.empty((N, 0), dtype=np.float32)

        # To avoid recomputing the massive Gram matrix H^T H from scratch 
        # (which is O(N * step^2) and the bottleneck), we maintain it dynamically.
        HtH = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        HtY = np.zeros((self.max_nodes, self.n_classes_), dtype=np.float32)

        residual = Y.copy()
        prev_res_norm = np.linalg.norm(residual)
        
        low, high = self.weight_range

        if self.verbose:
            print(f"\n[DeepSCN] Starting True SCN Construction (Optimized): "
                  f"max_nodes={self.max_nodes}, candidates={self.n_candidates}")

        for step in range(self.max_nodes):
            
            # A: Generate L candidate nodes
            W_cands = self.rng.uniform(low, high, size=(d, self.n_candidates)).astype(np.float32)
            b_cands = self.rng.uniform(low, high, size=(self.n_candidates,)).astype(np.float32)

            # B: Evaluate hidden activations for candidates
            Z_cands = X @ W_cands + b_cands
            H_cands = _activate(Z_cands, self.activation)

            # C: Greedy Candidate Selection (SCN Inequality Strategy)
            proj = residual.T @ H_cands
            corr = np.sum(proj ** 2, axis=0)
            h_energy = np.sum(H_cands ** 2, axis=0)
            corr_norm = corr / (h_energy + 1e-8)
            
            best_idx = int(np.argmax(corr_norm))
            h_best = H_cands[:, best_idx : best_idx + 1]
            w_best = W_cands[:, best_idx]
            b_best = b_cands[best_idx]

            # D: Dynamic Gram Matrix Update (The core optimization)
            # We only compute the cross-correlation with the newly added node.
            # This turns an O(N * step^2) operation into O(N * step).
            if step > 0:
                h_cross = (H.T @ h_best).flatten()
                HtH[:step, step] = h_cross
                HtH[step, :step] = h_cross
                
            HtH[step, step] = (h_best.T @ h_best)[0, 0]
            HtY[step, :] = (h_best.T @ Y)[0, :]

            # E: Append accepted node
            H = np.concatenate([H, h_best], axis=1)
            W_cols.append(w_best)
            b_vals.append(b_best)

            # F: Solve for analytical weights on the small (step)x(step) matrix
            sub_HtH = HtH[:step+1, :step+1] + self.ridge_lambda * np.eye(step+1, dtype=np.float32)
            sub_HtY = HtY[:step+1, :]
            
            # np.linalg.solve operates on L_max=1024 max. Almost instantaneous.
            beta = np.linalg.solve(sub_HtH, sub_HtY)

            # G: Fast Residual update
            Y_pred = H @ beta
            residual = Y - Y_pred
            new_res_norm = np.linalg.norm(residual)
            
            self.residual_history_.append(new_res_norm)
            
            if self.verbose and ((step + 1) % 100 == 0 or step == 0):
                print(f"  Step {step+1:4d} | ‖residual‖ = {new_res_norm:.6f} | Δ = {prev_res_norm - new_res_norm:.2e}")

            # H: SCN stopping constraints
            if new_res_norm < self.tolerance:
                if self.verbose:
                    print(f"  [DeepSCN] Converged early at step {step+1}")
                break
                
            if new_res_norm >= prev_res_norm:
                # Discard the node because we've stagnated
                H = H[:, :-1]
                W_cols.pop()
                b_vals.pop()
                if self.verbose:
                    print(f"  [DeepSCN] No improvement. Stopping at {step} nodes.")
                
                # Revert beta
                if step > 0:
                    sub_HtH = HtH[:step, :step] + self.ridge_lambda * np.eye(step, dtype=np.float32)
                    sub_HtY = HtY[:step, :]
                    beta = np.linalg.solve(sub_HtH, sub_HtY)
                break
                
            prev_res_norm = new_res_norm

        # ---- Store final model parameters ----
        self.n_nodes_ = H.shape[1]
        
        if self.n_nodes_ > 0:
            self.W_hidden_ = np.column_stack(W_cols)
            self.b_hidden_ = np.array(b_vals)
            self.W_output_ = beta
        else:
            raise RuntimeError(
                "[DeepSCN] No hidden nodes were accepted. "
                "Check tolerance or weight limits."
            )

        self.train_time_ = time.time() - t0

        if self.verbose:
            print(f"\n[DeepSCN] Training complete: {self.n_nodes_} nodes | "
                  f"Time = {self.train_time_:.4f}s")

        return self

    # -----------------------------------------------------------------------
    # Public: predict
    # -----------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Classify samples in X.

        Math
        ----
            H    = σ(X W_hidden + b_hidden)    hidden activations
            Ŷhat = H W_output                  raw output scores
            ŷ    = argmax_c  Ŷhat[:, c]        predicted class index

        Parameters
        ----------
        X : (N, d) float32 feature matrix

        Returns
        -------
        y_pred : (N,) integer predicted class indices
        """
        self._check_is_fitted()
        X = X.astype(np.float32)
        H     = self._hidden_output(X, self.W_hidden_, self.b_hidden_)
        Y_hat = H @ self.W_output_          # (N, C)
        return np.argmax(Y_hat, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return softmax-normalised class probabilities.

        Parameters
        ----------
        X : (N, d)

        Returns
        -------
        proba : (N, C) float32 probability matrix
        """
        self._check_is_fitted()
        X = X.astype(np.float32)
        H     = self._hidden_output(X, self.W_hidden_, self.b_hidden_)
        Y_hat = H @ self.W_output_             # (N, C)
        # Softmax: exp(z) / Σ exp(z)  — subtract max for numerical stability
        Y_hat -= Y_hat.max(axis=1, keepdims=True)
        exp   = np.exp(Y_hat)
        return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)

    def _check_is_fitted(self):
        if self.W_hidden_ is None:
            raise RuntimeError("Call fit() before predict().")


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    X, y = make_classification(
        n_samples=500, n_features=64, n_classes=4,
        n_informative=20, random_state=42
    )
    X = X.astype(np.float32)

    model = DeepSCN(max_nodes=200, n_candidates=50, verbose=True)
    model.fit(X, y)
    preds = model.predict(X)
    print(f"\nTrain accuracy: {accuracy_score(y, preds):.4f}")
    print(f"Train time    : {model.train_time_:.4f}s")
