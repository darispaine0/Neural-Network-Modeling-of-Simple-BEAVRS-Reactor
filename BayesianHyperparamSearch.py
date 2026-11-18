#!/usr/bin/env python3
"""
ds3_hparam_search_h5_gamma.py

Adaptive Bayesian hyperparameter search over 6 parameters:

  1. hidden_size      — number of neurons in the hidden layer
  2. learning_rate    — AdamW learning rate
  3. weight_decay     — AdamW weight decay (L2 regularisation)
  4. alpha            — FM MSE loss weight
  5. beta             — k-eff PCM² loss weight
  6. gamma            — fission-source vector loss weight   (NEW)

This is a merge of ds3_hparam_search.py (Bayesian GP search over the loss
weights) and ds3_sweep_fm_keff_loss_h5.py (HDF5-backed data loading, the
GAMMA fission-source term, and differentiable power iteration). Two things
changed relative to the original hparam search:

  * The search space now includes `gamma`, the weight on the fission-source
    MSE term (matched against the dominant eigenvector of the predicted FM,
    read straight from training_data.h5).
  * Gradient tracking through the power iteration is turned ON during
    training (DIFFERENTIABLE_POWER_ITER = True): k-eff and the fission
    source are computed with `power_iter_torch` on live tensors, so the BETA
    and GAMMA terms now contribute a real gradient back through the FM
    prediction, rather than being re-attached as constants after a numpy
    power iteration on a detached copy. Validation still uses the fast
    detached surrogate, since no gradient is needed there.

Search strategy
---------------
Uses scikit-optimize's Gaussian Process (GP) surrogate with Expected
Improvement (EI) acquisition. Each call to the objective function trains one
model to convergence and returns the mean |PCM| on the validation split —
the GP minimises mean |PCM| directly, not the composite training loss.

After every trial that sets a new best |PCM|, the model checkpoint, PCM
arrays, and summary JSON are written to disk immediately, so a crash never
loses the best result found so far.

Quick-start
-----------
  python ds3_hparam_search_h5_gamma.py              # 600 trials (default)
  python ds3_hparam_search_h5_gamma.py --trials 60  # faster exploratory run

Dependencies
------------
  pip install scikit-optimize h5py
  (everything else is already in your environment)
"""
import argparse
import json
import csv
import traceback
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# scikit-optimize — install once with: pip install scikit-optimize
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
except ImportError:
    raise ImportError(
        "scikit-optimize is required.  Install with:  pip install scikit-optimize"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Fixed config  (data paths, grid geometry, training budget)
# ─────────────────────────────────────────────────────────────────────────────

class GlobalConfig:
    # Data paths (relative to the repository folder; written by build_h5.py)
    H5_PATH   = Path("training_data.h5")
    MASK_PATH = Path("training_data_nonzero_mask.npy")

    # Grid geometry
    N_ROWS   = 17
    N_COLS   = 17
    N_CELLS  = N_ROWS * N_COLS   # 289
    N_CELLS2 = 264
    N_FUEL_COMPACT = N_CELLS2
    INPUT_SIZE     = N_FUEL_COMPACT + N_CELLS  # 553
    OUTPUT_SIZE    = N_CELLS2 * N_CELLS2        # 69 696

    # Training budget per trial (kept modest so many trials are affordable)
    BATCH_SIZE          = 32
    MAX_EPOCHS           = 500
    VALIDATION_SPLIT    = 0.2
    EARLY_STOP_PATIENCE = 5
    LR_REDUCE_PATIENCE  = 10

    # Power iteration mode — gradient tracking is ON: k-eff and the
    # fission-source vector are computed with the differentiable PyTorch
    # power iteration during training, so BETA/GAMMA carry a real gradient
    # back through the FM prediction. Validation always uses the fast
    # detached numpy surrogate (no gradient needed there).
    DIFFERENTIABLE_POWER_ITER = True
    POWER_ITER_STEPS          = 20

    # Reproducibility base seed (trial i uses BASE_SEED + i)
    BASE_SEED = 44906

    # Output
    RESULTS_ROOT     = Path("ds3_hparam_search_results")
    SAVE_CHECKPOINTS = True   # save checkpoint whenever a new best PCM is found

    EPSILON = 1e-8

    # k-eff target: OpenMC track-length k-eff (written by build_h5.py).
    # Set to "keff" to use the FM power-iteration eigenvalue instead.
    KEFF_DATASET = "keff_tracklength"

cfg = GlobalConfig()

# ─────────────────────────────────────────────────────────────────────────────
# Search space
# ─────────────────────────────────────────────────────────────────────────────
SEARCH_SPACE = [
    Integer(500,  10000,  name="hidden_size"),
    Real   (1e-8, 1e-4,  prior="log-uniform", name="learning_rate"),
    Real   (1e-9, 1e-5,  prior="log-uniform", name="weight_decay"),
    Real   (1e-9, 100,    prior="log-uniform", name="alpha"),
    Real   (1e-9,  1e5,   prior="log-uniform", name="beta"),
    Real   (1e-9, 1e4,   prior="log-uniform", name="gamma"),   # NEW: fission-source term weight
]

# ─────────────────────────────────────────────────────────────────────────────
# Data utilities (HDF5-backed)
# ─────────────────────────────────────────────────────────────────────────────

def load_fm_library_from_h5(h5file):
    """
    Build the {temp_K: (N_CELLS2, N_CELLS2)} reference FM library from the
    'Reference Matrices' / 'reference_temperatures' datasets.
    """
    ref_mats  = h5file["Reference Matrices"][:]        # (13, 69696)
    ref_temps = h5file["reference_temperatures"][:]    # (13,)

    fm_lib = {}
    for mat_flat, t in zip(ref_mats, ref_temps):
        fm = np.asarray(mat_flat, dtype=float).reshape(cfg.N_CELLS2, cfg.N_CELLS2)
        fm_lib[int(t)] = fm

    print(f"[INFO] FM library (from h5): {sorted(fm_lib.keys())} K")
    return fm_lib


def interpolate_rowwise_reference(fuel_temps_compact, fm_lib, mask):
    keys = sorted(fm_lib.keys())
    fuel_T   = np.asarray(fuel_temps_compact, dtype=float)
    baseline = np.zeros((cfg.N_CELLS2, cfg.N_CELLS2), dtype=float)

    if fuel_T.size != cfg.N_CELLS2:
        raise ValueError(f"fuel_temps_compact length {fuel_T.size} != {cfg.N_CELLS2}")

    for i in range(cfg.N_CELLS2):
        Ti = fuel_T[i]
        if Ti <= keys[0]:
            baseline[i] = fm_lib[keys[0]][i]
        elif Ti >= keys[-1]:
            baseline[i] = fm_lib[keys[-1]][i]
        else:
            for lo, hi in zip(keys[:-1], keys[1:]):
                if lo <= Ti <= hi:
                    a = (Ti - lo) / (hi - lo) if hi != lo else 0.0
                    baseline[i] = (1 - a) * fm_lib[lo][i] + a * fm_lib[hi][i]
                    break
    return baseline.flatten()


def build_ref_fm_cache(fuel_temps_all, fm_lib, mask):
    M   = fuel_temps_all.shape[0]
    out = np.zeros((M, cfg.N_CELLS2, cfg.N_CELLS2), dtype=np.float32)
    for i in range(M):
        out[i] = interpolate_rowwise_reference(fuel_temps_all[i], fm_lib, mask).reshape(
            cfg.N_CELLS2, cfg.N_CELLS2
        )
    return out


def prepare_dataset():
    """
    Loads temperatures, fission matrices, k-eff, and the fission-source
    vector straight from training_data.h5. The 'temperatures' dataset
    already stores the concatenated feature vector (264 compact fuel temps
    + 289 other temps), so the first 264 columns double as the compact fuel
    temps used for reference interpolation.
    """
    print("Loading data from training_data.h5...")
    with h5py.File(cfg.H5_PATH, "r") as h5:
        X_temps    = h5["temperatures"][:]     # (N, 553)
        fm_samples = h5["fission_matrix"][:]   # (N, 69696)
        if cfg.KEFF_DATASET not in h5:
            raise KeyError(
                f"'{cfg.KEFF_DATASET}' not found in {cfg.H5_PATH}. "
                f"Rebuild the file with build_h5.py, or set KEFF_DATASET = \"keff\" "
                f"to use the FM power-iteration eigenvalue."
            )
        keff_raw   = h5[cfg.KEFF_DATASET][:]   # (N,) track-length k-eff
        source_raw = h5["fission_source"][:]   # (N, 264) — compact fuel-cell order
        fm_lib     = load_fm_library_from_h5(h5)

    mask = np.load(cfg.MASK_PATH)

    n_pairs = min(X_temps.shape[0], fm_samples.shape[0], keff_raw.shape[0], source_raw.shape[0])
    print(f"Loaded {n_pairs} samples | mask {mask.shape}")

    features, fuel_temps_list, keff_list, source_list, good_indices, skipped = [], [], [], [], [], []

    for idx in range(n_pairs):
        k = keff_raw[idx]
        if not np.isfinite(k):
            skipped.append((idx, "invalid_keff"))
            continue

        feat = np.asarray(X_temps[idx], dtype=np.float32)
        if feat.shape[0] != cfg.INPUT_SIZE:
            skipped.append((idx, f"feat_shape:{feat.shape}"))
            continue
        fuel_compact = feat[:cfg.N_FUEL_COMPACT]

        fm_arr = np.asarray(fm_samples[idx], dtype=float)
        if fm_arr.size != cfg.OUTPUT_SIZE:
            skipped.append((idx, f"fm_shape:{fm_arr.shape}"))
            continue

        src_arr = np.asarray(source_raw[idx], dtype=np.float32)
        if src_arr.shape[0] != cfg.N_CELLS2:
            skipped.append((idx, f"source_shape:{src_arr.shape}"))
            continue

        features.append(feat)
        fuel_temps_list.append(fuel_compact)
        keff_list.append(float(k))
        source_list.append(src_arr)
        good_indices.append(idx)

    if skipped:
        print(f"[WARN] Skipped {len(skipped)} samples; first 10: {skipped[:10]}")
    if not features:
        raise RuntimeError("No valid samples found.")

    M              = len(features)
    features       = np.stack(features,        axis=0)
    fuel_temps_all = np.stack(fuel_temps_list, axis=0)
    Y_keff         = np.array(keff_list, dtype=np.float32)
    Y_source       = np.stack(source_list,      axis=0)   # (M, N_CELLS2)

    g_min    = float(np.min(features))
    g_max    = float(np.max(features))
    X_inputs = (features - g_min) / (g_max - g_min + 1e-12)

    print("Building ln-ratio FM targets (fuel-temp reference)...")
    Y_fm = np.zeros((M, cfg.OUTPUT_SIZE), dtype=np.float32)
    for i in range(M):
        fm_arr = np.asarray(fm_samples[good_indices[i]], dtype=float).reshape(
            cfg.N_CELLS2, cfg.N_CELLS2
        )
        fm_ref_flat = interpolate_rowwise_reference(fuel_temps_all[i], fm_lib, mask)
        fm_ref      = fm_ref_flat.reshape(cfg.N_CELLS2, cfg.N_CELLS2) + cfg.EPSILON
        ratio       = np.clip((fm_arr + cfg.EPSILON) / fm_ref, cfg.EPSILON, None)
        Y_fm[i, :]  = np.log(ratio).flatten().astype(np.float32)

    print(f"Dataset ready: X {X_inputs.shape}  Y_fm {Y_fm.shape}  "
          f"Y_source {Y_source.shape}  Y_keff {Y_keff.shape}")
    return X_inputs, Y_fm, Y_source, Y_keff, good_indices, fuel_temps_all, fm_lib, mask

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class FMCoefficientPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    def forward(self, x):
        return self.network(x)

# ─────────────────────────────────────────────────────────────────────────────
# Power iteration — numpy surrogate (validation) and differentiable (training)
# ─────────────────────────────────────────────────────────────────────────────

def power_iter_numpy(fm_np, tol=1e-7, max_iter=1000):
    n = fm_np.shape[0]
    v = np.ones((n, 1), dtype=float)
    eig = 1.0
    for i in range(max_iter):
        v_new   = fm_np.dot(v)
        eig_new = float(np.max(np.abs(v_new)))
        if eig_new == 0: break
        v = v_new / eig_new
        if i > 0 and abs(eig_new - eig) < tol:
            eig = eig_new; break
        eig = eig_new
    v = v.ravel(); s = v.sum()
    if s != 0: v /= s
    return float(eig), v


def power_iter_torch(fm_tensor, n_steps):
    """
    Differentiable power iteration in PyTorch. fm_tensor : (N, N), must stay
    in the compute graph. Returns (dominant eigenvalue, normalized
    fission-source vector), both differentiable tensors.
    """
    n = fm_tensor.shape[0]
    v = torch.ones(n, 1, dtype=fm_tensor.dtype, device=fm_tensor.device)
    eig = torch.tensor(1.0, dtype=fm_tensor.dtype, device=fm_tensor.device)
    for _ in range(n_steps):
        v_new   = fm_tensor.mm(v)
        eig_new = v_new.abs().max()
        if eig_new == 0:
            break
        v   = v_new / (eig_new + 1e-12)
        eig = eig_new
    v_norm = (v / (v.sum() + 1e-12)).reshape(-1)
    return eig, v_norm

# ─────────────────────────────────────────────────────────────────────────────
# Loss  (used during training and validation)
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_full_matrix_torch(cleaned_flat_tensor, mask_tensor):
    full = torch.zeros(cfg.N_CELLS, cfg.N_CELLS,
                       dtype=cleaned_flat_tensor.dtype,
                       device=cleaned_flat_tensor.device)
    full[mask_tensor] = cleaned_flat_tensor
    return full


def combined_loss(fm_pred, fm_true, keff_true_batch,
                  fm_ref_batch,        # (B, N_CELLS2, N_CELLS2) numpy, pre-built
                  source_true_batch,   # (B, N_CELLS2) numpy, true fission-source (compact order)
                  mask_tensor,
                  row_active_tensor,   # (N_CELLS,) bool tensor — which full-grid rows are fuel cells
                  alpha, beta, gamma,
                  differentiable=False,
                  power_iter_steps=20):
    """
    L = alpha * MSE(fm_pred, fm_true)
      + beta  * mean( ((k_pred - k_true) / k_true * 1e5)² ) / 1e10
      + gamma * ||v_pred - v_true||                      # fission source vector

    differentiable=True  → power iter runs in PyTorch on live tensors; the
                            BETA and GAMMA terms carry a real gradient back
                            through fm_pred (gradient tracking ON).
    differentiable=False → surrogate: power iter in numpy on a detached
                            copy of fm_pred; k_pred/v_pred are re-attached
                            as constants (used for validation only).
    """
    fm_mse_t = nn.functional.mse_loss(fm_pred, fm_true)

    if beta == 0.0 and gamma == 0.0:
        return alpha * fm_mse_t, float(fm_mse_t.detach()), 0.0, 0.0

    B      = fm_pred.shape[0]
    device = fm_pred.device

    if differentiable:
        fm_ref_t  = torch.from_numpy(fm_ref_batch).float().to(device)  # (B, N2, N2)
        ratio_mat = torch.exp(fm_pred).reshape(B, cfg.N_CELLS2, cfg.N_CELLS2)
        fm_clean  = ratio_mat * fm_ref_t                                 # (B, N2, N2)

        k_preds, v_preds = [], []
        for b in range(B):
            fm_full = reconstruct_full_matrix_torch(fm_clean[b].reshape(-1), mask_tensor)
            k_b, v_b_full = power_iter_torch(fm_full, power_iter_steps)
            v_b_compact = v_b_full[row_active_tensor]
            k_preds.append(k_b)
            v_preds.append(v_b_compact)
        k_pred_t = torch.stack(k_preds)
        v_pred_t = torch.stack(v_preds)
    else:
        with torch.no_grad():
            fm_det = fm_pred.detach().cpu().numpy()
        k_np = np.zeros(B, dtype=np.float32)
        v_np = np.zeros((B, cfg.N_CELLS2), dtype=np.float32)
        mask_np_local       = mask_tensor.cpu().numpy()
        row_active_np_local = row_active_tensor.cpu().numpy()
        for b in range(B):
            ratio_mat   = np.exp(fm_det[b]).reshape(cfg.N_CELLS2, cfg.N_CELLS2)
            fm_clean_np = ratio_mat * fm_ref_batch[b]
            full_np     = np.zeros((cfg.N_CELLS, cfg.N_CELLS), dtype=float)
            full_np[mask_np_local] = fm_clean_np.flatten()
            k_np[b], v_b_full = power_iter_numpy(full_np)
            v_np[b] = v_b_full[row_active_np_local]
        k_pred_t = torch.from_numpy(k_np).to(device)
        v_pred_t = torch.from_numpy(v_np).to(device)

    pcm      = (k_pred_t - keff_true_batch) / (keff_true_batch + 1e-8) * 1e5
    keff_l_t = (pcm ** 2).mean() / 1e10

    source_true_t = torch.from_numpy(source_true_batch).float().to(device)
    source_l_t    = torch.linalg.vector_norm(v_pred_t - source_true_t)

    total = alpha * fm_mse_t + beta * keff_l_t + gamma * source_l_t
    return total, float(fm_mse_t.detach()), float(keff_l_t.detach()), float(source_l_t.detach())

# ─────────────────────────────────────────────────────────────────────────────
# PCM evaluation  (runs on validation split during search, full set at end)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pcm_compact(model, mu, sig, features, keff_values,
                         fuel_temps, fm_lib, mask, batch_size=32):
    """
    Compute mean |PCM| for a subset of samples that are already row-aligned
    (features[i], keff_values[i], fuel_temps[i] all refer to the same
    sample — no separate good_indices indirection needed).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    Xn = (features - mu) / sig
    preds = []
    with torch.no_grad():
        for s in range(0, Xn.shape[0], batch_size):
            e   = min(s + batch_size, Xn.shape[0])
            out = model(torch.from_numpy(Xn[s:e].astype(np.float32)).to(device))
            preds.append(out.cpu().numpy())
    preds = np.vstack(preds)

    M        = preds.shape[0]
    pcm_list = np.full(M, np.nan)
    kfm_list = np.full(M, np.nan)

    for i in range(M):
        try:
            ratio_mat     = np.exp(preds[i]).reshape(cfg.N_CELLS2, cfg.N_CELLS2)
            fm_ref_flat   = interpolate_rowwise_reference(fuel_temps[i], fm_lib, mask)
            fm_pred_clean = ratio_mat * fm_ref_flat.reshape(cfg.N_CELLS2, cfg.N_CELLS2)
            fm_pred_full  = np.zeros((cfg.N_CELLS, cfg.N_CELLS), dtype=float)
            fm_pred_full[mask] = fm_pred_clean.flatten()
            k_pred, _     = power_iter_numpy(fm_pred_full)
            kfm_list[i]   = k_pred
            k_true = float(keff_values[i])
            if np.isfinite(k_pred) and np.isfinite(k_true) and k_true != 0:
                pcm_list[i] = (k_pred / k_true - 1.0) * 1e5
        except Exception as e:
            print(f"[WARN] PCM sample {i}: {e}")

    fin = pcm_list[np.isfinite(pcm_list)]
    mean_abs_pcm = float(np.mean(np.abs(fin))) if fin.size > 0 else float("nan")
    return pcm_list, kfm_list, mean_abs_pcm

# ─────────────────────────────────────────────────────────────────────────────
# Training (single trial)
# ─────────────────────────────────────────────────────────────────────────────

def train_trial(X, Y_fm, Y_source, Y_keff, ref_fm_cache, mask_np,
                hidden_size, learning_rate, weight_decay, alpha, beta, gamma,
                seed, verbose=False):
    """
    Train one model with the given hyperparameters. Gradient tracking
    through the power iteration is ON for the training loop
    (differentiable=cfg.DIFFERENTIABLE_POWER_ITER); validation always uses
    the fast detached surrogate.

    Returns
    -------
    best_val_loss : float   (composite training loss — for scheduler / early-stop)
    best_ep       : int
    model         : trained FMCoefficientPredictor
    mu, sig       : normalisation statistics (from training split)
    va_idx        : validation indices into X  (needed for PCM evaluation)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    M     = X.shape[0]
    n_val = int(M * cfg.VALIDATION_SPLIT)
    perm  = np.random.permutation(M)
    tr_idx, va_idx = perm[:M - n_val], perm[M - n_val:]

    tr_X,  tr_fm,  tr_keff, tr_ref, tr_src = (
        X[tr_idx], Y_fm[tr_idx], Y_keff[tr_idx], ref_fm_cache[tr_idx], Y_source[tr_idx]
    )
    va_X,  va_fm,  va_keff, va_ref, va_src = (
        X[va_idx], Y_fm[va_idx], Y_keff[va_idx], ref_fm_cache[va_idx], Y_source[va_idx]
    )

    mu  = tr_X.mean(axis=0)
    sig = tr_X.std(axis=0) + 1e-8
    tr_Xn = (tr_X - mu) / sig
    va_Xn = (va_X - mu) / sig

    def make_loader(Xn, fm, keff, shuffle):
        ds = TensorDataset(
            torch.from_numpy(Xn.astype(np.float32)),
            torch.from_numpy(fm.astype(np.float32)),
            torch.from_numpy(keff.astype(np.float32)),
            torch.arange(len(Xn)),
        )
        return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle)

    tr_loader = make_loader(tr_Xn, tr_fm, tr_keff, shuffle=True)
    va_loader = make_loader(va_Xn, va_fm, va_keff, shuffle=False)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_tensor = torch.from_numpy(mask_np).to(device)
    row_active_tensor = torch.from_numpy(mask_np.any(axis=1)).to(device)

    model  = FMCoefficientPredictor(cfg.INPUT_SIZE, hidden_size, cfg.OUTPUT_SIZE).to(device)
    opt    = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sched  = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=cfg.LR_REDUCE_PATIENCE
    )

    best_val, best_ep, no_improve, best_state = float('inf'), -1, 0, None

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        # ── train (gradient tracking ON through the power iteration) ─────────
        model.train()
        tr_tot = tr_fm_l = tr_kl = tr_sl = 0.0; nb = 0
        for bx, bfm, bk, bidx in tr_loader:
            bx, bfm, bk = bx.to(device), bfm.to(device), bk.to(device)
            b_ref_np = tr_ref[bidx.numpy()]
            b_src_np = tr_src[bidx.numpy()]
            opt.zero_grad()
            fm_p = model(bx)
            loss, fl, kl, sl = combined_loss(
                fm_p, bfm, bk, b_ref_np, b_src_np, mask_tensor, row_active_tensor,
                alpha, beta, gamma,
                differentiable=cfg.DIFFERENTIABLE_POWER_ITER,
                power_iter_steps=cfg.POWER_ITER_STEPS,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_tot += loss.item(); tr_fm_l += fl; tr_kl += kl; tr_sl += sl; nb += 1
        tr_tot /= max(1, nb); tr_fm_l /= max(1, nb); tr_kl /= max(1, nb); tr_sl /= max(1, nb)

        # ── validate (always the fast detached surrogate — no gradient needed) ─
        model.eval()
        va_tot = va_fm_l = va_kl = va_sl = 0.0; nbv = 0
        with torch.no_grad():
            for bx, bfm, bk, bidx in va_loader:
                bx, bfm, bk = bx.to(device), bfm.to(device), bk.to(device)
                b_ref_np = va_ref[bidx.numpy()]
                b_src_np = va_src[bidx.numpy()]
                fm_p = model(bx)
                loss, fl, kl, sl = combined_loss(
                    fm_p, bfm, bk, b_ref_np, b_src_np, mask_tensor, row_active_tensor,
                    alpha, beta, gamma,
                    differentiable=False,
                    power_iter_steps=cfg.POWER_ITER_STEPS,
                )
                va_tot += loss.item(); va_fm_l += fl; va_kl += kl; va_sl += sl; nbv += 1
        va_tot /= max(1, nbv); va_fm_l /= max(1, nbv); va_kl /= max(1, nbv); va_sl /= max(1, nbv)

        sched.step(va_tot)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"    Ep {epoch:4d} | "
                f"tr[total={tr_tot:.5f}  fm={tr_fm_l:.5f}  keff={tr_kl:.5f}  src={tr_sl:.5f}] | "
                f"va[total={va_tot:.5f}  fm={va_fm_l:.5f}  keff={va_kl:.5f}  src={va_sl:.5f}] | "
                f"lr={opt.param_groups[0]['lr']:.2e}"
            )

        if va_tot < best_val - 1e-12:
            best_val, best_ep, no_improve = va_tot, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= cfg.EARLY_STOP_PATIENCE:
            if verbose:
                print(f"    Early stop at epoch {epoch} (best {best_ep})")
            break

    if best_state:
        model.load_state_dict(best_state)

    return float(best_val), best_ep, model, mu, sig, va_idx

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / data saver  (called whenever a new best PCM is found)
# ─────────────────────────────────────────────────────────────────────────────

def save_best(run_dir, model, mu, sig, params, val_loss, best_ep,
              pcm_arr, kfm_arr, mean_abs_pcm, trial_record):
    """Persist everything about the current best trial to disk."""

    if cfg.SAVE_CHECKPOINTS:
        ckpt_path = run_dir / "best_model_checkpoint.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "train_mean":       mu,
            "train_std":        sig,
            "hyperparameters":  params,
            "best_val_loss":    val_loss,
            "best_epoch":       int(best_ep),
            "mean_abs_pcm":     mean_abs_pcm,
        }, ckpt_path)

    np.save(run_dir / "best_pcm_array.npy", pcm_arr)
    np.save(run_dir / "best_kfm_array.npy", kfm_arr)

    with open(run_dir / "best_so_far.json", "w") as f:
        json.dump({**trial_record, "mean_abs_pcm": mean_abs_pcm}, f, indent=2)

    print(f"  ✓ Saved new best  mean|PCM| = {mean_abs_pcm:.3f} pcm  → {run_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=600,
                        help="Total Bayesian optimisation calls (default 600)")
    parser.add_argument("--initial-points", type=int, default=35,
                        help="Random exploration points before GP kicks in (default 35)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = cfg.RESULTS_ROOT / f"search_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")
    print(f"Bayesian search: {args.trials} trials  "
          f"({args.initial_points} random seed points + "
          f"{args.trials - args.initial_points} GP-guided)")
    print(f"Gradient tracking through power iteration: {cfg.DIFFERENTIABLE_POWER_ITER}")
    print("Objective: minimise mean |PCM| on validation split\n")

    # ── Load data once ─────────────────────────────────────────────────────
    print("Loading data...")
    (X_all, Y_fm_all, Y_source_all, Y_keff_all, good_indices,
     fuel_temps_all, fm_lib, mask) = prepare_dataset()

    print("Pre-building reference FM cache...")
    ref_fm_cache = build_ref_fm_cache(fuel_temps_all, fm_lib, mask)
    print(f"ref_fm_cache: {ref_fm_cache.shape}  ({ref_fm_cache.nbytes/1e9:.2f} GB)")

    # ── Trial bookkeeping ──────────────────────────────────────────────────
    trial_log   = []
    trial_count = [0]
    best_pcm    = [float("inf")]   # running best mean |PCM| (mutable)

    csv_path  = run_dir / "trial_log.csv"
    csv_file  = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "trial", "hidden_size", "learning_rate", "weight_decay",
        "alpha", "beta", "gamma", "best_val_loss", "best_epoch",
        "mean_abs_pcm", "objective",
    ])

    # ── Objective function ────────────────────────────────────────────────
    @use_named_args(SEARCH_SPACE)
    def objective(hidden_size, learning_rate, weight_decay, alpha, beta, gamma):
        t = trial_count[0]
        trial_count[0] += 1
        seed = cfg.BASE_SEED + t

        print(f"\n{'='*70}")
        print(f"Trial {t+1}/{args.trials} | "
              f"hidden={hidden_size}  lr={learning_rate:.2e}  wd={weight_decay:.2e}  "
              f"α={alpha:.3g}  β={beta:.3g}  γ={gamma:.3g}")

        mean_abs_pcm = float("nan")
        val_loss     = float("nan")
        best_ep      = -1
        model = mu = sig = None
        va_idx = np.array([], dtype=int)

        try:
            val_loss, best_ep, model, mu, sig, va_idx = train_trial(
                X_all, Y_fm_all, Y_source_all, Y_keff_all, ref_fm_cache, mask,
                int(hidden_size), float(learning_rate),
                float(weight_decay), float(alpha), float(beta), float(gamma),
                seed=seed, verbose=False,
            )

            # PCM evaluation on the validation split (already row-aligned)
            va_features   = X_all[va_idx]
            va_keff       = Y_keff_all[va_idx]
            va_fuel_temps = fuel_temps_all[va_idx]

            pcm_arr, kfm_arr, mean_abs_pcm = evaluate_pcm_compact(
                model, mu, sig, va_features, va_keff, va_fuel_temps,
                fm_lib, mask, batch_size=cfg.BATCH_SIZE,
            )

        except Exception as e:
            print(f"[ERROR] Trial failed: {e}")
            traceback.print_exc()
            mean_abs_pcm = 1e9   # penalise crashed trials
            pcm_arr = kfm_arr = np.array([])

        print(f"  → val_loss = {val_loss:.6f}  |  mean|PCM| = {mean_abs_pcm:.3f} pcm"
              f"  (best epoch {best_ep})")

        record = {
            "trial":         t + 1,
            "hidden_size":   int(hidden_size),
            "learning_rate": float(learning_rate),
            "weight_decay":  float(weight_decay),
            "alpha":         float(alpha),
            "beta":          float(beta),
            "gamma":         float(gamma),
            "best_val_loss": float(val_loss) if np.isfinite(val_loss) else None,
            "best_epoch":    int(best_ep),
            "mean_abs_pcm":  float(mean_abs_pcm) if np.isfinite(mean_abs_pcm) else None,
            "objective":     float(mean_abs_pcm),
        }
        trial_log.append(record)
        csv_writer.writerow([record[k] for k in
            ["trial", "hidden_size", "learning_rate", "weight_decay",
             "alpha", "beta", "gamma", "best_val_loss", "best_epoch",
             "mean_abs_pcm", "objective"]])
        csv_file.flush()

        # ── Save immediately whenever we find a new best PCM ────────────
        if np.isfinite(mean_abs_pcm) and mean_abs_pcm < best_pcm[0]:
            best_pcm[0] = mean_abs_pcm
            save_best(
                run_dir, model, mu, sig,
                {k: record[k] for k in
                 ["hidden_size", "learning_rate", "weight_decay", "alpha", "beta", "gamma"]},
                val_loss, best_ep,
                pcm_arr, kfm_arr, mean_abs_pcm,
                record,
            )

        # skopt minimises the return value — use mean |PCM|
        return float(mean_abs_pcm) if np.isfinite(mean_abs_pcm) else 1e9

    # ── Run Bayesian optimisation ──────────────────────────────────────────
    print("Starting Bayesian optimisation...")
    result = gp_minimize(
        objective,
        SEARCH_SPACE,
        n_calls          = args.trials,
        n_initial_points = args.initial_points,
        acq_func         = "EI",
        noise            = 1e-6,
        random_state     = cfg.BASE_SEED,
        verbose          = False,
    )

    csv_file.close()

    # ── Best hyperparameters ───────────────────────────────────────────────
    best_params = {
        name: val for name, val in
        zip([d.name for d in SEARCH_SPACE], result.x)
    }
    best_mean_pcm = result.fun
    print("\n" + "="*70)
    print("BEST HYPERPARAMETERS FOUND")
    print("="*70)
    for k, v in best_params.items():
        print(f"  {k:20s} = {v}")
    print(f"  {'best_mean_abs_pcm':20s} = {best_mean_pcm:.3f} pcm")

    # ── Retrain best config with verbose + full-dataset PCM evaluation ─────
    print("\nRetraining best config (verbose) for full PCM evaluation...")
    best_seed = cfg.BASE_SEED + 9999
    val_loss_final, best_ep_final, best_model, best_mu, best_sig, _ = train_trial(
        X_all, Y_fm_all, Y_source_all, Y_keff_all, ref_fm_cache, mask,
        int(best_params["hidden_size"]),
        float(best_params["learning_rate"]),
        float(best_params["weight_decay"]),
        float(best_params["alpha"]),
        float(best_params["beta"]),
        float(best_params["gamma"]),
        seed=best_seed, verbose=True,
    )

    print("\nRunning full PCM evaluation on best model (all samples)...")
    pcm_arr_full, kfm_arr_full, mean_abs_pcm_full = evaluate_pcm_compact(
        best_model, best_mu, best_sig,
        X_all, Y_keff_all, fuel_temps_all, fm_lib, mask,
        batch_size=cfg.BATCH_SIZE,
    )
    print(f"Best model mean |PCM| (full dataset) = {mean_abs_pcm_full:.3f} pcm")

    # Overwrite with the full-dataset evaluation
    if cfg.SAVE_CHECKPOINTS:
        ckpt_path = run_dir / "best_model_checkpoint_final.pth"
        torch.save({
            "model_state_dict": best_model.state_dict(),
            "train_mean":       best_mu,
            "train_std":        best_sig,
            "hyperparameters":  best_params,
            "best_val_loss":    val_loss_final,
            "best_epoch":       int(best_ep_final),
            "mean_abs_pcm_full": mean_abs_pcm_full,
        }, ckpt_path)
        print(f"Final checkpoint saved → {ckpt_path}")

    np.save(run_dir / "final_pcm_array.npy",  pcm_arr_full)
    np.save(run_dir / "final_kfm_array.npy",  kfm_arr_full)

    # ── Full results summary ───────────────────────────────────────────────
    summary = {
        "best_hyperparameters":        best_params,
        "best_val_mean_abs_pcm":       best_mean_pcm,
        "final_full_mean_abs_pcm":     mean_abs_pcm_full,
        "n_trials":                    args.trials,
        "n_initial_points":            args.initial_points,
        "differentiable_power_iter":   cfg.DIFFERENTIABLE_POWER_ITER,
        "all_trials":                  trial_log,
    }
    with open(run_dir / "search_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {run_dir / 'search_summary.json'}")

    # ── Convergence plot ───────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    objectives  = [r["objective"] for r in trial_log]
    running_min = np.minimum.accumulate(objectives)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, len(objectives) + 1), objectives,
                 'o', alpha=0.4, markersize=4, label="Trial mean |PCM|")
    axes[0].plot(range(1, len(running_min) + 1), running_min,
                 '-', linewidth=2, color='red', label="Running best")
    axes[0].axvline(args.initial_points, linestyle='--', color='grey',
                    alpha=0.6, label="GP start")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("mean |PCM| (validation split)")
    axes[0].set_title("Bayesian search convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Parallel coordinates of top-20 trials
    top20 = sorted(trial_log, key=lambda r: r["objective"])[:20]
    dims  = ["hidden_size", "learning_rate", "weight_decay", "alpha", "beta", "gamma"]
    vals  = np.array([[np.log10(r[d]) if d != "hidden_size" else r[d]
                       for d in dims] for r in top20])
    for i, row in enumerate(vals):
        axes[1].plot(range(len(dims)), row,
                     alpha=0.6, linewidth=1.5,
                     color=plt.cm.viridis(i / max(1, len(vals) - 1)))
    axes[1].set_xticks(range(len(dims)))
    axes[1].set_xticklabels(["hidden", "log10(lr)", "log10(wd)",
                              "log10(α)", "log10(β)", "log10(γ)"], fontsize=9)
    axes[1].set_ylabel("Parameter value")
    axes[1].set_title("Top-20 trials (parallel coordinates)")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Hyperparameter search — best val mean|PCM| = {best_mean_pcm:.2f} pcm  |  "
        f"full-dataset mean|PCM| = {mean_abs_pcm_full:.2f} pcm",
        fontsize=11,
    )
    plt.tight_layout()
    plot_path = run_dir / "search_results.png"
    plt.savefig(plot_path, dpi=200)
    print(f"Plot saved → {plot_path}")
    print(f"Full results → {run_dir}")


if __name__ == "__main__":
    main()