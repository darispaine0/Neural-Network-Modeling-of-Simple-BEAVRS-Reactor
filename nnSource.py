#!/usr/bin/env python3
"""
ds3_sweep_fm_keff_loss_h5.py

Same as ds3_sweep_fm_keff_loss.py, but all training data (input temps,
fission matrices, k-eff, reference matrices, reference temperatures) is
read from training_data.h5 instead of individual .npy files.

Only the fuel/non-fuel cell mask (global_nonzero_mask.npy) is still loaded
from disk as .npy, since it isn't part of the HDF5 schema.

L = ALPHA * MSE(ln-ratio FM)
  + BETA  * mean( ((k_pred - k_true) / k_true * 1e5)² ) / 1e10
  + GAMMA * MSE(fission source vector)

k_pred is derived from the predicted FM via power iteration each batch
(surrogate: detached from autograd, so gradient flows only through FM MSE,
but the BETA term still pulls training toward FM states that yield good k-eff).

The fission source vector is the normalized dominant eigenvector of the FM
(the same power iteration that yields k-eff also yields this vector "for
free"). It represents the spatial neutron fission-source distribution across
the assembly, so matching it pushes training toward the correct power shape,
not just the correct eigenvalue.

For a fully differentiable version, set DIFFERENTIABLE_POWER_ITER = True,
which rewrites power iteration in PyTorch so gradients flow exactly through
the FM prediction. This is slower but more accurate.
"""
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import traceback
import random

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class GlobalConfig:
    # Data paths
    H5_PATH   = Path("training_data.h5")
    MASK_PATH = Path("training_data_nonzero_mask.npy")

    # Grid sizes
    N_ROWS   = 17
    N_COLS   = 17
    N_CELLS  = N_ROWS * N_COLS   # 289 — full grid
    N_CELLS2 = 264               # cleaned (non-zero rows/cols)

    # Feature: compact fuel temps (264) + full other temps (289) = 553
    N_FUEL_COMPACT = N_CELLS2
    INPUT_SIZE     = N_FUEL_COMPACT + N_CELLS   # 553
    OUTPUT_SIZE    = N_CELLS2 * N_CELLS2         # 69 696

    # Loss weights
    #   L = ALPHA * MSE(ln-ratio FM)  +  BETA * mean((PCM/1e5)²) / 1e10
    #       + GAMMA * MSE(fission source vector)
    # BETA = 0.0  → k-eff surrogate term disabled
    # GAMMA = 0.0 → fission-source term disabled
    ALPHA = 0.00287
    BETA  = 0.31
    GAMMA = 7.69e-06   # <-- NEW: weight on the fission-source MSE term; needs its own tuning pass, same as BETA

    # Power iteration mode
    # False → surrogate (fast, approximate gradient)
    # True  → differentiable PyTorch power iter (exact gradient, slower)
    DIFFERENTIABLE_POWER_ITER = True
    POWER_ITER_STEPS = 20   # iterations used during training (more = more accurate but slower)

    # Training
    BATCH_SIZE          = 32
    LEARNING_RATE       = 1e-04
    WEIGHT_DECAY        = 4.36e-08
    MAX_EPOCHS          = 1000
    VALIDATION_SPLIT    = 0.2
    EARLY_STOP_PATIENCE = 10
    LR_REDUCE_PATIENCE  = 5

    # Sweep
    HIDDEN_SIZE = 1744
    N_RUNS       = 1

    # Output
    RESULTS_ROOT     = Path("ds3_fm_keff_loss_results")
    SWEEP_SUBDIR     = None
    SAVE_CHECKPOINTS = True

    # Numeric safety
    EPSILON = 1e-8

    # k-eff target: OpenMC track-length k-eff (written by build_h5.py).
    # Set to "keff" to use the FM power-iteration eigenvalue instead.
    KEFF_DATASET = "keff_tracklength"

cfg = GlobalConfig()


def read_keff(h5file):
    """Read the k-eff target dataset; stop with a clear message if it's missing."""
    if cfg.KEFF_DATASET not in h5file:
        raise KeyError(
            f"'{cfg.KEFF_DATASET}' not found in {cfg.H5_PATH}. "
            f"Rebuild the file with build_h5.py, or set KEFF_DATASET = \"keff\" "
            f"to use the FM power-iteration eigenvalue."
        )
    return h5file[cfg.KEFF_DATASET][:]


# ─────────────────────────────────────────────────────────────────────────────
# Data utilities  (HDF5-backed)
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


def interpolate_rowwise_reference(fuel_temps_compact, fm_lib_temps, mask):
    keys          = sorted(fm_lib_temps.keys())
    per_temp_rows = {t: fm_lib_temps[t] for t in keys}
    fuel_T        = np.asarray(fuel_temps_compact, dtype=float)

    if fuel_T.size != cfg.N_CELLS2:
        raise ValueError(f"fuel_temps_compact length {fuel_T.size} != {cfg.N_CELLS2}")

    baseline = np.zeros((cfg.N_CELLS2, cfg.N_CELLS2), dtype=float)
    for i in range(cfg.N_CELLS2):
        Ti = fuel_T[i]
        if Ti <= keys[0]:
            baseline[i, :] = per_temp_rows[keys[0]][i, :]
        elif Ti >= keys[-1]:
            baseline[i, :] = per_temp_rows[keys[-1]][i, :]
        else:
            for low, high in zip(keys[:-1], keys[1:]):
                if low <= Ti <= high:
                    alpha = (Ti - low) / (high - low) if high != low else 0.0
                    baseline[i, :] = (
                        (1.0 - alpha) * per_temp_rows[low][i, :]
                        + alpha       * per_temp_rows[high][i, :]
                    )
                    break
    return baseline.flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset():
    """
    Loads temperatures, fission matrices, and k-eff straight from
    training_data.h5. The 'temperatures' dataset already stores the
    concatenated feature vector (264 compact fuel temps + 289 other temps),
    so no per-sample feature reconstruction is needed — the first 264
    columns are the compact fuel temps used for reference interpolation.
    """
    print("Loading data from training_data.h5...")
    with h5py.File(cfg.H5_PATH, "r") as h5:
        X_temps      = h5["temperatures"][:]     # (N, 553)
        fm_samples   = h5["fission_matrix"][:]   # (N, 69696)
        keff_raw     = read_keff(h5)             # (N,) track-length k-eff
        source_raw   = h5["fission_source"][:]   # (N, 264) — compact fuel-cell ordering, same as fission_matrix rows/cols
        fm_lib       = load_fm_library_from_h5(h5)

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
    Y_source       = np.stack(source_list,      axis=0)   # (M, N_CELLS2) — straight from h5, compact fuel-cell order

    g_min    = 300
    g_max    = 900
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
    return (X_inputs, Y_fm, Y_source, Y_keff, good_indices, fuel_temps_all, fm_lib, mask)


# ─────────────────────────────────────────────────────────────────────────────
# Model  (single FM head — no keff head)
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
# Power iteration — two flavours
# ─────────────────────────────────────────────────────────────────────────────

def power_iter_numpy(fm_np, tol=1e-7, max_iter=1000):
    """Standard numpy power iteration. Returns (dominant eigenvalue k-eff, normalized fission-source vector)."""
    n = fm_np.shape[0]
    v = np.ones((n, 1), dtype=float)
    eig = 1.0
    for i in range(max_iter):
        v_new   = fm_np.dot(v)
        eig_new = float(np.max(np.abs(v_new)))
        if eig_new == 0:
            break
        v = v_new / eig_new
        if i > 0 and abs(eig_new - eig) < tol:
            eig = eig_new
            break
        eig = eig_new
    v = v.ravel(); s = v.sum()
    if s != 0:
        v /= s
    return float(eig), v


def power_iter_torch(fm_tensor, n_steps):
    """
    Differentiable power iteration in PyTorch.
    fm_tensor : (N, N) — must stay in the compute graph.
    Returns (dominant eigenvalue, normalized fission-source vector), both
    differentiable tensors.
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
    v_norm = (v / (v.sum() + 1e-12)).reshape(-1)   # normalized fission source, differentiable
    return eig, v_norm   # eig: scalar tensor; v_norm: (N,) tensor


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_full_matrix_torch(cleaned_flat_tensor, mask_tensor):
    """
    Scatter a (N_CELLS2*N_CELLS2,) tensor into a full (N_CELLS, N_CELLS) tensor.
    Used for differentiable path only.
    mask_tensor : (N_CELLS, N_CELLS) bool tensor (on same device)
    """
    full = torch.zeros(cfg.N_CELLS, cfg.N_CELLS,
                       dtype=cleaned_flat_tensor.dtype,
                       device=cleaned_flat_tensor.device)
    full[mask_tensor] = cleaned_flat_tensor
    return full


def combined_loss(fm_pred, fm_true, keff_true_batch,
                  fm_ref_batch,       # (B, N_CELLS2, N_CELLS2) numpy array, pre-built
                  source_true_batch,  # (B, N_CELLS2) numpy array, true fission-source vectors straight from h5 (compact order)
                  mask_tensor,
                  row_active_tensor,  # (N_CELLS,) bool tensor — which full-grid rows are the N_CELLS2 fuel cells, ascending order
                  alpha, beta, gamma,
                  differentiable=False,
                  power_iter_steps=20):
    """
    L = alpha * MSE(fm_pred, fm_true)
      + beta  * mean( ((k_pred - k_true) / k_true * 1e5)² ) / 1e10
      + gamma * MSE(v_pred, v_true)                 # fission source vector

    Two modes for the k-eff / fission-source terms:
      differentiable=False  →  surrogate: power iter in numpy on detached FM,
                                k_pred/v_pred re-attached as constant tensors
                                so the BETA/GAMMA terms still contribute to the
                                total scalar loss (but their gradient w.r.t. FM
                                weights is zero). The FM MSE gradient still
                                dominates training.
      differentiable=True   →  power iter in PyTorch on live tensors, exact
                                gradient flows back through fm_pred for both
                                k-eff and the fission-source vector.

    The power iteration itself still runs on the full (N_CELLS, N_CELLS)
    reconstructed matrix (unchanged from before), since that's what k-eff is
    defined on. The resulting (N_CELLS,) eigenvector is then restricted to
    the N_CELLS2 active fuel rows via row_active_tensor before comparing
    against the h5-stored fission_source, which is already in that compact
    order (identical convention to how fm_clean is scattered into the full
    matrix via mask_tensor).

    Args:
        fm_pred             : (B, OUTPUT_SIZE) predicted ln-ratio, in compute graph
        fm_true              : (B, OUTPUT_SIZE) target ln-ratio tensor
        keff_true_batch      : (B,) k-eff truth tensor
        fm_ref_batch         : numpy (B, N_CELLS2, N_CELLS2) reference FM for this batch
        source_true_batch    : numpy (B, N_CELLS2) true fission-source vectors for this batch
        mask_tensor          : bool tensor (N_CELLS, N_CELLS) on device
        row_active_tensor    : bool tensor (N_CELLS,) on device
        alpha, beta, gamma   : loss weights
        differentiable        : whether to use differentiable power iteration
        power_iter_steps     : number of power iter steps (differentiable mode only)

    Returns:
        total    : differentiable scalar tensor
        fm_mse   : float (logged)
        keff_l   : float (logged, scaled)
        source_l : float (logged)
    """
    fm_mse_t = nn.functional.mse_loss(fm_pred, fm_true)

    if beta == 0.0 and gamma == 0.0:
        return alpha * fm_mse_t, float(fm_mse_t.detach()), 0.0, 0.0

    B      = fm_pred.shape[0]
    device = fm_pred.device

    if differentiable:
        # ── exact gradient path ──────────────────────────────────────────
        fm_ref_t  = torch.from_numpy(fm_ref_batch).float().to(device)  # (B, N2, N2)
        ratio_mat = torch.exp(fm_pred).reshape(B, cfg.N_CELLS2, cfg.N_CELLS2)
        fm_clean  = ratio_mat * fm_ref_t                                 # (B, N2, N2)

        k_preds = []
        v_preds = []
        for b in range(B):
            fm_full = reconstruct_full_matrix_torch(
                fm_clean[b].reshape(-1), mask_tensor
            )
            k_b, v_b_full = power_iter_torch(fm_full, power_iter_steps)  # v_b_full: (N_CELLS,)
            v_b_compact = v_b_full[row_active_tensor]                    # (N_CELLS2,), still differentiable
            k_preds.append(k_b)
            v_preds.append(v_b_compact)
        k_pred_t = torch.stack(k_preds)                                  # (B,)
        v_pred_t = torch.stack(v_preds)                                  # (B, N_CELLS2)

    else:
        # ── surrogate path (fast) ────────────────────────────────────────
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
        # Re-attach as constant tensors — contribute to total loss scalar
        # but carry no gradient back through fm_pred
        k_pred_t = torch.from_numpy(k_np).to(device)
        v_pred_t = torch.from_numpy(v_np).to(device)

    pcm      = (k_pred_t - keff_true_batch) / (keff_true_batch + 1e-8) * 1e5
    keff_l_t = (pcm ** 2).mean() / 1e10

    source_true_t = torch.from_numpy(source_true_batch).float().to(device)
    source_l_t = torch.linalg.vector_norm(v_pred_t - source_true_t)

    total = alpha * fm_mse_t + beta * keff_l_t + gamma * source_l_t
    return total, float(fm_mse_t.detach()), float(keff_l_t.detach()), float(source_l_t.detach())


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def build_ref_fm_cache(fuel_temps_all, fm_lib, mask):
    """
    Pre-compute the reference FM for every sample once so we don't repeat
    the interpolation inside every training batch.
    Returns (M, N_CELLS2, N_CELLS2) float32 array.
    """
    M   = fuel_temps_all.shape[0]
    out = np.zeros((M, cfg.N_CELLS2, cfg.N_CELLS2), dtype=np.float32)
    for i in range(M):
        ref = interpolate_rowwise_reference(fuel_temps_all[i], fm_lib, mask)
        out[i] = ref.reshape(cfg.N_CELLS2, cfg.N_CELLS2)
    return out


def train_one_run(X, Y_fm, Y_source, Y_keff, ref_fm_cache,
                  seed, hidden_size, run_dir, mask_np):
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

    # We store sample indices into ref_fm_cache/Y_source so we can retrieve
    # the reference FM slice and the true source vector inside the loop
    # without adding them as extra dense tensors to the DataLoader.
    def make_loader(Xn, fm, keff, shuffle):
        ds = TensorDataset(
            torch.from_numpy(Xn.astype(np.float32)),
            torch.from_numpy(fm.astype(np.float32)),
            torch.from_numpy(keff.astype(np.float32)),
            torch.arange(len(Xn)),   # local index → used to slice ref/source arrays
        )
        return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle)

    tr_loader = make_loader(tr_Xn, tr_fm, tr_keff, shuffle=True)
    va_loader = make_loader(va_Xn, va_fm, va_keff, shuffle=False)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_tensor = torch.from_numpy(mask_np).to(device)
    # Which of the N_CELLS full-grid rows are the N_CELLS2 active fuel cells,
    # in ascending order — this is the same ordering fm_clean is scattered
    # into full_np via mask_tensor, so restricting a full-grid eigenvector to
    # these rows recovers the compact ordering used by the h5 fission_source.
    row_active_tensor = torch.from_numpy(mask_np.any(axis=1)).to(device)

    model     = FMCoefficientPredictor(cfg.INPUT_SIZE, hidden_size, cfg.OUTPUT_SIZE).to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=cfg.LR_REDUCE_PATIENCE, verbose=False
    )

    best_val, best_ep, no_improve, best_state = float('inf'), -1, 0, None

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        tr_tot = tr_fm_l = tr_kl = tr_sl = 0.0
        nb = 0
        for bx, bfm, bk, bidx in tr_loader:
            bx   = bx.to(device)
            bfm  = bfm.to(device)
            bk   = bk.to(device)
            # Retrieve numpy ref FM / true source slices for this batch
            b_ref_np = tr_ref[bidx.numpy()]   # (B, N_CELLS2, N_CELLS2)
            b_src_np = tr_src[bidx.numpy()]   # (B, N_CELLS)

            optimizer.zero_grad()
            fm_p = model(bx)
            loss, fl, kl, sl = combined_loss(
                fm_p, bfm, bk, b_ref_np, b_src_np, mask_tensor, row_active_tensor,
                cfg.ALPHA, cfg.BETA, cfg.GAMMA,
                differentiable=cfg.DIFFERENTIABLE_POWER_ITER,
                power_iter_steps=cfg.POWER_ITER_STEPS,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_tot += loss.item(); tr_fm_l += fl; tr_kl += kl; tr_sl += sl; nb += 1
        tr_tot /= max(1, nb); tr_fm_l /= max(1, nb); tr_kl /= max(1, nb); tr_sl /= max(1, nb)

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        va_tot = va_fm_l = va_kl = va_sl = 0.0
        nbv = 0
        with torch.no_grad():
            for bx, bfm, bk, bidx in va_loader:
                bx  = bx.to(device)
                bfm = bfm.to(device)
                bk  = bk.to(device)
                b_ref_np = va_ref[bidx.numpy()]
                b_src_np = va_src[bidx.numpy()]
                fm_p = model(bx)
                loss, fl, kl, sl = combined_loss(
                    fm_p, bfm, bk, b_ref_np, b_src_np, mask_tensor, row_active_tensor,
                    cfg.ALPHA, cfg.BETA, cfg.GAMMA,
                    differentiable=False,       # always use surrogate for val
                    power_iter_steps=cfg.POWER_ITER_STEPS,
                )
                va_tot += loss.item(); va_fm_l += fl; va_kl += kl; va_sl += sl; nbv += 1
        va_tot /= max(1, nbv); va_fm_l /= max(1, nbv); va_kl /= max(1, nbv); va_sl /= max(1, nbv)

        scheduler.step(va_tot)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"    Ep {epoch:4d} | "
                f"tr[total={tr_tot:.5f}  fm={tr_fm_l:.5f}  keff={tr_kl:.5f}  src={tr_sl:.5f}] | "
                f"va[total={va_tot:.5f}  fm={va_fm_l:.5f}  keff={va_kl:.5f}  src={va_sl:.5f}] | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if va_tot < best_val - 1e-12:
            best_val, best_ep, no_improve = va_tot, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= cfg.EARLY_STOP_PATIENCE:
            print(f"    Early stop at epoch {epoch}  (best epoch {best_ep})")
            break

    if best_state:
        model.load_state_dict(best_state)

    if cfg.SAVE_CHECKPOINTS:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "train_mean": mu, "train_std": sig,
            "config": {
                "INPUT_SIZE":  cfg.INPUT_SIZE,
                "HIDDEN_SIZE": hidden_size,
                "OUTPUT_SIZE": cfg.OUTPUT_SIZE,
                "ALPHA":       cfg.ALPHA,
                "BETA":        cfg.BETA,
                "GAMMA":       cfg.GAMMA,
                "DIFFERENTIABLE_POWER_ITER": cfg.DIFFERENTIABLE_POWER_ITER,
            },
            "best_epoch":    best_ep,
            "best_val_loss": float(best_val),
        }, run_dir / f"checkpoint_hidden{hidden_size}.pth")

    return model, mu, sig, best_val, best_ep


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model_pcm(model, mu, sig,
                       features_all, good_indices,
                       keff_full, fuel_temps_all, fm_lib, mask,
                       source_true_all=None,
                       batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, features_all.shape[0], batch_size):
            e   = min(s + batch_size, features_all.shape[0])
            Xn  = (features_all[s:e] - mu) / sig
            out = model(torch.from_numpy(Xn.astype(np.float32)).to(device))
            preds.append(out.cpu().numpy())
    preds = np.vstack(preds)

    M        = preds.shape[0]
    pcm_list = np.full(M, np.nan)
    kfm_list = np.full(M, np.nan)
    src_mae_list = np.full(M, np.nan)   # mean abs error of the fission-source vector

    # Same active-row convention as combined_loss / row_active_tensor: which
    # of the N_CELLS full-grid rows correspond to the N_CELLS2 fuel cells,
    # in ascending order. Restricting the full-grid eigenvector to these
    # rows recovers the compact ordering used by the h5 fission_source.
    row_active = mask.any(axis=1)

    for local_i, global_idx in enumerate(good_indices):
        try:
            ratio_mat     = np.exp(preds[local_i]).reshape(cfg.N_CELLS2, cfg.N_CELLS2)
            fm_ref_flat   = interpolate_rowwise_reference(fuel_temps_all[local_i], fm_lib, mask)
            fm_pred_clean = ratio_mat * fm_ref_flat.reshape(cfg.N_CELLS2, cfg.N_CELLS2)
            fm_pred_full  = np.zeros((cfg.N_CELLS, cfg.N_CELLS), dtype=float)
            fm_pred_full[mask] = fm_pred_clean.flatten()
            k_pred, v_pred_full = power_iter_numpy(fm_pred_full)
            v_pred_compact = v_pred_full[row_active]   # (N_CELLS2,) — matches h5 fission_source ordering
            kfm_list[local_i] = k_pred

            if keff_full is not None and global_idx < keff_full.size:
                k_true = keff_full[global_idx]
                if np.isfinite(k_pred) and np.isfinite(k_true) and k_true != 0:
                    pcm_list[local_i] = (k_pred / k_true - 1.0) * 1e5

            if source_true_all is not None:
                src_mae_list[local_i] = float(np.linalg.norm(v_pred_compact - source_true_all[local_i]))
        except Exception as e:
            print(f"[WARN] Sample {local_i} (global {global_idx}): {e}")

    fin          = pcm_list[np.isfinite(pcm_list)]
    mean_abs_pcm = float(np.mean(np.abs(fin))) if fin.size > 0 else float("nan")

    fin_src      = src_mae_list[np.isfinite(src_mae_list)]
    mean_src_mae = float(np.mean(fin_src)) if fin_src.size > 0 else float("nan")

    return pcm_list, kfm_list, mean_abs_pcm, src_mae_list, mean_src_mae


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir = cfg.RESULTS_ROOT / f"train_fm_keff_loss_{timestamp}"
    cfg.SWEEP_SUBDIR = train_dir
    train_dir.mkdir(parents=True, exist_ok=True)
    print("Sweep output directory:", train_dir)
    print(f"Loss: ALPHA(FM MSE)={cfg.ALPHA}  BETA(keff PCM²)={cfg.BETA}  "
          f"GAMMA(fission source MSE)={cfg.GAMMA}  "
          f"diff_power_iter={cfg.DIFFERENTIABLE_POWER_ITER}")

    (X_all, Y_fm_all, Y_source_all, Y_keff_all, good_indices,
     fuel_temps_all, fm_lib, mask) = prepare_dataset()

    # Pre-build reference FM cache (avoids re-interpolating every batch)
    print("Pre-building reference FM cache for all samples...")
    ref_fm_cache = build_ref_fm_cache(fuel_temps_all, fm_lib, mask)
    print(f"ref_fm_cache shape: {ref_fm_cache.shape}  "
          f"({ref_fm_cache.nbytes / 1e9:.2f} GB)")

    # Full un-sliced keff for evaluation
    keff_full = None
    try:
        with h5py.File(cfg.H5_PATH, "r") as h5:
            keff_full = read_keff(h5)
        print(f"Loaded full keff array length {keff_full.size}")
    except Exception as e:
        print(f"[WARN] Failed to load keff: {e}")

    hidden_size = cfg.HIDDEN_SIZE
    print("\n" + "=" * 80)
    print(f"Hidden size: {hidden_size}  ({cfg.N_RUNS} run(s))")
    run_metrics = []

    for run_idx in range(cfg.N_RUNS):
        seed       = 44906 + hidden_size * 10 + run_idx
        run_subdir = train_dir / f"hidden_{hidden_size}" / f"run_{run_idx + 1}"
        print(f"\n--- Run {run_idx+1}/{cfg.N_RUNS}  seed={seed} ---")

        try:
            model, mu, sig, best_val, best_ep = train_one_run(
                X_all, Y_fm_all, Y_source_all, Y_keff_all, ref_fm_cache,
                seed, hidden_size, run_subdir, mask
            )
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            traceback.print_exc()
            run_metrics.append(float("nan"))
            continue

        print("Evaluating PCM and fission-source error...")
        pcm_arr, kfm_arr, mean_abs_pcm, src_mae_arr, mean_src_mae = evaluate_model_pcm(
            model, mu, sig,
            X_all, good_indices,
            keff_full, fuel_temps_all, fm_lib, mask,
            source_true_all=Y_source_all,
            batch_size=cfg.BATCH_SIZE,
        )
        print(f"  Run {run_idx+1} mean |PCM| = {mean_abs_pcm:.3f} pcm   "
              f"mean source MAE = {mean_src_mae:.6e}")
        run_metrics.append(mean_abs_pcm)

        run_subdir.mkdir(parents=True, exist_ok=True)
        np.save(run_subdir / "pcm_array.npy", pcm_arr)
        np.save(run_subdir / "kfm_array.npy", kfm_arr)
        np.save(run_subdir / "source_mae_array.npy", src_mae_arr)
        with open(run_subdir / "meta.json", "w") as f:
            json.dump({
                "hidden_size":   hidden_size,
                "run":           run_idx + 1,
                "seed":          seed,
                "best_epoch":    int(best_ep),
                "best_val_loss": float(best_val),
                "mean_abs_pcm":  mean_abs_pcm,
                "mean_source_mae": mean_src_mae,
                "alpha":         cfg.ALPHA,
                "beta":          cfg.BETA,
                "gamma":         cfg.GAMMA,
                "differentiable_power_iter": cfg.DIFFERENTIABLE_POWER_ITER,
            }, f, indent=2)

    a  = np.array(run_metrics, dtype=float)
    v  = np.isfinite(a)
    mn = float(np.nanmean(a[v])) if v.any() else float("nan")
    sd = float(np.nanstd( a[v])) if v.any() else float("nan")

    print(f"\nSUMMARY hidden={hidden_size}  mean |PCM| = {mn:.3f} ± {sd:.3f} pcm")

if __name__ == "__main__":
    main()