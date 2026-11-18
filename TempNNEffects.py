#!/usr/bin/env python3
"""
evaluate_fm_best_worst.py

Loads the best already-trained FMCoefficientPredictor checkpoint and evaluates
every temperature configuration in training_data.h5.

PCM = (k_pred / k_true - 1) * 1e5
"""

import numpy as np
import h5py
import torch
import torch.nn as nn
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

class Config:
    H5_PATH = Path("training_data.h5")

    CHECKPOINT_PATH = Path(
        "ds3_fm_keff_loss_results/"
        "train_fm_keff_loss_20261010_120627/"
        "hidden_1744/"
        "run_1/"
        "checkpoint_hidden1744.pth"
    )

    MASK_PATH = Path(
        "training_data_nonzero_mask.npy"
    )

    N_ROWS         = 17
    N_COLS         = 17
    N_CELLS        = 289
    N_CELLS2       = 264
    INPUT_SIZE     = 553
    OUTPUT_SIZE    = 69696
    N_FUEL_COMPACT = 264

    EPSILON    = 1e-8
    BATCH_SIZE = 256
    N_SAMPLES  = None   # None = all

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()


# ============================================================
# MODEL
# ============================================================

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


# ============================================================
# POWER ITERATION  — guarded against zero eigenvalue
# ============================================================

def power_iter_numpy(fm_np, tol=1e-7, max_iter=1000):
    n   = fm_np.shape[0]
    v   = np.ones((n, 1), dtype=float)
    eig = 1.0
    for i in range(max_iter):
        v_new   = fm_np.dot(v)
        eig_new = float(np.max(np.abs(v_new)))
        if eig_new == 0 or not np.isfinite(eig_new):
            break
        v = v_new / eig_new
        if i > 0 and abs(eig_new - eig) < tol:
            eig = eig_new
            break
        eig = eig_new
    v = v.ravel()
    s = v.sum()
    if s != 0:
        v /= s
    return float(eig), v


# ============================================================
# REFERENCE FM
# ============================================================

def load_fm_library_from_h5(h5file):
    ref_mats  = h5file["Reference Matrices"][:]
    ref_temps = h5file["reference_temperatures"][:]
    fm_lib = {}
    for mat_flat, t in zip(ref_mats, ref_temps):
        fm = np.asarray(mat_flat, dtype=float).reshape(cfg.N_CELLS2, cfg.N_CELLS2)
        fm_lib[int(t)] = fm
    print(f"  FM library keys: {sorted(fm_lib.keys())} K")
    return fm_lib


def interpolate_rowwise_reference(fuel_temps_compact, fm_lib_temps, mask):
    keys     = sorted(fm_lib_temps.keys())
    per_temp = {t: fm_lib_temps[t] for t in keys}
    fuel_T   = np.asarray(fuel_temps_compact, dtype=float)
    baseline = np.zeros((cfg.N_CELLS2, cfg.N_CELLS2), dtype=float)
    for i in range(cfg.N_CELLS2):
        Ti = fuel_T[i]
        if Ti <= keys[0]:
            baseline[i, :] = per_temp[keys[0]][i, :]
        elif Ti >= keys[-1]:
            baseline[i, :] = per_temp[keys[-1]][i, :]
        else:
            for low, high in zip(keys[:-1], keys[1:]):
                if low <= Ti <= high:
                    alpha = (Ti - low) / (high - low) if high != low else 0.0
                    baseline[i, :] = (
                        (1.0 - alpha) * per_temp[low][i, :]
                        + alpha       * per_temp[high][i, :]
                    )
                    break
    return baseline.flatten()


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print(f"Loading checkpoint: {cfg.CHECKPOINT_PATH}")
    ckpt = torch.load(
        cfg.CHECKPOINT_PATH,
        map_location=cfg.DEVICE,
        weights_only=False,
    )
    hidden_size = ckpt["config"]["HIDDEN_SIZE"]
    model = FMCoefficientPredictor(
        cfg.INPUT_SIZE, hidden_size, cfg.OUTPUT_SIZE
    ).to(cfg.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mu  = np.asarray(ckpt["train_mean"], dtype=np.float32)
    sig = np.asarray(ckpt["train_std"],  dtype=np.float32)
    print(f"  Hidden size  : {hidden_size}")
    print(f"  Best epoch   : {ckpt.get('best_epoch', 'N/A')}")
    print(f"  Best val loss: {ckpt.get('best_val_loss', 'N/A')}")
    print(f"  Device       : {cfg.DEVICE}")
    return model, mu, sig


# ============================================================
# DIAGNOSTICS — print HDF5 structure so we can see what's there
# ============================================================

def diagnose_h5():
    print("\n--- HDF5 DIAGNOSTICS ---")
    with h5py.File(cfg.H5_PATH, "r") as h5:
        print(f"  Keys: {list(h5.keys())}")
        for key in h5.keys():
            ds = h5[key]
            if hasattr(ds, 'shape'):
                arr = ds[:5]
                print(f"  '{key}': shape={ds.shape}  dtype={ds.dtype}  "
                      f"first5={arr.ravel()[:5]}")
    print("------------------------\n")


# ============================================================
# EVALUATE
# ============================================================

def evaluate_all(model, mu, sig, mask):

    diagnose_h5()

    print("Loading HDF5 data ...")
    with h5py.File(cfg.H5_PATH, "r") as h5:
        X_temps  = h5["temperatures"][:]
        keff_raw = h5["keff"][:]
        fm_lib   = load_fm_library_from_h5(h5)

    N = X_temps.shape[0]
    if cfg.N_SAMPLES is not None:
        N = min(N, cfg.N_SAMPLES)
    X_temps  = X_temps[:N]
    keff_raw = keff_raw[:N]

    # ── keff diagnostics ────────────────────────────────────────────────
    print(f"\n  keff array: shape={keff_raw.shape}  dtype={keff_raw.dtype}")
    print(f"  keff[:10] = {keff_raw[:10]}")
    print(f"  finite keff: {np.sum(np.isfinite(keff_raw))} / {N}")
    print(f"  keff range: [{np.nanmin(keff_raw):.6f}, {np.nanmax(keff_raw):.6f}]")
    valid_keff = np.isfinite(keff_raw) & (keff_raw != 0)
    print(f"  valid keff (finite & non-zero): {np.sum(valid_keff)} / {N}\n")

    # ── normalisation diagnostics ────────────────────────────────────────
    # The training script does TWO steps:
    #   Step 1: fixed range  X_range = (X_raw - 300) / (600 + 1e-12)
    #   Step 2: standardise  X_norm  = (X_range - mu) / sig
    # mu/sig in the checkpoint were computed on X_range, not X_raw.
    print("\n--- NORMALISATION DIAGNOSTICS ---")
    print(f"  Raw temps[:3, :5]   = {X_temps[:3, :5]}")
    print(f"  mu[:5]              = {mu[:5]}")
    print(f"  sig[:5]             = {sig[:5]}")
    print(f"  mu   range: [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"  sig  range: [{sig.min():.4f}, {sig.max():.4f}]")

    # Determine whether mu looks like it was computed on raw temps (300-900)
    # or on range-normalised temps (0-1).
    if mu.mean() > 10:
        print("  -> mu looks like RAW temperature scale: applying range norm first")
        X_range = (X_temps - 300.0) / (600.0 + 1e-12)
        X_norm  = ((X_range - mu) / sig).astype(np.float32)
    else:
        print("  -> mu looks like range-normalised scale: applying standardisation only")
        X_range = (X_temps - 300.0) / (600.0 + 1e-12)
        X_norm  = ((X_range - mu) / sig).astype(np.float32)

    sample0_norm = X_norm[0]
    print(f"  X_norm[0] range: [{sample0_norm.min():.4f}, {sample0_norm.max():.4f}]  "
          f"mean={sample0_norm.mean():.4f}")
    print("---------------------------------\n")

    preds  = np.empty((N, cfg.OUTPUT_SIZE), dtype=np.float32)

    print(f"Running inference on {N} samples ...")
    with torch.no_grad():
        for start in range(0, N, cfg.BATCH_SIZE):
            end   = min(start + cfg.BATCH_SIZE, N)
            batch = torch.from_numpy(X_norm[start:end]).to(cfg.DEVICE)
            preds[start:end] = model(batch).cpu().numpy()
    print("  Inference done.")

    # ── prediction diagnostics ───────────────────────────────────────────
    print("\n--- PREDICTION DIAGNOSTICS ---")
    print(f"  preds shape       : {preds.shape}")
    print(f"  preds[0] range    : [{preds[0].min():.4f}, {preds[0].max():.4f}]  "
          f"mean={preds[0].mean():.4f}")
    print(f"  exp(preds[0]) range: [{np.exp(preds[0]).min():.4f}, {np.exp(preds[0]).max():.4f}]")
    print(f"  % of |preds| > 10 : {100*np.mean(np.abs(preds)>10):.2f}%")
    print("------------------------------\n")

    # ── per-sample PCM ───────────────────────────────────────────────────
    pcm_errors = np.full(N, np.nan, dtype=np.float64)
    k_preds    = np.full(N, np.nan, dtype=np.float64)
    k_trues    = keff_raw.astype(np.float64)

    n_skip_keff  = 0
    n_skip_power = 0
    n_ok         = 0

    print(f"Running power iteration for {N} samples ...")
    for i in range(N):
        k_true = k_trues[i]

        if not np.isfinite(k_true) or k_true == 0:
            n_skip_keff += 1
            continue

        try:
            fuel_compact  = X_temps[i, :cfg.N_FUEL_COMPACT]
            fm_ref_flat   = interpolate_rowwise_reference(fuel_compact, fm_lib, mask)
            fm_ref        = fm_ref_flat.reshape(cfg.N_CELLS2, cfg.N_CELLS2) + cfg.EPSILON

            ratio_mat     = np.exp(preds[i]).reshape(cfg.N_CELLS2, cfg.N_CELLS2)
            fm_pred_clean = ratio_mat * fm_ref

            fm_pred_full  = np.zeros((cfg.N_CELLS, cfg.N_CELLS), dtype=float)
            fm_pred_full[mask] = fm_pred_clean.flatten()

            k_pred, _ = power_iter_numpy(fm_pred_full)

            if not np.isfinite(k_pred) or k_pred == 0:
                n_skip_power += 1
                continue

            k_preds[i]    = k_pred
            pcm_errors[i] = (k_pred / k_true - 1.0) * 1e5
            n_ok         += 1

        except Exception as e:
            print(f"  [WARN] sample {i}: {e}")
            n_skip_power += 1

        if (i + 1) % 500 == 0 or (i + 1) == N:
            print(f"  Power iter: {i+1}/{N}  "
                  f"(ok={n_ok}, skip_keff={n_skip_keff}, skip_power={n_skip_power})")

    print(f"\n  Completed: {n_ok} valid PCM values, "
          f"{n_skip_keff} skipped (bad k_true), "
          f"{n_skip_power} skipped (bad k_pred)")

    return X_temps, k_trues, k_preds, pcm_errors


# ============================================================
# REPORT
# ============================================================

def print_sample(rank, idx, temps, pcm, k_pred, k_true, label):
    print(f"\n  [{label} #{rank+1}]  sample index = {idx}")
    print(f"    PCM error   : {pcm:+.3f} pcm")
    print(f"    k_pred      : {k_pred:.6f}")
    print(f"    k_true      : {k_true:.6f}")
    print(f"    |Δk|        : {abs(k_pred - k_true):.2e}")
    print(f"    Temp vector (first 20 of {len(temps)}):")
    print(f"    {np.array2string(temps[:20], precision=2, separator=', ')}")
    if len(temps) > 20:
        print(f"    … ({len(temps) - 20} more values)")


def report(X_temps, k_trues, k_preds, pcm_errors, top_k=5):
    finite_mask = np.isfinite(pcm_errors)
    n_finite    = np.sum(finite_mask)

    sep = "=" * 70

    if n_finite == 0:
        print(f"\n{sep}")
        print("ERROR: No finite PCM values — nothing to rank.")
        print("Check the HDF5 diagnostics above.")
        print(sep)
        return np.array([], dtype=int), np.array([], dtype=int)

    finite_idx = np.where(finite_mask)[0]
    abs_pcm    = np.abs(pcm_errors[finite_idx])
    order      = np.argsort(abs_pcm)

    k_best  = min(top_k, len(order))
    best_local  = order[:k_best]
    worst_local = order[-k_best:][::-1]
    best_idx    = finite_idx[best_local]
    worst_idx   = finite_idx[worst_local]

    print(f"\n{sep}")
    print(f"TOP {k_best} BEST  — smallest |keff error| in PCM")
    print(sep)
    for rank, idx in enumerate(best_idx):
        print_sample(rank, idx, X_temps[idx],
                     pcm_errors[idx], k_preds[idx], k_trues[idx], "BEST")

    print(f"\n{sep}")
    print(f"TOP {k_best} WORST — largest |keff error| in PCM")
    print(sep)
    for rank, idx in enumerate(worst_idx):
        print_sample(rank, idx, X_temps[idx],
                     pcm_errors[idx], k_preds[idx], k_trues[idx], "WORST")

    fin = pcm_errors[finite_mask]
    print(f"\n{sep}")
    print("GLOBAL keff ERROR STATISTICS  (PCM)")
    print(sep)
    print(f"  N samples evaluated  : {n_finite}")
    print(f"  Mean   PCM           : {np.mean(fin):+.3f}")
    print(f"  Mean  |PCM|          : {np.mean(np.abs(fin)):.3f}")
    print(f"  Median |PCM|         : {np.median(np.abs(fin)):.3f}")
    print(f"  Std    PCM           : {np.std(fin):.3f}")
    print(f"  Min    PCM           : {np.min(fin):+.3f}")
    print(f"  Max    PCM           : {np.max(fin):+.3f}")
    print(f"  Worst |PCM|          : {np.max(np.abs(fin)):.3f}")

    return best_idx, worst_idx


# ============================================================
# SAVE
# ============================================================

def save_results(X_temps, k_trues, k_preds, pcm_errors,
                 best_idx, worst_idx,
                 out_path="fm_best_worst_pcm_results.npz"):
    np.savez(
        out_path,
        best_indices   = best_idx,
        worst_indices  = worst_idx,
        best_temps     = X_temps[best_idx] if len(best_idx) else np.array([]),
        worst_temps    = X_temps[worst_idx] if len(worst_idx) else np.array([]),
        best_pcm       = pcm_errors[best_idx] if len(best_idx) else np.array([]),
        worst_pcm      = pcm_errors[worst_idx] if len(worst_idx) else np.array([]),
        best_k_pred    = k_preds[best_idx] if len(best_idx) else np.array([]),
        worst_k_pred   = k_preds[worst_idx] if len(worst_idx) else np.array([]),
        best_k_true    = k_trues[best_idx] if len(best_idx) else np.array([]),
        worst_k_true   = k_trues[worst_idx] if len(worst_idx) else np.array([]),
        all_pcm_errors = pcm_errors,
        all_k_preds    = k_preds,
        all_k_trues    = k_trues,
    )
    print(f"\nResults saved → {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("FM PREDICTION — BEST & WORST TEMPERATURE DISTRIBUTIONS (PCM)")
    print("=" * 70)

    model, mu, sig = load_model()
    mask = np.load(cfg.MASK_PATH)
    print(f"Mask shape: {mask.shape}  active cells: {mask.sum()}")

    X_temps, k_trues, k_preds, pcm_errors = evaluate_all(model, mu, sig, mask)
    best_idx, worst_idx = report(X_temps, k_trues, k_preds, pcm_errors)
    save_results(X_temps, k_trues, k_preds, pcm_errors, best_idx, worst_idx)

    print("\nDone.")


if __name__ == "__main__":
    main()