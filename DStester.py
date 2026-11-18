"""
ds_sweep.py

Tests four data structure formats (DS1–DS4) with and without ln() transform
on the fission matrix targets, using a fixed small neural network
(HW=405, WD=1e-5, LR=1e-4) as specified in the methodology.

Data structures:
  DS1 = raw temps          + raw FM
  DS2 = normalized temps   + raw FM          [(T - 300) / 600]
  DS3 = raw temps          + ref-normalized FM  [ln(FM / FM_ref)]
  DS4 = normalized temps   + ref-normalized FM  [ln(FM / FM_ref)]

For each DS, two target variants are tested:
  - raw target (FM as-is for DS1/DS2, ln-ratio as-is for DS3/DS4)
  - ln() transform of target  [ln(|target| + eps)]

Each combination is trained 5 times with different seeds.
Results are saved to ds_sweep_results/ with a summary CSV.

Hyperparameters (fixed from flowchart):
  HIDDEN_SIZE = 405
  WEIGHT_DECAY = 1e-5
  LEARNING_RATE = 1e-4
  ALPHA = 1.0, BETA = 0.0, GAMMA = 0.0  (FM MSE only — no keff/source terms)
"""

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
import csv
import traceback
import random
from datetime import datetime

# =============================================================================
# Config
# =============================================================================

class Config:
    H5_PATH   = Path("training_data.h5")
    MASK_PATH = Path("training_data_nonzero_mask.npy")

    N_ROWS   = 17
    N_COLS   = 17
    N_CELLS  = N_ROWS * N_COLS   # 289
    N_CELLS2 = 264

    N_FUEL_COMPACT = N_CELLS2
    INPUT_SIZE     = N_FUEL_COMPACT + N_CELLS   # 553
    OUTPUT_SIZE    = N_CELLS2 * N_CELLS2        # 69 696

    # Fixed hyperparameters from flowchart
    HIDDEN_SIZE      = 405
    LEARNING_RATE    = 1e-4
    WEIGHT_DECAY     = 1e-5
    BATCH_SIZE       = 32
    MAX_EPOCHS       = 500
    VALIDATION_SPLIT = 0.2
    EARLY_STOP_PATIENCE = 10
    LR_REDUCE_PATIENCE  = 5

    N_RUNS  = 5       # runs per DS/transform combination
    EPSILON = 1e-8

    # Temp normalization constants for DS2/DS4
    TEMP_SHIFT = 300.0
    TEMP_SCALE = 600.0

    RESULTS_ROOT = Path("ds_sweep_results")

cfg = Config()


# =============================================================================
# Helpers
# =============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fm_library_from_h5(h5file):
    ref_mats  = h5file["Reference Matrices"][:]
    ref_temps = h5file["reference_temperatures"][:]
    fm_lib = {}
    for mat_flat, t in zip(ref_mats, ref_temps):
        fm = np.asarray(mat_flat, dtype=float).reshape(cfg.N_CELLS2, cfg.N_CELLS2)
        fm_lib[int(t)] = fm
    print(f"[INFO] FM library temps: {sorted(fm_lib.keys())} K")
    return fm_lib


def interpolate_rowwise_reference(fuel_temps_compact, fm_lib):
    keys = sorted(fm_lib.keys())
    fuel_T = np.asarray(fuel_temps_compact, dtype=float)
    baseline = np.zeros((cfg.N_CELLS2, cfg.N_CELLS2), dtype=float)
    for i in range(cfg.N_CELLS2):
        Ti = fuel_T[i]
        if Ti <= keys[0]:
            baseline[i, :] = fm_lib[keys[0]][i, :]
        elif Ti >= keys[-1]:
            baseline[i, :] = fm_lib[keys[-1]][i, :]
        else:
            for low, high in zip(keys[:-1], keys[1:]):
                if low <= Ti <= high:
                    alpha = (Ti - low) / (high - low) if high != low else 0.0
                    baseline[i, :] = (1.0 - alpha) * fm_lib[low][i, :] + alpha * fm_lib[high][i, :]
                    break
    return baseline


# =============================================================================
# Data loading
# =============================================================================

def load_raw_data():
    """Load temperatures, raw FM, and keff from HDF5."""
    print("Loading data from training_data.h5 ...")
    with h5py.File(cfg.H5_PATH, "r") as h5:
        X_temps    = h5["temperatures"][:]    # (N, 553)
        fm_samples = h5["fission_matrix"][:]  # (N, 69696)
        keff_raw   = h5["keff"][:]            # (N,)
        fm_lib     = load_fm_library_from_h5(h5)

    mask = np.load(cfg.MASK_PATH)

    n = min(X_temps.shape[0], fm_samples.shape[0], keff_raw.shape[0])
    print(f"Total samples: {n}")

    features, fuel_temps_list, fm_list, keff_list, valid_idx = [], [], [], [], []
    skipped = []

    for i in range(n):
        k = keff_raw[i]
        if not np.isfinite(k):
            skipped.append(i); continue

        feat = np.asarray(X_temps[i], dtype=np.float32)
        if feat.shape[0] != cfg.INPUT_SIZE:
            skipped.append(i); continue

        fm_arr = np.asarray(fm_samples[i], dtype=np.float32)
        if fm_arr.size != cfg.OUTPUT_SIZE:
            skipped.append(i); continue

        features.append(feat)
        fuel_temps_list.append(feat[:cfg.N_FUEL_COMPACT])
        fm_list.append(fm_arr)
        keff_list.append(float(k))
        valid_idx.append(i)

    if skipped:
        print(f"[WARN] Skipped {len(skipped)} invalid samples.")
    if not features:
        raise RuntimeError("No valid samples found.")

    features       = np.stack(features,       axis=0)   # (M, 553)
    fuel_temps_all = np.stack(fuel_temps_list, axis=0)  # (M, 264)
    fm_all         = np.stack(fm_list,         axis=0)  # (M, 69696)
    keff_all       = np.array(keff_list,       dtype=np.float32)

    print(f"Valid samples: {len(features)}")
    return features, fuel_temps_all, fm_all, keff_all, fm_lib, mask, valid_idx


def build_ref_fm_targets(fuel_temps_all, fm_all, fm_lib):
    """
    Build ln(FM / FM_ref) targets for DS3/DS4.
    Returns (M, OUTPUT_SIZE) float32 array.
    """
    M = fuel_temps_all.shape[0]
    Y = np.zeros((M, cfg.OUTPUT_SIZE), dtype=np.float32)
    print("Building reference-normalised FM targets (ln-ratio) ...")
    for i in range(M):
        fm_ref = interpolate_rowwise_reference(fuel_temps_all[i], fm_lib)   # (264, 264)
        fm_arr = fm_all[i].reshape(cfg.N_CELLS2, cfg.N_CELLS2)
        ratio  = np.clip((fm_arr + cfg.EPSILON) / (fm_ref + cfg.EPSILON), cfg.EPSILON, None)
        Y[i]   = np.log(ratio).flatten().astype(np.float32)
    return Y


# =============================================================================
# Dataset builder  —  one function per DS / ln variant
# =============================================================================

def build_dataset(features, fm_all, Y_ref_norm, ds_name, use_ln):
    """
    Returns (X, Y) numpy arrays ready for training.

    ds_name : 'DS1' | 'DS2' | 'DS3' | 'DS4'
    use_ln  : whether to apply ln(|Y| + eps) to the target
    """
    # ── Input features ──────────────────────────────────────────────────────
    if ds_name in ("DS1", "DS3"):
        # Raw temperatures (already in Kelvin from HDF5)
        X = features.copy()
    else:
        # DS2, DS4: normalized temps  (T - 300) / 600
        X = features.copy()
        X = (X - cfg.TEMP_SHIFT) / cfg.TEMP_SCALE

    # ── FM targets ──────────────────────────────────────────────────────────
    if ds_name in ("DS1", "DS2"):
        # Raw FM
        Y = fm_all.copy()
    else:
        # DS3, DS4: ln-ratio (already computed)
        Y = Y_ref_norm.copy()

    # ── Optional ln() transform on targets ──────────────────────────────────
    if use_ln:
        Y = np.log(np.abs(Y) + cfg.EPSILON).astype(np.float32)

    return X.astype(np.float32), Y.astype(np.float32)


# =============================================================================
# Model
# =============================================================================

class SingleHiddenNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Training
# =============================================================================

def train_one_run(X, Y, seed, run_dir):
    set_seed(seed)

    M     = X.shape[0]
    n_val = int(M * cfg.VALIDATION_SPLIT)
    perm  = np.random.permutation(M)
    tr_idx, va_idx = perm[:M - n_val], perm[M - n_val:]

    tr_X, tr_Y = X[tr_idx], Y[tr_idx]
    va_X, va_Y = X[va_idx], Y[va_idx]

    # Standardise inputs using training statistics
    mu  = tr_X.mean(axis=0)
    sig = tr_X.std(axis=0) + cfg.EPSILON
    tr_Xn = (tr_X - mu) / sig
    va_Xn = (va_X - mu) / sig

    def make_loader(Xn, Yn, shuffle):
        ds = TensorDataset(
            torch.from_numpy(Xn),
            torch.from_numpy(Yn),
        )
        return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle)

    tr_loader = make_loader(tr_Xn, tr_Y, shuffle=True)
    va_loader = make_loader(va_Xn, va_Y, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = SingleHiddenNN(cfg.INPUT_SIZE, cfg.HIDDEN_SIZE, cfg.OUTPUT_SIZE).to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=cfg.LR_REDUCE_PATIENCE, verbose=False
    )
    criterion = nn.MSELoss()

    best_val, best_ep, no_improve, best_state = float('inf'), -1, 0, None
    tr_history, va_history = [], []

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        # Train
        model.train()
        tr_loss = 0.0
        nb = 0
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item(); nb += 1
        tr_loss /= max(1, nb)

        # Validate
        model.eval()
        va_loss = 0.0
        nbv = 0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                va_loss += criterion(model(bx), by).item(); nbv += 1
        va_loss /= max(1, nbv)

        scheduler.step(va_loss)
        tr_history.append(tr_loss)
        va_history.append(va_loss)

        if epoch % 50 == 0 or epoch == 1:
            print(f"      Ep {epoch:4d} | tr={tr_loss:.6f}  va={va_loss:.6f} "
                  f"| lr={optimizer.param_groups[0]['lr']:.2e}")

        if va_loss < best_val - 1e-12:
            best_val, best_ep, no_improve = va_loss, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= cfg.EARLY_STOP_PATIENCE:
            print(f"      Early stop at epoch {epoch}  (best ep {best_ep})")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Save checkpoint
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "train_mean": mu,
        "train_std":  sig,
        "best_epoch": best_ep,
        "best_val_loss": float(best_val),
    }, run_dir / "checkpoint.pth")

    np.save(run_dir / "tr_loss.npy", np.array(tr_history))
    np.save(run_dir / "va_loss.npy", np.array(va_history))

    return model, mu, sig, best_val, best_ep


# =============================================================================
# Evaluation  —  mean |PCM| error
# =============================================================================

def power_iter_numpy(fm_np, tol=1e-7, max_iter=1000):
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
    return float(eig)


def evaluate_pcm(model, mu, sig,
                 features_all, fm_all, fuel_temps_all,
                 keff_all, fm_lib, mask,
                 ds_name, use_ln):
    """
    Reconstruct predicted FM from model output, run power iteration,
    compute mean |PCM| error against true keff.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Rebuild input features the same way as training
    if ds_name in ("DS1", "DS3"):
        X = features_all.copy().astype(np.float32)
    else:
        X = ((features_all - cfg.TEMP_SHIFT) / cfg.TEMP_SCALE).astype(np.float32)

    # Normalise with training stats
    Xn = (X - mu) / sig

    preds = []
    with torch.no_grad():
        for s in range(0, Xn.shape[0], cfg.BATCH_SIZE):
            e   = min(s + cfg.BATCH_SIZE, Xn.shape[0])
            out = model(torch.from_numpy(Xn[s:e]).to(device))
            preds.append(out.cpu().numpy())
    preds = np.vstack(preds)   # (M, OUTPUT_SIZE)

    M        = preds.shape[0]
    pcm_list = np.full(M, np.nan)

    for i in range(M):
        try:
            pred_i = preds[i]   # (OUTPUT_SIZE,)

            # Undo ln() transform if it was applied to targets
            if use_ln:
                pred_i = np.exp(pred_i) - cfg.EPSILON

            if ds_name in ("DS1", "DS2"):
                # Target was raw FM — pred is directly the FM (flat, compact)
                fm_pred_clean = pred_i.reshape(cfg.N_CELLS2, cfg.N_CELLS2)
            else:
                # Target was ln-ratio — reconstruct FM via exp(pred) * FM_ref
                fm_ref        = interpolate_rowwise_reference(fuel_temps_all[i], fm_lib)
                ratio_mat     = np.exp(pred_i).reshape(cfg.N_CELLS2, cfg.N_CELLS2)
                fm_pred_clean = ratio_mat * fm_ref

            # Scatter into full 289×289 grid for power iteration
            fm_full      = np.zeros((cfg.N_CELLS, cfg.N_CELLS), dtype=float)
            fm_full[mask] = fm_pred_clean.flatten()

            k_pred = power_iter_numpy(fm_full)
            k_true = float(keff_all[i])

            if np.isfinite(k_pred) and np.isfinite(k_true) and k_true != 0:
                pcm_list[i] = (k_pred / k_true - 1.0) * 1e5

        except Exception as e:
            print(f"    [WARN] Sample {i}: {e}")

    fin          = pcm_list[np.isfinite(pcm_list)]
    mean_abs_pcm = float(np.mean(np.abs(fin))) if fin.size > 0 else float("nan")
    return pcm_list, mean_abs_pcm


# =============================================================================
# Main sweep
# =============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root  = cfg.RESULTS_ROOT / f"sweep_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {out_root}")

    # Load everything once
    features, fuel_temps_all, fm_all, keff_all, fm_lib, mask, valid_idx = load_raw_data()

    # Pre-build ln-ratio targets (needed for DS3/DS4)
    Y_ref_norm = build_ref_fm_targets(fuel_temps_all, fm_all, fm_lib)

    # Combinations to test
    ds_names   = ["DS1", "DS2", "DS3", "DS4"]
    ln_options = [False, True]

    summary_rows = []

    for ds_name in ds_names:
        for use_ln in ln_options:
            label = f"{ds_name}{'_ln' if use_ln else ''}"
            print("\n" + "=" * 70)
            print(f"  Testing: {label}")
            print("=" * 70)

            X, Y = build_dataset(features, fm_all, Y_ref_norm, ds_name, use_ln)

            run_pcms = []
            for run_idx in range(cfg.N_RUNS):
                seed    = 42 + run_idx * 17
                run_dir = out_root / label / f"run_{run_idx + 1}"
                print(f"\n  Run {run_idx + 1}/{cfg.N_RUNS}  seed={seed}")

                try:
                    model, mu, sig, best_val, best_ep = train_one_run(X, Y, seed, run_dir)
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    traceback.print_exc()
                    run_pcms.append(float("nan"))
                    continue

                pcm_arr, mean_abs_pcm = evaluate_pcm(
                    model, mu, sig,
                    features, fm_all, fuel_temps_all,
                    keff_all, fm_lib, mask,
                    ds_name, use_ln
                )
                print(f"  --> Mean |PCM| = {mean_abs_pcm:.3f} pcm")
                run_pcms.append(mean_abs_pcm)

                np.save(run_dir / "pcm_array.npy", pcm_arr)
                with open(run_dir / "meta.json", "w") as f:
                    json.dump({
                        "label":         label,
                        "ds_name":       ds_name,
                        "use_ln":        use_ln,
                        "run":           run_idx + 1,
                        "seed":          seed,
                        "best_epoch":    int(best_ep),
                        "best_val_loss": float(best_val),
                        "mean_abs_pcm":  mean_abs_pcm,
                        "hidden_size":   cfg.HIDDEN_SIZE,
                        "lr":            cfg.LEARNING_RATE,
                        "wd":            cfg.WEIGHT_DECAY,
                    }, f, indent=2)

            arr   = np.array(run_pcms, dtype=float)
            valid = np.isfinite(arr)
            mean  = float(np.nanmean(arr[valid])) if valid.any() else float("nan")
            std   = float(np.nanstd( arr[valid])) if valid.any() else float("nan")
            best  = float(np.nanmin( arr[valid])) if valid.any() else float("nan")

            print(f"\n  {label}: mean |PCM| = {mean:.3f} ± {std:.3f}  best = {best:.3f}")

            summary_rows.append({
                "label":       label,
                "ds_name":     ds_name,
                "ln_transform": use_ln,
                "mean_pcm":    round(mean, 4),
                "std_pcm":     round(std,  4),
                "best_pcm":    round(best, 4),
                "n_runs":      cfg.N_RUNS,
                "hidden_size": cfg.HIDDEN_SIZE,
                "lr":          cfg.LEARNING_RATE,
                "wd":          cfg.WEIGHT_DECAY,
            })

    # Write summary CSV
    csv_path = out_root / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"{'Label':<15} {'Mean |PCM|':>12} {'Std':>8} {'Best':>8}")
    print("-" * 45)
    for r in sorted(summary_rows, key=lambda x: x["mean_pcm"]):
        print(f"{r['label']:<15} {r['mean_pcm']:>12.3f} {r['std_pcm']:>8.3f} {r['best_pcm']:>8.3f}")
    print(f"\nSummary CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()