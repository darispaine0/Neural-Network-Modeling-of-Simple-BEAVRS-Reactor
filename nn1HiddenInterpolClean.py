#!/usr/bin/env python3
"""
ds3_sweep_hidden_neurons.py - FIXED for cleaned matrices with proper reconstruction

Trains on cleaned 264×264 matrices, then reconstructs full 289×289 for PCM evaluation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import math
import csv
import sys
import traceback
import random

# ---------------- USER ADJUSTABLE CONFIG ----------------
class GlobalConfig:
    # Data paths
    INPUT_TEMPS_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/input_temps_final.npy")
    FM_NORMALIZED_PATH = Path("/home/daris/Monte_Carlo/Test/training_data2/output_fm_normalized_final_cleaned.npy")
    FM_294K_PATH = Path("/home/daris/Monte_Carlo/Test/training_data2/output_fm_normalized294k_cleaned.npy")
    KEFF_PATH = Path("/home/daris/Monte_Carlo/Test/training_data/output_keff_final.npy")
    FM_LIBRARY_DIR = Path("/home/daris/Monte_Carlo/Test/training_data2")
    MASK_PATH = Path("/home/daris/Monte_Carlo/Test/training_data2/global_nonzero_mask.npy")

    # Grid sizes
    N_ROWS = 17
    N_COLS = 17
    N_CELLS = N_ROWS * N_COLS          # 289 (full)
    N_CELLS2 = 264                      # cleaned
    INPUT_SIZE = N_CELLS * 2            # 578
    OUTPUT_SIZE = N_CELLS2 * N_CELLS2   # 69696

    # Best model hyperparameters from tuning
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-5
    MAX_EPOCHS = 500
    VALIDATION_SPLIT = 0.2
    EARLY_STOP_PATIENCE = 5
    LR_REDUCE_PATIENCE = 10

    # Sweep settings
    HIDDEN_START = 1030
    HIDDEN_STOP = 1020  # inclusive lower bound
    HIDDEN_STEP = -10
    N_RUNS = 1          # runs per hidden-size

    # Output / bookkeeping
    RESULTS_ROOT = Path("ds3_neuron_sweep_results")
    SWEEP_SUBDIR = None  # filled at runtime
    SAVE_CHECKPOINTS = True

    # numeric safety
    EPSILON = 1e-8
    LN_CLIP_MIN = -10.0
    LN_CLIP_MAX = 10.0

cfg = GlobalConfig()

# ---------------- Utilities (data loading / preprocessing) ----------------
def load_keff_values(path):
    arr = np.load(path, allow_pickle=True)
    keffs = []
    for val in arr:
        try:
            if hasattr(val, "nominal_value"):
                keffs.append(float(val.nominal_value))
            else:
                keffs.append(float(val))
        except Exception:
            keffs.append(np.nan)
    return np.array(keffs, dtype=float)

def make_feature_from_input(temp_tuple, n_cells):
    """Build 578-length feature from (fuel_vec, other_vec) and per-cell temperatures"""
    fuel_vec, other_vec = temp_tuple
    fuel_grid = np.zeros(n_cells, dtype=np.float32)
    fuel_idx = 0
    for i in range(n_cells):
        if fuel_idx < len(fuel_vec):
            try:
                val = float(fuel_vec[fuel_idx])
            except Exception:
                val = 0.0
            if val > 0:
                fuel_grid[i] = val
                fuel_idx += 1
        else:
            break
    other_grid = np.asarray(other_vec, dtype=np.float32).flatten()
    if other_grid.size != n_cells:
        raise ValueError(f"other_vec size {other_grid.size} != {n_cells}")
    feat = np.concatenate([fuel_grid, other_grid])
    # representative per-cell temperature
    cell_temps = np.where(fuel_grid > 0.0, fuel_grid, other_grid)
    return feat, cell_temps

def load_fm_library_if_available():
    """Load fission matrices from fission_matrix_300k through fission_matrix_900k"""
    fm_lib = {}
    if not cfg.FM_LIBRARY_DIR.exists():
        print(f"[WARN] FM library directory does not exist: {cfg.FM_LIBRARY_DIR}")
        return None
    
    missing_files = []
    for t in range(300, 901, 100):
        p = cfg.FM_LIBRARY_DIR / f"fission_matrix_{t}k_cleaned"
        if not p.exists():
            p_npy = cfg.FM_LIBRARY_DIR / f"fission_matrix_{t}k_cleaned.npy"
            if p_npy.exists():
                p = p_npy
            else:
                missing_files.append(f"fission_matrix_{t}k_cleaned")
                continue
        
        try:
            fm = np.load(p, allow_pickle=True)
            if isinstance(fm, np.ndarray) and fm.dtype == object:
                if fm.ndim == 0:
                    fm = fm.item()
                elif fm.size == 1:
                    fm = fm.flatten()[0]
            fm = np.asarray(fm, dtype=float)
            while fm.ndim > 2:
                fm = np.squeeze(fm)
            if fm.ndim == 1:
                fm = fm.reshape(cfg.N_CELLS2, cfg.N_CELLS2)
            elif fm.ndim == 2 and fm.shape == (cfg.N_CELLS2, cfg.N_CELLS2):
                pass
            else:
                print(f"[ERROR] Unexpected shape for {p}: {fm.shape}")
                return None
            fm_lib[t] = fm
            print(f"[INFO] Loaded {p.name} with shape {fm.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to load {p}: {e}")
            traceback.print_exc()
            return None
    
    if missing_files:
        print(f"[WARN] Missing FM library files: {missing_files}")
        return None
    
    print(f"[INFO] Successfully loaded FM library: {sorted(fm_lib.keys())} K")
    return fm_lib

def interpolate_rowwise_reference(cell_temps, fm_lib_temps, mask):
    """
    For each non-zero row in cleaned space, interpolate using corresponding cell temperature.
    Returns cleaned 264×264 matrix (flattened). 
    """
    keys = sorted(fm_lib_temps.keys())
    per_temp_rows = {t: fm_lib_temps[t] for t in keys}
    Tcell = np.asarray(cell_temps, dtype=float)
    
    baseline = np.zeros((cfg.N_CELLS2, cfg.N_CELLS2), dtype=float)
    
    # Map from full 289 cells to cleaned 264 rows
    # Get row indices where mask has at least one True value
    row_has_data = np.any(mask, axis=1)  # (289,) boolean
    active_rows = np.where(row_has_data)[0]  # indices of non-zero rows
    
    for i, full_row_idx in enumerate(active_rows):
        if i >= cfg.N_CELLS2:
            break
        Ti = Tcell[full_row_idx]
        
        if Ti <= keys[0]:
            baseline[i, :] = per_temp_rows[keys[0]][i, :]
        elif Ti >= keys[-1]:
            baseline[i, :] = per_temp_rows[keys[-1]][i, :]
        else:
            for low, high in zip(keys[:-1], keys[1:]):
                if low <= Ti <= high:
                    if high == low:
                        baseline[i, :] = per_temp_rows[low][i, :]
                    else:
                        alpha = (Ti - low) / (high - low)
                        baseline[i, :] = (1-alpha) * per_temp_rows[low][i, :] + alpha * per_temp_rows[high][i, :]
                    break
    
    return baseline.flatten()

def prepare_ds3_dataset():
    """
    Loads input_temps, fm_samples, fm_294k, and FM library.
    Returns:
      X_inputs: (M, 578) minmax normalized features
      Y_targets: (M, 69696) ln-transformed ratios relative to row-wise interpolated reference
      fm_294k: (264,264) cleaned array
      good_indices: list of original sample indices
      cell_temps_all: (M, 289) per-sample cell temperatures
      fm_lib: dict of library matrices
      mask: (289, 289) boolean mask
    """
    print("Loading raw arrays...")
    X_temps = np.load(cfg.INPUT_TEMPS_PATH, allow_pickle=True)
    fm_samples = np.load(cfg.FM_NORMALIZED_PATH, allow_pickle=True)
    fm_294k = np.load(cfg.FM_294K_PATH)
    fm_lib = load_fm_library_if_available()
    mask = np.load(cfg.MASK_PATH)
    
    if fm_lib is None:
        raise RuntimeError("FM library required for DS3 is not available!")
    
    n_pairs = min(len(X_temps), len(fm_samples))
    print(f"Loaded {n_pairs} samples; fm_294k shape {fm_294k.shape}, mask shape {mask.shape}")

    # Build features and cell temps
    features = []
    cell_temps_list = []
    good_indices = []
    skipped = []

    for idx in range(n_pairs):
        try:
            feat, cell_temps = make_feature_from_input(X_temps[idx], cfg.N_CELLS)
        except Exception as e:
            skipped.append((idx, f"feature_err:{e}"))
            continue

        # Validate FM shape
        raw_fm = fm_samples[idx]
        try:
            fm_arr = np.asarray(raw_fm, dtype=float)
            if fm_arr.ndim == 1 and fm_arr.size == cfg.OUTPUT_SIZE:
                fm_arr = fm_arr.reshape(cfg.N_CELLS2, cfg.N_CELLS2)
            elif fm_arr.ndim == 2 and fm_arr.shape == (cfg.N_CELLS2, cfg.N_CELLS2):
                pass
            else:
                skipped.append((idx, f"fm_shape_wrong:{fm_arr.shape}"))
                continue
        except Exception as e:
            skipped.append((idx, f"fm_coercion:{e}"))
            continue

        features.append(feat.astype(np.float32))
        cell_temps_list.append(cell_temps)
        good_indices.append(idx)

    if len(skipped) > 0:
        print(f"[WARN] skipped {len(skipped)} samples (examples): {skipped[:10]}")

    if len(features) == 0:
        raise RuntimeError("No valid samples found after cleaning.")

    M = len(features)
    features = np.stack(features, axis=0)
    cell_temps_all = np.stack(cell_temps_list, axis=0)

    # Apply minmax normalization (DS3 uses minmax input)
    global_min = float(np.min(features))
    global_max = float(np.max(features))
    X_inputs = (features - global_min) / (global_max - global_min + 1e-12)

    # Build targets with row-wise interpolated reference and ln transform
    print("Building DS3 targets with row-wise interpolated reference...")
    target_ratios = np.zeros((M, cfg.OUTPUT_SIZE), dtype=float)
    for i in range(M):
        fm_arr = np.asarray(fm_samples[good_indices[i]], dtype=float)
        if fm_arr.ndim == 1:
            fm_arr = fm_arr.reshape(cfg.N_CELLS2, cfg.N_CELLS2)
        
        # Row-wise interpolated reference
        fm_ref_flat = interpolate_rowwise_reference(cell_temps_all[i], fm_lib, mask)
        fm_ref = fm_ref_flat.reshape(cfg.N_CELLS2, cfg.N_CELLS2) + cfg.EPSILON
        ratio = (fm_arr + cfg.EPSILON) / fm_ref
        ratio = np.clip(ratio, cfg.EPSILON, None)
        target_ratios[i, :] = ratio.flatten()

    # Apply ln transform
    Y_targets = np.log(target_ratios)

    print(f"Prepared DS3 dataset: features {X_inputs.shape}, targets {Y_targets.shape}")
    return X_inputs, Y_targets, fm_294k, good_indices, cell_temps_all, fm_lib, mask

# ---------------- Model class ----------------
class FMCoefficientPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.network(x)

# ---------------- Training function ----------------
def train_one_run(X, Y, seed, hidden_size, run_dir):
    """Train a single-model run and return trained model + metadata"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # train/val split
    n_samples = X.shape[0]
    n_val = int(n_samples * cfg.VALIDATION_SPLIT)
    n_train = n_samples - n_val
    perm = np.random.permutation(n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_X = X[train_idx]
    train_y = Y[train_idx]
    val_X = X[val_idx]
    val_y = Y[val_idx]

    # normalization
    train_mean = train_X.mean(axis=0)
    train_std = train_X.std(axis=0) + 1e-8
    train_X_norm = (train_X - train_mean) / train_std
    val_X_norm = (val_X - train_mean) / train_std

    # dataloaders
    train_ds = TensorDataset(torch.from_numpy(train_X_norm).float(), torch.from_numpy(train_y).float())
    val_ds = TensorDataset(torch.from_numpy(val_X_norm).float(), torch.from_numpy(val_y).float())
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # model, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FMCoefficientPredictor(cfg.INPUT_SIZE, hidden_size, cfg.OUTPUT_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                     patience=cfg.LR_REDUCE_PATIENCE, verbose=False)

    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    best_state = None

    # training loop
    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        # train
        model.train()
        running_loss = 0.0
        nb = 0
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            nb += 1
        train_loss = running_loss / max(1, nb)

        # validate
        model.eval()
        running_val_loss = 0.0
        nbv = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                running_val_loss += float(loss.item())
                nbv += 1
        val_loss = running_val_loss / max(1, nbv)

        scheduler.step(val_loss)
        
        # early stopping logic
        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.EARLY_STOP_PATIENCE:
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # save checkpoint
    if cfg.SAVE_CHECKPOINTS:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "train_mean": train_mean,
            "train_std": train_std,
            "config": {"INPUT_SIZE": cfg.INPUT_SIZE, "HIDDEN_SIZE": hidden_size, "OUTPUT_SIZE": cfg.OUTPUT_SIZE},
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss)
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, run_dir / f"checkpoint_hidden{hidden_size}.pth")

    return model, train_mean, train_std, best_val_loss, best_epoch

# ---------------- Evaluation (compute PCM per sample) ----------------
def power_iteration_user(fm, tol=1e-6, max_iter=500):
    fm = np.asarray(fm, dtype=float)
    n = fm.shape[0]
    v = np.ones((n, 1))
    eig_val = 1.0
    for i in range(max_iter):
        v_new = fm.dot(v)
        eig_new = float(np.max(np.abs(v_new)))
        if eig_new == 0:
            eig_val = 0.0
            v = v_new
            break
        v = v_new / eig_new
        if i > 0 and abs(eig_new - eig_val) < tol:
            eig_val = eig_new
            break
        eig_val = eig_new
    v = v.ravel()
    s = v.sum()
    if s != 0:
        v = v / s
    return float(eig_val), v

def reconstruct_full_matrix(cleaned_flat, mask):
    """
    Reconstruct full 289×289 matrix from cleaned 69696 flattened array.
    
    Args:
        cleaned_flat: (69696,) array of non-zero values
        mask: (289, 289) boolean mask indicating non-zero positions
    
    Returns:
        full_matrix: (289, 289) array with zeros restored
    """
    full_matrix = np.zeros((cfg.N_CELLS, cfg.N_CELLS), dtype=float)
    full_matrix[mask] = cleaned_flat
    return full_matrix

def evaluate_model_pcm(model, train_mean, train_std, features_all, fm_294k, good_indices, 
                       keff_array, fm_lib, cell_temps_all, mask, batch_size=32):
    """
    Run batched inference and compute PCM errors using DS3 structure.
    FIXED: Properly reconstructs full 289×289 matrix before computing k-eff.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, features_all.shape[0], batch_size):
            end = min(start + batch_size, features_all.shape[0])
            Xb = features_all[start:end]
            Xn = (Xb - train_mean) / train_std
            Xt = torch.from_numpy(Xn).float().to(device)
            out = model(Xt).cpu().numpy()
            preds.append(out)
    preds = np.vstack(preds)  # (M, OUTPUT_SIZE=69696)

    M = preds.shape[0]
    pcm_list = np.full(M, np.nan, dtype=float)
    kfm_list = np.full(M, np.nan, dtype=float)

    for local_i, global_idx in enumerate(good_indices):
        try:
            pred_transformed_flat = preds[local_i]  # (69696,)
            
            # Inverse ln transform
            ratio_flat = np.exp(pred_transformed_flat)
            ratio_mat = ratio_flat.reshape(cfg.N_CELLS2, cfg.N_CELLS2)

            # DS3: rowwise interpolated reference (cleaned space)
            cell_temps = cell_temps_all[local_i]
            fm_ref_flat = interpolate_rowwise_reference(cell_temps, fm_lib, mask)
            fm_ref_mat = fm_ref_flat.reshape(cfg.N_CELLS2, cfg.N_CELLS2)
            
            # Predicted cleaned matrix
            fm_pred_cleaned = ratio_mat * fm_ref_mat
            
            # RECONSTRUCT FULL 289×289 MATRIX
            fm_pred_full = reconstruct_full_matrix(fm_pred_cleaned.flatten(), mask)

            # Compute k-eff on FULL matrix
            k_pred, _ = power_iteration_user(fm_pred_full)
            kfm_list[local_i] = k_pred

            if keff_array is not None:
                if global_idx < keff_array.size:
                    k_true = keff_array[global_idx]
                    if np.isfinite(k_pred) and np.isfinite(k_true) and k_true != 0:
                        pcm = (k_pred / k_true - 1.0) * 1e5
                        pcm_list[local_i] = pcm
        except Exception as e:
            print(f"[WARN] Sample {local_i} (global {global_idx}) failed: {e}")
            continue

    # compute mean absolute PCM
    finite_pcm = pcm_list[np.isfinite(pcm_list)]
    if finite_pcm.size > 0:
        mean_abs_pcm = float(np.mean(np.abs(finite_pcm)))
    else:
        mean_abs_pcm = float("nan")

    return pcm_list, kfm_list, mean_abs_pcm

# ---------------- Main sweep orchestration ----------------
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = cfg.RESULTS_ROOT / f"sweep_ds3_hidden_{timestamp}"
    cfg.SWEEP_SUBDIR = sweep_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print("Sweep output directory:", sweep_dir)

    # Prepare DS3 data once
    X_all, Y_all, fm_294k, good_indices, cell_temps_all, fm_lib, mask = prepare_ds3_dataset()

    # Load keff array if available
    keff_array = None
    if cfg.KEFF_PATH.exists():
        try:
            keff_array = load_keff_values(cfg.KEFF_PATH)
            keff_array = keff_array[good_indices]  # align with good_indices
            print(f"Loaded keff array length {keff_array.size}")
        except Exception as e:
            print(f"[WARN] Failed to load KEFF array: {e}")
            keff_array = None
    else:
        print("[WARN] KEFF file not found; PCM comparisons will be NaN.")

    neuron_values = list(range(cfg.HIDDEN_START, cfg.HIDDEN_STOP - 1, cfg.HIDDEN_STEP))
    results_summary = []

    for hidden_size in neuron_values:
        print("\n" + "="*80)
        print(f"Hidden size: {hidden_size} — running {cfg.N_RUNS} trainings")
        run_metrics = []
        run_pcms = []
        run_kfms = []
        
        for run_idx in range(cfg.N_RUNS):
            seed = 44906 + hidden_size * 10 + run_idx  # base seed from best config
            print(f"\n--- Training run {run_idx+1}/{cfg.N_RUNS} (seed={seed}) ---")
            run_subdir = sweep_dir / f"hidden_{hidden_size}" / f"run_{run_idx+1}"
            
            try:
                model, train_mean, train_std, best_val_loss, best_epoch = train_one_run(
                    X_all, Y_all, seed, hidden_size, run_subdir
                )
            except Exception as e:
                print(f"[ERROR] training failed for hidden {hidden_size} run {run_idx+1}: {e}")
                traceback.print_exc()
                run_metrics.append(float("nan"))
                run_pcms.append(None)
                run_kfms.append(None)
                continue

            # Evaluate model
            print("Evaluating trained model for PCM...")
            pcm_arr, kfm_arr, mean_abs_pcm = evaluate_model_pcm(
                model, train_mean, train_std, X_all, fm_294k, good_indices, 
                keff_array, fm_lib, cell_temps_all, mask, batch_size=cfg.BATCH_SIZE
            )
            print(f"  Run {run_idx+1} mean-abs-PCM: {mean_abs_pcm:.3f} pcm")
            run_metrics.append(mean_abs_pcm)
            run_pcms.append(pcm_arr)
            run_kfms.append(kfm_arr)

            # Save per-run outputs
            run_subdir.mkdir(parents=True, exist_ok=True)
            np.save(run_subdir / "pcm_array.npy", pcm_arr)
            np.save(run_subdir / "kfm_array.npy", kfm_arr)
            meta = {
                "hidden_size": hidden_size,
                "run": run_idx+1,
                "seed": seed,
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss)
            }
            with open(run_subdir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

        # aggregate per-hidden-size
        run_metrics_np = np.array([v for v in run_metrics], dtype=float)
        valid = np.isfinite(run_metrics_np)
        if np.any(valid):
            mean_of_mean_abs_pcm = float(np.nanmean(run_metrics_np[valid]))
            std_of_mean_abs_pcm = float(np.nanstd(run_metrics_np[valid]))
        else:
            mean_of_mean_abs_pcm = float("nan")
            std_of_mean_abs_pcm = float("nan")

        print("\nSUMMARY for hidden_size =", hidden_size)
        print(f"  runs mean-abs-pcm values: {run_metrics}")
        print(f"  mean_of_mean_abs_pcm = {mean_of_mean_abs_pcm:.3f} pcm, std = {std_of_mean_abs_pcm:.3f} pcm")

        results_summary.append({
            "hidden_size": int(hidden_size),
            "mean_of_mean_abs_pcm": mean_of_mean_abs_pcm,
            "std_of_mean_abs_pcm": std_of_mean_abs_pcm,
            "per_run_mean_abs_pcm": run_metrics
        })

        # Save intermediate CSV
        csv_path = sweep_dir / "sweep_summary.csv"
        with open(csv_path, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["hidden_size", "mean_of_mean_abs_pcm", "std_of_mean_abs_pcm", "per_run_mean_abs_pcm"])
            for row in results_summary:
                writer.writerow([row["hidden_size"], row["mean_of_mean_abs_pcm"], 
                               row["std_of_mean_abs_pcm"], json.dumps(row["per_run_mean_abs_pcm"])])

    # Final reporting + plotting
    hs = [r["hidden_size"] for r in results_summary]
    means = [r["mean_of_mean_abs_pcm"] for r in results_summary]
    stds = [r["std_of_mean_abs_pcm"] for r in results_summary]

    # Save numeric arrays
    np.save(sweep_dir / "sweep_hidden_sizes.npy", np.array(hs))
    np.save(sweep_dir / "sweep_means.npy", np.array(means))
    np.save(sweep_dir / "sweep_stds.npy", np.array(stds))

    # Print table
    print("\n" + "="*80)
    print("FINAL DS3 SWEEP SUMMARY")
    print("="*80)
    print("hidden_size | mean_abs_pcm (mean of runs) | std_of_runs")
    for h, m, s in zip(hs, means, stds):
        print(f"{h:10d} | {np.nan if np.isnan(m) else m:20.3f} | {np.nan if np.isnan(s) else s:10.3f}")

    # Plot
    plt.figure(figsize=(10, 6))
    x = np.array(hs)
    y = np.array(means, dtype=float)
    yerr = np.array(stds, dtype=float)
    plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=4, linewidth=2, markersize=6)
    plt.xlabel("Hidden layer neurons", fontsize=12)
    plt.ylabel("Mean absolute PCM (pcm)", fontsize=12)
    plt.title("DS3 Model (Cleaned 264x264): mean_abs_pcm vs hidden layer size", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plot_path = sweep_dir / "sweep_pcm_vs_hidden.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nSaved plot to {plot_path}")

    # Save full summary JSON
    with open(sweep_dir / "sweep_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\nDS3 neuron sweep complete. Results directory:", sweep_dir)
    print(f"Trained on cleaned {cfg.N_CELLS2}x{cfg.N_CELLS2} matrices, evaluated on full {cfg.N_CELLS}x{cfg.N_CELLS}")

if __name__ == "__main__":
    main()