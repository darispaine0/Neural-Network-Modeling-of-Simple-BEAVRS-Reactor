#!/usr/bin/env python3
"""
ds3_sweep_fm_keff_loss_h5_tracked.py

Same as ds3_sweep_fm_keff_loss_h5.py, but tracks and plots the weighted
contribution of each loss term per epoch, and saves the data to Excel.

Loss contributions tracked per epoch (train and validation):
  - alpha_fm_mse  = ALPHA * MSE(ln-ratio FM)
  - beta_keff     = BETA  * mean(((k_pred - k_true) / k_true * 1e5)^2) / 1e10
  - gamma_source  = GAMMA * MSE(fission source vector)
  - total         = sum of above

All per-epoch data is saved to:
  <run_dir>/loss_history.xlsx   (one sheet per run: train + val weighted terms)

A multi-panel plot is also saved to:
  <run_dir>/loss_contributions.png
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import traceback
import random

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, Reference

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class GlobalConfig:
    H5_PATH   = Path("training_data.h5")
    MASK_PATH = Path("training_data_nonzero_mask.npy")

    N_ROWS   = 17
    N_COLS   = 17
    N_CELLS  = N_ROWS * N_COLS   # 289
    N_CELLS2 = 264               # cleaned

    N_FUEL_COMPACT = N_CELLS2
    INPUT_SIZE     = N_FUEL_COMPACT + N_CELLS   # 553
    OUTPUT_SIZE    = N_CELLS2 * N_CELLS2         # 69 696

    ALPHA = 0.00287
    BETA  = 0.31
    GAMMA = 7.69e-06

    DIFFERENTIABLE_POWER_ITER = True
    POWER_ITER_STEPS = 20

    BATCH_SIZE          = 32
    LEARNING_RATE       = 1e-04
    WEIGHT_DECAY        = 4.36e-08
    MAX_EPOCHS          = 1000
    VALIDATION_SPLIT    = 0.2
    EARLY_STOP_PATIENCE = 10
    LR_REDUCE_PATIENCE  = 5

    HIDDEN_SIZE = 1744
    N_RUNS       = 1

    RESULTS_ROOT     = Path("ds3_fm_keff_loss_results")
    SWEEP_SUBDIR     = None
    SAVE_CHECKPOINTS = True

    EPSILON = 1e-8

cfg = GlobalConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_fm_library_from_h5(h5file):
    ref_mats  = h5file["Reference Matrices"][:]
    ref_temps = h5file["reference_temperatures"][:]
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
    print("Loading data from training_data.h5...")
    with h5py.File(cfg.H5_PATH, "r") as h5:
        X_temps      = h5["temperatures"][:]
        fm_samples   = h5["fission_matrix"][:]
        keff_raw     = h5["keff"][:]
        source_raw   = h5["fission_source"][:]
        fm_lib       = load_fm_library_from_h5(h5)

    mask = np.load(cfg.MASK_PATH)
    n_pairs = min(X_temps.shape[0], fm_samples.shape[0], keff_raw.shape[0], source_raw.shape[0])
    print(f"Loaded {n_pairs} samples | mask {mask.shape}")

    features, fuel_temps_list, keff_list, source_list, good_indices, skipped = [], [], [], [], [], []

    for idx in range(n_pairs):
        k = keff_raw[idx]
        if not np.isfinite(k):
            skipped.append((idx, "invalid_keff")); continue
        feat = np.asarray(X_temps[idx], dtype=np.float32)
        if feat.shape[0] != cfg.INPUT_SIZE:
            skipped.append((idx, f"feat_shape:{feat.shape}")); continue
        fuel_compact = feat[:cfg.N_FUEL_COMPACT]
        fm_arr = np.asarray(fm_samples[idx], dtype=float)
        if fm_arr.size != cfg.OUTPUT_SIZE:
            skipped.append((idx, f"fm_shape:{fm_arr.shape}")); continue
        src_arr = np.asarray(source_raw[idx], dtype=np.float32)
        if src_arr.shape[0] != cfg.N_CELLS2:
            skipped.append((idx, f"source_shape:{src_arr.shape}")); continue
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
    Y_source       = np.stack(source_list,     axis=0)

    g_min    = 300
    g_max    = 900
    X_inputs = (features - g_min) / (g_max - g_min + 1e-12)

    print("Building ln-ratio FM targets...")
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
# Power iteration
# ─────────────────────────────────────────────────────────────────────────────

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
    v = v.ravel(); s = v.sum()
    if s != 0:
        v /= s
    return float(eig), v


def power_iter_torch(fm_tensor, n_steps):
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
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_full_matrix_torch(cleaned_flat_tensor, mask_tensor):
    full = torch.zeros(cfg.N_CELLS, cfg.N_CELLS,
                       dtype=cleaned_flat_tensor.dtype,
                       device=cleaned_flat_tensor.device)
    full[mask_tensor] = cleaned_flat_tensor
    return full


def combined_loss(fm_pred, fm_true, keff_true_batch,
                  fm_ref_batch, source_true_batch,
                  mask_tensor, row_active_tensor,
                  alpha, beta, gamma,
                  differentiable=False,
                  power_iter_steps=20):
    """
    Returns:
        total        : differentiable scalar tensor
        fm_mse_raw   : float — unweighted MSE(ln-ratio FM)
        keff_l_raw   : float — unweighted keff term (before BETA scaling)
        source_l_raw : float — unweighted source norm (before GAMMA scaling)
        alpha_fm     : float — weighted FM contribution  = ALPHA * fm_mse_raw
        beta_keff    : float — weighted keff contribution = BETA  * keff_l_raw
        gamma_src    : float — weighted source contribution = GAMMA * source_l_raw
    """
    fm_mse_t = nn.functional.mse_loss(fm_pred, fm_true)

    if beta == 0.0 and gamma == 0.0:
        wfm = alpha * float(fm_mse_t.detach())
        return (alpha * fm_mse_t,
                float(fm_mse_t.detach()), 0.0, 0.0,
                wfm, 0.0, 0.0)

    B      = fm_pred.shape[0]
    device = fm_pred.device

    if differentiable:
        fm_ref_t  = torch.from_numpy(fm_ref_batch).float().to(device)
        ratio_mat = torch.exp(fm_pred).reshape(B, cfg.N_CELLS2, cfg.N_CELLS2)
        fm_clean  = ratio_mat * fm_ref_t
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
    source_l_t = torch.linalg.vector_norm(v_pred_t - source_true_t)

    total = alpha * fm_mse_t + beta * keff_l_t + gamma * source_l_t

    alpha_fm  = alpha * float(fm_mse_t.detach())
    beta_keff = beta  * float(keff_l_t.detach())
    gamma_src = gamma * float(source_l_t.detach())

    return (total,
            float(fm_mse_t.detach()), float(keff_l_t.detach()), float(source_l_t.detach()),
            alpha_fm, beta_keff, gamma_src)


# ─────────────────────────────────────────────────────────────────────────────
# Excel export
# ─────────────────────────────────────────────────────────────────────────────

HEADER_FILL  = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT  = Font(name="Arial", bold=True, color="FFFFFF", size=10)
DATA_FONT    = Font(name="Arial", size=10)
THIN         = Side(style="thin", color="BFBFBF")
THIN_BORDER  = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

# Column palette for weighted terms (fill colors)
TERM_FILLS = {
    "tr_alpha_fm":   PatternFill("solid", fgColor="BDD7EE"),
    "tr_beta_keff":  PatternFill("solid", fgColor="FCE4D6"),
    "tr_gamma_src":  PatternFill("solid", fgColor="E2EFDA"),
    "va_alpha_fm":   PatternFill("solid", fgColor="9DC3E6"),
    "va_beta_keff":  PatternFill("solid", fgColor="F4B183"),
    "va_gamma_src":  PatternFill("solid", fgColor="A9D18E"),
}


def write_loss_excel(history: dict, run_dir: Path, hidden_size: int, run_idx: int,
                     alpha: float, beta: float, gamma: float):
    """
    Write per-epoch weighted loss contributions to an Excel workbook.

    history keys expected:
        epoch, lr,
        tr_total, tr_alpha_fm, tr_beta_keff, tr_gamma_src,
        va_total, va_alpha_fm, va_beta_keff, va_gamma_src
    """
    wb = openpyxl.Workbook()

    # ── Sheet 1: raw data ──────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Loss History"

    headers = [
        "Epoch", "LR",
        "Train Total", f"Train α·FM MSE (α={alpha})",
        f"Train β·keff (β={beta})", f"Train γ·Source (γ={gamma:.2e})",
        "Val Total", f"Val α·FM MSE (α={alpha})",
        f"Val β·keff (β={beta})", f"Val γ·Source (γ={gamma:.2e})",
    ]
    col_keys = [
        "epoch", "lr",
        "tr_total", "tr_alpha_fm", "tr_beta_keff", "tr_gamma_src",
        "va_total", "va_alpha_fm", "va_beta_keff", "va_gamma_src",
    ]

    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font      = HEADER_FONT
        cell.fill      = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border    = THIN_BORDER

    for r, row_data in enumerate(zip(*[history[k] for k in col_keys]), start=2):
        for c, val in enumerate(row_data, 1):
            cell        = ws.cell(row=r, column=c, value=round(float(val), 10))
            cell.font   = DATA_FONT
            cell.border = THIN_BORDER
            col_key     = col_keys[c - 1]
            if col_key in TERM_FILLS:
                cell.fill = TERM_FILLS[col_key]
            if col_key == "lr":
                cell.number_format = "0.00E+00"
            elif col_key == "epoch":
                cell.number_format = "0"
            else:
                cell.number_format = "0.00000000"

    # Auto-width
    for c, h in enumerate(headers, 1):
        ws.column_dimensions[get_column_letter(c)].width = max(len(h) * 0.9, 14)
    ws.freeze_panes = "A2"

    # ── Sheet 2: weighted contribution % per epoch ─────────────────────────
    ws2 = wb.create_sheet("Contribution %")
    hdrs2 = ["Epoch",
             "Train α·FM %", "Train β·keff %", "Train γ·Source %",
             "Val α·FM %",   "Val β·keff %",   "Val γ·Source %"]
    for c, h in enumerate(hdrs2, 1):
        cell = ws2.cell(row=1, column=c, value=h)
        cell.font      = HEADER_FONT
        cell.fill      = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border    = THIN_BORDER

    for r, ep in enumerate(history["epoch"], start=2):
        i   = r - 2
        ttr = history["tr_total"][i] + 1e-30
        tva = history["va_total"][i] + 1e-30
        row_vals = [
            ep,
            history["tr_alpha_fm"][i]  / ttr * 100,
            history["tr_beta_keff"][i] / ttr * 100,
            history["tr_gamma_src"][i] / ttr * 100,
            history["va_alpha_fm"][i]  / tva * 100,
            history["va_beta_keff"][i] / tva * 100,
            history["va_gamma_src"][i] / tva * 100,
        ]
        col_keys2 = [None, "tr_alpha_fm", "tr_beta_keff", "tr_gamma_src",
                     "va_alpha_fm", "va_beta_keff", "va_gamma_src"]
        for c, val in enumerate(row_vals, 1):
            cell        = ws2.cell(row=r, column=c, value=round(float(val), 4))
            cell.font   = DATA_FONT
            cell.border = THIN_BORDER
            ck = col_keys2[c - 1]
            if ck in TERM_FILLS:
                cell.fill = TERM_FILLS[ck]
            cell.number_format = "0.00" if c > 1 else "0"

    for c, h in enumerate(hdrs2, 1):
        ws2.column_dimensions[get_column_letter(c)].width = max(len(h) * 0.9, 14)
    ws2.freeze_panes = "A2"

    # ── Metadata sheet ─────────────────────────────────────────────────────
    ws3 = wb.create_sheet("Metadata")
    meta = [
        ("Parameter", "Value"),
        ("Hidden size", hidden_size),
        ("Run index", run_idx + 1),
        ("ALPHA (FM MSE weight)", alpha),
        ("BETA (keff weight)", beta),
        ("GAMMA (source weight)", gamma),
        ("Differentiable power iter", str(cfg.DIFFERENTIABLE_POWER_ITER)),
        ("Power iter steps", cfg.POWER_ITER_STEPS),
        ("Batch size", cfg.BATCH_SIZE),
        ("Learning rate (initial)", cfg.LEARNING_RATE),
        ("Weight decay", cfg.WEIGHT_DECAY),
        ("Max epochs", cfg.MAX_EPOCHS),
        ("Early stop patience", cfg.EARLY_STOP_PATIENCE),
        ("LR reduce patience", cfg.LR_REDUCE_PATIENCE),
    ]
    for r, (k, v) in enumerate(meta, 1):
        kc = ws3.cell(row=r, column=1, value=k)
        vc = ws3.cell(row=r, column=2, value=v)
        kc.font = DATA_FONT; vc.font = DATA_FONT
        if r == 1:
            kc.font = HEADER_FONT; kc.fill = HEADER_FILL
            vc.font = HEADER_FONT; vc.fill = HEADER_FILL
    ws3.column_dimensions["A"].width = 32
    ws3.column_dimensions["B"].width = 20

    out_path = run_dir / f"loss_history_hidden{hidden_size}_run{run_idx+1}.xlsx"
    wb.save(str(out_path))
    print(f"  [Excel] Loss history saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_contributions(history: dict, run_dir: Path,
                            hidden_size: int, run_idx: int,
                            alpha: float, beta: float, gamma: float):
    """
    4-panel figure showing weighted loss contributions per epoch.
    Panel 1: Total loss (train vs val)
    Panel 2: α·FM MSE (train vs val)
    Panel 3: β·keff term (train vs val)
    Panel 4: γ·source term (train vs val)
    """
    epochs = history["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Weighted Loss Contributions per Epoch\n"
        f"Hidden={hidden_size}, Run {run_idx+1}   "
        f"α={alpha}, β={beta}, γ={gamma:.2e}",
        fontsize=13, fontweight="bold", y=1.01
    )

    panels = [
        ("Total Loss",                     "tr_total",     "va_total",     "#1F4E79", "#2E75B6"),
        (f"α·FM MSE  (α={alpha})",         "tr_alpha_fm",  "va_alpha_fm",  "#843C0C", "#ED7D31"),
        (f"β·keff term  (β={beta})",       "tr_beta_keff", "va_beta_keff", "#375623", "#70AD47"),
        (f"γ·source term  (γ={gamma:.2e})","tr_gamma_src", "va_gamma_src", "#4D1C7A", "#9DC3E6"),
    ]

    for ax, (title, tr_key, va_key, tr_color, va_color) in zip(axes.flatten(), panels):
        tr_vals = history[tr_key]
        va_vals = history[va_key]
        ax.plot(epochs, tr_vals, color=tr_color, linewidth=1.6, label="Train")
        ax.plot(epochs, va_vals, color=va_color,  linewidth=1.6, label="Validation",
                linestyle="--")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Loss contribution", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        # Log scale if values span more than 2 orders of magnitude
        combined = [v for v in tr_vals + va_vals if v > 0]
        if combined and max(combined) / (min(combined) + 1e-30) > 100:
            ax.set_yscale("log")

    fig.tight_layout()
    out_path = run_dir / f"loss_contributions_hidden{hidden_size}_run{run_idx+1}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot]  Loss contribution figure saved → {out_path}")

    # Second figure: stacked area chart of % contribution (train only)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    tr_total = np.array(history["tr_total"]) + 1e-30
    fm_pct   = np.array(history["tr_alpha_fm"])  / tr_total * 100
    keff_pct = np.array(history["tr_beta_keff"]) / tr_total * 100
    src_pct  = np.array(history["tr_gamma_src"]) / tr_total * 100
    ax2.stackplot(epochs, fm_pct, keff_pct, src_pct,
                  labels=[f"α·FM MSE (α={alpha})",
                          f"β·keff (β={beta})",
                          f"γ·source (γ={gamma:.2e})"],
                  colors=["#BDD7EE", "#F4B183", "#A9D18E"], alpha=0.85)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("% of total training loss", fontsize=10)
    ax2.set_title(f"Weighted Loss Term Contribution (%) — Train\n"
                  f"Hidden={hidden_size}, Run {run_idx+1}", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    out_path2 = run_dir / f"loss_stacked_pct_hidden{hidden_size}_run{run_idx+1}.png"
    fig2.savefig(str(out_path2), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [Plot]  Stacked % contribution figure saved → {out_path2}")


# ─────────────────────────────────────────────────────────────────────────────
# Reference FM cache
# ─────────────────────────────────────────────────────────────────────────────

def build_ref_fm_cache(fuel_temps_all, fm_lib, mask):
    M   = fuel_temps_all.shape[0]
    out = np.zeros((M, cfg.N_CELLS2, cfg.N_CELLS2), dtype=np.float32)
    for i in range(M):
        ref = interpolate_rowwise_reference(fuel_temps_all[i], fm_lib, mask)
        out[i] = ref.reshape(cfg.N_CELLS2, cfg.N_CELLS2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_run(X, Y_fm, Y_source, Y_keff, ref_fm_cache,
                  seed, hidden_size, run_dir, mask_np, run_idx):
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

    model     = FMCoefficientPredictor(cfg.INPUT_SIZE, hidden_size, cfg.OUTPUT_SIZE).to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=cfg.LR_REDUCE_PATIENCE, verbose=False
    )

    best_val, best_ep, no_improve, best_state = float('inf'), -1, 0, None

    # ── Per-epoch history for plotting/Excel ──────────────────────────────
    history = {k: [] for k in [
        "epoch", "lr",
        "tr_total", "tr_alpha_fm", "tr_beta_keff", "tr_gamma_src",
        "va_total", "va_alpha_fm", "va_beta_keff", "va_gamma_src",
    ]}

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        accum = {k: 0.0 for k in
                 ["total", "alpha_fm", "beta_keff", "gamma_src"]}
        nb = 0
        for bx, bfm, bk, bidx in tr_loader:
            bx   = bx.to(device)
            bfm  = bfm.to(device)
            bk   = bk.to(device)
            b_ref_np = tr_ref[bidx.numpy()]
            b_src_np = tr_src[bidx.numpy()]

            optimizer.zero_grad()
            fm_p = model(bx)
            loss, fl, kl, sl, wfm, wkeff, wsrc = combined_loss(
                fm_p, bfm, bk, b_ref_np, b_src_np, mask_tensor, row_active_tensor,
                cfg.ALPHA, cfg.BETA, cfg.GAMMA,
                differentiable=cfg.DIFFERENTIABLE_POWER_ITER,
                power_iter_steps=cfg.POWER_ITER_STEPS,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            accum["total"]    += loss.item()
            accum["alpha_fm"] += wfm
            accum["beta_keff"]+= wkeff
            accum["gamma_src"]+= wsrc
            nb += 1
        nb = max(1, nb)
        tr_tot  = accum["total"]    / nb
        tr_wfm  = accum["alpha_fm"] / nb
        tr_wk   = accum["beta_keff"]/ nb
        tr_ws   = accum["gamma_src"]/ nb

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        vaccum = {k: 0.0 for k in
                  ["total", "alpha_fm", "beta_keff", "gamma_src"]}
        nbv = 0
        with torch.no_grad():
            for bx, bfm, bk, bidx in va_loader:
                bx  = bx.to(device)
                bfm = bfm.to(device)
                bk  = bk.to(device)
                b_ref_np = va_ref[bidx.numpy()]
                b_src_np = va_src[bidx.numpy()]
                fm_p = model(bx)
                loss, fl, kl, sl, wfm, wkeff, wsrc = combined_loss(
                    fm_p, bfm, bk, b_ref_np, b_src_np, mask_tensor, row_active_tensor,
                    cfg.ALPHA, cfg.BETA, cfg.GAMMA,
                    differentiable=False,
                    power_iter_steps=cfg.POWER_ITER_STEPS,
                )
                vaccum["total"]    += loss.item()
                vaccum["alpha_fm"] += wfm
                vaccum["beta_keff"]+= wkeff
                vaccum["gamma_src"]+= wsrc
                nbv += 1
        nbv = max(1, nbv)
        va_tot = vaccum["total"]    / nbv
        va_wfm = vaccum["alpha_fm"] / nbv
        va_wk  = vaccum["beta_keff"]/ nbv
        va_ws  = vaccum["gamma_src"]/ nbv

        scheduler.step(va_tot)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["epoch"].append(epoch)
        history["lr"].append(current_lr)
        history["tr_total"].append(tr_tot)
        history["tr_alpha_fm"].append(tr_wfm)
        history["tr_beta_keff"].append(tr_wk)
        history["tr_gamma_src"].append(tr_ws)
        history["va_total"].append(va_tot)
        history["va_alpha_fm"].append(va_wfm)
        history["va_beta_keff"].append(va_wk)
        history["va_gamma_src"].append(va_ws)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"    Ep {epoch:4d} | "
                f"tr[total={tr_tot:.5f}  α·FM={tr_wfm:.5f}  β·keff={tr_wk:.5f}  γ·src={tr_ws:.5f}] | "
                f"va[total={va_tot:.5f}  α·FM={va_wfm:.5f}  β·keff={va_wk:.5f}  γ·src={va_ws:.5f}] | "
                f"lr={current_lr:.2e}"
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

    run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.SAVE_CHECKPOINTS:
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

    # ── Save Excel + plots ─────────────────────────────────────────────────
    write_loss_excel(history, run_dir, hidden_size, run_idx,
                     cfg.ALPHA, cfg.BETA, cfg.GAMMA)
    plot_loss_contributions(history, run_dir, hidden_size, run_idx,
                            cfg.ALPHA, cfg.BETA, cfg.GAMMA)

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
    src_mae_list = np.full(M, np.nan)
    row_active = mask.any(axis=1)

    for local_i, global_idx in enumerate(good_indices):
        try:
            ratio_mat     = np.exp(preds[local_i]).reshape(cfg.N_CELLS2, cfg.N_CELLS2)
            fm_ref_flat   = interpolate_rowwise_reference(fuel_temps_all[local_i], fm_lib, mask)
            fm_pred_clean = ratio_mat * fm_ref_flat.reshape(cfg.N_CELLS2, cfg.N_CELLS2)
            fm_pred_full  = np.zeros((cfg.N_CELLS, cfg.N_CELLS), dtype=float)
            fm_pred_full[mask] = fm_pred_clean.flatten()
            k_pred, v_pred_full = power_iter_numpy(fm_pred_full)
            v_pred_compact = v_pred_full[row_active]
            kfm_list[local_i] = k_pred

            if keff_full is not None and global_idx < keff_full.size:
                k_true = keff_full[global_idx]
                if np.isfinite(k_pred) and np.isfinite(k_true) and k_true != 0:
                    pcm_list[local_i] = (k_pred / k_true - 1.0) * 1e5

            if source_true_all is not None:
                src_mae_list[local_i] = float(np.linalg.norm(
                    v_pred_compact - source_true_all[local_i]))
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

    print("Pre-building reference FM cache for all samples...")
    ref_fm_cache = build_ref_fm_cache(fuel_temps_all, fm_lib, mask)
    print(f"ref_fm_cache shape: {ref_fm_cache.shape}  "
          f"({ref_fm_cache.nbytes / 1e9:.2f} GB)")

    keff_full = None
    try:
        with h5py.File(cfg.H5_PATH, "r") as h5:
            keff_full = h5["keff"][:]
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
                seed, hidden_size, run_subdir, mask, run_idx
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