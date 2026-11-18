#!/usr/bin/env python3
"""
plot_ref_temp_bars.py

Runs the trained model against every reference fission matrix
(300, 350, 400, ... 900 K) using a uniform assembly input at each
temperature, then plots |pcm| error as a bar chart vs reference temperature.

Usage:
    python plot_ref_temp_bars.py
    python plot_ref_temp_bars.py --save
"""

import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path


# ── palette (matches plot_pcm_distribution.py) ────────────────────────────────
BG_FIG    = "#ffffff"
BG_AX     = "#ffffff"
BORDER    = "#d0d7de"
TEXT_MAIN = "#1f2328"
TEXT_SUB  = "#57606a"

FONT_SCALE = 2.5


# ── config ────────────────────────────────────────────────────────────────────
class Config:
    H5_PATH         = Path("training_data.h5")
    CHECKPOINT_PATH = Path(
        "ds3_fm_keff_loss_results/"
        "train_fm_keff_loss_20261010_120627/"
        "hidden_1744/run_1/checkpoint_hidden1744.pth"
    )
    MASK_PATH   = Path(
        "training_data_nonzero_mask.npy"
    )
    N_CELLS     = 289
    N_CELLS2    = 264
    INPUT_SIZE  = 553
    OUTPUT_SIZE = 69696
    EPSILON     = 1e-8
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()


# ── model ─────────────────────────────────────────────────────────────────────
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


# ── power iteration ───────────────────────────────────────────────────────────
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
    if s:
        v /= s
    return float(eig), v


# ── inference ─────────────────────────────────────────────────────────────────
def run_inference():
    print(f"Loading checkpoint: {cfg.CHECKPOINT_PATH}")
    ckpt = torch.load(cfg.CHECKPOINT_PATH, map_location=cfg.DEVICE,
                      weights_only=False)
    model = FMCoefficientPredictor(
        cfg.INPUT_SIZE, ckpt["config"]["HIDDEN_SIZE"], cfg.OUTPUT_SIZE
    ).to(cfg.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mu  = np.asarray(ckpt["train_mean"], dtype=np.float32)
    sig = np.asarray(ckpt["train_std"],  dtype=np.float32)

    mask       = np.load(cfg.MASK_PATH)
    row_active = mask.any(axis=1)

    def to_full(fm_c):
        f = np.zeros((cfg.N_CELLS, cfg.N_CELLS))
        f[mask] = fm_c.flatten()
        return f

    with h5py.File(cfg.H5_PATH, "r") as h5:
        ref_mats  = h5["Reference Matrices"][:]
        ref_temps = h5["reference_temperatures"][:]

    pairs = sorted(zip(ref_temps.astype(int).tolist(), ref_mats),
                   key=lambda x: x[0])

    rows = []
    for temp_k, fm_flat in pairs:
        fm_ref  = fm_flat.astype(np.float64).reshape(cfg.N_CELLS2, cfg.N_CELLS2)
        x_raw   = np.full(cfg.INPUT_SIZE, float(temp_k), dtype=np.float32)
        x_range = (x_raw - 300.0) / (600.0 + 1e-12)
        x_norm  = ((x_range - mu) / sig).astype(np.float32)

        with torch.no_grad():
            ln_ratio = model(
                torch.from_numpy(x_norm).unsqueeze(0).to(cfg.DEVICE)
            ).cpu().numpy()[0]

        fm_pred = np.exp(ln_ratio).reshape(cfg.N_CELLS2, cfg.N_CELLS2) * \
                  (fm_ref + cfg.EPSILON)

        k_ref,  v_ref  = power_iter_numpy(to_full(fm_ref))
        k_pred, v_pred = power_iter_numpy(to_full(fm_pred))
        pcm = (k_pred / k_ref - 1.0) * 1e5

        print(f"  T = {temp_k} K  |PCM| = {abs(pcm):.3f}")
        rows.append({"T_K": temp_k, "pcm": pcm})

    return rows


# ── plot ──────────────────────────────────────────────────────────────────────
def plot(rows: list[dict], save: bool):
    temps   = [r["T_K"]      for r in rows]
    abs_pcm = [abs(r["pcm"]) for r in rows]
    def bar_color(v):
        if v < 90:
            return "#2da44e"   # green
        if v <= 135:
            return "#d4a72c"   # yellow
        return "#cf222e"       # red
    colors  = [bar_color(v) for v in abs_pcm]

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(BG_FIG)
    ax.set_facecolor(BG_AX)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT_SUB, labelsize=9 * FONT_SCALE)
    ax.grid(True, axis="y", color=BORDER, linewidth=0.5,
            linestyle="--", alpha=0.7, zorder=0)

    bars = ax.bar(
        [str(t) for t in temps], abs_pcm,
        color=colors, edgecolor=BORDER, linewidth=0.8, zorder=3
    )

    # value labels above each bar
    y_max = max(abs_pcm) if abs_pcm else 1.0
    for bar, val in zip(bars, abs_pcm):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.01,
            f"{val:.1f}",
            ha="center", va="bottom",
            fontsize=7 * FONT_SCALE, color=TEXT_MAIN
        )

    ax.set_xlabel("Reference temperature (K)", color=TEXT_SUB,
                  fontsize=9 * FONT_SCALE)
    ax.set_ylabel("absolute pcm error", color=TEXT_SUB, fontsize=9 * FONT_SCALE)
    ax.set_ylim(bottom=0, top=y_max * 1.15)

    fig.tight_layout()

    if save:
        out = Path("pcm_ref_temp_bars.png")
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    rows = run_inference()
    plot(rows, args.save)