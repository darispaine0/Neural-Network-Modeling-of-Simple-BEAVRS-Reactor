#!/usr/bin/env python3
"""
plot_pcm_distribution.py

Plots |pcm| error for all samples sorted best to worst (primary y-axis),
with secondary axes showing rolling average of k_eff and rolling average
of temperature std.

Usage:
    python plot_pcm_distribution.py
    python plot_pcm_distribution.py --npz path/to/results.npz
    python plot_pcm_distribution.py --h5  path/to/training_data.h5
    python plot_pcm_distribution.py --save
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import h5py
from pathlib import Path


# ── light theme palette ───────────────────────────────────────────────────────
BG_FIG   = "#ffffff"
BG_AX    = "#f6f8fa"
BORDER   = "#d0d7de"
TEXT_MAIN= "#1f2328"
TEXT_SUB = "#57606a"
ACCENT   = "#0969da"
PURPLE   = "#8250df"
ORANGE   = "#bc4c00"

FONT_SCALE = 3


def rolling(arr, window):
    return np.convolve(arr, np.ones(window) / window, mode="same")


def main(npz_path: Path, h5_path: Path, save: bool):

    # ── load data ─────────────────────────────────────────────────────────────
    data        = np.load(npz_path)
    all_pcm     = data["all_pcm_errors"]
    all_ktrue   = data["all_k_trues"]
    finite_mask = np.isfinite(all_pcm)
    finite_pcm  = all_pcm[finite_mask]
    abs_pcm     = np.abs(finite_pcm)
    ktrue       = all_ktrue[finite_mask]

    print(f"Loading temperatures from {h5_path} ...")
    with h5py.File(h5_path, "r") as h5:
        temps = h5["temperatures"][:]
    temp_stds = np.std(temps[finite_mask], axis=1)

    # ── sort by |pcm| ascending ───────────────────────────────────────────────
    order      = np.argsort(abs_pcm)
    sorted_pcm = abs_pcm[order]
    sorted_kt  = ktrue[order]
    sorted_std = temp_stds[order]
    ranks      = np.arange(1, len(sorted_pcm) + 1)


    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(20, 9))
    fig.patch.set_facecolor(BG_FIG)
    fig.subplots_adjust(right=0.75)

    ax1.set_facecolor(BG_AX)
    for sp in ax1.spines.values():
        sp.set_edgecolor(BORDER)
    ax1.tick_params(colors=TEXT_SUB, labelsize=9 * FONT_SCALE)
    ax1.xaxis.label.set_color(TEXT_SUB)
    ax1.yaxis.label.set_color(TEXT_SUB)
    ax1.grid(True, color=BORDER, linewidth=0.5, linestyle="--", alpha=0.7)

    # ── primary: |pcm| scatter ────────────────────────────────────────────────
    ax1.scatter(ranks, sorted_pcm, c=ACCENT, s=4, alpha=0.5,
                linewidths=0, zorder=3, label="absolute pcm error")
    ax1.set_xlabel("Sample rank (best → worst)", color=TEXT_SUB,
                   fontsize=9 * FONT_SCALE)
    ax1.set_ylabel("Absolute pcm error", color=TEXT_SUB, fontsize=9 * FONT_SCALE)
    ax1.set_xlim(0, len(ranks) + 1)
    ax1.set_ylim(bottom=0)

    # ── error-band shading (transparent) ──────────────────────────────────────
    y_top = ax1.get_ylim()[1]
    ax1.axhspan(0,   90,    color="green",  alpha=0.15, zorder=0, linewidth=0)
    ax1.axhspan(90,  135,   color="yellow", alpha=0.25, zorder=0, linewidth=0)
    ax1.axhspan(135, y_top, color="red",    alpha=0.12, zorder=0, linewidth=0)
    ax1.set_ylim(0, y_top)

    # ── secondary: k_eff raw scatter ─────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.set_facecolor("none")
    ax2.scatter(ranks, sorted_kt, c=PURPLE, s=2, alpha=0.4,
                linewidths=0, zorder=2)
    ax2.set_ylim(0, 1.3)
    ax2.set_ylabel(r"$k_\mathrm{eff}$", color=PURPLE, fontsize=9 * FONT_SCALE)
    ax2.tick_params(colors=PURPLE, labelsize=9 * FONT_SCALE)
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.2f}")
    )
    ax2.spines["right"].set_position(("axes", 1.02))
    for sp in ax2.spines.values():
        sp.set_edgecolor(BORDER)

    # ── tertiary: temperature std raw scatter ─────────────────────────────────
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.17))
    ax3.set_facecolor("none")
    ax3.scatter(ranks, sorted_std, c=ORANGE, s=2, alpha=0.4,
                linewidths=0, zorder=2)
    ax3.set_ylim(-200, 300)
    ax3.set_ylabel("Temperature std (K)", color=ORANGE, fontsize=9 * FONT_SCALE)
    ax3.tick_params(colors=ORANGE, labelsize=9 * FONT_SCALE)
    for sp in ax3.spines.values():
        sp.set_edgecolor(BORDER)
    ax3.spines["right"].set_edgecolor(ORANGE)

    # ── legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT,
               markersize=8, label="Absolute pcm error"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PURPLE,
               markersize=8, label=r"$k_\mathrm{eff}$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE,
               markersize=8, label="Temperature std (K)"),
        Patch(facecolor="green",  alpha=0.15, label="< 90 pcm"),
        Patch(facecolor="yellow", alpha=0.25, label="90–135 pcm"),
        Patch(facecolor="red",    alpha=0.12, label="> 135 pcm"),
    ]
    ax1.legend(handles=legend_elements, fontsize=7.5 * FONT_SCALE,
               facecolor=BG_FIG, labelcolor=TEXT_MAIN,
               edgecolor=BORDER, loc="upper left",
               bbox_to_anchor=(0.0, 0.7))

    ax1.set_title(
        rf""
        f"",
        color=TEXT_MAIN, fontsize=12 * FONT_SCALE, fontweight="bold"
    )

    if save:
        out = Path("pcm_distribution.png")
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="fm_best_worst_pcm_results.npz")
    parser.add_argument("--h5",  default="training_data.h5")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    main(Path(args.npz), Path(args.h5), args.save)