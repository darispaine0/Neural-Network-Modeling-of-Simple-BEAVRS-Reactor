#!/usr/bin/env python3
"""
plot_best_worst_temps.py

Loads fm_best_worst_pcm_results.npz and plots two rows of heatmaps:
  Row 0 – Fuel compact temperatures  (red  / YlOrRd)
  Row 1 – Moderator temperatures     (blue / YlGnBu)

Both rows show the single best and single worst sample (by |PCM|).

Usage:
    python plot_best_worst_temps.py
    python plot_best_worst_temps.py --npz path/to/results.npz
    python plot_best_worst_temps.py --save
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ── config ────────────────────────────────────────────────────────────────────
N_FUEL_COMPACT = 264
N_OTHER        = 289
N_ROWS         = 17
N_COLS         = 17

MASK_PATH = Path("training_data_nonzero_mask.npy")

BG_FIG      = "#ffffff"
BG_AX       = "#f6f8fa"
TEXT_MAIN   = "#1f2328"
TEXT_SUB    = "#57606a"
SPINE_COLOR = "#d0d7de"

FONT_SCALE  = 2.0         # scale all font sizes up or down (e.g. 1.5 = 50% larger)
FS          = 11 * FONT_SCALE  # base font size (pt) – all sizes relative to this
FUEL_CMAP   = "YlOrRd"
MOD_CMAP    = "YlGnBu"


# ── grid builders ─────────────────────────────────────────────────────────────
def load_npz(path: Path) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def fuel_to_grid(fuel_compact: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grid = np.full((N_ROWS, N_COLS), np.nan)
    flat_active = np.where(mask.any(axis=1))[0]
    if len(flat_active) != len(fuel_compact):
        raise ValueError(
            f"Active cells ({len(flat_active)}) != fuel_compact length "
            f"({len(fuel_compact)}). Check mask shape {mask.shape}."
        )
    for compact_i, flat_idx in enumerate(flat_active):
        grid[flat_idx // N_COLS, flat_idx % N_COLS] = fuel_compact[compact_i]
    return grid


def mod_to_grid(mod_temps: np.ndarray) -> np.ndarray:
    if len(mod_temps) != N_ROWS * N_COLS:
        raise ValueError(
            f"Moderator temp vector length ({len(mod_temps)}) != "
            f"{N_ROWS}×{N_COLS}={N_ROWS*N_COLS}."
        )
    return mod_temps.reshape(N_ROWS, N_COLS)


# ── heatmap helper ────────────────────────────────────────────────────────────
def plot_heatmap(ax, grid, line1, line2, vmin, vmax, cmap,
                 nan_color="#d0d7de"):
    """Draw the heatmap; two-line title avoids overflow."""
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal")

    nan_mask = np.isnan(grid)
    if nan_mask.any():
        ax.imshow(
            np.where(nan_mask, 1.0, np.nan),
            cmap=mcolors.ListedColormap([nan_color]),
            interpolation="nearest", aspect="equal",
            vmin=0, vmax=1,
        )

    # Two-line title: bold type on line 1, metadata on line 2
    ax.set_title(line1,
                 color=TEXT_MAIN, fontsize=FS * 1.05,
                 fontweight="bold", pad=8)


    ax.tick_params(colors=TEXT_SUB, labelsize=FS * 0.75)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COLOR)

    ax.set_xticks(range(N_COLS))
    ax.set_yticks(range(N_ROWS))
    ax.set_xticklabels(range(N_COLS), fontsize=FS * 0.70, color=TEXT_SUB)
    ax.set_yticklabels(range(N_ROWS), fontsize=FS * 0.70, color=TEXT_SUB)

    return im


def add_colorbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=TEXT_SUB, labelsize=FS * 0.80)
    cb.set_label(label, color=TEXT_SUB, fontsize=FS * 0.85)


# ── main ──────────────────────────────────────────────────────────────────────
def plot(npz_path: Path, save: bool):
    d = load_npz(npz_path)

    best_temps  = d["best_temps"][0]
    worst_temps = d["worst_temps"][0]
    best_pcm    = float(d["best_pcm"][0])
    worst_pcm   = float(d["worst_pcm"][0])
    best_idx    = int(d["best_indices"][0])
    worst_idx   = int(d["worst_indices"][0])

    best_fuel  = best_temps[:N_FUEL_COMPACT]
    worst_fuel = worst_temps[:N_FUEL_COMPACT]
    best_mod   = best_temps[N_FUEL_COMPACT:N_FUEL_COMPACT + N_OTHER]
    worst_mod  = worst_temps[N_FUEL_COMPACT:N_FUEL_COMPACT + N_OTHER]

    mask = np.load(MASK_PATH)

    best_fuel_grid  = fuel_to_grid(best_fuel,  mask)
    worst_fuel_grid = fuel_to_grid(worst_fuel, mask)
    best_mod_grid   = mod_to_grid(best_mod)
    worst_mod_grid  = mod_to_grid(worst_mod)

    fuel_vmin, fuel_vmax = (
        np.concatenate([best_fuel, worst_fuel]).min(),
        np.concatenate([best_fuel, worst_fuel]).max(),
    )
    mod_vmin, mod_vmax = (
        np.concatenate([best_mod, worst_mod]).min(),
        np.concatenate([best_mod, worst_mod]).max(),
    )

    # ── figure ────────────────────────────────────────────────────────────────
    # Each cell is 17×17 data points; we want them visually square.
    # Fix a cell size in inches and derive the figure size from it.
    CELL_IN  = 5.5    # inches per heatmap panel
    CB_IN    = 0.9    # inches for colorbar + gap (per column)
    L_IN     = 0.65  # left margin
    R_IN     = 0.20   # right margin
    T_IN     = 0.25   # top margin
    B_IN     = 0.50   # bottom margin
    HGAP_IN  = 0.90   # horizontal gap between columns
    VGAP_IN  = 1.40   # vertical gap between rows

    fig_w = L_IN + CELL_IN + CB_IN + HGAP_IN + CELL_IN + CB_IN + R_IN
    fig_h = T_IN + CELL_IN + VGAP_IN + CELL_IN + B_IN

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG_FIG)

    # Convert margins to figure fractions
    left   = L_IN  / fig_w
    right  = 1 - (CB_IN + R_IN) / fig_w
    top    = 1 - T_IN / fig_h
    bottom = B_IN  / fig_h

    gs = fig.add_gridspec(
        2, 2,
        left=left, right=right,
        top=top,   bottom=bottom,
        hspace=VGAP_IN / CELL_IN,
        wspace=(HGAP_IN + CB_IN) / CELL_IN,
    )

    ax_fb = fig.add_subplot(gs[0, 0])
    ax_fw = fig.add_subplot(gs[0, 1])
    ax_mb = fig.add_subplot(gs[1, 0])
    ax_mw = fig.add_subplot(gs[1, 1])

    for ax in (ax_fb, ax_fw, ax_mb, ax_mw):
        ax.set_facecolor(BG_AX)



    # ── fuel row ──────────────────────────────────────────────────────────────
    im_fb = plot_heatmap(
        ax_fb, best_fuel_grid,
        "Best Fuel Temperatures", "",
        fuel_vmin, fuel_vmax, FUEL_CMAP,
    )
    im_fw = plot_heatmap(
        ax_fw, worst_fuel_grid,
        "Worst Fuel Temperatures", "",
        fuel_vmin, fuel_vmax, FUEL_CMAP,
    )
    add_colorbar(fig, ax_fb, im_fb, "Fuel Temperature (K)")
    add_colorbar(fig, ax_fw, im_fw, "Fuel Temperature (K)")

    for ax in (ax_fb, ax_fw):
        ax.text(0.0, -0.08,
                "■  light grey = non-fuel / reflector cell",
                transform=ax.transAxes,
                fontsize=FS * 0.80, color=TEXT_SUB, va="top")

    # ── moderator row ─────────────────────────────────────────────────────────
    im_mb = plot_heatmap(
        ax_mb, best_mod_grid,
        "Best Moderator Temperatures", "",
        mod_vmin, mod_vmax, MOD_CMAP,
    )
    im_mw = plot_heatmap(
        ax_mw, worst_mod_grid,
        "Worst Moderator Temperatures", "",
        mod_vmin, mod_vmax, MOD_CMAP,
    )
    add_colorbar(fig, ax_mb, im_mb, "Moderator Temperature (K)")
    add_colorbar(fig, ax_mw, im_mw, "Moderator Temperature (K)")

    # ── output ────────────────────────────────────────────────────────────────
    if save:
        out = Path("best_worst_temps.png")
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {out}")
    else:
        plt.show()


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",  default="fm_best_worst_pcm_results.npz")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    plot(Path(args.npz), args.save)