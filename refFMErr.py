#!/usr/bin/env python3
"""
print_comparison_table.py

Loads the interpolation baseline (ref_matrix_pcm_results.npz) and the
neural network results (fm_best_worst_pcm_results.npz) and prints a
LaTeX table comparing summary statistics for both methods.

Usage:
    python print_comparison_table.py
    python print_comparison_table.py --ref path/to/ref_matrix_pcm_results.npz
    python print_comparison_table.py --nn  path/to/fm_best_worst_pcm_results.npz
"""

import argparse
import numpy as np
from pathlib import Path

SIGMA = 45  # OpenMC stochastic uncertainty (pcm)


def compute_stats(pcm_raw: np.ndarray) -> dict:
    pcm     = pcm_raw[np.isfinite(pcm_raw)]
    n       = len(pcm)
    abs_pcm = np.abs(pcm)
    return {
        "n":        n,
        "mae":      np.mean(abs_pcm),
        "median":   np.median(abs_pcm),
        "std":      np.std(pcm),
        "min":      np.min(pcm),
        "max":      np.max(abs_pcm),
        "below2":   int(np.sum(abs_pcm <  2 * SIGMA)),
        "between":  int(np.sum((abs_pcm >= 2 * SIGMA) & (abs_pcm < 3 * SIGMA))),
        "above3":   int(np.sum(abs_pcm >= 3 * SIGMA)),
    }


def fmt_si(value: float, sign: bool = False) -> str:
    v = int(round(value))
    if sign:
        return rf"\SI{{{v:+d}}}{{pcm}}"
    return rf"\SI{{{v}}}{{pcm}}"


def fmt_band(count: int, n: int) -> str:
    pct = 100 * count / n
    n_fmt = f"{n:,}".replace(",", "{,}")
    c_fmt = f"{count:,}".replace(",", "{,}")
    return rf"{c_fmt} of {n_fmt} \quad ({pct:.1f}\%)"


def main(ref_npz: Path, nn_npz: Path):
    ref_data = np.load(ref_npz)
    nn_data  = np.load(nn_npz)

    r = compute_stats(ref_data["pcm_errors"])
    n = compute_stats(nn_data["all_pcm_errors"])

    table = rf"""\begin{{center}}
\centering
\captionof{{table}}{{Summary statistics for the row-wise interpolated reference fission matrix
baseline and the neural network correction, evaluated across all {r['n']:,} test samples.}}
\label{{tab:comparison_stats}}
\begin{{tabular}}{{lll}}
\hline
\textbf{{Metric}} & \textbf{{Interpolation}} & \textbf{{Neural Network}} \\
\hline
Mean absolute error         & {fmt_si(r['mae'])}    & {fmt_si(n['mae'])}   \\
Median absolute error       & {fmt_si(r['median'])} & {fmt_si(n['median'])}    \\
Standard deviation          & {fmt_si(r['std'])}    & {fmt_si(n['std'])}   \\
Minimum signed error        & {fmt_si(r['min'], sign=True)} & {fmt_si(n['min'], sign=True)} \\
Maximum absolute error      & {fmt_si(r['max'], sign=True)} & {fmt_si(n['max'], sign=True)} \\
\hline
Samples below 2 sigma           & {fmt_band(r['below2'],  r['n'])} & {fmt_band(n['below2'],  n['n'])} \\
Samples between 2 and 3 sigma   & {fmt_band(r['between'], r['n'])} & {fmt_band(n['between'], n['n'])} \\
Samples above 3 sigma           & {fmt_band(r['above3'],  r['n'])} & {fmt_band(n['above3'],  n['n'])} \\
\hline
\end{{tabular}}
\end{{center}}"""

    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="ref_matrix_pcm_results.npz",
                        help="Path to the interpolation baseline NPZ")
    parser.add_argument("--nn",  default="fm_best_worst_pcm_results.npz",
                        help="Path to the neural network results NPZ")
    args = parser.parse_args()
    main(Path(args.ref), Path(args.nn))