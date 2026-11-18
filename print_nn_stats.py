#!/usr/bin/env python3
"""
print_nn_stats.py

Computes the same summary statistics as the linearly interpolated baseline
table, but for the neural network predictions.

Usage:
    python print_nn_stats.py
    python print_nn_stats.py --npz path/to/results.npz
"""

import argparse
import numpy as np
from pathlib import Path


def main(npz_path: Path):
    data    = np.load(npz_path)
    all_pcm = data["all_pcm_errors"]

    # Drop any non-finite values
    pcm = all_pcm[np.isfinite(all_pcm)]
    n   = len(pcm)

    abs_pcm = np.abs(pcm)
    SIGMA   = 45  # OpenMC stochastic uncertainty (pcm)

    mae    = np.mean(abs_pcm)
    median = np.median(abs_pcm)
    std    = np.std(pcm)
    mn     = np.min(pcm)
    mx     = np.max(abs_pcm)

    below2  = np.sum(abs_pcm <  2 * SIGMA)
    between = np.sum((abs_pcm >= 2 * SIGMA) & (abs_pcm < 3 * SIGMA))
    above3  = np.sum(abs_pcm >= 3 * SIGMA)

    print(f"{'Metric':<40} {'Value'}")
    print("-" * 60)
    print(f"{'Mean absolute error':<40} {mae:.0f} pcm")
    print(f"{'Median absolute error':<40} {median:.0f} pcm")
    print(f"{'Standard deviation':<40} {std:.0f} pcm")
    print(f"{'Minimum signed error':<40} {mn:+.0f} pcm")
    print(f"{'Maximum absolute error':<40} {mx:+.0f} pcm")
    print("-" * 60)
    print(f"{'Samples below 2 sigma':<40} {below2} of {n:,}  ({100*below2/n:.1f}%)")
    print(f"{'Samples between 2 and 3 sigma':<40} {between} of {n:,}  ({100*between/n:.1f}%)")
    print(f"{'Samples above 3 sigma':<40} {above3} of {n:,}  ({100*above3/n:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="fm_best_worst_pcm_results.npz",
                        help="Path to the NPZ results file")
    args = parser.parse_args()
    main(Path(args.npz))