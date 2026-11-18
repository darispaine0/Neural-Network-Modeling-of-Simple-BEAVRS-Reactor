#!/usr/bin/env python3
"""
check_temp_std.py

Computes the standard deviation of each temperature vector in the dataset
and reports the minimum, maximum, mean, and distribution.
"""

import numpy as np
import h5py
from pathlib import Path


H5_PATH = Path("training_data.h5")


def main():
    print("Loading temperature vectors ...")
    with h5py.File(H5_PATH, "r") as h5:
        temps = h5["temperatures"][:]   # (N, 553)

    N = temps.shape[0]
    print(f"  {N} samples, {temps.shape[1]} features each\n")

    stds = np.std(temps, axis=1)   # (N,) — one std per sample

    min_idx = int(np.argmin(stds))
    max_idx = int(np.argmax(stds))

    sep = "=" * 60
    print(sep)
    print("TEMPERATURE VECTOR STD DEVIATION ACROSS DATASET")
    print(sep)
    print(f"  Min  std : {stds.min():.4f} K  (sample {min_idx})")
    print(f"  Max  std : {stds.max():.4f} K  (sample {max_idx})")
    print(f"  Mean std : {stds.mean():.4f} K")
    print(f"  Median   : {np.median(stds):.4f} K")
    print(f"  Std of stds : {stds.std():.4f} K")

    print(f"\n  Sample with MIN std ({stds.min():.4f} K):")
    t = temps[min_idx]
    print(f"    mean={t.mean():.2f} K  min={t.min():.2f} K  max={t.max():.2f} K")
    print(f"    first 10 values: {t[:10].round(2)}")

    print(f"\n  Sample with MAX std ({stds.max():.4f} K):")
    t = temps[max_idx]
    print(f"    mean={t.mean():.2f} K  min={t.min():.2f} K  max={t.max():.2f} K")
    print(f"    first 10 values: {t[:10].round(2)}")

    print(f"\n  Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    {p:3d}th : {np.percentile(stds, p):.4f} K")


if __name__ == "__main__":
    main()