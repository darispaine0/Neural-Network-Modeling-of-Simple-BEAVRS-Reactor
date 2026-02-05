#!/usr/bin/env python3
"""
Remove ALL zero entries from fission matrices.

An entry is kept ONLY if it is non-zero (>|EPSILON|)
across:
  - all reference temperatures
  - all sample matrices

Outputs:
  - cleaned reference vectors
  - cleaned sample dataset
  - global nonzero mask + indices
"""

import numpy as np
from pathlib import Path
import math

# ---------------- CONFIG ----------------
REFERENCE_TEMPS = list(range(300, 901, 100))
EPSILON = 1e-10

# ---------------- UTILITIES ----------------
def robust_load_square_fm(path, name="FM"):
    raw = np.load(path, allow_pickle=True)

    if isinstance(raw, np.ndarray) and raw.dtype == object:
        if raw.ndim == 0:
            raw = raw.item()
        elif raw.size == 1:
            raw = raw.flatten()[0]

    fm = np.asarray(raw, dtype=float)

    while fm.ndim > 2:
        fm = np.squeeze(fm)

    if fm.ndim == 1:
        side = int(math.isqrt(fm.size))
        if side * side != fm.size:
            raise ValueError(f"{name}: flat size {fm.size} not square")
        fm = fm.reshape(side, side)

    if fm.ndim != 2 or fm.shape[0] != fm.shape[1]:
        raise ValueError(f"{name}: invalid shape {fm.shape}")

    if not np.all(np.isfinite(fm)):
        raise ValueError(f"{name}: contains NaN or inf")

    return fm

# ---------------- MAIN CLEANING ----------------
def clean_fission_matrix_dataset(data_dir):
    data_path = Path(data_dir)

    print("=" * 80)
    print("GLOBAL ZERO-REMOVAL FISSION MATRIX CLEANING")
    print("=" * 80)

    # ---------- Load reference matrices ----------
    print("\nLoading reference matrices...")
    fm_refs = []

    for T in REFERENCE_TEMPS:
        path = data_path / f"fission_matrix_{T}k.npy"
        fm = robust_load_square_fm(path, name=f"FM_{T}K")
        fm_refs.append(fm)
        print(f"  Loaded FM_{T}K  shape={fm.shape}")

    fm_shape = fm_refs[0].shape
    n_total = fm_shape[0] * fm_shape[1]

    # ---------- Load samples ----------
    fm_samples_path = data_path / "output_fm_normalized_final.npy"
    fm_samples_raw = np.load(fm_samples_path, allow_pickle=True)

    fm_samples = []
    for i, fm in enumerate(fm_samples_raw):
        fm = np.asarray(fm, dtype=float)
        if fm.ndim == 1:
            fm = fm.reshape(fm_shape)
        if fm.shape != fm_shape:
            raise ValueError(f"Sample {i} shape mismatch {fm.shape}")
        fm_samples.append(fm)

    print(f"\nLoaded {len(fm_samples)} sample matrices")

    # ---------- Build global non-zero mask ----------
    print("\nBuilding GLOBAL non-zero mask (no zeros allowed)...")

    all_matrices = np.stack(fm_refs + fm_samples, axis=0)
    mask = np.all(np.abs(all_matrices) > EPSILON, axis=0)

    n_nonzero = int(mask.sum())
    n_zero = n_total - n_nonzero

    print(f"  Total elements: {n_total}")
    print(f"  Kept elements: {n_nonzero} ({100*n_nonzero/n_total:.2f}%)")
    print(f"  Removed elements: {n_zero} ({100*n_zero/n_total:.2f}%)")

    nonzero_indices = np.argwhere(mask)

    # ---------- Save mask ----------
    np.save(data_path / "global_nonzero_mask.npy", mask)
    np.save(data_path / "global_nonzero_indices.npy", nonzero_indices)

    # ---------- Clean reference matrices ----------
    print("\nCleaning reference matrices...")
    flat_mask = mask.flatten()

    for T, fm in zip(REFERENCE_TEMPS, fm_refs):
        fm_cleaned = fm.flatten()[flat_mask]
        out_path = data_path / f"fission_matrix_{T}k_cleaned.npy"
        np.save(out_path, fm_cleaned)
        print(f"  Saved {out_path.name}  shape=({fm_cleaned.size},)")

    # ---------- Clean sample matrices ----------
    print("\nCleaning sample matrices...")
    fm_samples_cleaned = np.zeros((len(fm_samples), n_nonzero), dtype=float)

    for i, fm in enumerate(fm_samples):
        fm_samples_cleaned[i] = fm.flatten()[flat_mask]

        if not np.all(np.isfinite(fm_samples_cleaned[i])):
            raise ValueError(f"NaN/Inf in cleaned sample {i}")

        if (i + 1) % 100 == 0 or i == len(fm_samples) - 1:
            print(f"  Processed {i+1}/{len(fm_samples)}")

    np.save(
        data_path / "output_fm_normalized_final_cleaned.npy",
        fm_samples_cleaned
    )

    # ---------- Verification ----------
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    test_idx = 0
    recon = np.zeros(n_total)
    recon[flat_mask] = fm_samples_cleaned[test_idx]
    recon = recon.reshape(fm_shape)

    diff = np.abs(recon[mask] - fm_samples[test_idx][mask])

    print(f"Max reconstruction error: {np.max(diff):.2e}")
    print(f"Max value in removed cells: {np.max(np.abs(recon[~mask])):.2e}")

    if np.max(diff) < 1e-10:
        print("✓ Reconstruction PASSED")
    else:
        print("✗ Reconstruction FAILED")

    # ---------- Summary ----------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original FM size: {fm_shape} ({n_total})")
    print(f"Final vector size: ({n_nonzero},)")
    print(f"Compression ratio: {n_nonzero/n_total:.4f}")
    print(f"Space saved: {100*(1-n_nonzero/n_total):.2f}%")

    print("\nFiles created:")
    print("  global_nonzero_mask.npy")
    print("  global_nonzero_indices.npy")
    for T in REFERENCE_TEMPS:
        print(f"  fission_matrix_{T}k_cleaned.npy")
    print("  output_fm_normalized_final_cleaned.npy")

# ---------------- RUN ----------------
if __name__ == "__main__":
    data_dir = "/home/daris/Monte_Carlo/Test/training_data2"
    clean_fission_matrix_dataset(data_dir)
