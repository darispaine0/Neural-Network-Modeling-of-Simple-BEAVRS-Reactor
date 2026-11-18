#!/usr/bin/env python3
"""
build_h5.py

Builds training_data.h5 straight from the per-batch folders written by the
data generator, one batch at a time, so the full dataset never has to fit in
memory. This replaces the old chain

    FM_normalizer.py -> temp_cleaner.py -> consolidate_data.py -> FM_cleaner.py
    -> (scratch.py h5 scripts)

with a single script. Pipeline is now:

    ExampleDataGenerator.py -> build_h5.py -> nnSource.py

How it works
------------
Every sample FM is normalized from the raw tallies (output_tallies_raw.npy)
with the same formula as the data generator and FM_normalizer.py:
    FM[:, j] = raw[:, j] / sum(raw[j, :]) * sum(raw)
(columns whose row sum is zero are left as zeros).

Pass 1 (mask):  build the global non-zero mask by ANDing |FM| > EPSILON over
                the 13 reference FMs and every normalized sample FM, one batch
                at a time (same rule as FM_cleaner.py). Skipped if --mask is given.
Pass 2 (write): for each batch, normalize and trim every FM with the mask,
                pack the fuel temperatures 289 -> 264 (same rule as
                temp_cleaner.py), run power iteration for keff + fission
                source, and append the rows to the h5 file.

Batch folders are every folder under --data-dir that contains
output_tallies_raw.npy, sorted by path (same ordering as consolidate_data.py).

Inputs
------
    <data-dir>/**/output_tallies_raw.npy     per batch, N raw FM tallies (289*289)
    <data-dir>/**/input_temps_clean.npy      per batch, used if present,
      or input_temps.npy                     otherwise (packed on the fly)
    <data-dir>/**/output_keff.npy            per batch, OpenMC track-length k-eff
    <data-dir>/**/output_keff_uncertainty.npy  per batch, its 1-sigma std dev
    <ref-dir>/fission_matrix_{T}k.npy        T = 300..900 step 50, (289, 289),
                                             used as-is (NOT re-normalized)

Outputs
-------
    <out>                          the h5 file (schema below)
    <out stem>_nonzero_mask.npy    the (289, 289) mask used to trim the FMs,
                                   written next to the h5 file

    temperatures           (N, 553)    float32  [264 compact fuel | 289 other]
    fission_matrix         (N, 69696)  float64  trimmed 264x264 FM, flattened
    fission_source         (N, 264)    float32  normalized dominant eigenvector
    keff                   (N,)        float32  dominant eigenvalue of the FM
    keff_tracklength       (N,)        float32  OpenMC track-length k-eff
    keff_tracklength_std   (N,)        float32  its 1-sigma std dev
    Reference Matrices     (13, 69696) float64
    reference_temperatures (13,)       int32
    nonzero_mask           (289, 289)  bool     copy of the mask

Note: 'keff' and 'fission_source' come from power iteration on the trimmed FM
(as the original scripts did). The OpenMC track-length k-eff is stored
separately in 'keff_tracklength'. nnSource.py and HyperparamSearchTriforce.py
read h5["keff"]; change that to h5["keff_tracklength"] to train and evaluate
against the Monte Carlo k-eff instead.

Usage
-----
    python build_h5.py
    python build_h5.py --data-dir /path/to/training_data --out training_data_new.h5
    python build_h5.py --mask training_data/global_nonzero_mask.npy   # reuse a mask, skip pass 1
    python build_h5.py --overwrite                                    # replace an existing output file
"""

import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np


# ── config ────────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = Path("training_data")
DEFAULT_H5_PATH  = Path("training_data.h5")

REFERENCE_TEMPS = list(range(300, 901, 50))   # 300, 350, ..., 900 -> 13 values
EPSILON         = 1e-10                       # same threshold as FM_cleaner.py

N_CELLS     = 289                  # full 17x17 grid
N_FUEL      = 264                  # fuel cells after packing
N_FEATURES  = N_FUEL + N_CELLS     # 553
N_SIDE      = 264                  # trimmed FM side
FM_SIZE     = N_SIDE * N_SIDE      # 69696

LOG_EVERY = 100


# ── loading helpers ───────────────────────────────────────────────────────────
def find_batch_dirs(base):
    """Batch folders = folders holding output_tallies_raw.npy, sorted like consolidate_data.py."""
    files = sorted(base.rglob("*/output_tallies_raw.npy"), key=lambda p: str(p).lower())
    seen, dirs = set(), []
    for p in files:
        if str(p.parent) not in seen:
            seen.add(str(p.parent))
            dirs.append(p.parent)
    return dirs


def load_square_fm(path, name):
    """Load one (289, 289) reference FM (same handling as FM_cleaner.py)."""
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.item() if raw.ndim == 0 else raw.flatten()[0]
    fm = np.asarray(raw, dtype=float)
    while fm.ndim > 2:
        fm = np.squeeze(fm)
    if fm.ndim == 1:
        side = math.isqrt(fm.size)
        if side * side != fm.size:
            raise ValueError(f"{name}: flat size {fm.size} is not a perfect square")
        fm = fm.reshape(side, side)
    if fm.shape != (N_CELLS, N_CELLS):
        raise ValueError(f"{name}: expected ({N_CELLS}, {N_CELLS}), got {fm.shape}")
    if not np.all(np.isfinite(fm)):
        raise ValueError(f"{name}: contains NaN or Inf")
    return fm


def load_batch_fms(bdir):
    """Load one batch's raw FM tallies (one entry of 289*289 values per sample)."""
    return np.load(bdir / "output_tallies_raw.npy", allow_pickle=True)


def normalize_fm(raw_entry):
    """
    Normalize one raw FM tally into a (289, 289) fission matrix, identical to
    build_normalized_fission_matrix() in the data generator and
    normalize_matrix() in FM_normalizer.py:
        FM[:, j] = raw[:, j] / sum(raw[j, :]) * sum(raw)
    Columns whose row sum is zero (no source in that cell) stay zero.
    """
    raw = np.asarray(raw_entry, dtype=float).reshape(N_CELLS, N_CELLS)
    total = raw.sum()
    row_sums = raw.sum(axis=1)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    fm = raw / safe[np.newaxis, :] * total
    fm[:, row_sums <= 0] = 0.0
    return fm


def load_batch_temps(bdir):
    """
    Return one batch's temperatures as float32 (n, 553): 264 packed fuel temps
    followed by the 289 other temps. Uses input_temps_clean.npy if present,
    otherwise packs input_temps.npy by dropping the zero (non-fuel) entries
    of the fuel vector, exactly as temp_cleaner.py does.
    Returns (array, list_of_bad_sample_indices).
    """
    clean = sorted(bdir.glob("input_temps_clean*.npy"))
    path = clean[0] if clean else bdir / "input_temps.npy"
    raw = np.load(path, allow_pickle=True)

    rows, bad = [], []
    for idx, entry in enumerate(raw):
        try:
            fuel  = np.asarray(entry[0], dtype=np.float32).ravel()
            other = np.asarray(entry[1], dtype=np.float32).ravel()
        except Exception:
            bad.append(idx)
            continue
        if fuel.size != N_FUEL:
            fuel = fuel[fuel != 0]
        if fuel.size != N_FUEL or other.size != N_CELLS:
            bad.append(idx)
            continue
        rows.append(np.concatenate([fuel, other]))

    arr = np.stack(rows, axis=0) if rows else np.empty((0, N_FEATURES), np.float32)
    return arr, bad, path.name


def _to_floats(arr):
    """Object array of floats (or uncertainties ufloats) -> float32 array; bad entries -> NaN."""
    out = []
    for v in arr:
        try:
            out.append(float(v.nominal_value) if hasattr(v, "nominal_value") else float(v))
        except Exception:
            out.append(np.nan)
    return np.asarray(out, dtype=np.float32)


def load_batch_keff(bdir):
    """
    Return one batch's OpenMC track-length k-eff and its std dev as float32
    arrays, from output_keff.npy and output_keff_uncertainty.npy (written by
    the data generator). Returns (keff, keff_std, error_message_or_None).
    """
    keff_path = bdir / "output_keff.npy"
    std_path  = bdir / "output_keff_uncertainty.npy"
    if not keff_path.exists():
        return None, None, "missing output_keff.npy"
    keff = _to_floats(np.load(keff_path, allow_pickle=True))
    if std_path.exists():
        std = _to_floats(np.load(std_path, allow_pickle=True))
        if std.size != keff.size:
            return None, None, f"{keff.size} keff values vs {std.size} uncertainties"
    else:
        std = np.full(keff.size, np.nan, dtype=np.float32)
    return keff, std, None


# ── power iteration ───────────────────────────────────────────────────────────
def power_iter_numpy(fm_np, tol=1e-7, max_iter=1000):
    """Standard numpy power iteration. Returns (k-eff, normalized fission source)."""
    n = fm_np.shape[0]
    v = np.ones((n, 1), dtype=float)
    eig = 1.0
    for i in range(max_iter):
        v_new = fm_np.dot(v)
        eig_new = float(np.max(np.abs(v_new)))
        if eig_new == 0:
            break
        v = v_new / eig_new
        if i > 0 and abs(eig_new - eig) < tol:
            eig = eig_new
            break
        eig = eig_new
    v = v.ravel()
    s = v.sum()
    if s != 0:
        v /= s
    return float(eig), v


# ── pass 1: mask ──────────────────────────────────────────────────────────────
def build_mask(ref_fms, batch_dirs):
    print("\nPass 1: building global non-zero mask (one batch at a time)...")
    mask = np.ones((N_CELLS, N_CELLS), dtype=bool)
    for T, fm in zip(REFERENCE_TEMPS, ref_fms):
        mask &= np.abs(fm) > EPSILON
    print(f"  after reference FMs: {mask.sum()} non-zero entries")

    n_seen = 0
    for b, bdir in enumerate(batch_dirs, 1):
        fms = load_batch_fms(bdir)
        for entry in fms:
            mask &= np.abs(normalize_fm(entry)) > EPSILON
        n_seen += len(fms)
        print(f"  [{b}/{len(batch_dirs)}] {bdir.name}: {len(fms)} FMs "
              f"| {n_seen} total | {mask.sum()} non-zero entries")
        del fms
    return mask


def check_mask(mask):
    """The model and nnSource.py assume the mask keeps a full 264 x 264 block."""
    n_rows = int(mask.any(axis=1).sum())
    n_cols = int(mask.any(axis=0).sum())
    if mask.sum() != FM_SIZE or n_rows != N_SIDE or n_cols != N_SIDE:
        print(f"[ERROR] Mask keeps {mask.sum()} entries in {n_rows} rows x {n_cols} cols; "
              f"expected exactly {N_SIDE} x {N_SIDE} = {FM_SIZE}.")
        print("        Some sample FMs have extra zeros, so they can't be trimmed to "
              "264 x 264 without changing the model's output size.")
        sys.exit(1)


# ── h5 ────────────────────────────────────────────────────────────────────────
def create_schema(h5):
    h5.create_dataset("temperatures", shape=(0, N_FEATURES), maxshape=(None, N_FEATURES),
                      dtype="float32", chunks=(32, N_FEATURES), compression="gzip")
    h5.create_dataset("fission_matrix", shape=(0, FM_SIZE), maxshape=(None, FM_SIZE),
                      dtype="float64", chunks=(32, FM_SIZE), compression="gzip")
    h5.create_dataset("fission_source", shape=(0, N_SIDE), maxshape=(None, N_SIDE),
                      dtype="float32", chunks=(32, N_SIDE), compression="gzip")
    h5.create_dataset("keff", shape=(0,), maxshape=(None,),
                      dtype="float32", chunks=(32,), compression="gzip")
    h5.create_dataset("keff_tracklength", shape=(0,), maxshape=(None,),
                      dtype="float32", chunks=(32,), compression="gzip")
    h5.create_dataset("keff_tracklength_std", shape=(0,), maxshape=(None,),
                      dtype="float32", chunks=(32,), compression="gzip")
    h5.create_dataset("Reference Matrices", shape=(len(REFERENCE_TEMPS), FM_SIZE),
                      dtype="float64", chunks=(1, FM_SIZE), compression="gzip")
    h5.create_dataset("reference_temperatures", shape=(len(REFERENCE_TEMPS),),
                      dtype="int32")


def write_references(h5, ref_fms, mask):
    for idx, (T, fm) in enumerate(zip(REFERENCE_TEMPS, ref_fms)):
        h5["Reference Matrices"][idx, :] = fm[mask]
        h5["reference_temperatures"][idx] = T
    print(f"Wrote {len(ref_fms)} reference matrices ({REFERENCE_TEMPS[0]}-{REFERENCE_TEMPS[-1]} K)")


def append_batch(h5, temps, fms, keff_tl, keff_tl_std, mask):
    """Append one batch's rows to every per-sample dataset."""
    n = temps.shape[0]
    start = h5["keff"].shape[0]
    end = start + n
    for name in ("temperatures", "fission_matrix", "fission_source", "keff",
                 "keff_tracklength", "keff_tracklength_std"):
        h5[name].resize(end, axis=0)

    h5["temperatures"][start:end, :] = temps
    h5["keff_tracklength"][start:end] = keff_tl
    h5["keff_tracklength_std"][start:end] = keff_tl_std
    for i in range(n):
        flat = normalize_fm(fms[i])[mask]                    # (69696,) row-major
        keff, source = power_iter_numpy(flat.reshape(N_SIDE, N_SIDE))
        h5["fission_matrix"][start + i, :] = flat
        h5["fission_source"][start + i, :] = source.astype(np.float32)
        h5["keff"][start + i] = np.float32(keff)
    return start, end


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build training_data.h5 from batch folders")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help=f"folder containing the batch folders (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--ref-dir", default=None,
                        help="folder with fission_matrix_{T}k.npy (default: --data-dir)")
    parser.add_argument("--out", default=str(DEFAULT_H5_PATH),
                        help=f"output h5 path (default: {DEFAULT_H5_PATH})")
    parser.add_argument("--mask", default=None,
                        help="reuse an existing (289, 289) mask .npy and skip pass 1")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the output file if it already exists")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    ref_dir  = Path(args.ref_dir).expanduser() if args.ref_dir else data_dir
    h5_path  = Path(args.out)
    mask_out = h5_path.with_name(f"{h5_path.stem}_nonzero_mask.npy")

    if h5_path.exists() and not args.overwrite:
        print(f"[ERROR] {h5_path} already exists. Use --overwrite to replace it, "
              f"or --out to write somewhere else.")
        sys.exit(1)

    # ---- inputs ----
    batch_dirs = find_batch_dirs(data_dir)
    if not batch_dirs:
        print(f"[ERROR] No */output_tallies_raw.npy found under {data_dir}")
        sys.exit(1)
    print(f"Found {len(batch_dirs)} batch folders under {data_dir}")

    missing = [ref_dir / f"fission_matrix_{T}k.npy" for T in REFERENCE_TEMPS
               if not (ref_dir / f"fission_matrix_{T}k.npy").exists()]
    if missing:
        print("[ERROR] Missing reference matrices:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)
    ref_fms = [load_square_fm(ref_dir / f"fission_matrix_{T}k.npy", f"FM_{T}K")
               for T in REFERENCE_TEMPS]

    # ---- pass 1: mask ----
    if args.mask:
        mask = np.load(args.mask).astype(bool)
        print(f"Using existing mask {args.mask} ({mask.sum()} non-zero entries)")
    else:
        mask = build_mask(ref_fms, batch_dirs)
    check_mask(mask)
    np.save(mask_out, mask)
    print(f"Saved mask -> {mask_out}")

    # ---- pass 2: write ----
    print("\nPass 2: writing h5 (one batch at a time)...")
    skipped = []
    with h5py.File(h5_path, "w") as h5:
        create_schema(h5)
        h5.create_dataset("nonzero_mask", data=mask)
        write_references(h5, ref_fms, mask)

        for b, bdir in enumerate(batch_dirs, 1):
            fms = load_batch_fms(bdir)
            temps, bad_temps, temps_file = load_batch_temps(bdir)
            keff_tl, keff_tl_std, keff_err = load_batch_keff(bdir)

            reason = None
            if bad_temps:
                reason = f"{len(bad_temps)} unreadable temperature samples"
            elif temps.shape[0] != len(fms):
                reason = f"{temps.shape[0]} temperature rows vs {len(fms)} FMs"
            elif keff_err:
                reason = keff_err
            elif keff_tl.size != len(fms):
                reason = f"{keff_tl.size} track-length keff values vs {len(fms)} FMs"
            if reason:
                print(f"  [{b}/{len(batch_dirs)}] SKIPPED {bdir}: {reason}")
                skipped.append((str(bdir), reason))
                continue

            start, end = append_batch(h5, temps, fms, keff_tl, keff_tl_std, mask)
            print(f"  [{b}/{len(batch_dirs)}] {bdir.name}: rows [{start}:{end}) "
                  f"from {temps_file}")
            h5.flush()
            del fms, temps

        n_total = h5["keff"].shape[0]

    print(f"\nDone. Wrote {n_total} samples to {h5_path}")
    if skipped:
        print(f"[WARN] Skipped {len(skipped)} batch folder(s):")
        for d, reason in skipped:
            print(f"  {d}: {reason}")
    print(f"Point MASK_PATH in nnSource.py at {mask_out} (or read 'nonzero_mask' from the h5).")


if __name__ == "__main__":
    main()
