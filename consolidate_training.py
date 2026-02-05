#!/usr/bin/env python3
"""
consolidate_fm_data.py

Scans training_data batch folders (each contains output_fm_normalized.npy),
collects keff, keff_uncertainty, temperatures, and fission matrices.

Produces TWO consolidated datasets:

(A) WITH keff:
    output_keff_final.npy
    output_keff_uncertainty_final.npy
    input_temps_final.npy
    output_fm_normalized_final.npy

(B) WITHOUT keff:
    input_temps_nokef.npy
    output_fm_normalized_nokef.npy

Useful for training NN models both with and without keff targets.
"""

import argparse
from pathlib import Path
import numpy as np
import sys


# -------------------------------------------------------
# Utility: find batch dirs
# -------------------------------------------------------
def find_batch_dirs(base: Path):
    batch_files = sorted(base.rglob("*/output_fm_normalized.npy"), key=lambda p: str(p).lower())
    batch_dirs = []
    seen = set()
    for p in batch_files:
        if str(p.parent) not in seen:
            seen.add(str(p.parent))
            batch_dirs.append(p.parent)
    return batch_dirs


def load_npy_allow(p: Path):
    try:
        return np.load(p, allow_pickle=True)
    except Exception as e:
        print(f"[WARN] Failed loading {p}: {e}")
        return None


def extract_keff_values(arr):
    vals = []
    for e in arr:
        try:
            if hasattr(e, "nominal_value"):
                vals.append(float(e.nominal_value))
            else:
                vals.append(float(e))
        except Exception:
            vals.append(np.nan)
    return vals


def choose_best_file(candidate_paths, expected_len):
    if not candidate_paths:
        return None
    best = None
    best_diff = None
    for p in candidate_paths:
        arr = load_npy_allow(p)
        if arr is None:
            continue
        try:
            L = len(arr)
        except Exception:
            L = arr.size if hasattr(arr, "size") else 1
        diff = abs((expected_len or L) - L)
        if best is None or diff < best_diff:
            best = p
            best_diff = diff
            if diff == 0:
                break
    return best if best else candidate_paths[0]


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "-d", default="training_data")
    parser.add_argument("--out-dir", "-o", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = Path(args.data_dir).expanduser()
    outdir = Path(args.out-dir) if args.out_dir else base

    if not base.exists():
        print("[ERROR] data directory not found:", base)
        sys.exit(1)

    print("\nScanning batch directories...")
    batch_dirs = find_batch_dirs(base)
    print(f"Found {len(batch_dirs)} batch folders.\n")

    aligned_keffs = []
    aligned_keff_unc = []
    aligned_temps = []
    aligned_fm = []

    total_runs = 0
    report = []

    for bdir in batch_dirs:
        fm_path = bdir / "output_fm_normalized.npy"
        if not fm_path.exists():
            continue

        fm_arr = load_npy_allow(fm_path)
        if fm_arr is None:
            report.append((str(bdir), "fm_load_failed"))
            continue

        try:
            runs = len(fm_arr)
        except:
            runs = fm_arr.size if hasattr(fm_arr, "size") else 1

        total_runs += runs

        # ----------------------------------
        # FM collection
        # ----------------------------------
        try:
            if len(fm_arr) == runs:
                aligned_fm.extend(fm_arr)
            elif len(fm_arr) < runs:
                aligned_fm.extend(fm_arr)
                aligned_fm.extend([None] * (runs - len(fm_arr)))
                report.append((str(bdir), "fm_short"))
            else:
                aligned_fm.extend(fm_arr[:runs])
                report.append((str(bdir), "fm_long_truncated"))
        except:
            if runs == 1:
                aligned_fm.append(fm_arr)
            else:
                aligned_fm.extend([None] * runs)
                report.append((str(bdir), "fm_corrupt"))

        # ----------------------------------
        # KEFF lookup
        # ----------------------------------
        keff_candidates = sorted(list(bdir.glob("output_keff*.npy")), key=lambda p: str(p))

        keff_files = [p for p in keff_candidates if "unc" not in p.name.lower()]
        unc_files = [p for p in keff_candidates if "unc" in p.name.lower()]

        chosen_keff = choose_best_file(keff_files, runs) if keff_files else None
        chosen_unc = choose_best_file(unc_files, runs) if unc_files else None

        # ---- keff ----
        if chosen_keff is not None:
            arr = load_npy_allow(chosen_keff)
            if arr is None:
                aligned_keffs.extend([np.nan] * runs)
                report.append((str(bdir), "keff_load_failed"))
            else:
                vals = extract_keff_values(arr)
                if len(vals) == runs:
                    aligned_keffs.extend(vals)
                elif len(vals) < runs:
                    aligned_keffs.extend(vals)
                    aligned_keffs.extend([np.nan] * (runs - len(vals)))
                    report.append((str(bdir), "keff_short"))
                else:
                    aligned_keffs.extend(vals[:runs])
                    report.append((str(bdir), "keff_long_truncated"))
        else:
            aligned_keffs.extend([np.nan] * runs)

        # ---- keff uncertainty ----
        if chosen_unc is not None:
            arr = load_npy_allow(chosen_unc)
            if arr is None:
                aligned_keff_unc.extend([np.nan] * runs)
                report.append((str(bdir), "keff_unc_load_failed"))
            else:
                vals = extract_keff_values(arr)
                if len(vals) == runs:
                    aligned_keff_unc.extend(vals)
                elif len(vals) < runs:
                    aligned_keff_unc.extend(vals)
                    aligned_keff_unc.extend([np.nan] * (runs - len(vals)))
                else:
                    aligned_keff_unc.extend(vals[:runs])
        else:
            aligned_keff_unc.extend([np.nan] * runs)

        # ----------------------------------
        # Temps
        # ----------------------------------
        temp_files = sorted(list(bdir.glob("input_temps*.npy")), key=lambda p: str(p))
        chosen_temp = choose_best_file(temp_files, runs) if temp_files else None

        if chosen_temp is not None:
            arr = load_npy_allow(chosen_temp)
            if arr is None:
                aligned_temps.extend([None] * runs)
                report.append((str(bdir), "temps_load_failed"))
            else:
                try:
                    if len(arr) == runs:
                        aligned_temps.extend(arr)
                    elif len(arr) < runs:
                        aligned_temps.extend(arr)
                        aligned_temps.extend([None] * (runs - len(arr)))
                    else:
                        aligned_temps.extend(arr[:runs])
                except:
                    if runs == 1:
                        aligned_temps.append(arr)
                    else:
                        aligned_temps.extend([None] * runs)
        else:
            aligned_temps.extend([None] * runs)

    # -------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------
    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)

        # FULL dataset (with keff)
        np.save(outdir / "output_keff_final.npy", np.array(aligned_keffs, dtype=float))
        np.save(outdir / "output_keff_uncertainty_final.npy", np.array(aligned_keff_unc, dtype=float))
        np.save(outdir / "input_temps_final.npy", np.array(aligned_temps, dtype=object))
        np.save(outdir / "output_fm_normalized_final.npy", np.array(aligned_fm, dtype=object))

        # FM-only dataset
        np.save(outdir / "input_temps_nokef.npy", np.array(aligned_temps, dtype=object))
        np.save(outdir / "output_fm_normalized_nokef.npy", np.array(aligned_fm, dtype=object))

        print("\nSaved consolidated datasets:")
        print(" WITH KEFF:")
        print("  output_keff_final.npy")
        print("  output_keff_uncertainty_final.npy")
        print("  input_temps_final.npy")
        print("  output_fm_normalized_final.npy")
        print("\n WITHOUT KEFF:")
        print("  input_temps_nokef.npy")
        print("  output_fm_normalized_nokef.npy")

    print("\nDone.")


if __name__ == "__main__":
    main()
