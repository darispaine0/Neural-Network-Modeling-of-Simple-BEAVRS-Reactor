#!/usr/bin/env python3
"""
pack_fuel_temps.py

Iterates through each sample in a new-format input_temps.npy (where fuel_vec
is 289 elements with zeros at non-fuel positions) and packs it down to only
the nonzero entries (should be 264 fuel pins).

Reports:
  - Per-sample fuel_vec size before and after packing
  - Whether the packed size matches the expected 264
  - Summary statistics across all samples

Saves the repacked array to a new file so the original is untouched.

Usage:
    python pack_fuel_temps.py
    python pack_fuel_temps.py --input-dir /path/to/batch
    python pack_fuel_temps.py --input-dir /path/to/batch --expected-size 264
"""

import numpy as np
import argparse
from pathlib import Path

# ============================================================
# PATHS -- edit if needed
# ============================================================
DEFAULT_INPUT_DIR = Path("/home/daris/Monte_Carlo/training_data/batch_20260413_232715")
EXPECTED_FUEL_SIZE = 264   # number of fuel pins in 17x17 PWR assembly

N_CELLS = 289


# ============================================================
# Main
# ============================================================
def pack_fuel_temps(input_dir, expected_size):
    input_dir = Path(input_dir)
    input_path  = input_dir / "input_temps.npy"
    output_path = input_dir / "input_temps_packed.npy"

    if not input_path.exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    print("="*60)
    print("Fuel Temperature Packer")
    print("="*60)
    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Expected packed size: {expected_size}")
    print()

    temps_all = np.load(input_path, allow_pickle=True)
    n_samples = len(temps_all)
    print(f"Loaded {n_samples} samples  (array shape={temps_all.shape}, dtype={temps_all.dtype})\n")

    packed_data   = []
    size_report   = []   # (sample_idx, fuel_size_before, fuel_size_after, other_size, ok)
    bad_samples   = []

    for idx in range(n_samples):
        raw = temps_all[idx]

        # ---- decode ----
        try:
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                fuel_raw  = np.asarray(raw[0], dtype=float).flatten()
                other_raw = np.asarray(raw[1], dtype=float).flatten()
            elif isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == (2,):
                fuel_raw  = np.asarray(raw[0], dtype=float).flatten()
                other_raw = np.asarray(raw[1], dtype=float).flatten()
            elif isinstance(raw, np.ndarray) and raw.ndim == 2:
                # single-temp (17,17) -- treat both as same
                fuel_raw  = raw.flatten().astype(float)
                other_raw = raw.flatten().astype(float)
            else:
                raise ValueError(f"Unrecognised format: type={type(raw)}, "
                                 f"shape={getattr(raw,'shape','?')}")
        except Exception as e:
            print(f"  [ERROR] sample {idx}: {e}")
            bad_samples.append(idx)
            # keep original to avoid shifting indices
            packed_data.append(raw)
            size_report.append((idx, -1, -1, -1, False))
            continue

        size_before = fuel_raw.size
        n_zeros     = int(np.sum(fuel_raw == 0))
        n_nonzero   = int(np.sum(fuel_raw != 0))

        # ---- check if already packed ----
        if size_before == expected_size:
            # Already packed -- nothing to do
            fuel_packed = fuel_raw
            ok = True
            action = "already packed"
        elif size_before == N_CELLS:
            # Full 289 -- strip zeros
            fuel_packed = fuel_raw[fuel_raw != 0]
            ok = (fuel_packed.size == expected_size)
            action = f"stripped {n_zeros} zeros -> {fuel_packed.size} elements"
        else:
            # Unexpected size
            fuel_packed = fuel_raw[fuel_raw != 0]
            ok = (fuel_packed.size == expected_size)
            action = f"unexpected size {size_before} -> stripped to {fuel_packed.size}"

        size_after = fuel_packed.size

        # Per-sample report (only print mismatches or every 100th)
        if not ok or idx % 100 == 0:
            status = "OK " if ok else "BAD"
            print(f"  [{status}] sample {idx:5d} : "
                  f"fuel {size_before} -> {size_after}  |  {action}")

        if not ok:
            bad_samples.append(idx)

        size_report.append((idx, size_before, size_after, other_raw.size, ok))

        # ---- rebuild object array in packed format ----
        run_temps = np.empty(2, dtype=object)
        run_temps[0] = fuel_packed
        run_temps[1] = other_raw         # other_vec stays as-is (289)
        packed_data.append(run_temps)

    # ---- Save ----
    packed_array = np.array(packed_data, dtype=object)
    np.save(output_path, packed_array)

    # ---- Summary ----
    sizes_after = [r[2] for r in size_report if r[2] > 0]
    unique_sizes = sorted(set(sizes_after))
    n_ok  = sum(1 for r in size_report if r[4])
    n_bad = len(bad_samples)

    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples       : {n_samples}")
    print(f"Passed (size={expected_size:3d})  : {n_ok}")
    print(f"Failed              : {n_bad}")
    print(f"Unique packed sizes : {unique_sizes}")

    if bad_samples:
        print(f"\nBAD sample indices  : {bad_samples[:20]}"
              + (" ..." if len(bad_samples) > 20 else ""))

    # Verify a few samples from the saved file
    print("\nVerifying saved file...")
    verify = np.load(output_path, allow_pickle=True)
    for check_idx in [0, n_samples // 2, n_samples - 1]:
        if check_idx >= len(verify):
            continue
        v = verify[check_idx]
        f = np.asarray(v[0], dtype=float).flatten()
        o = np.asarray(v[1], dtype=float).flatten()
        zero_in_fuel = int(np.sum(f == 0))
        print(f"  sample {check_idx:5d} : fuel={f.size}  (zeros={zero_in_fuel})  "
              f"other={o.size}  "
              f"fuel T range [{f.min():.1f}, {f.max():.1f}] K")

    print(f"\nSaved packed array to:\n  {output_path}")
    print()

    if n_bad == 0:
        print("All samples packed successfully to size", expected_size)
    else:
        print(f"WARNING: {n_bad} samples could not be packed to size {expected_size}.")
        print("Check the bad sample indices above.")


def main():
    parser = argparse.ArgumentParser(
        description="Pack 289-element fuel_vec down to nonzero-only (expected: 264)"
    )
    parser.add_argument(
        '--input-dir', type=str, default=str(DEFAULT_INPUT_DIR),
        help=f'Directory containing input_temps.npy (default: {DEFAULT_INPUT_DIR})'
    )
    parser.add_argument(
        '--expected-size', type=int, default=EXPECTED_FUEL_SIZE,
        help=f'Expected number of nonzero fuel pins after packing (default: {EXPECTED_FUEL_SIZE})'
    )
    args = parser.parse_args()
    pack_fuel_temps(args.input_dir, args.expected_size)


if __name__ == "__main__":
    main()