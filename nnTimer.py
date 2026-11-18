
#!/usr/bin/env python3

import numpy as np
import h5py
import torch
import torch.nn as nn
import time
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

class Config:
    H5_PATH = Path("training_data.h5")

    # Train Checkpoint
    CHECKPOINT_PATH = Path(
        "ds3_fm_keff_loss_results/"
        "train_fm_keff_loss_20260831_100114/"
        "hidden_3522/"
        "run_1/"
        "checkpoint_hidden3522.pth"
    )

    N_ROWS = 17
    N_COLS = 17
    N_CELLS = 289
    N_CELLS2 = 264

    INPUT_SIZE = 553
    OUTPUT_SIZE = 69696

    # Number of simulations to time
    N_SIMULATIONS = 10

    # Number of power iterations
    POWER_ITER_STEPS = 1000

    # Use GPU if available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()


# ============================================================
# MODEL
# ============================================================

class FMCoefficientPredictor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# POWER ITERATION
# ============================================================

def power_iter_numpy(fm_np, tol=1e-7, max_iter=1000):

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


# ============================================================
# LOAD MASK
# ============================================================

def load_mask():

    mask_path = Path(
        ""
        "training_data_nonzero_mask.npy"
    )

    mask = np.load(mask_path)

    return mask


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading checkpoint:")
    print(cfg.CHECKPOINT_PATH)

    checkpoint = torch.load(
        cfg.CHECKPOINT_PATH,
        map_location=cfg.DEVICE,
        weights_only=False
    )

    hidden_size = checkpoint["config"]["HIDDEN_SIZE"]

    model = FMCoefficientPredictor(
        cfg.INPUT_SIZE,
        hidden_size,
        cfg.OUTPUT_SIZE
    ).to(cfg.DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    mu = np.asarray(checkpoint["train_mean"], dtype=np.float32)
    sig = np.asarray(checkpoint["train_std"], dtype=np.float32)

    print(f"Hidden size: {hidden_size}")
    print(f"Device: {cfg.DEVICE}")

    return model, mu, sig


# ============================================================
# RUN ONE NN SIMULATION
# ============================================================

def run_simulation(model, temperatures, mu, sig, mask):

    # --------------------------------------------------------
    # Normalize input exactly as during training
    # --------------------------------------------------------

    X = (temperatures - mu) / sig

    X_tensor = torch.from_numpy(
        X.astype(np.float32)
    ).unsqueeze(0).to(cfg.DEVICE)

    # --------------------------------------------------------
    # Neural network inference
    # --------------------------------------------------------

    if cfg.DEVICE.type == "cuda":
        torch.cuda.synchronize()

    nn_start = time.perf_counter()

    with torch.no_grad():
        fm_prediction = model(X_tensor)

    if cfg.DEVICE.type == "cuda":
        torch.cuda.synchronize()

    nn_end = time.perf_counter()

    nn_time = nn_end - nn_start

    # Move prediction to CPU
    fm_prediction = fm_prediction.cpu().numpy()[0]

    # --------------------------------------------------------
    # Convert predicted ln-ratio back to FM
    #
    # IMPORTANT:
    # The original model predicts a ln-ratio FM, so this
    # converts it back to the actual fission matrix.
    #
    # Here we need the reference matrix corresponding to
    # the temperature configuration.
    # --------------------------------------------------------

    return fm_prediction, nn_time


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NEURAL NETWORK SIMULATION TIMING")
    print("=" * 70)

    print(f"Device: {cfg.DEVICE}")
    print(f"Number of simulations: {cfg.N_SIMULATIONS}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, mu, sig = load_model()

    # --------------------------------------------------------
    # Load mask and data
    # --------------------------------------------------------

    mask = load_mask()

    with h5py.File(cfg.H5_PATH, "r") as h5:

        temperatures = h5["temperatures"][:cfg.N_SIMULATIONS]

    print(f"Loaded {len(temperatures)} temperature configurations.")

    # --------------------------------------------------------
    # Warm-up
    #
    # This prevents CUDA/PyTorch initialization from being
    # included in the actual timing.
    # --------------------------------------------------------

    print("\nWarming up neural network...")

    dummy = torch.from_numpy(
        ((temperatures[0] - mu) / sig)
        .astype(np.float32)
    ).unsqueeze(0).to(cfg.DEVICE)

    with torch.no_grad():

        for _ in range(5):

            _ = model(dummy)

            if cfg.DEVICE.type == "cuda":
                torch.cuda.synchronize()

    print("Warm-up complete.")

    # --------------------------------------------------------
    # Time simulations
    # --------------------------------------------------------

    nn_times = []

    print("\n" + "=" * 70)
    print("TIMING RESULTS")
    print("=" * 70)

    for i in range(cfg.N_SIMULATIONS):

        print(f"\nSimulation {i + 1}/{cfg.N_SIMULATIONS}")

        # ----------------------------------------------------
        # Neural network timing
        # ----------------------------------------------------

        fm_prediction, nn_time = run_simulation(
            model,
            temperatures[i],
            mu,
            sig,
            mask
        )

        nn_times.append(nn_time)

        print(
            f"  Neural network inference: "
            f"{nn_time * 1000:.4f} ms"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    nn_times = np.asarray(nn_times)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nNeural network inference:")
    print(f"  Mean:   {np.mean(nn_times) * 1000:.4f} ms")
    print(f"  Std:    {np.std(nn_times) * 1000:.4f} ms")
    print(f"  Min:    {np.min(nn_times) * 1000:.4f} ms")
    print(f"  Max:    {np.max(nn_times) * 1000:.4f} ms")

    print(
        f"\nTotal time for {cfg.N_SIMULATIONS} simulations: "
        f"{np.sum(nn_times):.6f} s"
    )

    print(
        f"Average simulations per second: "
        f"{1.0 / np.mean(nn_times):.2f}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
