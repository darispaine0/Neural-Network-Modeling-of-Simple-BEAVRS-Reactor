# Neural Network Modeling of a Simple BEAVRS Reactor

Code for *Learning Non-Uniform Temperature Feedback in the Fission Matrix Method via Differentiable Power Iteration and Bayesian Optimization* (Paine and Mascolino).

A feed-forward neural network predicts how non-uniform fuel and moderator temperatures change the fission matrix (FM) of the OpenMC 17×17 BEAVRS PWR assembly. The network does not predict the FM from scratch. It learns a correction to a reference FM that is interpolated row by row from 13 uniform-temperature reference matrices. Training uses a loss on the FM, on k_eff, and on the fission source, computed through a differentiable power iteration.

This branch (`evaluation`) holds the full pipeline from `main` plus the scripts used for the paper's figures and tables; see [Evaluation scripts](#evaluation-scripts) at the end.

## Pipeline

| Step | Script | Produces |
|---|---|---|
| 1 | `ConstantTempFM.py` | 13 uniform-temperature reference FMs, 300–900 K in 50 K steps |
| 2 | `ExampleDataGenerator.py` | randomized-temperature training samples, one folder per batch |
| 3 | `build_h5.py` | `training_data.h5` and the non-zero mask |
| 4 | `BayesianHyperparamSearch.py` | best hyperparameters from a 600-trial Gaussian-process search |
| 5 | `nnSource.py` | the final trained model and its k_eff errors |

Run every script from the repository folder. All paths are relative to it.

## Requirements

- Python 3.10+
- [OpenMC](https://docs.openmc.org) (steps 1–2), most easily installed with conda: `conda install -c conda-forge openmc`
- The ENDF/B-VIII.0 HDF5 nuclear data library for OpenMC
- `numpy`, `h5py`, `matplotlib`
- `torch` (steps 4–5; a CUDA GPU is strongly recommended)
- `scikit-optimize` (step 4)

OpenMC and PyTorch can live in separate conda environments. Steps 1–2 only need OpenMC; steps 3–5 only need PyTorch, NumPy and h5py.

Point OpenMC at the nuclear data before steps 1 and 2:

```bash
export OPENMC_CROSS_SECTIONS=/path/to/endfb-viii.0-hdf5/cross_sections.xml
export OPENMC_DATA=/path/to/endfb-viii.0-hdf5
```

## Step-by-step

### 1. Reference fission matrices

```bash
python ConstantTempFM.py
```

Runs one OpenMC simulation per uniform temperature (300, 350, …, 900 K) and saves `training_data/fission_matrix_{T}k.npy` for each, already normalized. These are the database used for the row-wise interpolation.

### 2. Training data

```bash
python ExampleDataGenerator.py --runs 500 --particles 100000 --batches 100 --inactive 30 --temp-min 300 --temp-max 900
```

Each run draws a new random temperature for every fuel pin and every moderator cell (uniform on 300–900 K), runs OpenMC, and records:

- the raw FM tally
- the temperatures
- OpenMC's track-length k_eff and its uncertainty

Each call writes one folder, `training_data/batch_<timestamp>/`. Generate data in batches of 100–500 runs, so that a failed job only costs one batch, until you have about 3,000 samples. The paper uses 3,000.

### 3. Build the HDF5 dataset

```bash
python build_h5.py
```

Reads every batch folder under `training_data/`, one batch at a time so the full dataset never has to fit in memory, and writes `training_data.h5`. For each sample it does the following:

1. **Normalizes the FM** from the raw tally (Eq. 4 of the paper).
2. **Trims the FM** from 289×289 to 264×264. This drops the 25 guide-tube and instrument positions, whose rows and columns are zero in every matrix, using a mask that `build_h5.py` builds from all samples and reference matrices. The mask is saved as `training_data_nonzero_mask.npy`.
3. **Packs the fuel temperatures** to the 264 fuel pins and joins them with the 289 moderator temperatures into a 553-value input.
4. **Runs power iteration** on the trimmed FM to get the fission source vector.
5. **Stores the k_eff values**: OpenMC's track-length k_eff (`keff_tracklength`) and the FM eigenvalue (`keff`).

The 13 reference matrices are stored in the same file. Useful options:

- `--data-dir`: read batches from another folder
- `--out`: write to a different file name
- `--overwrite`: replace an existing `training_data.h5`; without it the script refuses

### 4. Hyperparameter search (optional)

```bash
python BayesianHyperparamSearch.py --trials 600 --initial-points 50
```

Bayesian optimization (Gaussian-process surrogate with expected improvement) over six parameters:

- hidden-layer width
- learning rate
- weight decay
- the three loss weights α (FM), β (k_eff) and γ (fission source)

Each trial trains a model with early stopping and is scored by mean |Δk_eff| in pcm on a 20 % validation split. The best trial so far is saved under `ds3_hparam_search_results/` (`best_so_far.json` and `best_model_checkpoint.pth`).

The search is expensive: 600 full trainings. To reproduce the paper's model without it, skip to step 5, which already contains the paper's values.

### 5. Train the final model

```bash
python nnSource.py
```

Trains the final network with the hyperparameters in `GlobalConfig`, which are set to the optimum reported in the paper:

| Parameter | Value |
|---|---|
| Hidden layer width | 1,744 |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 4.36 × 10⁻⁸ |
| α (FM MSE weight) | 2.87 × 10⁻³ |
| β (k_eff weight) | 0.31 |
| γ (fission source weight) | 7.69 × 10⁻⁶ |

If you ran step 4, copy its best values into `GlobalConfig` first.

Training uses:

- the differentiable power iteration
- a learning-rate scheduler that halves the rate after 5 epochs without improvement
- early stopping on the validation loss

The checkpoint, per-sample pcm errors and a `meta.json` summary are written to `ds3_fm_keff_loss_results/`. The paper's model reaches a mean absolute k_eff error of about 23 pcm against the track-length k_eff.

## Notes

- **k_eff target.** `nnSource.py` and `BayesianHyperparamSearch.py` train and evaluate against OpenMC's track-length k_eff (`KEFF_DATASET = "keff_tracklength"` in each config). Set it to `"keff"` to use the FM eigenvalue instead.
- **Inputs.** Temperatures are scaled as (T − 300 K) / 600 K and then standardized with the training-split mean and standard deviation.
- **Targets.** The network predicts ln(a_ij / a_ij^interp), where a_ij^interp is linearly interpolated row by row using each destination cell's temperature.
- **Folder layout after a full run:**

  ```
  training_data/
      fission_matrix_300k.npy ... fission_matrix_900k.npy
      batch_<timestamp>/ ...
  training_data.h5
  training_data_nonzero_mask.npy
  ds3_hparam_search_results/
  ds3_fm_keff_loss_results/
  ```

## Evaluation scripts

These reproduce the figures and tables in the paper. Run them after step 5, from the repository folder.

Most of them load a trained checkpoint through a `CHECKPOINT_PATH` setting near the top of the file. Point it at the run folder that `nnSource.py` created under `ds3_fm_keff_loss_results/`.

| Script | Paper | What it does |
|---|---|---|
| `DStester.py` | §3.3, Table 4 | Trains the four data structures (DS1–DS4) with no transform, ln and sqrt, five seeds each |
| `hiddenLayerSweep.py` | §3.9, Fig. 11 (left) | Hidden-layer width sweep |
| `ConvergenceStudy.py` | §3.9, Fig. 11 (right) | Training-set size sweep |
| `ConvergenceEqnFit2.py` | §2.9, §3.9, Table 3 | Fits the ten candidate convergence models; the sweep results are typed into the script |
| `TempNNEffects.py` | §3.5 | Evaluates the trained model on every sample and writes `fm_best_worst_pcm_results.npz` |
| `plot_pcm_distribution.py` | §3.5, Fig. 8 | Per-sample pcm error ranked best to worst, with k_eff and temperature std |
| `BestWorstTemps.py` | §3.6, Fig. 9 | Fuel and moderator temperature maps of the best and worst samples |
| `MinMaxTemps.py` | §3.6 | Distribution of per-sample temperature standard deviations |
| `TempNNRefEffects.py` | §3.7, Fig. 10 | pcm error for uniform-temperature inputs at each reference temperature |
| `print_nn_stats.py` | §3.8, Table 6 | Neural-network summary statistics |
| `refFMErr.py` | §3.8, Table 6 | LaTeX table comparing interpolation-only and neural-network errors |
| `LossContribution.py` | §3.10, Fig. 13 | Trains the final model while tracking each weighted loss term per epoch |
| `DataGenerationTiming.py` | §3.11 | Average OpenMC runtime per sample |
| `nnTimer.py` | §3.11 | Average neural-network inference time |

Run order:

- `plot_pcm_distribution.py`, `BestWorstTemps.py`, `print_nn_stats.py` and `refFMErr.py` read `fm_best_worst_pcm_results.npz`, so run `TempNNEffects.py` first.
- `refFMErr.py` also needs `ref_matrix_pcm_results.npz` (the interpolation-only baseline). The script that makes that file is not in this repository.

`DStester.py`, `hiddenLayerSweep.py`, `ConvergenceStudy.py`, `LossContribution.py` and `TempNNEffects.py` read the FM eigenvalue (`h5["keff"]`) as the reference k_eff. To match `nnSource.py`, change it to `h5["keff_tracklength"]`.
