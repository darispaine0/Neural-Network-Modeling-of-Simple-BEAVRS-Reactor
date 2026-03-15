#!/usr/bin/env python3
"""
OpenMC Fission Matrix Data Generator - Neural Network Compatible
Version 2.0

Generates training data for fission matrix prediction neural networks.
Ensures proper data format compatibility with downstream consolidation and training scripts.

Usage:
    Interactive mode (will prompt for inputs):
        python openmc_data_generator_v2.py
    
    Command-line mode:
        python openmc_data_generator_v2.py --runs 100 [options]
        python openmc_data_generator_v2.py -n 50 --particles 100000 --temp-min 300 --temp-max 900
    
    Options:
        --runs, -n          Number of simulation runs (REQUIRED in CLI mode)
        --particles, -p     Particles per batch (default: 100000)
        --temp-min          Minimum temperature in K (default: 300)
        --temp-max          Maximum temperature in K (default: 900)
        --output-dir, -o    Custom output directory
        --show-plots        Display validation plots
        --batches           Number of simulation batches (default: 100)
        --inactive          Number of inactive batches (default: 30)
    
For local environment setup:
    1. conda activate openmc-env
    2. cd ~/Monte_Carlo/Test
    3. export OPENMC_CROSS_SECTIONS=/mnt/c/Users/Daris/Downloads/endfb80/endfb-viii.0-hdf5/cross_sections.xml
    4. export OPENMC_DATA=/mnt/c/Users/Daris/Downloads/endfb80/endfb-viii.0-hdf5
"""

import numpy as np
import openmc
import matplotlib.pyplot as plt
import re
import argparse
from pathlib import Path
from datetime import datetime
import json


# ============================================================================
# Configuration Class
# ============================================================================
class DataGenConfig:
    """Configuration parameters for data generation"""
    # Grid dimensions
    N_ROWS = 17
    N_COLS = 17
    N_CELLS = N_ROWS * N_COLS  # 289
    
    # Default simulation parameters
    DEFAULT_PARTICLES = 100000
    DEFAULT_BATCHES = 100
    DEFAULT_INACTIVE = 30
    DEFAULT_TEMP_MIN = 300  # Kelvin
    DEFAULT_TEMP_MAX = 900  # Kelvin
    DEFAULT_TOLERANCE = 400.0  # Temperature interpolation tolerance
    
    # Output directory structure
    OUTPUT_BASE = Path("training_data")
    
    # File naming conventions (match consolidation script expectations)
    INPUT_TEMPS_FILE = "input_temps.npy"
    OUTPUT_SOURCE_FILE = "output_source.npy"
    OUTPUT_FM_RAW_FILE = "output_tallies_raw.npy"
    OUTPUT_FM_NORMALIZED_FILE = "output_fm_normalized.npy"
    OUTPUT_KEFF_FILE = "output_keff.npy"
    OUTPUT_KEFF_UNC_FILE = "output_keff_uncertainty.npy"
    OUTPUT_SOURCE_UNC_FILE = "output_source_uncertainty.npy"
    
    # Validation settings
    VALIDATE_FM_SHAPE = True
    VALIDATE_TEMP_ASSIGNMENT = True


# ============================================================================
# Material Detection Utilities
# ============================================================================
def material_has_uranium(mat):
    """
    Robustly check if material contains uranium isotopes.
    
    Uses multiple detection methods to ensure reliability across OpenMC versions.
    
    Args:
        mat: OpenMC Material object
        
    Returns:
        bool: True if uranium is detected
    """
    # Method 1: Check material name
    try:
        if hasattr(mat, 'name') and mat.name:
            name_lower = mat.name.lower()
            if 'fuel' in name_lower or 'uo2' in name_lower:
                return True
    except Exception:
        pass
    
    # Method 2: Use get_nuclides() API
    try:
        nuclides = mat.get_nuclides()
        for nuclide in nuclides:
            if 'U' in str(nuclide):
                return True
    except Exception:
        pass
    
    # Method 3: Direct nuclides list inspection
    try:
        if hasattr(mat, 'nuclides') and mat.nuclides:
            for nuclide_tuple in mat.nuclides:
                nuclide_str = str(nuclide_tuple[0])
                if any(isotope in nuclide_str for isotope in ['U234', 'U235', 'U238']):
                    return True
                if re.search(r'U\d+', nuclide_str):  # Any uranium isotope pattern
                    return True
    except Exception:
        pass
    
    # Method 4: Try direct atom fraction query
    try:
        for isotope in ['U234', 'U235', 'U238']:
            if mat.get_nuclide_atom_fraction(isotope) > 0:
                return True
    except Exception:
        pass
    
    return False


# ============================================================================
# Temperature Assignment (Model 2 - Dual Temperature)
# ============================================================================
def assign_dual_temperatures(lattice, temp_min, temp_max, n_rows, n_cols, validate=True):
    """
    Assign separate random temperatures to fuel and non-fuel materials.
    
    This implements Model 2 from the original script, which is the model used
    by the neural network training pipeline.
    
    Args:
        lattice: OpenMC RectLattice object
        temp_min: Minimum temperature (K)
        temp_max: Maximum temperature (K)
        n_rows: Number of rows in lattice
        n_cols: Number of columns in lattice
        validate: Whether to validate temperature assignments
        
    Returns:
        tuple: (fuel_vector, other_vector) - flattened 1D arrays of assigned temperatures
    """
    # Generate random temperature matrices
    temps_fuel = np.random.uniform(temp_min, temp_max, size=(n_rows, n_cols)).astype(float)
    temps_other = np.random.uniform(temp_min, temp_max, size=(n_rows, n_cols)).astype(float)
    
    # Track actually assigned temperatures
    assigned_fuel = np.zeros((n_rows, n_cols), dtype=float)
    assigned_other = np.zeros((n_rows, n_cols), dtype=float)
    
    # Iterate through lattice cells
    for i in range(n_rows):
        for j in range(n_cols):
            uni = lattice.universes[i][j]
            if uni is None:
                continue
            
            fuel_temp = 0.0
            nonfuel_temp = 0.0
            
            # Assign temperatures based on material type
            for cell in uni.cells.values():
                if isinstance(cell.fill, openmc.Material):
                    mat = cell.fill
                    if material_has_uranium(mat):
                        fuel_temp = float(temps_fuel[i, j])
                        cell.temperature = fuel_temp
                    else:
                        nonfuel_temp = float(temps_other[i, j])
                        cell.temperature = nonfuel_temp
            
            # Store assigned temperatures (0 if not assigned)
            assigned_fuel[i, j] = fuel_temp
            assigned_other[i, j] = nonfuel_temp if nonfuel_temp > 0 else 294.0
    
    # Flatten to 1D vectors (289 elements each)
    fuel_vector = assigned_fuel.flatten()
    other_vector = assigned_other.flatten()
    
    # Validation
    if validate:
        assert fuel_vector.shape == (n_rows * n_cols,), f"Fuel vector shape mismatch: {fuel_vector.shape}"
        assert other_vector.shape == (n_rows * n_cols,), f"Other vector shape mismatch: {other_vector.shape}"
        
        n_fuel_assigned = np.sum(fuel_vector > 0)
        n_other_assigned = np.sum(other_vector > 0)
        print(f"  Assigned temperatures: {n_fuel_assigned} fuel pins, {n_other_assigned} non-fuel cells")
    
    return fuel_vector, other_vector


# ============================================================================
# Simulation Setup and Execution
# ============================================================================
def setup_pwr_assembly_geometry():
    """
    Create PWR assembly geometry and extract lattice.
    
    Returns:
        tuple: (assembly, geometry, lattice)
    """
    print("Creating PWR assembly geometry...")
    assembly = openmc.examples.pwr_assembly()
    geometry = assembly.geometry
    
    # Find the lattice in the geometry
    lattice = None
    for cell in geometry.get_all_cells().values():
        if isinstance(cell.fill, openmc.RectLattice):
            lattice = cell.fill
            break
    
    if lattice is None:
        raise RuntimeError("Could not find RectLattice in assembly geometry")
    
    print(f"  Found lattice with shape: {len(lattice.universes)}x{len(lattice.universes[0])}")
    return assembly, geometry, lattice


def setup_tallies(assembly, n_rows, n_cols):
    """
    Configure mesh-based tallies for fission source and fission matrix.
    
    Args:
        assembly: OpenMC assembly object
        n_rows: Number of mesh rows
        n_cols: Number of mesh columns
        
    Returns:
        tuple: (tally_source, tally_fm, mesh)
    """
    print("Setting up mesh and tallies...")
    
    # Create regular mesh aligned with assembly
    mesh = openmc.RegularMesh()
    mesh.dimension = (n_rows, n_cols)
    mesh.lower_left = (-10.71, -10.71)
    mesh.upper_right = (+10.71, +10.71)
    
    # Mesh filters
    mesh_filter = openmc.MeshFilter(mesh)
    born_filter = openmc.MeshBornFilter(mesh)
    
    # Fission source tally (where fission occurs)
    tally_source = openmc.Tally(name='Fission source')
    tally_source.filters = [mesh_filter]
    tally_source.scores = ['nu-fission']
    
    # Fission matrix tally (where born -> where fission)
    tally_fm = openmc.Tally(name='Fission matrix')
    tally_fm.filters = [mesh_filter, born_filter]
    tally_fm.scores = ['nu-fission']
    
    # Add tallies to assembly
    tallies = assembly.tallies
    tallies.append(tally_source)
    tallies.append(tally_fm)
    assembly.tallies = tallies
    
    return tally_source, tally_fm, mesh


def configure_settings(assembly, particles, batches=100, inactive=30, 
                       temp_method="interpolation", temp_default=294.0, temp_tolerance=400.0):
    """
    Configure simulation settings.
    
    Args:
        assembly: OpenMC assembly object
        particles: Number of particles per batch
        batches: Total number of batches
        inactive: Number of inactive batches
        temp_method: Temperature interpolation method
        temp_default: Default temperature (K)
        temp_tolerance: Temperature tolerance (K)
    """
    print("Configuring simulation settings...")
    settings = assembly.settings
    settings.particles = particles
    settings.batches = batches
    settings.inactive = inactive
    settings.temperature = {
        "method": temp_method,
        "default": temp_default,
        "tolerance": temp_tolerance
    }
    print(f"  Particles: {particles}, Batches: {batches} ({inactive} inactive)")


def run_simulation(assembly):
    """Execute OpenMC simulation"""
    print("Running OpenMC simulation...")
    assembly.run()
    print("  Simulation complete!")


# ============================================================================
# Results Processing
# ============================================================================
def extract_results(statepoint_file='statepoint.100.h5'):
    """
    Extract k-eff and tally results from statepoint file.
    
    Returns:
        dict: Contains keff, source, tallies_raw, and uncertainties
    """
    print("Analyzing results...")
    with openmc.StatePoint(filepath=statepoint_file) as output:
        keff = output.keff
        source_tally = output.get_tally(name='Fission source')
        fm_tally = output.get_tally(name='Fission matrix')
    
    results = {
        'keff_mean': keff.nominal_value.real,
        'keff_std': keff.std_dev.real,
        'source_mean': source_tally.mean.squeeze(),
        'source_std': source_tally.std_dev.squeeze(),
        'tallies_raw_mean': fm_tally.mean.squeeze(),
        'tallies_raw_std': fm_tally.std_dev.squeeze()
    }
    
    print(f"  k-eff: {results['keff_mean']:.6f} ± {results['keff_std']:.6f}")
    return results


def build_normalized_fission_matrix(tallies_raw, n_cells):
    """
    Construct normalized fission matrix from raw tally data.
    
    The normalization ensures each column sums to the total fission rate,
    representing the probability distribution of fission sites given a 
    birth location.
    
    Args:
        tallies_raw: Raw fission matrix data (flattened, length n_cells²)
        n_cells: Number of cells (289)
        
    Returns:
        numpy.ndarray: Normalized fission matrix (n_cells x n_cells)
    """
    print("Building normalized fission matrix...")
    fm = np.zeros((n_cells, n_cells), dtype=float)
    
    for i in range(n_cells):
        for j in range(n_cells):
            idx = i * n_cells + j
            if tallies_raw[idx] > 0:
                row_sum = np.sum(tallies_raw[j*n_cells:(j+1)*n_cells])
                if row_sum > 0: 
                    fm[i, j] = tallies_raw[idx] / row_sum * np.sum(tallies_raw)
    print(f"Sum of tallies_raw: {np.sum(tallies_raw)}")
    # Diagnostic: Check sparsity pattern
    nonzero_count = np.count_nonzero(fm)
    sparsity = 1.0 - (nonzero_count / fm.size)
    print(f"  Matrix sparsity: {sparsity*100:.2f}% ({nonzero_count}/{fm.size} nonzero)")
    
    return fm


def analyze_fission_matrix_sparsity(fm, n_cells):
    """
    Analyze and report fission matrix sparsity characteristics.
    
    This diagnostic helps identify the minimum number of rows needed
    to capture 99% of fission neutrons from each column.
    """
    n_rows_99_min = n_cells
    
    for j in range(n_cells):
        col_vec = fm[:, j]
        if np.all(np.isnan(col_vec)) or np.all(col_vec == 0):
            continue
        
        col_sum = np.sum(col_vec)
        if col_sum == 0 or not np.isfinite(col_sum):
            continue
            
        col_frac = col_vec / col_sum
        sorted_fracs = np.sort(col_frac)[::-1]
        cumulative = np.cumsum(sorted_fracs)
        n_rows_99 = np.argmax(cumulative >= 0.99) + 1
        
        if n_rows_99 < n_rows_99_min:
            n_rows_99_min = n_rows_99
            print(f"  Column {j}: {n_rows_99}/{n_cells} rows for 99% contribution")
            print(f"    Top row: {sorted_fracs[0]*100:.2f}%")
    
    return n_rows_99_min


def compute_uncertainties(source_mean, source_std, keff_mean, keff_std, n_rows, n_cols):
    """
    Compute relative uncertainties for source distribution and k-eff.
    
    Returns:
        dict: Uncertainty metrics
    """
    # Source relative uncertainty (element-wise)
    source_rel_unc = np.divide(
        source_std, 
        source_mean, 
        out=np.zeros_like(source_std), 
        where=source_mean != 0
    ).reshape(n_rows, n_cols)
    
    # k-eff relative uncertainty
    keff_rel_unc = keff_std / keff_mean if keff_mean != 0 else 0.0
    
    return {
        'source_rel_unc': source_rel_unc,
        'keff_rel_unc': keff_rel_unc
    }


# ============================================================================
# Power Iteration Validation
# ============================================================================
def power_iteration(fm, max_iter=500, tolerance=1e-6):
    """
    Find dominant eigenvalue and eigenvector via power iteration.
    
    Args:
        fm: Fission matrix (n_cells x n_cells)
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        tuple: (eigenvalue, eigenvector)
    """
    size = fm.shape[0]
    eig_value = 1.0
    eig_vector = np.ones((size, 1))
    
    for i_iter in range(max_iter):
        eig_vector_new = np.dot(fm, eig_vector)
        eig_value_new = np.max(np.abs(eig_vector_new))
        
        if eig_value_new == 0:
            break
            
        eig_vector = eig_vector_new / eig_value_new
        
        if abs(eig_value_new - eig_value) < tolerance:
            break
            
        eig_value = eig_value_new
    
    # Normalize eigenvector
    eig_vector = eig_vector.ravel()
    eig_vector /= eig_vector.sum()
    
    return eig_value, eig_vector


def validate_fission_matrix(fm, source_omc, keff_omc, n_rows, n_cols):
    """
    Validate fission matrix against OpenMC reference via power iteration.
    
    Args:
        fm: Computed fission matrix
        source_omc: OpenMC source distribution
        keff_omc: OpenMC k-eff value
        n_rows: Number of rows
        n_cols: Number of columns
        
    Returns:
        dict: Validation metrics
    """
    print("Validating fission matrix via power iteration...")
    
    # Power iteration on fission matrix
    kfm, source_fm = power_iteration(fm)
    
    # Reshape for comparison
    source_fm_2d = source_fm.reshape((n_rows, n_cols))
    source_omc_2d = source_omc.reshape((n_rows, n_cols))
    source_omc_normalized = source_omc_2d / source_omc_2d.sum()
    
    # k-eff comparison
    keff_rel_diff = (kfm / keff_omc - 1.0)
    keff_pcm = keff_rel_diff * 1e5
    
    print(f"  k-eff relative difference: {keff_pcm:.0f} pcm")
    
    # Source distribution comparison
    source_rel_diff = np.divide(
        source_fm_2d,
        source_omc_normalized,
        out=np.ones_like(source_fm_2d),
        where=source_omc_normalized != 0
    ) - 1.0
    
    source_rel_diff = np.nan_to_num(source_rel_diff, nan=0.0, posinf=0.0, neginf=0.0)
    
    max_loc = np.argmax(np.abs(source_rel_diff))
    rdiff_max = source_rel_diff.flatten()[max_loc]
    rdiff_mean = np.abs(source_rel_diff[source_rel_diff > 0]).mean()
    
    print(f"  Fission source largest relative difference: {rdiff_max*100:.2f}%")
    print(f"  Fission source mean absolute difference: {rdiff_mean*100:.2f}%")
    
    return {
        'keff_fm': kfm,
        'keff_pcm_diff': keff_pcm,
        'source_fm': source_fm_2d,
        'source_rel_diff': source_rel_diff,
        'source_max_diff_pct': rdiff_max * 100,
        'source_mean_diff_pct': rdiff_mean * 100
    }


# ============================================================================
# Visualization
# ============================================================================
def plot_validation_results(source_rel_diff, source_rel_unc, n_rows, n_cols, run_idx, output_dir=None):
    """
    Create side-by-side plots of relative difference and uncertainty.
    
    Args:
        source_rel_diff: Relative difference matrix
        source_rel_unc: Relative uncertainty matrix
        n_rows: Number of rows
        n_cols: Number of columns
        run_idx: Run index for title
        output_dir: Optional directory to save plot
    """
    plt.rcParams['font.size'] = 15
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Plot 1: Relative Difference
    im1 = axes[0].imshow(
        source_rel_diff * 100,
        cmap='jet',
        interpolation='nearest',
        origin='lower',
        aspect='equal'
    )
    axes[0].set_xticks(range(n_rows))
    axes[0].set_yticks(range(n_cols))
    axes[0].set_xticklabels(range(1, n_rows + 1))
    axes[0].set_yticklabels(range(1, n_cols + 1))
    axes[0].grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.6)
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Fission source relative difference (%)')
    axes[0].set_title(f'Fission Source - Relative Difference (Run {run_idx+1})')
    
    # Plot 2: Relative Uncertainty
    im2 = axes[1].imshow(
        source_rel_unc * 100,
        cmap='jet',
        interpolation='nearest',
        origin='lower',
        aspect='equal'
    )
    axes[1].set_xticks(range(n_rows))
    axes[1].set_yticks(range(n_cols))
    axes[1].set_xticklabels(range(1, n_rows + 1))
    axes[1].set_yticklabels(range(1, n_cols + 1))
    axes[1].grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.6)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Fission source relative uncertainty (%)')
    axes[1].set_title(f'Fission Source - Relative Uncertainty (Run {run_idx+1})')
    
    plt.tight_layout()
    
    if output_dir:
        plot_path = output_dir / f"validation_run_{run_idx+1}.png"
        plt.savefig(plot_path, dpi=150)
        print(f"  Saved plot to {plot_path}")
    
    plt.show()


# ============================================================================
# Data Management
# ============================================================================
class DataCollector:
    """Manages collection of data across multiple simulation runs"""
    
    def __init__(self):
        self.input_data = []
        self.source_data = []
        self.tallies_raw_data = []
        self.fm_normalized_data = []
        self.keff_data = []
        self.keff_uncertainty_data = []
        self.source_uncertainty_data = []
        self.metadata = []
    
    def add_run(self, temps_tuple, results, fm_normalized, uncertainties, validation_metrics):
        """Add data from a single simulation run"""
        # Store temps_tuple as-is - it's already (fuel_vec, other_vec) format
        # This will be saved as a (2,) shaped object array element
        self.input_data.append(temps_tuple)
        self.source_data.append(results['source_mean'])
        self.tallies_raw_data.append(results['tallies_raw_mean'])
        self.fm_normalized_data.append(fm_normalized)
        self.keff_data.append(results['keff_mean'])
        self.keff_uncertainty_data.append(uncertainties['keff_rel_unc'])
        self.source_uncertainty_data.append(uncertainties['source_rel_unc'])
        
        # Store metadata for this run
        self.metadata.append({
            'keff_mean': float(results['keff_mean']),
            'keff_std': float(results['keff_std']),
            'keff_fm': float(validation_metrics['keff_fm']),
            'keff_pcm_diff': float(validation_metrics['keff_pcm_diff']),
            'source_max_diff_pct': float(validation_metrics['source_max_diff_pct']),
            'source_mean_diff_pct': float(validation_metrics['source_mean_diff_pct'])
        })
    
    def save_to_directory(self, output_dir, config):
        """Save all collected data to output directory"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving data to {output_dir}...")
        
        # Convert tuples to the EXACT old format: (N, 2) object array
        # where each element [i] has shape (2,) containing two 289-element arrays
        temps_formatted = []
        for fuel_vec, other_vec in self.input_data:
            # Create 1D object array with 2 elements
            run_temps = np.empty(2, dtype=object)
            run_temps[0] = fuel_vec
            run_temps[1] = other_vec
            temps_formatted.append(run_temps)
        
        # Save data arrays
        np.save(output_dir / config.INPUT_TEMPS_FILE, np.array(temps_formatted, dtype=object))
        np.save(output_dir / config.OUTPUT_SOURCE_FILE, np.array(self.source_data, dtype=object))
        np.save(output_dir / config.OUTPUT_FM_RAW_FILE, np.array(self.tallies_raw_data, dtype=object))
        np.save(output_dir / config.OUTPUT_FM_NORMALIZED_FILE, np.array(self.fm_normalized_data, dtype=object))
        np.save(output_dir / config.OUTPUT_KEFF_FILE, np.array(self.keff_data, dtype=object))
        np.save(output_dir / config.OUTPUT_KEFF_UNC_FILE, np.array(self.keff_uncertainty_data, dtype=object))
        np.save(output_dir / config.OUTPUT_SOURCE_UNC_FILE, np.array(self.source_uncertainty_data, dtype=object))
        
        # Save metadata JSON
        metadata_path = output_dir / "run_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print("  All data saved successfully!")
        print(f"  Total samples: {len(self.keff_data)}")


# ============================================================================
# Main Execution
# ============================================================================
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate training data for fission matrix neural networks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--runs', '-n', type=int, required=True,
        help='Number of simulation runs to generate (REQUIRED)'
    )
    parser.add_argument(
        '--particles', '-p', type=int, default=DataGenConfig.DEFAULT_PARTICLES,
        help='Number of particles per batch'
    )
    parser.add_argument(
        '--temp-min', type=float, default=DataGenConfig.DEFAULT_TEMP_MIN,
        help='Minimum temperature (K)'
    )
    parser.add_argument(
        '--temp-max', type=float, default=DataGenConfig.DEFAULT_TEMP_MAX,
        help='Maximum temperature (K)'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, default=None,
        help='Output directory (default: training_data/batch_TIMESTAMP)'
    )
    parser.add_argument(
        '--show-plots', action='store_true',
        help='Display validation plots'
    )
    parser.add_argument(
        '--batches', type=int, default=DataGenConfig.DEFAULT_BATCHES,
        help='Number of simulation batches'
    )
    parser.add_argument(
        '--inactive', type=int, default=DataGenConfig.DEFAULT_INACTIVE,
        help='Number of inactive batches'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    # Parse arguments (with interactive fallback if no args provided)
    import sys
    if len(sys.argv) == 1:
        # No arguments provided - run interactively
        print("="*80)
        print("OpenMC Fission Matrix Data Generator v2.0 - Interactive Mode")
        print("="*80)
        try:
            n_runs = int(input("Number of simulation runs to generate: ").strip())
            particles = input(f"Particles per batch (default={DataGenConfig.DEFAULT_PARTICLES}): ").strip()
            particles = int(particles) if particles else DataGenConfig.DEFAULT_PARTICLES
            
            temp_min = input(f"Minimum temperature in K (default={DataGenConfig.DEFAULT_TEMP_MIN}): ").strip()
            temp_min = float(temp_min) if temp_min else DataGenConfig.DEFAULT_TEMP_MIN
            
            temp_max = input(f"Maximum temperature in K (default={DataGenConfig.DEFAULT_TEMP_MAX}): ").strip()
            temp_max = float(temp_max) if temp_max else DataGenConfig.DEFAULT_TEMP_MAX
            
            show_plots = input("Show validation plots? (y/N): ").strip().lower()
            show_plots = show_plots == 'y'
            
            # Create a mock args object
            class Args:
                def __init__(self):
                    self.runs = n_runs
                    self.particles = particles
                    self.temp_min = temp_min
                    self.temp_max = temp_max
                    self.output_dir = None
                    self.show_plots = show_plots
                    self.batches = DataGenConfig.DEFAULT_BATCHES
                    self.inactive = DataGenConfig.DEFAULT_INACTIVE
            args = Args()
        except (ValueError, KeyboardInterrupt) as e:
            print(f"\nError or interruption: {e}")
            print("Please provide valid inputs or use command-line arguments.")
            print("Usage: python openmc_data_generator_v2.py --runs N [options]")
            return
    else:
        args = parse_arguments()
    
    config = DataGenConfig()
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config.OUTPUT_BASE / f"batch_{timestamp}"
    
    print("="*80)
    print("OpenMC Fission Matrix Data Generator v2.0")
    print("="*80)
    print(f"Configuration:")
    print(f"  Runs: {args.runs}")
    print(f"  Particles per batch: {args.particles}")
    print(f"  Temperature range: {args.temp_min}-{args.temp_max} K")
    print(f"  Output directory: {output_dir}")
    print("="*80)
    
    # Initialize data collector
    collector = DataCollector()
    
    # Run simulations
    for run_idx in range(args.runs):
        print(f"\n{'='*80}")
        print(f"Run {run_idx+1}/{args.runs}")
        print(f"{'='*80}")
        
        # Setup geometry
        assembly, geometry, lattice = setup_pwr_assembly_geometry()
        
        # Assign temperatures (Model 2: dual temperature)
        fuel_temps, other_temps = assign_dual_temperatures(
            lattice, args.temp_min, args.temp_max,
            config.N_ROWS, config.N_COLS,
            validate=config.VALIDATE_TEMP_ASSIGNMENT
        )
        temps_tuple = (fuel_temps.copy(), other_temps.copy())
        
        # Setup tallies
        tally_source, tally_fm, mesh = setup_tallies(assembly, config.N_ROWS, config.N_COLS)
        
        # Configure settings
        configure_settings(
            assembly, args.particles,
            batches=args.batches,
            inactive=args.inactive,
            temp_tolerance=config.DEFAULT_TOLERANCE
        )
        
        # Run simulation
        run_simulation(assembly)
        
        # Extract results
        results = extract_results()
        
        # Build normalized fission matrix
        fm_normalized = build_normalized_fission_matrix(results['tallies_raw_mean'], config.N_CELLS)
        
        # Analyze sparsity
        analyze_fission_matrix_sparsity(fm_normalized, config.N_CELLS)
        
        # Compute uncertainties
        uncertainties = compute_uncertainties(
            results['source_mean'], results['source_std'],
            results['keff_mean'], results['keff_std'],
            config.N_ROWS, config.N_COLS
        )
        
        # Validate via power iteration
        validation_metrics = validate_fission_matrix(
            fm_normalized, results['source_mean'],
            results['keff_mean'], config.N_ROWS, config.N_COLS
        )
        
        # Collect data
        collector.add_run(temps_tuple, results, fm_normalized, uncertainties, validation_metrics)
        
        # Plot if requested
        if args.show_plots:
            plot_validation_results(
                validation_metrics['source_rel_diff'],
                uncertainties['source_rel_unc'],
                config.N_ROWS, config.N_COLS,
                run_idx, output_dir
            )
        
        print(f"Run {run_idx+1} complete!")
    
    # Save all data
    collector.save_to_directory(output_dir, config)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)
    print(f"Total runs completed: {args.runs}")
    print(f"Output directory: {output_dir}")
    print("\nk-eff Statistics:")
    keff_array = np.array(collector.keff_data)
    print(f"  Mean: {keff_array.mean():.6f}")
    print(f"  Std:  {keff_array.std():.6f}")
    print(f"  Min:  {keff_array.min():.6f}")
    print(f"  Max:  {keff_array.max():.6f}")
    
    print("\nValidation Statistics (FM vs OpenMC):")
    pcm_diffs = [m['keff_pcm_diff'] for m in collector.metadata]
    print(f"  Mean |PCM diff|: {np.mean(np.abs(pcm_diffs)):.2f} pcm")
    print(f"  Max |PCM diff|:  {np.max(np.abs(pcm_diffs)):.2f} pcm")
    
    source_diffs = [m['source_mean_diff_pct'] for m in collector.metadata]
    print(f"  Mean source diff: {np.mean(source_diffs):.2f}%")
    print(f"  Max source diff:  {np.max([m['source_max_diff_pct'] for m in collector.metadata]):.2f}%")
    
    print("\n" + "="*80)
    print("Data generation complete!")
    print("="*80)


if __name__ == "__main__":
    main()