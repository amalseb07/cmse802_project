"""
ising_mc.py
------------
This module provides functions to generate 2D Ising model spin configurations
using the Metropolis Monte Carlo algorithm with periodic boundary conditions.

For each specified temperature, it simulates the spin dynamics on an LxL lattice
and saves a series of equilibrated spin configurations. These configurations can
later be used for  training data for Convoluted Neural Network.

Key features:
- Random initialization of spin lattices (+1 / -1)
- Energy change computation for single-spin flips
- Metropolis update rule implementation
- Configurable simulation parameters (temperature, coupling J, equilibration steps, etc.)
- Automatic dataset generation and saving to disk

Default behavior:
- Generates 100 lattices per temperature
- Coupling constant J = 1
- Saves lattices as NumPy .npy files in data/ where the naming is ising_L{Size of Lattice}_T{Temperature}.npy .... where size of lattice by default is 32
  and temperature varies from 1 to 5 in spaces of 0.4.

Author: Amal Sebastian
Date : October 2025

"""

import numpy as np
import os
from tqdm import tqdm

# -------------------------------
# Ising Model (2D, periodic BCs)
# -------------------------------

def initial_lattice(L):
    """Initialize an LxL lattice with random spins (+1 or -1).
    
    Parameters
    ----------
    L : int
        Size (length) of one side of the square lattice.
    
    Returns
    -------
    numpy.ndarray
        2D array of shape (L, L) with randomly assigned spins (+1 or -1).
    """
    return np.random.choice([-1, 1], size=(L, L))

def delta_energy(lattice, i, j, J=1.0):
    """Compute the energy change associated with flipping a single spin.

    Parameters
    ----------
    lattice : numpy.ndarray
        2D array representing the spin configuration of the lattice, 
        where each element is either +1 or -1.
    i : int
        Row index of the spin to flip.
    j : int
        Column index of the spin to flip.
    J : float, optional
        Coupling constant between neighboring spins (default is 1.0).

    Returns
    -------
    float
        The change in energy (ΔE) resulting from flipping the selected spin.
    """
    L = lattice.shape[0]
    spin = lattice[i, j]
    # Periodic boundary conditions
    neighbors = lattice[(i+1)%L, j] + lattice[(i-1)%L, j] + lattice[i, (j+1)%L] + lattice[i, (j-1)%L]
    dE = 2 * J * spin * neighbors        # energy calculation
    return dE

def metropolis_step(lattice, T, J=1.0):
    """Perform one Metropolis sweep (LxL spin updates) of the lattice.

    Each sweep attempts to flip every spin once on average, using the
    Metropolis-Hastings criterion to decide whether a spin flip is accepted.

    Parameters
    ----------
    lattice : numpy.ndarray
        2D array representing the spin configuration of the lattice,
        where each element is either +1 or -1.
    T : float
        Temperature of the system (in reduced units, k_B = 1).
    J : float, optional
        Coupling constant between neighboring spins (default is 1.0).

    Returns
    -------
    numpy.ndarray
        Updated 2D lattice configuration after one Metropolis sweep.
    """
    L = lattice.shape[0]
    for _ in range(L * L):
        i, j = np.random.randint(0, L), np.random.randint(0, L)
        dE = delta_energy(lattice, i, j, J)
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):      # condition for whether to flip to the spin , depends on temperarure
            lattice[i, j] *= -1
    return lattice

def run_simulation(L=32, T=2.5, J=1.0, n_steps=5000, equilibration=1000):
    
    """Run a full Monte Carlo simulation of the 2D Ising model at a given temperature.

    The simulation first equilibrates the lattice to reach thermal stability, 
    then performs additional Metropolis sweeps to sample configurations.
    Every 10th configuration after equilibration is saved for analysis.

    Parameters
    ----------
    L : int, optional
        Size (length) of one side of the square lattice (default is 32).
    T : float, optional
        Temperature of the system (default is 2.5).
    J : float, optional
        Coupling constant between neighboring spins (default is 1.0).
    n_steps : int, optional
        Number of Monte Carlo steps (sweeps) performed after equilibration (default is 5000).
    equilibration : int, optional
        Number of initial steps discarded to allow the system to reach equilibrium (default is 1000).

    Returns
    -------
    numpy.ndarray
        3D array containing the saved spin configurations with shape 
        (n_saved, L, L), where `n_saved = n_steps / 10`.
    """
    lattice = initial_lattice(L)
    for step in range(equilibration):        # loop to equilibrate at a given temperature 
        metropolis_step(lattice, T, J)
    
    configs = []
    for step in range(n_steps):
        metropolis_step(lattice, T, J)
        if step % 10 == 0:  # Save every 10th configuration
            configs.append(np.copy(lattice))
    return np.array(configs)

# -------------------------------
# Data Generation
# -------------------------------

def generate_dataset(L=32, temps=np.linspace(1.0, 5.0, 11), out_dir="../data", n_samples=200):
    """
    Generate and save Ising model spin configurations for a range of temperatures.

    For each temperature in the specified range, this function runs a Monte Carlo
    simulation using the 2D Ising model and saves the resulting lattice configurations
    as `.npy` files. These datasets can later be used for training data for Convoluted Neural Network.

    Parameters
    ----------
    L : int, optional
        Size (length) of one side of the square lattice (default is 32).
    temps : array-like, optional
        Sequence of temperatures at which to generate configurations 
        (default is np.linspace(1.0, 5.0, 11)).
    out_dir : str, optional
        Output directory where generated `.npy` files will be saved (default is "../data").
    n_samples : int, optional
        Number of Monte Carlo steps used to generate configurations per temperature 
        (default is 200).

    Returns
    -------
    None
        The function saves `.npy` files for each temperature in the specified directory.

    Notes
    -----
    - Each saved file is named as `ising_L{L}_T{T:.2f}.npy`.
    - Existing directories are automatically created if they do not exist.
  
    """


    os.makedirs(out_dir, exist_ok=True)
    for T in tqdm(temps, desc="Generating data"):                                      # form a certain number of example lattices at a given temperature 
        configs = run_simulation(L=L, T=T, n_steps=n_samples)
        np.save(os.path.join(out_dir, f"ising_L{L}_T{T:.2f}.npy"), configs)
    print(" Data generation complete!")

if __name__ == "__main__":
    generate_dataset(L=32, temps=np.linspace(1.0, 5.0, 11), n_samples=1000)

