import numpy as np
import os
from tqdm import tqdm

# -------------------------------
# Ising Model (2D, periodic BCs)
# -------------------------------

def initial_lattice(L):
    """Initialize LxL lattice with random spins (+1 or -1)."""
    return np.random.choice([-1, 1], size=(L, L))

def delta_energy(lattice, i, j, J=1.0):
    """Compute the energy change for flipping a single spin."""
    L = lattice.shape[0]
    spin = lattice[i, j]
    # Periodic boundary conditions
    neighbors = lattice[(i+1)%L, j] + lattice[(i-1)%L, j] + lattice[i, (j+1)%L] + lattice[i, (j-1)%L]
    dE = 2 * J * spin * neighbors
    return dE

def metropolis_step(lattice, T, J=1.0):
    """Perform one Metropolis sweep (LxL spin updates)."""
    L = lattice.shape[0]
    for _ in range(L * L):
        i, j = np.random.randint(0, L), np.random.randint(0, L)
        dE = delta_energy(lattice, i, j, J)
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            lattice[i, j] *= -1
    return lattice

def run_simulation(L=32, T=2.5, J=1.0, n_steps=5000, equilibration=1000):
    """Run full Monte Carlo simulation at a given temperature."""
    lattice = initial_lattice(L)
    for step in range(equilibration):
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
    """Generate Ising configurations for a range of temperatures."""
    os.makedirs(out_dir, exist_ok=True)
    for T in tqdm(temps, desc="Generating data"):
        configs = run_simulation(L=L, T=T, n_steps=n_samples)
        np.save(os.path.join(out_dir, f"ising_L{L}_T{T:.2f}.npy"), configs)
    print(" Data generation complete!")

if __name__ == "__main__":
    generate_dataset(L=32, temps=np.linspace(1.0, 5.0, 11), n_samples=1000)

