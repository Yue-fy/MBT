# Layer Hall Effect in Antiferromagnetic Topological Insulators

## Project Overview

This project implements numerical calculations and theoretical analysis for the **Layer Hall Effect** in antiferromagnetic topological insulators, with a focus on **MnBi₂Te₄** systems.

### Physical Phenomena

1. **Layer Hall Effect**: Transport phenomenon where an in-plane electric field generates a transverse layer current/accumulation
2. **Berry Curvature**: Geometric property of Bloch states in momentum space
3. **Quantum Metric**: Real part of the quantum geometric tensor
4. **Antiferromagnetic Topological Insulator**: Material with broken time-reversal symmetry but compensated magnetization

### Key Materials

- **MnBi₂Te₄**: Intrinsic magnetic topological insulator
- Layered structure with antiferromagnetic coupling between Mn layers
- Hosts quantum anomalous Hall effect (QAHE) and axion insulator states

## Project Structure

```
MBT/
├── code/               # Numerical implementation
│   ├── tb_model.py            # Tight-binding Hamiltonian
│   ├── berry_curvature.py     # Berry curvature calculations
│   ├── quantum_metric.py      # Quantum metric tensor
│   ├── layer_hall.py          # Layer Hall conductivity
│   └── visualization.py       # Plotting and analysis
├── theory/             # Theoretical derivations
│   ├── 01_berry_physics.md    # Berry curvature and connection
│   ├── 02_layer_hall.md       # Layer Hall effect theory
│   └── 03_computational.md    # Numerical methods
├── notebooks/          # Example calculations
├── results/            # Output data and figures
├── mypaper/            # Working documents
└── paper/              # Reference papers
```

## Computational Tools

- **pyqula**: Python library for quantum lattice calculations
- **Kwant**: Quantum transport simulations
- **NumPy/SciPy**: Numerical computations
- **Matplotlib**: Visualization

## References

Key papers in `paper/` directory:
- Chen et al. (2025) - Nonlinear Layer Hall Effect and Berry Curvature Dipole
- Gao et al. (2021) - Layer Hall Effect in 2D Topological Axion Antiferromagnet
- Gao et al. (2023) - Quantum Metric Nonlinear Hall Effect
- Wang et al. (2023) - Quantum-Metric-Induced Nonlinear Transport
- Deng et al. (2020) - QAHE in MnBi₂Te₄

## Getting Started

See `theory/03_computational.md` for detailed workflow and `notebooks/` for examples.

## Author

Yue - November 2025
