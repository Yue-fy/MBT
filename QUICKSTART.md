# Quick Start Guide - Layer Hall Effect Calculations

## Installation

### Step 1: Install Python Dependencies

```powershell
# Navigate to project directory
cd d:\Yue\MBT

# Install required packages
pip install numpy scipy matplotlib jupyter ipykernel

# Optional: Install Kwant for advanced features
pip install kwant

# Optional: Install pyqula (may require specific setup)
# pip install git+https://github.com/joselado/pyqula
```

### Step 2: Verify Installation

```powershell
# Test Python imports
python -c "import numpy; import scipy; import matplotlib; print('All packages installed successfully!')"
```

## Running the Code

### Test 1: Tight-Binding Model

```powershell
cd code
python tb_model.py
```

**Expected output:**
- Model parameters
- Hamiltonian at Γ point
- Eigenvalues and gap
- Layer projection operators
- High-symmetry path

### Test 2: Berry Curvature Calculation

```powershell
python berry_curvature.py
```

**Expected output:**
- Berry curvature at test k-point
- Values for each band
- Sum of Berry curvatures

### Test 3: Quantum Metric Calculation

```powershell
python quantum_metric.py
```

**Expected output:**
- Quantum metric components (g^xx, g^yy, g^xy)
- Trace of quantum metric tensor

### Test 4: Layer Hall Conductivity

```powershell
python layer_hall.py
```

**Expected output:**
- Layer Berry curvature contrast
- Model initialization confirmation

## Basic Usage Example

Create a Python script `test_layer_hall.py`:

```python
import numpy as np
import sys
sys.path.append('d:/Yue/MBT/code')

from tb_model import MnBi2Te4_Model
from layer_hall import LayerHallCalculator

# Initialize model
print("Initializing MnBi2Te4 model...")
model = MnBi2Te4_Model(
    a=4.38,
    t=1.0,
    lambda_SO=0.3,
    M=0.5,
    t_perp_0=0.2,
    mu=0.0
)

# Calculate band structure at Gamma point
print("\nCalculating bands at Gamma point...")
energies, eigvecs = model.solve_bands(0.0, 0.0)
print(f"Energies: {energies}")
print(f"Gap: {energies[2] - energies[1]:.4f} eV")

# Initialize layer Hall calculator
print("\nInitializing Layer Hall calculator...")
calc = LayerHallCalculator(model, eta=1e-6, temperature=0.0)

# Calculate layer Berry curvature contrast at a k-point
print("\nCalculating layer Berry curvature contrast...")
kx, ky = 0.1, 0.1
delta_omega = calc.layer_berry_curvature_contrast(kx, ky, band_index=1)
print(f"ΔΩ at k=({kx}, {ky}), band 1: {delta_omega:.6f} Ų")

# Calculate intrinsic layer Hall conductivity (small grid for testing)
print("\nCalculating intrinsic layer Hall conductivity...")
print("This may take a few minutes...")

k_range = (-np.pi/4.38, np.pi/4.38)  # First BZ
nk = 50  # Start with coarse grid for testing

result = calc.intrinsic_layer_hall(
    k_range=k_range,
    nk=nk,
    occupied_bands=[0, 1],  # Assume first two bands occupied
    mu=0.0,
    dk=1e-4
)

sigma_layer = result['sigma_layer_intrinsic']
print(f"\nLayer Hall conductivity: {sigma_layer:.6f} e²/h")
print(f"(Note: Increase nk to 100-200 for converged results)")

print("\n" + "="*60)
print("Calculation completed successfully!")
print("="*60)
```

Run it:

```powershell
python test_layer_hall.py
```

## Parameter Scanning Example

Create `parameter_scan.py`:

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('d:/Yue/MBT/code')

from tb_model import MnBi2Te4_Model
from layer_hall import LayerHallCalculator

# Scan over exchange field M
M_values = np.linspace(0.1, 1.0, 10)
sigma_values = []

k_range = (-np.pi/4.38, np.pi/4.38)
nk = 50  # Coarse for speed

print("Scanning over exchange field M...")
for M in M_values:
    print(f"  M = {M:.2f} eV...")
    
    model = MnBi2Te4_Model(
        a=4.38, t=1.0, lambda_SO=0.3,
        M=M, t_perp_0=0.2, mu=0.0
    )
    
    calc = LayerHallCalculator(model, eta=1e-6, temperature=0.0)
    
    result = calc.intrinsic_layer_hall(
        k_range=k_range, nk=nk,
        occupied_bands=[0, 1], mu=0.0
    )
    
    sigma_values.append(result['sigma_layer_intrinsic'])

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(M_values, sigma_values, 'o-', linewidth=2, markersize=8)
plt.xlabel('Exchange Field M (eV)', fontsize=14)
plt.ylabel('Layer Hall Conductivity (e²/h)', fontsize=14)
plt.title('Layer Hall vs Exchange Field', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/layer_hall_vs_M.png', dpi=300)
print("\nPlot saved to results/layer_hall_vs_M.png")
plt.show()
```

## Troubleshooting

### Issue 1: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'numpy'`

**Solution:**
```powershell
pip install numpy scipy matplotlib
```

### Issue 2: Slow Calculation

**Problem:** Layer Hall calculation takes too long

**Solution:**
- Start with smaller k-mesh: `nk = 30-50`
- Increase gradually to `nk = 100-200` for converged results
- Use coarser `dk` step: `dk = 1e-3` instead of `1e-4`

### Issue 3: Memory Error

**Problem:** Out of memory for large k-meshes

**Solution:**
- Reduce `nk` value
- Calculate band by band instead of all at once
- Use chunked calculation (process k-points in batches)

### Issue 4: Numerical Instability

**Problem:** Large oscillations or NaN values

**Solution:**
- Increase regularization: `eta = 1e-5` or `1e-4`
- Check for band degeneracies at high-symmetry points
- Use smaller `dk` for velocity calculation: `dk = 1e-5`

## Performance Tips

1. **Start Small**: Use `nk = 30-50` for testing, increase to `nk = 100-200` for production
2. **Parallel Processing**: Future version can use `multiprocessing` for k-point loops
3. **Caching**: Save intermediate results (band structures, Berry curvature maps)
4. **Optimized k-mesh**: Focus on regions with large Berry curvature

## Next Steps

1. **Visualize band structure**:
   ```python
   k_path, labels, positions = model.get_high_symmetry_path()
   distances, bands = model.band_structure_path(k_path, num_points=100)
   # Plot bands vs distances
   ```

2. **Calculate Berry curvature maps**:
   ```python
   from berry_curvature import calculate_berry_curvature_map
   bc_result = calculate_berry_curvature_map(
       model, k_range=(-np.pi/4.38, np.pi/4.38), nk=100
   )
   # Plot heatmap of Berry curvature
   ```

3. **Compare with literature**:
   - Check against Chen et al. (2025)
   - Verify parameter ranges match experiments
   - Calculate at different temperatures/dopings

## Documentation

- **Theory**: See `theory/01_berry_physics.md` and `theory/02_layer_hall.md`
- **API Reference**: Docstrings in each Python module
- **Examples**: Coming soon in `notebooks/` directory

## Getting Help

If you encounter issues:
1. Check the docstrings: `help(MnBi2Te4_Model)`
2. Review theory documents for formula details
3. Verify parameters are in correct units (eV, Ångström)
4. Check that k-range matches Brillouin zone size

---

**Ready to calculate!** 🚀
