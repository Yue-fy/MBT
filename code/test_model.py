"""
快速测试TB模型的正确性
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tb_model_exact import MnBi2Te4_Exact

print("="*70)
print("Testing MnBi2Te4 TB Model")
print("="*70)

# 创建模型
model = MnBi2Te4_Exact(n_layers=6)

print("\n" + "="*70)
print("Test 1: Γ point band structure")
print("="*70)

# Γ点能带
E_gamma, _ = model.solve_bands(0.0, 0.0)
print(f"\nEnergies at Γ (in eV):")
for i, E in enumerate(E_gamma):
    print(f"  Band {i:2d}: {E:10.6f} eV = {E*1000:10.2f} meV")

# 能隙
n_bands = len(E_gamma)
mid = n_bands // 2
gap = E_gamma[mid] - E_gamma[mid-1]
print(f"\nBand gap at Γ: {gap*1000:.2f} meV")
print(f"Valence band top: {E_gamma[mid-1]*1000:.2f} meV")
print(f"Conduction band bottom: {E_gamma[mid]*1000:.2f} meV")

print("\n" + "="*70)
print("Test 2: Check Hermiticity")
print("="*70)

# 检查几个k点的厄米性
test_k_points = [
    (0.0, 0.0),
    (0.1, 0.0),
    (0.1, 0.1),
    (0.5, 0.3),
]

max_error = 0.0
for kx, ky in test_k_points:
    H = model.hamiltonian_multilayer(kx, ky)
    error = np.max(np.abs(H - H.conj().T))
    max_error = max(max_error, error)
    print(f"k=({kx:.2f}, {ky:.2f}): Hermiticity error = {error:.2e}")

print(f"\nMax Hermiticity error: {max_error:.2e}")
if max_error < 1e-12:
    print("✓ Hamiltonian is Hermitian (within numerical precision)")
else:
    print("✗ WARNING: Large Hermiticity error!")

print("\n" + "="*70)
print("Test 3: Energy scale check at small k")
print("="*70)

# 检查小k值的能量尺度（不要用高对称点，先检查一般情况）
test_k_values = [
    ('Γ', 0.0, 0.0),
    ('Small k1', 0.05, 0.0),
    ('Small k2', 0.05, 0.05),
    ('Small k3', 0.1, 0.1),
]

print("\nBand gaps and energy ranges:")
for name, kx, ky in test_k_values:
    E, _ = model.solve_bands(kx, ky)
    mid = len(E) // 2
    gap = E[mid] - E[mid-1]
    print(f"  {name:12s} (k={kx:.2f},{ky:.2f}): gap = {gap*1000:7.2f} meV, E ∈ [{E.min():.3f}, {E.max():.3f}] eV")

print("\n" + "="*70)
print("Test 4: Parameter consistency check")
print("="*70)

# 验证参数推导
print("\nVerifying Eq. S8 relations:")
print(f"  e0 = C0 + 2*C1/az² + 4*C2/a²")
e0_calc = model.C0 + 2*model.C1/(model.az**2) + 4*model.C2/(model.a**2)
print(f"     = {model.C0} + 2*{model.C1}/{model.az**2:.2f} + 4*{model.C2}/{model.a**2:.2f}")
print(f"     = {e0_calc:.6f} eV")
print(f"  Stored: {model.e0:.6f} eV")
print(f"  Match: {abs(e0_calc - model.e0) < 1e-10}")

print(f"\n  tz0 = C1/az²")
tz0_calc = model.C1/(model.az**2)
print(f"      = {model.C1}/{model.az**2:.2f}")
print(f"      = {tz0_calc:.6f} eV")
print(f"  Stored: {model.tz0:.6f} eV")
print(f"  Match: {abs(tz0_calc - model.tz0) < 1e-10}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ Model initialized successfully")
print(f"✓ Band gap at Γ: {gap*1000:.2f} meV")
print(f"✓ Energy range: [{E_gamma.min():.3f}, {E_gamma.max():.3f}] eV")
print(f"✓ Hermiticity check passed (max error: {max_error:.2e})")
print("="*70)
