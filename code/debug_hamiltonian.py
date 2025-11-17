"""
调试：检查Γ点哈密顿量的构成
"""
import numpy as np

# Nature SI参数
C0 = -0.0048
C2 = 0.0
M0 = -0.1165
M2 = 9.4048
a = 4.334

# 在k=0时
cos_term = 3  # cos(0) + cos(0) + cos(0) = 3

epsilon_k0 = C0 + 2*C2/(a**2) * (2 - cos_term)
mass_k0 = M0 + 2*M2/(a**2) * (2 - cos_term)

print("Γ点 (k=0) 的哈密顿量参数：")
print(f"  epsilon = C0 + 2*C2/a² * (2-3)")
print(f"          = {C0} + 2*{C2}/{a**2:.2f} * (-1)")
print(f"          = {epsilon_k0:.4f} eV")
print(f"\n  mass    = M0 + 2*M2/a² * (2-3)")  
print(f"          = {M0} + 2*{M2}/{a**2:.2f} * (-1)")
print(f"          = {mass_k0:.4f} eV")

print(f"\n问题：mass = {mass_k0:.4f} eV 太大了！")
print(f"这会导致能带在 ±{abs(mass_k0):.2f} eV 范围")

print("\n" + "="*70)
print("检查：是不是理解错了公式？")
print("="*70)

# 也许公式是这样的？
epsilon_alt = C0 + 2*C2/(a**2) * (cos_term)
mass_alt = M0 + 2*M2/(a**2) * (cos_term)

print(f"\n备选理解 (没有2-...):")
print(f"  epsilon = {epsilon_alt:.4f} eV")
print(f"  mass    = {mass_alt:.4f} eV")

# 或者参数定义不同？
print("\n" + "="*70)
print("检查Eq. S8的推导")
print("="*70)

print(f"\n从Eq. S8:")
print(f"  e0 = C0 + 2*C1/az² + 4*C2/a²")
print(f"  e5 = M0 + 2*M1/az² + 4*M2/a²")

az = 13.64
C1 = 2.7232
M1 = 11.9048

e0 = C0 + 2*C1/(az**2) + 4*C2/(a**2)
e5 = M0 + 2*M1/(az**2) + 4*M2/(a**2)

print(f"\n计算结果:")
print(f"  e0 = {e0:.6f} eV")
print(f"  e5 = {e5:.6f} eV")

print(f"\n在Γ点，单层2D哈密顿量 (kz=0):")
print(f"  H_2D(k=0) 的对角元 ≈ e0*I + e5*σz")
print(f"            ≈ {e0:.4f}*I ± {e5:.4f}")
print(f"            ≈ [{e0-e5:.4f}, {e0+e5:.4f}] eV")
