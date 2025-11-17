"""
MnBi2Te4 Bulk Model - 使用第755行的简化参数
这组参数用于计算J和能带结构
"""
import numpy as np
from numpy import linalg
from typing import Tuple

class MnBi2Te4_Bulk_Simple:
    """
    简化的bulk模型 - 使用TBmodel.pdf第755行参数
    """
    
    def __init__(self):
        # 第755行参数
        self.A1 = 2.7023    # eV·Å
        self.A2 = 3.1964    # eV·Å
        self.M0 = -0.04     # eV
        self.M1 = 11.9048   # eV·Å²
        self.M2 = 9.4048    # eV·Å²
        self.m5 = 0.03      # eV (AFM交换)
        
        self.a = 4.334      # Å
        self.c = 40.91      # Å (完整c轴)
        
        # Pauli矩阵
        self.I2 = np.eye(2, dtype=complex)
        self.sx = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sz = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # τ矩阵
        self.t0 = np.eye(2, dtype=complex)
        self.t1 = np.array([[0, 1], [1, 0]], dtype=complex)
        self.t2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.t3 = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Γ矩阵（第755行定义）
        # Γ1,2,..,5 = (τ1σ3, -τ1σ2, τ1σ1, τ3σ0, τ0σ3)
        self.Gamma1 = np.kron(self.t1, self.sz)  # τ1⊗σ3
        self.Gamma2 = -np.kron(self.t1, self.sy) # -τ1⊗σ2
        self.Gamma3 = np.kron(self.t1, self.sx)  # τ1⊗σ1
        self.Gamma4 = np.kron(self.t3, self.I2)  # τ3⊗σ0
        self.Gamma5 = np.kron(self.t0, self.sz)  # τ0⊗σ3
        
        print("MnBi2Te4 Bulk Model (Simplified - Line 755 parameters)")
        print(f"  Parameters:")
        print(f"    A1={self.A1} eV·Å, A2={self.A2} eV·Å")
        print(f"    M0={self.M0} eV, M1={self.M1} eV·Å²")
        print(f"    M2={self.M2} eV·Å², m5={self.m5} eV")
        print(f"    a={self.a} Å, c={self.c} Å")
    
    def hamiltonian(self, kx: float, ky: float, kz: float, 
                    afm_layer: int = 0) -> np.ndarray:
        """
        简化的bulk哈密顿量 - 第755行公式
        
        H = Σ_i d_i Γ_i
        
        d1 = A1/c * sin(kz*c)
        d2 = A2/a * sin(kx*a)
        d3 = -A2/a * cos(ky*a)
        d4 = M0 + 2*M1/c² * (1-cos(kz*c)) + 2*M2/a² * [2-cos(kx*a)-cos(ky*a)]
        d5 = (-1)^l * m5  (l是层编号)
        """
        a = self.a
        c = self.c
        
        # 计算d系数
        d1 = self.A1/c * np.sin(kz*c)
        d2 = self.A2/a * np.sin(kx*a)
        d3 = -self.A2/a * np.cos(ky*a)
        d4 = (self.M0 + 
              2*self.M1/(c**2) * (1 - np.cos(kz*c)) +
              2*self.M2/(a**2) * (2 - np.cos(kx*a) - np.cos(ky*a)))
        
        # AFM: d5 = (-1)^l * m5
        # l=0(A层): +m5, l=1(B层): -m5
        sign = 1.0 if afm_layer == 0 else -1.0
        d5 = sign * self.m5
        
        # 构建哈密顿量
        H = (d1*self.Gamma1 + d2*self.Gamma2 + d3*self.Gamma3 + 
             d4*self.Gamma4 + d5*self.Gamma5)
        
        return H
    
    def solve_bands(self, kx: float, ky: float, kz: float,
                    afm_layer: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解能带
        """
        H = self.hamiltonian(kx, ky, kz, afm_layer)
        
        # 确保厄米
        H = 0.5 * (H + H.conj().T)
        
        try:
            E, V = linalg.eigh(H)
        except linalg.LinAlgError:
            eigenvalues, V = linalg.eig(H)
            E = np.real(eigenvalues)
            idx = np.argsort(E)
            E = E[idx]
            V = V[:, idx]
        
        return E, V


if __name__ == "__main__":
    print("="*70)
    print("Testing Simplified Bulk Model")
    print("="*70)
    
    model = MnBi2Te4_Bulk_Simple()
    
    # Γ点
    print("\n" + "="*70)
    print("Γ point (0,0,0)")
    print("="*70)
    
    E_gamma, _ = model.solve_bands(0.0, 0.0, 0.0, afm_layer=0)
    
    for i, E in enumerate(E_gamma):
        print(f"  Band {i+1}: {E:10.4f} eV = {E*1000:10.1f} meV")
    
    gap = E_gamma[2] - E_gamma[1]
    print(f"\nBand gap: {gap*1000:.1f} meV")
    
    # 能量范围检查
    print("\n" + "="*70)
    print("Energy range check")
    print("="*70)
    
    # 使用小k值测试（k单位：Å⁻¹）
    test_points = [
        ("Γ", 0.0, 0.0, 0.0),
        ("Small k1", 0.05, 0.0, 0.0),
        ("Small k2", 0.1, 0.0, 0.0),
        ("Small k3", 0.1, 0.1, 0.0),
        ("Small kz", 0.0, 0.0, 0.05),
    ]
    
    for name, kx, ky, kz in test_points:
        E, _ = model.solve_bands(kx, ky, kz, afm_layer=0)
        print(f"  {name:12s}: E ∈ [{E.min():.3f}, {E.max():.3f}] eV, gap={E[2]-E[1]:.3f} eV")
