"""
简化的MnBi2Te4双层模型 - 用于验证TB方法
基于Bi2Te3的4带模型 + AFM序
"""
import numpy as np
from numpy import linalg
from typing import Tuple

class MnBi2Te4_Bilayer:
    """
    最简单的双层MnBi2Te4模型
    每层4个轨道 -> 总共8个轨道
    """
    
    def __init__(self):
        # 使用Nature SI Eq. S9的精确参数
        self.C0 = -0.0048  # eV
        self.C1 = 2.7232   # eV·Å²
        self.C2 = 0.0      # eV·Å²  ← 注意这里是0！
        self.M0 = -0.1165  # eV
        self.M1 = 11.9048  # eV·Å²
        self.M2 = 9.4048   # eV·Å²
        self.A1 = 4.0535   # eV·Å
        self.A2 = 3.1964   # eV·Å
        
        self.a = 4.334    # Å
        self.d = 13.64    # Å (= az from SI)
        
        # AFM交换场
        self.m_AFM = 0.030  # eV (30 meV)
        
        # Pauli矩阵
        self.I2 = np.eye(2, dtype=complex)
        self.sx = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sz = np.array([[1, 0], [0, -1]], dtype=complex)
        
        self.tx = np.array([[0, 1], [1, 0]], dtype=complex)
        self.ty = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.tz = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Gamma矩阵 (4x4)
        self.G0 = np.eye(4, dtype=complex)
        self.G1 = np.kron(self.sx, self.tx)
        self.G2 = np.kron(self.sy, self.tx)
        self.G3 = np.kron(self.sz, self.tx)
        self.G5 = np.kron(self.I2, self.tz)
        
        print("MnBi2Te4 Bilayer Model")
        print(f"  Parameters (simplified):")
        print(f"    M0={self.M0} eV, C0={self.C0} eV")
        print(f"    A1={self.A1} eV·Å, A2={self.A2} eV·Å")
        print(f"    Layer spacing d={self.d} Å")
        print(f"    AFM exchange={self.m_AFM*1000} meV")
    
    def h_2d(self, kx: float, ky: float) -> np.ndarray:
        """
        单层2D哈密顿量 (kz=0)
        """
        a = self.a
        
        # 晶体动量
        k1 = kx
        k2 = 0.5 * (-kx + np.sqrt(3)*ky)
        k3 = 0.5 * (-kx - np.sqrt(3)*ky)
        
        # 能量参数
        epsilon = self.C0 + 2*self.C2/(a**2) * (
            2 - np.cos(k1*a) - np.cos(k2*a) - np.cos(k3*a)
        )
        
        mass = self.M0 + 2*self.M2/(a**2) * (
            2 - np.cos(k1*a) - np.cos(k2*a) - np.cos(k3*a)
        )
        
        # 动量项
        v1 = -self.A2/(3*a) * (
            2*np.sin(k1*a) - np.sin(k2*a) - np.sin(k3*a)
        )
        v2 = -np.sqrt(3)*self.A2/(3*a) * (
            np.sin(k2*a) - np.sin(k3*a)
        )
        
        # 哈密顿量
        H = epsilon*self.G0 + v1*self.G1 + v2*self.G2 + mass*self.G5
        
        return H
    
    def h_hop(self) -> np.ndarray:
        """
        层间跳跃矩阵 (最简单形式)
        """
        # 只用最基本的层间跳跃
        t_perp = -self.C1 / (self.d**2)  # 约0.012 eV
        return t_perp * self.G0
    
    def h_afm(self, layer: int) -> np.ndarray:
        """
        AFM哈密顿量
        layer=0: 上层 (自旋向上)
        layer=1: 下层 (自旋向下)
        """
        sign = 1.0 if layer == 0 else -1.0
        return sign * self.m_AFM * np.kron(self.sz, self.I2)
    
    def hamiltonian(self, kx: float, ky: float) -> np.ndarray:
        """
        完整双层哈密顿量 (8x8)
        """
        H_total = np.zeros((8, 8), dtype=complex)
        
        # 层1
        H_total[0:4, 0:4] = self.h_2d(kx, ky) + self.h_afm(0)
        
        # 层2
        H_total[4:8, 4:8] = self.h_2d(kx, ky) + self.h_afm(1)
        
        # 层间跳跃
        H_hop = self.h_hop()
        H_total[0:4, 4:8] = H_hop
        H_total[4:8, 0:4] = H_hop.T
        
        # 强制厄米化
        H_total = 0.5 * (H_total + H_total.conj().T)
        
        return H_total
    
    def solve_bands(self, kx: float, ky: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解能带
        """
        H = self.hamiltonian(kx, ky)
        
        try:
            E, V = linalg.eigh(H)
        except linalg.LinAlgError:
            # 使用普通eig
            eigenvalues, V = linalg.eig(H)
            E = np.real(eigenvalues)
            idx = np.argsort(E)
            E = E[idx]
            V = V[:, idx]
        
        return E, V


if __name__ == "__main__":
    print("="*70)
    print("Testing Bilayer Model")
    print("="*70)
    
    model = MnBi2Te4_Bilayer()
    
    # Γ点
    print("\n" + "="*70)
    print("Γ point")
    print("="*70)
    E_gamma, _ = model.solve_bands(0.0, 0.0)
    
    for i, E in enumerate(E_gamma):
        print(f"  Band {i}: {E:10.4f} eV = {E*1000:10.1f} meV")
    
    gap = E_gamma[4] - E_gamma[3]
    print(f"\nBand gap: {gap*1000:.1f} meV")
    
    # 小k值测试
    print("\n" + "="*70)
    print("Small k test")
    print("="*70)
    
    test_k = [
        (0.0, 0.0),
        (0.05, 0.0),
        (0.1, 0.1),
    ]
    
    for kx, ky in test_k:
        E, _ = model.solve_bands(kx, ky)
        gap = E[4] - E[3]
        print(f"k=({kx:.2f},{ky:.2f}): gap={gap*1000:6.1f} meV, E_range=[{E.min():.3f}, {E.max():.3f}] eV")
