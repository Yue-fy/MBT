"""
MnBi2Te4 Bulk Model - 最简单的验证
严格按照Nature SI的公式实现
"""
import numpy as np
from numpy import linalg
from typing import Tuple

class MnBi2Te4_Bulk:
    """
    MnBi2Te4 bulk 模型 (3D)
    基于Nature SI Eq. S7-S9
    """
    
    def __init__(self):
        # Nature SI Eq. S9 参数
        self.C0 = -0.0048    # eV
        self.C1 = 2.7232     # eV·Å²
        self.C2 = 0.0        # eV·Å²
        self.M0 = -0.1165    # eV
        self.M1 = 11.9048    # eV·Å²
        self.M2 = 9.4048     # eV·Å²
        self.A1 = 4.0535     # eV·Å
        self.A2 = 3.1964     # eV·Å
        
        self.a = 4.334       # Å
        self.az = 13.64      # Å (c/3)
        
        # AFM交换场（bulk中每层交替）
        self.m_AFM = 0.030   # eV (30 meV)
        
        # 派生参数 Eq. S8
        self.e0 = self.C0 + 2*self.C1/(self.az**2) + 4*self.C2/(self.a**2)
        self.e5 = self.M0 + 2*self.M1/(self.az**2) + 4*self.M2/(self.a**2)
        self.t0 = 2*self.C2/(3*self.a**2)
        self.tz0 = self.C1/(self.az**2)
        self.t1 = -self.A2/(3*self.a)
        self.tz3 = -self.A1/(2*self.az)
        self.t5 = 2*self.M2/(3*self.a**2)
        self.tz5 = self.M1/(self.az**2)
        
        # Pauli矩阵
        self.I2 = np.eye(2, dtype=complex)
        self.sx = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sz = np.array([[1, 0], [0, -1]], dtype=complex)
        
        self.tx = np.array([[0, 1], [1, 0]], dtype=complex)
        self.ty = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.tz = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Gamma矩阵 Eq. S7
        self.Gamma0 = np.eye(4, dtype=complex)
        self.Gamma1 = np.kron(self.sx, self.tx)
        self.Gamma2 = np.kron(self.sy, self.tx)
        self.Gamma3 = np.kron(self.sz, self.tx)
        self.Gamma4 = np.kron(self.I2, self.ty)
        self.Gamma5 = np.kron(self.I2, self.tz)
        
        print("MnBi2Te4 Bulk Model Initialized")
        print(f"  Parameters (Nature SI Eq. S9):")
        print(f"    C0={self.C0} eV, M0={self.M0} eV")
        print(f"    A1={self.A1} eV·Å, A2={self.A2} eV·Å")
        print(f"  Derived (Eq. S8):")
        print(f"    e0={self.e0:.6f} eV, e5={self.e5:.6f} eV")
        print(f"    t0={self.t0:.6f} eV, t1={self.t1:.6f} eV")
        print(f"    tz0={self.tz0:.6f} eV, tz3={self.tz3:.6f} eV")
    
    def hamiltonian(self, kx: float, ky: float, kz: float, 
                    include_AFM: bool = True, afm_layer: int = 0) -> np.ndarray:
        """
        Bulk 哈密顿量 - Eq. S7
        
        hTB = [e0 - 2t0(cos k1a + cos k2a + cos k3a) - 2tz0 cos k4az]Γ0
              - t1(2 sin k1a - sin k2a - sin k3a)Γ1
              - √3 t1(sin k2a - sin k3a)Γ2
              - 2tz3 sin k4az Γ3
              + [e5 - 2t5(cos k1a + cos k2a + cos k3a) - 2tz5 cos k4az]Γ5
        
        Parameters:
            kx, ky, kz: 动量 (单位: Å⁻¹)
            include_AFM: 是否包含AFM序
            afm_layer: AFM层编号 (0=A层向上, 1=B层向下)
        """
        # 晶体动量
        k1 = kx
        k2 = 0.5 * (-kx + np.sqrt(3)*ky)
        k3 = 0.5 * (-kx - np.sqrt(3)*ky)
        k4 = kz
        
        # 三角函数
        cos_k1a = np.cos(k1 * self.a)
        cos_k2a = np.cos(k2 * self.a)
        cos_k3a = np.cos(k3 * self.a)
        cos_k4az = np.cos(k4 * self.az)
        
        sin_k1a = np.sin(k1 * self.a)
        sin_k2a = np.sin(k2 * self.a)
        sin_k3a = np.sin(k3 * self.a)
        sin_k4az = np.sin(k4 * self.az)
        
        # 构建哈密顿量
        H = (self.e0 - 2*self.t0*(cos_k1a + cos_k2a + cos_k3a) 
             - 2*self.tz0*cos_k4az) * self.Gamma0
        
        H += -self.t1 * (2*sin_k1a - sin_k2a - sin_k3a) * self.Gamma1
        
        H += -np.sqrt(3) * self.t1 * (sin_k2a - sin_k3a) * self.Gamma2
        
        H += -2 * self.tz3 * sin_k4az * self.Gamma3
        
        # 注意：Γ4项在Eq. S8中没有定义t4，所以忽略
        
        H += (self.e5 - 2*self.t5*(cos_k1a + cos_k2a + cos_k3a)
              - 2*self.tz5*cos_k4az) * self.Gamma5
        
        # 添加AFM序 Eq. S10
        if include_AFM:
            # mA·s⊗τ0 for A层, mB·s⊗τ0 for B层
            # mA = +m_AFM * ẑ, mB = -m_AFM * ẑ
            sign = 1.0 if afm_layer == 0 else -1.0
            H_AFM = sign * self.m_AFM * np.kron(self.sz, self.I2)
            H += H_AFM
        
        return H
    
    def solve_bands(self, kx: float, ky: float, kz: float,
                    include_AFM: bool = True, afm_layer: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解能带
        """
        H = self.hamiltonian(kx, ky, kz, include_AFM, afm_layer)
        
        # 确保厄米
        H = 0.5 * (H + H.conj().T)
        
        try:
            E, V = linalg.eigh(H)
        except linalg.LinAlgError:
            # 使用eig备选
            eigenvalues, V = linalg.eig(H)
            E = np.real(eigenvalues)
            idx = np.argsort(E)
            E = E[idx]
            V = V[:, idx]
        
        return E, V


if __name__ == "__main__":
    print("="*70)
    print("Testing Bulk Model")
    print("="*70)
    
    model = MnBi2Te4_Bulk()
    
    # 测试Γ点 (kx=ky=kz=0)
    print("\n" + "="*70)
    print("Γ point (0,0,0) - with AFM")
    print("="*70)
    
    E_gamma, _ = model.solve_bands(0.0, 0.0, 0.0, include_AFM=True, afm_layer=0)
    
    for i, E in enumerate(E_gamma):
        print(f"  Band {i}: {E:10.4f} eV = {E*1000:10.1f} meV")
    
    gap = E_gamma[2] - E_gamma[1]
    print(f"\nBand gap: {gap*1000:.1f} meV")
    
    # 测试Z点 (0,0,π/az)
    print("\n" + "="*70)
    print("Z point (0,0,π/az) - with AFM")
    print("="*70)
    
    kz_Z = np.pi / model.az
    E_Z, _ = model.solve_bands(0.0, 0.0, kz_Z, include_AFM=True, afm_layer=0)
    
    for i, E in enumerate(E_Z):
        print(f"  Band {i}: {E:10.4f} eV = {E*1000:10.1f} meV")
    
    gap_Z = E_Z[2] - E_Z[1]
    print(f"\nBand gap: {gap_Z*1000:.1f} meV")
    
    # 能量尺度检查
    print("\n" + "="*70)
    print("Energy scale check")
    print("="*70)
    
    test_points = [
        ("Γ", 0.0, 0.0, 0.0),
        ("X", 0.5*np.pi/model.a, 0.0, 0.0),
        ("Z", 0.0, 0.0, np.pi/model.az),
    ]
    
    for name, kx, ky, kz in test_points:
        E, _ = model.solve_bands(kx, ky, kz, include_AFM=True)
        print(f"  {name}: E ∈ [{E.min():.3f}, {E.max():.3f}] eV")
