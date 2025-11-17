"""
MnBi2Te4 精确紧束缚模型 - 基于 Nature SI (TBmodel.pdf)
Accurate Tight-Binding Model for MnBi2Te4 based on Nature Supplementary Information

核心课题：Disorder-Filtered Quantum Geometry
Universal Crossover from Hidden Berry Curvature to Quantum Metric
in PT-Symmetric Antiferromagnets

参考文献：
- Nature Supplementary Information (TBmodel.pdf) - 精确参数
- workflow.pdf - 计算流程和核心物理

物理图像：
- 基组：|P1z+,↑⟩, |P2z-,↑⟩, |P1z+,↓⟩, |P2z-,↓⟩
- 4带模型：Bi pz和Te pz轨道形成的bonding/anti-bonding态
- 反铁磁序：层间交替磁化 mA=30meV, mB=-30meV
- 6SL结构：6个septuple layer的层状体系

Author: Yue
Date: 2025-11-18
"""

import numpy as np
import numpy.linalg as linalg  # 使用numpy.linalg避免SciPy兼容性问题
from typing import Tuple, Optional

# 注意：由于NumPy 2.x兼容性问题，matplotlib暂时无法使用
# 用户需要先修复NumPy版本：conda install 'numpy<2.0' -y
# import matplotlib.pyplot as plt


class MnBi2Te4_Exact:
    """
    MnBi2Te4 精确紧束缚模型
    
    基于 Nature SI 的完整参数，用于研究：
    1. Hidden Berry Curvature Dipole (BCD)
    2. Quantum Metric Dipole (QMD)
    3. 无序对量子几何的过滤效应
    """
    
    def __init__(
        self,
        n_layers: int = 6,
        # === 精确参数 from TBmodel.pdf Eq. S9 ===
        C0: float = -0.0048,         # eV
        C1: float = 2.7232,          # eV·Å²
        C2: float = 0.0,             # eV·Å²
        M0: float = -0.1165,         # eV
        M1: float = 11.9048,         # eV·Å²
        M2: float = 9.4048,          # eV·Å²
        A1: float = 4.0535,          # eV·Å
        A2: float = 3.1964,          # eV·Å
        a: float = 4.334,            # Å (面内晶格常数)
        az: float = 13.64,           # Å (层间距离, c/3)
        m_AFM: float = 0.030,        # eV (30 meV, 交换场)
    ):
        """
        初始化 MnBi2Te4 模型
        
        Parameters:
            n_layers: 层数（默认6层）
            C0, C1, C2, M0, M1, M2, A1, A2: 紧束缚参数
            a: 面内晶格常数
            az: 层间距离
            m_AFM: 反铁磁交换场强度
        """
        self.n_layers = n_layers
        
        # 存储参数
        self.C0, self.C1, self.C2 = C0, C1, C2
        self.M0, self.M1, self.M2 = M0, M1, M2
        self.A1, self.A2 = A1, A2
        self.a, self.az = a, az
        self.m_AFM = m_AFM
        
        # 计算派生参数 (Eq. S8)
        self.e0 = C0 + 2*C1/(az**2) + 4*C2/(a**2)
        self.e5 = M0 + 2*M1/(az**2) + 4*M2/(a**2)
        self.t0 = 2*C2/(3*a**2)
        self.tz0 = C1/(az**2)
        self.t1 = -A2/(3*a)
        self.tz3 = -A1/(2*az)
        self.t5 = 2*M2/(3*a**2)
        self.tz5 = M1/(az**2)
        
        # Pauli矩阵
        self.sigma_0 = np.eye(2, dtype=complex)
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        self.tau_0 = np.eye(2, dtype=complex)
        self.tau_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.tau_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.tau_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Gamma矩阵 (Eq. S7)
        self.Gamma0 = np.eye(4, dtype=complex)  # I_4
        self.Gamma1 = np.kron(self.sigma_x, self.tau_x)
        self.Gamma2 = np.kron(self.sigma_y, self.tau_x)
        self.Gamma3 = np.kron(self.sigma_z, self.tau_x)
        self.Gamma4 = np.kron(self.sigma_0, self.tau_y)
        self.Gamma5 = np.kron(self.sigma_0, self.tau_z)
        
        print(f"MnBi2Te4 Model Initialized ({n_layers} layers)")
        print(f"  Parameters from Nature SI Eq. S9:")
        print(f"    C0={C0} eV, C1={C1} eV·Å², M0={M0} eV")
        print(f"    M1={M1} eV·Å², M2={M2} eV·Å²")
        print(f"    A1={A1} eV·Å, A2={A2} eV·Å")
        print(f"    a={a} Å, az={az} Å")
        print(f"    m_AFM={m_AFM*1000} meV")
        print(f"  Derived (Eq. S8):")
        print(f"    e0={self.e0:.6f} eV, e5={self.e5:.6f} eV")
        print(f"    t0={self.t0:.6f} eV, tz0={self.tz0:.6f} eV")
        print(f"    t1={self.t1:.6f} eV, tz3={self.tz3:.6f} eV")
        print(f"    t5={self.t5:.6f} eV, tz5={self.tz5:.6f} eV")
    
    def hamiltonian_bulk(self, kx: float, ky: float, kz: float) -> np.ndarray:
        """
        体块哈密顿量 (无磁化) - Eq. S7
        
        hTB = [e0 - 2t0(cos k1a + cos k2a + cos k3a) - 2tz0 cos k4az]Γ0
              - t1(2 sin k1a - sin k2a - sin k3a)Γ1
              - √3 t1(sin k2a - sin k3a)Γ2
              - 2tz3 sin k4az Γ3
              - 2t4(sin k1a + sin k2a + sin k3a)Γ4
              + [e5 - 2t5(cos k1a + cos k2a + cos k3a) - 2tz5 cos k4az]Γ5
        
        晶体动量：
            k1 = kx
            k2 = (−kx + √3 ky)/2
            k3 = (−kx − √3 ky)/2
            k4 = kz
        """
        # 计算晶体动量
        k1 = kx
        k2 = 0.5 * (-kx + np.sqrt(3)*ky)
        k3 = 0.5 * (-kx - np.sqrt(3)*ky)
        k4 = kz
        
        # 三角函数项
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
        
        # 注意：t4在原文中没有明确给出，这里设为0
        # H += -2 * t4 * (sin_k1a + sin_k2a + sin_k3a) * self.Gamma4
        
        H += (self.e5 - 2*self.t5*(cos_k1a + cos_k2a + cos_k3a)
              - 2*self.tz5*cos_k4az) * self.Gamma5
        
        return H
    
    def hamiltonian_AFM_layer(self, layer_index: int) -> np.ndarray:
        """
        单层的反铁磁哈密顿量 - Eq. S10
        
        hAFM = mA·s⊗τ0  (A层, 奇数层)
        hAFM = mB·s⊗τ0  (B层, 偶数层)
        
        其中 mA = m_AFM * ẑ, mB = -m_AFM * ẑ (反平行)
        
        Parameters:
            layer_index: 层编号 (0, 1, ..., n_layers-1)
        
        Returns:
            4x4反铁磁哈密顿量
        """
        # 判断奇偶层
        if layer_index % 2 == 0:  # A层（偶数索引）
            m_sign = 1.0
        else:  # B层（奇数索引）
            m_sign = -1.0
        
        # m·s = m_z * sigma_z (只有z分量)
        H_AFM = m_sign * self.m_AFM * np.kron(self.sigma_z, self.tau_0)
        
        return H_AFM
    
    def hamiltonian_multilayer(
        self, 
        kx: float, 
        ky: float,
        Ez: float = 0.0,
        disorder_scalar: Optional[np.ndarray] = None,
        disorder_magnetic: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        多层系统的完整哈密顿量
        
        H_total = Σ_layers [H_bulk(kx,ky,0) + H_AFM + H_disorder]
                  + H_interlayer_coupling
                  + H_gate_field
        
        Parameters:
            kx, ky: 面内动量
            Ez: 垂直栅压电场 (用于分离BCD和QMD)
            disorder_scalar: 标量无序势 [n_layers, 4] (每层每轨道)
            disorder_magnetic: 磁性无序 [n_layers, 3] (每层的δm向量)
        
        Returns:
            完整哈密顿量 [4*n_layers, 4*n_layers]
        """
        dim = 4 * self.n_layers  # 总维度
        H_total = np.zeros((dim, dim), dtype=complex)
        
        # === 1. 每层的哈密顿量 ===
        for i in range(self.n_layers):
            # 单层块的起始索引
            idx = 4 * i
            
            # (a) 面内动力学 (kz=0的bulk哈密顿)
            H_layer = self.hamiltonian_bulk(kx, ky, kz=0.0)
            
            # (b) 反铁磁序
            H_layer += self.hamiltonian_AFM_layer(i)
            
            # (c) 标量无序
            if disorder_scalar is not None:
                V_disorder = np.diag(disorder_scalar[i, :])
                H_layer += V_disorder
            
            # (d) 磁性无序
            if disorder_magnetic is not None:
                dm = disorder_magnetic[i, :]  # [mx, my, mz]
                H_mag_disorder = (
                    dm[0] * np.kron(self.sigma_x, self.tau_0) +
                    dm[1] * np.kron(self.sigma_y, self.tau_0) +
                    dm[2] * np.kron(self.sigma_z, self.tau_0)
                )
                H_layer += H_mag_disorder
            
            # 放入总哈密顿量
            H_total[idx:idx+4, idx:idx+4] = H_layer
        
        # === 2. 层间耦合 ===
        # 通过离散化 -2*tz0*cos(kz*az)*Γ0 得到：
        # cos(kz*az) → [c†(n)c(n+1) + c†(n+1)c(n) + 2]/2 的期望 → [c†(n)c(n+1) + h.c.]/2
        # 所以: -2*tz0*cos(kz*az) → -2*tz0*[{c†(n)c(n+1)+h.c.}/2 + 1]
        #                          = -tz0*[c†(n)c(n+1)+h.c.] - 2*tz0 (对角项)
        # 类似地对Γ3项：-2*tz3*sin(kz*az) → -i*tz3*[c†(n)c(n+1) - c†(n+1)c(n)]
        
        for i in range(self.n_layers - 1):
            idx_i = 4 * i
            idx_j = 4 * (i + 1)
            
            # 层间跳跃（最近邻）
            # 从cos项: -tz0*Γ0
            # 从sin项: 实空间中sin变换为反对称，但这里我们用最简单的最近邻近似
            # 注意：原文说"symmetry allowed interlayer hoppings"，最简单就是Γ0项
            H_hop = -self.tz0 * self.Gamma0
            
            H_total[idx_i:idx_i+4, idx_j:idx_j+4] = H_hop
            H_total[idx_j:idx_j+4, idx_i:idx_i+4] = H_hop.T
        
        # cos项的对角贡献：每层额外加 -2*tz0*Γ0
        for i in range(self.n_layers):
            idx = 4 * i
            # 两个最近邻的贡献 (除了边界层)
            if i == 0 or i == self.n_layers - 1:
                # 边界：只有一个最近邻
                H_total[idx:idx+4, idx:idx+4] += -self.tz0 * self.Gamma0
            else:
                # 体内：两个最近邻
                H_total[idx:idx+4, idx:idx+4] += -2.0 * self.tz0 * self.Gamma0
        
        # === 3. 栅压电场 (用于BCD/QMD分离) ===
        if Ez != 0.0:
            for i in range(self.n_layers):
                idx = 4 * i
                # 电势沿z线性变化：V(z) = Ez * z
                # z坐标：从-(n_layers-1)/2到+(n_layers-1)/2
                z_coord = (i - (self.n_layers - 1) / 2) * self.az
                potential = Ez * z_coord
                
                # 加到对角项
                H_total[idx:idx+4, idx:idx+4] += potential * self.Gamma0
        
        # === 4. 确保厄米性（数值稳定性）===
        # 强制对称化：H = (H + H†)/2
        H_total = 0.5 * (H_total + H_total.conj().T)
        
        return H_total
    
    def solve_bands(
        self, 
        kx: float, 
        ky: float,
        Ez: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解能带和本征态
        
        Returns:
            energies: 能量本征值 [4*n_layers]
            eigenvectors: 本征矢 [4*n_layers, 4*n_layers]
        """
        H = self.hamiltonian_multilayer(kx, ky, Ez=Ez)
        
        # 检查厄米性
        hermiticity_error = np.max(np.abs(H - H.conj().T))
        if hermiticity_error > 1e-10:
            print(f"Warning: Hamiltonian not Hermitian at k=({kx:.6f}, {ky:.6f})")
            print(f"  Max |H - H†|: {hermiticity_error:.2e}")
        
        try:
            energies, eigenvectors = linalg.eigh(H)
        except linalg.LinAlgError as e:
            print(f"\nLinAlgError at k=({kx:.6f}, {ky:.6f}), Ez={Ez:.6f}")
            print(f"  Error: {e}")
            print(f"  Hamiltonian condition number: {np.linalg.cond(H):.2e}")
            print(f"  Max |H|: {np.max(np.abs(H)):.2e}")
            print(f"  Hermiticity error: {hermiticity_error:.2e}")
            
            # 尝试使用更稳健的求解器
            print("  Attempting to use eig() instead of eigh()...")
            eigenvalues, eigenvectors = linalg.eig(H)
            # eig返回的可能是复数，取实部
            energies = np.real(eigenvalues)
            # 排序
            idx = np.argsort(energies)
            energies = energies[idx]
            eigenvectors = eigenvectors[:, idx]
            print(f"  Success with eig()")
        
        return energies, eigenvectors
    
    def velocity_operator(self, kx: float, ky: float, direction: str = 'x') -> np.ndarray:
        """
        速度算符 v_μ = ∂H/∂k_μ (数值微分)
        
        Parameters:
            kx, ky: k点坐标
            direction: 'x' or 'y'
        
        Returns:
            速度算符矩阵
        """
        dk = 1e-5  # 微分步长
        
        if direction == 'x':
            H_plus = self.hamiltonian_multilayer(kx + dk, ky)
            H_minus = self.hamiltonian_multilayer(kx - dk, ky)
        else:  # 'y'
            H_plus = self.hamiltonian_multilayer(kx, ky + dk)
            H_minus = self.hamiltonian_multilayer(kx, ky - dk)
        
        v = (H_plus - H_minus) / (2 * dk)
        return v


def test_model():
    """测试模型构建"""
    print("="*60)
    print("Testing MnBi2Te4 Exact Model")
    print("="*60)
    
    # 创建模型
    model = MnBi2Te4_Exact(n_layers=6)
    
    # 测试Γ点能带
    print("\nSolving band structure at Γ point...")
    energies, _ = model.solve_bands(0.0, 0.0)
    
    print(f"\nEnergies at Γ (kx=0, ky=0):")
    for i, E in enumerate(energies):
        print(f"  Band {i:2d}: {E:10.6f} eV")
    
    # 检查能隙
    mid = len(energies) // 2
    gap = energies[mid] - energies[mid-1]
    print(f"\nEnergy gap at Γ: {gap*1000:.2f} meV")
    
    # 测试不同k点
    print("\nTesting different k points...")
    k_points = [
        (0.0, 0.0),  # Γ
        (0.5, 0.0),  # M
        (0.5, 0.5),  # K
    ]
    
    for kx, ky in k_points:
        E, _ = model.solve_bands(kx, ky)
        print(f"  k=({kx:.1f}, {ky:.1f}): E_min={E.min():.3f}, E_max={E.max():.3f}")
    
    print("\n" + "="*60)
    print("✓ Model test completed!")
    print("="*60)


if __name__ == "__main__":
    test_model()
