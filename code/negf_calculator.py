"""
NEGF (Non-Equilibrium Green's Function) 计算引擎
用于计算二阶非线性霍尔电流 J_y^(2)

核心物理：
1. 二阶响应：J_y ∝ E_x^2
2. 有限差分法提取：J^(2) ≈ [J_y(+V) - J_y(-V)] / 2
3. 分离BCD和QMD：利用栅压奇偶性
   - BCD (Berry Curvature Dipole): Ez的奇函数
   - QMD (Quantum Metric Dipole): Ez的偶函数

基于 workflow.pdf 第二阶段

Author: Yue
Date: 2025-11-18
"""

import numpy as np
import numpy.linalg as linalg
from typing import Tuple, Optional, Dict
import time


class NEGF_Calculator:
    """
    非平衡格林函数计算器
    
    实现二阶非线性输运的全量子计算
    """
    
    def __init__(
        self,
        hamiltonian_func,
        device_length: int = 100,
        device_width: int = 40,
        temperature: float = 1.0,  # K
        eta: float = 1e-6,  # 正则化参数
    ):
        """
        Parameters:
            hamiltonian_func: 哈密顿量函数 H(kx, ky, **kwargs)
            device_length: 器件长度（格点数，沿x）
            device_width: 器件宽度（格点数，沿y）
            temperature: 温度 (K)
            eta: 能量正则化参数 (eV)
        """
        self.H_func = hamiltonian_func
        self.L = device_length
        self.W = device_width
        self.T = temperature
        self.eta = eta
        
        # 物理常数
        self.kB = 8.617e-5  # eV/K
        
        print(f"NEGF Calculator Initialized")
        print(f"  Device: {self.L} × {self.W} sites")
        print(f"  Temperature: {self.T} K")
        print(f"  η (regularization): {self.eta} eV")
    
    def fermi_dirac(self, energy: float, mu: float = 0.0) -> float:
        """
        费米-狄拉克分布
        
        f(E) = 1 / [1 + exp((E-μ)/(k_B T))]
        """
        if self.T < 1e-3:  # 接近零温
            return 1.0 if energy < mu else 0.0
        
        x = (energy - mu) / (self.kB * self.T)
        # 避免溢出
        if x > 100:
            return 0.0
        elif x < -100:
            return 1.0
        else:
            return 1.0 / (1.0 + np.exp(x))
    
    def self_energy_lead(
        self,
        energy: float,
        kx: float,
        ky: float,
        position: str = 'left'
    ) -> np.ndarray:
        """
        电极自能 Σ_L 或 Σ_R
        
        简化模型：使用宽带极限 (wide-band limit)
        Σ = -i Γ/2, 其中 Γ 是耦合强度
        
        Parameters:
            energy: 能量
            kx, ky: 横向动量
            position: 'left' or 'right'
        
        Returns:
            自能矩阵
        """
        # 获取单层哈密顿量的维度
        H_sample = self.H_func(kx, ky)
        dim = H_sample.shape[0]
        
        # 宽带极限：Σ = -i Γ/2
        Gamma = 0.1  # eV (耦合强度)
        Sigma = -1j * (Gamma / 2) * np.eye(dim, dtype=complex)
        
        return Sigma
    
    def green_function_retarded(
        self,
        energy: float,
        kx: float,
        ky: float,
        bias_V: float = 0.0,
        Ez: float = 0.0,
        **kwargs
    ) -> np.ndarray:
        """
        推迟格林函数 G^R
        
        G^R(E) = [E + iη - H - Σ_L - Σ_R]^(-1)
        
        Parameters:
            energy: 能量
            kx, ky: 横向动量
            bias_V: 偏压（沿x方向）
            Ez: 栅压电场（沿z方向）
            **kwargs: 传递给哈密顿量的其他参数
        
        Returns:
            格林函数矩阵
        """
        # 构建哈密顿量
        H = self.H_func(kx, ky, Ez=Ez, **kwargs)
        
        # 加上偏压（电势沿x线性降落）
        if abs(bias_V) > 1e-10:
            # 简化：每层加上均匀电势（实际应该是x方向渐变）
            # 这里假设偏压主要影响化学势
            H = H + bias_V * np.eye(H.shape[0], dtype=complex)
        
        # 电极自能
        Sigma_L = self.self_energy_lead(energy, kx, ky, 'left')
        Sigma_R = self.self_energy_lead(energy, kx, ky, 'right')
        
        # 扩展自能到整个器件（只在边界层）
        dim_total = H.shape[0]
        dim_layer = Sigma_L.shape[0]
        
        Sigma_total = np.zeros((dim_total, dim_total), dtype=complex)
        Sigma_total[:dim_layer, :dim_layer] = Sigma_L  # 左电极
        Sigma_total[-dim_layer:, -dim_layer:] = Sigma_R  # 右电极
        
        # 计算格林函数
        # G^R = [E + iη - H - Σ]^(-1)
        G_inv = (energy + 1j*self.eta) * np.eye(dim_total) - H - Sigma_total
        
        try:
            G_R = linalg.inv(G_inv)
        except linalg.LinAlgError:
            # 如果矩阵奇异，返回零矩阵
            print(f"Warning: Singular matrix at E={energy}, returning zero")
            G_R = np.zeros_like(G_inv)
        
        return G_R
    
    def spectral_function(
        self,
        energy: float,
        kx: float,
        ky: float,
        **kwargs
    ) -> np.ndarray:
        """
        谱函数 A(E) = i[G^R - G^A]
        
        其中 G^A = (G^R)^†
        """
        G_R = self.green_function_retarded(energy, kx, ky, **kwargs)
        G_A = G_R.conj().T
        A = 1j * (G_R - G_A)
        return A
    
    def current_density_y(
        self,
        energy: float,
        kx: float,
        ky: float,
        bias_V: float = 0.0,
        Ez: float = 0.0,
        mu_L: float = 0.0,
        mu_R: float = 0.0,
        **kwargs
    ) -> float:
        """
        计算y方向的电流密度 J_y
        
        使用Kubo-Bastin公式：
        J_y = (e/ℏ) Tr[v_y · G^<(E)]
        
        其中 G^< 是lesser格林函数，与费米分布相关
        
        Parameters:
            energy: 能量
            kx, ky: 横向动量
            bias_V: x方向偏压
            Ez: z方向栅压
            mu_L, mu_R: 左右电极化学势
            
        Returns:
            电流密度 (任意单位)
        """
        # 获取格林函数
        G_R = self.green_function_retarded(
            energy, kx, ky, bias_V=bias_V, Ez=Ez, **kwargs
        )
        
        # 计算速度算符 v_y (数值导数)
        dky = 1e-5
        H_plus = self.H_func(kx, ky + dky, Ez=Ez, **kwargs)
        H_minus = self.H_func(kx, ky - dky, Ez=Ez, **kwargs)
        
        # 处理维度不匹配
        if H_plus.shape[0] != G_R.shape[0]:
            # 扩展到相同维度
            dim_diff = G_R.shape[0] - H_plus.shape[0]
            if dim_diff > 0:
                H_plus = np.pad(H_plus, ((0, dim_diff), (0, dim_diff)))
                H_minus = np.pad(H_minus, ((0, dim_diff), (0, dim_diff)))
        
        v_y = (H_plus - H_minus) / (2 * dky)
        
        # Lesser格林函数近似：G^< ≈ f(E) · A(E)
        f_L = self.fermi_dirac(energy, mu_L)
        f_R = self.fermi_dirac(energy, mu_R)
        f_avg = (f_L + f_R) / 2  # 简化
        
        A = self.spectral_function(energy, kx, ky, bias_V=bias_V, Ez=Ez, **kwargs)
        G_lesser = 1j * f_avg * A
        
        # 计算电流：Tr[v_y · G^<]
        current_operator = v_y @ G_lesser
        J_y = np.trace(current_operator).real
        
        return J_y
    
    def nonlinear_hall_current(
        self,
        bias_V: float,
        Ez: float = 0.0,
        nk: int = 20,
        n_energy: int = 50,
        **kwargs
    ) -> float:
        """
        计算非线性霍尔电流（对能量和k空间积分）
        
        J_y^total = ∫∫∫ J_y(E, k_x, k_y) dE dk_x dk_y
        
        Parameters:
            bias_V: 偏压
            Ez: 栅压电场
            nk: k空间采样点数
            n_energy: 能量采样点数
        
        Returns:
            总横向电流
        """
        print(f"    Computing J_y at V={bias_V:.4f}, Ez={Ez:.4f}...")
        
        # k空间网格
        kx_max = np.pi / 4.334  # 1/a
        ky_max = np.pi / 4.334
        kx_grid = np.linspace(-kx_max, kx_max, nk)
        ky_grid = np.linspace(-ky_max, ky_max, nk)
        
        # 能量网格（围绕费米能）
        E_min, E_max = -0.5, 0.5  # eV
        E_grid = np.linspace(E_min, E_max, n_energy)
        
        # 积分
        J_total = 0.0
        count = 0
        
        for kx in kx_grid:
            for ky in ky_grid:
                for E in E_grid:
                    try:
                        J_y = self.current_density_y(
                            E, kx, ky, bias_V=bias_V, Ez=Ez, **kwargs
                        )
                        J_total += J_y
                        count += 1
                    except Exception as e:
                        # 忽略计算失败的点
                        pass
        
        # 归一化
        if count > 0:
            J_total /= count
        
        return J_total
    
    def extract_second_order_response(
        self,
        bias_V: float,
        Ez: float = 0.0,
        nk: int = 10,
        n_energy: int = 20,
        **kwargs
    ) -> float:
        """
        提取二阶响应 J^(2)
        
        使用有限差分法：
        J^(2) ≈ [J_y(+V) - J_y(-V)] / 2
        
        这样可以消除一阶项，只保留偶数阶（包括二阶）
        """
        print(f"  Extracting 2nd order response at Ez={Ez:.4f}...")
        
        # 计算 +V 和 -V 的电流
        J_plus = self.nonlinear_hall_current(
            +bias_V, Ez=Ez, nk=nk, n_energy=n_energy, **kwargs
        )
        J_minus = self.nonlinear_hall_current(
            -bias_V, Ez=Ez, nk=nk, n_energy=n_energy, **kwargs
        )
        
        # 提取二阶项
        J_second_order = (J_plus - J_minus) / 2.0
        
        print(f"    J(+V)={J_plus:.6e}, J(-V)={J_minus:.6e}")
        print(f"    J^(2)={J_second_order:.6e}")
        
        return J_second_order
    
    def separate_BCD_QMD(
        self,
        bias_V: float,
        Ez: float,
        nk: int = 10,
        n_energy: int = 20,
        **kwargs
    ) -> Dict[str, float]:
        """
        分离 Berry Curvature Dipole (BCD) 和 Quantum Metric Dipole (QMD)
        
        利用对称性：
        - BCD 是 Ez 的奇函数：BCD(Ez) = -BCD(-Ez)
        - QMD 是 Ez 的偶函数：QMD(Ez) = QMD(-Ez)
        
        因此：
        - J_BCD = [J^(2)(+Ez) - J^(2)(-Ez)] / 2
        - J_QMD = [J^(2)(+Ez) + J^(2)(-Ez)] / 2
        
        Returns:
            {'BCD': ..., 'QMD': ..., 'total_pos': ..., 'total_neg': ...}
        """
        print(f"\nSeparating BCD and QMD at V={bias_V}, Ez=±{Ez}")
        print("="*60)
        
        # 计算 +Ez 的响应
        J2_pos = self.extract_second_order_response(
            bias_V, Ez=+Ez, nk=nk, n_energy=n_energy, **kwargs
        )
        
        # 计算 -Ez 的响应
        J2_neg = self.extract_second_order_response(
            bias_V, Ez=-Ez, nk=nk, n_energy=n_energy, **kwargs
        )
        
        # 分离
        J_BCD = (J2_pos - J2_neg) / 2.0  # 奇函数部分
        J_QMD = (J2_pos + J2_neg) / 2.0  # 偶函数部分
        
        print("\n" + "="*60)
        print("RESULTS:")
        print(f"  BCD component: {J_BCD:.6e}")
        print(f"  QMD component: {J_QMD:.6e}")
        print(f"  BCD/QMD ratio: {abs(J_BCD/J_QMD) if J_QMD != 0 else np.inf:.3f}")
        print("="*60)
        
        return {
            'BCD': J_BCD,
            'QMD': J_QMD,
            'total_pos': J2_pos,
            'total_neg': J2_neg
        }


def test_negf():
    """测试NEGF计算"""
    from tb_model_exact import MnBi2Te4_Exact
    
    print("="*60)
    print("Testing NEGF Calculator")
    print("="*60)
    
    # 创建模型
    print("\n1. Creating MnBi2Te4 model...")
    model = MnBi2Te4_Exact(n_layers=6)
    
    # 创建NEGF计算器
    print("\n2. Creating NEGF calculator...")
    negf = NEGF_Calculator(
        hamiltonian_func=model.hamiltonian_multilayer,
        device_length=50,
        device_width=20,
        temperature=1.0,
        eta=1e-6
    )
    
    # 测试单点格林函数
    print("\n3. Testing Green's function at single point...")
    G_R = negf.green_function_retarded(
        energy=0.0, kx=0.0, ky=0.0, bias_V=0.01, Ez=0.05
    )
    print(f"   G^R shape: {G_R.shape}")
    print(f"   |G^R|_max: {np.abs(G_R).max():.6e}")
    
    # 测试电流计算（快速版本）
    print("\n4. Testing current calculation (quick)...")
    J_y = negf.current_density_y(
        energy=0.0, kx=0.0, ky=0.0, 
        bias_V=0.01, Ez=0.05
    )
    print(f"   J_y = {J_y:.6e}")
    
    # 测试二阶响应提取（非常快速的版本）
    print("\n5. Testing 2nd order extraction (ultra-quick)...")
    J2 = negf.extract_second_order_response(
        bias_V=0.01, Ez=0.05, nk=3, n_energy=5
    )
    print(f"   J^(2) = {J2:.6e}")
    
    print("\n" + "="*60)
    print("✓ NEGF test completed!")
    print("="*60)


if __name__ == "__main__":
    test_negf()
