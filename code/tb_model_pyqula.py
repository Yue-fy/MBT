"""
MnBi2Te4 模型使用 PyQula 实现
Implementation using PyQula library for quantum lattice calculations

PyQula: https://github.com/joselado/pyqula
专门用于量子格点模型、拓扑物性和Berry曲率计算

Author: Yue
Date: 2025-11-17
"""

import numpy as np

try:
    # PyQula 核心模块
    from pyqula import geometry
    from pyqula import hamiltonian
    from pyqula import berry
    from pyqula import topology
    from pyqula import operators
    PYQULA_AVAILABLE = True
    print("✓ PyQula imported successfully!")
except ImportError:
    PYQULA_AVAILABLE = False
    print("✗ PyQula not available. Install: pip install git+https://github.com/joselado/pyqula")


class MnBi2Te4_PyQula:
    """
    MnBi2Te4 双层反铁磁模型 - PyQula 实现
    
    优势：
    - 自动计算Berry曲率
    - 内置拓扑不变量计算
    - 优化的k空间积分
    - 可视化工具
    """
    
    def __init__(
        self,
        a: float = 4.38,          # 晶格常数 (Å)
        t: float = 1.0,           # 跳跃 (eV)
        lambda_SO: float = 0.3,   # 自旋轨道耦合 (eV)
        M: float = 0.5,           # 交换场 (eV)
        t_perp: float = 0.2,      # 层间耦合 (eV)
        mu: float = 0.0           # 化学势 (eV)
    ):
        if not PYQULA_AVAILABLE:
            raise ImportError("PyQula is required. Install with: pip install git+https://github.com/joselado/pyqula")
        
        self.a = a
        self.t = t
        self.lambda_SO = lambda_SO
        self.M = M
        self.t_perp = t_perp
        self.mu = mu
        
        # 创建几何结构
        self.geometry = self._build_geometry()
        
        # 创建哈密顿量
        self.hamiltonian = self._build_hamiltonian()
        
    def _build_geometry(self):
        """
        创建双层三角格子几何
        
        PyQula 自动处理周期性边界条件
        """
        # 创建单层三角格子
        g = geometry.triangular_lattice()
        
        # 设置晶格常数
        g = g.supercell(1)
        g.a1 = g.a1 * self.a
        g.a2 = g.a2 * self.a
        
        # 扩展到双层
        g = g.get_supercell([1, 1, 2])  # 沿z方向复制
        
        # 设置层间距
        g.z = g.z * 3.0  # 层间距 ~3Å
        
        # 标记层指标
        for i, r in enumerate(g.r):
            if r[2] < 1.0:
                g.atoms[i] = 0  # 层1
            else:
                g.atoms[i] = 1  # 层2
        
        g.dimensionality = 2  # 2D系统
        
        return g
    
    def _build_hamiltonian(self):
        """
        构建双层反铁磁哈密顿量
        
        PyQula自动处理自旋和层自由度
        """
        h = self.geometry.get_hamiltonian()
        
        # 设置最近邻跳跃
        h.add_onsite(-self.mu)  # 化学势
        h.add_hopping(self.t)   # 最近邻跳跃
        
        # 添加自旋轨道耦合（Rashba + Kane-Mele）
        h.add_rashba(self.lambda_SO)
        
        # 添加交换场（层依赖）
        def exchange_field(r):
            """层依赖的交换场"""
            layer = 0 if r[2] < 1.0 else 1
            M_eff = self.M if layer == 0 else -self.M
            return np.array([0, 0, M_eff])
        
        h.add_zeeman(exchange_field)
        
        # 添加层间耦合
        h.add_hopping(lambda r1, r2: self.t_perp if abs(r1[2] - r2[2]) > 0.5 else 0)
        
        # 启用自旋
        h.has_spin = True
        
        return h
    
    def calculate_bands(self, kpath=None, nk=100):
        """
        计算能带结构
        
        Parameters:
            kpath: 高对称点路径 (默认: Γ-M-K-Γ)
            nk: k点数
            
        Returns:
            (k_distances, bands)
        """
        if kpath is None:
            kpath = self.hamiltonian.geometry.get_kpath()
        
        k_vec, k_dist, bands = self.hamiltonian.get_bands(
            kpath=kpath,
            num_bands=8,  # 4个轨道 × 2个自旋
            operator=None
        )
        
        return k_dist, bands
    
    def calculate_berry_curvature(
        self,
        nk=50,
        operator="sz",  # 层算符
        band_indices=None
    ):
        """
        计算Berry曲率（使用PyQula内置功能）
        
        Parameters:
            nk: k空间网格数
            operator: 层投影算符 ("sz" for spin, custom for layer)
            band_indices: 要计算的能带
            
        Returns:
            Dictionary with Berry curvature data
        """
        # 创建k空间网格
        kxs = np.linspace(-np.pi, np.pi, nk)
        kys = np.linspace(-np.pi, np.pi, nk)
        
        # 计算Berry曲率
        # PyQula有专门的berry模块
        bc_calculator = berry.berry_curvature(
            self.hamiltonian,
            nk=nk
        )
        
        omega = bc_calculator.get_berry_curvature()
        
        return {
            'kx': kxs,
            'ky': kys,
            'berry_curvature': omega
        }
    
    def calculate_chern_number(self, band_index=0, nk=50):
        """
        计算Chern数
        
        PyQula自动积分Berry曲率
        """
        chern = topology.chern(
            self.hamiltonian,
            nk=nk,
            noccupied=band_index + 1
        )
        
        return chern
    
    def calculate_layer_hall(self, nk=50, occupied_bands=None):
        """
        计算层霍尔电导率
        
        使用PyQula的层投影算符
        """
        if occupied_bands is None:
            occupied_bands = [0, 1]  # 默认占据前两条带
        
        # 定义层投影算符
        def layer_projector(layer=1):
            """返回层投影算符"""
            def proj_op(r):
                if layer == 1:
                    return 1.0 if r[2] < 1.0 else 0.0
                else:
                    return 1.0 if r[2] > 1.0 else 0.0
            return proj_op
        
        # 计算层1的Berry曲率
        P1 = operators.get_operator(layer_projector(1), self.hamiltonian)
        omega1 = berry.berry_curvature_map(
            self.hamiltonian,
            operator=P1,
            nk=nk
        )
        
        # 计算层2的Berry曲率
        P2 = operators.get_operator(layer_projector(2), self.hamiltonian)
        omega2 = berry.berry_curvature_map(
            self.hamiltonian,
            operator=P2,
            nk=nk
        )
        
        # 层Berry曲率对比
        delta_omega = omega1 - omega2
        
        # 积分得到层霍尔电导率
        sigma_layer = -np.sum(delta_omega) * (2*np.pi/nk)**2 / (2*np.pi)
        
        # 转换为e²/h单位
        e2_h = 1.0
        sigma_layer *= e2_h
        
        return {
            'sigma_layer': sigma_layer,
            'delta_omega': delta_omega,
            'omega1': omega1,
            'omega2': omega2
        }
    
    def visualize_bands(self, show=True):
        """
        可视化能带结构
        
        PyQula内置绘图功能
        """
        import matplotlib.pyplot as plt
        
        k_dist, bands = self.calculate_bands()
        
        plt.figure(figsize=(8, 6))
        for band in bands.T:
            plt.plot(k_dist, band, 'b-', alpha=0.6)
        
        plt.xlabel('k', fontsize=14)
        plt.ylabel('Energy (eV)', fontsize=14)
        plt.title('MnBi₂Te₄ Band Structure', fontsize=16)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def visualize_berry_curvature(self, show=True):
        """
        可视化Berry曲率分布
        """
        import matplotlib.pyplot as plt
        
        result = self.calculate_berry_curvature(nk=50)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(
            result['berry_curvature'],
            origin='lower',
            extent=[-np.pi, np.pi, -np.pi, np.pi],
            cmap='RdBu',
            aspect='auto'
        )
        plt.colorbar(label='Berry Curvature (Ų)')
        plt.xlabel('$k_x$ (2π/a)', fontsize=14)
        plt.ylabel('$k_y$ (2π/a)', fontsize=14)
        plt.title('Berry Curvature Distribution', fontsize=16)
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return plt.gcf()


def example_pyqula():
    """
    PyQula 使用示例
    """
    if not PYQULA_AVAILABLE:
        print("Please install PyQula first:")
        print("  pip install git+https://github.com/joselado/pyqula")
        return
    
    print("="*60)
    print("MnBi2Te4 Model with PyQula")
    print("="*60)
    
    # 创建模型
    print("\nInitializing model...")
    model = MnBi2Te4_PyQula(
        a=4.38,
        t=1.0,
        lambda_SO=0.3,
        M=0.5,
        t_perp=0.2
    )
    
    print(f"  Geometry: {model.geometry.dimensionality}D")
    print(f"  Atoms: {len(model.geometry.r)}")
    print(f"  Layers: 2")
    
    # 计算能带
    print("\nCalculating band structure...")
    k_dist, bands = model.calculate_bands(nk=100)
    print(f"  Number of bands: {bands.shape[1]}")
    print(f"  Gap: {np.min(bands[:, 2]) - np.max(bands[:, 1]):.4f} eV")
    
    # 计算Chern数
    print("\nCalculating Chern number...")
    try:
        chern = model.calculate_chern_number(band_index=1, nk=30)
        print(f"  Chern number (band 0-1): {chern:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 计算层霍尔电导率
    print("\nCalculating layer Hall conductivity...")
    try:
        result = model.calculate_layer_hall(nk=30, occupied_bands=[0, 1])
        print(f"  σ_layer = {result['sigma_layer']:.6f} e²/h")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "="*60)
    print("PyQula calculation completed!")
    print("="*60)


if __name__ == "__main__":
    example_pyqula()
