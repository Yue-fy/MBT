"""
MnBi2Te4 模型使用 Kwant 实现
Implementation using Kwant library for quantum transport

Kwant: https://kwant-project.org/
专门用于量子输运、紧束缚模型和拓扑系统

Features:
- 精确的紧束缚模型构建
- 高效的波函数计算
- 输运性质（电导、散射矩阵）
- Lead连接和边界条件

Author: Yue
Date: 2025-11-17
"""

import numpy as np

try:
    import kwant
    from kwant.continuum import discretize
    KWANT_AVAILABLE = True
    print("✓ Kwant imported successfully!")
except ImportError:
    KWANT_AVAILABLE = False
    print("✗ Kwant not available. Install: pip install kwant")


class MnBi2Te4_Kwant:
    """
    MnBi2Te4 双层反铁磁模型 - Kwant 实现
    
    优势：
    - 精确的实空间紧束缚模型
    - 可以添加 leads 计算输运
    - 支持无序和散射
    - 波函数可视化
    """
    
    def __init__(
        self,
        a: float = 4.38,          # 晶格常数 (Å)
        t: float = 1.0,           # 跳跃 (eV)
        lambda_SO: float = 0.3,   # SOC (eV)
        M: float = 0.5,           # 交换场 (eV)
        t_perp: float = 0.2,      # 层间耦合 (eV)
        mu: float = 0.0,          # 化学势 (eV)
        L: int = 20               # 系统线度（用于有限系统）
    ):
        if not KWANT_AVAILABLE:
            raise ImportError("Kwant is required. Install with: pip install kwant")
        
        self.a = a
        self.t = t
        self.lambda_SO = lambda_SO
        self.M = M
        self.t_perp = t_perp
        self.mu = mu
        self.L = L
        
        # Pauli矩阵
        self.sigma_0 = np.eye(2, dtype=complex)
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # 创建系统
        self.syst = None
        self.lead_syst = None
    
    def build_infinite_system(self):
        """
        构建无限周期系统（用于能带计算）
        
        Kwant 使用 Builder 构建格点
        """
        # 定义晶格（双层 + 自旋）
        # 每个格点有4个自由度：2层 × 2自旋
        lat_layer1 = kwant.lattice.general(
            [(self.a, 0), (self.a/2, self.a*np.sqrt(3)/2)],
            norbs=2,  # 2个自旋
            name='layer1'
        )
        lat_layer2 = kwant.lattice.general(
            [(self.a, 0), (self.a/2, self.a*np.sqrt(3)/2)],
            norbs=2,
            name='layer2'
        )
        
        # 创建Builder
        syst = kwant.Builder(kwant.TranslationalSymmetry(*lat_layer1.prim_vecs))
        
        # 层1的onsite能量和交换场
        def onsite_layer1(site):
            return -self.mu * self.sigma_0 + self.M * self.sigma_z
        
        # 层2的onsite能量和交换场（反平行）
        def onsite_layer2(site):
            return -self.mu * self.sigma_0 - self.M * self.sigma_z
        
        # 添加格点
        syst[lat_layer1(0, 0)] = onsite_layer1
        syst[lat_layer2(0, 0)] = onsite_layer2
        
        # 层内最近邻跳跃（包含SOC）
        def hopping_intra(site1, site2, t, lambda_SO):
            """层内跳跃 + 自旋轨道耦合"""
            # 基本跳跃
            hop = -t * self.sigma_0
            
            # Kane-Mele SOC: t_SO * (s × d)_z
            # 其中 d 是键方向
            d = np.array(site2.pos) - np.array(site1.pos)
            # SOC项：i * lambda_SO * (sigma_x * d_y - sigma_y * d_x)
            soc_term = 1j * lambda_SO * (self.sigma_x * d[1] - self.sigma_y * d[0]) / self.a
            
            return hop + soc_term
        
        # 层1内跳跃
        syst[kwant.builder.HoppingKind((1, 0), lat_layer1)] = lambda s1, s2: hopping_intra(s1, s2, self.t, self.lambda_SO)
        syst[kwant.builder.HoppingKind((0, 1), lat_layer1)] = lambda s1, s2: hopping_intra(s1, s2, self.t, self.lambda_SO)
        syst[kwant.builder.HoppingKind((-1, 1), lat_layer1)] = lambda s1, s2: hopping_intra(s1, s2, self.t, self.lambda_SO)
        
        # 层2内跳跃
        syst[kwant.builder.HoppingKind((1, 0), lat_layer2)] = lambda s1, s2: hopping_intra(s1, s2, self.t, self.lambda_SO)
        syst[kwant.builder.HoppingKind((0, 1), lat_layer2)] = lambda s1, s2: hopping_intra(s1, s2, self.t, self.lambda_SO)
        syst[kwant.builder.HoppingKind((-1, 1), lat_layer2)] = lambda s1, s2: hopping_intra(s1, s2, self.t, self.lambda_SO)
        
        # 层间跳跃
        syst[lat_layer1(0, 0), lat_layer2(0, 0)] = -self.t_perp * self.sigma_0
        
        self.syst = syst.finalized()
        
        return self.syst
    
    def build_finite_system(self, shape='square'):
        """
        构建有限尺寸系统（用于局域态、输运）
        
        Parameters:
            shape: 'square' or 'circle'
        """
        lat_layer1 = kwant.lattice.general(
            [(self.a, 0), (self.a/2, self.a*np.sqrt(3)/2)],
            norbs=2,
            name='layer1'
        )
        lat_layer2 = kwant.lattice.general(
            [(self.a, 0), (self.a/2, self.a*np.sqrt(3)/2)],
            norbs=2,
            name='layer2'
        )
        
        syst = kwant.Builder()
        
        # 定义系统形状
        if shape == 'square':
            def shape_func(pos):
                return abs(pos[0]) < self.L*self.a/2 and abs(pos[1]) < self.L*self.a/2
        else:  # circle
            def shape_func(pos):
                return np.sqrt(pos[0]**2 + pos[1]**2) < self.L*self.a/2
        
        # 添加格点
        syst[lat_layer1.shape(shape_func, (0, 0))] = lambda site: -self.mu * self.sigma_0 + self.M * self.sigma_z
        syst[lat_layer2.shape(shape_func, (0, 0))] = lambda site: -self.mu * self.sigma_0 - self.M * self.sigma_z
        
        # 添加跳跃（同上）
        def hopping_intra(site1, site2):
            hop = -self.t * self.sigma_0
            d = np.array(site2.pos) - np.array(site1.pos)
            soc_term = 1j * self.lambda_SO * (self.sigma_x * d[1] - self.sigma_y * d[0]) / self.a
            return hop + soc_term
        
        # 层内
        syst[lat_layer1.neighbors()] = hopping_intra
        syst[lat_layer2.neighbors()] = hopping_intra
        
        # 层间
        # 需要遍历每个格点对
        for site1 in syst.sites():
            if site1.family == lat_layer1:
                try:
                    site2 = lat_layer2(*site1.tag)
                    if site2 in syst.sites():
                        syst[site1, site2] = -self.t_perp * self.sigma_0
                except:
                    pass
        
        self.finite_syst = syst.finalized()
        
        return self.finite_syst
    
    def calculate_bands(self, nk=100):
        """
        计算能带结构
        
        使用Kwant的无限系统功能
        """
        if self.syst is None:
            self.build_infinite_system()
        
        # 定义k路径：Γ → M → K → Γ
        # 对于三角格子
        b1 = 2*np.pi/self.a * np.array([1, 1/np.sqrt(3)])
        b2 = 2*np.pi/self.a * np.array([0, 2/np.sqrt(3)])
        
        Gamma = np.array([0, 0])
        M = 0.5 * b1
        K = (2/3) * b1 + (1/3) * b2
        
        # 计算每个k点的能量
        k_path = []
        for k_start, k_end, n in [(Gamma, M, nk//3), (M, K, nk//3), (K, Gamma, nk//3)]:
            k_seg = np.linspace(k_start, k_end, n, endpoint=False)
            k_path.extend(k_seg)
        
        k_path = np.array(k_path)
        
        # 计算能带
        bands = []
        for k in k_path:
            ham = self.syst.hamiltonian_submatrix(params=dict(k_x=k[0], k_y=k[1]))
            energies = np.linalg.eigvalsh(ham)
            bands.append(energies)
        
        bands = np.array(bands)
        
        return np.arange(len(k_path)), bands
    
    def calculate_berry_curvature_kubo(self, kx, ky, band_index=0):
        """
        在k点计算Berry曲率（使用Kubo公式）
        
        Kwant直接提供哈密顿量，我们用自己的Kubo公式
        """
        # 获取哈密顿量
        def get_ham(kx, ky):
            # 这里需要手动构建动量空间哈密顿量
            # Kwant的无限系统在特定k点的哈密顿量
            # 实际实现较复杂，需要傅里叶变换
            pass
        
        # TODO: 实现Kubo公式计算
        # 可以参考我们之前的berry_curvature.py
        
        return 0.0
    
    def calculate_ldos(self, energy=0.0):
        """
        计算局域态密度（Local Density of States）
        
        需要有限系统
        """
        if not hasattr(self, 'finite_syst'):
            self.build_finite_system()
        
        ldos = kwant.ldos(self.finite_syst, energy=energy)
        
        return ldos
    
    def plot_system(self):
        """
        可视化系统结构
        """
        if not hasattr(self, 'finite_syst'):
            self.build_finite_system()
        
        kwant.plot(self.finite_syst)
    
    def add_leads_for_transport(self):
        """
        添加leads用于输运计算
        
        计算霍尔电导需要4个leads（Hall bar几何）
        """
        # TODO: 实现Hall bar几何
        # 左右leads（电流方向）
        # 上下leads（电压测量）
        pass


def example_kwant():
    """
    Kwant 使用示例
    """
    if not KWANT_AVAILABLE:
        print("Please install Kwant first:")
        print("  pip install kwant")
        return
    
    print("="*60)
    print("MnBi2Te4 Model with Kwant")
    print("="*60)
    
    # 创建模型
    print("\nInitializing model...")
    model = MnBi2Te4_Kwant(
        a=4.38,
        t=1.0,
        lambda_SO=0.3,
        M=0.5,
        t_perp=0.2,
        L=10
    )
    
    # 构建无限系统
    print("\nBuilding infinite system...")
    try:
        syst = model.build_infinite_system()
        print(f"  System built successfully")
        print(f"  Hamiltonian size: {syst.cell_hamiltonian().shape}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 构建有限系统
    print("\nBuilding finite system...")
    try:
        finite_syst = model.build_finite_system(shape='circle')
        print(f"  Number of sites: {len(finite_syst.sites())}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "="*60)
    print("Kwant system built!")
    print("Note: Full band and Berry curvature calculations")
    print("      require additional implementation.")
    print("="*60)


if __name__ == "__main__":
    example_kwant()
