"""
可视化和绘图工具
用于生成论文质量的图表

Author: Yue
Date: 2025-11-18
"""

import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    MATPLOTLIB_AVAILABLE = True
    
    # 设置论文质量的绘图参数
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 11
    rcParams['figure.titlesize'] = 16
    rcParams['figure.dpi'] = 150
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available")


def plot_band_structure(k_path, bands, k_labels=None, E_fermi=0.0, save_path=None):
    """
    绘制能带结构
    
    Parameters:
        k_path: k点路径坐标 [nk]
        bands: 能带 [nk, n_bands]
        k_labels: 高对称点标签
        E_fermi: 费米能级
        save_path: 保存路径
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制所有能带
    for i in range(bands.shape[1]):
        ax.plot(k_path, bands[:, i], 'b-', linewidth=1.5, alpha=0.7)
    
    # 费米能级
    ax.axhline(E_fermi, color='r', linestyle='--', linewidth=1, alpha=0.7, label='$E_F$')
    
    # 高对称点标记
    if k_labels is not None:
        nk = len(k_path)
        k_points = [0, nk//3, 2*nk//3, nk-1]
        for kp in k_points:
            ax.axvline(k_path[kp], color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xticks(k_path[k_points])
        ax.set_xticklabels(k_labels)
    
    ax.set_ylabel('Energy (eV)', fontsize=14)
    ax.set_xlabel('Wave vector', fontsize=14)
    ax.set_title('Band Structure of 6SL MnBi$_2$Te$_4$', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Band structure saved to: {save_path}")
    
    plt.close()


def plot_dos(energies, dos, save_path=None):
    """
    绘制态密度
    
    Parameters:
        energies: 能量点 [n_energy]
        dos: 态密度 [n_energy]
        save_path: 保存路径
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(6, 8))
    
    ax.plot(dos, energies, 'b-', linewidth=2)
    ax.axhline(0, color='r', linestyle='--', linewidth=1, alpha=0.7, label='$E_F$')
    ax.fill_betweenx(energies, 0, dos, alpha=0.3)
    
    ax.set_xlabel('DOS (states/eV)', fontsize=14)
    ax.set_ylabel('Energy (eV)', fontsize=14)
    ax.set_title('Density of States', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ DOS saved to: {save_path}")
    
    plt.close()


def plot_band_and_dos(k_path, bands, energies, dos, k_labels=None, save_path=None):
    """
    组合绘制能带和态密度
    
    Parameters:
        k_path: k点路径
        bands: 能带
        energies: DOS能量
        dos: 态密度
        k_labels: 高对称点标签
        save_path: 保存路径
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return
    
    fig = plt.figure(figsize=(12, 6))
    
    # 左侧：能带
    ax1 = plt.subplot(1, 2, 1)
    for i in range(bands.shape[1]):
        ax1.plot(k_path, bands[:, i], 'b-', linewidth=1.5, alpha=0.7)
    
    ax1.axhline(0, color='r', linestyle='--', linewidth=1, alpha=0.7)
    
    if k_labels is not None:
        nk = len(k_path)
        k_points = [0, nk//3, 2*nk//3, nk-1]
        for kp in k_points:
            ax1.axvline(k_path[kp], color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax1.set_xticks(k_path[k_points])
        ax1.set_xticklabels(k_labels)
    
    ax1.set_ylabel('Energy (eV)', fontsize=14)
    ax1.set_xlabel('Wave vector', fontsize=14)
    ax1.set_title('Band Structure', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.5, 0.5])
    
    # 右侧：态密度
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(dos, energies, 'b-', linewidth=2)
    ax2.axhline(0, color='r', linestyle='--', linewidth=1, alpha=0.7)
    ax2.fill_betweenx(energies, 0, dos, alpha=0.3)
    
    ax2.set_xlabel('DOS (states/eV)', fontsize=14)
    ax2.set_ylabel('Energy (eV)', fontsize=14)
    ax2.set_title('Density of States', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.5, 0.5])
    
    plt.suptitle('6SL MnBi$_2$Te$_4$ Electronic Structure', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Combined plot saved to: {save_path}")
    
    plt.close()


def plot_disorder_scan(W_values, BCD_mean, QMD_mean, BCD_std=None, QMD_std=None, 
                       disorder_type='scalar', save_path=None):
    """
    绘制无序扫描结果
    
    Parameters:
        W_values: 无序强度
        BCD_mean: BCD平均值
        QMD_mean: QMD平均值
        BCD_std: BCD标准差
        QMD_std: QMD标准差
        disorder_type: 'scalar' or 'magnetic'
        save_path: 保存路径
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 转换为数组
    W_values = np.array(W_values)
    BCD_mean = np.abs(np.array(BCD_mean))
    QMD_mean = np.abs(np.array(QMD_mean))
    
    # 绘制BCD
    ax.loglog(W_values, BCD_mean, 'o-', color='red', linewidth=2.5, 
              markersize=10, label='BCD (Hidden Berry Curvature)', alpha=0.8)
    if BCD_std is not None:
        ax.fill_between(W_values, BCD_mean - np.array(BCD_std), 
                        BCD_mean + np.array(BCD_std), color='red', alpha=0.2)
    
    # 绘制QMD
    ax.loglog(W_values, QMD_mean, 's-', color='blue', linewidth=2.5,
              markersize=10, label='QMD (Quantum Metric)', alpha=0.8)
    if QMD_std is not None:
        ax.fill_between(W_values, QMD_mean - np.array(QMD_std),
                        QMD_mean + np.array(QMD_std), color='blue', alpha=0.2)
    
    ax.set_xlabel('Disorder Strength $W$ (eV)', fontsize=14)
    ax.set_ylabel(r'$|\sigma_{xy}^{(2)}|$ (arb. units)', fontsize=14)
    
    title = f'{"Magnetic" if disorder_type == "magnetic" else "Scalar"} Disorder Effect'
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Disorder scan plot saved to: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    if MATPLOTLIB_AVAILABLE:
        print("✓ Plotting utilities loaded successfully")
        print(f"  DPI: {rcParams['figure.dpi']}")
        print(f"  Save DPI: {rcParams['savefig.dpi']}")
    else:
        print("✗ matplotlib not available")
