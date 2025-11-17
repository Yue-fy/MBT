"""
无序扫描主程序 - Workflow 第三阶段
Disorder Scanning: The Core PRL Experiment

核心科学问题：
为什么实验只看到QMD而没有BCD？
→ 答案：磁性无序是量子几何的"过滤器"

物理机制：
1. Hidden BCD依赖于完美的层间磁矩抵消（被Ez微扰打破）
2. 一旦引入磁性无序（反位缺陷/热涨落），Hidden BCD瞬间消失
3. QMD具有拓扑鲁棒性，对无序不敏感

实验设计：
- 标量无序 vs 磁性无序对比
- 无序强度扫描 W = [0.01, 0.05, 0.1, 0.2, 0.5] eV
- 每个W值：100个样本平均

Author: Yue
Date: 2025-11-18
"""

import numpy as np
from typing import List, Dict, Tuple
import time
import pickle
from tb_model_exact import MnBi2Te4_Exact
from negf_calculator import NEGF_Calculator


class DisorderScanner:
    """
    无序扫描器 - 实现workflow第三阶段
    """
    
    def __init__(
        self,
        model: MnBi2Te4_Exact,
        negf_calc: NEGF_Calculator
    ):
        self.model = model
        self.negf = negf_calc
        
        print("Disorder Scanner Initialized")
        print(f"  Model: {model.n_layers} layers")
        print(f"  NEGF: {negf_calc.L}×{negf_calc.W} device")
    
    def generate_scalar_disorder(
        self,
        W: float,
        n_layers: int,
        n_orbitals: int = 4
    ) -> np.ndarray:
        """
        生成标量无序势
        
        V_disorder[layer, orbital] ∈ [-W, W]
        
        模拟杂质doping效应
        """
        disorder = np.random.uniform(-W, W, size=(n_layers, n_orbitals))
        return disorder
    
    def generate_magnetic_disorder(
        self,
        W_mag: float,
        n_layers: int,
        m_AFM: float = 0.03
    ) -> np.ndarray:
        """
        生成磁性无序
        
        δm[layer, 3] = 随机方向的磁矩扰动
        
        模拟反位缺陷或热涨落
        """
        # 随机方向的单位向量
        theta = np.random.uniform(0, np.pi, n_layers)
        phi = np.random.uniform(0, 2*np.pi, n_layers)
        
        # 球坐标转笛卡尔坐标
        delta_m = np.zeros((n_layers, 3))
        delta_m[:, 0] = np.sin(theta) * np.cos(phi)
        delta_m[:, 1] = np.sin(theta) * np.sin(phi)
        delta_m[:, 2] = np.cos(theta)
        
        # 缩放到强度W_mag
        delta_m *= W_mag
        
        return delta_m
    
    def single_disorder_realization(
        self,
        W: float,
        disorder_type: str,
        Ez: float = 0.05,
        bias_V: float = 0.01,
        nk: int = 5,
        n_energy: int = 10
    ) -> Dict[str, float]:
        """
        单次无序样本的计算
        
        Parameters:
            W: 无序强度
            disorder_type: 'scalar' or 'magnetic'
            Ez: 栅压电场
            bias_V: 偏压
            nk, n_energy: k空间和能量采样点数
        
        Returns:
            {'BCD': ..., 'QMD': ...}
        """
        # 生成无序
        if disorder_type == 'scalar':
            disorder_scalar = self.generate_scalar_disorder(W, self.model.n_layers)
            disorder_magnetic = None
        elif disorder_type == 'magnetic':
            disorder_scalar = None
            disorder_magnetic = self.generate_magnetic_disorder(W, self.model.n_layers)
        else:
            raise ValueError(f"Unknown disorder type: {disorder_type}")
        
        # 计算BCD和QMD
        results = self.negf.separate_BCD_QMD(
            bias_V=bias_V,
            Ez=Ez,
            nk=nk,
            n_energy=n_energy,
            disorder_scalar=disorder_scalar,
            disorder_magnetic=disorder_magnetic
        )
        
        return results
    
    def disorder_average(
        self,
        W: float,
        disorder_type: str,
        n_samples: int = 10,
        Ez: float = 0.05,
        bias_V: float = 0.01,
        nk: int = 5,
        n_energy: int = 10
    ) -> Dict[str, float]:
        """
        对无序样本求平均
        
        Parameters:
            W: 无序强度
            disorder_type: 'scalar' or 'magnetic'
            n_samples: 样本数量（默认10，实际应该100+）
            
        Returns:
            平均后的 {'BCD': ..., 'QMD': ..., 'BCD_std': ..., 'QMD_std': ...}
        """
        print(f"\n{'='*60}")
        print(f"Disorder Averaging: {disorder_type.upper()}, W={W:.4f} eV")
        print(f"  Samples: {n_samples}")
        print(f"{'='*60}")
        
        BCD_list = []
        QMD_list = []
        
        for i in range(n_samples):
            print(f"\n  Sample {i+1}/{n_samples}")
            try:
                results = self.single_disorder_realization(
                    W=W,
                    disorder_type=disorder_type,
                    Ez=Ez,
                    bias_V=bias_V,
                    nk=nk,
                    n_energy=n_energy
                )
                BCD_list.append(results['BCD'])
                QMD_list.append(results['QMD'])
            except Exception as e:
                print(f"    Error in sample {i+1}: {e}")
                continue
        
        # 统计
        BCD_array = np.array(BCD_list)
        QMD_array = np.array(QMD_list)
        
        result = {
            'BCD_mean': np.mean(BCD_array),
            'BCD_std': np.std(BCD_array),
            'QMD_mean': np.mean(QMD_array),
            'QMD_std': np.std(QMD_array),
            'n_valid': len(BCD_list)
        }
        
        print(f"\n{'='*60}")
        print(f"AVERAGED RESULTS:")
        print(f"  BCD: {result['BCD_mean']:.6e} ± {result['BCD_std']:.6e}")
        print(f"  QMD: {result['QMD_mean']:.6e} ± {result['QMD_std']:.6e}")
        print(f"  Valid samples: {result['n_valid']}/{n_samples}")
        print(f"{'='*60}")
        
        return result
    
    def scan_disorder_strength(
        self,
        W_values: List[float],
        disorder_type: str,
        n_samples: int = 10,
        Ez: float = 0.05,
        bias_V: float = 0.01,
        nk: int = 5,
        n_energy: int = 10,
        save_file: str = None
    ) -> Dict:
        """
        扫描无序强度 - 核心PRL实验
        
        Parameters:
            W_values: 无序强度列表，如 [0.01, 0.05, 0.1, 0.2, 0.5]
            disorder_type: 'scalar' or 'magnetic'
            n_samples: 每个W的样本数
            save_file: 保存结果的文件名
        
        Returns:
            完整扫描结果
        """
        print("\n" + "="*60)
        print("DISORDER STRENGTH SCAN")
        print("="*60)
        print(f"Type: {disorder_type.upper()}")
        print(f"W values: {W_values}")
        print(f"Samples per W: {n_samples}")
        print(f"Ez: {Ez} eV, V: {bias_V} eV")
        print("="*60)
        
        results = {
            'W_values': W_values,
            'disorder_type': disorder_type,
            'BCD_mean': [],
            'BCD_std': [],
            'QMD_mean': [],
            'QMD_std': [],
            'parameters': {
                'Ez': Ez,
                'bias_V': bias_V,
                'n_samples': n_samples,
                'nk': nk,
                'n_energy': n_energy
            }
        }
        
        start_time = time.time()
        
        for W in W_values:
            avg_result = self.disorder_average(
                W=W,
                disorder_type=disorder_type,
                n_samples=n_samples,
                Ez=Ez,
                bias_V=bias_V,
                nk=nk,
                n_energy=n_energy
            )
            
            results['BCD_mean'].append(avg_result['BCD_mean'])
            results['BCD_std'].append(avg_result['BCD_std'])
            results['QMD_mean'].append(avg_result['QMD_mean'])
            results['QMD_std'].append(avg_result['QMD_std'])
        
        elapsed = time.time() - start_time
        results['elapsed_time'] = elapsed
        
        # 保存结果
        if save_file:
            with open(save_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"\n✓ Results saved to: {save_file}")
        
        print(f"\n{'='*60}")
        print(f"SCAN COMPLETED in {elapsed/60:.1f} minutes")
        print(f"{'='*60}")
        
        return results
    
    def compare_scalar_vs_magnetic(
        self,
        W_values: List[float],
        n_samples: int = 10,
        save_prefix: str = "disorder_scan"
    ):
        """
        对比标量无序 vs 磁性无序 - 论文核心图
        
        预期结果：
        - 标量无序：BCD和QMD都下降，但QMD稍强
        - 磁性无序：BCD极速崩溃（Hidden BCD被破坏），QMD鲁棒
        """
        print("\n" + "#"*60)
        print("# COMPARING SCALAR vs MAGNETIC DISORDER")
        print("# This is the KEY PRL experiment!")
        print("#"*60)
        
        # 扫描标量无序
        print("\n\n### PART 1: SCALAR DISORDER ###\n")
        results_scalar = self.scan_disorder_strength(
            W_values=W_values,
            disorder_type='scalar',
            n_samples=n_samples,
            save_file=f"{save_prefix}_scalar.pkl"
        )
        
        # 扫描磁性无序
        print("\n\n### PART 2: MAGNETIC DISORDER ###\n")
        results_magnetic = self.scan_disorder_strength(
            W_values=W_values,
            disorder_type='magnetic',
            n_samples=n_samples,
            save_file=f"{save_prefix}_magnetic.pkl"
        )
        
        print("\n" + "#"*60)
        print("# COMPARISON COMPLETE")
        print("#"*60)
        print("\nKey Findings:")
        print("  - Check the .pkl files for detailed results")
        print("  - Use plot_disorder_results.py to visualize")
        print("  - Expected: Magnetic disorder kills BCD, preserves QMD")
        print("#"*60)
        
        return results_scalar, results_magnetic


def quick_test():
    """快速测试（少量样本和k点）"""
    print("="*60)
    print("Quick Test of Disorder Scanning")
    print("="*60)
    
    # 创建模型
    print("\n1. Creating model...")
    model = MnBi2Te4_Exact(n_layers=6)
    
    # 创建NEGF
    print("\n2. Creating NEGF calculator...")
    negf = NEGF_Calculator(
        hamiltonian_func=model.hamiltonian_multilayer,
        device_length=30,
        device_width=15,
        temperature=1.0
    )
    
    # 创建扫描器
    print("\n3. Creating disorder scanner...")
    scanner = DisorderScanner(model, negf)
    
    # 快速测试：只扫描2个W值，每个2个样本
    print("\n4. Running quick scan...")
    W_values = [0.01, 0.05]  # 只测试2个点
    
    results = scanner.scan_disorder_strength(
        W_values=W_values,
        disorder_type='magnetic',
        n_samples=2,  # 只2个样本
        nk=3,  # 很少的k点
        n_energy=5,  # 很少的能量点
        save_file='test_disorder_scan.pkl'
    )
    
    print("\n" + "="*60)
    print("✓ Quick test completed!")
    print("="*60)


if __name__ == "__main__":
    quick_test()
