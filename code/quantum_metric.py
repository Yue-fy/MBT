"""
Quantum Metric Tensor Calculation

This module calculates the quantum metric tensor, which is the real part
of the quantum geometric tensor.

Key formula:
    g_n^{μν}(k) = Re Σ_{m≠n} v^μ_nm (v^ν_nm)* / (E_n - E_m)^2

The quantum metric:
- Measures "distance" between Bloch states at nearby k-points
- Related to band flatness and localization
- Contributes to nonlinear optical responses
- Enters superfluid weight in flat band systems

Author: Yue
Date: November 2025
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict
import warnings


class QuantumMetricCalculator:
    """
    Calculate quantum metric tensor for tight-binding models.
    """
    
    def __init__(
        self,
        hamiltonian_func: Callable,
        eta: float = 1e-6
    ):
        """
        Initialize quantum metric calculator.
        
        Parameters:
            hamiltonian_func: Function H(kx, ky) returning Hamiltonian matrix
            eta: Regularization parameter for energy denominators (eV)
        """
        self.hamiltonian_func = hamiltonian_func
        self.eta = eta
    
    def velocity_matrix(
        self,
        kx: float,
        ky: float,
        direction: str,
        dk: float = 1e-4
    ) -> np.ndarray:
        """
        Calculate velocity operator matrix using finite differences.
        
        v_μ = (1/ℏ) ∂H/∂k_μ (with ℏ=1)
        
        Parameters:
            kx, ky: Momentum point
            direction: 'x' or 'y'
            dk: Finite difference step
            
        Returns:
            Velocity operator matrix
        """
        if direction == 'x':
            H_plus = self.hamiltonian_func(kx + dk, ky)
            H_minus = self.hamiltonian_func(kx - dk, ky)
        elif direction == 'y':
            H_plus = self.hamiltonian_func(kx, ky + dk)
            H_minus = self.hamiltonian_func(kx, ky - dk)
        else:
            raise ValueError("direction must be 'x' or 'y'")
        
        v_matrix = (H_plus - H_minus) / (2 * dk)
        
        return v_matrix
    
    def quantum_metric(
        self,
        kx: float,
        ky: float,
        band_indices: Optional[list] = None,
        dk: float = 1e-4
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate quantum metric tensor components.
        
        Returns g^{xx}, g^{yy}, g^{xy} for each band.
        
        Formula:
            g_n^{μν} = Re Σ_{m≠n} v^μ_nm (v^ν_nm)* / (E_n - E_m)^2
        
        Parameters:
            kx, ky: Momentum point
            band_indices: Bands to calculate (default: all)
            dk: Finite difference step
            
        Returns:
            Dictionary: {band_index: {'gxx': ..., 'gyy': ..., 'gxy': ...}}
        """
        # Diagonalize Hamiltonian
        H = self.hamiltonian_func(kx, ky)
        energies, eigvecs = np.linalg.eigh(H)
        num_bands = len(energies)
        
        if band_indices is None:
            band_indices = range(num_bands)
        
        # Calculate velocity matrices
        v_x = self.velocity_matrix(kx, ky, 'x', dk)
        v_y = self.velocity_matrix(kx, ky, 'y', dk)
        
        # Transform to band basis
        v_x_band = eigvecs.T.conj() @ v_x @ eigvecs
        v_y_band = eigvecs.T.conj() @ v_y @ eigvecs
        
        # Calculate quantum metric for each band
        quantum_metrics = {}
        
        for n in band_indices:
            g_xx = 0.0
            g_yy = 0.0
            g_xy = 0.0
            
            for m in range(num_bands):
                if m == n:
                    continue
                
                # Energy denominator with regularization
                dE = energies[n] - energies[m]
                denom = dE**2 + self.eta**2
                
                # Velocity matrix elements
                v_x_nm = v_x_band[n, m]
                v_y_nm = v_y_band[n, m]
                
                # Quantum metric components
                # g^{xx} = Re[v^x_nm * (v^x_nm)*]
                g_xx += np.real(v_x_nm * np.conj(v_x_nm)) / denom
                
                # g^{yy} = Re[v^y_nm * (v^y_nm)*]
                g_yy += np.real(v_y_nm * np.conj(v_y_nm)) / denom
                
                # g^{xy} = Re[v^x_nm * (v^y_nm)*]
                g_xy += np.real(v_x_nm * np.conj(v_y_nm)) / denom
            
            quantum_metrics[n] = {
                'gxx': g_xx,
                'gyy': g_yy,
                'gxy': g_xy,
                'trace': g_xx + g_yy  # Tr(g) is gauge-invariant
            }
        
        return quantum_metrics
    
    def quantum_metric_layer(
        self,
        kx: float,
        ky: float,
        layer_projector: np.ndarray,
        band_indices: Optional[list] = None,
        dk: float = 1e-4
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate layer-resolved quantum metric.
        
        g_n^{(l),μν} = Re Σ_{m≠n} <n|P_l v^μ|m><m|v^ν|n>* / (E_n - E_m)^2
        
        Parameters:
            kx, ky: Momentum point
            layer_projector: Projection operator for the layer
            band_indices: Bands to calculate
            dk: Finite difference step
            
        Returns:
            Dictionary with layer-resolved quantum metric components
        """
        # Diagonalize Hamiltonian
        H = self.hamiltonian_func(kx, ky)
        energies, eigvecs = np.linalg.eigh(H)
        num_bands = len(energies)
        
        if band_indices is None:
            band_indices = range(num_bands)
        
        # Calculate velocity matrices
        v_x = self.velocity_matrix(kx, ky, 'x', dk)
        v_y = self.velocity_matrix(kx, ky, 'y', dk)
        
        # Layer-weighted velocities
        v_x_layer = layer_projector @ v_x
        v_y_layer = layer_projector @ v_y
        
        # Transform to band basis
        v_x_layer_band = eigvecs.T.conj() @ v_x_layer @ eigvecs
        v_y_layer_band = eigvecs.T.conj() @ v_y_layer @ eigvecs
        
        # Calculate layer quantum metric
        quantum_metrics_layer = {}
        
        for n in band_indices:
            g_xx_layer = 0.0
            g_yy_layer = 0.0
            g_xy_layer = 0.0
            
            for m in range(num_bands):
                if m == n:
                    continue
                
                dE = energies[n] - energies[m]
                denom = dE**2 + self.eta**2
                
                v_x_layer_nm = v_x_layer_band[n, m]
                v_y_layer_nm = v_y_layer_band[n, m]
                
                g_xx_layer += np.real(v_x_layer_nm * np.conj(v_x_layer_nm)) / denom
                g_yy_layer += np.real(v_y_layer_nm * np.conj(v_y_layer_nm)) / denom
                g_xy_layer += np.real(v_x_layer_nm * np.conj(v_y_layer_nm)) / denom
            
            quantum_metrics_layer[n] = {
                'gxx': g_xx_layer,
                'gyy': g_yy_layer,
                'gxy': g_xy_layer,
                'trace': g_xx_layer + g_yy_layer
            }
        
        return quantum_metrics_layer
    
    def quantum_metric_dipole(
        self,
        kx: float,
        ky: float,
        band_index: int,
        dk: float = 1e-4,
        dk_deriv: float = 1e-4
    ) -> Dict[str, float]:
        """
        Calculate quantum metric dipole for nonlinear transport.
        
        D^{μν} = ∂E/∂k_μ * ∂g^{νν}/∂k_ν
        
        Parameters:
            kx, ky: Momentum point
            band_index: Band to calculate
            dk: Step for velocity calculation
            dk_deriv: Step for derivative of quantum metric
            
        Returns:
            Dictionary with dipole components
        """
        # Calculate quantum metric at nearby points
        g_center = self.quantum_metric(kx, ky, [band_index], dk)[band_index]
        
        g_x_plus = self.quantum_metric(kx + dk_deriv, ky, [band_index], dk)[band_index]
        g_x_minus = self.quantum_metric(kx - dk_deriv, ky, [band_index], dk)[band_index]
        
        g_y_plus = self.quantum_metric(kx, ky + dk_deriv, [band_index], dk)[band_index]
        g_y_minus = self.quantum_metric(kx, ky - dk_deriv, [band_index], dk)[band_index]
        
        # Derivatives of quantum metric
        dg_xx_dx = (g_x_plus['gxx'] - g_x_minus['gxx']) / (2 * dk_deriv)
        dg_yy_dy = (g_y_plus['gyy'] - g_y_minus['gyy']) / (2 * dk_deriv)
        dg_xy_dx = (g_x_plus['gxy'] - g_x_minus['gxy']) / (2 * dk_deriv)
        dg_xy_dy = (g_y_plus['gxy'] - g_y_minus['gxy']) / (2 * dk_deriv)
        
        # Energy derivatives (band velocity)
        H = self.hamiltonian_func(kx, ky)
        energies, _ = np.linalg.eigh(H)
        
        H_x_plus = self.hamiltonian_func(kx + dk_deriv, ky)
        H_x_minus = self.hamiltonian_func(kx - dk_deriv, ky)
        E_x_plus, _ = np.linalg.eigh(H_x_plus)
        E_x_minus, _ = np.linalg.eigh(H_x_minus)
        
        H_y_plus = self.hamiltonian_func(kx, ky + dk_deriv)
        H_y_minus = self.hamiltonian_func(kx, ky - dk_deriv)
        E_y_plus, _ = np.linalg.eigh(H_y_plus)
        E_y_minus, _ = np.linalg.eigh(H_y_minus)
        
        dE_dx = (E_x_plus[band_index] - E_x_minus[band_index]) / (2 * dk_deriv)
        dE_dy = (E_y_plus[band_index] - E_y_minus[band_index]) / (2 * dk_deriv)
        
        # Quantum metric dipoles
        # D^{xx} = ∂E/∂kx * ∂g^{xx}/∂kx
        D_xx = dE_dx * dg_xx_dx
        
        # D^{yy} = ∂E/∂ky * ∂g^{yy}/∂ky
        D_yy = dE_dy * dg_yy_dy
        
        # D^{xy} = ∂E/∂kx * ∂g^{yy}/∂ky (for nonlinear Hall)
        D_xy = dE_dx * dg_yy_dy
        
        # D^{yx} = ∂E/∂ky * ∂g^{xx}/∂kx
        D_yx = dE_dy * dg_xx_dx
        
        dipoles = {
            'D_xx': D_xx,
            'D_yy': D_yy,
            'D_xy': D_xy,
            'D_yx': D_yx,
            'dE_dx': dE_dx,
            'dE_dy': dE_dy
        }
        
        return dipoles


def calculate_quantum_metric_map(
    model,
    k_range: Tuple[float, float],
    nk: int = 100,
    band_indices: Optional[list] = None,
    component: str = 'trace',
    dk: float = 1e-4,
    eta: float = 1e-6
) -> dict:
    """
    Calculate quantum metric over 2D k-space grid.
    
    Parameters:
        model: Tight-binding model with hamiltonian(kx, ky) method
        k_range: (k_min, k_max) for both kx and ky
        nk: Number of k-points along each direction
        band_indices: Bands to calculate
        component: 'gxx', 'gyy', 'gxy', or 'trace'
        dk: Finite difference step
        eta: Regularization parameter
        
    Returns:
        Dictionary with quantum metric maps
    """
    k_min, k_max = k_range
    kx_array = np.linspace(k_min, k_max, nk)
    ky_array = np.linspace(k_min, k_max, nk)
    
    KX, KY = np.meshgrid(kx_array, ky_array)
    
    # Initialize calculator
    calc = QuantumMetricCalculator(model.hamiltonian, eta=eta)
    
    # Determine bands
    H_test = model.hamiltonian(0.0, 0.0)
    num_bands = H_test.shape[0]
    
    if band_indices is None:
        band_indices = list(range(num_bands))
    
    # Calculate quantum metric
    quantum_metric_maps = {}
    
    for n in band_indices:
        qm_map = np.zeros((nk, nk))
        
        for i in range(nk):
            for j in range(nk):
                kx, ky = KX[i, j], KY[i, j]
                
                qm = calc.quantum_metric(kx, ky, [n], dk)
                qm_map[i, j] = qm[n][component]
        
        quantum_metric_maps[n] = qm_map
    
    result = {
        'kx': KX,
        'ky': KY,
        'quantum_metric': quantum_metric_maps,
        'component': component,
        'band_indices': band_indices
    }
    
    return result


def calculate_layer_quantum_metric_map(
    model,
    layer: int,
    k_range: Tuple[float, float],
    nk: int = 100,
    band_indices: Optional[list] = None,
    component: str = 'trace',
    dk: float = 1e-4,
    eta: float = 1e-6
) -> dict:
    """
    Calculate layer-resolved quantum metric over 2D k-space.
    
    Parameters:
        model: Model with layer_projection_operator method
        layer: Layer index (1 or 2)
        k_range: (k_min, k_max)
        nk: Number of k-points
        band_indices: Bands to calculate
        component: Metric component to extract
        dk: Finite difference step
        eta: Regularization
        
    Returns:
        Dictionary with layer-resolved quantum metric maps
    """
    k_min, k_max = k_range
    kx_array = np.linspace(k_min, k_max, nk)
    ky_array = np.linspace(k_min, k_max, nk)
    
    KX, KY = np.meshgrid(kx_array, ky_array)
    
    # Get layer projection operator
    P_layer = model.layer_projection_operator(layer)
    
    # Initialize calculator
    calc = QuantumMetricCalculator(model.hamiltonian, eta=eta)
    
    # Determine bands
    H_test = model.hamiltonian(0.0, 0.0)
    num_bands = H_test.shape[0]
    
    if band_indices is None:
        band_indices = list(range(num_bands))
    
    # Calculate layer quantum metric
    quantum_metric_layer_maps = {}
    
    for n in band_indices:
        qm_layer_map = np.zeros((nk, nk))
        
        for i in range(nk):
            for j in range(nk):
                kx, ky = KX[i, j], KY[i, j]
                
                qm_layer = calc.quantum_metric_layer(kx, ky, P_layer, [n], dk)
                qm_layer_map[i, j] = qm_layer[n][component]
        
        quantum_metric_layer_maps[n] = qm_layer_map
    
    result = {
        'kx': KX,
        'ky': KY,
        'quantum_metric_layer': quantum_metric_layer_maps,
        'component': component,
        'layer': layer,
        'band_indices': band_indices
    }
    
    return result


if __name__ == "__main__":
    """
    Example usage and tests.
    """
    print("=" * 60)
    print("Quantum Metric Calculator - Test")
    print("=" * 60)
    
    # Simple 2-band model for testing
    def simple_hamiltonian(kx, ky):
        """Simple 2-band model with SOC."""
        t = 1.0
        Delta = 0.5
        lambda_so = 0.2
        
        h_k = -2 * t * (np.cos(kx) + np.cos(ky))
        h_x = lambda_so * np.sin(kx)
        h_y = lambda_so * np.sin(ky)
        
        H = np.array([
            [Delta + h_k, h_x - 1j * h_y],
            [h_x + 1j * h_y, -Delta + h_k]
        ], dtype=complex)
        
        return H
    
    # Test quantum metric calculation
    calc = QuantumMetricCalculator(simple_hamiltonian, eta=1e-6)
    
    kx, ky = 0.5, 0.3
    qm = calc.quantum_metric(kx, ky)
    
    print(f"\nQuantum metric at k=({kx}, {ky}):")
    for band, metrics in qm.items():
        print(f"\n  Band {band}:")
        print(f"    g^{{xx}} = {metrics['gxx']:.6f}")
        print(f"    g^{{yy}} = {metrics['gyy']:.6f}")
        print(f"    g^{{xy}} = {metrics['gxy']:.6f}")
        print(f"    Tr(g) = {metrics['trace']:.6f}")
    
    print(f"\n{'='*60}")
    print("Quantum metric calculator initialized successfully!")
    print('='*60)
