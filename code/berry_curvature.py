"""
Berry Curvature Calculation for Tight-Binding Models

This module implements numerical methods for calculating Berry curvature
using the Kubo formula and related approaches.

Key formulas:
    Ω_n(k) = -2 Im Σ_{m≠n} <n|v_x|m><m|v_y|n> / (E_n - E_m)^2

Author: Yue
Date: November 2025
"""

import numpy as np
from typing import Tuple, Optional, Callable
import warnings


class BerryCurvatureCalculator:
    """
    Calculate Berry curvature for tight-binding models.
    
    Supports:
    - Kubo formula method
    - Finite difference of Berry connection
    - Wilson loop (plaquette) method
    """
    
    def __init__(
        self,
        hamiltonian_func: Callable,
        eta: float = 1e-6
    ):
        """
        Initialize Berry curvature calculator.
        
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
        
        v_μ = (1/ℏ) ∂H/∂k_μ
        
        Parameters:
            kx, ky: Momentum point
            direction: 'x' or 'y'
            dk: Finite difference step
            
        Returns:
            Velocity operator matrix (in eV·Å units with ℏ=1)
        """
        if direction == 'x':
            H_plus = self.hamiltonian_func(kx + dk, ky)
            H_minus = self.hamiltonian_func(kx - dk, ky)
        elif direction == 'y':
            H_plus = self.hamiltonian_func(kx, ky + dk)
            H_minus = self.hamiltonian_func(kx, ky - dk)
        else:
            raise ValueError("direction must be 'x' or 'y'")
        
        # Central difference
        v_matrix = (H_plus - H_minus) / (2 * dk)
        
        return v_matrix
    
    def berry_curvature_kubo(
        self,
        kx: float,
        ky: float,
        band_indices: Optional[list] = None,
        dk: float = 1e-4
    ) -> np.ndarray:
        """
        Calculate Berry curvature using Kubo formula.
        
        Ω_n(k) = -2 Im Σ_{m≠n} v^x_nm v^y_mn / (E_n - E_m)^2
        
        Parameters:
            kx, ky: Momentum point
            band_indices: List of band indices to calculate (default: all)
            dk: Finite difference step for velocity
            
        Returns:
            Array of Berry curvature for each band
        """
        # Diagonalize Hamiltonian
        H = self.hamiltonian_func(kx, ky)
        energies, eigvecs = np.linalg.eigh(H)
        num_bands = len(energies)
        
        if band_indices is None:
            band_indices = range(num_bands)
        
        # Calculate velocity matrices in band basis
        v_x = self.velocity_matrix(kx, ky, 'x', dk)
        v_y = self.velocity_matrix(kx, ky, 'y', dk)
        
        # Transform to band basis
        v_x_band = eigvecs.T.conj() @ v_x @ eigvecs
        v_y_band = eigvecs.T.conj() @ v_y @ eigvecs
        
        # Calculate Berry curvature for each band
        berry_curv = np.zeros(num_bands, dtype=float)
        
        for n in band_indices:
            omega_n = 0.0
            for m in range(num_bands):
                if m == n:
                    continue
                
                # Energy denominator with regularization
                dE = energies[n] - energies[m]
                denom = dE**2 + self.eta**2
                
                # Velocity matrix elements
                v_x_nm = v_x_band[n, m]
                v_y_mn = v_y_band[m, n]
                
                # Berry curvature contribution
                omega_n += -2 * np.imag(v_x_nm * v_y_mn) / denom
            
            berry_curv[n] = omega_n
        
        return berry_curv
    
    def berry_curvature_kubo_layer(
        self,
        kx: float,
        ky: float,
        layer_projector: np.ndarray,
        band_indices: Optional[list] = None,
        dk: float = 1e-4
    ) -> np.ndarray:
        """
        Calculate layer-resolved Berry curvature.
        
        Ω_n^(l)(k) = -2 Im Σ_{m≠n} <n|P_l v_x|m><m|v_y|n> / (E_n - E_m)^2
        
        Parameters:
            kx, ky: Momentum point
            layer_projector: Projection operator for the layer (P_l)
            band_indices: List of band indices to calculate
            dk: Finite difference step
            
        Returns:
            Array of layer-resolved Berry curvature for each band
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
        
        # Layer-weighted velocity: P_l @ v_x
        v_x_layer = layer_projector @ v_x
        
        # Transform to band basis
        v_x_layer_band = eigvecs.T.conj() @ v_x_layer @ eigvecs
        v_y_band = eigvecs.T.conj() @ v_y @ eigvecs
        
        # Calculate layer Berry curvature
        berry_curv_layer = np.zeros(num_bands, dtype=float)
        
        for n in band_indices:
            omega_n_layer = 0.0
            for m in range(num_bands):
                if m == n:
                    continue
                
                dE = energies[n] - energies[m]
                denom = dE**2 + self.eta**2
                
                v_x_layer_nm = v_x_layer_band[n, m]
                v_y_mn = v_y_band[m, n]
                
                omega_n_layer += -2 * np.imag(v_x_layer_nm * v_y_mn) / denom
            
            berry_curv_layer[n] = omega_n_layer
        
        return berry_curv_layer
    
    def berry_connection(
        self,
        kx: float,
        ky: float,
        direction: str,
        band_indices: Optional[list] = None,
        dk: float = 1e-4
    ) -> np.ndarray:
        """
        Calculate Berry connection A_n^μ(k) = i<u_n|∂_k_μ u_n>.
        
        Uses finite difference of eigenstates.
        
        Parameters:
            kx, ky: Momentum point
            direction: 'x' or 'y'
            band_indices: Bands to calculate
            dk: Finite difference step
            
        Returns:
            Berry connection for each band
        """
        # Get eigenstates at k
        H0 = self.hamiltonian_func(kx, ky)
        E0, U0 = np.linalg.eigh(H0)
        
        # Get eigenstates at k + dk
        if direction == 'x':
            H1 = self.hamiltonian_func(kx + dk, ky)
        elif direction == 'y':
            H1 = self.hamiltonian_func(kx, ky + dk)
        else:
            raise ValueError("direction must be 'x' or 'y'")
        
        E1, U1 = np.linalg.eigh(H1)
        
        num_bands = len(E0)
        if band_indices is None:
            band_indices = range(num_bands)
        
        berry_conn = np.zeros(num_bands, dtype=float)
        
        for n in band_indices:
            # Overlap <u_n(k)|u_n(k+dk)>
            overlap = np.vdot(U0[:, n], U1[:, n])
            
            # Berry connection from logarithmic derivative
            # A_n = Im[ln(overlap)] / dk
            berry_conn[n] = np.angle(overlap) / dk
        
        return berry_conn
    
    def berry_curvature_finite_diff(
        self,
        kx: float,
        ky: float,
        band_indices: Optional[list] = None,
        dk: float = 1e-4
    ) -> np.ndarray:
        """
        Calculate Berry curvature from finite difference of Berry connection.
        
        Ω_n = ∂_kx A_n^y - ∂_ky A_n^x
        
        Parameters:
            kx, ky: Momentum point
            band_indices: Bands to calculate
            dk: Finite difference step
            
        Returns:
            Berry curvature for each band
        """
        # A^y at kx ± dk
        A_y_plus = self.berry_connection(kx + dk, ky, 'y', band_indices, dk)
        A_y_minus = self.berry_connection(kx - dk, ky, 'y', band_indices, dk)
        
        # A^x at ky ± dk
        A_x_plus = self.berry_connection(kx, ky + dk, 'x', band_indices, dk)
        A_x_minus = self.berry_connection(kx, ky - dk, 'x', band_indices, dk)
        
        # Finite difference
        dA_y_dkx = (A_y_plus - A_y_minus) / (2 * dk)
        dA_x_dky = (A_x_plus - A_x_minus) / (2 * dk)
        
        berry_curv = dA_y_dkx - dA_x_dky
        
        return berry_curv
    
    def chern_number(
        self,
        k_mesh: np.ndarray,
        band_index: int,
        method: str = 'kubo',
        dk: float = 1e-4
    ) -> float:
        """
        Calculate Chern number for a single band.
        
        C_n = (1/2π) ∫ dk Ω_n(k)
        
        Parameters:
            k_mesh: Array of shape (nk, nk, 2) with (kx, ky)
            band_index: Band to calculate
            method: 'kubo' or 'finite_diff'
            dk: Finite difference step
            
        Returns:
            Chern number (should be integer for gapped bands)
        """
        nk_x, nk_y = k_mesh.shape[:2]
        
        # Calculate Berry curvature at each k-point
        berry_curv_map = np.zeros((nk_x, nk_y))
        
        for i in range(nk_x):
            for j in range(nk_y):
                kx, ky = k_mesh[i, j]
                
                if method == 'kubo':
                    berry_curv = self.berry_curvature_kubo(
                        kx, ky, [band_index], dk
                    )
                elif method == 'finite_diff':
                    berry_curv = self.berry_curvature_finite_diff(
                        kx, ky, [band_index], dk
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                berry_curv_map[i, j] = berry_curv[band_index]
        
        # Integrate over BZ
        # Area element: dk_x * dk_y
        dk_x = np.abs(k_mesh[1, 0, 0] - k_mesh[0, 0, 0])
        dk_y = np.abs(k_mesh[0, 1, 1] - k_mesh[0, 0, 1])
        
        chern = np.sum(berry_curv_map) * dk_x * dk_y / (2 * np.pi)
        
        return chern


def calculate_berry_curvature_map(
    model,
    k_range: Tuple[float, float],
    nk: int = 100,
    band_indices: Optional[list] = None,
    method: str = 'kubo',
    dk: float = 1e-4,
    eta: float = 1e-6
) -> dict:
    """
    Calculate Berry curvature over 2D k-space grid.
    
    Parameters:
        model: Tight-binding model object with hamiltonian(kx, ky) method
        k_range: (k_min, k_max) for both kx and ky
        nk: Number of k-points along each direction
        band_indices: Bands to calculate (default: all)
        method: 'kubo' or 'finite_diff'
        dk: Finite difference step
        eta: Regularization parameter
        
    Returns:
        Dictionary with 'kx', 'ky', 'berry_curvature', 'chern_numbers'
    """
    k_min, k_max = k_range
    kx_array = np.linspace(k_min, k_max, nk)
    ky_array = np.linspace(k_min, k_max, nk)
    
    KX, KY = np.meshgrid(kx_array, ky_array)
    
    # Initialize calculator
    calc = BerryCurvatureCalculator(model.hamiltonian, eta=eta)
    
    # Determine number of bands
    H_test = model.hamiltonian(0.0, 0.0)
    num_bands = H_test.shape[0]
    
    if band_indices is None:
        band_indices = list(range(num_bands))
    
    # Calculate Berry curvature
    berry_curv_maps = np.zeros((len(band_indices), nk, nk))
    
    for i in range(nk):
        for j in range(nk):
            kx, ky = KX[i, j], KY[i, j]
            
            if method == 'kubo':
                berry_curv = calc.berry_curvature_kubo(kx, ky, band_indices, dk)
            elif method == 'finite_diff':
                berry_curv = calc.berry_curvature_finite_diff(kx, ky, band_indices, dk)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            for idx, n in enumerate(band_indices):
                berry_curv_maps[idx, i, j] = berry_curv[n]
    
    # Calculate Chern numbers
    dk_area = (k_max - k_min)**2 / (nk**2)
    chern_numbers = {}
    
    for idx, n in enumerate(band_indices):
        chern = np.sum(berry_curv_maps[idx]) * dk_area / (2 * np.pi)
        chern_numbers[n] = chern
    
    result = {
        'kx': KX,
        'ky': KY,
        'berry_curvature': {n: berry_curv_maps[idx] 
                            for idx, n in enumerate(band_indices)},
        'chern_numbers': chern_numbers,
        'band_indices': band_indices
    }
    
    return result


def calculate_layer_berry_curvature_map(
    model,
    layer: int,
    k_range: Tuple[float, float],
    nk: int = 100,
    band_indices: Optional[list] = None,
    dk: float = 1e-4,
    eta: float = 1e-6
) -> dict:
    """
    Calculate layer-resolved Berry curvature over 2D k-space.
    
    Parameters:
        model: Model with layer_projection_operator method
        layer: Layer index (1 or 2)
        k_range: (k_min, k_max) for kx and ky
        nk: Number of k-points
        band_indices: Bands to calculate
        dk: Finite difference step
        eta: Regularization
        
    Returns:
        Dictionary with layer-resolved Berry curvature maps
    """
    k_min, k_max = k_range
    kx_array = np.linspace(k_min, k_max, nk)
    ky_array = np.linspace(k_min, k_max, nk)
    
    KX, KY = np.meshgrid(kx_array, ky_array)
    
    # Get layer projection operator
    P_layer = model.layer_projection_operator(layer)
    
    # Initialize calculator
    calc = BerryCurvatureCalculator(model.hamiltonian, eta=eta)
    
    # Determine bands
    H_test = model.hamiltonian(0.0, 0.0)
    num_bands = H_test.shape[0]
    
    if band_indices is None:
        band_indices = list(range(num_bands))
    
    # Calculate layer Berry curvature
    berry_curv_layer_maps = np.zeros((len(band_indices), nk, nk))
    
    for i in range(nk):
        for j in range(nk):
            kx, ky = KX[i, j], KY[i, j]
            
            berry_curv_layer = calc.berry_curvature_kubo_layer(
                kx, ky, P_layer, band_indices, dk
            )
            
            for idx, n in enumerate(band_indices):
                berry_curv_layer_maps[idx, i, j] = berry_curv_layer[n]
    
    result = {
        'kx': KX,
        'ky': KY,
        'berry_curvature_layer': {n: berry_curv_layer_maps[idx] 
                                   for idx, n in enumerate(band_indices)},
        'layer': layer,
        'band_indices': band_indices
    }
    
    return result


if __name__ == "__main__":
    """
    Example usage and tests.
    """
    print("=" * 60)
    print("Berry Curvature Calculator - Test")
    print("=" * 60)
    
    # Simple 2-band model for testing: Haldane model
    def haldane_hamiltonian(kx, ky):
        """Simple Haldane model for testing."""
        t1 = 1.0
        t2 = 0.3
        M = 0.5
        phi = np.pi / 2
        
        # Nearest neighbor
        h = -2 * t1 * (np.cos(kx) + np.cos(ky) + np.cos(kx + ky))
        
        # Next-nearest neighbor (complex)
        h_nnn = -2 * t2 * (np.cos(kx - ky) * np.cos(phi) + 
                           np.cos(2*kx + ky) * np.cos(phi) + 
                           np.cos(kx + 2*ky) * np.cos(phi))
        
        h_nnn_im = -2 * t2 * (np.sin(kx - ky) * np.sin(phi) + 
                              np.sin(2*kx + ky) * np.sin(phi) + 
                              np.sin(kx + 2*ky) * np.sin(phi))
        
        H = np.array([
            [M + h_nnn, h - 1j * h_nnn_im],
            [h + 1j * h_nnn_im, -M + h_nnn]
        ], dtype=complex)
        
        return H
    
    # Test Berry curvature calculation
    calc = BerryCurvatureCalculator(haldane_hamiltonian, eta=1e-6)
    
    kx, ky = 0.1, 0.2
    berry_curv = calc.berry_curvature_kubo(kx, ky)
    
    print(f"\nBerry curvature at k=({kx}, {ky}):")
    print(f"  Band 0: {berry_curv[0]:.6f}")
    print(f"  Band 1: {berry_curv[1]:.6f}")
    print(f"  Sum: {berry_curv[0] + berry_curv[1]:.6f}")
    
    print(f"\n{'='*60}")
    print("Berry curvature calculator initialized successfully!")
    print('='*60)
