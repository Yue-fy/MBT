"""
Layer Hall Conductivity Calculation

This module implements the calculation of layer Hall conductivity for
antiferromagnetic bilayer systems.

Key formulas:
    σ_xy^(l) = -(e²/ℏ) Σ_n ∫ dk/(2π)² f(E_n) Ω_n^(l)(k)
    σ_xy^layer = σ_xy^(1) - σ_xy^(2)  (layer antisymmetric part)

Author: Yue
Date: November 2025
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, List
import warnings
from berry_curvature import BerryCurvatureCalculator
from quantum_metric import QuantumMetricCalculator


class LayerHallCalculator:
    """
    Calculate layer Hall conductivity for antiferromagnetic bilayer systems.
    
    The layer Hall conductivity has two contributions:
    1. Intrinsic (Berry curvature): dominant in gapped systems
    2. Quantum metric dipole: important in doped/semimetallic systems
    """
    
    def __init__(
        self,
        model,
        eta: float = 1e-6,
        temperature: float = 0.0
    ):
        """
        Initialize layer Hall calculator.
        
        Parameters:
            model: Tight-binding model with hamiltonian and layer_projection_operator
            eta: Regularization parameter (eV)
            temperature: Temperature in Kelvin (default: 0 for T=0)
        """
        self.model = model
        self.eta = eta
        self.temperature = temperature
        self.kB = 8.617333e-5  # Boltzmann constant in eV/K
        
        # Initialize sub-calculators
        self.berry_calc = BerryCurvatureCalculator(model.hamiltonian, eta)
        self.metric_calc = QuantumMetricCalculator(model.hamiltonian, eta)
    
    def fermi_dirac(self, energy: float, mu: float = 0.0) -> float:
        """
        Fermi-Dirac distribution function.
        
        Parameters:
            energy: Energy value (eV)
            mu: Chemical potential (eV)
            
        Returns:
            Occupation number f(E)
        """
        if self.temperature < 1e-10:
            # Zero temperature: step function
            return 1.0 if energy < mu else 0.0
        else:
            beta = 1.0 / (self.kB * self.temperature)
            x = beta * (energy - mu)
            # Avoid overflow
            if x > 50:
                return 0.0
            elif x < -50:
                return 1.0
            else:
                return 1.0 / (1.0 + np.exp(x))
    
    def fermi_derivative(self, energy: float, mu: float = 0.0) -> float:
        """
        Derivative of Fermi-Dirac distribution: -∂f/∂ε.
        
        This appears in transport formulas for doped systems.
        
        Parameters:
            energy: Energy value (eV)
            mu: Chemical potential (eV)
            
        Returns:
            -∂f/∂ε (positive at Fermi level)
        """
        if self.temperature < 1e-10:
            # Zero temperature: delta function at mu
            # Return 0 for numerical purposes (integrate carefully!)
            return 0.0
        else:
            beta = 1.0 / (self.kB * self.temperature)
            x = beta * (energy - mu)
            # Avoid overflow
            if abs(x) > 50:
                return 0.0
            else:
                sech_squared = 1.0 / (np.cosh(x / 2))**2
                return beta * sech_squared / 4.0
    
    def layer_berry_curvature_contrast(
        self,
        kx: float,
        ky: float,
        band_index: int,
        dk: float = 1e-4
    ) -> float:
        """
        Calculate layer Berry curvature contrast.
        
        ΔΩ_n(k) = Ω_n^(1)(k) - Ω_n^(2)(k)
        
        Parameters:
            kx, ky: Momentum point
            band_index: Band to calculate
            dk: Finite difference step
            
        Returns:
            Layer Berry curvature contrast
        """
        # Layer 1 (top)
        P1 = self.model.layer_projection_operator(1)
        omega1 = self.berry_calc.berry_curvature_kubo_layer(
            kx, ky, P1, [band_index], dk
        )
        
        # Layer 2 (bottom)
        P2 = self.model.layer_projection_operator(2)
        omega2 = self.berry_calc.berry_curvature_kubo_layer(
            kx, ky, P2, [band_index], dk
        )
        
        # Contrast
        delta_omega = omega1[band_index] - omega2[band_index]
        
        return delta_omega
    
    def intrinsic_layer_hall(
        self,
        k_range: Tuple[float, float],
        nk: int = 100,
        occupied_bands: Optional[List[int]] = None,
        mu: float = 0.0,
        dk: float = 1e-4
    ) -> Dict:
        """
        Calculate intrinsic layer Hall conductivity from Berry curvature.
        
        σ_xy^layer = -(e²/ℏ) Σ_n ∫ dk/(2π)² f(E_n) ΔΩ_n(k)
        
        Parameters:
            k_range: (k_min, k_max) for kx and ky
            nk: Number of k-points
            occupied_bands: List of occupied band indices (if None, use mu)
            mu: Chemical potential (eV)
            dk: Finite difference step
            
        Returns:
            Dictionary with conductivity and details
        """
        k_min, k_max = k_range
        kx_array = np.linspace(k_min, k_max, nk)
        ky_array = np.linspace(k_min, k_max, nk)
        
        dk_area = (k_max - k_min)**2 / (nk * nk)
        
        # Determine bands
        H_test = self.model.hamiltonian(0.0, 0.0)
        num_bands = H_test.shape[0]
        
        if occupied_bands is None:
            # Use all bands (will weight by Fermi function)
            bands_to_calc = list(range(num_bands))
        else:
            bands_to_calc = occupied_bands
        
        # Initialize storage
        sigma_layer_intrinsic = 0.0
        delta_omega_maps = {n: np.zeros((nk, nk)) for n in bands_to_calc}
        
        # Integration over BZ
        for i, kx in enumerate(kx_array):
            for j, ky in enumerate(ky_array):
                # Get energies
                energies, _ = self.model.solve_bands(kx, ky)
                
                for n in bands_to_calc:
                    # Fermi weight
                    f_n = self.fermi_dirac(energies[n], mu)
                    
                    # Layer Berry curvature contrast
                    delta_omega_n = self.layer_berry_curvature_contrast(
                        kx, ky, n, dk
                    )
                    
                    delta_omega_maps[n][i, j] = delta_omega_n
                    
                    # Accumulate conductivity
                    sigma_layer_intrinsic += f_n * delta_omega_n * dk_area
        
        # Convert to physical units: σ = -(e²/ℏ) × integral
        # In natural units (ℏ = e = 1), and area in (2π)²:
        e2_h = 1.0  # e²/h in units of conductance quantum
        sigma_layer_intrinsic *= -e2_h / (2 * np.pi)**2
        
        result = {
            'sigma_layer_intrinsic': sigma_layer_intrinsic,
            'delta_omega_maps': delta_omega_maps,
            'kx': kx_array,
            'ky': ky_array,
            'mu': mu,
            'temperature': self.temperature
        }
        
        return result
    
    def layer_resolved_conductivity(
        self,
        k_range: Tuple[float, float],
        nk: int = 100,
        layer: int = 1,
        occupied_bands: Optional[List[int]] = None,
        mu: float = 0.0,
        dk: float = 1e-4
    ) -> Dict:
        """
        Calculate layer-resolved Hall conductivity for a single layer.
        
        σ_xy^(l) = -(e²/ℏ) Σ_n ∫ dk/(2π)² f(E_n) Ω_n^(l)(k)
        
        Parameters:
            k_range: (k_min, k_max)
            nk: Number of k-points
            layer: Layer index (1 or 2)
            occupied_bands: Occupied bands
            mu: Chemical potential
            dk: Finite difference step
            
        Returns:
            Dictionary with layer-resolved conductivity
        """
        k_min, k_max = k_range
        kx_array = np.linspace(k_min, k_max, nk)
        ky_array = np.linspace(k_min, k_max, nk)
        
        dk_area = (k_max - k_min)**2 / (nk * nk)
        
        # Layer projector
        P_layer = self.model.layer_projection_operator(layer)
        
        # Determine bands
        H_test = self.model.hamiltonian(0.0, 0.0)
        num_bands = H_test.shape[0]
        
        if occupied_bands is None:
            bands_to_calc = list(range(num_bands))
        else:
            bands_to_calc = occupied_bands
        
        # Calculate conductivity
        sigma_layer = 0.0
        omega_layer_maps = {n: np.zeros((nk, nk)) for n in bands_to_calc}
        
        for i, kx in enumerate(kx_array):
            for j, ky in enumerate(ky_array):
                energies, _ = self.model.solve_bands(kx, ky)
                
                omega_layer = self.berry_calc.berry_curvature_kubo_layer(
                    kx, ky, P_layer, bands_to_calc, dk
                )
                
                for n in bands_to_calc:
                    f_n = self.fermi_dirac(energies[n], mu)
                    omega_n_layer = omega_layer[n]
                    
                    omega_layer_maps[n][i, j] = omega_n_layer
                    sigma_layer += f_n * omega_n_layer * dk_area
        
        # Convert to physical units
        e2_h = 1.0
        sigma_layer *= -e2_h / (2 * np.pi)**2
        
        result = {
            'sigma_layer': sigma_layer,
            'omega_layer_maps': omega_layer_maps,
            'layer': layer,
            'kx': kx_array,
            'ky': ky_array,
            'mu': mu
        }
        
        return result
    
    def quantum_metric_contribution(
        self,
        k_range: Tuple[float, float],
        nk: int = 100,
        tau: float = 1e-12,  # Scattering time in seconds
        mu: float = 0.0,
        dk: float = 1e-4
    ) -> Dict:
        """
        Calculate quantum metric contribution to layer Hall effect.
        
        This is important in doped or semimetallic systems.
        
        σ_xy^metric ~ e²τ Σ_n ∫ dk/(2π)² (-∂f/∂ε) ∂E_n/∂kx ∂g_n^layer/∂ky
        
        Parameters:
            k_range: (k_min, k_max)
            nk: Number of k-points
            tau: Scattering time (s)
            mu: Chemical potential (eV)
            dk: Finite difference step
            
        Returns:
            Dictionary with quantum metric contribution
        """
        k_min, k_max = k_range
        kx_array = np.linspace(k_min, k_max, nk, endpoint=False)
        ky_array = np.linspace(k_min, k_max, nk, endpoint=False)
        
        dk_area = (k_max - k_min)**2 / (nk * nk)
        dk_deriv = (k_max - k_min) / nk
        
        H_test = self.model.hamiltonian(0.0, 0.0)
        num_bands = H_test.shape[0]
        
        sigma_metric = 0.0
        
        # This is computationally expensive!
        # We need derivatives of quantum metric
        for i, kx in enumerate(kx_array[::2]):  # Subsample for speed
            for j, ky in enumerate(ky_array[::2]):
                energies, _ = self.model.solve_bands(kx, ky)
                
                for n in range(num_bands):
                    # Fermi derivative weight
                    df_dE = self.fermi_derivative(energies[n], mu)
                    
                    if df_dE < 1e-10:
                        continue  # Skip if far from Fermi level
                    
                    # Band velocity
                    E_x_plus, _ = self.model.solve_bands(kx + dk_deriv, ky)
                    E_x_minus, _ = self.model.solve_bands(kx - dk_deriv, ky)
                    dE_dkx = (E_x_plus[n] - E_x_minus[n]) / (2 * dk_deriv)
                    
                    # Layer quantum metric contrast
                    P1 = self.model.layer_projection_operator(1)
                    P2 = self.model.layer_projection_operator(2)
                    
                    g1_center = self.metric_calc.quantum_metric_layer(
                        kx, ky, P1, [n], dk
                    )[n]
                    g1_y_plus = self.metric_calc.quantum_metric_layer(
                        kx, ky + dk_deriv, P1, [n], dk
                    )[n]
                    g1_y_minus = self.metric_calc.quantum_metric_layer(
                        kx, ky - dk_deriv, P1, [n], dk
                    )[n]
                    
                    g2_center = self.metric_calc.quantum_metric_layer(
                        kx, ky, P2, [n], dk
                    )[n]
                    g2_y_plus = self.metric_calc.quantum_metric_layer(
                        kx, ky + dk_deriv, P2, [n], dk
                    )[n]
                    g2_y_minus = self.metric_calc.quantum_metric_layer(
                        kx, ky - dk_deriv, P2, [n], dk
                    )[n]
                    
                    # Derivative of layer metric contrast (use trace)
                    dg1_trace_dky = (g1_y_plus['trace'] - g1_y_minus['trace']) / (2 * dk_deriv)
                    dg2_trace_dky = (g2_y_plus['trace'] - g2_y_minus['trace']) / (2 * dk_deriv)
                    dg_layer_dky = dg1_trace_dky - dg2_trace_dky
                    
                    # Contribution
                    sigma_metric += df_dE * dE_dkx * dg_layer_dky * dk_area * 4  # ×4 for subsampling
        
        # Physical units: e²τ/(2π)²
        e2_h = 1.0
        hbar = 6.582119e-16  # eV·s
        sigma_metric *= e2_h * tau / (hbar * (2 * np.pi)**2)
        
        result = {
            'sigma_layer_metric': sigma_metric,
            'tau': tau,
            'mu': mu,
            'temperature': self.temperature
        }
        
        return result
    
    def total_layer_hall_conductivity(
        self,
        k_range: Tuple[float, float],
        nk: int = 100,
        occupied_bands: Optional[List[int]] = None,
        mu: float = 0.0,
        tau: Optional[float] = None,  # If None, skip metric contribution
        dk: float = 1e-4
    ) -> Dict:
        """
        Calculate total layer Hall conductivity (intrinsic + metric).
        
        Parameters:
            k_range: (k_min, k_max)
            nk: Number of k-points
            occupied_bands: Occupied bands
            mu: Chemical potential
            tau: Scattering time (if None, only intrinsic)
            dk: Finite difference step
            
        Returns:
            Dictionary with total conductivity and components
        """
        print("Calculating intrinsic contribution...")
        intrinsic = self.intrinsic_layer_hall(
            k_range, nk, occupied_bands, mu, dk
        )
        
        sigma_total = intrinsic['sigma_layer_intrinsic']
        
        if tau is not None and self.temperature > 0:
            print("Calculating quantum metric contribution...")
            metric = self.quantum_metric_contribution(
                k_range, nk, tau, mu, dk
            )
            sigma_total += metric['sigma_layer_metric']
        else:
            metric = {'sigma_layer_metric': 0.0}
        
        result = {
            'sigma_layer_total': sigma_total,
            'sigma_layer_intrinsic': intrinsic['sigma_layer_intrinsic'],
            'sigma_layer_metric': metric['sigma_layer_metric'],
            'delta_omega_maps': intrinsic['delta_omega_maps'],
            'mu': mu,
            'tau': tau,
            'temperature': self.temperature
        }
        
        return result


if __name__ == "__main__":
    """
    Example usage and tests.
    """
    print("=" * 60)
    print("Layer Hall Conductivity Calculator - Test")
    print("=" * 60)
    
    # Import model
    from tb_model import MnBi2Te4_Model
    
    # Initialize model
    model = MnBi2Te4_Model(
        a=4.38,
        t=1.0,
        lambda_SO=0.3,
        M=0.5,
        t_perp_0=0.2,
        mu=0.0
    )
    
    print("\nModel initialized with parameters:")
    print(f"  M = {model.M} eV")
    print(f"  λ_SO = {model.lambda_SO} eV")
    print(f"  t_⊥ = {model.t_perp_0} eV")
    
    # Initialize calculator
    calc = LayerHallCalculator(model, eta=1e-6, temperature=0.0)
    
    print(f"\n{'='*60}")
    print("Testing layer Berry curvature contrast...")
    print('='*60)
    
    kx, ky = 0.1, 0.1
    delta_omega = calc.layer_berry_curvature_contrast(kx, ky, band_index=1)
    
    print(f"\nΔΩ at k=({kx}, {ky}), band 1:")
    print(f"  {delta_omega:.6f} Å²")
    
    print(f"\n{'='*60}")
    print("Layer Hall calculator initialized successfully!")
    print('='*60)
