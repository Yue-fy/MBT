"""
Tight-Binding Model for MnBi2Te4 Antiferromagnetic Bilayer

This module implements a multi-orbital tight-binding Hamiltonian for MnBi2Te4,
including:
- Triangular lattice kinetic terms
- Spin-orbit coupling (Kane-Mele type)
- Exchange coupling (antiferromagnetic between layers)
- Inter-layer hopping

Author: Yue
Date: November 2025
"""

import numpy as np
import scipy.linalg as la
from typing import Tuple, Optional, Dict
import warnings

class MnBi2Te4_Model:
    """
    Tight-binding model for MnBi2Te4 antiferromagnetic bilayer.
    
    The model includes two layers with opposite magnetization and
    spin-orbit coupling on a triangular lattice.
    
    Attributes:
        a: Lattice constant (Angstrom)
        t: Nearest-neighbor hopping (eV)
        lambda_SO: Spin-orbit coupling strength (eV)
        M: Exchange field magnitude (eV)
        t_perp_0: Inter-layer hopping (constant term) (eV)
        t_perp_1: Inter-layer hopping (k-dependent term) (eV)
        mu: Chemical potential (eV)
    """
    
    def __init__(
        self,
        a: float = 4.38,          # Lattice constant in Angstrom
        t: float = 1.0,           # Hopping energy in eV
        lambda_SO: float = 0.3,   # SOC strength in eV
        M: float = 0.5,           # Exchange field in eV
        t_perp_0: float = 0.2,    # Inter-layer hopping
        t_perp_1: float = 0.05,   # Inter-layer hopping (k-dependent)
        mu: float = 0.0           # Chemical potential in eV
    ):
        """
        Initialize the MnBi2Te4 tight-binding model.
        
        Parameters:
            a: Lattice constant
            t: Nearest-neighbor hopping energy
            lambda_SO: Spin-orbit coupling strength
            M: Exchange field magnitude (AFM: opposite for each layer)
            t_perp_0: Inter-layer hopping (constant)
            t_perp_1: Inter-layer hopping (k-dependent)
            mu: Chemical potential
        """
        self.a = a
        self.t = t
        self.lambda_SO = lambda_SO
        self.M = M
        self.t_perp_0 = t_perp_0
        self.t_perp_1 = t_perp_1
        self.mu = mu
        
        # Define nearest-neighbor vectors for triangular lattice
        self.delta = self._get_nn_vectors()
        
        # Pauli matrices
        self.sigma_0 = np.eye(2, dtype=complex)
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
    def _get_nn_vectors(self) -> np.ndarray:
        """
        Get nearest-neighbor vectors for triangular lattice.
        
        Returns:
            delta: Array of shape (3, 2) containing the three nn vectors
        """
        delta = np.array([
            [self.a, 0],
            [self.a * (-0.5), self.a * (np.sqrt(3)/2)],
            [self.a * (-0.5), self.a * (-np.sqrt(3)/2)]
        ])
        return delta
    
    def h0(self, kx: float, ky: float) -> float:
        """
        Kinetic energy term (nearest-neighbor hopping on triangular lattice).
        
        Parameters:
            kx, ky: Momentum components
            
        Returns:
            Kinetic energy
        """
        k = np.array([kx, ky])
        h = 0.0
        for i in range(3):
            h += -2 * self.t * np.cos(np.dot(k, self.delta[i]))
        return h
    
    def h_SOC(self, kx: float, ky: float) -> Tuple[float, float, float]:
        """
        Spin-orbit coupling terms (Kane-Mele type).
        
        Returns the vector (h_x, h_y, h_z) for Pauli matrix expansion.
        
        Parameters:
            kx, ky: Momentum components
            
        Returns:
            Tuple of (h_SOC_x, h_SOC_y, h_SOC_z)
        """
        k = np.array([kx, ky])
        
        # SOC has the form: sum_i sin(k·delta_i) (z × delta_i)
        h_x = 0.0
        h_y = 0.0
        
        for i in range(3):
            sin_term = np.sin(np.dot(k, self.delta[i]))
            # z × delta_i gives (-delta_y, delta_x)
            h_x += self.lambda_SO * sin_term * (-self.delta[i, 1])
            h_y += self.lambda_SO * sin_term * (self.delta[i, 0])
        
        h_z = 0.0  # No z-component for in-plane SOC
        
        return h_x, h_y, h_z
    
    def t_perp(self, kx: float, ky: float) -> float:
        """
        Inter-layer hopping (k-dependent).
        
        Parameters:
            kx, ky: Momentum components
            
        Returns:
            Inter-layer hopping amplitude
        """
        k = np.array([kx, ky])
        t_k = self.t_perp_0
        for i in range(3):
            t_k += self.t_perp_1 * np.cos(np.dot(k, self.delta[i]))
        return t_k
    
    def hamiltonian_layer(
        self, 
        kx: float, 
        ky: float, 
        layer_sign: int
    ) -> np.ndarray:
        """
        Construct intra-layer Hamiltonian for a single layer.
        
        H_layer = h0(k) * I_2x2 + h_SOC · sigma + M * layer_sign * sigma_z
        
        Parameters:
            kx, ky: Momentum components
            layer_sign: +1 for top layer, -1 for bottom layer
            
        Returns:
            2x2 Hamiltonian matrix (spin space)
        """
        # Kinetic term
        h_kin = self.h0(kx, ky)
        
        # SOC terms
        h_x, h_y, h_z = self.h_SOC(kx, ky)
        
        # Exchange field (opposite for each layer)
        M_eff = layer_sign * self.M
        
        # Construct Hamiltonian
        H = (h_kin * self.sigma_0 + 
             h_x * self.sigma_x + 
             h_y * self.sigma_y + 
             (h_z + M_eff) * self.sigma_z)
        
        return H
    
    def hamiltonian(self, kx: float, ky: float) -> np.ndarray:
        """
        Construct full bilayer Hamiltonian.
        
        Structure:
            H = [[H_11, H_12],
                 [H_21, H_22]]
        
        where H_11, H_22 are 2x2 intra-layer blocks (spin space)
        and H_12, H_21 are inter-layer coupling.
        
        Parameters:
            kx, ky: Momentum components (in units of 1/Angstrom)
            
        Returns:
            4x4 Hamiltonian matrix
        """
        # Intra-layer Hamiltonians
        H_11 = self.hamiltonian_layer(kx, ky, layer_sign=+1)  # Top layer
        H_22 = self.hamiltonian_layer(kx, ky, layer_sign=-1)  # Bottom layer
        
        # Inter-layer coupling
        t_k = self.t_perp(kx, ky)
        H_12 = t_k * self.sigma_0
        H_21 = t_k * self.sigma_0  # Hermitian
        
        # Assemble full Hamiltonian
        H = np.block([
            [H_11, H_12],
            [H_21, H_22]
        ])
        
        # Subtract chemical potential
        H = H - self.mu * np.eye(4)
        
        return H
    
    def solve_bands(
        self, 
        kx: float, 
        ky: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize Hamiltonian at a single k-point.
        
        Parameters:
            kx, ky: Momentum components
            
        Returns:
            energies: Array of eigenvalues (sorted)
            eigenvectors: Array of eigenvectors (columns)
        """
        H = self.hamiltonian(kx, ky)
        energies, eigenvectors = la.eigh(H)
        return energies, eigenvectors
    
    def band_structure_path(
        self, 
        k_path: np.ndarray, 
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate band structure along a specified path in k-space.
        
        Parameters:
            k_path: Array of shape (N_kpts, 2) with high-symmetry points
            num_points: Number of points along the path
            
        Returns:
            k_distances: Array of k-space distances along path
            bands: Array of shape (num_bands, num_points) with energies
        """
        # Interpolate path
        k_interp = []
        distances = [0]
        
        for i in range(len(k_path) - 1):
            k_start = k_path[i]
            k_end = k_path[i + 1]
            segment = np.linspace(k_start, k_end, num_points, endpoint=False)
            k_interp.append(segment)
            
            # Calculate distance
            segment_length = np.linalg.norm(k_end - k_start)
            for j in range(num_points):
                distances.append(distances[-1] + segment_length / num_points)
        
        # Add final point
        k_interp.append([k_path[-1]])
        k_interp = np.vstack(k_interp)
        distances = np.array(distances)
        
        # Calculate bands
        num_bands = 4  # Bilayer with spin
        bands = np.zeros((num_bands, len(k_interp)))
        
        for i, k in enumerate(k_interp):
            energies, _ = self.solve_bands(k[0], k[1])
            bands[:, i] = energies
        
        return distances, bands
    
    def fermi_surface(
        self, 
        k_range: Tuple[float, float],
        nk: int = 100,
        band_indices: Optional[list] = None,
        energy: float = 0.0
    ) -> Dict:
        """
        Calculate Fermi surface (or constant energy contour) in BZ.
        
        Parameters:
            k_range: (k_min, k_max) for both kx and ky
            nk: Number of k-points along each direction
            band_indices: List of band indices to include (default: all)
            energy: Energy at which to calculate contour (default: 0)
            
        Returns:
            Dictionary with 'kx', 'ky', 'bands' arrays
        """
        k_min, k_max = k_range
        kx_array = np.linspace(k_min, k_max, nk)
        ky_array = np.linspace(k_min, k_max, nk)
        
        KX, KY = np.meshgrid(kx_array, ky_array)
        
        num_bands = 4
        bands = np.zeros((num_bands, nk, nk))
        
        for i in range(nk):
            for j in range(nk):
                energies, _ = self.solve_bands(KX[i, j], KY[i, j])
                bands[:, i, j] = energies
        
        result = {
            'kx': KX,
            'ky': KY,
            'bands': bands,
            'energy': energy
        }
        
        return result
    
    def layer_projection_operator(self, layer: int) -> np.ndarray:
        """
        Return projection operator onto a specific layer.
        
        For a bilayer system with spin, the basis is:
        [layer1_up, layer1_dn, layer2_up, layer2_dn]
        
        Parameters:
            layer: 1 for top layer, 2 for bottom layer
            
        Returns:
            4x4 projection matrix
        """
        P = np.zeros((4, 4), dtype=complex)
        
        if layer == 1:  # Top layer
            P[0, 0] = 1.0  # Spin up
            P[1, 1] = 1.0  # Spin down
        elif layer == 2:  # Bottom layer
            P[2, 2] = 1.0  # Spin up
            P[3, 3] = 1.0  # Spin down
        else:
            raise ValueError("layer must be 1 or 2")
        
        return P
    
    def get_high_symmetry_path(self) -> Tuple[np.ndarray, list, list]:
        """
        Get standard high-symmetry path for triangular lattice.
        
        Returns:
            k_path: Array of k-points
            labels: Labels for high-symmetry points
            positions: Positions of labels along path
        """
        # High-symmetry points in triangular BZ
        # In units of 2π/a
        G = np.array([0.0, 0.0])  # Gamma
        K = np.array([4*np.pi/(3*self.a), 0.0])  # K
        M = np.array([np.pi/self.a, np.pi/(np.sqrt(3)*self.a)])  # M
        
        k_path = np.array([G, M, K, G])
        labels = ['Γ', 'M', 'K', 'Γ']
        positions = [0, 1, 2, 3]
        
        return k_path, labels, positions


def get_brillouin_zone_2d(nk: int, lattice_type: str = 'triangular') -> np.ndarray:
    """
    Generate k-point mesh for 2D Brillouin zone.
    
    Parameters:
        nk: Number of k-points along each direction
        lattice_type: 'square' or 'triangular'
        
    Returns:
        k_mesh: Array of shape (nk, nk, 2) with (kx, ky) values
    """
    if lattice_type == 'square':
        kx = np.linspace(-np.pi, np.pi, nk, endpoint=False)
        ky = np.linspace(-np.pi, np.pi, nk, endpoint=False)
        KX, KY = np.meshgrid(kx, ky)
        k_mesh = np.stack([KX, KY], axis=-1)
    
    elif lattice_type == 'triangular':
        # For triangular lattice, BZ is hexagonal
        # Use rectangular mesh covering the hexagon
        kx = np.linspace(-4*np.pi/3, 4*np.pi/3, nk, endpoint=False)
        ky = np.linspace(-4*np.pi/(np.sqrt(3)), 4*np.pi/(np.sqrt(3)), nk, endpoint=False)
        KX, KY = np.meshgrid(kx, ky)
        k_mesh = np.stack([KX, KY], axis=-1)
    
    else:
        raise ValueError(f"Unknown lattice_type: {lattice_type}")
    
    return k_mesh


if __name__ == "__main__":
    """
    Example usage and tests.
    """
    print("=" * 60)
    print("MnBi2Te4 Tight-Binding Model")
    print("=" * 60)
    
    # Initialize model with default parameters
    model = MnBi2Te4_Model(
        a=4.38,
        t=1.0,
        lambda_SO=0.3,
        M=0.5,
        t_perp_0=0.2,
        t_perp_1=0.05,
        mu=0.0
    )
    
    print(f"\nModel parameters:")
    print(f"  Lattice constant: {model.a:.2f} Å")
    print(f"  Hopping: {model.t:.2f} eV")
    print(f"  SOC: {model.lambda_SO:.2f} eV")
    print(f"  Exchange field: {model.M:.2f} eV")
    print(f"  Inter-layer hopping: {model.t_perp_0:.2f} eV")
    
    # Test at Gamma point
    print(f"\n{'='*60}")
    print("Testing at Γ point (k=0):")
    print('='*60)
    
    kx, ky = 0.0, 0.0
    H = model.hamiltonian(kx, ky)
    energies, eigvecs = model.solve_bands(kx, ky)
    
    print(f"\nHamiltonian shape: {H.shape}")
    print(f"Eigenvalues: {energies}")
    print(f"Gap: {energies[2] - energies[1]:.4f} eV")
    
    # Test layer projection
    print(f"\n{'='*60}")
    print("Layer projection operators:")
    print('='*60)
    
    P1 = model.layer_projection_operator(1)
    P2 = model.layer_projection_operator(2)
    
    print(f"\nLayer 1 projection:")
    print(P1)
    print(f"\nLayer 2 projection:")
    print(P2)
    print(f"\nOrthogonality check (P1·P2): {np.linalg.norm(P1 @ P2):.2e}")
    print(f"Completeness check (P1+P2): {np.linalg.norm(P1 + P2 - np.eye(4)):.2e}")
    
    # Test high-symmetry path
    print(f"\n{'='*60}")
    print("High-symmetry path:")
    print('='*60)
    
    k_path, labels, positions = model.get_high_symmetry_path()
    print(f"\nPath: {' → '.join(labels)}")
    print(f"K-points:")
    for label, k in zip(labels, k_path):
        print(f"  {label}: ({k[0]:.4f}, {k[1]:.4f}) [2π/a]")
    
    print(f"\n{'='*60}")
    print("Model initialized successfully!")
    print('='*60)
