# Berry Curvature and Quantum Geometry

## 1. Bloch States and Quantum Geometry

### 1.1 Bloch Wavefunctions

For a periodic crystal, the single-particle Bloch states are:

$$
|\psi_{n\mathbf{k}}\rangle = e^{i\mathbf{k}\cdot\mathbf{r}}|u_{n\mathbf{k}}\rangle
$$

where $|u_{n\mathbf{k}}\rangle$ is the periodic part with lattice periodicity, and $n$ is the band index.

The Bloch Hamiltonian satisfies:

$$
\mathcal{H}(\mathbf{k})|u_{n\mathbf{k}}\rangle = E_n(\mathbf{k})|u_{n\mathbf{k}}\rangle
$$

### 1.2 Berry Connection

The **Berry connection** (gauge field in momentum space) is defined as:

$$
\mathcal{A}_{n}^{\mu}(\mathbf{k}) = i\langle u_{n\mathbf{k}}|\frac{\partial}{\partial k_{\mu}}|u_{n\mathbf{k}}\rangle
$$

where $\mu = x, y, z$ denotes the momentum direction.

**Key properties:**
- $\mathcal{A}_n^{\mu}$ is real-valued (from Hermiticity)
- Gauge-dependent: under $|u_{n\mathbf{k}}\rangle \to e^{i\chi(\mathbf{k})}|u_{n\mathbf{k}}\rangle$, we have $\mathcal{A}_n^{\mu} \to \mathcal{A}_n^{\mu} + \partial_{k_\mu}\chi$

**Numerical calculation:**

For discrete tight-binding models:

$$
\mathcal{A}_{n}^{\mu}(\mathbf{k}) = i\langle u_{n\mathbf{k}}|\frac{u_{n,\mathbf{k}+\Delta k_\mu} - u_{n,\mathbf{k}-\Delta k_\mu}}{2\Delta k_\mu}\rangle + O(\Delta k_\mu^2)
$$

More accurate: use the lattice gauge-invariant form:

$$
\mathcal{A}_{n}^{\mu}(\mathbf{k}) = \text{Im}\ln\langle u_{n\mathbf{k}}|u_{n,\mathbf{k}+\Delta k_\mu}\rangle
$$

## 2. Berry Curvature

### 2.1 Definition

The **Berry curvature** is the "magnetic field" in momentum space:

$$
\Omega_{n}^{\mu\nu}(\mathbf{k}) = \partial_{k_\mu}\mathcal{A}_{n}^{\nu}(\mathbf{k}) - \partial_{k_\nu}\mathcal{A}_{n}^{\mu}(\mathbf{k})
$$

In 2D systems (most relevant for layer Hall effect):

$$
\Omega_n(\mathbf{k}) = \Omega_n^{xy}(\mathbf{k}) = \partial_{k_x}\mathcal{A}_{n}^{y} - \partial_{k_y}\mathcal{A}_{n}^{x}
$$

### 2.2 Kubo Formula Expression

**Alternative formula** (gauge-invariant, easier for numerics):

$$
\Omega_n^{\mu\nu}(\mathbf{k}) = -2\text{Im}\sum_{m\neq n}\frac{\langle u_{n\mathbf{k}}|\partial_{k_\mu}\mathcal{H}|u_{m\mathbf{k}}\rangle\langle u_{m\mathbf{k}}|\partial_{k_\nu}\mathcal{H}|u_{n\mathbf{k}}\rangle}{(E_n - E_m)^2}
$$

where $\partial_{k_\mu}\mathcal{H}$ is the velocity operator in direction $\mu$.

**In matrix form:**

$$
\Omega_n^{\mu\nu}(\mathbf{k}) = -2\text{Im}\sum_{m\neq n}\frac{v_{nm}^{\mu}(\mathbf{k})v_{mn}^{\nu}(\mathbf{k})}{(E_n(\mathbf{k}) - E_m(\mathbf{k}))^2}
$$

with velocity matrix elements:

$$
v_{nm}^{\mu}(\mathbf{k}) = \langle u_{n\mathbf{k}}|\partial_{k_\mu}\mathcal{H}(\mathbf{k})|u_{m\mathbf{k}}\rangle = \begin{cases}
\frac{\partial E_n}{\partial k_\mu} & n = m \\
\frac{\langle u_{n\mathbf{k}}|\partial_{k_\mu}\mathcal{H}|u_{m\mathbf{k}}\rangle}{1} & n \neq m
\end{cases}
$$

### 2.3 Numerical Discretization

For tight-binding models on a discrete k-point mesh:

**Method 1: Finite difference of Berry connection**

$$
\Omega_n(\mathbf{k}) \approx \frac{\mathcal{A}_n^y(\mathbf{k}+\Delta k_x\hat{\mathbf{x}}) - \mathcal{A}_n^y(\mathbf{k}-\Delta k_x\hat{\mathbf{x}})}{2\Delta k_x} - \frac{\mathcal{A}_n^x(\mathbf{k}+\Delta k_y\hat{\mathbf{y}}) - \mathcal{A}_n^x(\mathbf{k}-\Delta k_y\hat{\mathbf{y}})}{2\Delta k_y}
$$

**Method 2: Plaquette formula** (gauge-invariant, most stable):

For a small plaquette with corners at $\mathbf{k}, \mathbf{k}+\Delta k_x, \mathbf{k}+\Delta k_x+\Delta k_y, \mathbf{k}+\Delta k_y$:

$$
\Omega_n(\mathbf{k}) = \text{Im}\ln\left[\frac{\langle u_{n\mathbf{k}}|u_{n,\mathbf{k}+\Delta k_x}\rangle\langle u_{n,\mathbf{k}+\Delta k_x}|u_{n,\mathbf{k}+\Delta k_x+\Delta k_y}\rangle}{\langle u_{n,\mathbf{k}+\Delta k_y}|u_{n,\mathbf{k}+\Delta k_x+\Delta k_y}\rangle\langle u_{n\mathbf{k}}|u_{n,\mathbf{k}+\Delta k_y}\rangle}\right]
$$

This is the **Wilson loop** around an infinitesimal plaquette.

**Method 3: Kubo formula** (direct from Hamiltonian):

$$
\Omega_n(\mathbf{k}) = -2\text{Im}\sum_{m\neq n}\frac{\langle u_{n\mathbf{k}}|v_x|u_{m\mathbf{k}}\rangle\langle u_{m\mathbf{k}}|v_y|u_{n\mathbf{k}}\rangle}{(E_n - E_m)^2}
$$

where the velocity operators are:

$$
v_x = \frac{\partial \mathcal{H}}{\partial k_x}, \quad v_y = \frac{\partial \mathcal{H}}{\partial k_y}
$$

## 3. Quantum Metric Tensor

### 3.1 Definition

The **quantum metric** is the real part of the quantum geometric tensor:

$$
g_{n}^{\mu\nu}(\mathbf{k}) = \text{Re}\sum_{m\neq n}\frac{\langle \partial_{k_\mu}u_{n\mathbf{k}}|u_{m\mathbf{k}}\rangle\langle u_{m\mathbf{k}}|\partial_{k_\nu}u_{n\mathbf{k}}\rangle}{1}
$$

Alternative form using projection operators:

$$
g_{n}^{\mu\nu}(\mathbf{k}) = \text{Re}\langle\partial_{k_\mu}u_{n\mathbf{k}}|(1-|u_{n\mathbf{k}}\rangle\langle u_{n\mathbf{k}}|)|\partial_{k_\nu}u_{n\mathbf{k}}\rangle
$$

### 3.2 Connection to Velocity Matrix Elements

$$
g_{n}^{\mu\nu}(\mathbf{k}) = \text{Re}\sum_{m\neq n}\frac{v_{nm}^{\mu}(\mathbf{k})v_{mn}^{\nu}(\mathbf{k})}{(E_n - E_m)^2}
$$

**Note the sign difference** with Berry curvature:
- Berry curvature: $\Omega_n^{\mu\nu} = -2\text{Im}[\cdots]$
- Quantum metric: $g_n^{\mu\nu} = \text{Re}[\cdots]$

### 3.3 Physical Interpretation

The quantum metric:
- Measures the "distance" between Bloch states at nearby k-points
- Related to the band flatness and localization
- Contributes to nonlinear optical responses
- Enters the superfluid weight in flat band systems

### 3.4 Numerical Calculation

**Using Kubo formula:**

$$
g_n^{xy}(\mathbf{k}) = \sum_{m\neq n}\frac{\text{Re}[v_{nm}^x(\mathbf{k})(v_{nm}^y(\mathbf{k}))^*]}{(E_n(\mathbf{k}) - E_m(\mathbf{k}))^2}
$$

**Regularization for near-degeneracies:**

When $|E_n - E_m| < \epsilon_{\text{tol}}$ (band crossing or near-degeneracy), we need regularization:

$$
\frac{1}{(E_n - E_m)^2 + \eta^2}
$$

where $\eta$ is a small regularization parameter (typically $\eta \sim 10^{-6}$ eV).

## 4. Chern Number and Topology

### 4.1 Chern Number

The **Chern number** (topological invariant) is:

$$
C_n = \frac{1}{2\pi}\int_{\text{BZ}} d^2\mathbf{k}\, \Omega_n(\mathbf{k})
$$

For a 2D insulator with a gap, $C = \sum_{\text{occ}} C_n$ is an integer.

**Physical meaning:**
- $C \neq 0$: topologically nontrivial, edge states exist
- Quantum anomalous Hall conductivity: $\sigma_{xy} = C \frac{e^2}{h}$

### 4.2 Numerical Evaluation

On a discrete k-mesh with $N_x \times N_y$ points:

$$
C_n = \frac{1}{2\pi}\sum_{i,j} \Omega_n(\mathbf{k}_{ij}) \Delta k_x \Delta k_y
$$

where $\Delta k_x = \frac{2\pi}{a N_x}$, $\Delta k_y = \frac{2\pi}{a N_y}$ for a square lattice with lattice constant $a$.

## 5. Layer-Resolved Berry Curvature

### 5.1 Multilayer Systems

For a multilayer system with layer index $l$, the **layer-resolved Berry curvature** is:

$$
\Omega_n^{(l)}(\mathbf{k}) = \langle u_{n\mathbf{k}}|\hat{P}_l \Omega_n(\mathbf{k})|u_{n\mathbf{k}}\rangle
$$

where $\hat{P}_l$ projects onto layer $l$.

More explicitly:

$$
\Omega_n^{(l)}(\mathbf{k}) = -2\text{Im}\sum_{m\neq n}\frac{\langle u_{n\mathbf{k}}|\hat{P}_l v_x|u_{m\mathbf{k}}\rangle\langle u_{m\mathbf{k}}|v_y|u_{n\mathbf{k}}\rangle + \text{c.c.}}{(E_n - E_m)^2}
$$

### 5.2 Layer Current Operator

The layer current operator in direction $\mu$ for layer $l$ is:

$$
j_{\mu}^{(l)} = e v_{\mu} \hat{P}_l
$$

where $v_\mu = \frac{1}{\hbar}\partial_{k_\mu}\mathcal{H}$ is the velocity operator.

## 6. Berry Curvature Dipole

### 6.1 Definition

The **Berry curvature dipole** measures the first moment of Berry curvature:

$$
D^{\mu\nu\lambda} = \int_{\text{BZ}} \frac{d^2\mathbf{k}}{(2\pi)^2} \frac{\partial f}{\partial k_\mu} \frac{\partial E(\mathbf{k})}{\partial k_\nu} \Omega^{\lambda}(\mathbf{k})
$$

where $f$ is the Fermi-Dirac distribution.

At zero temperature:

$$
D^{\mu\nu\lambda} = \sum_{\text{occ}} \int_{\text{BZ}} \frac{d^2\mathbf{k}}{(2\pi)^2} \frac{\partial E_n(\mathbf{k})}{\partial k_\mu} \frac{\partial \Omega_n^{\lambda}(\mathbf{k})}{\partial k_\nu}
$$

### 6.2 Symmetry Requirements

For a nonzero Berry curvature dipole:
- **Time-reversal symmetry** must be broken
- **Inversion symmetry** must be broken
- At least $C_{2v}$ symmetry is needed for in-plane response

## 7. Implementation Strategy

### Recommended Numerical Approach

1. **Build tight-binding Hamiltonian** $\mathcal{H}(\mathbf{k})$ on k-mesh
2. **Diagonalize** at each k-point to get $E_n(\mathbf{k})$ and $|u_{n\mathbf{k}}\rangle$
3. **Calculate velocity matrices** $v_{nm}^{\mu}(\mathbf{k})$ from $\partial_{k_\mu}\mathcal{H}$
4. **Compute Berry curvature** using Kubo formula
5. **Compute quantum metric** using same velocity matrices
6. **Integrate** over BZ for Chern numbers and dipole moments

### Numerical Considerations

- **k-mesh density**: At least 100×100 for smooth bands, 300×300 for sharp features
- **Regularization**: Add $\eta \sim 10^{-6}$ eV to avoid divergences at degeneracies
- **Velocity operator**: Use analytical derivative when possible, finite difference as backup
- **Gauge fixing**: Use smooth gauge for Berry connection methods

## References

1. Xiao, D., Chang, M.-C., & Niu, Q. (2010). Berry phase effects on electronic properties. *Rev. Mod. Phys.* **82**, 1959.
2. Resta, R. (2011). The insulating state of matter: A geometrical theory. *Eur. Phys. J. B* **79**, 121.
3. Fukui, T., Hatsugai, Y., & Suzuki, H. (2005). Chern numbers in discretized Brillouin zone. *J. Phys. Soc. Jpn.* **74**, 1674.
4. Sodemann, I., & Fu, L. (2015). Quantum nonlinear Hall effect induced by Berry curvature dipole. *Phys. Rev. Lett.* **115**, 216806.
