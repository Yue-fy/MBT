# Layer Hall Effect in Antiferromagnetic Topological Insulators

## 1. Introduction to Layer Hall Effect

### 1.1 Physical Concept

The **Layer Hall Effect** (LHE) is a transport phenomenon in layered materials where:
- An **in-plane electric field** $\mathbf{E}$ is applied
- A **transverse layer current** $j_z^{(l)}$ is induced perpendicular to both $\mathbf{E}$ and the layer normal
- Different layers accumulate charges with opposite signs

**Analogy:**
- Conventional Hall effect: $\mathbf{E} \perp \mathbf{B}$ → transverse charge current
- Layer Hall effect: $\mathbf{E}$ (in-plane) → transverse **layer-resolved** current

### 1.2 Antiferromagnetic Context

In antiferromagnetic topological insulators like **MnBi₂Te₄**:
- Alternating magnetization in adjacent layers: $\uparrow\downarrow\uparrow\downarrow$
- Time-reversal symmetry $\mathcal{T}$ is broken
- Combined symmetry $\mathcal{S} = \mathcal{T} \times \mathcal{I}_z$ can be preserved ($\mathcal{I}_z$: layer inversion)
- Layer Hall conductivity measures layer-antisymmetric response

## 2. Layer Hall Conductivity

### 2.1 Definition

The **layer Hall conductivity** relates the layer current to the electric field:

$$
j_{\mu}^{(l)} = \sum_{\nu} \sigma_{\mu\nu}^{(l)} E_{\nu}
$$

where:
- $j_{\mu}^{(l)}$: current in direction $\mu$ at layer $l$
- $\sigma_{\mu\nu}^{(l)}$: layer-resolved conductivity tensor
- $E_{\nu}$: electric field in direction $\nu$

The **layer Hall conductivity** is specifically:

$$
\sigma_{xy}^{(l)} = \frac{j_x^{(l)}}{E_y} \bigg|_{E_x=0}
$$

### 2.2 Layer Antisymmetric Part

Define the **layer antisymmetric** conductivity (odd under layer inversion):

$$
\sigma_{xy}^{\text{layer}} = \sum_l (-1)^l \sigma_{xy}^{(l)}
$$

For a bilayer (or two-layer unit cell):

$$
\sigma_{xy}^{\text{layer}} = \sigma_{xy}^{(\text{top})} - \sigma_{xy}^{(\text{bottom})}
$$

## 3. Kubo Formula for Layer Hall Conductivity

### 3.1 Linear Response Theory

From the Kubo formula, the DC layer Hall conductivity is:

$$
\sigma_{xy}^{(l)} = -\frac{e^2}{\hbar} \sum_n \int_{\text{BZ}} \frac{d^2\mathbf{k}}{(2\pi)^2} f(E_n(\mathbf{k})) \Omega_n^{(l)}(\mathbf{k})
$$

where:
- $f(E)$: Fermi-Dirac distribution
- $\Omega_n^{(l)}(\mathbf{k})$: layer-resolved Berry curvature for band $n$

$$
\Omega_n^{(l)}(\mathbf{k}) = -2\text{Im}\sum_{m\neq n}\frac{\langle u_{n\mathbf{k}}|\hat{P}_l v_x|u_{m\mathbf{k}}\rangle\langle u_{m\mathbf{k}}|v_y|u_{n\mathbf{k}}\rangle}{(E_n - E_m)^2}
$$

with $\hat{P}_l$ being the projection operator onto layer $l$.

### 3.2 Zero Temperature Formula

At $T = 0$, summing over occupied bands:

$$
\sigma_{xy}^{(l)} = -\frac{e^2}{\hbar} \sum_{n\in\text{occ}} \int_{\text{BZ}} \frac{d^2\mathbf{k}}{(2\pi)^2} \Omega_n^{(l)}(\mathbf{k})
$$

The **layer antisymmetric** part:

$$
\sigma_{xy}^{\text{layer}} = -\frac{e^2}{\hbar} \sum_{n\in\text{occ}} \int_{\text{BZ}} \frac{d^2\mathbf{k}}{(2\pi)^2} \Delta\Omega_n(\mathbf{k})
$$

where $\Delta\Omega_n(\mathbf{k}) = \Omega_n^{(\text{top})}(\mathbf{k}) - \Omega_n^{(\text{bottom})}(\mathbf{k})$ is the **layer Berry curvature contrast**.

## 4. Intrinsic vs. Quantum Metric Contribution

### 4.1 Full Expression

The complete layer Hall conductivity includes two contributions:

$$
\sigma_{xy}^{\text{layer}} = \sigma_{xy}^{\text{layer,intrinsic}} + \sigma_{xy}^{\text{layer,metric}}
$$

### 4.2 Intrinsic Contribution (Berry Curvature)

From the Berry curvature:

$$
\sigma_{xy}^{\text{layer,intrinsic}} = -\frac{e^2}{\hbar} \sum_{n\in\text{occ}} \int \frac{d^2\mathbf{k}}{(2\pi)^2} \Delta\Omega_n(\mathbf{k})
$$

This is the dominant contribution in **gapped** systems.

### 4.3 Quantum Metric Contribution

At finite doping or in **semimetals**, the quantum metric dipole contributes:

$$
\sigma_{xy}^{\text{layer,metric}} = e^2 \tau \sum_n \int \frac{d^2\mathbf{k}}{(2\pi)^2} \left(-\frac{\partial f}{\partial \epsilon}\right) D_n^{\text{layer}}(\mathbf{k})
$$

where $\tau$ is the scattering time and:

$$
D_n^{\text{layer}}(\mathbf{k}) = \frac{\partial E_n}{\partial k_x} \frac{\partial}{\partial k_y}\left[g_n^{(l)}(\mathbf{k})\right]
$$

is the **layer quantum metric dipole**.

The layer-resolved quantum metric is:

$$
g_n^{(l)}(\mathbf{k}) = \text{Re}\sum_{m\neq n}\frac{|\langle u_{n\mathbf{k}}|\hat{P}_l v_x|u_{m\mathbf{k}}\rangle|^2}{(E_n - E_m)^2}
$$

## 5. MnBi₂Te₄ Tight-Binding Model

### 5.1 Lattice Structure

MnBi₂Te₄ has a **layered rhombohedral structure**:
- Septuple layer (SL) stacking: Te-Bi-Te-Mn-Te-Bi-Te
- Triangular lattice in each layer
- Antiferromagnetic coupling: Mn spins alternate $\uparrow\downarrow$

### 5.2 Effective Bilayer Model

For simplicity, consider an effective **bilayer model** with:
- Two layers: $l = 1$ (top), $l = 2$ (bottom)
- Each layer: Te-Bi-Te-Mn sublattice
- Opposite magnetization: $\mathbf{M}_1 = -\mathbf{M}_2$

### 5.3 Multi-Orbital Tight-Binding Hamiltonian

The Hamiltonian in momentum space:

$$
\mathcal{H}(\mathbf{k}) = \begin{pmatrix}
\mathcal{H}_{11}(\mathbf{k}) & \mathcal{H}_{12}(\mathbf{k}) \\
\mathcal{H}_{21}(\mathbf{k}) & \mathcal{H}_{22}(\mathbf{k})
\end{pmatrix}
$$

where $\mathcal{H}_{ll'}(\mathbf{k})$ couples layer $l$ to $l'$.

**Intra-layer block** $\mathcal{H}_{ll}(\mathbf{k})$:

$$
\mathcal{H}_{ll}(\mathbf{k}) = h_0(\mathbf{k}) \otimes \mathbb{I}_{\sigma} + \mathbf{h}_{\text{SOC}}(\mathbf{k}) \cdot \boldsymbol{\sigma} + M_l \sigma_z
$$

where:
- $h_0(\mathbf{k})$: kinetic term (hopping on triangular lattice)
- $\mathbf{h}_{\text{SOC}}(\mathbf{k})$: spin-orbit coupling (Rashba + Kane-Mele)
- $M_l = (-1)^l M$: exchange field (opposite for each layer)
- $\boldsymbol{\sigma}$: Pauli matrices for spin

**Inter-layer coupling** $\mathcal{H}_{12}(\mathbf{k})$:

$$
\mathcal{H}_{12}(\mathbf{k}) = t_{\perp}(\mathbf{k}) \mathbb{I}_{\sigma}
$$

with $t_{\perp}(\mathbf{k})$ the inter-layer hopping.

### 5.4 Detailed Terms

**Kinetic term** (nearest-neighbor hopping on triangular lattice):

$$
h_0(\mathbf{k}) = -2t \sum_{i=1}^{3} \cos(\mathbf{k}\cdot\boldsymbol{\delta}_i)
$$

where $\boldsymbol{\delta}_i$ are the nearest-neighbor vectors:
$$
\boldsymbol{\delta}_1 = a(1,0), \quad \boldsymbol{\delta}_2 = a\left(-\frac{1}{2}, \frac{\sqrt{3}}{2}\right), \quad \boldsymbol{\delta}_3 = a\left(-\frac{1}{2}, -\frac{\sqrt{3}}{2}\right)
$$

**Spin-orbit coupling** (Kane-Mele type):

$$
\mathbf{h}_{\text{SOC}}(\mathbf{k}) = \lambda_{\text{SO}} \sum_{i=1}^{3} \sin(\mathbf{k}\cdot\boldsymbol{\delta}_i) \hat{\mathbf{z}} \times \boldsymbol{\delta}_i
$$

This gives a $k$-dependent Zeeman-like term:

$$
h_{\text{SOC}}^z(\mathbf{k}) = 2\lambda_{\text{SO}} \left[\sin(k_x a) + \sin\left(\frac{k_x a}{2} - \frac{\sqrt{3}k_y a}{2}\right) + \sin\left(\frac{k_x a}{2} + \frac{\sqrt{3}k_y a}{2}\right)\right]
$$

**Inter-layer hopping**:

$$
t_{\perp}(\mathbf{k}) = t_{\perp}^{(0)} + t_{\perp}^{(1)} \sum_{i=1}^3 \cos(\mathbf{k}\cdot\boldsymbol{\delta}_i)
$$

## 6. Symmetry Analysis

### 6.1 Time-Reversal and Inversion

For antiferromagnetic bilayer:
- **Time-reversal** $\mathcal{T}$: broken by magnetization
- **Spatial inversion** $\mathcal{P}$: may be broken depending on stacking
- **Layer inversion** $\mathcal{I}_z$: exchanges layers $1 \leftrightarrow 2$ and flips spins

Combined symmetry $\mathcal{S} = \mathcal{T} \times \mathcal{I}_z$:
- If $[\mathcal{H}, \mathcal{S}] = 0$, then $\Omega_n^{(1)}(\mathbf{k}) = \Omega_n^{(2)}(\mathcal{T}\mathbf{k})$
- For layer Hall: need $\Omega_n^{(1)}(\mathbf{k}) \neq \Omega_n^{(2)}(\mathbf{k})$

### 6.2 Conditions for Nonzero Layer Hall

For **nonzero** $\sigma_{xy}^{\text{layer}}$:

1. **Break $\mathcal{T}$**: Requires magnetization or magnetic field
2. **Break layer-exchange symmetry**: $\mathcal{H}_{11} \neq \mathcal{H}_{22}$ (different layers)
3. **Preserve some in-plane symmetry** for clean response

**Example scenarios:**
- External electric field breaks $\mathcal{I}_z$ (displacement field)
- Substrate breaks top-bottom symmetry
- Ferroelectric substrate induces layer asymmetry

## 7. Layer Berry Curvature Dipole

### 7.1 Nonlinear Layer Hall Effect

Beyond linear response, the **nonlinear layer Hall effect** arises:

$$
j_x^{(l)} = \chi_{xxyy}^{(l)} E_y^2
$$

The nonlinear conductivity is related to the **layer Berry curvature dipole**:

$$
\chi_{xxyy}^{\text{layer}} = \frac{e^3\tau^2}{\hbar^2} \sum_n \int \frac{d^2\mathbf{k}}{(2\pi)^2} \left(-\frac{\partial f}{\partial \epsilon}\right) \frac{\partial E_n}{\partial k_y} \frac{\partial \Delta\Omega_n}{\partial k_x}
$$

### 7.2 Connection to Wavepacket Dynamics

The layer Hall current can be understood from semiclassical wavepacket dynamics:

$$
\dot{\mathbf{r}}_l = \frac{1}{\hbar}\frac{\partial E_n}{\partial \mathbf{k}} + \frac{e}{\hbar}\mathbf{E} \times \boldsymbol{\Omega}_n^{(l)}
$$

The second term is the **anomalous velocity** with layer-resolved Berry curvature.

## 8. Experimental Signatures

### 8.1 Transport Measurement

Apply in-plane voltage $V_y$ and measure:
- **Layer current**: Using layer-resolved contacts or tunneling probes
- **Layer charge accumulation**: Via capacitance or scanning probe

Expected signal:

$$
\Delta n_{\text{layer}} = \frac{\sigma_{xy}^{\text{layer}}}{e} \frac{V_y}{L_x}
$$

### 8.2 Optical Response

- **Second-harmonic generation** (SHG) with layer resolution
- **Circular photogalvanic effect** (CPGE)
- Both sensitive to layer Berry curvature

### 8.3 Parameter Dependence

Study $\sigma_{xy}^{\text{layer}}$ as a function of:
- **Magnetization** $M$: Controls band gap and Berry curvature
- **Chemical potential** $\mu$: Tunes between gapped and doped regimes
- **Electric displacement field** $D$: Breaks layer symmetry
- **Temperature** $T$: Thermally activates carriers

## 9. Computational Strategy

### 9.1 Workflow

1. **Construct TB Hamiltonian** $\mathcal{H}(\mathbf{k})$ with layer structure
2. **Diagonalize** to get bands $E_n(\mathbf{k})$ and eigenstates $|u_{n\mathbf{k}}\rangle$
3. **Define layer projector** $\hat{P}_l$ (projects onto layer $l$ orbitals)
4. **Calculate velocity matrices**: 
   - $v_x^{(l)} = \hat{P}_l v_x$
   - $v_y = \frac{1}{\hbar}\partial_{k_y}\mathcal{H}$
5. **Compute layer Berry curvature**:
   $$\Omega_n^{(l)}(\mathbf{k}) = -2\text{Im}\sum_{m\neq n}\frac{v_{nm,x}^{(l)} v_{mn,y}}{(E_n - E_m)^2}$$
6. **Integrate over BZ**:
   $$\sigma_{xy}^{(l)} = -\frac{e^2}{\hbar}\sum_{n\in\text{occ}}\int \frac{d^2\mathbf{k}}{(2\pi)^2} \Omega_n^{(l)}(\mathbf{k})$$
7. **Extract layer antisymmetric part**: $\sigma_{xy}^{\text{layer}} = \sigma_{xy}^{(1)} - \sigma_{xy}^{(2)}$

### 9.2 Numerical Considerations

- **k-mesh**: Dense enough to resolve Berry curvature peaks (200×200 minimum)
- **Regularization**: Add $\eta \sim 10^{-5}$ eV to energy denominators
- **Layer projection**: Carefully define which orbitals belong to each layer
- **Symmetry check**: Verify $\Omega_n^{(1)}(-\mathbf{k}) = -\Omega_n^{(2)}(\mathbf{k})$ if $\mathcal{S}$ symmetry holds

## 10. Physical Insights

### 10.1 Relation to Quantum Anomalous Hall Effect

In MnBi₂Te₄:
- **Ferromagnetic state**: $M_1 = M_2$ → QAHE with $\sigma_{xy} = \frac{e^2}{h}$
- **Antiferromagnetic state**: $M_1 = -M_2$ → QAHE cancels, but layer Hall survives

The layer Hall effect is the **remnant topological response** in the AFM state.

### 10.2 Connection to Axion Insulator

MnBi₂Te₄ in AFM state is an **axion insulator**:
- Topological magnetoelectric effect: $\theta = \pi$
- Zero anomalous Hall: $\sigma_{xy}^{\text{total}} = 0$
- Nonzero layer Hall: $\sigma_{xy}^{\text{layer}} \neq 0$

The layer Hall conductivity is a **layer-resolved topological invariant**.

### 10.3 Layer Hall vs. Spin Hall

Comparison:
| Effect | Layer Hall | Spin Hall |
|--------|------------|-----------|
| Conserved quantity | Layer index $l$ | Spin $\sigma$ |
| Symmetry breaking | $\mathcal{I}_z$ | $\mathcal{P} \times \mathcal{T}$ |
| Origin | Layer Berry curvature | Spin Berry curvature |
| Material | Layered AFM | SOC materials |

Both are **topological** transport phenomena!

## References

1. Gao, A., et al. (2021). Layer Hall effect in a 2D topological axion antiferromagnet. *Nature* **595**, 521.
2. Chen, J., et al. (2025). Nonlinear layer hall effect and detection of the hidden berry curvature dipole. *Nature Physics* (in press).
3. Gao, Y., et al. (2023). Quantum metric nonlinear Hall effect in a topological antiferromagnetic heterostructure. *Phys. Rev. Lett.* **130**, 106301.
4. Wang, N., et al. (2023). Quantum-metric-induced nonlinear transport in a topological antiferromagnet. *Nature* **621**, 487.
5. Deng, Y., et al. (2020). Quantum anomalous Hall effect in intrinsic magnetic topological insulator MnBi₂Te₄. *Science* **367**, 895.
