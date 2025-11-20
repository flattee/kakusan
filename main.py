"""
claude sonnet4.5 input:
I want to solve the neutron diffusion equation numerically. 
For now, please use the finite difference method with a one-dimensional slab system,
setting neutron energy to a single group and applying zero neutron flux boundary conditions at both ends of the slab.
Define the diffusion coefficient = 1, absorption cross-section = 1, and production = 2. Please write the program in Python.
"""


"""
1D Neutron Diffusion Equation Solver
Using Finite Difference Method with Single Energy Group

Equation: -D d²φ/dx² + Σ_a φ = (1/k) νΣ_f φ
where:
  D = diffusion coefficient
  Σ_a = absorption cross-section
  νΣ_f = production term (nu * fission cross-section)
  k = effective multiplication factor (eigenvalue)
  φ = neutron flux (eigenfunction)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Problem parameters
D = 1.0           # Diffusion coefficient
sigma_a = 1.0     # Absorption cross-section
nu_sigma_f = 2.0  # Production term (nu * Sigma_f)

# Geometry
slab_length = np.pi  # Length of the slab [cm]
n_points = 101       # Number of spatial points (including boundaries)

# Spatial discretization
dx = slab_length / (n_points - 1)
x = np.linspace(0, slab_length, n_points)

# Number of interior points (excluding boundaries with zero flux)
n_interior = n_points - 2

print("=" * 60)
print("1D Neutron Diffusion Equation Solver")
print("=" * 60)
print(f"Diffusion coefficient D = {D}")
print(f"Absorption cross-section Σ_a = {sigma_a}")
print(f"Production term νΣ_f = {nu_sigma_f}")
print(f"Slab length = {slab_length} cm")
print(f"Number of mesh points = {n_points}")
print(f"Mesh spacing dx = {dx:.4f} cm")
print("=" * 60)

# Build the loss matrix A (diffusion + absorption terms)
# -D d²φ/dx² + Σ_a φ can be written as matrix A operating on φ
A = np.zeros((n_interior, n_interior))

for i in range(n_interior):
    if i > 0:
        A[i, i-1] = -D / dx**2  # Coefficient for φ_{i-1}
    
    A[i, i] = 2*D / dx**2 + sigma_a  # Coefficient for φ_i
    
    if i < n_interior - 1:
        A[i, i+1] = -D / dx**2  # Coefficient for φ_{i+1}

# Build the production matrix F (fission term)
F = np.zeros((n_interior, n_interior))
for i in range(n_interior):
    F[i, i] = nu_sigma_f

# Solve the generalized eigenvalue problem: A φ = (1/k) F φ
# or equivalently: F φ = k A φ
eigenvalues, eigenvectors = eigh(A, F)

# The k-effective is the largest eigenvalue
# (we need to invert because we solved A φ = λ F φ where λ = 1/k)
k_eff_values = 1.0 / eigenvalues
fundamental_mode_idx = np.argmax(k_eff_values)
k_eff = k_eff_values[fundamental_mode_idx]

# Extract the fundamental mode flux (corresponding to largest k-eff)
flux_interior = eigenvectors[:, fundamental_mode_idx]

# Normalize flux so that maximum value is 1
flux_interior = flux_interior / np.max(np.abs(flux_interior))

# Construct the full flux array including boundary conditions (φ = 0)
flux_full = np.zeros(n_points)
flux_full[1:-1] = flux_interior

# Display results
print("\nResults:")
print(f"k-effective = {k_eff:.6f}")
print(f"Maximum flux = {np.max(flux_full):.6f}")
print(f"Flux at center (x={slab_length/2:.1f} cm) = {flux_full[n_points//2]:.6f}")

# Plot the flux distribution
plt.figure(figsize=(10, 6))
plt.plot(x, flux_full, 'b-o', linewidth=2, label='Neutron Flux φ(x)')
plt.xlabel('Position x [cm]', fontsize=12)
plt.ylabel('Normalized Neutron Flux', fontsize=12)
plt.title(f'1D Neutron Flux Distribution (k-eff = {k_eff:.6f})', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xlim([0, slab_length])
plt.ylim([0, 1.1])

# Add text box with parameters
textstr = f'D = {D}\nΣ_a = {sigma_a}\nνΣ_f = {nu_sigma_f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.75, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('./neutron_flux_1d.png', dpi=150)
print("\nPlot saved to: neutron_flux_1d.png")

# Display the first few eigenvalues
print("\nFirst 5 k-effective values:")
sorted_k_values = np.sort(k_eff_values)[::-1]
for i, k_val in enumerate(sorted_k_values[:5]):
    print(f"  Mode {i}: k-eff = {k_val:.6f}")

plt.show()

