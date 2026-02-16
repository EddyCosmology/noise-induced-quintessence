"""
Sensitivity Test with Parameter Tweaks for Observational Alignment
==================================================================

Integrates stochastic quintessence model, with options to tweak parameters for better w_DE ≈ -1 under noise.
Includes deterministic baseline comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Model Parameters (tweakable)
# =============================================================================
Omega_m   = 0.3
nu        = 0.03
m         = 0.5      # Mass; try 1.0 if needed (but avoid as it pushes w positive)
beta      = 0.5     # Nonlinear; tweak to 1.0 for stability
kappa     = 0.5     # Hyperdiffusion; tweak to 0.1+ for more damping
phi0      = 1.0
dt        = 0.005
initial_phi = 3.5    # High-z phi; tweak to 4.0 for better freezing

# Sigma values
sigma_values = [0.020, 0.050, 0.100]

# Realizations per sigma (start small for tests, e.g. 100; use 10k+ for convergence)
n_realizations = 500000

# Flag to apply tweaks (beta=1.0, initial_phi=4.0, kappa=0.1)
use_tweaks = True  # Set True for observationally aligned runs

if use_tweaks:
    beta = 0.5
    initial_phi = 3.5
    kappa = 0.5  # Optional; test without if spread is still low

# =============================================================================
# Acceleration function
# =============================================================================
def get_acceleration(phi, psi, H, eta, m, beta, kappa, phi0):
    b = 3 * H * psi
    V_prime = m**2 * phi
    noise_term = eta * (phi**2 / phi0**2)
    base = 3 * H * psi + V_prime + beta * phi * psi**4 + noise_term

    if kappa == 0:
        return -base

    A = kappa
    B = 1 + 2 * kappa * b
    C = kappa * b**2 + base

    discriminant = B**2 - 4 * A * C
    u_det = -base

    if discriminant < 0:
        return u_det

    sqrtD = np.sqrt(discriminant)
    u1 = (-B + sqrtD) / (2 * A)
    u2 = (-B - sqrtD) / (2 * A)

    return u1 if abs(u1 - u_det) < abs(u2 - u_det) else u2

# =============================================================================
# Single realization
# =============================================================================
def run_single_realization(sigma):
    a = 1e-3
    phi = initial_phi
    psi = 0.0
    rho_m = Omega_m / a**3
    rho_scalar = 0.5 * psi**2 + 0.5 * m**2 * phi**2
    H = np.sqrt((rho_m + rho_scalar + 1.0) / (1 - 3 * nu))

    while a < 1.0:
        eta = np.random.normal(0, sigma / np.sqrt(dt)) if sigma > 0 else 0.0
        u = get_acceleration(phi, psi, H, eta, m, beta, kappa, phi0)
        a += a * H * dt
        phi += psi * dt
        psi += u * dt
        rho_m = Omega_m / a**3
        rho_scalar = 0.5 * psi**2 + 0.5 * m**2 * phi**2
        H = np.sqrt((rho_m + rho_scalar + 1.0) / (1 - 3 * nu))

    p_scalar = 0.5 * psi**2 - 0.5 * m**2 * phi**2
    rho_vac = 1.0 + 3 * nu * H**2
    p_vac = -rho_vac
    rho_de = rho_vac + rho_scalar
    p_de = p_vac + p_scalar
    w = p_de / rho_de if rho_de != 0 else -1.0
    return w

# =============================================================================
# Deterministic baseline (σ=0, single run)
# =============================================================================
print("Running deterministic baseline...")
w_det = run_single_realization(0.0)
print(f"Deterministic w(0) = {w_det:.4f}")

# =============================================================================
# Stochastic sensitivity test
# =============================================================================
results = {}
print(f"\nRunning stochastic test (n={n_realizations:,}, tweaks={use_tweaks})...\n")

for sigma in sigma_values:
    w_values = [run_single_realization(sigma) for _ in range(n_realizations)]
    mean_w = np.mean(w_values)
    std_w = np.std(w_values)
    results[sigma] = (mean_w, std_w)
    print(f"For sigma = {sigma:.3f}, mean w(0) = {mean_w:+.4f} ± {std_w:.4f}")

# =============================================================================
# Plot with deterministic baseline
# =============================================================================
sigmas = np.array(list(results.keys()))
means = np.array([results[s][0] for s in sigmas])
stds = np.array([results[s][1] for s in sigmas])

plt.figure(figsize=(9, 6))
plt.errorbar(sigmas, means, yerr=stds, fmt='o-', capsize=6, linewidth=2, markersize=10, color='darkblue')

# Add deterministic as point at small σ
plt.plot(1e-4, w_det, 's', markersize=12, color='red', label='Deterministic (σ=0)')
plt.text(1e-4 * 1.5, w_det + 0.05, f"{w_det:.2f}", color='red', fontsize=11)

plt.axhline(y=-1, color='gray', linestyle='--', linewidth=1.0, label='Phantom divide (w = -1)')
plt.xscale('log')
plt.xlabel('Noise strength σ', fontsize=14)
plt.ylabel('Effective w_DE(z=0)', fontsize=14)
plt.title(f'Stochastic Hyperdiffusion Sensitivity (Tweaks: {use_tweaks})\n(n = {n_realizations:,} realizations per σ, dt = {dt})', fontsize=15)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("w_DE_vs_sigma_tweaked.png", dpi=300)
plt.show()

print("\nPlot generated. Check results for observational alignment (w ≈ -1).")
