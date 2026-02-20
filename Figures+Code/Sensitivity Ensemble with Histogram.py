"""
Stochastic Hyperdiffusion Sensitivity – Tweaked for Observational Alignment
===========================================================================

Default parameters produce w_DE(z=0) ≈ -1 at low noise.
Includes side-by-side comparison with original parameters.
Automatically saves w_values arrays and generates histogram for selected σ.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Model Parameters – TWEAKED DEFAULTS (locked in)
# =============================================================================
Omega_m     = 0.3
nu          = 0.03
m           = 0.2
beta        = 1.0          # Reduced from 10.0 → less nonlinear amplification
kappa       = 0.1          # Increased from 0.01 → stronger regularization
phi0        = 1.0
dt          = 0.005
initial_phi = 4.0          # Reduced from 5.0 → better frozen attractor

# Sigma values to test
sigma_values = [0.020, 0.050, 0.100]

# Number of realizations per sigma
n_realizations = 50000  # change to 50000 for production runs

# Which sigma to plot histogram for (must be in sigma_values)
histogram_sigma = 0.050

# Run comparison mode? (tweaked vs original)
run_comparison = True

# Original (untweaked) parameters for comparison
orig_beta        = 10.0
orig_kappa       = 0.01
orig_initial_phi = 5.0

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
def run_single_realization(sigma, beta=beta, kappa=kappa, initial_phi=initial_phi):
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
# Run ensemble for given parameter set + save w_values
# =============================================================================
def run_ensemble(sigma_list, beta_val, kappa_val, phi_val, label):
    results = {}
    print(f"\nRunning {label} (n={n_realizations:,})...")
    for sigma in sigma_list:
        w_values = []
        for i in range(n_realizations):
            w = run_single_realization(sigma, beta_val, kappa_val, phi_val)
            w_values.append(w)
            if (i + 1) % 5000 == 0:
                print(".", end="", flush=True)
        print(" done.")

        mean_w = np.mean(w_values)
        std_w = np.std(w_values)
        results[sigma] = (mean_w, std_w)

        # Save raw w_values for histogram later
        filename = f"w_values_{label.lower().replace(' ', '_')}_sigma_{sigma:.3f}.npy"
        np.save(filename, np.array(w_values))
        print(f"  Saved: {filename}")

        print(f"  σ = {sigma:.3f} → mean w(0) = {mean_w:+.4f} ± {std_w:.4f}")

    return results

# =============================================================================
# Deterministic baseline
# =============================================================================
print("Deterministic baseline (tweaked parameters)...")
w_det = run_single_realization(0.0)
print(f"  w_DE(z=0) = {w_det:+.4f}\n")

# =============================================================================
# Run ensembles
# =============================================================================
tweaked_results = run_ensemble(sigma_values, beta, kappa, initial_phi, "Tweaked")

if run_comparison:
    print("\n" + "="*60 + "\n")
    original_results = run_ensemble(sigma_values, orig_beta, orig_kappa, orig_initial_phi, "Original")

# =============================================================================
# Main sensitivity plot (w vs σ)
# =============================================================================
plt.figure(figsize=(10, 7))

def plot_results(results, title, color='darkblue', marker='o', label_prefix=""):
    sigmas = np.array(list(results.keys()))
    means = np.array([results[s][0] for s in sigmas])
    stds = np.array([results[s][1] for s in sigmas])
    plt.errorbar(sigmas, means, yerr=stds, fmt=f'{marker}-', capsize=6,
                 linewidth=2, markersize=10, color=color,
                 label=label_prefix + title, elinewidth=1.5, capthick=1.5)

# Tweaked
plot_results(tweaked_results, "Tweaked parameters", color='blue', marker='o', label_prefix="")

# Original (if enabled)
if run_comparison:
    plot_results(original_results, "Original parameters", color='darkorange', marker='s', label_prefix="")

# Deterministic reference
plt.plot(1e-4, w_det, 'P', markersize=14, color='limegreen',
         label=f'Deterministic (σ=0): {w_det:.4f}')

plt.axhline(y=-1, color='gray', linestyle='--', linewidth=1.2, label='w = −1 (Λ)')
plt.axhline(y=-1/3, color='darkred', linestyle=':', linewidth=1.0, label='w = −1/3 (no acceleration)')

plt.xscale('log')
plt.xlabel('Noise strength σ', fontsize=14)
plt.ylabel('Effective w_DE(z=0)', fontsize=14)
plt.title(f'Stochastic Hyperdiffusion Sensitivity Comparison\n(n = {n_realizations:,} per σ, dt = {dt})', fontsize=15)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=11, loc='upper left')
plt.ylim(-1.2, 1.0)
plt.tight_layout()
plt.savefig("w_DE_comparison_tweaked_vs_original.png", bbox_inches='tight')
plt.savefig("w_DE_comparison_tweaked_vs_original.pdf", format='pdf', bbox_inches='tight')
plt.show()

print("Main sensitivity plot saved.")

# =============================================================================
# Histogram for chosen sigma
# =============================================================================
print(f"\nGenerating histogram for σ = {histogram_sigma:.3f}")

# Load saved tuned values (adjust filename if needed)
tuned_filename = f"w_values_tweaked_sigma_{histogram_sigma:.3f}.npy"
w_tuned = np.load(tuned_filename)

fig, ax = plt.subplots(figsize=(8.0, 5.5))

ax.hist(w_tuned, bins=80, alpha=0.75, density=True,
        color='blue', edgecolor='none', label='Tweaked parameters')

ax.axvline(-1.0, color='gray', linestyle='--', linewidth=1.4,
           label='w = −1 (cosmological constant)')
ax.axvline(-1/3, color='darkred', linestyle=':', linewidth=1.4,
           label='w = −1/3 (no acceleration)')

ax.set_xlabel('Effective w_DE(z=0)', fontsize=11)
ax.set_ylabel('Probability Density', fontsize=11)
ax.set_title(f'Distribution of w(0) at σ = {histogram_sigma:.3f} (n={n_realizations:,})', fontsize=11, pad=12)
ax.legend(fontsize=11, loc='upper right', frameon=True, edgecolor='gray')
ax.grid(True, which='major', linestyle='--', alpha=0.35)
ax.tick_params(axis='both', labelsize=11)
ax.set_xlim(-1.3, 1.3)

plt.tight_layout()
plt.savefig(f"w_distribution_sigma_{histogram_sigma:.3f}.png", dpi=400, bbox_inches='tight')
plt.savefig(f"w_distribution_sigma_{histogram_sigma:.3f}.pdf", format='pdf', bbox_inches='tight')
plt.show()

print(f"Histogram saved as 'w_distribution_sigma_{histogram_sigma:.3f}.png/pdf'")
print("All done.")
