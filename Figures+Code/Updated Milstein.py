import numpy as np
import matplotlib.pyplot as plt

def quadratic_solve(a, b, c, tol=1e-8):
    disc = b**2 - 4*a*c
    if disc < -tol:  # Allow tiny negative due to numerics
        disc = 0
    if disc < 0:
        # Approximate with midpoint if no real roots
        return -b / (2*a) if a != 0 else 0
    sqrt_disc = np.sqrt(disc)
    root1 = (-b + sqrt_disc) / (2*a)
    root2 = (-b - sqrt_disc) / (2*a)
    approx_z = -b / (2*a)  # Rough midpoint
    if abs(root1 - approx_z) < abs(root2 - approx_z):
        return root1
    return root2

def single_realization(sigma, params, dt_base=0.002, t_max=2.0, phi0=0.013, Omega_m=1.0, H0=1.0):
    beta = params['beta']
    kappa = params['kappa']
    m = params['m']
    phi_init = params['phi_init']
    
    t = 0.0
    phi = phi_init
    dphi = 0.0
    history = {'t': [t], 'phi': [phi], 'dphi': [dphi], 'H': [H0]}
    
    max_phi = 10 * abs(phi_init)  # Clip bound
    
    while t < t_max:
        # Approximate scale factor a ≈ t + small offset (conformal time proxy)
        a_approx = t + 1e-3
        rho_m = Omega_m / a_approx**3
        rho_phi = 0.5 * dphi**2 + 0.5 * m**2 * phi**2
        H = H0 * np.sqrt(rho_m + rho_phi)  # Dynamic Hubble
        
        dt = dt_base
        retries = 0
        max_retries = 12
        success = False
        
        while not success and retries < max_retries:
            normal = np.random.normal()
            dW = normal * np.sqrt(dt)
            
            f = (phi**2 / phi0**2)
            b = sigma * f
            db_dphi = sigma * (2 * phi / phi0**2)
            milstein = 0.5 * b * db_dphi * (dW**2 - dt)
            
            rhs_base = -3 * H * dphi - m**2 * phi - beta * phi * dphi**4
            noise_term = b * dW
            effective_rhs = rhs_base + milstein + noise_term
            
            qa = kappa
            qb = 1 + 6 * kappa * H * dphi
            qc = 9 * kappa * (H * dphi)**2 - effective_rhs
            
            Z = quadratic_solve(qa, qb, qc)
            Z = np.clip(Z, -1e5, 1e5)
            
            new_phi = phi + dphi * dt + 0.5 * Z * dt**2
            new_dphi = dphi + Z * dt
            
            new_phi = np.clip(new_phi, -max_phi, max_phi)
            new_dphi = np.clip(new_dphi, -max_phi, max_phi)
            
            phi, dphi = new_phi, new_dphi
            t += dt
            history['t'].append(t)
            history['phi'].append(phi)
            history['dphi'].append(dphi)
            history['H'].append(H)
            success = True
        
        if not success:
            print(f"Max retries exceeded at t={t:.4f}, sigma={sigma}. Falling back to Euler-Maruyama.")
            rhs_fallback = rhs_base + noise_term
            qa_f = kappa
            qb_f = 1 + 6 * kappa * H * dphi
            qc_f = 9 * kappa * (H * dphi)**2 - rhs_fallback
            
            Z = quadratic_solve(qa_f, qb_f, qc_f)
            Z = np.clip(Z, -1e5, 1e5)
            
            new_phi = phi + dphi * dt_base + 0.5 * Z * dt_base**2
            new_dphi = dphi + Z * dt_base
            
            new_phi = np.clip(new_phi, -max_phi, max_phi)
            new_dphi = np.clip(new_dphi, -max_phi, max_phi)
            
            phi, dphi = new_phi, new_dphi
            t += dt_base
            history['t'].append(t)
            history['phi'].append(phi)
            history['dphi'].append(dphi)
            history['H'].append(H)
    
    kinetic = 0.5 * dphi**2
    potential = 0.5 * m**2 * phi**2
    total = kinetic + potential
    w = (kinetic - potential) / total if total > 1e-10 else np.nan
    return w, history

# Tuned parameters (adjust m, phi_init, Omega_m, H0 to freeze better)
tuned_params = {
    'beta': 1.0,
    'kappa': 0.1,
    'phi_init': 4.0,
    'm': 3.0          # Try 0.2 to 0.5 range
}

# Single deterministic test (σ=0)
print("Deterministic test (σ=0):")
w_det, hist_det = single_realization(0.0, tuned_params)
print(f"w(0) deterministic: {w_det:.4f}")

plt.plot(hist_det['t'], hist_det['phi'], label='ϕ(t) deterministic')
plt.xlabel('Time (conformal)')
plt.ylabel('ϕ')
plt.legend()
plt.show()

# Ensemble function
def ensemble_avg(sigma, n_realiz=50000, params=tuned_params):
    ws = []
    for i in range(n_realiz):
        w, _ = single_realization(sigma, params)
        if not np.isnan(w):
            ws.append(w)
        if i % 10000 == 0:
            print(f"Realization {i+1}/{n_realiz} for σ={sigma}")
    if len(ws) > 0:
        return np.mean(ws), np.std(ws)
    return np.nan, np.nan

# Test σ values (increase n_realiz for production)
sigmas = [0.001]
for sig in sigmas:
    mean_w, std_w = ensemble_avg(sig, n_realiz=50000)
    print(f"Sigma {sig:.3f}: mean w(0) = {mean_w:.4f} ± {std_w:.4f}")
