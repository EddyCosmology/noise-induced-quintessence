import numpy as np
import matplotlib.pyplot as plt

# Data from your 10k run
sigmas = np.array([0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
mean_w = np.array([-0.9992, -0.9900, -0.9566, -0.8731, -0.6697, -0.2323, 0.3453, 0.6719, 0.6451, 0.4608, 0.2798])
std_w = np.array([0.0000, 0.0067, 0.0280, 0.0735, 0.1814, 0.3963, 0.4866, 0.3492, 0.3140, 0.3529, 0.3478])

fig, ax = plt.subplots(figsize=(10, 6))

# Main plot
ax.errorbar(sigmas, mean_w, yerr=std_w, fmt='o-', capsize=5, 
            color='darkblue', linewidth=2, markersize=8, 
            label='Ensemble mean w(0) ± 1σ (n=10,000 per point)')

# Reference lines
ax.axhline(-1.0, color='black', linestyle='--', linewidth=1.2, label='Frozen (w = −1)')
ax.axhline(-0.86, color='green', linestyle='-', linewidth=1.5, label='DESI-like local value (w ≈ −0.86)')
ax.axhline(-1/3, color='red', linestyle='-.', linewidth=1.5, label='No acceleration (w = −1/3)')

# Viable window shading
ax.axvspan(0.001, 0.045, alpha=0.12, color='green', label='Viable acceleration window')

# Threshold marker
ax.axvline(0.045, color='orange', linestyle=':', linewidth=1.5, 
           label='Approx. σ_c ≈ 0.045–0.05 (this tuning)')

# Axis and title
ax.set_xscale('log')
ax.set_xlabel('Noise amplitude σ', fontsize=14)
ax.set_ylabel('Effective equation of state w(0)', fontsize=14)
ax.set_title('Noise-Induced Transition in Quintessence-Like Dark Energy\n(Deterministic w(0) ≈ −0.9992)', fontsize=15)
ax.grid(True, which="both", ls="--", alpha=0.5)
ax.legend(fontsize=11, loc='upper left')
ax.set_xlim(0.0008, 0.15)
ax.set_ylim(-1.05, 0.8)

# Annotations
ax.annotate('Frozen near −1', xy=(0.001, -1.0), xytext=(0.002, -0.75),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5), fontsize=11)
ax.annotate('w ≈ −0.86 (DESI-like)', xy=(0.03, -0.86), xytext=(0.005, -0.92),
            arrowprops=dict(facecolor='green', shrink=0.05, width=1.5), fontsize=11)
ax.annotate('Crosses −1/3', xy=(0.045, -0.333), xytext=(0.01, -0.5),
            arrowprops=dict(facecolor='red', shrink=0.05, width=1.5), fontsize=11)

plt.tight_layout()
plt.savefig('thornton_noise_threshold_10k.png', dpi=300, bbox_inches='tight')
plt.show()
