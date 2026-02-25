# Noise-Induced Transitions in Quintessence-Like Dark Energy  
Sensitivity to Stochastic Vacuum Fluctuations

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the code, data, and figures for the paper:

**"Noise-Induced Transitions in Quintessence-Like Dark Energy: Sensitivity to Stochastic Vacuum Fluctuations"**  
Author: Micah David Thornton  
Draft version: February 2026  

Companion to: Thornton (2026a), "A Phenomenological Model for Evolving Dark Energy Inspired by DESI DR2" (https://eddycosmology/eddy-de-collapse)

## Overview

This work presents large-scale numerical ensembles (50,000 realizations per noise strength σ) exploring how multiplicative stochastic noise affects late-time dark energy behavior in a scalar field model with nonlinear advection, hyperdiffusion, and running vacuum.  

Key findings:
We uncover a previously unidentified critical noise threshold-hereafter the **Thornton Noise Threshold (σ_c ≈ 0.045-0.05 in tuned parameters)** that constrains vacuum fluctuation amplitudes cosmologically. (Updated σ_c value based on new Milstein (n=10k) analysis, was σ_c ≈ 0.06 and will be updated and noted to reflect stronger statistical analysis in both papers for transparency.)
- At low noise (σ ≲ 0.02), the field remains frozen near w ≈ -1 (Λ-like). 
- At moderate noise (σ ≈ 0.05), w(0) shifts to \~ -0.85 (quintessence-like, consistent with DESI DR2 hints).
- At higher noise (σ ≳ 0.1), mean w(0) becomes positive → loss of acceleration.
- Parameter tuning (reduced β, increased κ) extends the viable acceleration window.

# IMPORTANT AUTHOR'S NOTE / DISCLAIMER  
**(Last updated: February 25, 2026)**

**Previous draft versions of both preprints are now obsolete with respect to certain quantitative claims.**

The documents currently in this repository (dated February 20, 2026) represent **early drafts**. Since their initial upload, I have continued numerical work with larger ensembles (up to 10,000 realizations per σ), refined parameter tuning (e.g., m=3, ϕ₀=0.01, Ω_m=0.5), and improved integration stability (Milstein scheme). These updates have slightly shifted some numerical results:

- The critical noise threshold (now named the **Thornton Noise Threshold**) is found at σ_c ≈ **0.045–0.05** in the current fiducial tuning, rather than the approximate σ_c ≈ 0.06 stated in the February 20 drafts.
- The viable acceleration window (where w(0) ≲ −0.85) extends to σ ≈ 0.045–0.05, not 0.05 exactly as previously estimated.
- Moderate-noise w(0) is ≈ −0.87 in this tuning (still consistent with DESI DR2 hints of ≈ −0.86 locally).

**All quantitative claims about σ_c, the exact location of the transition, and the boundaries of the viable window in the February 20 drafts should be considered superseded by the results presented in the most recent PDF versions (look for files dated February 24–25, 2026 or later).**

The qualitative conclusions remain unchanged and robust:
- A sharp noise-induced transition exists where multiplicative stochastic noise destabilizes the frozen quintessence-like attractor.
- Mean w(0) shifts from near −1 at low σ → ≈ −0.87 at moderate σ → crosses −1/3 and becomes positive at higher σ.
- The model provides a mechanism consistent with DESI DR2 hints of evolving dark energy, with testable predictions (f_NL ∼10–50, suppressed σ₈, etc.).

Updated PDFs, figures, and code reflecting the latest results are in the repository. Future arXiv submission will use the most recent version.

Thank you for your understanding — this is an active, iterative project. Feedback is welcome!

### Status
Exploratory independent research project by Micah David Thornton (@EddyCosmology). 

Feedback, comments, or collaboration welcome via Issues or X. 
If using/citing: Please reference the draft and this repo. Thank you and star if you like my work.

**Keywords**: cosmology, dark energy, DESI DR2, evolving DE, stochastic collapse, vacuum suppression, objective collapse models, quintessence

### License Information
The preprint manuscript (PDF and LaTeX content) is additionally liscensed under Creative Commons Attribution 4.0 International (CC-BY-4.0): https://creativecommons.org/licenses/by/4.0/. Code and figures remain licensed under the MIT License (see LICENSE file).
