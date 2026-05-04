# DiffuserCam 2026 Study Workspace

This repo is a staged learning and simulation workspace for DiffuserCam-style lensless imaging. The presentation branch focuses on the robust 2D story: forward modeling, inverse-problem instability, Wiener filtering, projected-gradient/FISTA reconstruction, measured Waller-Lab sample data, TV-ADMM, and robustness diagnostics.

The 3D notebooks and data are intentionally treated as future-work references for this branch. They are useful context for the long-term project, but they should not be included as completed implementation in the presentation.

## Start Here

Use notebooks 01-07 as the main 2D learning and presentation path:

```text
notebooks/01_forward_model_wiener.ipynb
notebooks/02_padded_linear_pgd_fista.ipynb
notebooks/03_support_constraints_l2_regularization.ipynb
notebooks/04_waller_lab_sample_data.ipynb
notebooks/05_total_variation_admm.ipynb
notebooks/06_admm_diagnostics_parameter_sweeps.ipynb
notebooks/07_robustness_parameter_tuning_diagnostics.ipynb
```

Each notebook explains the math, runs the simulation, saves figures, and computes quantitative metrics. The reusable implementation lives in:

```text
src/diffusercam_sim/
```

The scripts in `scripts/` are command-line versions of selected milestones for reproducibility. Notebook builder scripts for 08-09 are retained as future-work utilities, not part of the 2D presentation path.

## What Exists

```text
notebooks/                  Main tutorial sequence and presentation narrative
src/diffusercam_sim/         Reusable FFT, forward-model, solver, metric, data-loading, and visualization utilities
scripts/                    Reproducible command-line runs plus notebook builders
docs/                       Short derivation notes and presentation planning material
data/external/tutorial/     Waller-Lab 2D sample PSF and raw hand measurement
data/external/test_images/  Candidate display scenes for later hardware work
data/external/3d/           Future-work Waller 3D sample data
results/                    Regenerated figures, metrics, and notebook outputs
```

## Presentation Scope

For this branch, present the 2D DiffuserCam workflow:

1. Lensless imaging motivation.
2. DiffuserCam idea: a diffuser turns each point source into a calibrated PSF pattern.
3. PSF calibration and measured raw data.
4. 2D shift-invariant forward model: `b = A x + n`.
5. Why direct inversion is unstable.
6. Wiener filtering as the first stabilized inverse.
7. Projected gradient descent and FISTA for constrained least squares.
8. TV regularization and ADMM for sharper piecewise-smooth reconstructions.
9. Robustness experiments: noise, PSF mismatch, shifts, crop errors, saturation, quantization, and parameter tuning.
10. Limitations and next steps: more robust 3D reconstruction and building a Raspberry Pi DiffuserCam.

## Learning Roadmap

### Phase 1: Incoherent 2D Forward Model

- Interpret a lensless camera as a linear system: `y = Hx + n`.
- Use a measured or synthetic point spread function (PSF) as the impulse response of the system.
- Start with the shift-invariant approximation: `y = psf * x + noise`.
- Learn why Fourier-domain direct inversion is unstable when the optical transfer function has small values.
- Implement Wiener deconvolution and evaluate with PSNR, SSIM, residual error, and saved visual outputs.

### Phase 2: Iterative Reconstruction

- Implement projected gradient descent for `0.5 ||Hx - y||_2^2`.
- Add nonnegativity projection because scene intensities cannot be negative.
- Implement FISTA for faster convergence.
- Compare convergence curves and residuals against Wiener deconvolution.

### Phase 3: Robustness Experiments

- Sweep SNR, PSF mismatch, PSF shifts, cropping errors, saturation, quantization, and regularization strength.
- Save metrics as CSV/JSON and generate comparison montages.
- Track when reconstruction quality fails gracefully versus catastrophically.

### Phase 4: Regularization

- Add Tikhonov regularization in the Fourier and iterative views.
- Add total variation regularization and compare against nonnegative FISTA.
- Introduce ADMM after the simpler projected-gradient view is solid.

### Future Work: 3D DiffuserCam Model

- Simulate `y = sum_z H_z x_z + n` using depth-dependent PSFs.
- Reconstruct a small depth volume.
- Evaluate lateral recovery and depth localization.
- Keep this out of the main presentation claims until the implementation is robust.

### Phase 6: Hardware Preparation

- Only after the simulation tools are trustworthy, plan Raspberry Pi 5 and Pi Camera Module 3 calibration, raw capture, PSF measurement, dynamic range, saturation, and mechanical layout.

## Command-Line Simulations

The first runnable milestone is the circular-convolution/Wiener demo:

```powershell
& "C:\Users\Zhen\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" scripts\run_minimal_2d.py --mode both
```

Outputs are written to:

```text
results/minimal_2d/
```

Available modes:

```powershell
# DiffuserCam-style sparse random PSF toy model
& "C:\Users\Zhen\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" scripts\run_minimal_2d.py --mode diffuser

# Professor-reference 5x5 averaging PSF inverse/Wiener demo
& "C:\Users\Zhen\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" scripts\run_minimal_2d.py --mode professor
```

This minimal script uses NumPy and Pillow only. Later notebook figures increasingly use Matplotlib for presentation-quality labels and diagnostic plots.

The second runnable milestone is padded linear convolution with projected GD and FISTA:

```powershell
& "C:\Users\Zhen\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" scripts\run_iterative_2d.py
```

Outputs are written to:

```text
results/iterative_2d/
```

## Notebook Environment

The bundled Python runtime in this Codex workspace can run the simulations, but it does not currently include Jupyter. To use the notebooks interactively, create or use an environment with:

```text
jupyterlab
ipykernel
numpy
pillow
```

The presentation notebooks should prefer Matplotlib for labeled figures and captions. Some older script utilities still use Pillow for lightweight montage generation; those are candidates for a later cleanup pass.

## Theory Notes for Step 1

For a spatially incoherent scene, intensities add. If every scene point creates a shifted copy of the same sensor pattern, then the image formation is approximately shift-invariant:

```text
y = h * x + n
```

where `h` is the PSF, `x` is the unknown scene, and `n` is measurement noise. With circular boundary assumptions, the convolution matrix is diagonalized by the discrete Fourier transform:

```text
Y = H X + N
```

Direct inversion uses `X = Y / H`, but this explodes wherever `|H|` is small. Wiener deconvolution stabilizes the inverse:

```text
X_hat = conj(H) Y / (|H|^2 + lambda)
```

The regularization parameter `lambda` trades sharpness for noise suppression.

## Primary References

- DiffuserCam project: https://waller-lab.github.io/DiffuserCam/
- DiffuserCam tutorial: https://waller-lab.github.io/DiffuserCam/tutorial.html
- Gradient descent / FISTA tutorial: https://waller-lab.github.io/DiffuserCam/tutorial/GD.html
- ADMM tutorial: https://waller-lab.github.io/DiffuserCam/tutorial/ADMM.html
- Recent advances in lensless imaging: https://doi.org/10.1364/OPTICA.431361
- LenslessPiCam docs: https://lensless.readthedocs.io/en/latest/index.html
