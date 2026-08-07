# DiffuserCam 2026

An educational study of 2D DiffuserCam-style lensless imaging and inverse reconstruction.

The project covers forward simulation, Wiener deconvolution, projected-gradient methods, FISTA, total-variation regularization, ADMM, and robustness testing using synthetic measurements and Waller Lab sample data.

## Attribution

This is an independent educational reimplementation and is not an official Waller Lab project.

The project is based on the DiffuserCam research, tutorials, and sample data published by Nick Antipa, Grace Kuo, Laura Waller, and collaborators.

The files under:

```text
data/external/tutorial/
```

contain a sample point-spread function and raw sensor measurement from the Waller Lab DiffuserCam tutorial. Credit and licensing information for third-party materials should be retained with those files.


The current implementatio uses a simplified 2D shift-invariant model:

'''text
b = A x + n
'''

where:

- `x` is the scene
- `A` is convolution with a calibrated or simulated point-spread function
- `b` is the sensor measurement
- `n` is measurement noise

- ## Notebooks

The main study is organized into seven notebooks:

```text
notebooks/
  01_forward_model_wiener.ipynb
  02_padded_linear_pgd_fista.ipynb
  03_support_constraints_l2_regularization.ipynb
  04_waller_lab_sample_data.ipynb
  05_total_variation_admm.ipynb
  06_admm_diagnostics_parameter_sweeps.ipynb
  07_robustness_parameter_tuning_diagnostics.ipynb
```

The notebooks introduce the forward model and reconstruction methods in sequence. Generated figures and metrics are written to:

```text
results/notebooks/
```

## Repository Structure

```text
notebooks/                  Main experiments
src/diffusercam_sim/        Forward models, solvers, metrics, and utilities
data/external/tutorial/     Waller Lab tutorial data
results/                    Generated figures and measurements
```

## Setup

Create a Python environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install jupyterlab ipykernel numpy scipy pillow matplotlib
```

Start JupyterLab:

```bash
jupyter lab
```

Begin with:

```text
notebooks/01_forward_model_wiener.ipynb
```

## Limitations

The current work assumes a 2D shift-invariant forward model. A physical DiffuserCam may exhibit spatially varying PSFs, calibration errors, sensor noise, and depth-dependent measurements that are not represented by this simplified model.

The measured-data experiment is intended to study reconstruction behavior rather than reproduce the complete Waller Lab 3D pipeline.

## Future Work

- extend the model to depth-dependent 3D reconstruction
- evaluate additional regularization methods
- improve reconstruction under calibration mismatch
- build and calibrate a Raspberry Pi DiffuserCam prototype

## References

- [DiffuserCam project](https://waller-lab.github.io/DiffuserCam/)
- [DiffuserCam tutorial](https://waller-lab.github.io/DiffuserCam/tutorial.html)
- [Gradient descent and FISTA tutorial](https://waller-lab.github.io/DiffuserCam/tutorial/GD.html)
- [ADMM tutorial](https://waller-lab.github.io/DiffuserCam/tutorial/ADMM.html)
- [Recent advances in lensless imaging](https://doi.org/10.1364/OPTICA.431361)
- [LenslessPiCam documentation](https://lensless.readthedocs.io/en/latest/)

### DiffuserCam Paper

```bibtex
@article{antipa2018diffusercam,
  title   = {DiffuserCam: Lensless Single-Exposure 3D Imaging},
  author  = {Antipa, Nick and Kuo, Grace and Heckel, Reinhard and
             Mildenhall, Ben and Bostan, Emrah and Ng, Ren and Waller, Laura},
  journal = {Optica},
  volume  = {5},
  number  = {1},
  pages   = {1--9},
  year    = {2018}
}
```
