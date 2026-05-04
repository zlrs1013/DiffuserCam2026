"""Build Notebook 09 for Waller-Lab DiffuserCam 3D real-data reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("notebooks/09_waller_lab_3d_real_data.ipynb")


def md(text: str) -> dict:
    text = dedent(text).strip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.splitlines()]}


def code(text: str) -> dict:
    text = dedent(text).strip("\n")
    lines = [line + "\n" for line in text.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


cells = [
    md(
        r"""
        # 09 - Waller-Lab DiffuserCam 3D Real Data

        Notebook 08 built a synthetic 3D forward model:

        $$
        b = \sum_z A_z x_z + n
        $$

        This notebook applies the same idea to the original Waller-Lab DiffuserCam example data:

        - `example_psfs.mat`: measured depth-dependent PSF stack,
        - `example_raw.png`: raw sensor measurement,
        - `DiffuserCam_settings.m`: original MATLAB preprocessing/solver settings.

        This is a first Python reconstruction notebook. The goal is not to exactly reproduce the Waller MATLAB ADMM solver yet. The goal is to:

        1. load and inspect the real 3D calibration data,
        2. preprocess PSF and measurement using the settings file,
        3. build a multi-depth forward/adjoint model,
        4. run a CPU-friendly projected FISTA reconstruction,
        5. diagnose residuals, depth energy, and reconstructed depth slices.
        """
    ),
    md(
        r"""
        ## Important Practical Note

        The full Waller example has shape:

        ```text
        PSF stack: 270 x 320 x 44
        Raw image: 270 x 320
        ```

        Reconstructing all 44 planes at full resolution is heavier than our earlier notebooks. For a first CPU notebook, we use:

        ```text
        lateral_downsample = 2
        axial_downsample = 4
        ```

        This creates an 11-plane volume at `135 x 160`. Once the pipeline is verified, you can increase resolution or reduce axial downsampling. This is the same engineering pattern as hardware work: start with a small trusted pipeline, then scale it.
        """
    ),
    md("## Setup"),
    code(
        r"""
        from pathlib import Path
        import sys
        import json
        from dataclasses import dataclass
        from time import perf_counter

        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        from scipy.io import loadmat


        def find_project_root(start: Path) -> Path:
            for candidate in [start, *start.parents]:
                if (candidate / 'src' / 'diffusercam_sim').exists():
                    return candidate
            raise RuntimeError('Could not find project root containing src/diffusercam_sim')


        PROJECT_ROOT = find_project_root(Path.cwd().resolve())
        SRC_DIR = PROJECT_ROOT / 'src'
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        from diffusercam_sim import PaddedLinearConvolution, residual_relative_l2

        DATA_DIR = PROJECT_ROOT / 'data' / 'external' / '3d'
        PSF_PATH = DATA_DIR / 'example_psfs.mat'
        RAW_PATH = DATA_DIR / 'example_raw.png'
        SETTINGS_PATH = DATA_DIR / 'DiffuserCam_settings.m'

        RESULT_DIR = PROJECT_ROOT / 'results' / 'notebooks' / '09_waller_lab_3d_real_data'
        RESULT_DIR.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update(
            {
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.edgecolor': '#263238',
                'axes.labelcolor': '#263238',
                'axes.titleweight': 'bold',
                'axes.titlesize': 11,
                'axes.labelsize': 10,
                'xtick.color': '#263238',
                'ytick.color': '#263238',
                'grid.color': '#d9e0e3',
                'grid.linewidth': 0.8,
                'legend.frameon': True,
                'legend.framealpha': 0.95,
                'legend.facecolor': 'white',
                'font.size': 10,
                'figure.max_open_warning': 80,
            }
        )

        DATA_DIR
        """
    ),
    code(
        r"""
        def save_figure(fig, path: Path):
            fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
            return fig


        def display_scale(image: np.ndarray, percentiles=(1, 99.7)) -> np.ndarray:
            image = np.asarray(image, dtype=np.float64)
            low, high = np.percentile(image, percentiles)
            if high <= low:
                return np.zeros_like(image)
            return np.clip((image - low) / (high - low), 0.0, 1.0)


        def plot_image_grid(items, path: Path, columns: int = 3, title: str | None = None):
            rows = int(np.ceil(len(items) / columns))
            fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.4 * rows), constrained_layout=True)
            axes = np.atleast_1d(axes).ravel()
            for ax, (label, image) in zip(axes, items):
                ax.imshow(display_scale(image), cmap='gray', vmin=0, vmax=1)
                ax.set_title(label)
                ax.set_xticks([])
                ax.set_yticks([])
            for ax in axes[len(items):]:
                ax.axis('off')
            if title is not None:
                fig.suptitle(title, fontsize=15, fontweight='bold')
            return save_figure(fig, path)


        def add_panel_note(ax, text: str, y: float = 0.05):
            ax.text(
                0.03,
                y,
                text,
                transform=ax.transAxes,
                fontsize=8.7,
                va='bottom',
                bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'edgecolor': '#cfd8dc', 'alpha': 0.92},
            )


        def normalize_l2(image: np.ndarray) -> np.ndarray:
            image = np.asarray(image, dtype=np.float64)
            return image / max(float(np.linalg.norm(image.ravel())), 1e-12)


        def normalize_sum(image: np.ndarray) -> np.ndarray:
            image = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
            return image / max(float(np.sum(image)), 1e-12)


        def box_downsample(image: np.ndarray, factor: int) -> np.ndarray:
            if factor < 1 or int(factor) != factor:
                raise ValueError('factor must be a positive integer')
            factor = int(factor)
            result = np.asarray(image, dtype=np.float64)
            if factor == 1:
                return result
            rows = result.shape[0] - result.shape[0] % factor
            cols = result.shape[1] - result.shape[1] % factor
            result = result[:rows, :cols]
            new_shape = (rows // factor, factor, cols // factor, factor)
            return result.reshape(new_shape).mean(axis=(1, 3))


        def axial_average(psf_stack: np.ndarray, factor: int) -> np.ndarray:
            if factor < 1 or int(factor) != factor:
                raise ValueError('factor must be a positive integer')
            factor = int(factor)
            if factor == 1:
                return psf_stack
            depth = psf_stack.shape[2] - psf_stack.shape[2] % factor
            trimmed = psf_stack[:, :, :depth]
            rows, cols, _ = trimmed.shape
            return trimmed.reshape(rows, cols, depth // factor, factor).mean(axis=3)
        """
    ),
    md(
        r"""
        ## Load and Inspect the Real Data

        The MATLAB settings file says:

        ```text
        impulse_var_name = 'psf'
        image_bias = 100
        psf_bias = 102
        lateral_downsample = 1
        axial_downsample = 1
        start_z = 1
        end_z = 0
        ```

        We will respect the bias correction and variable name, but use stronger downsampling for this first CPU-friendly notebook.
        """
    ),
    code(
        r"""
        settings_text = SETTINGS_PATH.read_text(encoding='utf-8', errors='replace')
        mat_data = loadmat(PSF_PATH)
        raw_psf_stack = np.asarray(mat_data['psf'], dtype=np.float64)
        raw_measurement = np.asarray(Image.open(RAW_PATH), dtype=np.float64)

        raw_info = {
            'psf_shape_rows_cols_depth': raw_psf_stack.shape,
            'psf_dtype': str(raw_psf_stack.dtype),
            'raw_measurement_shape': raw_measurement.shape,
            'raw_measurement_min': float(raw_measurement.min()),
            'raw_measurement_max': float(raw_measurement.max()),
            'raw_measurement_mean': float(raw_measurement.mean()),
            'settings_file_present': SETTINGS_PATH.exists(),
        }

        raw_info
        """
    ),
    code(
        r"""
        representative_depths = [0, raw_psf_stack.shape[2] // 2, raw_psf_stack.shape[2] - 1]
        plot_image_grid(
            [(f'Raw PSF depth {depth + 1}', raw_psf_stack[:, :, depth]) for depth in representative_depths]
            + [('Raw sensor measurement', raw_measurement)],
            RESULT_DIR / '01_raw_inputs.png',
            columns=4,
            title='Raw Waller-Lab DiffuserCam 3D example data',
        )
        """
    ),
    code(
        r"""
        def plot_raw_dynamic_range(raw_image: np.ndarray, path: Path):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
            fig.suptitle('Raw measurement dynamic range', fontsize=15, fontweight='bold')
            axes[0].hist(raw_image.ravel(), bins=140, color='#1b7f79')
            axes[0].axvline(65535, color='#b24c63', linewidth=2, label='16-bit maximum')
            axes[0].set_title('1. Raw value histogram')
            axes[0].set_xlabel('Raw pixel value')
            axes[0].set_ylabel('Pixel count')
            axes[0].legend(loc='best')
            saturated_fraction = float(np.mean(raw_image >= 65535))
            add_panel_note(axes[0], f'saturated pixels: {100 * saturated_fraction:.2f}%')

            axes[1].imshow(raw_image >= 65535, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('2. Saturation mask')
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            for ax in axes:
                ax.grid(True, which='major')
                ax.spines[['top', 'right']].set_visible(False)
            return save_figure(fig, path)


        plot_raw_dynamic_range(raw_measurement, RESULT_DIR / '02_raw_dynamic_range.png')
        """
    ),
    md(
        r"""
        ## Preprocessing

        We apply the original bias corrections:

        $$
        \text{PSF}_{\text{corrected}} = \max(\text{PSF}_{\text{raw}} - 102, 0)
        $$

        $$
        b_{\text{corrected}} = \max(b_{\text{raw}} - 100, 0)
        $$

        Then we downsample and normalize:

        - each PSF plane is sum-normalized so every depth has comparable flux,
        - the measurement is max-normalized for numerical scale.

        This is not yet an exact copy of the Waller MATLAB ADMM pipeline. It is a clean Python baseline for inspecting and reconstructing the real 3D sample.
        """
    ),
    code(
        r"""
        image_bias = 100.0
        psf_bias = 102.0

        lateral_downsample = 2
        axial_downsample_factor = 4

        psf_corrected = np.maximum(raw_psf_stack - psf_bias, 0.0)
        measurement_corrected = np.maximum(raw_measurement - image_bias, 0.0)

        psf_downsampled = np.stack(
            [box_downsample(psf_corrected[:, :, depth], lateral_downsample) for depth in range(psf_corrected.shape[2])],
            axis=2,
        )
        measurement_downsampled = box_downsample(measurement_corrected, lateral_downsample)
        psf_axial = axial_average(psf_downsampled, axial_downsample_factor)

        psf_stack = np.stack([normalize_sum(psf_axial[:, :, depth]) for depth in range(psf_axial.shape[2])], axis=2)
        measurement = measurement_downsampled / max(float(np.max(measurement_downsampled)), 1e-12)

        depth_count = psf_stack.shape[2]
        depth_labels = [f'z{index + 1}' for index in range(depth_count)]

        preprocess_info = {
            'lateral_downsample': lateral_downsample,
            'axial_downsample_factor': axial_downsample_factor,
            'processed_psf_shape': psf_stack.shape,
            'processed_measurement_shape': measurement.shape,
            'depth_count': depth_count,
            'measurement_max_after_normalization': float(measurement.max()),
            'psf_plane_sums_min_max': [float(np.min(np.sum(psf_stack, axis=(0, 1)))), float(np.max(np.sum(psf_stack, axis=(0, 1))))],
        }

        preprocess_info
        """
    ),
    code(
        r"""
        depth_preview = sorted(set([0, depth_count // 4, depth_count // 2, 3 * depth_count // 4, depth_count - 1]))
        plot_image_grid(
            [(f'Processed PSF {depth_labels[depth]}', psf_stack[:, :, depth]) for depth in depth_preview]
            + [('Processed measurement', measurement)],
            RESULT_DIR / '03_processed_inputs.png',
            columns=3,
            title='Bias-corrected, downsampled, normalized data',
        )
        """
    ),
    md(
        r"""
        ## Build the 3D Forward and Adjoint Model

        The real-data model is:

        $$
        b = \sum_{z=1}^{Z} A_z x_z + n
        $$

        Each \(A_z\) is the padded linear-convolution operator built from PSF plane \(h_z\).

        The gradient of the least-squares data term is:

        $$
        \nabla_{x_z}
        \frac{1}{2}
        \left\|
        \sum_k A_k x_k - b
        \right\|_2^2
        =
        A_z^T
        \left(
        \sum_k A_k x_k - b
        \right)
        $$

        We use this for projected FISTA with nonnegativity and optional L2 regularization.
        """
    ),
    code(
        r"""
        operators = [PaddedLinearConvolution(psf_stack[:, :, depth]) for depth in range(depth_count)]


        def forward_volume(operators: list[PaddedLinearConvolution], volume: np.ndarray) -> np.ndarray:
            prediction = np.zeros(operators[0].image_shape, dtype=np.float64)
            for operator, plane in zip(operators, volume):
                prediction += operator.forward(plane)
            return prediction


        def adjoint_volume(operators: list[PaddedLinearConvolution], residual: np.ndarray) -> np.ndarray:
            gradients = []
            for operator in operators:
                gradients.append(operator.crop(operator.adjoint_padded(residual)))
            return np.stack(gradients, axis=0)


        def adjoint_inner_product_error_3d(operators: list[PaddedLinearConvolution], seed: int = 0) -> float:
            rng = np.random.default_rng(seed)
            x = rng.standard_normal((len(operators), *operators[0].image_shape))
            y = rng.standard_normal(operators[0].image_shape)
            lhs = float(np.vdot(forward_volume(operators, x), y).real)
            rhs = float(np.vdot(x, adjoint_volume(operators, y)).real)
            return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)


        operator_info = {
            'depth_count': depth_count,
            'image_shape': operators[0].image_shape,
            'padded_shape': operators[0].padded_shape,
            'adjoint_inner_product_error_3d': adjoint_inner_product_error_3d(operators),
        }

        operator_info
        """
    ),
    md(
        r"""
        ## Reconstruct with Projected FISTA

        We solve a simple nonnegative L2-regularized problem:

        $$
        \min_{x_z \ge 0}
        \frac{1}{2}
        \left\|
        \sum_z A_z x_z - b
        \right\|_2^2
        +
        \frac{\lambda}{2}
        \sum_z \|x_z\|_2^2
        $$

        This is not the full Waller 3D-TV ADMM solver yet. Think of it as a baseline reconstruction that verifies our data loading, forward model, adjoint, and diagnostics.
        """
    ),
    code(
        r"""
        @dataclass
        class VolumeRecord:
            iteration: int
            objective: float
            relative_residual_l2: float
            volume_l2: float
            max_value: float


        @dataclass
        class VolumeResult:
            volume: np.ndarray
            history: list[VolumeRecord]
            prediction: np.ndarray
            step_size: float
            seconds: float


        def volume_objective(operators, volume, measurement, l2_regularization):
            residual = forward_volume(operators, volume) - measurement
            data_fit = 0.5 * float(np.sum(residual * residual))
            l2_term = 0.5 * l2_regularization * float(np.sum(volume * volume))
            return data_fit + l2_term, residual


        def reconstruct_volume_fista(
            operators,
            measurement,
            iterations: int = 120,
            l2_regularization: float = 2e-3,
            record_every: int = 5,
        ) -> VolumeResult:
            start = perf_counter()
            volume = np.zeros((len(operators), *operators[0].image_shape), dtype=np.float64)
            momentum_volume = volume.copy()
            t_value = 1.0
            lipschitz_bound = sum(operator.lipschitz_bound for operator in operators) + l2_regularization
            step_size = 1.0 / max(lipschitz_bound, 1e-12)
            history = []

            for iteration in range(iterations + 1):
                if iteration % record_every == 0 or iteration == iterations:
                    objective, residual = volume_objective(operators, volume, measurement, l2_regularization)
                    history.append(
                        VolumeRecord(
                            iteration=iteration,
                            objective=objective,
                            relative_residual_l2=residual_relative_l2(measurement, measurement + residual),
                            volume_l2=float(np.linalg.norm(volume.ravel())),
                            max_value=float(np.max(volume)),
                        )
                    )

                if iteration == iterations:
                    break

                residual = forward_volume(operators, momentum_volume) - measurement
                gradient = adjoint_volume(operators, residual) + l2_regularization * momentum_volume
                previous_volume = volume
                volume = np.maximum(momentum_volume - step_size * gradient, 0.0)
                next_t_value = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_value * t_value))
                momentum = (t_value - 1.0) / next_t_value
                momentum_volume = volume + momentum * (volume - previous_volume)
                t_value = next_t_value

            prediction = forward_volume(operators, volume)
            seconds = perf_counter() - start
            return VolumeResult(volume=volume, history=history, prediction=prediction, step_size=step_size, seconds=seconds)


        reconstruction = reconstruct_volume_fista(
            operators,
            measurement,
            iterations=140,
            l2_regularization=2e-3,
            record_every=5,
        )

        reconstruction_summary = {
            'seconds': reconstruction.seconds,
            'step_size': reconstruction.step_size,
            'final_record': reconstruction.history[-1].__dict__,
        }

        reconstruction_summary
        """
    ),
    code(
        r"""
        def plot_convergence(history: list[VolumeRecord], path: Path):
            iterations = np.array([record.iteration for record in history])
            objective = np.array([record.objective for record in history])
            residual = np.array([record.relative_residual_l2 for record in history])
            volume_l2 = np.array([record.volume_l2 for record in history])
            max_value = np.array([record.max_value for record in history])

            fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), constrained_layout=True)
            fig.suptitle('Projected FISTA convergence on Waller 3D real data', fontsize=15, fontweight='bold')

            axes[0].plot(iterations, objective, marker='o', linewidth=2.2)
            axes[0].set_yscale('log')
            axes[0].set_title('1. Objective')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Objective, log scale')

            axes[1].plot(iterations, residual, marker='o', linewidth=2.2, color='#1b7f79')
            axes[1].set_yscale('log')
            axes[1].set_title('2. Sensor residual')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Relative residual, log scale')

            axes[2].plot(iterations, volume_l2, marker='o', linewidth=2.2, color='#b24c63')
            axes[2].set_title('3. Volume L2 norm')
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('||x||_2')

            axes[3].plot(iterations, max_value, marker='o', linewidth=2.2, color='#d17a22')
            axes[3].set_title('4. Max voxel value')
            axes[3].set_xlabel('Iteration')
            axes[3].set_ylabel('max(x)')

            for ax in axes:
                ax.grid(True, which='major')
                ax.spines[['top', 'right']].set_visible(False)
            return save_figure(fig, path)


        plot_convergence(reconstruction.history, RESULT_DIR / '04_reconstruction_convergence.png')
        """
    ),
    md(
        r"""
        ## Visualize the Reconstructed Volume

        With no ground truth, we inspect:

        - depth slices,
        - maximum-intensity projection,
        - depth winner map,
        - energy per depth.

        The depth winner map labels each lateral pixel by the depth plane with the largest reconstructed intensity.
        """
    ),
    code(
        r"""
        volume = reconstruction.volume
        depth_energy = np.sum(volume, axis=(1, 2))
        depth_max = np.max(volume, axis=(1, 2))
        mip = np.max(volume, axis=0)
        sum_projection = np.sum(volume, axis=0)
        depth_winner = np.argmax(volume, axis=0)
        depth_confidence = np.max(volume, axis=0) / np.maximum(np.sum(volume, axis=0), 1e-12)

        slice_indices = sorted(set([0, depth_count // 5, 2 * depth_count // 5, 3 * depth_count // 5, 4 * depth_count // 5, depth_count - 1]))
        plot_image_grid(
            [(f'Depth slice {index + 1}', volume[index]) for index in slice_indices]
            + [('Max intensity projection', mip), ('Sum projection', sum_projection)],
            RESULT_DIR / '05_reconstructed_depth_slices.png',
            columns=4,
            title='Waller 3D real-data reconstruction slices',
        )
        """
    ),
    code(
        r"""
        def plot_depth_diagnostics(depth_energy, depth_max, depth_winner, depth_confidence, path: Path):
            fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), constrained_layout=True)
            fig.suptitle('Depth diagnostics for Waller 3D real-data reconstruction', fontsize=15, fontweight='bold')

            x = np.arange(1, len(depth_energy) + 1)
            axes[0, 0].bar(x, depth_energy, color='#5d6fb3')
            axes[0, 0].set_title('1. Reconstructed energy per depth')
            axes[0, 0].set_xlabel('Reconstructed depth index')
            axes[0, 0].set_ylabel('sum of voxel intensities')

            axes[0, 1].bar(x, depth_max, color='#b24c63')
            axes[0, 1].set_title('2. Maximum voxel value per depth')
            axes[0, 1].set_xlabel('Reconstructed depth index')
            axes[0, 1].set_ylabel('max voxel value')

            im0 = axes[1, 0].imshow(depth_winner + 1, cmap='turbo', vmin=1, vmax=len(depth_energy))
            axes[1, 0].set_title('3. Depth winner map')
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])
            fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04, label='depth index')

            im1 = axes[1, 1].imshow(depth_confidence, cmap='magma', vmin=0, vmax=1)
            axes[1, 1].set_title('4. Depth confidence max/sum')
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
            fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04, label='confidence')

            for ax in axes[:1].ravel():
                ax.grid(True, axis='y')
                ax.spines[['top', 'right']].set_visible(False)
            return save_figure(fig, path)


        plot_depth_diagnostics(
            depth_energy,
            depth_max,
            depth_winner,
            depth_confidence,
            RESULT_DIR / '06_depth_diagnostics.png',
        )
        """
    ),
    md(
        r"""
        ### Depth-Diagnostic Observation

        The energy curve is a diagnostic, not a final depth estimate. In this baseline, reconstructed energy may drift toward later depth planes because:

        - every PSF plane was independently sum-normalized,
        - the solver uses only L2 regularization, not 3D TV or depth sparsity,
        - the measurement contains saturated or near-saturated bright regions,
        - the original Waller solver uses ADMM with autotuned penalties, while this notebook uses a simpler projected FISTA baseline.

        So the current volume is useful for validating the real-data pipeline and seeing depth-dependent structure, but the depth labels should not yet be interpreted as calibrated physical depths.
        """
    ),
    md(
        r"""
        ## Forward-Model Residual Diagnostics

        We compare the measured sensor image \(b\), the predicted sensor image \(\hat{b} = A\hat{x}\), and the residual:

        $$
        r = A\hat{x} - b
        $$

        Structured residuals are expected in a first baseline reconstruction. They can come from saturation, preprocessing mismatch, model mismatch, missing 3D TV, or insufficient iterations.
        """
    ),
    code(
        r"""
        residual = reconstruction.prediction - measurement

        plot_image_grid(
            [
                ('Measured sensor b', measurement),
                ('Predicted sensor A x_hat', reconstruction.prediction),
                ('Residual magnitude |A x_hat - b|', np.abs(residual)),
                ('Reconstruction sum projection', sum_projection),
            ],
            RESULT_DIR / '07_prediction_residual.png',
            columns=2,
            title='Forward-model residual check',
        )
        """
    ),
    code(
        r"""
        def plot_residual_histogram_and_fft(measurement, residual, path: Path):
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
            fig.suptitle('Residual statistics and Fourier structure', fontsize=15, fontweight='bold')

            axes[0].hist(residual.ravel(), bins=100, color='#5d6fb3')
            axes[0].set_title('1. Residual histogram')
            axes[0].set_xlabel('A x_hat - b')
            axes[0].set_ylabel('Pixel count')
            axes[0].ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))

            residual_fft = np.fft.fftshift(np.fft.fft2(residual))
            measurement_fft = np.fft.fftshift(np.fft.fft2(measurement))
            ratio = np.log10(np.abs(residual_fft) + 1e-12) - np.log10(np.abs(measurement_fft) + 1e-12)
            im = axes[1].imshow(ratio, cmap='magma')
            axes[1].set_title('2. Fourier residual ratio')
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='log residual FFT - log measurement FFT')

            for ax in axes:
                ax.grid(True, which='major')
                ax.spines[['top', 'right']].set_visible(False)
            return save_figure(fig, path)


        plot_residual_histogram_and_fft(measurement, residual, RESULT_DIR / '08_residual_histogram_fft.png')
        """
    ),
    md(
        r"""
        ## What This First Real-Data Notebook Tells Us

        This notebook verifies the Python pipeline against the Waller-Lab real 3D sample:

        - MATLAB `.mat` PSF stack loading works.
        - Bias correction and downsampling are explicit.
        - The 3D forward and adjoint operators pass an inner-product check.
        - A simple projected FISTA solver produces a volume and sensor prediction.
        - We can inspect depth energy, depth winner maps, and residual structure.

        The next step is to get closer to the original Waller reconstruction:

        1. Compare against full-resolution or less-downsampled reconstruction.
        2. Add 3D TV regularization instead of only L2.
        3. Add ADMM for the real 3D data.
        4. Investigate the saturated raw measurement pixels and whether masking/clipping helps.
        5. Compare depth slices against the Waller MATLAB output if we generate it.
        """
    ),
    code(
        r"""
        summary = {
            'raw_info': raw_info,
            'preprocess_info': preprocess_info,
            'operator_info': operator_info,
            'reconstruction_summary': reconstruction_summary,
            'depth_energy': depth_energy.tolist(),
            'depth_max': depth_max.tolist(),
            'relative_residual_l2': residual_relative_l2(measurement, reconstruction.prediction),
        }

        (RESULT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(NOTEBOOK_PATH)
