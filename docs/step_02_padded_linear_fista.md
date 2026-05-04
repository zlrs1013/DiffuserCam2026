# Step 2: Padded Linear Convolution, PGD, and FISTA

## Why Move Beyond Circular Convolution?

The first simulation used same-size circular convolution because it made Fourier inversion easy. Real sensors do not wrap around. Light that would fall outside the finite sensor is cropped away, so a better model is:

```text
b = A x + n
```

where `A` means:

```text
pad scene -> convolve with padded PSF by FFT -> crop sensor-sized region
```

This is the convention used in the Waller Lab DiffuserCam tutorial.

## Operator Convention

Real-space arrays are stored with their origin at the center. Before an FFT, the origin is moved to the top-left corner using `ifftshift`; after an inverse FFT, `fftshift` returns the origin to the center.

The forward model is:

```text
A x = crop( F^-1( F(h_pad) * F(x_pad) ) )
```

The adjoint is:

```text
A^H r = F^-1( conj(F(h_pad)) * F(pad(r)) )
```

The gradient of the least-squares data term is:

```text
grad f(x) = A^H(Ax - b)
```

## Projected Gradient Descent

We solve:

```text
minimize 0.5 ||A x - b||_2^2
subject to x >= 0
```

The projected gradient update is:

```text
x_{k+1} = max( x_k - alpha A^H(Ax_k - b), 0 )
```

The projection is physical: scene intensities should not be negative.

## FISTA

FISTA uses the same gradient and projection, but evaluates the gradient at a momentum point. It often reaches a useful reconstruction in fewer iterations:

```text
y_k = x_k + momentum * (x_k - x_{k-1})
x_{k+1} = max( y_k - alpha A^H(A y_k - b), 0 )
```

## Verification

The implementation checks the adjoint numerically:

```text
<A x, y> should equal <x, A^H y>
```

This matters because a wrong adjoint gives the wrong gradient, even if the forward simulation looks plausible.

The optimizer stores the estimate on the padded grid, following the tutorial code. The displayed reconstruction is the center crop of that padded estimate. The summary file reports residuals for both the optimized padded estimate and the cropped reconstruction, because these are not identical once nonzero values appear outside the displayed crop.

## Run

Open `notebooks/02_padded_linear_pgd_fista.ipynb` and run the notebook from top to bottom. This notebook now owns the former command-line PGD/FISTA milestone.

Outputs are saved to:

```text
results/notebooks/02_padded_linear_pgd_fista/
```
