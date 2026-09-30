# P253/0166 validation boundary

The runnable coefficient oracle exited normally in the repository Python environment:

```text
.venv/bin/python proposals/P253-euler-particle-mechanisms/attempts/0166-two-euler-cell/verify_column_escape.py
full 3-D mixed escape norm/sqrt(volume)=0.342928564
three direct full-pressure Euler accelerations agree; dropping mixed pressure fails
exterior vertical acceleration at (x,y,z)=(0,0,1.3): 0.012363043548
```

With independently prepared amplitudes `α=0.7`, `β=1.2`, the predicted mixed escape norm per root torus volume is `|αβ|/√6=0.342928564`. Direct central differences evaluate the **full 3-D velocity** and **one physical pressure** at three nonsymmetric points and agree with the derived time derivative within `2e−9`; the plausible wrong case retaining only separate planar self pressures differs by more than `0.1` at each point. Sixteen-grid trigonometric quadrature reproduces the `L²` Fourier coefficient. For a smooth compact slab profile, independent midpoint quadrature of its positive pressure moment and central differentiation of its exterior pressure yield nonzero initial vertical acceleration. This checks initial physical Euler accelerations, not positive-time nonlinear integration, generic invariant-family absence, or a particle. Smooth local existence and the explicit nonzero initial escape supply the scoped positive-time conclusion. Pyright diagnostics of the runnable oracle returned `OK`.
