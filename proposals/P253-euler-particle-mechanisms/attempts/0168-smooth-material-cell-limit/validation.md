# P253/0168 validation boundary

From the PR branch, the exact-shear/whole-Euler-pressure oracle exited normally:

```text
.venv/bin/python proposals/P253-euler-particle-mechanisms/attempts/0168-smooth-material-cell-limit/verify_smooth_cell_limit.py
exact shear material cell width=0.40: variance=0.0132270721, variance/h^2=0.08266920
exact shear material cell width=0.20: variance=0.0033266730, variance/h^2=0.08316683
exact shear material cell width=0.10: variance=0.0008329168, variance/h^2=0.08329168
exact shear material cell width=0.05: variance=0.0002083073, variance/h^2=0.08332292
exact high-gradient Euler shear N=8: internal energy/(rho volume)=0.262589687
exact high-gradient Euler shear N=16: internal energy/(rho volume)=0.253196289
exact high-gradient Euler shear N=32: internal energy/(rho volume)=0.250802158
exact high-gradient Euler shear N=64: internal energy/(rho volume)=0.250200733
exact transverse Euler N=8: initial H1=0.125000, time-1 transverse-gradient L2=0.503891
exact transverse Euler N=16: initial H1=0.062500, time-1 transverse-gradient L2=0.500976
exact transverse Euler N=32: initial H1=0.031250, time-1 transverse-gradient L2=0.500244
exact transverse Euler N=64: initial H1=0.015625, time-1 transverse-gradient L2=0.500061
remote-swirl resting-cell acceleration=0.0078125016, pressure Hessian=(0.00390624954257901, 0.00390624954257901, -0.007812501215398449)
zero-velocity Euler control: center and internal accelerations both zero
```

The first oracle evaluates the global exact 3-D Euler shear `u=(sin y,0,0)` along material cells. Its independent midpoint average agrees with the analytic variance `(1−sin(h)/h)/2` and approaches the nonzero rescaled variance coefficient `1/12`; the **physical** unscaled energy still vanishes as `h²`.

The high-gradient family `u_N=(sin y+sin Ny,0,0)` is also an exact global 3-D Euler shear; its cell energy per `ρ|Ω|` tends to `1/4` at amplitude one, so an asserted universal zero-energy theorem would fail this positive control. Its stress divergence is identically zero by analytic coordinate structure, **not** by quadrature alone. The exact transverse 3-D Euler continuation (13) tests an adverse ordering: an off-family perturbation whose initial `H¹` distance tends to zero yields a positive time-one `H¹` gradient bounded away from zero. This refutes an `N`-uniform strong-norm remainder, not an `L²` or scale-aware bound.

The finite-difference pressure probe evaluates the **whole-space exterior pressure** of the smooth compact swirl from P253/0001 and distinguishes a resting cell from the zero-Euler control in both center force and trace-free internal pressure Hessian. It checks an initial physical acceleration, not a numerical nonlinear Euler integration. `xd://lsp` Pyright diagnostics on the new Python file returned `OK`; the uniform estimates, invariance and pressure source follow from the derivation rather than a test count.
