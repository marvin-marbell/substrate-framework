# Exact momentum reduction and the universal sea-response logarithm

The filled sea now predicts a photon coefficient instead of borrowing one. Two constructive results make that calculation scalable and identifiable: an exact finite Fourier reduction, and a thermodynamic susceptibility with a derived infrared logarithm.

## 1. Exact reduction of the same microscopic law

Use the primitive Pauli-selected fermionic sector and the four-orbital law of `RotationBand.couplings()`. With unit lattice spacing, a forward displacement d contributes

The executable API now requires `transverse_spectrum(L, band, mode=1, *, gradient=False, polarization_axis=1)` and `momentum_response(L, band, mode=1, *, gradient=False)`. The actual `RotationBand` owns a,m,r,v,e and its hopping flag; no internal `ma` factory or legacy alias remains. The CLI and `vacuum_log` explicitly supply their previous bands, preserving those workflows' parameters. For general spacing, momentum is physical `p=2pi n/(a L)`, Fourier phases are `exp(i a p.d)`, transfer angle is `qangle=a q`, and the response source is compact link angle `theta=a A`. The compact Maxwell difference is `2sin(qangle/2)`, not the physical difference divided by a. Physical-A current and contact derivatives are respectively a and a² times the compact-angle derivatives. The unit-spacing equations and prior executed numbers below retain their original convention.

\[
h(p)=h_0+\sum_d[T_d e^{ip\cdot d}+T_d^\dagger e^{-ip\cdot d}].
\]

Consider `theta_i(x)=Re[s_i exp(i Q x)]`, `Q=2pi n/L` along x. Differentiating the actual straight primitive-link path gives

\[
P_d(Q)=s_i\sum_{r=0}^{d_i-1}e^{iQr\delta_{ix}},\qquad
J_{p+Q,p}=-\frac{iq}{2}\sum_d P_d(Q)
[T_d e^{ip\cdot d}-T_d^\dagger e^{-i(p+Q)\cdot d}],
\]

with integer microscopic charge `q=-1`. Both +Q and -Q channels enter. The diagonal contact is

\[
K_{p,p}=-\frac{q^2}{2}\sum_d|P_d(Q)|^2
[T_d e^{ip\cdot d}+T_d^\dagger e^{-ip\cdot d}].
\]

These factors assume distinct nonzero +Q/-Q Fourier channels; the executable rejects zero and Nyquist modes rather than double counting them. `s_y=1` is the transverse cosine; `s_x=exp(iQ)-1` is the exact forward gradient of a site cosine.

Diagonalize only 4-by-4 momentum blocks. Let o be an occupied negative mode, u an empty positive mode, `Delta=E_u-E_o>0`, and `w=|J_uo|^2`. Then

\[
E''=C-2\sum_{\pm Q,p,u,o}\frac w\Delta,\quad
C=\sum_{p,o}\langle o|K|o\rangle,\quad
\delta\chi_B(Q)=\frac{E''}{L^3\widehat Q^2/2}.
\]

This is exactly the dense finite Hamiltonian susceptibility, not a continuum replacement. The uniform temporal-frequency coefficient is obtained from the same link derivative and spectral gaps:

\[
\delta\chi_E=\frac2{L^3}\sum_{p,u,o}\frac{|J_y(p)_{uo}|^2}{\Delta^3}>0.
\]

`vacuum_limit.py` verifies the finite reduction against the independently assembled `PeriodicBand` dense matrices at L4/L6. Maximum observed electric error is `2.49e-13`; magnetic error `5.21e-17`; contact and pair errors below `2.85e-14`. The longitudinal Ward cancellation and the deleted-contact/deleted-pair alternatives all reach their physical assertions.

## 2. The infrared/volume explanation becomes measurable

At `ma=.05`, the large uniform-electric coefficient decreases from **7.84878** at L8 to **.0753786** at L96; it was a real small-box response, dominated by the unresolved low-energy region. It is not an arbitrary sign failure to erase. At `ma=.2`, L48/L64/L96 give electric `.0450462/.0450385/.0450382`; the static holonomy curvature per cell tends from `-2.98e-7` to `-1.31e-11`. Magnetic coefficients at the respective nonzero modes are `.0731277/.0734346/.0736529`.

The formerly negative L4 magnetic datum therefore does not describe the same low-momentum/large-volume regime. The new calculation resolves how the construction evolves between regimes. Electric and magnetic finite-cutoff coefficients differ; the next photon theorem uses both rather than declaring their equality.

The separate fixed `mL=6.4` sequence varies ma with L and keeps physical momentum/mass fixed. Its receipt is not confused with increasing physical volume at fixed cutoff. `vacuum_limit_large_receipt.json` preserves the original executed source `0585ae7...` for L48/L64/L96; the integration run records the current factored source and small-volume controls separately.

## 3. Thermodynamic trace formula

The selected algebra has `h=d_mu A_mu`, `E=|d|`, and spectral projectors `P_±=(I±h/E)/2`. For `J_y=u_mu A_mu`, where `u=partial_y d`, the derived Clifford trace gives

\[
\operatorname{tr}(P_+J_yP_-J_y)=2\left(|u|^2-\frac{(d\cdot u)^2}{E^2}\right).
\]

Consequently the thermodynamic coefficient is

\[
\delta\chi_E(ma)=\int_{[-\pi,\pi]^3}\frac{d^3p}{(2\pi)^3}
\frac{|u|^2-(d\cdot u)^2/E^2}{2E^3}.
\]

For the improved law, `d=(ma+sum_i(1-cos p_i)^2,sin p)` and `u_0=2(1-cos p_y)sin p_y`, `u_y=cos p_y`. The ordinary Wilson law instead has `u_0=sin p_y`. `vacuum_log.py` evaluates this formula with explicitly subdivided tensor Gauss quadrature near zero momentum, not a coarse grid that misses the small-mass region. It independently checks the formula against actual spectral eigenvectors at 24 seeded momenta for each law and against the finite Fourier sum: maximum errors `4.44e-16` and `1.18e-16`.

## 4. A derived universal coefficient

In the shell `ma << |p| << 1`, either law has `d=(ma,p)+higher orders` and `u=(0,e_y)+higher orders`. Angular averaging gives

\[
\delta\chi_E\big|_{\rm shell}=
\frac1{4\pi^2}\int dp\,\frac{p^2}{(p^2+ma^2)^{3/2}}
\left(1-\frac{p^2}{3(p^2+ma^2)}\right).
\]

The logarithmic integrand is therefore `1/(6pi^2 p)`, yielding

\[
\boxed{\delta\chi_E(ma)=\frac1{6\pi^2}\log\frac1{ma}+C_{\rm regulator}+o(1).}
\]

The isolated light cone supplies this logarithm; the Wilson corners stay massive. The ordinary law's O(p²) scalar correction and the improved law's O(p⁴) correction change finite terms, not the shell coefficient. The formula is a derived free-sea susceptibility theorem, not an inserted QED loop coefficient.

The actual ma-halving sequence `.4` through `.000390625` gives final logarithmic slope divided by `1/(6pi²)` **1.00744** for ordinary Wilson and **1.00151** for improved Wilson. Differences between quadrature orders8/12 stay below `1.3e-8` on this sequence; these are numerical convergence observations, not rigorous quadrature error bounds. The limiting coefficient follows from the shell derivation.

## 5. Constructive continuation and execution

```sh
python proposals/P256-qed-backward/vacuum_limit.py --output proposals/P256-qed-backward/vacuum_limit_receipt.json
python proposals/P256-qed-backward/vacuum_log.py --output proposals/P256-qed-backward/vacuum_log_receipt.json
```

Both real CLIs completed with their controls true. The logarithm receipt source is `e591d46073b5a1f2e8fa1009fe8bf1b823fba68114368fe6c61ca639ce65c52b`. The same spectral channels feed the [positive-residue photon construction](photon_matching.md), which in turn reaches the actual Pauli Compton response. These results provide a concrete route to charge normalization and cutoff matching within the declared quantum microscopic model. Photon nonlinear fluctuations and further matter interactions are subsequent improvements of that theorem, not replacements for its demonstrated mechanism.
