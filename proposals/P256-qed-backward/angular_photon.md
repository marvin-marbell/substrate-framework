# Continuous-transfer dressed photons and nonforward primitive scattering

This construction joins the **same** `RotationBand.couplings()` sea, the Gaussian rotor with `U_E=.09, K=1/.09`, and the actual bosonic-word `FilledSeaConsumer` at a genuinely nonforward recoil. It supplies the differential rate, not only a forward response. Its approximation is Gaussian photon dynamics plus the quadratic free-sea kernel and the connected tree electron response with dressed external photon legs. It does not include electron self-energy, hard vertex loops, or nonlinear compact-photon corrections.

## Continuous transfer from primitive midpoint links

Internal spectral units have lattice spacing one and rest gap `ma`. Use the deterministic periodic midpoint product quadrature

\[
p_i=-\pi+\frac{2\pi}{n}(j_i+\tfrac12),\qquad j_i=0,\ldots,n-1,
\]

with equal weights `1/n³`. Construct each actual four-orbital block from onsite and the one-/two-link couplings. Diagonalize its two occupied and two empty bands at **both** `p` and `p+sigma Q`, `sigma=±1`, evaluating the latter Hamiltonians at their genuine continuous arguments. There is no grid shift or interpolation. Only 4-by-4 matrices are diagonalized; the quadrature is a thermodynamic BZ integral, not an incommensurate finite-torus field.

For a positive axial displacement `d=t e_i`, the elementary-midpoint straight-link sum is

\[
P_{d,\sigma}(Q)=\sum_{r=0}^{t-1}e^{i\sigma Q_i(r+1/2)}.
\]

The Fourier vertex for the real cosine profile, including its factor one half, is

\[
V_{\sigma,i}(p,Q)=-\frac{iq}{2}P_{d,\sigma}(Q)
 [T_d e^{ip\cdot d}-T_d^\dagger e^{-i(p+\sigma Q)\cdot d}],\qquad q=-1,
\]

summed over the corresponding axis's primitive paths. The contact in the same normalization is diagonal in link axes:

\[
C_{ii}=2\left\langle\operatorname{tr}_{-}\sum_{d\parallel e_i}
 -\frac{q^2}{2}|P_d|^2(T_de^{ip\cdot d}+T_d^\dagger e^{-ip\cdot d})\right\rangle.
\]

The factor two divides by the real-cosine mean square `1/2`; it is the thermodynamic counterpart of `norm=N/2` in `photon_matching.py`. For occupied label `b`, empty label `a`, define

\[
\Delta_{\sigma ab}=E_{+,a}(p+\sigma Q)-E_{-,b}(p)>0,\quad
A_{\sigma iab}=\langle u_{+,a}(p+\sigma Q)|V_{\sigma,i}|u_{-,b}(p)\rangle,
\]

\[
W_{ij}=\operatorname{Re}(A_i^*A_j),\qquad
\Pi_{ij}(s,Q)=C_{ij}-4\left\langle\sum_{\sigma,a,b}
 \frac{W_{ij}\Delta}{\Delta^2-s}\right\rangle.
\]

Each real `W` is positive semidefinite on real polarizations. Inversion and time reversal cancel the imaginary antisymmetric component. Both Fourier channels are mandatory. The code evaluates the static contact/pair cancellation once and uses the subtracted finite-frequency kernel

\[
\Pi(s)=\Pi(0)-4s\left\langle\sum\frac{W}{\Delta(\Delta^2-s)}\right\rangle.
\]

Gauge vectors use the elementary-link momentum `Qhat_i=2 sin(Q_i/2)`. A midpoint pure gradient is proportional to `Qhat`, **not** `Q`; two-link hopping does not change this gauge vector. In the BZ integral, momentum translation gives `Qhatᵀ Pi(0) Qhat=0`. At finite quadrature order a continuous translation is not a permutation. The executable reports and refines this residual rather than claiming an exact finite-torus identity. No residual is subtracted by hand and no mass counterterm is inserted.

## Two-polarization pole theorem and normalization

Let `B(Q)` have orthonormal columns perpendicular to `Qhat`. The full projected transverse matrix is

\[
D(s,Q)=K|\widehat Q|^2 I-U_E^{-1}sI+B^T\Pi(s,Q)B.
\]

For `0<=s<g²`, with `g=min Delta`,

\[
-\partial_sD=U_E^{-1}I+4B^T\left\langle\sum
 \frac{W\Delta}{(\Delta^2-s)^2}\right\rangle B\succ0.
\]

If `D(0)` is positive and `b=lambda_max(D(0))/(1/U_E)<g²`, then each ordered eigenvalue strictly decreases from positive to nonpositive by `b`. Thus there are two real subthreshold crossings, with unique roots for each ordered eigenvalue, including a symmetry-protected coincident root. At a normalized null vector `v_alpha` define

\[
s_\alpha=\omega_\alpha^2,\quad
\epsilon_\alpha=Bv_\alpha,\quad
Z_\alpha^{-1}=v_\alpha^T[-\partial_sD(s_\alpha)]v_\alpha>0.
\]

These energies, eigenpolarizations and derivative residues all use the same finite-frequency channels. The code retains the entire two-by-two matrix for oblique transfers and reports its split poles; it does not replace a general tensor with a scalar isotropic fit.

For the scattering theorem select incoming `Q=q e_x` and outgoing `Q'=q' e_z`: **right-angle, nonforward** scattering. Cubic `C4` symmetry around either axis makes its transverse tensor exactly scalar, and transverse-density/longitudinal mixing is forbidden by the same symmetry. Reflection symmetry forbids transverse components of the group velocity. The invariant tensor is checked numerically and its degenerate space uses a deterministic orthonormal basis. Therefore averaging the two incoming polarizations here is legitimate; this is not an unpolarized formula for split oblique branches. Oblique output is the projected transverse tensor diagnostic, not a claim to have eliminated general longitudinal/density mixing in the full coupled Gaussian system.

## Actual recoil and bosonic Pauli amplitude

Physical units use `m=ma/a`, `k=Q/a`, `omega_phys=omega/a`; the canonical external link field is `theta=a sqrt(Z_alpha) A_alpha`. Keep the microscopic current charge `q=-1`. Each external insertion separately carries its own source factor:

\[
e_\alpha=\sqrt{Z_\alpha},\quad
\varepsilon_\alpha=e_\alpha(0,\epsilon_\alpha),\quad
\varepsilon'_\beta=e'_\beta(0,\epsilon'_\beta).
\]

The primitive two-photon contact consequently scales by **both** factors, just as the two current insertions do. There is no global replacement of the microscopic integer charge or reuse of the incoming residue for the outgoing leg.

With a rest electron solve the actual nonlinear recoil equation

\[
E(k-rn')+\omega(rn')=m+\omega(k),\quad n'=e_z,
\]

on the explicit low-BZ positive branch. The selected momenta are `q=.12,.2`, `ma=.4`, `a=.4` (mass one); the bracket `[q/(4a),q/a]` is checked, not presumed for arbitrary inputs. The consumer obtains both external energies from the continuous pole supplier and acts with its actual bosonic Pauli words and filled-sea occupation-state denominators, including electron–pair intermediates. An independent comparator contracts the band spinors with

\[
T=V'\,[E_i+\omega-h(p+k)]^{-1}V
 +V\,[E_i-\omega'-h(p-k')]^{-1}V'
 +C_{\gamma\gamma}.
\]

This expression does not generate the Pauli consumer's intermediate states. Ward substitutions are `sqrt(Z)(omega_phys, 2 sin(a k/2)/a)` on each appropriate external leg. The executable compares every spin/polarization amplitude and exercises both Ward legs, omitted-pair and omitted-contact controls, and independently computed bare-source rates.

## Differential rate: residue on the legs, dispersion in phase space

With norm-one band spinors define `A` to be the connected Pauli amplitude containing both source factors, and

\[
\overline{|A|^2}=\frac14\sum_{s,s',\alpha,\beta}|A_{\beta s',\alpha s}|^2.
\]

The canonical oscillator wavefunctions contribute `1/sqrt(2omega)` on each leg, independently of the derivative residues already contained in `A`. Golden rule in volume-normalized electron states supplies `2pi delta(E_f+omega'-E_i-omega)` and outgoing measure `d³k'/(2pi)³`. Divide by the incident one-photon flux `|v_gamma(k)|/V`. Integrating the radial delta gives

\[
\boxed{\frac{d\sigma}{d\Omega_{k'}}=
 \frac{r^2\,\overline{|A|^2}}
 {16\pi^2\omega\omega'\,|v_\gamma(k)|
  |[v_\gamma(k')-v_e(p_f)]\cdot n'|}.}
\]

The solid angle is wave-vector solid angle, not ray-direction solid angle. No continuum Lorentz-invariant phase-space assumption is made. Polarization-resolved rates average initial electron spin only; their sum divided by two is the displayed degenerate-unpolarized rate.

The group velocity is the derivative of the constructed continuous pole, evaluated by central differences with steps `2e-4` and `1e-4` in dimensionless `Q`. Since physical energy and momentum both scale as `1/a`, this derivative needs no extra factor of `a`. The code checks stencil agreement and quadrature refinement. Electron velocity is the analytic derivative of the same improved band. A rate computed using bare-photon group speed is an adverse control distinct from using bare source normalization: a correct external residue does not license a bare flux/Jacobian.

Bare-source and bare-speed relative rate errors are measurements, not required
minimum-sized effects. The heavy-neutral join at bare light mass .4 produces
bare-source errors only 2.48e−5 to 3.89e−5, while its actual Pauli/resolvent,
Ward, recoil and source-pole checks pass. An earlier fixed 1e−4 adverse-effect
threshold incorrectly rejected that valid regime. Both incidental
minimum-rate-change checks are removed rather than re-pinned; the damaged
rates and their relative errors remain in every consumer receipt.

## Reproduce

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/angular_photon.py --output proposals/P256-qed-backward/angular_photon_receipt.json
```

Default orders `24,32,48` provide three independent thermodynamic quadratures at fixed continuous physical transfers. The JSON contains actual recoil, both photon tensors/residues, finite-frequency roots, amplitude and Ward residuals, branch-labelled rates, measured group flux, outgoing radial energy-delta Jacobian, adverse controls, and rate/gradient refinement evidence. It checks the last relative rate change is below `.002`, the rate change decreases, and the gradient Ward relative residual decreases to below `2e-5`. These are explicit selected-domain numerical error checks, not a global analytic convergence rate or exact incommensurate finite-torus Ward theorem.

Scoped consumer smoke runs observed the following (the complete default CLI separately produces the multi-order rate receipt):

| BZ order | q | dσ/dΩ | Incident group flux | Pauli/comparator residual | Midpoint Ward residual |
|---|---:|---:|---:|---:|---:|
| 24 | .12 | 1.6127786180e-5 | .9992845811 | 1.73e-14 | 2.01e-14 |
| 48 | .20 | 1.3242233576e-5 | .9960885714 | 3.47e-17 | 2.32e-17 |

Both scoped cases passed every scattering/adverse check. At `q=.20`, order 48, the energy-balance residual was `-4.45e-16` and the outgoing radial Jacobian was `1.2835617397`. Independently evaluated continuous-transfer gradient relative residuals at orders `24,32,48` were `[2.4752e-7,1.7564e-8,1.2387e-11]` for `q=.12` and `[6.1745e-7,1.8936e-9,2.8544e-12]` for `q=.20`. The order-48 oblique projected tensor at `Q=(.16,.09,.04)` resolved a pole-energy splitting `9.7090e-8`; it is not replaced by the cubic-axis degeneracy formula.
