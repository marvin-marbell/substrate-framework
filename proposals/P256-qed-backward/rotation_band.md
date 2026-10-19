# Local orbital propagation, selected isotropy, and derived rotation response

This supplier owns `rotation_band.py`. It constructs the existing continuous-time
Wilson-link band from local internal matrix units rather than importing gamma
matrices. It derives its effective rotation action, projectors, finite-transfer
current and magnetic response. It does **not** demonstrate that generic bosonic
dynamics chooses this pattern, nor that a faithful orbital representation alone
explains a physical electron. The mass and charge remain declared inputs.

## Microscopic degrees of freedom and exact finite-range law

At a cubic site use four initially unassigned orbital states `|t,s>`, with two
binary labels. Neither label is declared spatial spin. Define matrix units
`E_ab=|a><b|` on either binary factor and the transitions

\[
D_1=E_{01}+E_{10},\quad D_2=-i(E_{01}-E_{10}),\quad
D_3=E_{00}-E_{11}.
\]

These are elementary qudit transition operators. Set

\[
C_i=D_3\otimes D_i,\qquad B=D_1\otimes 1_2.
\]

The two `t` blocks have opposite linear propagation patterns; a scalar internal
mixing couples them. For spacing `a`, the single-excitation microscopic generator
is the Fourier transform of an onsite matrix `O` and hopping matrices `T`:

\[
h(p)=O+\sum_{i,\ell}[T_{i\ell}e^{i\ell ap_i}
                         +T_{i\ell}^\dagger e^{-i\ell ap_i}].
\]

For the standard range-one law,

\[
O=(m+3r/a)B,\qquad T_{i1}=-iv C_i/(2a)-rB/(2a).
\]

For the improved range-two law,

\[
O=(m+9r/(2a))B,\quad T_{i1}=-iv C_i/(2a)-rB/a,
\quad T_{i2}=rB/(4a).
\]

The implemented `couplings()` supplies these matrices and integer displacement
vectors. `h_from_couplings()` independently Fourier-transforms them. They give

\[
h(p)=v\sum_i C_i\sin(ap_i)/a+M(p)B,
\quad M=m+\frac r a\sum_i\begin{cases}
1-\cos(ap_i)&\text{range one},\\
(1-\cos(ap_i))^2&\text{range two}.
\end{cases}
\]

There is no discrete-time logarithm. Continuous time is a declared microscopic
law, so the generator is exactly range one/two; its propagator need not have
strictly finite support. The extra cosine hopping is actual ultraviolet dynamics,
not a postprocessing change to the continuum amplitude. It gaps the unwanted
corners and, in the improved case, removes the leading `O(a)` mass-curvature term.

A literal commuting-site bosonic realization is a site tensor product containing
vacuum, the four single occupations, and the ten symmetric double occupations:
15 states per site. `local_bosonic_operators()` constructs the truncated bosonic
creation/annihilation matrices and number operator. `bosonic_dimer_probe()` builds
the actual 36-dimensional two-total-boson sector of two sites and resolves the
extra finite-penalty double occupations. Use

\[
H_b=\sum_x b_x^\dagger(O+\Omega 1)b_x
 +\sum_{x,i,\ell}(b_x^\dagger T_{i\ell}b_{x+\ell e_i}+\mathrm{h.c.})
 +\frac\Delta2\sum_x N_x(N_x-1).
\]

Number conservation makes the total-one-excitation band above exact for every
finite positive `Delta`; no infinite penalty limit is needed for that statement.
All site Hilbert spaces are finite and their intersite operators commute. Double
occupations cost `Delta` and interact; the full many-body spectrum is not a CAR
spectrum. An optional `Omega` above a band-norm bound (for `m>0,r>=0`,
`Omega>m+12r/a+3|v|/a` suffices for either law) keeps the bosonic vacuum from
condensing into the negative branch. It shifts both bands and must be subtracted
when comparing their *relative* single-excitation energies. **This bosonic law
alone does not provide a filled negative-energy sea or antiparticles.**

To join the separate bosonic-statistics supplier, use the same four orbital labels
as encoded fermion modes and expand every matrix entry of `O,T` as the coefficient
of its encoded even quadratic monomial. The supplier's finite constraint penalty,
extra sectors and sector preparation remain part of the microscopic model; they
are not replaced by the bare 15-state boson construction. Gauge each charged
hopping with the same `q=-e` used below. Range-two hopping follows the straight
ordered product of its two elementary links, not a new independent charge or
an ungauged shortcut. This is a coefficient interface, not proof that either
statistics or a relativistic many-body vacuum follows from the band alone.

## Why this pattern: a conditional minimal-representation derivation

A nontrivial alternative to simply declaring physical Pauli spin is to impose a
spectral condition on initially spinless orbital hopping:

1. Each gapless block has **minimal dimension two**, a traceless Hermitian linear
   symbol `w(p)=sum_i p_i W_i` and no tilt term.
2. Its two eigenvalues are `+v|p|,-v|p|` for every sufficiently small direction,
   not merely for three coordinate axes.
3. A local gap must mix two opposite blocks without splitting the target
   double degeneracy.

For a traceless two-by-two matrix, Cayley-Hamilton implies
`w(p)^2=v^2 |p|^2 1`. Matching all quadratic coefficients gives

\[
\{W_i,W_j\}=2v^2\delta_{ij}1.
\]

Every traceless two-by-two Hermitian operator is a real combination of the three
`D_i`. Its coefficient vectors must be an orthonormal triad; an orientation and
internal basis change select the displayed `D_i`. Thus the equivalent Clifford
condition is selected precisely by **three-dimensional spectral isotropy,
tracelessness, linearity and minimal block dimension**. It is not inferred from
bosonic locality or scalar dispersion data without these conditions. A commuting
three-direction hopping can fit the axis speeds and still fail badly obliquely.

There is no nonzero two-by-two matrix anticommuting with all three `D_i`, so a
single two-level block cannot carry the selected gap. With opposite blocks
`diag(D.p,-D.p)`, an off-diagonal mass matrix `K` must commute with all `D_i`;
irreducibility forces `K` to be scalar. Removing its relative phase by changing
the `t` basis gives `m D_1 tensor 1`. Four orbitals are therefore the minimal
local gapped representation under the stated requirements. This reduces the
spin/mass matrix choices to a spectral-symmetry classification. It is a material
constraint, **not a relaxation dynamics selecting those requirements**. Small
anisotropies remain allowed microscopic perturbations and persist as unequal
continuum speeds unless separately tuned or shown to flow away. No isotropizing
renormalization flow is claimed.

## Derived effective rotations and the physical-rotation qualification

Solve the vector-action condition `[J_i,C_j]=i epsilon_ijk C_k` together with
`[J_i,B]=0`. The minimal irreducible gapped representation gives

\[
J_i=-\frac i2 C_jC_k=\tfrac12 1\otimes D_i
\quad ((i,j,k)\text{ cyclic}),\qquad
[J_i,J_j]=i\epsilon_{ijk}J_k,\quad \sum_i J_i^2=\tfrac34 1.
\]

The normalization `1/2` was derived by the vector commutator, not assigned to the
orbital labels. The mass intertwiner makes the common commutant scalar, so an
extra constant vector of identity generators cannot preserve the rotation Lie
algebra. For `D(R)=exp(-i theta n.J)` and the ordinary right-handed vector rotation
`R`,

\[
D(R)[v C\cdot p+mB]D(R)^\dagger=v C\cdot(Rp)+mB.
\]

Together with coordinate rotation the continuum field transforms with total
`L+J`, where `L=-i x cross grad`; this is the band's effective spatial symmetry.
`D(2pi)=-1`, `D(4pi)=1`, and both rest sectors have `J_z=+/-1/2`.
Continuous spatial rotations are not exact on a cubic lattice: `sin(ap)` and the
Wilson mass break them by quantified irrelevant corrections. The numerical
covariance test separates the exact continuum action from the finite-spacing
residual.

The minus sign is an *internal orbital band action*. A globally sign-changed
single-state ray does not by itself prove physical fermionic spin. On a bare
boson Fock space its second-quantized central action is number parity, not the
identity action expected for ordinary microscopic scalar spatial rotations.
Promoting the combined effective rotation to a physically realized spinor
quasiparticle therefore requires the independently encoded statistics sector,
its gauge/central identification and the joined low-energy transformation law.
Nothing here establishes that ordinary scalar bosons spontaneously acquire that
identification. The classification explains why the selected low-energy band
has a spinor symmetry action; it does not conceal its spectral inputs as an
origin of real-space rotations.

## Sectors, mass, dispersion, and unwanted species

Anticommutation now gives consequences rather than additional inputs:

\[
E_\pm(p)=\pm\sqrt{v^2\sum_i\sin^2(ap_i)/a^2+M(p)^2},\qquad
P_\pm(p)=\tfrac12(1\pm h(p)/E(p)).
\]

Both projectors have rank two. `projector()` and `spinor()` provide the projectors
and a normalized positive-sector basis. The antiunitary matrix
`C=D_2 tensor D_2` obeys `C h(p)^* C^dagger=-h(-p)`; it supplies the particle/hole
band map, not a many-body vacuum. After negative-sector fermionic filling,
removing a `q`-charged occupied mode gives a hole of charge `-q`. Before that
filling the negative branch is just a bosonic orbital branch.

For the range-one law at low momentum,

\[
E^2=m^2+(v^2+mra)|p|^2+\frac{r^2a^2}{4}|p|^4
 -\left(\frac{v^2a^2}{3}+\frac{mra^3}{12}\right)\sum_i p_i^4
 +O(|p|^6).
\]

For range two,

\[
E^2=m^2+v^2|p|^2+
 \left(-\frac{v^2a^2}{3}+\frac{mra^3}{2}\right)\sum_i p_i^4
 +O(|p|^6).
\]

Thus the rest energy is exactly the declared scalar mixing `m`; the standard
kinetic coefficient is not exactly `v^2` at finite spacing. The improved law
repairs that mismatch without a nonlocal generator. Oblique versus axis momenta
of equal length distinguish cubic corrections. At a corner with `n` nonzero
`pi/a` components, the positive gap is `m+2rn/a` (range one) or `m+4rn/a`
(range two). For `r=0` all eight corners have mass `m`; the runtime prints that
negative control. Heavy corners decouple only in the low-energy limit with
positive `r` and decreasing `a`, not from the full finite-spacing Hilbert space.

## Link current, two-photon contact, and leading magnetic moment

Put a Peierls factor on each elementary hop and the ordered product on a two-hop
path. Functional differentiation of these finite paths gives the current in
`current(pout,pin)`. With `s_i=a(pout_i+pin_i)/2` and
`t_i=a(pout_i-pin_i)/2`, its charge-divided matrices are

\[
V_i=v\cos s_i C_i+r\sin s_i B\quad\text{(range one)},
\]
\[
V_i=v\cos s_i C_i+r[2\sin s_i-\cos t_i\sin(2s_i)]B
\quad\text{(range two)}.
\]

They obey the finite-transfer identity

\[
\sum_i\frac{2\sin t_i}{a}V_i=h(pout)-h(pin).
\]

Continuous-time density is exactly `q 1`; `vertex=q(eps_0 1-eps_i V_i)` with
`q=-e`. There is no discrete-time frequency sine or implicit time-Ward repair.
The second derivative yields `contact=e^2 sum_i epsin_i epsout_i K_i`:

\[
K_i=a[-v\sin s_i C_i+r\cos s_i B]\quad\text{(range one)},
\]
\[
K_i=a\left[-v\sin s_i C_i+
 r\{2\cos s_i-2\cos(ak_i/2)\cos(akout_i/2)\cos(2s_i)\}B\right]
\quad\text{(range two)}.
\]

The cosine factors are the separated insertions on the two-link path. They are
not optional: replacing the finite-transfer current/contact by ordinary uniform
momentum derivatives generally fails finite-transfer Ward cancellations.

For smooth weak magnetic fields `pi=-i grad-q A`,
`[pi_i,pi_j]=i q epsilon_ijk B_k`. The derived internal transition algebra gives

\[
(C\cdot\pi)^2=\pi^2-2q J\cdot B.
\]

Eliminating the negative block to leading order in `v|pi|/m` gives

\[
h_+=m+\frac{v^2\pi^2}{2m}-\frac{qv^2}{m}J\cdot B+\cdots.
\]

Relative to inertial mass `m_kin=m/v^2`, the coefficient is `g=2`; no physical
spin label or Pauli magnetic term was supplied independently. The Wilson
range-one term adds `ra pi^2/2`, changing `m_kin` to `m/(v^2+mra)` without adding
that same leading magnetic coefficient; hence its finite-spacing value is
`g_eff=2v^2/(v^2+mra)`. The improved hopping removes this quadratic Wilson term.
A finite oscillator weak-field calculation in `magnetic_probe()` uses the
*derived* continuum matrices and checks actual energy eigenvalues, squared-state
residuals and splitting as `B` decreases. Its `g` result is explicitly a continuum
weak-field limit, not an unmeasured exact lattice moment or a radiative anomalous
moment. `v` still needs to match the photon speed in the joined construction.

## Exact existing-consumer map and numerical evidence interface

With `U=((D_1+D_3)/sqrt(2)) tensor 1`,

\[
U C_i U^\dagger=\alpha_i^{Wilson},\qquad U B U^\dagger=\beta^{Wilson}.
\]

The module does not import these consumer matrices at module load. `run()` alone
imports `WilsonLink` for a comparison. The same unitary maps `h`, one-photon
vertices, two-photon contacts and positive-band spinors. Parent integration can
use `U^dagger` on the existing external states and evaluate the same resolvents
with the microscopic band; no new cross-section normalization is needed.

Run with the declared project interpreter:

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/rotation_band.py
```

The deterministic JSON includes:

- local-coupling Fourier equality, exact unitary Hamiltonian/current/contact maps,
  and finite-transfer Ward residuals;
- paired positive/negative spectra, projector algebra, charge conjugation and
  normalized-band state residuals;
- relativistic coefficients, rest spin Casimir, `2pi/4pi` response, exact continuum
  covariance, finite-spacing covariance and equal-length oblique dispersion;
- all eight corner gaps and the ungapped-species `r=0` negative control;
- a direct weak-field continuum spectral splitting approaching `g=2`;
- the actual bare bosonic dimer spectrum and double-occupation admixture at finite
  penalties, explicitly separated from the encoded-statistics supplier;
- commuting, anisotropic and deleted-hopping false positives that an axis-only
  or rest-mass-only check would miss.

Numerical outputs must be attached by the integration owner after the coordinated
execution; this derivation document does not preclaim a successful run. Fundamental
quantum evolution, selected isotropy/minimality, scalar mass, common charge,
sector realization and photon-speed matching remain separate explicit inputs or
joined obligations. A completed band map is not completion of the campaign.
