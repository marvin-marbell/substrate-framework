# Filled-sea feedback into the primitive compact photon stiffness

## Boundary and runnable mechanism

`vacuum_feedback.py` takes the **same selected four-orbital band and its filled
negative sea**, after the primitive Pauli/code-sector statistics supplier. It
assembles actual finite periodic spatial matrices using
`RotationBand.couplings()`. No Dirac vacuum-polarization coefficient or fitted
charge enters. The reduction to the sum of occupied single-particle energies
uses the supplied quadratic, exclusion-respecting filled-sea sector; it does
not independently derive the quantum ground-state postulate or that sector.
The mass, hopping, velocity, Wilson improvement and integer source charge are
still inputs. Physical compact rotors are on spatial links, not internal code
edges; this calculation differentiates their classical background phases.

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 /home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/vacuum_feedback.py
```

Defaults: improved band, `L=4,6`, spatial modes `n=1,2`, `ma=.05,.2`, `a=1`,
finite-difference amplitude `.002` and its half, finite gauge amplitude `.31`,
and one BLAS thread. The largest matrix is `4*6^3=864` square. The CLI prints
JSON and exits nonzero if any gauge/contact/empty-sea discriminator fails.
A negative magnetic coefficient is **reported**, not suppressed by a pass
criterion requiring the desired answer. Scientific execution belongs to the
integration owner; this document derives the mechanism and does not claim an
unobserved numerical outcome.
The standalone CLI sets native BLAS thread limits before importing NumPy;
`VACUUM_FEEDBACK_THREADS` overrides its default of one. Importing this module
as a library does not change the host process's threading configuration.

Useful adverse/regime probes:

```sh
# Minimal full discriminator, including wrongly gauged two-link hopping:
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/vacuum_feedback.py \
  --lengths 4 --modes 1 --mass-ratios .05 --gauge-amplitude .7
# Delete improvement legitimately; the wrong-two-link discriminator is inapplicable:
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/vacuum_feedback.py \
  --lengths 4 6 --modes 1 --mass-ratios .05 .2 --unimproved
# Finer low-momentum box, larger matrix (not the default):
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/vacuum_feedback.py \
  --lengths 8 --modes 1 --mass-ratios .1 .2
```

The adverse constructions are calculated alongside the correct one on the
same unperturbed spectrum; there is no switch that silently replaces the
physical output. Changing `--a` at fixed `ma` changes physical units, not the
cutoff/mass ratio. A genuine fixed-physical-mass cutoff sequence instead varies
`ma=m*a` and increases `L` to keep physical size and the momentum fixed.

## Source phase and exact finite gauge identity

Let cells be `x in (Z/LZ)^3`, four orbitals per cell, with physical displacement
`a d`. The supplier defines

\[
h(p)=h_{\rm on}+\sum_d[T_d e^{iap\cdot d}+T_d^\dagger e^{-iap\cdot d}].
\]

Accordingly the assembled matrix has row `x`, column `x+d` equal to `T_d`.
The unnormalized compact variable is

\[
\theta_i(x)=a\widetilde A_i(x),\qquad q_{\rm integer}=-1.
\]

The field is a primitive link integral (the equation is exact for a
link-averaged field), and every coupling is gauged as

\[
H_{x,x+d}(\theta)=T_d\exp[-iq_{\rm integer}
    \sum_{\ell\in P(x,d)}\theta_\ell].
\]

Nearest paths contain one primitive link. Improved straight two-link paths
contain **both** consecutive links, including periodic wraparound. Reverse
hopping is the Hermitian conjugate, not an independently assigned phase.
On-site mixing has no spatial phase. This convention gives
`h(p-q_integer Atilde)` for a uniform field.

For a real periodic site function `lambda`, choose
`theta_i(x)=t[lambda(x+e_i)-lambda(x)]`. The entire path sum telescopes. With
`U_x=exp(i q_integer t lambda(x))` on all four orbitals,

\[
H(t\nabla\lambda)=U H(0)U^\dagger.
\]

This is a **nonzero-amplitude finite-matrix** statement on the torus, including
improved links. The code checks matrix entries, the full spectrum and occupied
ground energy at `t=.31`. The matched wrong construction keeps every coupling
but replaces a two-link phase sum by its first link only. It must lose this
unitary identity, change the spectrum and sea energy, and give a nonzero
pure-gradient curvature. Merely proving a small-amplitude Ward cancellation
would not suffice.

## Derivative, virtual pairs and mandatory contact

For an arbitrary real link profile `s`, set `theta=t s` and let
`S_d=sum_path s`. Differentiating the **assembled** link exponential gives

\[
J_{x,x+d}=-iq S_d T_d,\qquad
K_{x,x+d}=-q^2 S_d^2 T_d
\]

at zero amplitude, with conjugate reverse blocks. `K` is the link-derived
contact; it is not a separately fitted counterterm. There are `2 L^3`
occupied negative eigenstates and `2 L^3` empty positive ones. The CLI rejects
a probe that closes the gap at its sampled amplitudes. The filled energy is

\[
E_{\rm sea}(t)=\sum_{o\in-}\epsilon_o(t).
\]

Ordinary finite-matrix eigenvalue perturbation gives

\[
E''_{\rm sea}(0)=
 \underbrace{\sum_{o\in-}\langle o|K|o\rangle}_{C}
 -2\sum_{o\in-,u\in+}
  \frac{|\langle u|J|o\rangle|^2}{\epsilon_u-\epsilon_o}.
\]

Occupied-to-occupied terms cancel pairwise in the filled sum, including basis
freedom inside degenerate occupied subspaces. The second term describes
virtual electron-hole states, **not an empty negative band or a primitive CAR
input**. The code computes this sum using the actual finite eigenvectors and
checks it against the symmetric second derivative of independently
diagonalized `H(+t)` and `H(-t)`, at two step sizes.

For a pure gradient, the two terms are individually nonzero and cancel.
Deleting the contact leaves a spurious negative gauge kernel; deleting the
virtual filled-negative contribution leaves a spurious positive gauge kernel.
Both fail the exact zero-response criterion. Deleting the negative sea
altogether gives zero for every response: that tempting false positive passes
the zero-gradient condition but fails the **nonzero transverse sea-energy and
susceptibility** positive control. The transverse spectrum also must actually
change. A uniform static vector potential is not used as a mass discriminator:
on a periodic box it changes the torus holonomy and need not be gauge-trivial.

## Volume, momentum denominator and dimensionless induced stiffness

Use `hbar=c=1` and an isotropic reference lattice time spacing `a_t=a`.
`H`, `m`, energy and frequency have inverse-length units. `theta`, `t`, `s`
and `chi` are dimensionless. There are `N=L^3` cells and physical volume
`V=N a^3`. The transverse profile is

\[
s_y(x)=\cos(2\pi n x_1/L),\quad s_x=s_z=0,
\qquad k=2\pi n/(La),\quad \widehat k=2\sin(\pi n/L)/a.
\]

The code explicitly sums the squared oriented plaquette curls
`D=sum_(x,i<j) (curl s)_ij(x)^2`. For a non-Nyquist mode,
`D=N (a khat)^2/2`; at the Nyquist mode the cosine mean square is one rather
than one-half, so **the explicit sum, not that shortcut, is the denominator**.
Pure-gradient `D` is zero up to roundoff and is never used as a stiffness
denominator.

The small-field compact magnetic energy with stiffness `chi` is

\[
E_B=\frac{\chi}{2a}\sum_{x,i<j}\theta_{ij}(x)^2
   =\frac{\chi}{2}\int_V d^3x\,\widetilde B^2,
\qquad \widetilde B_{ij}=\theta_{ij}/a^2.
\]

Thus the finite static matter correction is **derived**, without canonical
charge factors, as

\[
\boxed{\Delta\chi_B(k,L,ma)=a E''_{\rm sea}(0)/D}.
\]

The JSON exposes `D`, `D/a`, `V`, both momenta, both spectral terms and finite
energy derivatives, so a phase/volume/momentum normalization error is visible.
There is no insertion of the pure-gauge `q_R=.30350041` into `J` or `K`; doing
so would double-use a normalization that is itself changed by matter.

The two masses probe cutoff/mass ratios `1/(ma)=20,5`; the two boxes and modes
probe different `ka` and `mLa`. These are finite regulator data, not a clean
continuum log. In particular `ma=.05` on `L=4,6` has `mLa=.2,.3`, so winding
and unresolved infrared modes can dominate. The CLI reports the sign at every
point, mass-direction comparisons and momentum/volume comparisons. Positive
`DeltaChi_B` supports an increased static photon stiffness in that sampled
sector; negative values or strong momentum/volume dependence defeat a universal
weak-photon inference there. Neither sign establishes a continuum coefficient.

## Electric-frequency counterpart and charge/residue conditions

The same finite virtual pairs supply the imaginary-frequency current kernel
for a fixed spatial profile:

\[
\mathcal K_s(i\omega)=C_s-
 2\sum_{o,u}\frac{\Delta_{uo}|J_{uo}|^2}{\Delta_{uo}^2+\omega^2},
\quad \Delta_{uo}=\epsilon_u-\epsilon_o>0.
\]

At zero frequency this is the energy curvature above. For a **uniform**
`y`-link profile, the code also calculates

\[
\mathcal K(i\omega)-\mathcal K(0)=
 2\omega^2\sum_{o,u}\frac{|J_{uo}|^2}
 {\Delta_{uo}(\Delta_{uo}^2+\omega^2)},
\quad
\Delta\chi_E=\frac{2}{aN}\sum_{o,u}\frac{|J_{uo}|^2}{\Delta_{uo}^3}.
\]

The denominator follows from the electric Maxwell action
`(chi_E a/2) sum_x (d theta_y/dt_physical)^2`. The code reports the low-frequency
coefficient and actual kernels at `.1m,.2m`, normalized frequency-difference
coefficients and the discrepancy from the static transverse magnetic value.
The uniform **static** curvature is finite-torus holonomy sensitivity and is
subtracted, not mislabelled a photon mass. This electric calculation is not yet
a complete temporal compact-link determinant or a density/current Ward test;
nonuniform temporal fields require the supplied density coupling and consistent
`Atilde_0`, not just a time-varying spatial link.

Only after the same regulator/ensemble, long-wavelength and low-frequency
limits, full temporal gauge identities, magnetic/electric matching (or a
measured physical light-speed rescaling), and interacting unquenched matching
have been established may one identify a Lorentz Maxwell coefficient

\[
\chi_{\rm full}=\chi_{\rm puregauge}+\Delta\chi,
\quad A_{\rm canonical}=\sqrt{\chi_{\rm full}}\,\widetilde A,
\quad |q_{\rm canonical}|=1/\sqrt{\chi_{\rm full}}.
\]

The unnormalized transverse propagator then has residue `1/chi_full` in those
matched units; the canonical field has unit Maxwell residue. Anisotropic
`chi_E != chi_B` instead gives speed squared `chi_B/chi_E` and does not permit
that single-coefficient identification without further physical matching.
The CLI's optional `--pure-gauge-chi 10.85629` is only a labelled **conditional
magnetic normalization** reference from a cold pure compact sector. It is not
an equilibrium/unquenched calibration or a full dressed electron-photon charge.
The reported matter feedback invalidates independently fixing a photon
endpoint normalization and then leaving it untouched when the same charged
sea is joined.

No radiative Compton correction, continuum renormalization scheme, controlled
logarithmic extrapolation, dynamically selected mass/hopping, interacting
resummation or hot-sector Coulomb-phase inference is supplied by this finite
calculation. Those are distinct joined obligations, not names for this probe.
