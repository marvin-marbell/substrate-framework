# Actual-current quantum soft clouds and a finite-resolution Compton outcome

This is a **conditional joined leading-soft calculation**, not a derivation of
quantum mechanics, a compact Coulomb ground state, hard radiative Compton, or a
general infraparticle theorem. It replaces the interpretation “the outgoing bare
charged Fock state is the physical final answer” with a precisely delimited
transverse quantum cloud and a detector probability. The selected microscopic
matter/rotor Hamiltonian, Hilbert space, commutators, Born rule, positive input
charge magnitude and photon normalization remain explicit inputs.

## Executable and ownership

From `proposals/P256-qed-backward`:

```sh
/home/administrator/substrate-framework/.venv/bin/python soft_quantum.py
```

`--charge` selects a positive input magnitude. The default `.30350041` is the
campaign's **model normalization**, not an independently derived fully dressed
physical charge or measured fine structure. The CLI emits JSON and asserts its
current/Hellmann–Feynman, oscillator cutoff, angular, gauge and infrared detector
controls. Scientific execution is reserved to the integrating parent; this
artifact does not claim an unobserved successful run or prefill a receipt.

The default candidate is `RotationBand(a=.025, m=1, e=.30350041, r=1, v=1,
improved=True)` and a parameter-matched `WilsonLink` with photon speed `c=1`.
The **actual** `WilsonLink.kinematics(.2, .7)` supplies all four hard momenta.
The electron charge is `q=band.charge=-band.e`, including its sign. The hard
amplitude is computed through `FilledSeaConsumer(band, photons, kinematics)`:
primitive Pauli currents, filled negative sea, virtual electron–hole channels,
contact term and connected vacuum subtraction. No gamma-matrix comparator is
evaluated by this new consumer. `WilsonLink` remains an existing conditional
supplier; its use is not a new microscopic explanation.

## 1. Velocity belongs to the actual current

For either positive-energy eigenvector `u_s(p)` of the actual band,

\[
 v_j(p)=\langle u_s(p)|J_j(p,p)|u_s(p)\rangle,
 \qquad J_j(p,p)=\partial_{p_j}h(p),
 \qquad qv_j=\langle j_j\rangle.
\]

`RotationBand.current` is explicitly **divided by charge**. Multiplying its
expectation by `q` once, not twice, produces the oscillator source. The program
checks the eigenstate equation, both spin labels and a symmetric finite
difference of `RotationBand.energy`. Off-axis probes include the unimproved
finite-spacing band with a different matter speed: inserting `p/E`, a continuum
velocity, or a photon group velocity in place of the matter current would not
satisfy that obligation. The initial and final velocities used below are computed
from the same hard momenta, not fitted to a desired soft coefficient.

At finite spacing the band need not be Lorentz invariant. The soft photon
approximation below uses `omega=c|k|`, the long-wavelength limit of the selected
canonical photon band, not the full finite-spacing dispersion at all momenta.
The default upper soft energy `.001` is checked to be below one percent of the
smaller incoming/outgoing hard photon energy and rest gap. This hierarchy is a
regime check, not an estimate of all omitted corrections.

## 2. Derivation from a forced quantum oscillator

Use `hbar=1`, canonical transverse photon normalization

\[
 H_\gamma=\sum_{k,r}\omega_k a^\dagger_{k,r}a_{k,r},\qquad
 A_T(x)=\sum_{k,r}\frac{\epsilon_r}{\sqrt{2\omega_k V}}
   \left(a_{k,r}e^{ik\cdot x}+a^\dagger_{k,r}e^{-ik\cdot x}\right).
\]

The zero-point constant is immaterial here. Projecting the selected matter
coupling `-j.A_T` on a narrow asymptotic wave packet traveling at the
**current-derived** velocity gives a single-mode Hamiltonian

\[
 H_v(t)=\omega a^\dagger a
 -g_v\left(e^{ik\cdot vt}a+e^{-ik\cdot vt}a^\dagger\right),\qquad
 g_v=\frac{q\,v\cdot\epsilon}{\sqrt{2\omega V}}.
\]

This is an asymptotic classical **source approximation inside a quantum photon
Hamiltonian**, not a claim that the full electron is a prescribed classical
trajectory. Soft recoil, velocity spread and structure of the hard event are
neglected at leading order. Adiabatic switching selects the stable forced
solution on each asymptotic half-line. The Heisenberg equation

\[
 \dot a=-i\omega a+ig_v e^{-ik\cdot vt}
\]

has particular solution `a=eta_v exp(-ik.v t)`, where

\[
 \eta_v=\frac{g_v}{\omega-k\cdot v}.
\]

Equivalently the rotating/displaced oscillator has the generator

\[
 H_v^{\rm mov}=(\omega-k\cdot v)a^\dagger a-g_v(a+a^\dagger)
 =\Omega_v(a^\dagger-\eta_v)(a-\eta_v)-g_v^2/\Omega_v,
 \quad \Omega_v=\omega-k\cdot v>0.
\]

It is a moving-frame generator, not an assertion that the free photon laboratory
frequency changes to `Omega_v`. Its coherent eigenstate is the ground state of
this generator. The negative shift includes the source interaction and must not
be confused with the positive radiated free-field energy.

Model the hard event as sudden on soft time scales at `t=0`: velocities change
from `vi` to `vf`, while the oscillator wavefunction is initially unchanged.
Expansion in the final displaced basis gives relative displacement

\[
 \alpha_{k,r}=\eta_f-\eta_i
 =\frac{q}{\sqrt{2\omega V}}\left[
 \frac{v_f\cdot\epsilon_r}{\omega-k\cdot v_f}
 -\frac{v_i\cdot\epsilon_r}{\omega-k\cdot v_i}\right]
\]

up to an overall phase/sign. This is obtained from the Hamiltonian/current, not
inserted as a QED formula. Straight asymptotic currents and the sudden-event
limit are explicitly chosen approximations. Finite event duration changes
non-leading frequency dependence; it does not authorize an arbitrary spectral
floor that discards genuinely soft modes.

### An actual quantum eigensystem, not just the classical solution

The code constructs truncated ladder matrices and **diagonalizes both actual
forced Hamiltonians** with `scipy.linalg.eigh` at cutoffs 4, 12 and 32. It obtains
`P_n=|<n;vf|0;vi>|^2` from the two numerical eigenvector matrices. The independent
completed-square prediction is

\[
 P_n=e^{-\mu}\mu^n/n!,\quad \mu=|\eta_f-\eta_i|^2,
 \qquad |\langle0;vf|0;vi\rangle|^2=e^{-\mu}.
\]

No analytic coherent vector is supplied as the numerical eigensystem. The last
cutoff must recover overlap, first six probabilities, energy shift and coordinate
variance, while the deliberately inadequate cutoff must fail the energy check.
The boundary Fock occupation is reported. A one-mode stress source with final
`|eta|=1.2` makes the cutoff error discriminating rather than invisible.

Its `k`, `omega` and normalization volume are compatible with an elongated
rectangular periodic box: length `2pi*c/omega` along `k`, transverse area `V/length`.
This is a **single-mode cutoff probe**, not an isotropic thermodynamic box or a
simulation of the subsequent three-dimensional IR integral. The program also
reports the much smaller displacement for a cubic box with its first mode at
this same frequency. It does not pretend the stress-box dimensions are the bulk
infrared regulator.

The classical forced solution reproduces the mean transverse field but does not
supply `P_0`, a photon-number distribution or its measurement interpretation.
The actual ground state has canonical coordinate variance `1/(2omega)`, not zero
variance of a deterministic classical wave. The numerical photon probabilities
and overlap, combined with the assumed Born rule, are the quantum distinction.
The program exposes two adverse predictions: replacing `omega-k.v` by `omega`,
and dropping the canonical factor two. It also checks the exact soft scaling of
sources at frequencies down to `1e-30` without replacing a small frequency by an
absolute numerical floor.

## 3. Number divergence, finite energy and distinct velocity sectors

Let `n=k/|k|` and integrate the two transverse polarizations with
`sum eps eps^T=I-nn^T`. Replacing the bulk mode sum by
`V int d^3k/(2pi)^3` cancels precisely the canonical `V` in `|alpha|^2`. With
`omega=c|k|`, and lower/upper **photon energy** cutoffs `lambda,Lambda`,

\[
 N=C\log(\Lambda/\lambda),\qquad
 C=\frac{q^2}{2c(2\pi)^3}\int d\Omega\sum_{r=1}^2
 \left|\frac{v_f\cdot\epsilon_r}{c-v_f\cdot n}
       -\frac{v_i\cdot\epsilon_r}{c-v_i\cdot n}\right|^2.
\]

The angular calculation uses Gauss–Legendre polar quadrature and periodic
azimuthal quadrature, applying the transverse projector directly rather than
selecting a preferred polarization. At `c=1`, for rest to velocity of magnitude
`beta`, independently integrate

\[
 \int d\Omega\frac{\beta^2(1-\cos^2\theta)}
 {(1-\beta\cos\theta)^2}
 =8\pi\left(\frac{\operatorname{atanh}\beta}{\beta}-1\right),
 \qquad C=\frac{q^2}{2\pi^2}
 \left(\frac{\operatorname{atanh}\beta}{\beta}-1\right).
\]

The CLI compares that independent formula with the actual recoil velocity,
small/medium/large subluminal beta controls, and two angular orders. It also
checks distinct non-rest velocities, swapping in/out, and the exactly vanishing
same-velocity difference. Subluminality is required: this is not a treatment of
Cherenkov emission or a collinear massless charged particle.

Free-field cloud energy is instead

\[
 E_{\rm soft}=\int_\lambda^\Lambda\omega\,dN
 =C(\Lambda-\lambda)\longrightarrow C\Lambda.
\]

For `C>0`, `N` diverges, the squared overlap of **different-velocity** clouds is
`exp(-N)->0`, and energy remains finite. The program retains `log(lambda)` for an
arbitrarily long IR sequence; underflow of `exp(log(lambda))` is used only when
computing an already negligible correction to the finite energy, never as a
mode cutoff or replacement of `log(Lambda/lambda)`. These are explicit
leading-soft continuum limits, not a proof of a thermodynamic compact spectrum.
In particular rest has no transverse current in this approximation, but its
longitudinal charge field remains: neither a rest-state pole theorem nor absence
of all charged Fock descriptions is established here.

## 4. Precisely defined detector and hard-consumer join

Condition on the selected hard Compton bin. Ignore occupations of **every
individual additional photon with energy below `Delta`**; veto any additional
photon with energy in `Delta..Lambda`. `Lambda` is below the hard scales. This is
**not** a cut on the sum of unresolved photon energies. A real detector imposing
a total missing-energy bound has a different soft function. The unrestricted
individual-photon definition neglects recoil at this leading-soft order; it is
not an exact all-orders energy-conserving event generator. Emission above
`Lambda` and finite/hard radiative terms require separate hard matching and are
not calculated by this result.

Independent coherent modes yield a vacuum/no-emission normalization
`exp(-N_total)` in the probability. Calling this the “virtual” factor means the
coherent displacement normalization and its leading IR role; it is **not** a
computed full loop correction. The complete sum over arbitrary unresolved
occupations is

\[
 e^{-N_{\rm total}}
 \prod_{\omega<\Delta}\sum_{n=0}^{\infty}
 \frac{|\alpha_k|^{2n}}{n!}
 =e^{-N_{\rm total}+N_{\rm unresolved}}
 =e^{-C\log(\Lambda/\Delta)}
 =\left(\frac\Delta\Lambda\right)^C.
\]

The executable performs the aggregate occupation sum in log space; its generous
reported cutoff is a numerical bound on a Poisson tail, not a restriction in the
detector definition. Multinomial expansion of all unresolved modes gives
`N_unresolved^n/n!`. Multiple lower regulators must give the same veto
probability. Keeping only the vacuum occupation incorrectly preserves regulator
dependence and tends to zero; omitting the coherent normalization produces an
unphysical exponentially increasing sum. Both wrong alternatives are reported.

For the same primitive Pauli hard amplitudes, average over incident spin and
polarization and sum outgoing labels:

\[
 R_{\rm tree}=\frac14\sum|M_{\rm Pauli,connected}|^2.
\]

Using actual canonical photon energies, incident group flux, outgoing wavevector
magnitude and radial energy-conservation Jacobian, the code evaluates the same
finite-spacing `d sigma_tree/d Omega_k` convention as `WilsonLink.cross_section`.
The recoil velocity in that Jacobian is again the `RotationBand.current`
expectation. It then reports

\[
 \frac{d\sigma_{\rm veto}}{d\Omega_k}\bigg|_{\rm leading\ soft}
 =\frac{d\sigma_{\rm tree}}{d\Omega_k}
  \left(\frac\Delta\Lambda\right)^C.
\]

This is a tree hard result dressed by its conditional leading-soft observable,
**not hard radiative Compton**. No unexplained measured-charge substitution is
made between the current, oscillator and hard calculation.

## 5. Gauge conservation and compact longitudinal debt

A gauge test cannot shift only a transverse spatial vector and omit the density
component. For four-polarization `(eps0,eps)` use numerator
`v.eps-eps0` with denominator `|k|(c-v.n)`. Under

\[
 (\epsilon^0,\epsilon)\mapsto(\epsilon^0,\epsilon)+z(c,n),
\]

each leg changes by `-q*z/|k|`. The difference cancels for conserved incoming and
outgoing charge. The executable checks this cancellation and deliberately
rejects unequal in/out charges and the spatial-only substitution. This soft
continuum gauge check is distinct from the finite-transfer lattice Ward identity
already supplied by the hard consumer.

The transverse coherent cloud does not solve Gauss's law for the charged matter.
The campaign already has an actual finite Pauli charged creation prototype:
physical occupations exclude the neutral parity-marker leaf, the boundary
ladder creates charge, and the quantum electric string preserves the finite
Gauss operators with charge equal to boundary outgoing flux. That is a genuine
longitudinal/string construction, not this transverse oscillator displacement.
It is not yet the Coulomb-minimized asymptotic charged state.

Compact link rotors have periodic angles and integer electric flux in the
selected elementary charge units. An arbitrary fractional shift of a link's
integer flux is not a unitary operation on that compact rotor Hilbert space:
`exp(i f theta)` is nonperiodic for noninteger `f` and is **not an exact compact
dressing**. Integer strings and superpositions of allowed configurations can
obey Gauss, but their Coulomb-minimized state must be constructed within the
actual compact/matter Hilbert space. The harmonic noncompact transverse
description here is a selected weak-field/continuum approximation. It must not
be projected backward into an exact arbitrary fractional electric string.

Outstanding joins remain consequential: Coulomb minimization, compact
thermodynamic/continuous-time photon transfer, matter feedback on electric and
magnetic normalization, charge renormalization, equality of matter and photon
speeds, infrared asymptotic dynamics beyond prescribed-current approximation,
and quantum/microscopic-law/constant input debt. A finite oscillator, convergent
angular integral, and finite-resolution leading-soft factor answer this scoped
interpretation problem without claiming those campaign obligations are solved.
