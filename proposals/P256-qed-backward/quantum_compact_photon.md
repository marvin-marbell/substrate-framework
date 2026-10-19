# Quantum compact-photon response with charged Pauli matter

`quantum_compact_photon.py` replaces a Gaussian photon pole by a **finite,
positive Lehmann measure of the actual coupled Gauss Hamiltonian**. It does not
factor the ground into matter and a photon sea. The construction is a quantum
one-plaquette response theorem and executable consumer, not a claim of a
three-dimensional photon phase or a QED scattering cross section.

## 1. Actual source, Hamiltonian and units

The model is `code_phase.CoupledSquare`: four physical edge Pauli qubits,
noncyclic integer electric fields, fixed background `(1,0,1,0)`, and the raw
Gauss basis. Its Hamiltonian, without matter or code-sector projection, is

\[
H=H_{\rm onsite}+H_{\rm transfer}
 +\frac{U_E}{2}\sum_{\ell=1}^4E_\ell^2
 -\frac K2(W+W^\dagger)+\frac\Delta2(1-S).
\]

`H_transfer` contains the existing gauge-invariant Pauli path operators and
rotor shifts, with their existing complex coefficients. `W` is precisely the
oriented loop shift already used by `CoupledSquare.operators()`: its field
increment is `s=(-1,-1,-1,+1)`. There is no electric wraparound at the cutoff.
The code rebuilds this operator and verifies that its cosine is the Hamiltonian
cosine. All spectra below are spectra of the raw Gauss matrix, including both
code sectors; the positive-sector ground bound is not used to discard states.

The observable and source are

\[
B=\frac{W-W^\dagger}{2i},\qquad H_j(t)=H-j(t)B.
\]

This is the oriented dimensionless compact magnetic loop observable, locally
`sin(phi)`, not a continuum magnetic field strength in tesla or an integrated
3D vector potential. Its orientation determines its sign, not its spectral
weights. The second reported observable is

\[
n_{\rm loop}=\frac14\sum_\ell s_\ell E_\ell,
\qquad [n_{\rm loop},W]=W.
\]

For charged configurations this coordinate can be affine/noninteger; no
neutral-vacuum identification is imposed on the full ground. Both observables
preserve Gauss because the loop field shift has zero vertex divergence.

Units are `hbar=1`: `phi`, integer edge electric fields, `n_loop`, `B` and `W`
are dimensionless; `K`, `UE`, `Delta`, path hopping, source `j`, and frequency
are energies. The Fourier-domain susceptibility has inverse-energy units.
No Euclidean pure-gauge stiffness is imported as `K` or `UE`; these are the
specified Hamiltonian coefficients. Tensor/Born dynamics and those scalar
coefficients remain model inputs.

## 2. Finite spectral response theorem

**Theorem.** Let `H` be this finite Hermitian Gauss Hamiltonian at any positive
cutoff and let `|0>` be a normalized ground eigenvector. For Hermitian `B`, an
orthonormal eigenbasis gives

\[
\epsilon_a=E_a-E_0>0,\quad
w_a=|\langle a|B|0\rangle|^2\geq0,\quad
\rho_B(d\epsilon)=\sum_{\epsilon_a>0}w_a\,
\delta_{\epsilon_a}(d\epsilon).
\]

The following causal consumer is exact on this finite domain:

\[
\chi_B(t)=i\theta(t)\langle[B(t),B(0)]\rangle
 =2\theta(t)\sum_a w_a\sin(\epsilon_a t),
\]
\[
\delta\langle B\rangle(t)=\int dt'\,\chi_B(t-t')j(t'),
\qquad
\chi_B(z)=\sum_a\frac{2\epsilon_aw_a}{\epsilon_a^2-z^2},
\quad \operatorname{Im}z>0.
\]

The sign convention is for **`-jB`** and Fourier transform `exp(+i z t)`;
using the common `-i` commutator Green function without changing its source sign
would reverse this susceptibility. At positive frequency,
`Im chi(omega+i0)=pi sum w_a delta(omega-epsilon_a)` is nonnegative. Even at a
finite positive broadening, every summand has nonnegative imaginary part for
positive real frequency. Its static value is `sum 2w_a/epsilon_a` for the
inelastic retarded response.

There are two exact sum rules:

\[
\sum_{\epsilon_a>0}w_a+w_{\rm zero,conn}
 =\langle B^2\rangle-|\langle B\rangle|^2,
\]
\[
\sum_{\epsilon_a>0}\epsilon_aw_a
 =\tfrac12\langle[B,[H,B]]\rangle.
\]

Here `w_zero,conn` is the connected weight inside a possibly degenerate ground
space. For a nondegenerate ground it vanishes. In a degenerate case the code
reports this weight and the selected eigenvector; the inelastic retarded
response is not asserted to equal a unique static ground-state Hessian.

**Proof.** Insert the complete physical eigenbasis into `B(t)B(0)` and the
reversed order. Their difference is `-2i sum w_a sin(epsilon_a t)`; the zero
energy contributions cancel. Multiplication by `i theta(t)` and the convergent
upper-half-plane Fourier integral give the stated kernel. Completeness gives
the equal-time identity including the zero-energy subspace. Expanding the
double commutator gives `2<BHB>-<B^2H>-<HB^2>`; acting `H` on the ground gives
`2 sum (E_a-E_0)w_a`. Each weight is an absolute square. These arguments use
neither a factorized sea nor harmonic/Gaussian statistics. They apply equally
to the electric-loop response.

The double commutator in the executable uses the **whole Hamiltonian**, so
matter transfer and cutoff boundary terms cannot silently disappear. Pole
selection uses nonzero `B` matrix elements, not the smallest overall gap.
The displayed rows omit weights below `1e-10` only for readability; the consumer
and both sums retain every positive-energy eigenstate. Energies below `1e-10`
are treated as numerically zero, an explicit finite diagonalization tolerance.

## 3. Deriving, not importing, the quartic correction

This calculation belongs to a clearly identified invariant domain:
zero hopping **and zero improved transfers**, occupation `(0,2)` equal to the
fixed background, and `S=+`. The full zero-transfer ground is **not** presumed
to occupy this domain: onsite energies and charged Coulomb energies can select
another occupation. The executable constructs this neutral block from actual
`sector_basis(1)` columns and checks invariance under the full Hamiltonian.

In that neutral block Gauss gives `E_ell=s_ell n`, so

\[
\frac{U_E}{2}\sum_\ell E_\ell^2=2U_E n^2,
\qquad H_{\rm loop}=2U_E n^2-K\cos\phi
\]

up to its fixed onsite constant. On the infinite rotor, `W=exp(i phi)` and
`n=-i partial_phi`. At finite cutoff `W` is a truncated shift, not a unitary
angle operator; no global finite-dimensional canonical commutator is claimed.
The weak-angle approximation is local near `phi=0`:

\[
H_{\rm loop}=-K+2U_E n^2+\frac K2\phi^2
 -\frac K{24}\phi^4+O(K\phi^6).
\]

Writing the harmonic part as `n^2/(2M)+M omega^2 phi^2/2` gives

\[
M=(4U_E)^{-1},\quad \omega_0=2\sqrt{U_EK},\quad
v=\langle\phi^2\rangle_0=\frac1{2M\omega_0}=\sqrt{U_E/K}.
\]

Set `phi=sqrt(v)(a+a^dagger)`. Applying `(a+a^dagger)^2` to `|m>` gives three
orthogonal components with coefficients
`sqrt(m(m-1))`, `2m+1`, and `sqrt((m+1)(m+2))`. Squaring their norms yields

\[
\langle m|(a+a^\dagger)^4|m\rangle
 =m(m-1)+(2m+1)^2+(m+1)(m+2)
 =3(2m^2+2m+1).
\]

Consequently

\[
\delta E_m^{(4)}=-\frac K{24}\,3(2m^2+2m+1)v^2
 =-\frac{U_E}{8}(2m^2+2m+1),
\]
\[
\boxed{\epsilon_{1}^{\rm compact}
 =2\sqrt{U_EK}-\frac{U_E}{2}
 +O\!\left(U_E\sqrt{U_E/K}\right).}
\]

In particular `delta E_0=-UE/8` and `delta E_1=-5UE/8`; their difference,
not a guessed mass correction, is `-UE/2`. Sextic first order and quartic
second order both scale as `UE*v`. Compact winding and electric cutoff effects
are separate from this local asymptotic remainder. The prediction is not an
exact identity at arbitrary cutoff/coupling. For `UE=.09,K=100/9`,
the harmonic prediction is `omega0=2,v=.09` and the leading corrected
gap is `1.955`. The [executed higher-order construction](compact_asymptotics.md)
now derives and measures the next pole and source coefficients on the same
actual invariant rotor, including explicit electric-cutoff refinement.

## 4. Genuine Gaussian alternatives and discriminating outputs

There are two different Gaussian approximations, neither silently relabeled as
the compact model:

1. Linearize both the Hamiltonian and observable: `B approx phi`, a single pole
   at `omega0` with weight `v`.
2. Keep `B=sin(phi)` but use a harmonic ground/Hamiltonian. The oscillator
   displacement identity gives, for odd `m`,
   `w_m=exp(-v) v^m/m!`, `epsilon_m=m omega0`; even weights vanish. This already
   has multiple poles, so "more than one pole" alone does **not** prove a
   non-Gaussian Hamiltonian. The executable reports the first eight odd poles
   of this alternative and their partial static sum.

The compact prediction changes the first active pole by `-UE/2` at leading
order, unlike either Gaussian Hamiltonian. Actual charged matter transfers
can further move/split active poles and redistribute positive weights. The
full consumer computes these changes directly rather than attaching an
unmodified Gaussian photon to an independent matter spectrum. It reports the
smallest overall gap, first active gap, dominant active gap, dominant fraction,
all active rows, total weight, moment and static response. Raw-embedding matter
purity and occupation probabilities diagnose ground matter/rotor correlation;
this purity is an embedding diagnostic, not an assertion of an autonomous
physical matter tensor factor after Gauss reduction.

## 5. Cutoff control and reproducible execution

The default complete run uses:

- cutoffs `2,4,6`, `K=100/9`, `UE=.09`, `Delta=3K`;
- transfer cases `(hopping,improved)=(0,0),(.6,0),(.6,.19)` and the existing
  path phases/onsite coefficients;
- neutral weak-angle comparisons `UE=.2,.1,.09`, `K=1/UE`, hence fixed
  `omega0=2` and decreasing variance, at cutoff 8;
- `Im z=.02` for a sampled retarded response, and explicit negative-time
  samples of the causal kernel.

Increasing the cutoff nests the actual Gauss bases and their Galerkin
Hamiltonians. The ground energy therefore decreases variationally. The code
embeds the smaller ground into the next physical basis and calculates
`r=||(H_large-E_small) embedded_ground||`. The spectral theorem guarantees
`dist(E_small,spec(H_large)) <= r`. This is a rigorous finite-to-finite
spectral-distance bound, **not** a guaranteed error to the infinite-cutoff
ground and not a guarantee identifying one photon-active eigenstate.

Ground cutoff-shell probability and lost raising/lowering norms expose the
nonunitary shift boundary. Changes in active gap, equal-time weight, energy
moment and static response expose response convergence, which cannot be
inferred from the ground energy alone. Small shell probability is a useful
indicator, not a rigorous complete response-error estimate. The neutral run
also reports shell probability and the quartic scaled remainder; increase
`--weak-cutoff` if shell error competes with the quartic remainder. No default
cutoff is predeclared converged.

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 /home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/quantum_compact_photon.py
```

A stricter cutoff comparison can use `--cutoffs 4 6 8 --weak-cutoff 10`.
The output is JSON; sum-rule, Hamiltonian/operator, neutral-invariance and
variational violations raise errors. The complete default CLI has naturally
exited zero; `quantum_compact_photon_receipt.json` preserves its actual
Hamiltonian, positive spectral weights, entanglement and response diagnostics.
For the improved matter transfer at cutoff 6, the dominant B-active excitation
is about `1.9895` while the first small-weight active excitation is about
`.8758`. A dominant pole is not silently identified with the smallest gap.
These are spinless charged-matter configurations in the specified Gauss
background, not the total odd-charge sectors of the later spinor consumer.

The natural next physical join is to feed this positive causal spectral
measure, with an explicitly derived source normalization, into the same
continuous angular/Pauli consumer as the matter construction. That would
replace its Gaussian finite supplier rather than append an unrelated photon
model. A continuum multi-plaquette limit, transverse propagating modes, and
the real electron-photon scattering normalization remain outside this
one-plaquette theorem.
