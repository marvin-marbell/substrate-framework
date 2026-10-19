# Full quantum charged current/contact response on the open square

`quantum_compton_response.py` uses the **actual interacting**
`SpinorPlaquette` of `quantum_spinor_plaquette.py`. It constructs a finite
matter-source two-current kernel, not an asymptotic continuum Compton
cross section or a propagating-photon S matrix. The compact photon is quantum
and participates in every Hamiltonian resolvent; the applied source is an
external, prescribed matter Peierls source. This does not solve the remaining
infinite-volume, photon external-state, or continuum scattering obligations.

## Source and Hilbert-space ownership

The default parameters are `K=.25, UE=.6, Delta=1.2, m=.8, r=.04, v=.08`;
default cases are `(R,N)=(1,7),(1,9),(2,7),(2,9)`, in the `S=+1` code sector.
No replacement CAR Hamiltonian, free sea, appended spin, or frozen rotor is
introduced. The owner supplies:

- `PauliCode.element(mask,new,word,S)`, evaluated on its actual two-word code
  projector;
- each `SpinorPlaquette.terms` entry `(source,target,coefficient,word,shifts)`;
- every allowed raw `(mask,fields)` Gauss state, its index and hard electric
  cutoff, and the complete sparse `H`, magnetic raising operator `W`, and `T`;
- `solve_low`, which obtains a ground eigenvector, constructs its **actual**
  `apply_T` partner, and probes the entire orthogonal complement for the first
  excitation rather than trusting a single-vector Krylov degeneracy count.

The raw domain fixes `N=7` or `9` in sixteen orbitals. It contains every
occupation mask compatible with Gauss and `|E_e|<=R`, including arbitrary
allowed internal particle-hole rearrangements. Rotor offsets are solved by
Gauss; the rotor loop is not restricted to its ground value. The hidden marker
is neutral and only carries the code parity; the charged boundary is `N-8`.
All response intermediates live in this complete same-`N` domain. The receipt
records its dimension, number of masks, raw-domain hash, cell-occupation range,
cutoff-shell dimension, and ground-frame shell probability eigenvalues.
Neither a small shell weight nor the two default cutoffs proves infinite-cutoff
convergence.

## Peierls differentiation of actual matrix entries

Orient the four links in cycle order `0->1->2->3->0`, with physical cell
coordinates `(0,0),(1,0),(1,1),(0,1)`. A microscopic transfer lowers or raises
rotor fields by the owner-provided shifts. Define its oriented displacement
vector `d=-shifts`. An allowed matrix entry is

\[
T_t= h_t\,\langle\mathrm{code}(m')|w_t|\mathrm{code}(m)\rangle
 |m',E+\mathrm{shifts}_t\rangle\langle m,E|.
\]

The same cutoff rejection and Gauss-index membership as the owner are used.
For a matter-only source `a` the definition is

\[
H(a)=H(0)+\sum_{t:\,d_t\ne0}(e^{i d_t\cdot a}-1)T_t,
\quad J_e=\partial_{a_e}H|_0=\sum_t i d_{te}T_t,
\quad C_{ef}=\partial_{a_e}\partial_{a_f}H|_0
            =-\sum_t d_{te}d_{tf}T_t.
\]

The program builds all four edge currents and all sixteen contacts, and checks
their Hermiticity. Actual single-link terms happen to give zero off-diagonal
edge contacts; that is derived from `shifts`, not assumed in the differentiation.
Onsite transfers have zero spatial displacement and are unaffected. Electric
energy and the code penalty are unchanged. In particular, the magnetic term
`-K(W+W†)/2` is **held fixed**. A general external compact background would also
alter magnetic terms when its curl is nonzero and would require their own
first/second derivatives. No result here is called that total background
response.

For arbitrary real planar momentum `k` and polarization `epsilon`, the edge
amplitude is the straight-link integral, not a site phase or guessed vertex:

\[
a_e(k,\epsilon)=\int_0^1\!ds\,
 \epsilon\cdot\ell_e\,e^{i k\cdot(x_e+s\ell_e)}
 =(\epsilon\cdot\ell_e)e^{i k\cdot(x_e+\ell_e/2)}
   \operatorname{sinc}(k\cdot\ell_e/2),
\]

where `sinc(z)=sin(z)/z` with its analytic zero limit. Defaults use distinct
`k_in=(.65,.2)` and `k_out=(-.25,.55)` with unit transverse polarizations.
These momenta label the applied source shape, not quantized asymptotic photon
states. Let `J_b=J[a_in]`, `J_a=J[conj(a_out)]`, and
`C_ab=C[conj(a_out),a_in]`, using bilinear, not Hermitian, differentiation.

## Complete two-source resolvent and ground poles

Write `F` for the two-column Kramers frame, `P=FF†`, `Q=1-P`, and
`D=H-E0`. At a nonzero frequency below the first measured complement gap,

\[
\begin{split}
M_{ab}(\omega)={}&F^\dagger C_{ab}F
 -F^\dagger J_a Q(D-\omega)^{-1}QJ_bF
 -F^\dagger J_b Q(D+\omega)^{-1}QJ_aF\\
 &+\frac{j_a j_b-j_b j_a}{\omega},\qquad
 j_a=F^\dagger J_aF,\quad j_b=F^\dagger J_bF.
\end{split}
\]

The last line is the **analytic ground-doublet Born term**, obtained from the
actual full same-`N` intermediate `P` denominators `-omega` and `+omega`.
It is never silently discarded. On a square with only one independent loop
current, this commutator can vanish; a measured zero is not omitted poles.
The signs correspond to contact minus the two positive-excitation resolvents.

The frequency is `.23` times the measured first excitation gap by default,
not an arbitrary spectral floor. For each sign of frequency, matrix-free CG
acts on

\[
 Q(D\mp\omega)Q+gP,
\]

with right-hand side `Q J F` and measured gap `g`. The harmless `gP` extension
makes the full-dimensional numerical operator invertible; it does **not**
replace any complement eigenvalue. Projection is applied inside the action
and after solving. There is no intermediate-state spectral truncation and no
dense Hamiltonian diagonalization.

Each solve reports actual CG status, iterations, absolute/relative residual,
and ground leakage. Coarse `rtol=1e-8` and fine `rtol=2e-11` calculations are
both performed and their kernel differences reported. Ground eigenpair,
T-partner, first-complement residuals and complement leakage come from
`solve_low`; inadequate eigen accuracy, unresolved additional ground states,
failed CG, or failed kernel convergence raises, rather than producing a
fallback answer. All numerical identities use an explicit user-visible
absolute acceptance tolerance (default `2e-6`), not a denominator floor.

Only doublet-frame invariants are published: matrix Frobenius norm, singular
values, eigenvalues and trace. The program independently repeats the
nonforward resolvent calculation in `F U` for a nontrivial unitary `U`, then
checks `M[F U]=U† M[F] U` and norm invariance. It does not interpret one chosen
basis vector as a canonical physical spin axis.

## Exact static longitudinal Ward identity

For real site potentials `chi=(0,.37,-.23,.61)`, put
`a_e=chi_target-chi_source`, and let
`G_chi=sum_c chi_c n_c`, evaluated from the actual occupation masks.
This gradient has exactly zero cycle curl. Magnetic `W` commutes with `G_chi`,
so keeping its potential unchanged is also the correct pure-gradient choice.
The hard-cutoff projector is diagonal in occupations and commutes with
`G_chi`; cutoff rejection therefore does not spoil this identity:

\[
H(t\nabla\chi)=e^{itG_\chi}H(0)e^{-itG_\chi},\quad
J_\chi=i[G_\chi,H],\quad
C_{\chi\chi}=-[G_\chi,[G_\chi,H]].
\]

The receipt reports sparse full-operator residuals for both derivative
identities, the density/magnetic commutator and gradient curl. Degenerate
ground energies imply `P J_chi P=0`. The exact complement solution is

\[
QD^{-1}QJ_\chi F=-iQG_\chi F,
\]

whose unprojected Hamiltonian residual is separately observed. Differentiating
the invariant ground energy gives the complete matrix identity

\[
 F^\dagger C_{\chi\chi}F
 -2F^\dagger J_\chi QD^{-1}QJ_\chi F=0.
\]

This is checked with the converged full-domain solves at **omega=0**, not
inferred from a finite-frequency longitudinal source. Contact and complete
pair matrices must be nontrivial for the check to discriminate.

The same Hermitian current also obeys complete closure and an energy-weighted
sum identity. The code separately constructs total, `P` and `Q` current
weights and compares their sum. It compares the complement energy moment
`(QJF)†(H-E0)(QJF)` with the actual half-double-commutator action
`(JF)†H(JF)-[(J²F)†HF+(HF)†J²F]/2`. It does not use a Lanczos quadrature as an
exact list of all intermediate states.

Two **damaged response calculations** must fail:

1. Remove contact but keep the full computed pair. The remaining Ward matrix
   is minus that pair, not zero.
2. Keep only actual ground intermediates. Their two pole terms for this one
   current are `-j_chi²/omega` and `+j_chi²/omega`; they cancel identically
   and have zero static continuation. The microscopic projected contact is
   retained and therefore does not cancel. The receipt includes the actual
   ground pole numerator norm as well as the damaged kernel norm.

The frozen control is not an assertion that the static inverse of a degenerate
ground Hamiltonian exists. It uses the analytic two-pole cancellation before
taking the static limit. Both damaged norms must exceed the observed full-Ward
error by a factor of 100 and exceed the absolute identity tolerance. If the
chosen model/source has no usable contact/pair weight, the program refuses to
claim successful discrimination.

## Reproducible parent execution

From `proposals/P256-qed-backward`:

```sh
/home/administrator/substrate-framework/.venv/bin/python quantum_compton_response.py --output quantum_compton_response_receipt.json
```

JSON is printed on stdout and, with `--output`, written identically to the
selected path. It binds the response source and all three direct scientific
supplier files by SHA-256. A smaller first scientific probe is
`--cutoffs 1 --numbers 7`; the default remains the full four-case calculation.
The parent executed all four default cases successfully. At R2, N7 and N9,
the nonforward full-kernel Frobenius norm is 4.98477793779e−5. Fine/coarse
resolvent changes are respectively 2.01e−12 and 1.99e−12, and independent
doublet-frame covariance residuals are below 1.1e−16. Removing contact or
keeping only ground intermediates leaves a longitudinal Ward residual
0.0146550558299; the full response cancels it. Independent source review
accepted the derivative identities, density solution and complete resolvent.
This is a finite quantum matter-source response, not a photon S-matrix or
continuum cross-section.
