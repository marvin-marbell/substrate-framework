# A magnetic-coupling-independent charged-gap bound on the actual compact plaquette

## Result, owner and scope

For the finite, noncyclic Gauss domain of `quantum_spinor_plaquette.py`, the
pure photon ground energy in **every integer occupation-offset block** is at
least the neutral photon ground energy. This holds for every `K>=0`, `UE>0`
and finite `R>=0` for which the offset domain is nonempty. It is not an
angle-space diamagnetic argument applied without justification to a Galerkin
cutoff: the proof explicitly keeps every physical electric-domain indicator.

Consequently, on the same fully coupled planar four-cell, four-orbital,
primitive-Pauli spinor Hamiltonian, with `M=m+2r>0` and `r,v>=0`,

`min(E7,E9)-E8 >= M + UE/2 - 8 max(r,v)`.

This removes the previous sufficient estimate's `-K` term, without changing
its Hamiltonian, prescribing the interacting photon's wavefunction, or
claiming a measured charged spectrum. All energies below are in the actual
lattice Hamiltonian's units (`hbar=a=1`); inverse temperature `t` has inverse
energy units. The parent owns integration and numerical science execution.
Only the new `compact_charge_gap.py/.md` are this worker's write surface.
This is an active issue222 theorem iteration, not canonical promotion,
campaign completion, merging or issue closure.

## Exact photon blocks, including the cutoff

For a fixed orbital occupation, the actual oriented square Gauss solver gives

`E_j=ell+a_j`, `ell in Z`, `j=0,1,2,3`,

where all `a_j` are integers. In the actual consumer, `a0=0`, and if `n_c`
is the cell's occupation count, `b=N-8`, then

`q_c=n_c-2-b delta_(c0)`,

`a=(0,q1,q1+q2,-q0)`.

The charges satisfy `sum q_c=0`. The external boundary flux is `b`; it is
not carried by the neutral parity marker. The internal physical domain is

`D_a = {ell in Z : |ell+a_j|<=R for all j}`.

Thus `D_a` is the integer intersection of four intervals, not a separately
chosen loop cutoff. On `l2(D_a)`, let `S_a` raise `ell` by one if the result
remains in `D_a`, and annihilate it otherwise. In particular it never wraps.
The exact pure block is

`H_a = (UE/2) sum_j (ell+a_j)^2 - (K/2)(S_a+S_a^dag)`.

The reference is on `D0={ell in Z: |ell|<=R}`:

`H0 = 2 UE ell^2 - (K/2)(S0+S0^dag)`, `epsilon0=inf spectrum(H0)`.

This is the actual nonlinear compact rotor in electric representation, not
a quadratic angle-space approximation.

## Positive closed-walk proof at finite electric cutoff

Fix `t>0`. Use a common measure on **integer closed nearest-neighbor walks**
on all of `Z`. A walk consists of `n>=0`, vertices
`x0,x1,...,xn=x0`, each increment equal to `+1` or `-1`, and ordered jump times
`0<tau1<...<taun<t`. Set `tau0=0`, `tau_(n+1)=t` and keep vertex `x_k` on
`[tau_k,tau_(k+1))` for `0<=k<=n`. For `n=0` this is the constant walk
at each integer starting point. The measure is counting measure over all
these vertices and starting points times

`(K/2)^n d tau1 ... d taun`.

There is **no extra `1/n!`**: the ordered simplex already supplies that
factor in its volume. There is no cyclic edge, no loop-coordinate winding
from a wrap operation, and no offset-dependent magnetic weight. A normalized
Poisson-walk representation would have a common `exp(Kt)` prefactor; the
Duhamel representation above avoids that bookkeeping entirely.

For each physical edge define the nonnegative function

`f_j(path) = exp[-(UE/2) int_0^t (x(s)+a_j)^2 ds]`

`             * product_(k=0)^n 1_{|x_k+a_j|<=R}`.

The indicators include **all vertices**, not merely the initial point or
a almost-everywhere occupation-time condition. This covers the constant
walk and every admitted jump. Jumps exactly at `0,t` or coincident jumps
form a zero-measure set and are unnecessary.

The finite-dimensional Duhamel expansion, with diagonal electric potential
and off-diagonal positive kernel `K/2`, gives exactly

`Z_a(t)=Tr exp(-t H_a) = int product_(j=0)^3 f_j d mu`.

Indeed the product potential is the defining four-edge electric energy,
and the product indicators restrict every vertex to exactly `D_a`. Every
term is nonnegative. The expansion converges: for any finite admitted
starting domain, the unrestricted number of length-`n` jump words is at
most `2^n`, the simplex volume is `t^n/n!`, and the electric exponential is
at most one. Thus each relevant trace is at most `|D| exp(Kt)`. The common
measure on all integer starts is sigma-finite, though its total mass need
not be finite. Each `f_j` is in `L4(mu)`, which is what Holder requires.

Apply Holder with all four exponents equal to four:

`Z_a(t) <= product_j [int f_j^4 d mu]^(1/4)`.

The fourth power does not change an indicator, and its potential is
`2 UE (x+a_j)^2`. The integer change of vertices `y_k=x_k+a_j` is a
bijection of the **common all-integer closed-walk space**. It preserves
nearest-neighbor steps, closure, the ordered times and their magnetic
weight. The single-edge indicators become `|y_k|<=R`. Therefore each
factor is exactly

`int f_j^4 d mu = Tr_(D0) exp(-t H0) = Z0(t)`.

It would be incorrect to begin with a measure already restricted to
`D_a`: then its single-link Holder factors would not equal the translated
neutral trace. The common measure must come first, and the four domain
restrictions must live inside the functions.

We obtain, for all `t>0`,

`Z_a(t)<=Z0(t)`.

Both domains are finite and nonempty, so
`lim_(t->infinity) [-log Z(t)/t]=inf spectrum(H)`. Hence

`inf spectrum(H_a) >= epsilon0`.

The proof includes `K=0`, one-state domains, integer translations of all
four offsets, and even a real-valued cutoff `R`: the physical coordinate
and the offsets, not the cutoff itself, supply the integer translation.
There is no exchange of a thermodynamic limit with a cutoff limit. For
integer `R->infinity`, the confining quadratic potential and bounded
nearest-neighbor hopping also permit the usual increasing-domain
Rayleigh-Ritz limit; the inequality then survives that limit.

### Unequal positive electric coefficients

The proof also works for `UE_j>0`. Put `U=sum_j UE_j`, `w_j=UE_j/U` and
use generalized Holder with exponents `1/w_j`, or replace `f_j` by
`g_j=exp[-(U/2) int (x+a_j)^2] * single-edge indicator` and integrate
`product g_j^(w_j)`. Each translated single factor is the same neutral
trace with potential `(U/2) ell^2`. Thus the identical trace and ground
comparison holds. The executable and the actual spinor charge estimate
use the equal-coefficient model; no unequal-coefficient numerical consumer
is claimed.

## From photon-block comparison to the coupled charged gap

Decompose the actual operator as

`H_N = H_onsite,N + H_photon,N + H_hop,N + (UE/2)(N-8)^2`

in the selected `S=+` code sector. A discarded `S=-` sector additionally
has the nonnegative `Delta` selector penalty. The model's existing
`Delta>2K` selector hypothesis can be retained unchanged. Neither this
photon comparison nor the charge estimate assumes commuting onsite,
photon and hopping operators.

1. **Photon lower bound on the full matter space.** `H_photon,N` is the
   direct sum of the exact `H_a` over fixed occupations. The theorem gives
   `H_photon,N >= epsilon0 I`, including superpositions and entangled
   matter/electric states. The same `epsilon0(R,UE,K)` occurs in all three
   particle-number sectors.
2. **Onsite lower bounds.** The actual planar onsite matrix has eigenvalues
   `(-M,-M,+M,+M)` per cell, where `M=m+2r`. The primitive Pauli matrix-unit
   code intertwiner gives its ordinary fixed-number exterior spectrum.
   Filling the least `N` of the sixteen eigenvalues yields `-8M` at `N=8`
   and `-7M` at either `N=7,9`. Local orbital rotations preserve each cell's
   total occupation and therefore preserve its Gauss constraint.
3. **Hopping norm without freezing the rotor.** An actual spatial link has
   singular values `|r+v|/2` twice and `|r-v|/2` twice. Their sum is
   `2 max(r,v)` for `r,v>=0`. After local singular-vector rotations, each
   two-orbital hopping channel is an off-diagonal operator between the
   occupied-empty and empty-occupied subspaces, multiplied by the same
   physical electric-shift contraction and its adjoint. Its norm is at
   most its singular value. The empty-empty/full-full subspaces contribute
   zero. All four orbital paths on a given spatial link transport the same
   physical edge flux, so the contraction is genuinely common. The
   noncyclic finite-domain compression cannot increase its norm. Triangle
   inequality over channels and the four physical links gives
   `||H_hop,N||<=8 max(r,v)`.
4. **Physical neutral variational upper bound.** Fill both negative onsite
   orbitals in each cell, in the actual code basis, and tensor this matter
   state with the **actual neutral rotor ground vector** on `D0`. Every
   occupation in its Slater expansion has cell counts `(2,2,2,2)`, so every
   loop component obeys all Gauss equations with `b=0`. A spatial hop
   changes cell counts and therefore has zero expectation. This state's
   energy expectation is exactly `-8M+epsilon0`, giving
   `E8<=-8M+epsilon0`. It is a variational state, not an assumed factorized
   interacting vacuum; its nonzero full-H residual is not a failure.

The charged lower bound and the neutral upper bound are therefore

`E_(7 or 9) >= -7M + epsilon0 + UE/2 - 8 max(r,v)`,

`E8 <= -8M + epsilon0`.

Subtracting proves

`E_(7 or 9)-E8 >= M + UE/2 - 8 max(r,v)`.

The bound is **uniform in cutoff** and independent of `K`, although the
individual ground energies and `epsilon0` certainly depend on `K`.
A nonpositive bound is inconclusive, not a no-go. Positivity proves a
finite-model charged gap; it does not prove continuum dispersion,
thermodynamic many-body stability, a photon phase or a QED scattering law.

## Weak-photon parameter family at the same matter velocity

In the bare small-angle matching, `c=sqrt(UE K)` and after `t'=v t`,
the dimensionless electric/magnetic coefficients are `UE/v,K/v`.
Choosing `c=v` and a dimensionless bare charge `q` gives

`UE=v q^2`, `K=v/q^2`.

For `v=.08`, `q=.3`, the actual input is

`UE=.0072`, `K=.888888888888...`, `Delta=2.2>2K`,

`m=.8`, `r=.04`, `M=.88`.

The proven sufficient gap is

`.88 + .0036 - .64 = .2436`,

or `3.045` in the energy units conjugate to `t'`. The old estimate is
`.2436-.888888888888...`, which is merely inconclusive. Its `-K` term was
conservative, not an incorrect magnetic Hamiltonian.

At fixed matter parameters, every `q>0` gives the family
`UE=.08 q^2`, `K=.08/q^2`, with `Delta>2K`, and sufficient gap
`.24+.04 q^2>0`. This is an algebraic family in the same finite quantum
model. The stated `c,q` are bare harmonic matching parameters, **not a
measured nonlinear compact-photon pole, continuum speed, or renormalized
charge**. Joining the weak-photon parameters to the full coupled photon
consumer remains the parent's numerical work; this worker supplies no
invented weak-photon quantum data.

## Parent CLI and exposing controls

Run from the repository root:

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/compact_charge_gap.py
```

For an additional nested actual-operator domain, without coupled ARPACK:

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/compact_charge_gap.py --cutoffs 1 2 4 8 --actual-cutoff 2
```

The default executable performs:

- Exact tiny tridiagonal spectra and heat traces at `t=.125,1,8,64` for
  **every** actual `N=7,8,9` cell-count offset pattern, with orbital-mask
  multiplicities, at nested `R=1,2,4,8`. Empty intersections are counted,
  not treated as spectra. It checks the proposed weak parameters, `K=0`,
  a stiffer rotor and a high-magnetic rotor. It verifies ground/trace
  inequalities, nested-domain energy monotonicity and integer-translation
  spectral equality. Numerical tolerances are discriminants, not rigorous
  floating-point spectral enclosures; the analytic proof supplies rigor.
- Actual `PauliCode` and `SpinorPlaquette` construction at `N=7,8,9` and
  default `R=1`, including their full Gauss/code/time-reversal checks.
  It decomposes the **actual sparse H**, verifies its pure-photon operator,
  audits every actual fixed-occupation domain against the physical
  intersection, and checks actual basis dimensions against the complete
  pattern enumeration. It reports the scalar onsite and link-singular-value
  bounds. It runs **zero coupled ground eigensolvers**.
- The physical neutral trial using the derived primitive-Pauli intertwiner,
  determinants of the actual negative onsite orbitals and the actual
  neutral tridiagonal photon ground vector. It checks its norm, onsite and
  photon eigenvector residuals, zero hopping expectation and full-H
  variational expectation, and separately labels its full-H residual.
- **Boundary omission:** in both actual charged matrices, omitting the
  boundary electric energy subtracts `UE/2` times identity and lowers the
  sufficient estimate by exactly `UE/2`. Setting boundary flux to zero
  also fails the complete charged Gauss law; these are distinct effects.
- **Wrong cyclic rotor:** at `R=1,UE=1,K=10`, true Gauss offsets
  `(0,0,1,1)` have two loop states. A cyclic raising permutation gives
  their ground energy exactly `1-10=-9`. The similarly wrapped neutral
  three-state rotor has energy `(-3-sqrt(209))/2`, approximately `-8.7284`.
  Thus the proposed comparison is actually false for this wrong wrapping.
  The two-state cyclic shift is its own adjoint, so its off-diagonal
  hopping is doubled; it is not the noncyclic compression. This control
  checks the physical boundary prescription, not just an option label.
- **Noninteger offsets:** the production function explicitly rejects them.
  A separate `K=0,R=1,t=1` half-shift calculation has trace
  `2 exp(-1/2)`, whereas the neutral integer trace is `1+2 exp(-2)`.
  They are unequal: the single-link integer-translation equality in the
  proof cannot simply be asserted for a fractional shift. This does not
  assert that every fractional shift violates the ground comparison.
- **Old constant subtraction:** the report displays the old and new
  estimates side by side and identifies the discarded common `epsilon0`
  cancellation. It does not falsely label a less sharp valid bound as
  wrong physics.

The JSON fingerprints this new consumer and the actual imported spinor
source. No numerical science execution was performed by this worker,
as required by the parent execution/ownership contract. Parent commands
above are executable consumers, not claimed observed receipts. The
non-science `--help` CLI smoke exited zero, exercising the real imports
and argument surface without constructing a scientific model. Language
intelligence was exercised with positive ast-grep 0.45.3 Python matching and
the actual Pyright 1.1.414 stdio protocol in the repository root; this new
consumer has zero final LSP diagnostics. The child has no mounted `xd://lsp`
adapter. Existing representative spinor diagnostics include SciPy-stub
argument reports and are not silently treated as clean science. The parent
owns the first-run science receipt and integration of the stronger estimate
into any shared spinor report.
