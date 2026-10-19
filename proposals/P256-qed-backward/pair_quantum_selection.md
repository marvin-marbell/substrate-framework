# Pair harmonic selection on the complete native domain

## Result and scientific scope

This is a new producer and proof for the **unchanged** nearest-x two-cell
Hamiltonian of `pair_full_field_concentration.md`. All existing owners are
read-only. The producer extracts the **actual binary native coefficients**
as exact dyadic rationals, rather than silently rounding c to.32:

```
c=1441151880758559/4503599627370496,
d=1080863910568919/18014398509481984,
t=5404319552844595/36028797018963968,
kappa=2, J=3602879701896397/18014398509481984,
R=((c^4+t^4)/kappa)^(1/3), q=t^4/(c^4+t^4).
```

These are the mathematical coefficients of the executed primitive owner;
its floating-point Hessian evaluation is compared separately. In particular
the native onsite subtraction produces c=.32000000000000006. The initial
nominal-decimal certificate was materially upgraded to these exact native
values before delivery. No statement about other coefficients is inferred.

The computer-assisted global theorem is

```
C0(delta)=1/2 Tr sqrt(H120(delta)), 0<=delta<=q,
argmin C0 = {q}.
```

The square root includes every positive native Cartesian Hessian eigenvalue;
zero eigenvalues contribute zero. This theorem uses exact invariant
characteristic factors and an outward-rounded interval cover of the entire
closed classified quotient, **not** sampled spectra or an optimizer.
The endpoint is non-Clifford, with one Y block of rank zero. The preliminary
nominal-decimal evaluations were C0(0)=32.41556894769723060342 and
C0(q)=32.41349193665206038549. They are not used as proof of the actual native
law: the final receipt recomputes exact rational factors and outward decimal
interval bounds using the extracted dyadics.

The analytic variational construction below upgrades this coefficient to
**leading neutral-ground selection** for the specified finite pair law:

```
E0(Lambda)=-3R Lambda + C0(q) Lambda^(3/4) + o(Lambda^(3/4)).
```

The neutral ground-projector density concentrates on the delta=q minimum
stratum as Lambda tends to infinity with fixed positive UE4 and UE1 and fixed
finite magnetic K. This is a leading asymptotic theorem, not a computation of
the full finite-Lambda ground, not a subleading effective Hamiltonian, and
not a charged gap, physical spin, bulk photon phase or joined campaign result.
Its analytic bounds have uniform finite constants obtained by compactness;
no numerical value for the onset Lambda or the remainder constant is claimed.

## 1. The actual Hessian and its complete invariant partition

The preceding complete classification gives, up to the two full local SU4
transformations, every classical minimum:

```
a1^2=1-q+delta, a2^2=1-q-delta,
y1^2=q-delta,   y2^2=q+delta,
A=diag(a1,-a1,a2,-a2), Y=diag(y1 sigma_x,y2 sigma_x),
Phi0,L=Phi0,R=(R/c)A, Phi1,L=Phi1,R=(R/t)Y,
Phi2=Phi3=0, U=I.
```

This is the complete delta interval, including rank loss; no extra orbits
are excluded by a representation assumption. The unreduced active minimum
manifold M is the smooth compact 15-dimensional S7 bundle over Gr(2,C4)
proved in `pair_minimizer_moduli.md`. Its common SU4 orbit dimensions are
12,14,13 at Clifford/interior/rank-loss, respectively.

Put Q=I_site tensor A + tau_y tensor Y and Z=tau_x tensor A. Then
Q^2=I and [Q,Z]=0. The source convention gives hH=RQ-(dR/c)Z;
changing the sign of d merely relabels the commuting shifts. Heavy fills
**TOP4** in -dGamma(hH), while light fills **BOTTOM4** in +dGamma(hL).
The occupied-empty heavy gap on every minimum is at least

```
g=2R(1-|d|/c)>0.
```

For each colour block i, let Mi be its site-extended projector and
Wi=tau_x tensor sigma_z on that block. Joint rank-one projectors are

```
P(i,s,r)=(Mi+s Q Mi)(Mi+r Wi)/4, s,r in {-1,+1},
e(i,s,r)=s R-r(dR/c) ai.
```

They are polynomial, affine in ai,yi, with no division by yi. In particular,
they remain regular at y1=0 and at equal blocks. For any native field
coordinates v,w the exact sea Hessian is

```
Hsea[v,w] = -2 Re sum_(empty e, occupied o)
             Tr(Pe dh[v] Po dh[w])/(eo-ee).
```

This is the separated-cluster derivative, not a per-block filling ansatz.
All denominators have the displayed positive global lower bound.

Use the HS/4-orthonormal colour matrices rho_i tensor sigma_j (excluding
00), and orthonormal site symmetric/antisymmetric coordinates. Let name
`s001`, for example, mean symmetric site parity, temporal field 0,
colour matrix rho_0 sigma_1. The active60 partition is:

| Coordinate block | Positive rank | Multiplicity |
|---|---:|---:|
| s001,s031,s103,s133 | 2 | 1 |
| s003,s033,s101,s131 | 3 | 1 |
| s010,s122,a012 | 2 | 2 |
| s011,s113,a023 | 2 | 2 |
| s012,s120,a010 | 2 | 2 |
| s013,s111,a021 | 2 | 2 |
| s030,a002,a032 | 3 | 1 |
| a001,a031,a103,a133 | 4 | 1 |
| a003,a033 | 2 | 1 |
| a030 | 1 | 1 |
| a101,a131 | 2 | 1 |
| remaining antisymmetric spatial singletons | 1 each | 11 |
| s130 | 1 | 1 |
| s002,s032,s102,s132 | 0 each | 4 |

The second copies replace rho_1 by rho_2, with the paired labels permuted as
explicitly recorded in the partition producer. There are 45 positive and
15 zero active eigenvalues. Exact Pauli multiplication verifies all 1723
off-block pairs, including four nontrivial rational cancellations, and
verifies the repeated characteristic factors. This is not an approximate
sparsity graph. The independent integer computation separates the onsite,
temporal-hop and spatial-hop Pauli matrices, computes their integer trace
coefficients, then contracts the exact native c,d,t rationals. Every trace
is bounded by32768, far below int64 overflow. The other derivation uses exact
SymPy rational matrices directly.

For field Gram G and variations dv,dw the elastic Hessian is obtained by
varying the actual gradient 2 kappa G Phi; alignment adds 2J in antisymmetric
site coordinates. The source coordinate metric is HS/4, not HS or HS/2.
Exact polynomial reduction uses

```
a1^2=1-q+delta, a2^2=1-q-delta,
y1^2=q-delta, y2^2=q+delta, R^3=(c^4+t^4)/kappa.
```

Every reduced characteristic coefficient is emitted in the receipt. Newton's
identities compute it from the exact matrix powers, and coefficients beyond
the asserted rank must vanish **exactly**. This includes the two zero factors
in the first block and all single zero factors in the rank-two triples.

The quotient reduction is performed **before** multivariate rational
cancellation. A literal cancel-then-reduce dyadic run timed out without a
receipt; this algebraically identical reduction order changes that failure
route rather than reverting to nominal coefficients. The discriminating
first-block probe completed its native exact characteristic proof in23.73s.

### The unused60 are not silently removed

Let u=c^2/(kappa R), v=t^2/(kappa R). For each unused field the 15 symmetric
modes have eigenvalues 2 kappa u, 2 kappa v, and thirteen zeros. Its 15
antisymmetric modes have eigenvalues 2 kappa u+2J, 2 kappa v+2J, and thirteen
copies of2J. Thus the full unused60 contain 34 positive modes and26 quartic
soft modes. Their half-trace contribution is exactly

```
Cunused=sqrt(2kappa u)+sqrt(2kappa v)
        +sqrt(2kappa u+2J)+sqrt(2kappa v+2J)+13sqrt(2J),
```

independent of delta. Dropping them would change the full energy coefficient,
although it would not change this scalar minimizer. The whole120 rank is79,
constant over M, with41 Hessian kernel directions.

## 2. The global interval proof

Only the eleven invariant blocks above need exact characteristic algebra;
constant positive singletons and unused modes are included separately.
For a rank-two block with positive elementary invariants e1,e2,

```
Tr sqrt(Hpositive)=sqrt(e1+2sqrt(e2)).
```

For a rank-three block, t=Tr sqrt(Hpositive) is the unique root above
sqrt(e1) of

```
(t^2-e1)^2-4e2-8sqrt(e3)t=0.
```

On that range this polynomial is strictly convex, starts negative, and has
one crossing. The producer encloses the crossing with sign-certified interval
bisection and verifies its positive implicit derivative denominator. It
then differentiates the equation twice with interval arithmetic; individual
eigenvalue differentiation, including divisions at repeated eigenvalues,
is never used. The sole rank-four block has constant characteristic factors;
exact division verifies roots2J and2J+2kappa v and reduces the remaining
positive pair to the rank-two formula.

Parameterize the scalar invariant, solely for this proof, by

```
x=4theta/pi in[0,1], delta=q cos(2theta),
y1=sqrt(2q)sin(theta), y2=sqrt(2q)cos(theta).
```

All expressions are smooth at both ends. This is **not** an assumed delta
measure or a one-dimensional kinetic operator. The full coefficient is even
under theta -> -theta (the signed-Y cross blocks exchange) and is even in
delta at Clifford; therefore C0'(x)=0 exactly at x=0,1.

Interval automatic differentiation encloses C0',C0'' on closed dyadic cells.
A cell is accepted only if:

* it meets x=0 and its entire C0'' interval is positive; or
* it meets x=1 and its entire C0'' interval is negative; or
* its entire C0' interval is strictly positive.

Endpoint curvature plus the exact endpoint derivative proves the same
strict derivative sign in the adjacent open cell. The accepted cells must
join exactly from0 to1; any unresolved cell aborts without a proof receipt.
Consequently C0 is strictly increasing in x, equivalently strictly decreasing
in delta in the interior, and delta=q is the unique global minimizer.
The preliminary nominal run certified145 cells at45 decimal digits; the
native-dyadic receipt supplies the final cover. Changing the precision or
interval subdivision is not sampling: every accepted enclosure
covers its continuum cell. The receipt serializes exact dyadic bounds with
outward decimal rounding. The trusted arithmetic is mpmath's interval
implementation plus exact integer/rational algebra; this is a
computer-assisted proof, not a machine-checked proof-assistant formalization.

## 3. Coupled soft26 and uniform native paired-fibre control

For two HS/4 vectors w2,w3 orthogonal to A,Y, let the symmetric unused
perturbation be Phi_mu,L=Phi_mu,R=s_mu/sqrt(2). This is a **unit full120**
normalization when |s2|^2+|s3|^2=1. The exact restoring polynomial is

```
Vsoft(s)=kappa/4 Tr Gram(s2,s3)^2,
kappa/8 |s|^4 <= Vsoft(s) <= kappa/4 |s|^4.
```

One component therefore has coefficient **kappa/4**, not kappa. The
per-site-unit probe instead has full120 norm sqrt(2) and coefficient kappa.
The producer exercises the one-component, orthogonal two-component and
parallel two-component cases, including the Gram cross term and the two
sharp bounds. Heavy hH is unchanged; alignment remains zero.
The inherited global excess decomposition also gives, for arbitrary fields,

```
F-Fmin >= kappa/8 |Phi_unused|_Cartesian^4.
```

This follows from kappa/2 times the sum of the four squared unused diagonal
Grams and Cauchy. It is a full-domain inequality, not just a frozen soft probe.

The actual primitive section first projects the light onto the negative
local Phi0 two-plane at each site, then projects heavy onto top4 and light
onto bottom4. At a minimum the local seed is the product of the complementary
heavy-positive and light-negative two-site Slaters. Its **paired** projection
weight is exactly

```
weight=product_i ((1+ai)/2)^4
      >= ((1+sqrt(1-2q))/2)^8 >0.
```

Indeed the four restricted heavy projector eigenvalues are (1+ai)/2,
each twice; the light complementary projector has the same four principal
angles. This formula is compared to the actual normalized native4900 section.
The local Phi0 gap is at least 2(R/c)sqrt(1-2q). Together with g this proves a
smooth positive-weight native section on a uniform compact tube around all
minima, including both isotropy endpoints, and with physical U1 loop angle
sufficiently near0. The relative U1 gauge orbit need not be localized.

For any **finite** common stabilizer gSU4, the paired determinant-line
character is det(g8 restricted to heavy P) times det(g8 restricted to its
light complement), equal to det(g8)=det(gSU4)^2=1. This also covers disconnected
stabilizer components, not just Lie generators. Heavy alone is not a trivial
endpoint fibre: its stabilizer charge can be+1 while light has-1. The normal
oscillator Gaussian is invariant because the exact Hessian commutes with
isotropy. Hence its paired vacuum is an admissible scalar-isotropy state;
no angular representation cut is imposed on the Hamiltonian.

Projector derivative norms obey the separated-cluster bound
||dP|| <= const ||dh||/g. In particular the paired field Born-Huang coefficient
on M is at most

```
(sum_field ||dhH||HS^2 + sum_field ||dhL||HS^2)/(2g^2).
```

In a smaller uniform tube with both gaps at least g/2 the bound is four times
larger. The actual source derivative tensors give sum||dh||HS^2=13.854 per
species. The receipt reports this conservative bound and the actual complete
field/link/phase metrics. Smooth normalized native sections give uniformly
bounded Berry and quantum-metric terms on this tube. Field contributions
therefore cost O(sqrt(Lambda)); fixed electric link and phase fibre terms
cost O(1). A global light gap on **all** T2 is neither claimed nor needed:
the lower bound uses the full light operator norm, and the upper construction
uses only the positive-gap physical-loop tube while retaining the complete
Hamiltonian domain.

## 4. Full-domain leading lower bound, without a BO spectral cut

Write F=Vquartic+VJ-sum_top4(hH), Fmin=-3R, and h=Lambda^(-1/4).
The full heavy operator is bounded below pointwise by Lambda F. There are
explicit global tail constants, uniform over **every** field, SU4 link and
both compact phases. Put rho=|u|_Cartesian and

```
a=kappa/16, b=4sqrt(max(c^2+d^2,t^2)),
r0=max(1,(2b/a)^(1/3)).
F>=a rho^4-b rho,  ||HL||<=b rho,
rho<=r0+(2/a)(F+3R).
```

The quartic inequality follows from Tr G^2 >= (Tr G)^2/4 at each site and
the two-site Cauchy inequality. Endpoint averaging and Hermitian cross-term
cancellation give ||h||HS^2 <=4 max(c^2+d^2,t^2)rho^2; each light4 Slater
energy has magnitude at most2||h||HS. The U1 path mixture has modulus<=1.
For rho>=r0, F+3R>=a rho^4/2; the last inequality follows using rho>=1,
while rho<r0 is immediate. Thus B0=b r0 and B1=2b/a in
||HL||<=B0+B1(F-Fmin). These are constructive whole-tail inequalities, not
near-minimum gap assumptions. Their constants are interval-enclosed by the
CLI. The exact complete matrix Hamiltonian obeys, after dropping only
positive link/phase kinetic terms,

```
H >= Lambda Fmin -B0-|K|
     +sqrt(Lambda)/2 (-Delta_u)
     +(Lambda-B1)(F-Fmin).
```

The same scalar lower bound holds for arbitrary native4900 vector-valued
wavefunctions and for their exact Gauss subspace, by the norm/diamagnetic
inequality. For each fixed U, rotating the right fields by U preserves the
Cartesian measure and kinetic form and converts F to the U=I potential.
No matter-band, angular or normal truncation is used in this inequality.

Here is the needed **uniform degenerate harmonic lower construction**.
The exact constant rank79 gives a smooth positive normal bundle N79 over
M. The CLI also certifies a strictly positive **uniform** least-eigenvalue
lower bound: each PSD rank-r block satisfies lambda_min>=e_r/e1^(r-1), which
is interval-enclosed on every accepted continuum cell, including rank loss.
Constant singletons and unused modes are included in this normal-gap bound.
Its complementary kernel splits as TM15 plus the explicit symmetric unused
E26. In local orthonormal bundle charts write u=m+s+n.
The implicit function theorem in N79 solves

```
partial_n F(m+s+N(m,s))=0,
N(m,0)=0, D_s N(m,0)=0.
```

This holds uniformly on finitely many compact charts. Put z=n-N(m,s).
The shifted potential is

```
F=Fmin+W(m,s)+1/2 z Hnormal(m,s) z+O(|z|^3),
W>=0, Hnormal(m,s)=Hpositive(m)+O(|s|).
```

The global unused-field quartic gives W>=c|s|^4: the unused projection is
s+O(|s|^2), so shrinking the common chart makes its norm at least |s|/2.
Positive normal curvature gives, on the tube,
F-Fmin>=c(|z|^2+|s|^4), and hence F-Fmin>=c dist(u,M)^4.
Outside a fixed compact tube coercivity and absence of other zeros supply
a positive gap. These are uniform statements, not fitted local coefficients.

Localize with bounded-overlap ambient Cartesian IMS cells of radius epsilon
covering M, plus the outside tube. In the shifted coordinates inside each
cell the metric differs from its orthonormal value by O(epsilon), the positive
normal Hessian differs from Hpositive(m) by O(epsilon), and the nonnegative
valley term W can be dropped. The **79-dimensional** oscillator inequality
then gives

```
(-h^2 Delta/2 + F-Fmin)_cell
 >= h [C0(m)-C epsilon] - C h^2/epsilon^2.
```

One may extend each Dirichlet-localized cell function by zero when applying
the oscillator inequality. Tangent and soft derivatives remain positive;
they are not replaced by a quotient kinetic term. Jacobian corrections are
bounded in the same compact charts. The outside potential is at least
c epsilon^4. Choose epsilon=h^(1/5); it is larger than the quantum soft width,
and epsilon^4>>h, while all IMS and metric debts are o(h).
This proves

```
liminf (E0+3R Lambda)/Lambda^(3/4) >= min_M C0=C0(q).
```

For example the inside coefficient debt scales as Lambda^(7/10) and the
Cartesian IMS debt as Lambda^(3/5); only existence of uniform constants is
used. The unscaled light and magnetic debts do not affect this limit.

## 5. Matching exact-Gauss upper construction on the endpoint orbit

The selected common SU4 orbit is a smooth compact13-dimensional embedded
orbit, although its quotient is an isotropy endpoint. Its ambient field
normal space has79 positive and28 kernel directions: **26 unused soft plus
two active tangent components**. In its genuine equivariant tubular charts,
put the invariant79-mode Gaussian with covariance sqrt(Hpositive) in the
positive directions, and a smooth radial compact profile of width h^(1/3)
in all28 kernel directions. Transport these profiles along the orbit and
use its actual tubular Jacobian. Isotropy invariance of the Gaussian and the
radial kernel profile makes this a well-defined real common-SU4-invariant
amplitude f_h. This is not a delta-line wavefunction or a choice of flat
quotient measure; both rank-loss tangent components are present.

Extend to every link by
f_h(uL,uR,U)=f_h(uL,Ad_U uR,I). Multiply by a fixed smooth function of the
**physical** loop A0+A1 supported in the positive light-gap tube, not by a
function localizing its gauge difference. Finally multiply by the actual
smooth normalized primitive paired section. Its exact all30 colour and both
EM covariance identities make this wavefunction satisfy the full Gauss law.
The electromagnetic background is the owner's light n_site-2; heavy has no
EM Peierls phase. The global EM stabilizer is trivial on light number4.
Independent local SU4 rotations and arbitrary relative U1 gauge phases are
retained. Haar and both-circle measures are unchanged.

The normalized real amplitude times a normalized fibre obeys

```
|D(f psi)|^2=|Df|^2+f^2|Dpsi|^2,
```

because Re<psi,Dpsi>=0. Thus Berry and Born-Huang contributions enter through
the bounded native fibre derivative term, O(sqrt(Lambda)) in fields, not a
spurious singular quotient connection. The link derivative of f_h is bounded
by C|uR||D_field f_h|, giving fixed UE4 kinetic O(h^-1)=O(Lambda^(1/4)).
The loop-profile and EM fibre energies are O(1).

Every purely kernel cubic Taylor coefficient is zero: a nonzero cubic on a
Hessian-kernel line would violate the global minimum under sign reversal.
Uniform Taylor bounds therefore have the anisotropic form

```
|F-Fmin-1/2 n H n|
 <= C(|n|^3+|n|^2|s|+|n||s|^2+|s|^4).
```

For |n|~h^(1/2), |s|~h^(1/3), every right-hand term is o(h); the largest
absolute mixed bound is h^(7/6). Kernel kinetic h^2/h^(2/3)=h^(4/3) and pure
soft quartic h^(4/3) are also o(h), corresponding to Lambda^(2/3).
Compact Gaussian truncation tails and smooth tubular metric debts are o(h).
Consequently the exact admissible packet has energy

```
-3R Lambda + C0(q) Lambda^(3/4) + o(Lambda^(3/4)).
```

Together with the lower bound this proves the leading full-domain asymptotic.
The displayed absolute upper Taylor debt would allow an unspecified
O(Lambda^(17/24)) bound after fixing all chart constants; no finite-Lambda
constant or crossover is numerically certified here.

Complete native135 principal coordinates have cometric
I120 plus eta DD^T after gauge fixing, eta=2UE4/sqrt(Lambda). The inherited
exact principal bound C0<=C_eta<=sqrt(1+eta||D||^2)C0 is uniform since
||D||^2<=4(u+v). This agrees with the full-domain construction: finite-link
principal corrections cannot change the Lambda^(3/4) selector. All79
positive oscillators remain after the full mass-normalized Gauss quotient.
Full135 flat dimensions are56; subtracting gauge ranks27,29,28 yields
29,27,28 physical linear kernel components at Clifford/interior/endpoint.
These counts are not replaced by one-dimensional kinetics at either end.

The same IMS lower bound retaining C0(m), together with compact coercivity
and the matching upper bound, shows that any fixed closed set of minimum
orbits with delta bounded away from q has ground-projector mass tending to
zero: its strict continuous coefficient gap cannot fit the o(Lambda^(3/4))
energy debt. This is gauge-invariant density concentration, not selection of
a gauge orientation or a claim of a unique finite-Lambda vector.

## Verification, changed decision and remaining obligations

The default consumer recomputes exact blocks, all off-block cancellations,
constant characteristic factors, the full closed interval cover, and native
full120 comparisons at0,.013,.37,.999,1. Clifford/interior/endpoint also run
actual135 mixed Gauss principal controls, native4900 all30 colour and both EM
identities, isotropy of the paired normal vacuum, and complete Born-Huang
metrics. A separately gauge-rotated endpoint with a genuinely noncentral
link and physical loop angle.056 runs adverse full-Cartesian Gauss controls.

The wrong-source comparison omits the commuting d term while keeping c,t
and the same classical minimum fields. It must change the actual full120
Hessian; commuting with Q does **not** justify deleting d from virtual
occupied-empty denominators. The unit-full120 soft probe also rejects the
per-site coefficient error by a factor4. These are actual native adverse
controls, not wording assertions.

The selection decision is now **rank-loss non-Cliff at leading neutral quantum
order**, rather than a sampled prediction or an imported unique Gamma frame.
Remaining exact obligations are the subleading coupled soft/tangent and
compact-phase spectral problem, a global subleading BO operator through any
light gap loss, and explicit finite-Lambda error constants. None is needed
for the leading variational theorem above. Charged excitations, genuine
mechanical spin, the thermodynamic/bulk quantum photon phase and the joined
campaign are still open; the different three-axis bulk geometry is not
invalidated by this finite-pair selector.
No commit, merge or issue closure is authorized by this result.

Run from the repository root:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/pair_quantum_selection.py \
  --output proposals/P256-qed-backward/pair_quantum_selection_receipt.json
```

`--interval-only` omits native finite probes, not exact algebra or the continuum
certificate. It is not the parent's complete acceptance command. Every own
producer/proof/probe and inherited source is SHA256-bound; an intervening
change aborts. The two uniquely named algebra/partition probes are independent
runnable exact-arithmetic scientific controls and dependencies of the CLI.
The exploratory numerical block-finding probe is not retained. Child LSP is
not mounted; actual positive Python AST navigation worked, and parent owns
final native diagnostics and integration. No test suite, build, linter or
formatter is used as a scientific proof.
