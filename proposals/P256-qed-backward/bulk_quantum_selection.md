# A different local heavy law with full quantum 3D field selection

## Decision, failure and explicit changed law

This is **not** a new proof for the original itinerant-heavy microscopic law.
The actual periodic `N=3` source, with `a=1,m=.2,r=.12,v=.3,kappa=2`, has
uniform Clifford identity-link minimum `-.5594484493551865` per site. Allowed
central pi holonomies on all three seams have both alignment and colour
plaquette cost zero, but their source-reoptimized minimum is
`-.5644728806787747`. Thus arbitrarily large positive alignment and plaquette
coefficients do not justify setting all torons to identity. This falsifies that
identity-link global supplier, **not** Clifford selection or a thermodynamic
original-law construction. The producer also actually optimizes a noncentral
Cartan holonomy against a fixed reducible field, evaluates its native full-box
matrix and tests its filling improvement. Neither control uses an open cube.

The materially changed design below makes the **local** Gram theorem apply
without a flat-link or spatial-uniformity hypothesis. It adds matter, not a
Clifford potential, determinant sign prescription, Gram budget or field cut.
The added representations and large-scale trajectory are microscopic debts,
not facts about the original primitive bosonic statistics supplier.

Let `V=N^3,N>=3`, `C[x,mu,a]` range over all `60V` real coordinates and
`Phi[x,mu]=sum_a C[x,mu,a] B[a]`, where `Tr B[a]B[b]=4 delta_ab` is the
executing `sea_selection.matrix_unit_basis`. Retain all `3V` independent SU4
links and all `3V` compact U1 phases. The original nearest-axis source is
extracted by `local_orbital_stability.scalar_paths` from `RotationBand.couplings`:

```
c0=.56;  s[a,0]=-.06; s[a,a+1]=-.15i.
F[x,a]=sum_mu s[a,mu] Phi[x,mu].
hH[x,x]=c0 Phi[x,0],
hH[x,y]=(F[x,a] U[x,a]+U[x,a] F[y,a])/2, y=x+e_a,
hL[x,y]=exp(i A[x,a]) hH[x,y],
```

with every adjoint included. The producer checks the identity-link matrix
against the read-only native `local_hamiltonian`, and checks arbitrary local
colour covariance of the full link/field/phase endpoint matrix. Original
heavy/light occupation remains the **whole** `NH=NL=2V`, not two per site or
momentum. The original heavy sign is minus and the light sign is plus.

Freeze 27 *local flavour coefficients*, not spatial momenta of the physical
box, by sampling that same actual source at
`k=2pi(n1,n2,n3)/3`, `n_i=0,1,2`:

```
x[f]=(m+r sum_a(1-cos k_a), v sin k_1, v sin k_2, v sin k_3), f=1..F, F=27.
q[x,f]=sum_mu x[f,mu] Phi[x,mu].
```

For each `f` add a fundamental and an antifundamental **EM-neutral** heavy CAR
flavour. The local matrices are `q` and `q^T`; the latter transforms by
`g* q^T g^T`, so the anti representation is genuine. Fill the entire added
`8FV`-mode sector at total `Nsel=4FV`; no occupation of any flavour/site block is
prescribed. These additional CAR representations are explicit assumptions;
this note does not derive them from the original qudit/Pauli marker.

For every finite `Lambda>=1` the full physical neutral Hamiltonian is

```
H_Lambda = sqrt(Lambda)/2 sum_xmu_a p_C^2
         + u4 sum_links Casimir_SU4 + u1/2 sum_links p_A^2
         + K1 sum_plaquettes(1-cos curl A)
         + Lambda [ kappa/2 sum_x ||G[x]||_F^2 + J W/8 + K4 P
                    - dGamma( direct_sum_xf(q[x,f],q[x,f]^T) )/(2F) ]
         - dGamma(hH) + dGamma(hL),
G[x,mu,nu]=Tr(Phi[x,mu]Phi[x,nu])/4,
W=sum_xa ||Phi[x]-U[x,a]Phi[x+e_a]U[x,a]^dagger||_HS^2,
P=sum_plaquettes (4-ReTr U_plaquette)=1/2 sum ||I-U_plaquette||_HS^2.
```

Declared coefficients: `kappa=2,J=.2,K4=.7,u4=.3,u1=.4,K1=.7`. Field kinetic,
quartic, selector, alignment and colour-plaquette powers are as displayed.
The original itinerant heavy coefficient is now **1**, not Lambda; the new
on-site selector carries the large heavy scale. Nothing else is silently
relabelled. All colour/EM Gauss constraints act, all electric operators are
positive, and all Fock states in the three aggregate number sectors remain.

## Exact global lower bound, arbitrary spectra and aggregate filling

For a traceless Hermitian four-by-four block `q`, the sum of **any** subset of
its eigenvalues is at most its positive eigenvalue sum. Trace zero gives
`sum positive=Tr|q|/2`; HS Cauchy gives

```
sum positive <= sqrt(Tr q^2)=2 sqrt(x^T G x).
```

Apply this to every block of both colour representations. It bounds the
largest `4FV` eigenvalue sum even when optimal occupation transfers among
sites/flavours. Therefore the actual selector operator obeys

```
Vquartic - dGamma(q_all)/(2F) >= sum_x F_local(G[x]),
F_local(G)=kappa/2 ||G||_F^2 - 2/F sum_f sqrt(x[f]^T G x[f]).
```

This is an inequality, **not** replacing the physical energy by `Tr|q|`,
and not a per-block filling hypothesis. The executing consumer explicitly
sorts all modes. `diag(5,1,1,-7)` in the temporal field supplies an adverse
case where forcing two per flavour gives a different physical energy.

`F_local` is kappa-strongly convex on the full PSD cone. Its unique exact
minimizer `Gstar` is diagonal `diag(g0,gS,gS,gS)` by the actual cubic sampling
symmetry. Both scales are **analytically positive**: on the `gS=0,g0>0` boundary
the spatial inward derivative is strictly negative because the actual source
has nonzero spatial sin coefficients; on the `g0=0` boundary the sample at k0
has a temporal square-root term with strictly negative infinite inward
derivative. The all-zero point is not a minimum either. Thus no numerical
rank decision is needed to infer full rank. The source rank10 has the existing
exact witness determinant `8 m^2 mu^3 |t|^15>0`, using k0, the six signed axial
momenta and the three positive planar sums, with `mu=.38,t=.3 sqrt(3)/2`.
All ten witnesses belong to the actual 3-grid sample.

The existing `soft_budget.select_gram` solver supplies floating approximations,
**not** an interval endpoint. Its full ten-component gradient and the quadratic
design rank and analytic witness are recomputed. Analytically use the exact
minimizer. Numerically `g0=.42042038805775556`, `gS=.05694884768250374`.
At every configuration

```
F_local(G) >= F_local(Gstar)+kappa/2 ||G-Gstar||_F^2.
```

Let `slack(q)` be the nonnegative difference between the actual aggregate
selector minimum plus quartic and `sum F_local(G[x])`. Define

```
S = kappa/2 sum_x ||G[x]-Gstar||_F^2 + slack(q) + J W/8 + K4 P.
```

Define the actual nonnegative fibre operator
`Bsel=-dGamma(q_all)/(2F)-Esel_min(q)`, where `Esel_min` is the true aggregate
CAR minimum, not a per-block replacement. The leading operator is at least
`V F_local(Gstar)+S+Bsel` on the **entire** Hilbert space. Uniform Clifford
fields attain equality with aggregate half occupation, and any central seam
torons do too. This is a true full-domain lower bound; no nonuniform field,
link, or Fock state is first excluded.

At `S=0`, equality in every local HS bound forces `q[x,f]^2` scalar. The
rank-10 quadratic design then forces
`{Phi[x,mu],Phi[x,nu]}=2 Gstar[mu,nu] I` for every site. This inference uses the
actual sampled coefficients, not an imposed Clifford identity. Positive `J`
makes fields parallel under every colour link; positive `K4` makes every
colour plaquette identity. A full-rank complex four-generator Clifford frame
has scalar commutant, so all colour holonomies are central. On the periodic
connected box the zero set is locally SU4-gauge-equivalent to one uniform
frame with **all 64** `Z4^3` toron choices retained. No local positive term here
fixes them to I. A flat noncentral toron has zero plaquette cost but strictly
positive alignment on an irreducible frame. A centre-flux plaquette has zero
alignment but positive `P`. Both are actual controls in the producer.
All U1 configurations remain in this leading zero set.

## Full physical Gauss trial: not an assumed tensor singlet

Use on each site an all-60-coordinate ball of radius `r` about the selected
frame, and on each colour link an exponential 15-coordinate ball of radius
`r` about identity. Their normalized Euclidean amplitudes are
`f_r(z) proportional to (1-|z|^2/r^2)^2_+`; the colour density divides by the
exponential Haar Jacobian. All EM phases have normalized constant Haar
amplitude. These are trial envelopes only, **not cuts on the Hilbert space**.

Gauge-average the *normalized product density* over independent site SU4
transformations, and take its square root. Its normalization is exactly one,
its support is the full gauge tube and its scalar part is gauge invariant.
For any gauge-invariant potential its expectation equals the ungauged product
expectation. Fisher convexity bounds its kinetic cost by that of the original
product density: averaging a density avoids the unproved norm/cancellation
bound of averaging an amplitude. The same statement applies to the gauge-
covariant combined field and left/right electric differential metric.

The matter section is concrete and phase-free. For each positive-rank-2
projector `Pplus(q)` put `Pi=wedge^2 Pplus(q)`, a rank-one six-by-six Slater
projector, and vectorize `Pi`. It is normalized in the **actual**
`wedge^2 C4 tensor wedge^2 conjugate(C4)` sector and transforms as

```
Pi(g q g^dagger)=wedge^2(g) Pi(q) wedge^2(g)^dagger.
```

Thus its derivative under every local colour generator is the *sum of the
actual fundamental and antifundamental CAR charges*, cancelled by the
Cartesian adjoint field derivative. It is not a coordinate-independent
assumed singlet. The producer builds the genuine 36-state CAR matrix and
checks the spectral eigenstate, colour covariance and all-generator
differential Gauss identities; using `q` rather than `q^T` in the anti
Hamiltonian is a deliberately rejected control.

For the original two-heavy/two-light local trial, use `Pi(Phi0) epsilon`,
where `epsilon` is the exterior-square Levi-Civita pairing. It transforms as
`wedge^2(g) [Pi epsilon] wedge^2(g)^T`, and is an actual complementary
heavy-positive/light-negative CAR state. Its local occupancy is exactly
`nL=2`, so `nL-background2=0` at **each** site. Added selector charges are zero;
the scalar phase envelope is constant. Hence every EM Gauss operator also
annihilates the trial. Colour links supply the appropriate endpoint terms in
the invariant density; the matter section has no link derivatives. This
constructs a full primitive-Gauss physical state without a normal bundle,
angular Fourier, electromagnetic, or light-sector truncation. Restricting
occupations in a variational trial is allowed; restricting the operator domain
would not be.

The source sign gaps remain separated throughout the field balls if
`r<=min(.01,min_f rho_f/(8 max_f|x_f|),sqrt(g0)/8)`, where
`rho_f=sqrt(x_f^T Gstar x_f)`. The projector derivative gives the conservative
all-60 sum

```
sum_a,mu ||d(Pi_f)/dC_mu_a||^2 <= 960 |x_f|^2/rho_f^2;
sum_a ||d(Pi_0 epsilon)/dC_0_a||^2 <= 960/g0.
```

Products have no cross Berry term: vectorized Hermitian rank-one projectors
have zero inner product with their derivative, and epsilon is constant. Thus
`M=V/2 [960 sum_f |x_f|^2/rho_f^2+960/g0]` bounds the actual section contribution
to the field kinetic energy. The actual producer sums finite-difference costs
over all60 field directions for all27 paired flavours and the original
complementary state, at a **non-Clifford perturbed** trial field; all15
differential Gauss generators are also exercised there. These checks account
for the moving section, rather than charging only the scalar envelope.

For a d-ball the exact bump Fisher is `d(d+8)/(3r^2)`: d60 gives 1360 and d15
gives 115. On the SU4 chart with `Tr T_a T_b=delta_ab/2`, `r<=.01`, the inverse
metric is bounded by4 and `|grad log HaarJacobian|<=100r` (the product of the
six positive-root sinc-squares suffices). Using `|a+b|^2<=2|a|^2+2|b|^2`
bounds colour electric Fisher by `(920+20000 r0^4)/r^2` per link.
The producer uses `A=680V` and `B=u4 3V(920+20000r0^4)`. These are intentionally
conservative analytic costs, not a sampled estimate of the quantum ground.

## Retained itinerant matter and exact-ground concentration

Let `Z=sum_x TrG[x]`, `D=sum_x ||G[x]-Gstar||^2`, and extract
`b=|c0|+2 sum_paths |s_path|` from the actual scalar paths. Endpoint averaging,
unitary colour/EM links and HS triangle give
`||hH||_HS,||hL||_HS <=2b sqrt(Z)`. Ky Fan on each original aggregate-half
sector therefore gives

```
-dGamma(hH)+dGamma(hL) >= -4b sqrt(2V Z),
Z <= V TrGstar + 2 sqrt(V D),
>= -B0-B1 D^(1/4),
B0=4b sqrt(2V^2 TrGstar), B1=8b V^(3/4).
```

Young absorption with half the Gram remainder gives

```
B1 D^(1/4) <= Lambda kappa D/4 + C Lambda^(-1/3),
C=(3/4) B1^(4/3) kappa^(-1/3),
H_Lambda >= Lambda V F_local(Gstar) + Lambda (S+Bsel)/2 -B0-C Lambda^(-1/3).
```

Photon and kinetic terms are positive. There is no lost light-sea term.
The producer evaluates both retained native whole-half seas at adverse
nonuniform SU4/U1 points and tests this lower bound.

The packet potential has an explicit uniform upper
`leading<=V F_local(Gstar)+L r`. The displayed producer computes `L` from
quartic/selector norm changes, `W<=3V[4+2sqrt(2)(sqrt(TrGstar)+r0)]^2 r^2`,
and `P<=12V r^2`. The original matter cost is bounded by
`B0+B1 V^(1/4) sqrt(2r(2sqrt(TrGstar)+r))`; the photon cost is at most `6VK1`.
Consequently

```
E0/Lambda - V F_local(Gstar)
 <= Lr + A/(sqrt(Lambda)r^2) + M/sqrt(Lambda) + B/(Lambda r^2)
    + [B0+B1 V^(1/4)sqrt(2r(2sqrt(TrGstar)+r))+6VK1]/Lambda.
```

Choosing `r=min(r0,(2A/L)^(1/3) Lambda^(-1/6))` and comparing with the physical
operator lower bound proves, for **any exact neutral ground**,
`<S+Bsel>=O(Lambda^(-1/6))` at fixed finite N. Thus the **actual** added Fock
excitation above its coordinate-dependent aggregate minimum concentrates too;
it is not merely a frozen Slater assertion. Quartic coercivity and compact
links give concentration in probability on the full zero set described above:
on a compact sublevel away from that set the continuous `S` has positive
minimum, while Gram control removes the field tail. This proves a limiting
full-rank covariantly uniform Clifford supplier for this changed quantum law,
not merely concentration on a fixed frame or a Hessian classification.

Floating receipts evaluate exact analytic formulas at `Ghat`. They report
full-gradient residual and its strong-convex energy-debt formula; they are not
interval enclosures of `Gstar`, and do not claim to solve a quantum eigenstate.

## Carrier, photon, spin and volume limits

At the selected classical limiting shape the original source light band has
`mass=m sqrt(g0)`, `r_eff=r sqrt(g0)`, `velocity=v sqrt(gS)`. These are frozen
carrier parameters, not quantum pole parameters. The new selector is EM
neutral, so it does not insert a new electromagnetic current; the actual
retained light endpoint matrix has the compact link-current/contact obtained
by differentiating its phase. The actual nonuniform SU4/U1 consumer exercises
both first and second full U1 Ward identities, finite phase differences and
a finite local gauge transformation; a deleted-contact control is rejected.
These operator checks give no photon/current quantum pole theorem.
All centre torons and dynamical U1 fields remain.

Concentration alone does **not** prove an adiabatic effective Hamiltonian,
normal/colour gap uniform in volume, rotor photon phase, charged lower gap,
physical mechanical spin, scattering, Lorentz matching or a joined electron-
photon construction. The constants grow with volume and the extra27 paired
representations; no order of N and Lambda limits is exchanged. Original-law
thermodynamic toron irrelevance is a different supplier owned elsewhere.

## Observed discriminating runs

Both actual physical lattices N3 and N4 were exercised; the source-sampled
selector law stays F27 in both, rather than secretly tracking the physical
box momentum blocks. Native identity-link endpoint errors were exactly0.
For N3 the added aggregate consumer filled 2916 of5832 modes. The adverse
temporal field gave ranks1 and3, with an actual forced-two energy error2.88.
Nonuniform field/link/phase controls at coordinate scales .001,.03,.2,.7
satisfied the full-domain bound and actual local colour covariance.

The original fixed Cartan field improved under noncentral torons by
.017800298446891877 per site at N3 and .001159011714995023 at N4; its W/P were
below7e-31. This proves the adverse finite-torus filling mechanism is present,
not that it beats the original optimized Clifford branch. The original
identity-versus-pi Clifford advantage shrank from .005024431323588541 at N3
to .0007325596594475181 at N4; no extrapolated phase claim follows.

For the changed law at N3, a flat noncentral seam gave W7.4326814650275335
with P0; a single centre-flux link gave W0 and P32. Allowed centre torons
kept the exact leading energy -15.105108132590031 unchanged.
At a non-Clifford trial field inside the radius.001 tube, the actual
all60/all27-flavour section metric sum was702.7481475161779, below the
derived84340.7724003395 upper. All15 selector/original differential Gauss
residuals were below1.7e-9, EM Gauss exactly0; the wrong anti transpose
failed by .17793389031779028. The actual light current/contact Ward residuals
were below4e-17; omitting the contact failed by .15686312332078534.

The **formal-bound evaluation**, not a computed quantum eigenstate, gives
`<S+Bsel><=2.260881390286267e-7` at N3,Lambda1e60, and
`<=5.359126258456336e-7` at N4,Lambda1e60. Every receipt identifies its exact
producer/proof/source bytes and explicitly labels the non-interval status.

Run the deterministic actual-model consumer, including nonuniform fields,
noncentral/central holonomies, centre flux, aggregate occupation, physical
Gauss section and full quantum-bound controls:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/bulk_quantum_selection.py --size 3 \
  --output proposals/P256-qed-backward/bulk_quantum_selection_receipt.json
```

Child LSP was not mounted (`xd://lsp` unavailable); the actual project root
marker and positive Python structural AST query were inspected. Parent owns
native typing and composed integration. No shared source/proof was edited.
