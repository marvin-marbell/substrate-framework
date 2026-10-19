# Primitive improved-Phi light band with the original heavy phase

This is a separate constructive light-law successor. It does not rename the
original bare branch or its scattering result. The original heavy Hamiltonian
and scalar potential retain `a=1,m=.2,r=.12,v=.3,improved=False`, stiffness2,
alignment.2 and colour magnetic.7. The original full-Phi phase, all64 exact
normal offsets and Gauss/Friedrichs construction remain owner dependencies.

## The actual microscopic change

At each site there are four arbitrary traceless Hermitian Cartesian matrices
`Phi_mu`, all60 coordinates. Let `P_xy=U_xj` for one link and
`P_xz=U_xj U_(x+j)j` for two. No fixed Gamma matrix is an input coefficient.
The new light onsite term is `(m+9r/2)Phi_0`. Its directed endpoint couplings are

* one link: `F_x=-r Phi_0(x)-i(v/2)Phi_j(x)`;
* two links: `F_x=(r/4)Phi_0(x)`;
* each coefficient: `exp(i sum_path A)*(F_x P+P F_y)/2`, plus its adjoint.

Every SU4 product and both real-U1 Peierls insertions are retained. For a photon
profile f, J multiplies each directed coefficient by `i sum_path f`. The mixed
contact multiplies by `-sum_path f sum_path g`; hence it includes the two
same-link and two cross-link insertions for a length-two path. The source is the
whole-group screened source with actual Cartesian frame charts, not an imported
Dirac endpoint. A range-two gauge/source phase telescopes through BOTH links.
Primitive coefficients lift using Root's unique onsite four-mode Pauli trees
and a common spatial path factor for all16 colour entries. This is an even
Pauli mechanism; exterior/CAR matrices in the native consumer are comparators.

## Which metric governs propagation

The selected Clifford field belongs to the **unchanged original heavy**
full10 convex Gram functional. It is not reselected using the changed light
band. Nor is the fullRef4 source creation Gram a kinetic metric. Directly,
`{Phi_mu,Phi_nu}=2 G_mu_nu I`, and the actual Fourier Hamiltonian satisfies

`h(p)=Phi_0[m+r sum(1-cos p_j)^2]+v sum Phi_j sin p_j`.

Thus `h(p)^2=x(p)^T G x(p) I`. The native fullPhi matrix and actual current
measure this dispersion. Source Gram only normalizes created states.
The authoritative original Q64/Arb metric trial is
`G00=.4187253908699568, Gjj=.0632221974274214`; its full10 continuum distance
is bounded by the owner gradient plus analytic quadrature remainder, divided
by strong convexity2. The all-real-shift finite-N additional alias is bounded
by the same positive polystrip factor. At selected `N=5*2^96` its positive
cap is `2^-1024`, not floating zero. Schur completion controls noncubic
cross terms and proves `E(p)>=m sqrt(Schur(G))` for every real p and retained
centre well.

The **exact** calibrated parameter is `v_star=c/sqrt(G_infinity,11)`;
`c=1130202988249777/9007199254740992` remains the physical photon speed.
The executed binary64 parameter `.4990360130634525` is calibrated to the
certified trial and pays the nonzero metric/rounding debt explicitly. This
corrects the old improved-heavy/N9 proxy `.41402766991208007`, which cannot
be used when the heavy phase remains unchanged. Trial mass is
`.12941798806502236`; the legal-volume Schur bound is exported by the CLI.

## Full-domain light and source bounds

Let rho be the global Cartesian field radius in the established HS/4 metric.
The actual endpoint row coefficient is
`C_L=m+12r+3v=3.1371080391903575<16/5`, not old `11/10`.
On the entire fixed-NL exterior factor, `||H_L u||<=2 NL C_L ||rho u||`.
For q Peierls/colour derivatives, the exact coefficient sums are obtained
from the primitive scalar paths with their length^q. For q0..q6 they obey
`Cq_new <=4*2^q*(11/10)`. This coefficient domination is not automatically a
state-moment domination: the changed neutral trial, Young/quartic baseline,
current pullthrough and source graph must use renewed constants.

Original A4 and the heavy/scalar geometry stay unchanged. On the largest raw
exterior factor use `BL=2(4V+1)(16/5)`, `C0<=8 C0_old` and `C1=2 BL/A4`.
Set `slope=BH_original+BL` and choose a dyadic radius with
`radius^3>=2 slope/A4`. The exported new CD is independently checked above
`slope*radius`, so the fresh full-domain bound retains
`Vpot_new>=Lambda A4 rho^4/2-Lambda CD_new`. This is not an unproved transfer of
old-state moments. The renewed form gives
`Crho=(2(CD_new+3)/A4)^1/4` and `CQ=(2/A4)^1/4`.
The squared-form integration-by-parts graph uses the newly enlarged
`E=2^20(105V)^2(1+2+16/5)^2` and explicitly checks its Lambda decay.
Unbounded Phi is never replaced by a global bounded Hamiltonian norm.

The changed neutral trial is bounded by
`Usharp=64V+UE V N^2/4+3UE/(2V)` relative to the SAME exact scalar minimum,
and the renewed photon form shift is `S=Usharp+1`. The changed actual
ground current obeys `||J_k Omega_new||<=32(2V+1)(16/5) Crho`.
For all complex unit-l2 coexact packets, the all-mode pullthrough weight
is `sqrt(2V/omega_min)`, `omega_min>=c/N`. Resolvent functional calculus gives
the exported new U norm, Qhalf, QU, rho/rho2/rho4, chiPi and photon graphs.
Every Q here is `H_new-E_new`, never Q_original on Omega_original.
The positive quartic form controls rho2; only the independently proved
squared form controls the rho4 OPERATOR norm. Source angular q1/q2, both
actual currents, and the full screen CY reserve fit the finite2^66048 envelope.

The selector increases the original local t and exterior ell exponents by256,
retaining the exact phase height and all64 unknown Etheta offsets. Full
exterior counting and q6 light jets enlarge old light residual constants by
at most2^9, whereas the slowest `t^-5` residual improves by2^-1280.
Thus the old **nonzero upper residual**, not the old state, remains an
upper bound. Hvac Lambda^-1/8 is separately checked below1 for chiPi.
The weighted normal/auxiliary graph debt is below2^-33024, much less than
the exact physical target UE/(V 2^40); it is counted once in the Compton join.
Source-specific spectral/current/occupation flight is rederived by that
same-law consumer, not inferred from this auxiliary allocation.

Root vacuum domination is separately explicit: compute
`B_N(Lambda)=a_N Lambda+b_N+c_N/Lambda` from the NEW scalar paths. These larger
coefficients conservatively also bound original heavy terms without changing
the heavy leading law. The owner's admissible full-Gauss trial remains valid:
`U_trial=V[300 Lambda+(132/5)Lambda^(5/4)+15 Lambda^(-1/2)+4c]`.
Its centred full60-coordinate Gaussian kills the onsite expectation; fixed
onsite singlets kill every nearest/range-two hopping expectation for legalN>=3.
The charged trial co-locates matter and the compensating static reference;
it is only a vacuum upper, never the physical prepared scattering input.
Choose `G=max(2^132096,B_N(Lambda)+U_trial+window+beta+1)` and
`Delta=B_N(Lambda)+U_trial+window+beta+G`, with beta1/16 and each nonzero spatial
Z defect coefficient1/(96V). This pays full-Phi vacuum advantage. Merely
choosing the weighted graph minimum2^132096 would not do so. All8 auxiliary
flat histories and all64 colour wells remain distinct and retained.

The source is exactly scattering19, eta18, alpha2^371, tree wordN/2+2,
fullRef4. The separate RAW profile has nativeN41, radius20/countupper68921,
wordN/2+60, CY2^64; no profile is relabelled. PH uses fullPhi parity+C and
kappa0 or explicit kappa/-kappa paired histories; timed B is not assumed
Phi-even. The scalar screen retains
`s_eta_second(0)=-(eta/2)sech(2eta)^2`.
Physical Noether is the intrinsic S plus actual orbital, static-reference,
scalar-EM and normal/loop terms, not totalJ1/2 for an arbitrary19-site source.
Coulomb `UE rho G rho`, harmonic `UE/V p^2` and the real quadratic curl remain.

The photon boundary prefactor alone has conservative dyadic exponent-2081.
The actual prepared-source Hph-half graph is2^1726, so the complete photon
Noether upper has log2-356.871483193132 and outward dyadic exponent-356.
The before-sqrtUE coefficient has log2=1936.638352857681 and is not
binary64-representable; its float field is null, not the small pre-graph
coefficient. Separately named prefactor fields omit only the prepared graph.
This remains a partial physical-Noether boundary bound, not a totalJ eigenvalue.

## The true charged gap

Let `S_j` be the actual unitary Peierls shift in any real electromagnetic
history. The improved Wilson mass is the positive operator
`m+r sum[1-(S_j+S_jdagger)/2]^2`, with BOTH two-link products. Schur completion
of the actual field Gram therefore gives a coercive mass at least
`mu=m sqrt(Schur G)` in every history. Chirality, arbitrary-temporal-background
coercivity, the positive canonical determinant and the dense-source-range
argument in `noncompact_charge_threshold.md` apply unchanged to this positive
polynomial mass. Consequently the effective all-fibre compensated
charged-minus-neutral infimum is at least mu, with **no weak-photon loss**.
The microscopic transfer loses the actual nonzero `2d+d^2/(Gnormal-Usharp)`
and the finite auxiliary `beta^2/Groot`, not an arbitrary assigned tolerance.
The exported outward rational lower is .129417988064 for both charge signs.
This is an infimum gap, not an isolated charged pole or an R3 photon eigen-ground.


## Controlled spacetime and gyromagnetic meaning

For |p|<=R the primitive Hamiltonian differs from its massive Dirac limit by
`c R^3/6+sqrt(G00)r R^4/4`, and its rest current differs by at most
`c R^2/2+sqrt(G00)r R^3`. T6 propagation costs at most six times the Hamiltonian
remainder, plus finiteN/calibration/normal transfer debts. The same exact
finite-transfer Ward identity is exercised on actual reframed current.
Because the improved Wilson term begins at fourth order, the positive-sector
quadratic Foldy operator is `(sigma.pi)^2/(2M)` with its matching magnetic
coefficient, giving **rest low-energy g=2**. This is not exact Lorentz or exact
g at all momenta; the fixed hard Compton momenta are outside cutoff.05.

## Executable evidence and limits

`improved_phi_band.py --metric METRIC --normal NORMAL --graph GRAPH --output OUT`
consumes source-bound owner receipts rather than replaying their old leaves.
It runs the literal N3 full1620 Cartesian directions/independent SU4/U1
endpoint matrix, all first/mixed-second link derivatives, fullPhi sigma/PH,
and the actual primitive Fourier/current cone. A quantum Phi/SU4-Cartan/
transverse coexact `A_y(x)` Galerkin slice has three propagation sites,
every charged7/12 CAR occupation, compensated Coulomb, kappa0 harmonic
fibre, original heavy BO potential and all4 static reference columns.
An additional2048-dimensional primitive Pauli quantum consumer evolves the
same folded field-dependent coefficients with finite penalty20 and nonzero
defect.13; auxiliary-complement occupation is measured, not projected away.
These witness UE/chi/penalty values are explicitly different representable
diagnostic parameters, not replacements of the exact selected family.
Wrong frozen Phi, nearest-only current, source-as-kinetic metric and N9 scale
controls reject. The literal scalar second curvature, full-Haar core Lie and
renewed actual photon/orbital/ref/EM boundary graphs are retained.
Hashes, natural CLI output and elapsed runtime are exported. Compton/source
flight and actual physical probability remain the same-law consumer's work;
no original probability is copied.
