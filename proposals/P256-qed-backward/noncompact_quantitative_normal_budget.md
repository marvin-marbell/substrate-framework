# Computable original-law normal and entire-photon transfer budget

Governing plan `issue-222-synthesis-continuation-20261006`, assignment revision
`168949e5f2df57656471db8e62d4ae809ab290b1d09ea28c620f52349324a549`, general template
`89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`.
Work item campaign, outcome/verification. The new pair does not modify the locked
`original_slow_kinetic_bo` or `original_normal_gap_bound` pairs.

## 1. Outcome, inputs and scientific boundary

The improvement over those original proofs is a rational, executable set of
normal, heavy-projector, local scalar, moment, current and transfer constants.
The physical normal dimension is **90N^3**, with 45 soft uniform normals and
90N^3−45 stiff normals, after all15N^3 Gauss equations. Heavy is the entire
fixed2N^3 cluster of the native endpoint matrix, not independently assigned
momentum occupations. Light is the whole raw fixed-NL CAR fibre, including the
opposite bar4/4 reference. All64 unknown exact scalar offsets stay in Heff.

The new computable local constants **do not infer global phase from N25**.
`GramBounds.native_a1()` uses NativeMetricAdmission's actual all-ten Arb128,
all-real-shift enclosure at N>=25. It widens that enclosure to
`(1/20)I <= Gtheta <= (1/2)I`. A different spacing requires its actual Gram
bounds and provenance. The independent scalar-phase supplier owns global N0,
flat reducible/noncentral separation and full-domain distance coercivity.
`choose_lambda` without its packet explicitly returns conditional local
admission. This is not a claimed microscopic or campaign completion.

Only integers and rational arithmetic supply proof bounds. Floating native
checks below are discriminating consumers, not outward Hessian certificates.

## 2. A quantitative normal constant without a Hessian sweep

Write the native coefficients

```
x0=1/5+(3/(25a)) sum_j(1-cos kj),  xj=(3/(10a)) sin kj.
kappa=2/a, Cx²=(1/5+18/(25a))²+3(3/(10a))².
```

For the ten unordered pairs p=(mu,nu), define fp=xmu*xnu and the exact rational
matrix Dpq=integral fp fq. `coefficient_design` constructs it by Laurent
polynomial multiplication and taking the zero coefficient. All its positive
LDL pivots and its inverse are calculated rationally. Products have degree at
most4 in each coordinate. Therefore their average on **every real shifted**
N-grid equals the integral for N>=5; no finite scan, centre sample or asymptotic
Riemann estimate enters this fact. Put

```
eta=1/Tr(D^-1)>0; D>=eta I.
```

At a positive metric-Clifford frame set
Cmu,nu=({Phi_mu,dPhi_nu}+{Phi_nu,dPhi_mu})/2. Its scalar component is dGmu,nu.
The trace-free component gives
`{h,dh}_0=sum_mu,nu xmu*xnu Cmu,nu,0`.
The original sea-Cauchy identity gives its Hessian contribution at least

```
eta/(2 rmax³) sum_mu,nu ||Cmu,nu,0||_(HS/2)²,
rmax²=gmax Cx².
```

Here `||M||_(HS/2)²=Tr(M†M)/4`; the offdiagonal polynomial coefficient is twice
Cmu,nu, so the unordered coefficient norm bounds the ordered tensor norm.
Strict Gram convexity gives the additional
`kappa sum_mu,nu |dGmu,nu|²`. The canonical complete60 real-coordinate
Clifford map C has exact spectrum

```
0 (multiplicity15), 4 (10), 6 (20), 8 (15).
```

This follows by grading the15 Hermitian Clifford monomials and is also
reproduced by exact Gaussian-integer symbolic matrices in the archived
`local://issue222-normal-clifford-exact.py`. Its kernel is exactly the gauge
orbit. Transport Phi=S Gamma and dPhi=S dY, S=G^(1/2), gives tensor lower
gmin² and coordinate upper gmax. Distance to the transported gauge orbit
is treated before choosing a horizontal slice. Consequently

```
alpha = 4 gmin²/gmax * min(kappa,eta/(2 rmax³))
```

is a sufficient lower on all45 physical uniform modes at each exact Gram
minimum. This is an analytic positive number, not the original CLI's alpha
estimate. It covers all35 nonmetric slack normals and all10 Gram directions.

## 3. All-q stiffness, Haar mass and heavy gap

With actual generators Ta=Ba/sqrt8, Tr(TaTb)=delta_ab/2, the canonical field
orbit inertia eigenvalues lie in `{1/2,1,3/2,2}`. Thus
`Dgauge†Dgauge >= gmin/2`; the15 generators cannot be discarded. In particular
using `4gmin` in the native generator convention would be an eightfold error.
The actual link mass is1/.6 and the field mass is1. For each q the alignment
map is `Lj=(dj I,-Dgauge on link j)`, dj=1-exp(iqj). Its positive singular
values on the mass-horizontal complement are bounded below by either
`sum|dj|²` or `.6 gmin/2`. At nonzero grid q,
`sum|dj|²>=16/N²`, from sin(pi/N)>=2/N. Hence alignment alone supplies

```
b=(1/5) min(16/N²,3gmin/10)>0.
```

At q0 the45 uniform kernel modes remain and all45 link normals are positive.
Plaquettes add a positive form. Using ||d||²<=12 in the three-axis curl
matrix supplies the intentionally wider `Bmax=10+gmax`.

The full-heavy gap at each exact frame is
`Delta=2m sqrt(gmin)`. The original all-q endpoint/contact proof gives

```
SF<=(3r²+3v²)gmax/a²,
L=6kappa gmax+(60Cx²+30SF)/Delta
  +sqrt(5400(r²+v²)/a²+450SF).
```

Roots are outward rational bounds, computed with integer square roots.
The Schur onset is `Lambda^(1/4)>=2(L+2L²/alpha)/b`. It gives soft frequencies
`sqrt(alpha/2)Lambda^(1/4)`, stiff frequencies
`sqrt(b/2)Lambda^(3/8)`, and the all90V zero-point upper
`45V sqrt(L+Bmax)Lambda^(3/8)`. Real/conjugate Fourier momenta are not doubled.

## 4. Explicit analytic-chart and finite-jet majorants

Here are intentionally very wasteful constants for the regular tube. They
exist to make a finite legal choice computable, not to estimate practical
Lambda. Let d=105V, rho=sqrt(4Vgmax),

```
h=2^12 d²(1+rho+Cx),
s=min(1/4,sqrt(gmin/2),sqrt(b),Delta/8),
r=min(1/64,(s/(1024h))^8,Delta/(64Vh)),
J=(1024h/s)^32.
```

The analytic-coordinate construction has four finite operations: the
product exponential coordinates on SU4^(3V); the mass-horizontal gauge
slice through the regular orbit; the uniform field gauge slice; and the
orthonormal normal-frame identification which follows the zero stiff
bundle. The last is the analytic polar identification of the moving
projector with its value at the origin, not a fixed Euclidean complement.
The uniform slice and its projector give coordinates (z,xi) with
`V1(z,0)=d_xi V1(z,0)=0`. This identity, rather than a generic Cartesian
Taylor estimate, removes spurious pure-soft stiff quartics.

For completeness the coarse analytic majorant used here is obtained as
follows. Each product exponential and adjoint action has derivatives bounded
by `2^q q! d(1+rho)` on the unit complex polydisc; native endpoint products
have at most two field endpoints and two link factors. The first two
Jacobians have least singular value at least s, and the projector operation
uses a contour of distance at least s²/4 from its complementary spectrum.
The elementary analytic inverse estimate shrinks a disc by at most
`(s/(1024h))²` per inverse/projector/polar operation. This follows by writing
the nonlinear part as its integral derivative and summing the Neumann series
at ||Dnonlinear||<=s/4; the spectral projector uses the same series for its
resolvent. Four such operations fit within the eighth-power radius above.
Each resulting coordinate/cometric/connection coefficient has q-th
contracted derivative at most `q! J^(q+1)` through q=8. The mass-normal
cometric at the origin is identity; shrinking to r retains bounds1/4 and4.
The charge representation norm is bounded by4V+3 for the whole raw light,
heavy and opposite-reference fibre; no selected CAR vector enters it.

The rank-one **whole many-body** projector also has a direct, independent
Riesz bound. Native endpoint q-th derivatives are at most
`q! 2^q 16(1+Cx)(1+rho)`. Multiplying by NH=2V bounds dGamma on the entire
finite Fock fibre, without forming it. A contour around the actual original
heavy ground with radiusDelta/2 stays separated in the r tube. Differentiate
the resolvent's Neumann series; the displayed
`2 q! [64 NH endpoint/Delta]^q` bounds the projector. Kato's differential
transport then supplies the exported section bound. The heavy state is
EM-neutral and this construction is independent of A on the entire tube.

## 5. Actual scalar well: form comparison and moments

A single sufficient relative-comparison constant is

```
C=2^16 d³ J^4 (1+1/alpha+1/b)(4V+3)².
Z=45V sqrt(L+Bmax), H=4Z+4C.
```

Taylor's integral remainder, the flat-adapted V1 identity, and the coefficient
bounds above give on epsilon=Lambda^(-1/5)

```
(1-Cepsilon)Q-Cchi/epsilon <= S_theta-Lambda e_theta
 <=(1+Cepsilon)Q+Cchi/epsilon, chi=Lambda^(-1/2).
```

All Berry, Born-Huang, quotient density and heavy-only angular terms stay in
S_theta. Light/reference connections are bounded by the full representation
norm and are compared separately. Young's inequality spends relative
epsilon kinetic energy and at most Cchi/epsilon; it never multiplies the
negative classical baseline by1-epsilon. These constants also cover that
full-fibre comparison. The code's explicit local onset pays Cepsilon<=1/64,
the Schur threshold, and a core barrier64 times the full vacuum upper.

Minmax with the complete90V quadratic states, full-domain IMS, and the
shrinking-core barrier gives

```
E_theta-Lambda e_theta <= H Lambda^(3/8),
g_theta >=(frequency/4)Lambda^(1/4)
          -8C(H+1)Lambda^(7/40).
```

The exact Etheta is **not computed** and is never replaced by Z or H. Those
are inequalities only. All64 Etheta−Emin remain in the effective family.
Positive kinetic, potential restoration and the above energy inequality give

```
||p phi|| <= sqrt(8(H+C)) Lambda^(7/16),
||z phi|| <= sqrt(8(H+C)/alpha) Lambda^(-5/16),
||xi phi|| <= sqrt(8(H+C)/b) Lambda^(-7/16).
```

These sufficient coarse moments do not assert a sharp frozen-Gaussian
variance. Momentum includes the bounded connection/density contribution.

The core Agmon weight can have coefficient
`aAg=min(1/64,sqrt(min(alpha,b))/256)` in the action
`aAg Lambda^(7/20)`. Normalization, first/second cutoff derivatives and
elliptic interior estimates are covered by
`64C(1+H) Lambda^(d+10) exp(-aAg Lambda^(7/20))`.
The large dimension-dependent polynomial is retained, not silently treated
as a universal constant. To avoid exponentials entirely, use
`exp(-u)<=j!u^-j<=j^j u^-j`,
`j=ceil((80(d+10)+5)/28)`. With Lambda=t^80 this tail is at most
`2^cutoff_log2/t^5`; cutoff_log2 is computed by exact integer logarithmic
upper bounds without forming j! or2^cutoff_log2.

## 6. Entire-photon/current residual and noncompact field exterior

The complete colour-orbit Haar integral has stabilizer character
`i^(NH+NL)*reference_character=1` on every raw CAR vector. Disjoint core
supports give exact isometries. This does not restrict Haar angles, angular
irreps, fermion vectors or photon energy. W is independent of A and therefore
both unbounded photon P and the pure magnetic operator intertwine exactly.
For the noncompact successor this also includes the unbounded quadratic
magnetic multiplication form in A: fibrewise isometry and A independence give
exact form-domain intertwining, not a bounded-magnetic approximation.
A derivative of A does not hit the heavy/scalar factor, even on arbitrarily
high photon-energy vectors in D(Hph).

On the scalar image the physical kinetic comparison has norm at most
`16C(momentum_prefactor+1)Lambda^(-1/16)`. The native light/current/contact
endpoint coefficients are Lipschitz uniformly all compact phases, with
`32(2V+1)Cx(1+rho)J`. This gives the exported
Lambda^(-5/16) image difference. Phase jets of a literal endpoint are
`(i f_edge)^q DeltaT_edge exp(iA_edge)`; absolute edge components, not a
cancellation-prone sum, are controlled. Hence the bounded residual extends
over the **entire effective photon Hilbert space**, not only oscillator
Gaussians or a photon window.

The full Phi domain is unbounded. With

```
A4=kappa/(8V), BH=4V Cx, BL=2(2V+1)Cx,
rho0>=max(1,(2BH/A4)^(1/3)),
C0=BL rho0, C1=2BL/A4,
```

native quartic/endpoint bounds give
`V0>=A4 rho^4−BH rho`, `||Hlight||<=BL rho`, and globally
`||Hlight||<=C0+C1(U_Lambda−v*)`.
The code uses an integer-power cube upper for rho0. Exterior comparison is
`(Lambda−C1)(U−v*)−H Lambda^(3/8)−C0`, not a fictitious global constant
light norm. Reducible/singular exterior regions are treated on the original
unreduced physical form, not by a regular quotient chart. The independent
`noncompact_quantitative_scalar_phase.phase_bounds(N)` now supplies the exact
distance coercivity for the full original a1 domain, N>=2^96. Its CAD/root-height
packet has A=100(100000V²)², phase onset2^(8B), B=2^(2^A), and c=2^(-8B²).
This is the source's analytic integer majorant, not a native finite-N scan.
Adapted core distance may shrink by J, so the selector also pays c/J². With
D=8B²=8*2^(2^(A+1)) and
ell>=log2[256(H+C+C0+requested_gap+1)J²], its exact choice is
`Lambda=2^(80*(k+ell+D))`. It pays the exterior condition
`Lambda^(9/40)>=64(H+C+C0+requested_gap+1)J²/c` and the phase onset, in addition
to every local requirement paid by k. No tower is materialized to make this
choice; the returned symbolic expression is a finite integer.

## 7. Consumer capability and same-current source jets

Call `native_normal_budget(N)` then
`choose_lambda(budget,error,time,norm_budget,requested_gap,source_multiplier,phase)`.
Every exact power is made integer using Lambda=t^80. The selector uses
outward logarithms, so its Lambda is an explicit finite `2^(integer)` or the
fully paid scalar-phase tower expression from the preceding section. It
never converts a huge finite parameter to floating infinity. Errors and
requested complementary gap are certified by monotonic power inequalities.
`resolvent_budget` implements `3d/Im(z)^2+1/(g−|Re(z)|)` exactly.

For a requested six-current graph, `orientation_source_jets` exports the
CentreSourceEnergy whole-Haar analytic q<=6 bound

```
4 *2^(q+1) q! [16q(1+alpha sqrt((L+1)B²)+eta sqrt L)]^q /sqrt(G0lower).
```

It is a source Frechet/ordered-link bound on all Phi, not a total physical
Noether bound. `source_transfer_multiplier` pays every source derivative,
the actual positive weighted `R_B aOmega` graph norm, and detector norm in
a Leibniz upper. The PhotonAmputationGraph's D1/D2 estimates are not promoted
to q3..6. No unweighted photon norm substitutes for its positive graph norm.
Actual physical Noether derivatives require contraction with Y(Phi), orbital
and photon moments; ScreenedGaussDressing reports that separate localized
Noether admission remains unpaid. This pair does not erase that limit.

`state_normal_moments` supplies actual ground/resolvent moment bounds without
setting the negative sea baseline to zero. Its neutral trial pays the entire
light matrix, positive photon oscillators, an actual normalized harmonic
Gaussian and matched Coulomb law `UE rho G rho` (not the obsolete half).
It uses the maximum all64 scalar vacuum upper plus the finite neutral trial,
C0 and a unit allowance for d0. Field momentum squared is at most
`(2/chi)[relative_energy+(H Lambda^(3/8)+trial+C0+1)*norm²]`; links use
`(1/(.3chi))` instead. Unknown exact offsets stay symbolic.

For the parent-requested additional localization capability,
`physical_noether_filter_budget` receives actual Gauss/mechanical-connection
velocity suprema and the **actual** projected matrix norm from the Noether
owner, as well as source field/link and photon/orbital/reference graph norms.
On W's shrinking-core support it bounds the unbounded Jphys graph by
`(YPhi+Ylink)*Ppref Lambda^(7/16)*source_norm`
plus the stated source/photon/projected terms. The core has bounded Phi but
the operator's exterior domain is not cut off; this is a graph estimate only
on the specified source. Harmonic A moments are explicit inputs because
photon energy alone does not bound them.

For a source energy window E<=gLambda/2 and tau=Lambda^(-1/32), a time kernel
with integrable eighth derivative gives the separated offnormal spectral
factor `||f^(8)||1/[(gLambda−E)tau]^8` for the **compressed normal block K_C**
against the effective window. Since gLambda>=frequency Lambda^(1/4)/8,
multiplying the normal momentum graph gives the sufficient leading decay
**Lambda^(-21/16)**. The exported exact dyadic logarithm bounds this product;
temporal-halo and energy-tail inputs remain separate. Applying it to the full
microscopic spectral filter requires an actual weighted spectral graph/mixing
bound from that consumer. C need not be invariant under K. Naively transferring
an unbounded J using dLambda tau ||JW|| can grow like Lambda^(11/32), so the
bounded Hamiltonian residual emphatically does not supply that missing bound.
No bare local Phi2pi is identified with physical rotation.


## 8. Evidence, adverse inputs and successors

The live existing Pyright language server was exercised before editing on
`original_normal_gap_bound.py`: symbols110, diagnostics[]. A positive
structural AST query found its native reduced_matrices and full-Gauss shape
checks. Exact symbolic Clifford arithmetic returned the spectrum and gauge
normalization stated above. The final CLI smoke additionally exercises the
actual native500x500 whole-heavy matrix, all15 Gauss/full90 horizontal
consumer, independent whole-heavy curvature, native current/contact Ward
and arbitrary compact EM covariance on centre(1,2,3), N5. N5 is a diagnostic
source control, not the native N25 metric or global-phase onset.

Observed natural exits: the native quantitative CLI completed in3.07s with
N25 constants and the N5 full source consumer; phase integration completed
in1.65s at N=2^96 and target1e-12/time6, with no grid allocation. The capability
smoke compared the exact Laurent design with an actual real-shifted N5 grid
(maximum discrepancy5.55e-17), paid a six-jet orientation/weighted graph
multiplier, rejected a mismatched phase N, and bounded the compressed r8
normal filter by2^-3937942897 on its explicit supplied graph inputs. That
last run did **not** compute an actual physical Noether projection.


The adverse actual source scaling Phi->32Phi disproves a global constant
light-matrix norm. The fast-kinetic adverse input additionally recomputes the
actual105-coordinate mass-horizontal alignment principal with all15 Gauss
directions: changing the field cometric changes its positive90-mode trace.
The harmonic-vacuum adverse input evaluates the literal radial native
potential `quartic*(1+t)^4+whole_heavy_energy*(1+t)` against its quadratic
Taylor polynomial, retaining a nonzero cubic/quartic remainder. A Gaussian
of the quadratic potential cannot be an exact eigenfunction of that operator;
this test is **not** an exact Etheta calculation. Capability checks also reject
fast chi, harmonic-vac subtraction and bounded-Phi assumptions. Their
rejection is not the evidence for the constructive bounds; the rational
derivation and actual full native consumers are. The complete-domain exact W
proof remains the original slow-kinetic source, now with computable normal
constants. Raw CAR/reference centre descent is proved algebraically for the
whole fixed-number fibre, not by a tautological arithmetic diagnostic.

Two decision-changing successors are (1) continue the same-current compact
descriptor and certified jet/graph constants from the diagnostic N2000 to
the **same N>=2^96** global-phase range, selected; (2) contract the actual
source jets with the physical Noether field and graph moments to establish
individual localized Noether current control. The integrated phase/normal
Lambda chooser is complete as a capability, but these consumer-specific
joins are not inferred. N2000 remains conditional for the global original
phase even though its actual Gram and local normal bounds are admitted.

## 9. Microscopic compensated mass transfer across all harmonic fibres

This addition uses the existing **untruncated effective** theorem in
`noncompact_charge_threshold.md` sections1–4, not a charged Ritz observation
or an isolated charged pole. The real-link Coulomb Gaussian/canonical positive
weight and dense full-CAR/reference range already give, at every retained
centre, `inf Heff_q,theta >= inf Heff_0,theta +mu_theta`. All harmonic fibres,
coexact oscillator coordinates, raw CAR states and opposite reference
components remain. The current metric certificate's actual outward endpoint
is `s>=.4182722595473932` (`noncompact_native_metric_admission.md` line171).
The nine-decimal **downward** rational floor used here is .418272259;
.418272260 would be a rounded-up number and is not used. Thus
`mu=(1/5)*sqrt_lower(.418272259)>0` is computable and common to all64 classes
at a1,N>=25. Metric admission still does not imply global microscopic phase.

Let `nu=inf Heff_0` for the direct sum containing **every common exact scalar
offset**. Classwise threshold inequalities imply
`inf Heff_q>=nu+mu`, because adding the same Etheta−Emin to both sectors and
then minimizing preserves the common lower mu. No minimizing class is picked
using floating energies. Neutral trial states transferred by W give

```
inf K0 <=nu+d0.
```

This variational argument uses an infimum-approximating sequence in the full
operator domain; a normalizable full-R harmonic ground is unnecessary.
In the charged P/C decomposition the image block is at least
`a=nu+mu−dq`, its complement at least g, and the offdiagonal norm at most dq.
For g>a, a scalar Schur/Young bound gives

```
Kq >= [a−dq²/(g−a)] I.
```

Indeed, subtract the displayed lower from the two-by-two comparison matrix.
Its diagonal entries are nonnegative and its determinant is nonnegative.
This is an operator form inequality on the original charged domain, not an
assertion that the physical spectrum is two dimensional. Use an actual
neutral effective trial `nu<=U0`. If `g>U0+mu`, then `g−a>=g−(U0+mu)`, and
the complete microscopic compensated threshold is

```
inf Hq−inf H0 >= mu−d0−dq−dq²/[g−(U0+mu)].
```

`microscopic_charge_lower` exports this exact rational bound for either
charged sign. With common residual upper d it is the requested
`mu−2d−d²/(g−Ebound)`, Ebound=U0+mu. The common Emin cancels; none of the
other exact Etheta offsets are dropped or replaced by a harmonic vacuum.

`native_microscopic_charge_budget(budget,UE,K,target_error,phase)` computes
U0 with the actual finite neutral photon/light/Coulomb trial already exported.
It chooses error<=mu/16 and requests g>=U0+mu+1 from the existing exact
normal/phase selector. That selector gives d<=error/4<=mu/64, so the loss is
at most `mu/32+mu²/4096`, strictly less than mu. This produces an explicit
finite Lambda and a positive full microscopic lower **conditional on the
underlying global BO constants/phase proof**, now supplied with its precise
review status. With no global phase packet the output remains conditional;
N2000 is not promoted to the phase range. A supplied same-N phase at N>=2^96
admits the mass capability without any photon/Fock/normal truncation.

The addition's scoped smoke evaluates exact two-block determinant inequalities,
including a negative neutral effective infimum, zero-error equality, a
nonpositive mass lower at excessive residual, and the inadmissible
complement-denominator boundary. It also exercises the new same-N native
mass capability at N=2^96 with exact positive UE,K. It does not rerun the
existing author CLI/native checks and does not independently certify the
earlier quantitative chart/CAD/Agmon construction. No isolated charged-pole
claim or reference removal follows from a positive compensated threshold.
