# Full-field transfer of the all-bosonic internal-frame pulse

Governing plan `issue-222-continuing-backward-20260930`, assignment revision
`ed42a4078d9dca07c06716c3774d2b67477eb55266b92f03b60c53bd0ece3c3c`,
Latest nonblocking world-model steering revision
`0004af9d7c06f05f2fb7b9969804d0634014cc82d83934ddf7b563356e800a8e`;
scientific requirements unchanged. The world-model writer owns its separate
private directory; this assignment owns only this new source/proof pair.
pinned `general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`,
work campaign/outcome/verification. Parent owns scientific execution and plan
updates. This pair does not report an unexecuted scientific result.

## 1. Law, operation, and theorem boundary

Use the **different noncompact electromagnetic law** of the existing
noncompact suppliers: real links, positive quadratic curl potential, exact
unitary matter phases, reduced longitudinal Coulomb energy, and the full
harmonic direct integral. Retain all original 60V scalar coordinates, actual
SU4 links, the original itinerant whole NH=2V heavy CAR and entire NL=2V or
2V+1 light CAR sector. Charged creation includes the static opposite EM charge
and all four bar-four components at the same outside reference R. Canonical
quantization, CAR, and the chosen microscopic bare parameters are assumptions.
No 27-selector energy or extra selector CAR enters this argument.

At fixed a and finite N in the **existence-only globality range** of
`original_slow_kinetic_bo.md`, use its explicitly different slow trajectory:
field cometric Lambda^(-1/2), colour electric coefficient .3 Lambda^(-1/2),
alignment and colour magnetic coefficients proportional Lambda^(5/4).
Native N5 frame diagnostics do not certify membership in that globality range.
Changing compact EM to the positive noncompact quadratic EM form does not
alter the normal/heavy construction: phases still have norm one uniformly
all real A, the photon form is positive, and the normal/heavy section is
independent of A. In particular the exterior light bound in BO section 6
still applies. The photon potential need not be bounded above.

The source is specifically the one in `bosonic_frame_torque.md`:

```
g_x(Phi)=b_x chi_frame(Phi_x) S_i(Phi_x),
K_bos=1/2 sum_xa {g_x^a,Q_Phi,x^a+(div E4)_x^a}.
```

The retracted S comes from the actual four scalar matrices, not an assigned
colour basis. Choose fixed real bounded b, equal one at every packet site
and its chosen source halo, zero at R. All Hamiltonian edges, including those
crossing this window, remain. A smooth invariant source/mark cutoff is one
on every retained BO core and supported in the regular frame tube. It cuts
coefficients, never the Hamiltonian domain, states or Fock occupations.

On the exact complete Gauss quotient the operator is the bounded multiplication
matrix

```
K_bos=-Q_H(g)-Q_L(g)-Q_ref(g).
```

The field/link velocity is an actual gauge tangent at every field value in
the support. Therefore its quotient normal principal component and every
normal derivative are exactly zero. Fixed-generator gauge fields preserve
Cartesian/Haar measure. Equivariance gives sum_a X_a g^a=0, including the
invariant cutoff, so symmetric ordering has no coefficient-divergence
remainder. These are exact full-field facts, not a frozen-normal estimate.

This is an **internal colour-frame drive**, not local Cartesian/mechanical
rotation. Its equality to a vector-index rotation fails off the Clifford
core, and can fail at a nonisometric true finite-N Gram. The global proper-cube
lift already proved elsewhere is not being recast as a new local mechanical
Spin3. A localized 2pi colour path changes crossing Wilson links; it is not a
closed bosonic spatial rotation.

**Pulse theorem.** Let B denote the complete-reference four-label finite
packet preparation with genuine oriented Wilson words, Coulomb dressing,
and a smooth common mark cutoff as above. For initial densities lifted by the
unperturbed BO isometry W0, retain their actual initial four-by-four source
metric G and assume G>=g_* I with g_*>0. For each of two legs choose a real
angle theta_j, the same frame axis and window, with theta_1+theta_2=2pi, and
leg duration tau=Lambda^(-p), 0<p<1/4. Evolve the **same unreversed complete
microscopic Hamiltonian** in both neutral and charged sectors:

```
gamma_j=theta_j/tau,
U_s,j=exp[-i tau (H_s+gamma_j K_bos,s)],
U_s=U_s,2 U_s,1,       s=0,+.
```

There are constants depending on a,N, path lengths, couplings, sectors and
windows, not on the photon vector, such that the actual relative echo obeys

```
||(U_+ B+B U_0) W0 G^(-1/2)||
 <= C/sqrt(g_*) [ Lambda^(-5/16)
                 + max_j |gamma_j|/Lambda
                 + sum_j tau (Lambda^(-1/16)
                               + |gamma_j| Lambda^(-5/16)
                               + gamma_j^2/Lambda)
                 + tau ]
    + polynomial(Lambda,gamma) exp[-c Lambda^(7/20)]/sqrt(g_*).
```

The notation for a mixed state means the corresponding preparation map on
its purification, or the largest eigenvalue of its complete error Gram.
The entire initial density and all well weights are retained. The neutral
RIGHT is U0 of that same density, including its evolved heavy/normal part.
There is no final RIGHT Gram inversion. For p=1/8 the displayed terms tend
zero (the slower displayed contribution is the ordinary light relative
motion O(Lambda^(-1/8))). This theorem is not a numerical scalar vacuum,
thermodynamic/uniform-N theorem, total interacting charged spin multiplet,
photon phase or Compton scattering theorem.

## 2. Why the unmodified heavy line fails, and the changed construction

Write the actual heavy Hamiltonian as H_H=-Lambda dGamma(h(y)), where y is
all physical normal coordinates in a regular all15V Gauss chart. Write
q(y)=directsum_x g_x(Phi). Then K_H=-dGamma(q). Generally
[H_H,dGamma(q)] is nonzero, including at the localized BODY window. The
unmodified line therefore has a heavy-excited pulse residual of size
|gamma| ||(1-P_H)dGamma(q) Omega_H||, not O(gamma |y|). Zero mean does not
mean zero variance, and even NH does not imply invariance.

The changed material construction is to diagonalize the **actual pulsed
fast block** before comparing low-energy dynamics:

```
h_gamma(y)=h(y)+gamma q(y)/Lambda,
H_H+gamma K_H=-Lambda dGamma(h_gamma).
```

The plus sign is forced by this repository's H_H=-Lambda dGamma(h) and
K_H=-Q convention. Equivalently, if q_K=-q is called the source onebody,
h_gamma=h-gamma q_K/Lambda. Using h-gamma q/Lambda with K_H=-Q would evolve
the opposite physical pulse. No sign is chosen from a desired endpoint.

Let P_gamma(y) be the top NH occupied spectral cluster of h_gamma, by a
Riesz contour separated from its entire remaining spectrum. Its rank is
NH; no per-cell or per-momentum occupation is imposed. For |gamma|/Lambda
small enough, the actual unpulsed onebody gap on the fixed tube and Weyl's
inequality give a common contour and heavy many-body gap c Lambda. Riesz
resolvent differentiation gives, for r=0,1,2,

```
||D^r(P_gamma-P_0)|| <= C_r |gamma|/Lambda,
||D^r P_gamma|| <= C_r.
```

For example D R=R (D h_gamma) R; the second derivative has the two ordered
products R Dh_gamma R Dh_gamma R and R D^2h_gamma R. Every factor on the
fixed tube is bounded. Subtracting the gamma=0 identities puts one
O(gamma/Lambda) factor in each difference. There is no gamma-enhanced
unbounded normal derivative hidden in this assertion.

Use the canonical near-identity Kato unitary between P0 and Pgamma, then
its complete exterior-power lift, to define the normalized occupied Slater
line Omega_gamma. Fix its phase by that near-identity lift, not independent
eigenvector determinants at different points. On the contractible normal
tube its derivatives through order two differ by O(|gamma|/Lambda). On the
full orbit the construction is gauge equivariant. The stabilizer character
is unchanged; the entire light/reference fixed-number fibre still descends
under all SU4 Gauss constraints. Heavy remains electromagnetically neutral.

Let u_theta be the **actual** normalized scalar-well state/cutoff of the
unperturbed BO construction, including its exact heavy Berry, Born-Huang,
quotient density and scalar offsets. Define Wgamma using the same u_theta
and actual Omega_gamma. It is an exact isometry, not a projected heavy
Hamiltonian approximation. In full orbit measure,

```
||Wgamma-W0|| <= C |gamma|/Lambda.
```

No claim is made that the original heavy state is invariant during a pulse.
At a leg switch change comparison isometries, not the actual state. The
switch cost is at most C(|gamma_1|+|gamma_2|)/Lambda.

## 3. Heavy energy, its normal derivatives, and exact offsets

At every native metric-Clifford star, the full endpoint h anticommutes with
a common chirality J. Every local bivector q commutes with J and is traceless.
Thus P0=(1+sgn h)/2 gives

```
Tr(P0 q)=0.
```

This holds for an arbitrary real spatial b and full itinerant filling; it
is not a two-particles-per-site assumption. Normal variation of this mean
is O(|y|), since P0 and q are smooth. Other colour means and all variances
remain present.

Let e_gamma(y) be the actual lowest fixed-NH eigenvalue of
-Lambda dGamma(h_gamma(y)), and e0(y) the unpulsed value. Hellmann-Feynman,
integrated over s gamma, gives

```
e_gamma(y)-e0(y)=-gamma integral_0^1 Tr[P_(s gamma)(y) q(y)] ds,
|e_gamma-e0| <= C (|gamma| |y|+gamma^2/Lambda).
```

Its first and second normal derivatives are bounded by C|gamma| on the
fixed tube, by the same Riesz identities. The exact well-dependent pulse
scalar offset is

```
delta_theta,gamma=e_gamma(0)-e0(0),
|delta_theta,gamma| <= C gamma^2/Lambda.
```

Retain delta **exactly** in the effective generator. Never replace it by a
truncated perturbation series or erase its phase between wells. In addition
retain every original exact E_theta-E_min. Each offset is common to neutral
and charged sectors, and commutes with B, which is diagonal in the well
comparison maps. Distinct wells need not have equal offsets or uniform
weights; finite-Lambda centre tunnelling remains in the microscopic law.

After subtracting delta the variation is bounded by
C(|gamma|+gamma^2/Lambda)|y|. The coarser envelope including gamma^2/Lambda
above is deliberately safe and suffices. The Berry/Born-Huang and quotient
kinetic coefficients changed by the line trivialization have derivatives
O(gamma/Lambda); applying them to u costs at most
C chi (|gamma|/Lambda)(||p u||+1). This is smaller than the stated residual.
There is no time derivative of a projector within a constant leg. Jumps
between constant legs are paid by their lift differences, not a fictitious
adiabatic time derivative.

## 4. Quantum norm moments, bounded residual and exterior gap

The BO estimates used here are **quantum norm estimates on u**, not normal
samples, pointwise amplitudes, or products of separate means:

```
chi=Lambda^(-1/2), epsilon=Lambda^(-1/5),
||p u|| <= C Lambda^(7/16),
||z u|| <= C Lambda^(-5/16),
||xi u|| <= C Lambda^(-7/16),
cutoff residual <= polynomial(Lambda) exp[-c Lambda^(7/20)].
```

z comprises the 45 physical uniform normals, xi the remaining restoring
normals. The complete mass-horizontal quotient, density, Berry and angular
terms are the ones in the existing BO proof. These moments arise from its
full zero-point budget O(Lambda^(3/8)), positive kinetic form and weighted
potential, not a product Gaussian assumption. Since |y|<=C(|z|+|xi|), they
imply ||(e_gamma-e0-delta)u||<=C(|gamma|+gamma^2/Lambda)
Lambda^(-5/16). The displayed theorem uses the weaker safe envelope.

In sector s put Q_s,theta=Q_L(q_theta)+Q_ref(q_theta), retaining the actual
reference representation (-S_i^* for bar4); b_R=0 gives its zero coefficient,
not deletion of its four components. The effective pulse generator is

```
F_s,gamma=directsum_theta [ (E_theta-E_min)+delta_theta,gamma
                          +H_ph+H_C,s+H_L,s(theta,A)
                          -gamma Q_s,theta ].
```

On D(H_ph), the actual full operator satisfies a bounded image residual

```
||(H_s-E_min+gamma K_bos,s) Wgamma-Wgamma F_s,gamma||
 <= C [Lambda^(-1/16)+|gamma| Lambda^(-5/16)+gamma^2/Lambda]
    + polynomial(Lambda,gamma) exp[-c Lambda^(7/20)].
```

Indeed the native light difference is Lipschitz normal times bounded
unitary phases. The extra light/reference pulse difference is
|gamma| ||(q(y)-q(0))u||. Off-heavy kinetic terms use the actual differentiated
Pgamma and cost chi ||p u||=Lambda^(-1/16), not gamma times the original
heavy variance. Heavy energy and derivative costs are those in section 3.
All projected scalar terms keep their exact unperturbed auxiliary energy;
no term of order one is dismissed individually on a product normal state.

The complete complementary gap withstands this pulse. Globally K_bos is a
bounded finite-CAR multiplication matrix, ||K_bos||<=C at fixed sectors and
N. The full physical H+gamma K has the same operator domain as H. In each
core the modified heavy excited gap is c Lambda-C|gamma|. The scalar normal
gap is at least

```
c Lambda^(1/4)-C Lambda^(7/40)-C|gamma|-C.
```

To apply this to 1-Wgamma Wgamma^*, trivialize its actual heavy line,
retain the scalar positive relative kinetic/potential form, and discount
only that form by O(gamma/Lambda) plus the original BO C epsilon. Its vacuum
budget loss is O((gamma/Lambda) Lambda^(3/8)), negligible. The additional
scalar energy is bounded by C|gamma| on the fixed tube; orthogonality to u
is componentwise in the ENTIRE effective photon/light/reference fibre.
One must not discount the negative Lambda e_theta classical baseline.
The exact line offsets contribute at most C gamma^2/Lambda to this bound.

Exterior IMS uses the original full-domain quartic/light domination and
barrier c Lambda^(3/5)-C Lambda^(3/8). Bounded pulse loss C|gamma| and the
unchanged chi/epsilon^2 IMS cost cannot close it. The cutoff vanishes before
Gram/sign singularities; no exterior regular Gauss chart is assumed.
Reducible, singular and arbitrarily large fields remain in the Hamiltonian.
Agmon tail bounds retain their exponent since these bounded perturbations
are below the core barrier. For p<1/4 the normal complement tends to
infinity and the displayed image residual tends to zero. This is the
restricted legal joint scale; an arbitrary sudden gamma~Lambda pulse is
not covered.

## 5. Full noncompact photon/Coulomb and preparation domains

Wgamma is independent of every real EM coordinate and preserves all
n_L(x), since its matter changes are colour/heavy only. Hence the unbounded
pure photon electric, quadratic curl and harmonic direct-integral operators
intertwine **exactly**, on their full domains. Longitudinal Gauss reduction
produces the unchanged sector-specific positive Coulomb matrix
UE rho^T(DD^T)^+rho. Wgamma intertwines that too. Neutral rho=n-2; charged
rho=n-2-delta_R, including the static opposite charge. The heavy species is
EM neutral, and total graph charge remains zero.

Before reduction the creation coefficient includes the actual fractional
longitudinal phase with vector D^T(DD^T)^+(delta_x-delta_R); it is a unitary
momentum translation with zero total charge, not a prescribed photon phase.
After exact reduction it becomes the sector change of H_C, while transverse
and harmonic quantum photons remain. The resulting Coulomb-prepared B
commutes with pure transverse/harmonic H_ph. It does **not** commute with
H_C; that genuine bounded finite-sector commutator enters the relative
light error. No arbitrary scalar electric momentum is put to zero.

For M_a(x)=eta_x chi_mark(y) V_a(Phi_x) W_xR(U), V=(I,2iS_i), the complete
reference CAR identity and bound are

```
B_M^* B_N=Tr(M^*N)-dGamma(N M^*),
||B_alpha|| <= ||M_alpha||_HS <= C ||alpha||.
```

All reference columns, off-site coherent packet terms and complex eta are
retained. Derivatives obey the corresponding HS coefficient bounds. Finite
Wilson path lengths enter constants; they are not declared cost-free.
On regular cores the marks are smooth, and

```
||B Wgamma-Wgamma B_theta|| <= C Lambda^(-5/16).
```

Heavy commutes with preparation; the estimate concerns its actual field,
link and reference coefficient, not a changed heavy occupation. B maps the
common smooth cylindrical/finite-CAR form core between the two physical
Gauss sectors, with Coulomb dressing as above. Applying it to Wgamma
D(H_ph) lies in the required operator domains by bounded first/second normal
jets and u's local elliptic regularity. No L2 longitudinal Gauss kernel on
unreduced real links is asserted.

The positive initial-metric hypothesis has concrete full-domain suppliers,
not a post-pulse renormalization. For example use the maximally mixed
**entire** neutral fixed-NL sector on the effective fibre and any finite-energy
physical transverse/harmonic photon density, then lift by W0. Its onebody
density is I/2 pointwise in the quotient colour chart. Complete CAR and
Tr(V_a^*V_b)=4 delta_ab give G=2I exactly on the torque/mark plateau,
including arbitrary complex eta and every retained well weight. This is a
legal finite-energy preparation, not a claim that a product density is the
coupled neutral ground. The numerical consumer instead uses the actual
coupled engine and measures its G. Alternatively the microscopic CT identity
in `noncompact_marked_metric.md` applies to these **sign-even retracted**
marks: a CT-invariant neutral density gives
G=2 <sum_x |eta_x|^2 chi_mark,x^2> I. The cutoff expectation, or the actual
coupled-source minimum eigenvalue, remains visible; it is never set to one
by a final metric division.

## 6. Both legs, orientations, actual RIGHT, and Duhamel

For a constant leg, the bounded image residual and selfadjoint domains give

```
||exp[-it(H+gamma K-E_min)] Wgamma
   -Wgamma exp[-it F_gamma]|| <= |t| d_gamma.
```

Start from W0, pay ||Wgamma-W0||, evolve the actual state, and at a switch
pay only ||Wgamma_2-Wgamma_1||. Apply this separately in the two complete
sectors and compose both legs. The second leg inherits the first's actual
state, not a new normal or heavy vacuum. No H reversal or heavy-ground
invariance enters. Common phases E_min and each exact effective offset are
retained and cancel only in the stated relative preparation comparison.

At every packet row b=chi_frame=1 and at R b=0. Actual bosonic differentiation
of V_a W_xR is i S_i V_a W_xR: the frame commutator and Wilson endpoint terms
cancel internally. Thus the label action is **left multiplication**,
S_i V_b=sum_a V_a (s_i)_ab. Right multiplication V_b S_i is a different
orientation and is not substituted. On the whole neutral/charged CAR,

```
Q_+,i B_b-B_b Q_0,i=sum_a B_a(s_i)_ab.
```

For angles totalling 2pi, the source-only complete-reference identity is
U_source,+ B=-B U_source,0. It includes the full neutral RIGHT; it does not
assert U_source,0=I or a charged-state invariant spin multiplet.

Replace the effective full-H legs by source-only legs **relatively**, not
by discarding unbounded photon H. In the source interaction picture,
H_ph and all scalar well offsets commute with B and the spin source.
The remaining H_L, H_C and reference commutator with the bounded creator
is bounded uniformly every photon coordinate at fixed graph. Relative
Duhamel therefore costs C(tau_1+tau_2); it needs no photon energy window or
photon ground. Compose with the two microscopic image estimates, the
endpoint preparation coefficient estimates and initial lift changes. This
gives section 1's bound. If the initial full-field density has an extra
norm distance eta to ran W0, add C eta/sqrt(g_*); it is not silently treated
as the BO state. A microscopic exact-ground supplier must supply that
separate distance and the actual G, not merely an initial point norm2.

For arbitrary complex alpha the exact initial norm is alpha^* G alpha.
The estimate applies after this one initial metric choice. RIGHT may acquire
a different metric at finite strength. Neither that metric nor a failed
raw cross is inverted to restore a desired minus sign.

## 7. Runnable actual consumer and discriminators

The Python companion constructs genuine non-Clifford scalar sections,
nonuniform colour-gauge-moved frames and crossing Wilson links. It evaluates
native endpoint-average heavy/light matrices on all four graph edges, with
no imported 27-selector potential. It uses the complete NH6 exterior924,
NL6/NL7 sectors924/792, all four bar4 reference components, and the existing
**coupled** oscillator/harmonic/light/Coulomb neutral spectral engine. No
occupation filter or hand-selected photon Gaussian is supplied.

For each leg and scale it computes the actual fast h+gamma q/Lambda Riesz
line, full-CAR eigenline residual, exact scalar offset, cluster gap and
unmodified-line residual. The latter reaches and fails the **same** modified
Riesz-line assertion: setting the source's excited-heavy part to zero is
not allowed to pass because its mean is zero. Actual heavy evolution uses
expm[i(tau Lambda h+theta q)] on occupied columns followed by all924 Slater
minors. This is exact exterior evolution of that state under the complete
onebody CAR Hamiltonian, not a per-cell projection or BCH split; h and q
need not commute. Its large common exact offsets are retained as phases.

Finite section heavy and quantum light factors separate **only in this
numerical discriminator**. The full tensor history Gram/cross is evaluated
from their exact factors without allocating their product. The analytic
full-normal result above does not assume this factorization. The consumer
also propagates a legal heavy ground/excited complex superposition in the
entire924 sector, together with the actual full-H quantum light histories.
Holding its RIGHT heavy factor frozen, or projecting it away, fails the
same actual tensor-history Gram/cross cancellation assertion. Restoring its
actual evolved RIGHT passes that same boundary; the finite-strength light
minus error is not falsely assumed zero. This superposition is not claimed
to be a low-energy BO state. Cutting the packet's actual site1 halo separately
fails the source-only minus endpoint. Actual pulse full-H endpoint
errors and RIGHT metrics are reported without a forced small-T threshold.

The photon harmonic calculation is one declared Bloch fibre with noncyclic
Galerkin compression; it is not a normalizable full-R harmonic ground.
The analytic transfer acts on the complete direct integral. The numerical
fields are fixed normal sections with no normal kinetic operator or quantum
colour vacuum: they do not simulate all60V quantum fields, certify the BO
moments, or constitute a native cubic admission calculation. Exact finite
CAR/Riesz arithmetic and coupled finite photon propagation are the actual
observable checks it supplies.

Recommended parent command, from this proposal directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
 noncompact_frame_pulse_transfer.py --levels 3 --harmonics 2 \
 --noises 0 .0003 --scales 64 512 --power .125
```

Then refine the actual consumers independently: `--levels 5 --harmonics 3`,
`--scales 512 4096`, `--axis 0` and `--axis 1`; use unequal two-leg strengths
`--angles 1.2 5.083185307179586`. The legal scale parser rejects p outside
(0,1/4) and non-2pi total angles rather than labeling them paid echoes.
The adverse heavy amplitude and duration can be varied with
`--adverse-scale` and `--adverse-duration`; accidental recurrence is a visible
wrong-construction-passed error, not a forced adverse rejection. All hashes
are read from actual source/proof bytes by the CLI.

This supplies the stated fixed-a,N full-normal **internal-frame** transfer
under the explicit original BO hypotheses and a genuinely different pulsed
heavy Riesz construction. Scientific numerical values and diagnostics are
integration-owner observations. It does not remove the external reference,
identify total mechanical spin of a mobile interacting electron, prove
thermodynamic photon/Coulomb/Z4 phases, supply uniform volume/cutoff bounds,
or close issue222's joined electron-photon campaign.
