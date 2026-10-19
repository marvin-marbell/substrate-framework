# Wilson-relative all-background kernel and stronger positive Maxwell family

Governing successor: `issue-222-synthesis-continuation-20261006`, assignment
revision `14d4836b213e2bdca4c489ec23d8489fdb5ffcb76ebfa1a4fad87f7216b9e8c0`,
general template `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`,
campaign/outcome+verification. This pair constructs a sharper admitted family
and a reusable kernel; it does not rerun the predecessor phase consumer.

## 1. Actual native operator and all-background relative odd estimate

Keep LAW-05, a=1, m=.2, r=.12, v=.3, nearest-axis N>=5, whole 2V CAR,
background two and all64 wells. In each well use its full true positive Gram,
not an isotropic approximation. Let

    s=G00-G0s Gss^(-1) Gs0,
    beta=(Phi0-sum_j (Gss^(-1)Gs0)_j Phi_j)/sqrt(s),
    Phi0_perp=Phi0-sqrt(s) beta.

Then beta is a fixed Hermitian involution, anticommutes with every spatial
Phi_j and Phi0_perp, and ||Phi0_perp||=sqrt(G00-s). Central colour seams are
scalar phases on the actual torus endpoint, included in T_j. Arbitrary REAL
spatial phases, with no smallness or smoothness assumption, are admitted:
T_j is unitary, L_j=2-T_j-T_j*, S_j=(T_j-T_j*)/(2i). Functional calculus of the
single unitary proves S_j^2<=L_j. No commutativity between distinct axes is used.
The actual endpoint law is exactly

    W=m+r L_A/2, L_A=sum_j L_j,
    M_A=sqrt(s) W,
    h=beta M_A+Dodd,
    Dodd=Phi0_perp W+v sum_j Phi_j S_j.

In particular M_A>=mu=m sqrt(s). Wilson mass is NOT replaced by its onsite
part and its derivative remains in the native current.

For arbitrary psi, Cauchy over the three vectors S_j psi gives

    ||sum_j Phi_j S_j psi||^2
      <=Tr(Gss) sum_j ||S_j psi||^2
      <=Tr(Gss) <psi,L_A psi>.

Using Hermiticity, |<psi,D_v psi>|<=|v|sqrt(Tr Gss)||psi||||L_A^(1/2)psi||.
The scalar AM inequality

    m||psi||^2+(r/2)||L_A^(1/2)psi||^2
       >=sqrt(2mr)||psi||||L_A^(1/2)psi||

therefore bounds the quadratic form by kappa_v<psi,M_A psi>, with

    kappa_v=sqrt(v^2 Tr(Gss)/(2mr s)).

Polarization/self-adjoint spectral calculus, NOT a product-norm shortcut,
proves ||M_A^(-1/2) D_v M_A^(-1/2)||<=kappa_v. The mass-odd term commutes
with W and has relative norm exactly sqrt((G00-s)/s). Consequently

    kappa=kappa_v+sqrt((G00-s)/s)
    ||M_A^(-1/2) Dodd M_A^(-1/2)||<=kappa.

This is an analytic bound for every phase background at each declared G.
The CLI computes all64 actual full-ten-coordinate tilted G at N5 to guide
admission; those finite samples do not prove global metric admission.

## 2. Time-varying mass congruence, relative expansion and ordered heat

For arbitrary piecewise measurable spatial histories and scalar site-U1
temporal sources/interfaces, including actual global Haar holonomy, on the
AP circle D_E=partial_t-i a0+h. All interfaces commute with fixed beta.
For smooth approximants, with the same boundary conditions,

    beta D_E=M_A+beta(partial_t-i a0)+beta Dodd,
    Q=M_A^(-1/2) beta D_E M_A^(-1/2)=I+B+R,
    B=M_A^(-1/2) beta(partial_t-i a0) M_A^(-1/2),
    R=M_A^(-1/2) beta Dodd M_A^(-1/2).

Both B and R are skew-adjoint as forms; integration by parts is on the ENTIRE
congruence, not on the derivative alone. In particular the derivative of
M_A(t) is already owned by B. Its adjoint includes precisely the opposite
product derivative. Hence Re Q=I and

    ||M_A^(1/2) D_E^(-1) beta M_A^(1/2)||=||Q^(-1)||<=1.

The kernel assertion is an operator norm on spacetime L2 with Wilson-energy
weights, not a pointwise or anchored absolute-value Hessian envelope. Closed
forms/finite Fourier approximants yield the same coercive inverse bound;
scalar temporal Brownian kicks are unitary interfaces. kappa<1 is unnecessary
for coercivity, but supplies a uniformly convergent useful relative expansion:

    Q^(-1)=sum_{n>=0}[-(I+B)^(-1)R]^n (I+B)^(-1),
    remainder after n=p <=kappa^(p+1)/(1-kappa).

Beta is constant and commutes with I+B and anticommutes with R. Thus every
closed odd-R trace vanishes when the trace is defined. The determinant loop
expansion starts at n=2, not n=1. The mass-only filled whole-half sector has
energy -2 Tr M_A=-2V sqrt(s)(m+3r), independent of phases, even for histories.
Its finiteT/holonomy remainder is not zero: the contraction of each forward
mass-only evolution is <=exp(-mu T), giving a logdet remainder <=
4V[-log(1-exp(-mu T))]. This is a bound on the mass-only boundary remainder,
not permission to erase the actual odd-sector canonical coefficient.

For scalar forward mass evolution, the ACTUAL ordered kernel obeys

    |U_M(t,s)(x,y)| <= [U_0(t,s)](x,y),
    U_0=exp[-(t-s)sqrt(s)(m+r L_0/2)].

Proof: at a Trotter step, the offdiagonal Wilson hopping has magnitude
r sqrt(s)/2, its diagonal is sqrt(s)(m+3r), and all phase/temporal-kick factors
are unit modulus. The power/Dyson expansion is dominated termwise by the
positive free nearest-axis walk sum. Multiply the positive kernels in their
ACTUAL time order and take the product limit. The backward beta sector uses
reverse order with adjoints and the identical domination. This bound holds
for arbitrary time histories, not exp of an averaged mass. Coupled odd loops
still require their relative insertions and differentiated mass/current/contact
terms; this note does not silently turn a weighted inverse into a new CE/CB.

## 3. An explicit stronger admitted Maxwell family now

A useful immediate construction consumes the proven contact-completed
anchored density envelope in `original_massive_photon_locality.md` AppendixA
and the paid-seam theorem in `noncompact_maxwell_phase.md` sections3–5. It
improves BOTH the actual inputs and the excessive demand to cover every small
N with the same largest seam constant. No unproved relative-loop Hessian is
used for the numerical coupling claim.

### Same predecessor metric family

Retain exactly the predecessor hypothesis s>=.01, ||G||_F<=t_star,
t_star=(2sqrt(2)sqrt((m+6r)^2+3v^2))^(2/3). Thus every Gaa<=t_star and

    Ch=sum_j (r sqrt(G00_upper)+v sqrt(Gjj_upper)),
    B=(m+3r)sqrt(G00_upper)+Ch,
    d0_upper=(m+3r)sqrt(G00_upper-s_floor),
    c_upper=(m+3r)sqrt(G00_upper), c_lower=(m+3r)sqrt(s_floor).

These are native hopping/onsite bounds from the actual Clifford endpoints.
In particular the onsite odd piece is a Schur tilt, not the entire onsite
mass. Choose gamma=g mu, xi=x mu with g+x<1, alpha=log(1+xi/Ch). The CLI
selects the largest UE from five explicitly legal scalar budgets. This finite
optimization does not assert anything about sampled physical backgrounds.
Use AppendixA's exact qtail and Mtail_upper=qtail-c_lower, preserving the
subtracted c. Evaluate delta,Cbar,P,Hlow,H,CE_bulk=2CE0,CB_bulk=2CB0 at these
upper inputs, including all contact atoms and both oriented cycles.

### Optional tighter actual-metric admission

For a stronger declared family use s>=.40, G00<=.43, Gjj<=.065 for EACH j,
and sqrt((G00-s)/s)<=.002. These are inequalities on the true tilted Gram in
every retained well, not a replacement Gram. Then kappa<=
sqrt(.09*.195/(2*.2*.12*.40))+.002<1. Tight mode uses the corresponding true
row bounds and d0_upper=(m+3r)*.002*sqrt(.43). Native N5 samples can support
choosing this box, but do not certify its infinite-N membership. Any use of
this family must separately retain the actual metric admission N_metric.
The SAME-metric construction above needs no such new hypothesis.

### Pay seams by an explicit exhaustion onset, not a global overcount

Keep the predecessor's full Rstar and both cross/contact/spatial/time seams:

    R_NT=Rstar[exp(-alpha N/64)+exp(-gamma T/64)],
    tau=max(1,(alpha+1)/gamma), T>=tau N^2.

For N>=5, 1+qmin^(-2)<=N^2/8. Set epsilon=max(CE_bulk,CB_bulk)/100. Choose
N0 at or above max(5,2/(alpha/64),sqrt(1/((gamma/64)tau))) so that

    (Rstar N0^2/8)[exp(-alpha N0/64)
                   +exp(-(gamma/64)tau N0^2)]<=epsilon.

Both N^2 exponentials decrease past that onset. This is then uniform for
ALL N>=N0 and ALL T>=tau N^2; it is not a fit at a single diagonal T. The
CLI computes the smallest sufficient integer after this monotonicity floor
using logarithms, doubling and bisection. Retain N>=max(N_metric,N0).
Set CE=CE_bulk+epsilon, CB=CB_bulk+epsilon,

    hE=16 max(CE,CB/c^2,1), UE=1/(2hE), K=c^2 hE, c=.1.

They are fixed strictly positive constants, not N-dependent. rho<=1/16;
the bare speed sqrt(2UE K)=.1 remains exact. This family is materially larger
than the predecessor's uniform-small-N seam supremum while retaining its
same metric hypotheses. It spends an explicit possibly large thermodynamic
onset, not an unearned finite-N physical claim. In particular the N5 consumer
is an operator/current probe and is NOT asserted to meet this onset.

## 4. Canonical, harmonic, centre and contact ownership

The exact conditional measure, background phase, whole 2V Haar coefficient,
scalar temporal interfaces, positivity/chirality homotopy, full harmonic
histories/real-link windings and all64 actual well offsets remain precisely
those in `noncompact_maxwell_phase.md` sections1–2. None is replaced by a
Gaussian/vacuum measure. CE/CB are conditional at each such history, alpha
and centre. Spatial k!=0 source means are zero by translation, so the true
mixture preserves the two-sided covariance and subGaussian bounds with
rho<=1/16. A log of the canonical integral has score covariance; it is NOT
identified with a conditional W''. Zero-spatial-momentum/harmonic response
is not assigned Maxwell convexity. No generic temporal SU4 twist is admitted.

Both spatial seam and circle seam are present in R_NT before limits.
Canonical/global-point errors use the predecessor's actual Floquet estimate,
4V log[(1+exp(-mu T))/(1-exp(-mu T))], tending to zero along the exhaustion;
finiteT is not quietly replaced by a filled free sea. The BO/full-field legal
Lambda(N) isometric lift, all Gauss equations, heavy/common reference and
existence-only normal gaps are inherited, not numerically recertified here.
The result is a stronger conditional LAW-05 family, not native compact LAW-01,
UE=1e-4 admission, exact microscopic Gibbs equality, a sharp pole or joined
physical half-spin/scattering completion.

## 5. Current variance capability and actual consumer

On the static native whole-half negative sea P, the SAME Peierls derivative
j_eta and contact c_eta are used. For N>=3, distinct bonds give

    Tr j_eta^2=2 sum_edges (r^2 G00+v^2 Gjj) eta_edge^2,
    Var_P(dGamma j_eta)=Tr[P j_eta (1-P)j_eta]<=Tr j_eta^2/2,
    chi_static=2 sum_occ,empty |j_empty,occ|^2/(E_empty-E_occ)
                  <=Tr j_eta^2/(2mu).

The variance bound follows by retaining the two equal occupied-empty blocks
inside Tr j^2; the susceptibility uses the actual gap about zero >=mu.
These are static quasi-free conditional capabilities. They do NOT prove an
interacting-ground variance bound or turn a Peierls contact into equal-time
J^2. A full interacting photon packet must use its actual CAR/state measure
and contact-completed anchored score derivative, with true harmonic/Haar
conditioning. This scope was sent directly to PhotonPacketSupplier; the
relative inverse/density reach was sent to MobileChargedPacket.

The CLI is a new genuine3D native500-mode consumer: exact all64 full-ten-G
selection, three arbitrary large unwrapped phase histories, native Wilson
reconstruction, all-phase sine-form inequality, beta odd decomposition,
relative odd eigenvalues, ordered diamagnetic heat, phase-independent whole
half mass trace, noncommuting time mass, both-sided congruence and relative
Neumann action, actual current and contact finite differences, static canonical
Wick/susceptibility. It retains adverse one-sided mass congruence, onsite-mass
odd norm and odd-only current, all against the same native endpoints. The time
matrix is a skew spectral Galerkin identity check, not a discretized claim about
the continuous determinant. Source/runtime/input hashes and natural-exit output
are recorded by the run packet; failed attempts are retained, not overwritten.

The first stable consumer exited0 in16.94s. It gave SAME-metric UE=
1.397258562896859e-59, K=3.5784357546779154e56, CE=2.2143798605502907e53,
CB=2.236522346673697e55, N0_seam=2068737 and tau=335.5300430:
UE improves by1.4258042521840973e89 over the predecessor. The tighter DECLARED
metric box gives UE=1.1857403960026757e-39, K=4.2167746134447445e36,
N0_seam=67357 and tau=60.66457058, improving by1.209964815031781e109.
These are formula evaluations, not outward-rounded interval certification.
All64 finite N5 kappa lie in[.9200711796,.9229983213]. At centre123 the
static actual Wick variance=.00575397907<=.01174031854 and susceptibility=
.01605525505<=.09071852590. The whole interacting variance remains separate.
Adverse onsite-mass odd norm1.265011497 exceeds kappa=.9224905784;
one-sided congruence has Hermitian defect5.298808779 and odd-only current
relative error.7166304296. This receipt remains retained when the continued
two-jet construction below is executed.

Language tooling: actual Pyright stdio initialize, document symbols, definition
of original_frame and diagnostics exercised at project root; OMP has no mounted
LSP device. Positive ast-grep math.sqrt query worked. Pyright1.1.414 initially
could not resolve NumPy/SciPy under its default interpreter. Configuring the
existing scientific venv through the actual workspace-configuration protocol
resolved those imports; explicit ndarray boundaries resolved SciPy's broad
Array stubs. Changed-consumer pull diagnostics then returned zero items.

## 6. Selected next construction

Two concrete successors are (A) differentiate the relative even-band Wilson
heat loops with a gauge-covariant mass derivative and native contact at BOTH
source slots to replace Hlow entirely, or (B) use the beta-cone Riccati graph
with weighted spatial coercivity and the exact canonical seam. Select A:
the actual contraction and ordered heat have been derived/executed here, and
source/current agreement locates the mandatory Wilson derivative. This selected
construction begins as the differentiable relative-Wilson supplier below, within
this pair. Neither successor has permission to replace the already proved
anchored field-strength envelope with a bare operator norm.

### Differentiable relative-Wilson source supplier

For a bounded link source eta, let W'=dW, Dodd'=dDodd. The mass derivative is
M'=sqrt(s) W'; Phi and beta are source independent. The actual derivative is

    h'=beta M'+Dodd',
    h''=beta M''+Dodd''.

A source-dependent mass congruence also differentiates its inverse square root;
these terms cannot be omitted. A reusable noncommuting formula is

    X=M^(-1/2),
    X'=-1/pi integral_0^infty t^(-1/2)(M+t)^(-1) M' (M+t)^(-1) dt,
    ||X'||<=||M'||/(2 mu^(3/2)).

Differentiating the identical resolvent again gives two ordered M' insertions
minus the single M'' insertion, yielding

    ||X''||<=3||M'||^2/(4 mu^(5/2))+||M''||/(2 mu^(3/2)).

Consequently, for R=X beta Dodd X,

    R'=X' beta Dodd X+X beta Dodd' X+X beta Dodd X',

and its second derivative retains X'', both mixed X'/Dodd' terms, both source
orders and the actual Dodd'' contact. This controls finite source differentiation
of the relative Neumann expansion uniformly on any closed metric box with
kappa<1. It is not yet an anchored positive Hessian-moment estimate; the next
material payoff is combining these differentiated insertions with the ordered
positive heat kernels and Ward filling, rather than paying the old high/low-mass
pointwise Cbar.

The continued consumer independently rebuilds native endpoints at +/-epsilon
and extracts M=(beta h+h beta)/2. It computes both mass-weight derivatives
without finite differences: if Y=sqrt(M), then

    Y Y'+Y' Y=M',
    Y Y''+Y'' Y=M''-2(Y')^2,
    X'=-X Y' X,
    X''=2X Y' X Y' X-X Y'' X.

These Sylvester equations use the full noncommuting mass. The reconstructed
R' and R'' are compared against independent perturbed relative endpoints;
freezing X must fail the SAME first-relative-current boundary. This is the
source two-jet supplier needed by the selected loop construction, not a claim
that its remaining anchored positive loop sum has already been bounded.

The continued native two-jet consumer exited0 in13.85s with empty stderr.
Relative first/second source-jet errors were3.9545434e-9/1.3213568e-8.
Freezing relative mass weights failed with relative current error.259909423.
The explicit native Tr(j^2) identity agreed exactly in this execution.
The100-term relative Neumann action error was1.0757005e-15, below its
uniform analytic remainder.0037307949. Both-sided time-varying-mass
coercivity error was4.7746347e-16 and the inverse-action norm was.5938305873.
This second successful attempt is retained separately from the first;
there were no failed scientific execution attempts in this pair.

## 7. Executed successor: even-band anchored loops replace Hlow and Htail

This continuation follows parent selection at successor-plan revision
`3ca65a9f55e5621b0b73f03592e5d2c8fef040a870fcabf2159af830fb0bc858`,
without changing pinned physical requirements or publishing a shared plan.
The first representation considered here was direct positive domination of
every relative Neumann path by its absolute hopping row sum. It is insufficient:
native kappa is about.923, whereas the unweighted odd row/mu bound is about1.8.
An energy-norm contraction is not an absolute path contraction. That failed
estimate is retained; no current-variance or CE claim is made from it.

The replacement keeps the selected relative even-band representation, but
uses the beta-cone cancellation to obtain its pointwise envelope. Thus the
cone is a kernel supplier inside the selected construction, not a switch to
another physical law. All source derivatives below are the actual source
derivatives of h, M and Dodd; no mass-derivative/contact terms are discarded.

### 7.1. Exact odd interpolation and anchored current/contact identity

Set h_lambda=beta M+lambda Dodd, lambda in[0,1], solely as an identity tool.
Every h_lambda has the SAME positive M and hence the original canonical
positivity/chirality lemma applies. Beta conjugation sends lambda to-lambda
and commutes with all admitted scalar temporal sources/interfaces. The actual
ordered determinant W_lambda is consequently even in lambda. With
K_lambda=D_lambda^(-1), the exact finite-spatial continuous-time identity is

    W_1-W_0=integral_0^1 dlambda integral dt Tr[K_lambda Dodd].

Tr Dodd=0, so equal-time prescriptions agree. At lambda0 the trace is zero
at EACH anchor, since K0 is beta-diagonal and the inserted Dodd is beta-odd.
Equivalently the regularized relative trace-log starts at two odd insertions.
This fixes normalization without an auxiliary mass endpoint. It is not an
unjustified interchange with canonical projection.

Allocate each spatial odd bond half to each endpoint, and keep the onsite odd
piece at its site. Call this Hermitian local insertion D_x. Then sum_x D_x=
Dodd, every D_x is beta-odd, supported at x and its six neighbours and covariant
under site-U1. A localized odd-coupling derivative proves its density real;
tracing K_lambda D_x proves its exact anchored gauge invariance. Thus its
source Hessian annihilates a pure gauge in EITHER slot, as needed by rooted
Ward filling. The h_lambda interpolation is a calculation, not a replacement
Hamiltonian/state or a light-reference energy subtraction.

For a source eta let j_lambda=beta M'+lambda Dodd' and
c_lambda=beta M''+lambda Dodd''. Differentiating the anchored density gives

    2 Tr[K j K j K D_x] - Tr[K c K D_x]
      -2 Tr[K j K D_x'] + Tr[K D_x''].

The two distinct source orders are understood before symmetrization. The
last TWO terms are indispensable: the insertion's own source jets are
physical, not a fixed spectator. Temporal currents are -i site projections;
spatial first/second jets are the same native Peierls currents/contacts.
This identity is what the new ordered consumer integrates, not an equal-time
resolvent substituted for an ordered evolution.

### 7.2. A small point-kernel supplier from the true beta cone

For the declared true metric box define

    mu=m sqrt(s_floor),
    u=sqrt(G00_upper-s_floor)                (universal box),
    u=.002 sqrt(G00_upper)                  (optional tight box),
    d=(m+6r)u+3v sqrt(Gjj_upper),
    Ch=3[r sqrt(G00_upper)+v sqrt(Gjj_upper)].

These are actual tilted-odd and endpoint row bounds, not isotropic changes to
G. Spatial conjugation by any real alpha-Lipschitz site weight changes h_lambda
by E_alpha with ||E_alpha||<=xi=Ch expm1(alpha). This holds on a torus with
the actual wrapped distance, including central seams, for ALL lambda and all
real phases. Site temporal U1 factors commute with the weight and beta.

Choose 0<xi<mu/2, alpha=log(1+xi/Ch), and

    R=(d+xi)/(sqrt(mu^2-2mu xi+d^2)+mu-xi)<1,
    c_beta=(1-R^2)/(1+R^2), zeta=mu-xi.

Here is the load-bearing cancellation. In beta blocks an unperturbed
Hermitian h_lambda has positive diagonal mass blocks and adjoint offdiagonal
odd blocks. The stable graph Z obeys its Riccati equation. At a largest
singular pair Zb=Ra, Z* a=Rb, the real part of the offdiagonal term
Z D Z-D* is bounded by d(1-R^2), NOT d(1+R^2). Diagonal positive mass contributes
at least2mu R. The entire weighted perturbation contributes at most
xi(1+R)^2. Thus backward stable-graph invariance and forward unstable-graph
invariance follow from

    2mu R=d(1-R^2)+xi(1+R)^2.

Scalar temporal connection contributes a purely imaginary term at this
singular pair and its unitary jumps preserve the cone. Finite-horizon graphs
with terminal/initial zero graph remain in this ball; bounded subsequential
limits give the dichotomy. Applying the same argument to the complementary
time direction gives both graphs and their transversality.

On the stable graph q=<psi,beta psi>>=c_beta||psi||^2. The weighted generator
has Re(beta h_alpha)>=zeta, hence q'<=-2zeta||psi||^2<=-2zeta q.
Stable evolution therefore has norm <=c_beta^(-1/2)exp(-zeta(t-s)).
The complementary graph has the reverse-time bound. Their oblique projection
norms are <=c_beta^(-1); multiplying yields c_beta^(-3/2). Periodize the
infinite-line dichotomy with the SAME scalar AP/holonomy interfaces. The
image sum is bounded by

    ||K_alpha(t,s)|| <=
       2 c_beta^(-3/2)/(1-exp(-zeta T)) exp[-zeta d_T(t,s)].

Transversality/coercivity fixes this as the actual inverse, not a different
boundary Green function. Consequently at T>=T0=max(1,log2/zeta), with
gamma=zeta/2 and C=4 c_beta^(-3/2),

    ||K_lambda(x,t;y,s)|| <= C exp[-alpha d_X(x,y)-gamma d_T(t,s)].

Every lambda, arbitrary real spatial history, scalar temporal source/kick,
global Haar phase, central seam and full true well share this bound. It does
not involve B, an evaluation-interval delta, a high-mass cutoff, or a mass
integral. In particular the large Cbar in Hlow is genuinely replaced.

### 7.3. Positive anchored two-source measure and explicit moments

Use the scalar point envelope in7.2 on each of the four differentiated
contractions in7.1, retaining the actual one/two-site incidence supports.
Take their positive contractions BEFORE summing anchors; integrate lambda
over[0,1]. Contact derivatives retain collapsed-time unit mass. This constructs
a background-independent positive measure on the anchor and both source
offsets, with actual first/contact-anchor incidences included.

Let J=max(max_j ||F_j||,1), A=3(ru+v sqrt(Gjj_upper)); A is an ANCHOR ROW
jet bound, paying all six incident links rather than one link. Set

    P=(16/gamma)[1+2/expm1(alpha/2)]^3,
    H_triangle=16d (2C)^3 J^2 exp(6alpha) P^2,
    H_contact=8d (2C)^2 J exp(4alpha) P,
    H_anchor_first=16(2C)^2 J A exp(4alpha) P,
    H_anchor_contact=8(2C) A exp(2alpha),
    H_even=H_triangle+H_contact+H_anchor_first+H_anchor_contact.

The rank-four trace, both source orders and incident orientations are paid
by8/16; A already sums the anchor's incident link norms. Two independent
translated source offsets cost P^2 for the triangle, one costs P for the
contact/mixed contractions, and neither costs P for the insertion contact.
The factors exp(2alpha), exp(4alpha), exp(6alpha) pay endpoint supports in
one-, two-, three-kernel contractions. As in AppendixA's incidence argument,
half of each kernel exponential is retained after spending the anchored
diameter moment; P sums the other half. Tonelli yields

    integral exp[diam_Q(anchor,source1,source2)] dnu_even <=H_even.

The anchor itself may include one native bond endpoint; the inherited
minus16 seam support allowance includes this radius-one insertion. Applying
the SAME rooted Ward filling and paid torus/circle cutoffs consequently gives

    CE_bulk=16 H_even/(e^2 gamma^2),
    CB_bulk=4*64^2 H_even P4(alpha).

Hlow, Mtail and Htail no longer occur. This is a new anchored two-source
envelope for W1-W0, not a numerical bound inferred from a finite response.
Evenness sets the relative trace normalization; it is not used to erase any
contact or an absolute-value term in these positive contractions.

### 7.4. Own the mass-only ordered finiteT/canonical remainder

W0's filled whole-half trace is exactly2 Tr(M)T, independent of spatial phases.
Its actual ordered forward/backward beta-sector propagators contract by
q=exp(-mu T), even though M(t) at different times need not commute. After
extracting the filled trace and centred temporal background, the four V-mode
logdet remainders are logs of I plus these contractions with the SAME unitary
holonomies. For any quotient source, native mass currents satisfy

    ||delta X(t)|| <=sqrt(1+3r^2 G00_upper)||eta(t)||_2,
    ||delta^2 X(t)|| <=r sqrt(G00_upper)||eta_A(t)||_2^2.

Ordered Duhamel with contraction factors, followed by the logdet derivatives,
therefore bounds the source Hessian remainder at q<=1/2 by

    |W0_remainder''| <=16 V [T Lmass^2+cmass] exp(-mu T)||eta||_2^2,
    Lmass^2=1+3r^2 G00_upper, cmass=r sqrt(G00_upper).

The mass reference's finiteT current is not zero, and no unprojected boundary
is substituted for Haar. Using qmin^(-2)<=N^2/16 pays an additional

    N^5[T Lmass^2+cmass] exp(-mu T)

in BOTH field-strength coefficients. Add it to the full Rstar spatial/time
seam from section3, then choose N0 so their total is <=
epsilon=max(CE_bulk,CB_bulk)/100 for every N>=N0,T>=tau N^2,
tau=max(T0,(alpha+1)/gamma). The CLI includes this thermal term in the
logarithmic bisection; its monotonicity floor also includes
N>=sqrt(7/(2mu tau)). No spatial/temporal/canonical seam is dropped.

Set CE=CE_bulk+epsilon, CB=CB_bulk+epsilon and use the SAME fixed positive
Maxwell prescription hE=16max(CE,CB/.1^2,1), UE=1/(2hE), K=.1^2hE.
All exact conditional canonical/Haar/harmonic/centre and full-field lift
ownership from section4 survives. The actual mixture covariance/score proof
consumes these coefficients, not the static Wick variance as an interacting
variance surrogate. Current-score variance suppliers can use H_even with
their own conditional anchored-to-score Schur and seam accounting.

### 7.5. New consumer boundary, not baseline revalidation

Run `noncompact_wilson_photon_envelope.py --size 5 --loop-only`. It skips the
previous native baseline. The new consumer evaluates the continuous ordered
three-history native500-mode determinant source Hessian at lambda=0,+1,-1,
then integrates the independently differentiated exact lambda first-response
at4/8 Gauss nodes. Every source perturbation rebuilds M and Dodd from native
endpoints. Agreement must reach the actual determinant difference W1''-W0'';
the mass-only finiteT response is retained separately. It also demands
ordered even-band parity and anchored pure-gradient Ward cancellation, and
requires the frozen-odd-anchor variant to FAIL that same Ward boundary.
The actual ordered Green at two sites/times is compared with the finiteT
version of the derived cone envelope. Finite native execution validates these
identities/adverses, not global box admission or the entire positive-moment
theorem by sampling.

### 7.6. Observed payoff, selection and next constructions

The new `--loop-only` consumer naturally exited0 in163.8089845s with empty
stderr, source SHA256
`a4569e78021fa359363d074c55ccdf044654b7481d1e7440c29587840507d62a`,
unchanged through exit. Exact native W1''=2.2124292408713515e-8 and
W0''=-1.1180884082093906e-8. The even-loop difference3.330517649080742e-8
agreed with4-node/8-node independent ordered lambda integrals at relative
errors2.1504204e-5/1.1764900e-7. The4-node value was insufficient for the
selected2e-5 consumer accuracy; the8-node refinement succeeded and BOTH
values remain in stdout. Ordered lambda-sign parity error was8.78645e-12.
Correct anchored Ward Hessian was-3.46945e-12; frozen anchor jets produced
-8.15948028e-4 and were rejected at the same physical boundary. The actual
ordered offsite Green norm1.37473764e-4 was below its finiteT point envelope.

For the optional tight true-G box, new H_even=1.3764192778061936e22 is
110397.6762 times smaller than section3's H. New CE=1.7001720160178975e29,
CB=1.7169956146408055e31 give UE=1.8200395932017268e-35 and
K=2.7471929834252887e32: UE gains15349.3935 over the prior tighter family.
N0=108999 and tau=26.53508134094653 pay BOTH loop seams and the actual
mass-only ordered thermal remainder. T0=8.4304763242; rho remains1/16.
The scalar budget chosen from six legal candidates has xi=.35mu,
alpha=.0908468342, gamma=.0411096096 and cone radius.8510269752.

For the SAME loose predecessor metric box, H_even is4.25845 times smaller,
but its longer spatial moment offsets that gain: new UE=6.94653209e-60 is
only.497154376 of section3's UE. **Retain section3's stronger same-metric
family.** Do not label this regression an improvement or erase it. The new
kernel is materially useful on the actual near-native tight Gram box, not a
uniform superiority claim over every looser metric box.

PhotonPacketSupplier received H_even/CE/CB and the anchor-current/contact
ownership immediately. Evaluating its proposed anchored-score variance
formula with the new tight H,UE,alpha,gamma gives a110397.7 improvement
factor; that is an arithmetic capability comparison, NOT independent proof
of the supplier's incidence/Schur constants. MobileChargedPacket receives the
larger conditional UE and reduced temporal onset, with Euclidean/current
scope kept distinct from real-time transport. Neither supplier waits for
publication or global tight-box admission.

Two next constructions can now change the native reach:

1. **Selected: rigorous native metric-box admission.** Use the actual
   full-ten-coordinate Gram stationarity, its strong-convexity floor, cubic
   continuum parity and analytic shifted-grid quadrature error to enclose
   the continuum minimizer and all64 finite-N offsets. Produce an explicit
   N_metric for s>=.40,G00<=.43,Gjj<=.065,tilt<=.002. This removes a real
   membership hypothesis from the stronger family; it is not a repeat sample
   or an independent-review/approval gate. Original m=.2,r=.12,v=.3,kappa=2
   and native whole filling stay unchanged.
2. Diffusive marked Wilson-loop moments: combine the ordered positive Wilson
   heat and paired odd insertions with the source two-jet supplier to pay
   field-strength filling by actual walk fourth/time moments rather than
   the exponential-distance P^2 P4 overcount. Compare its new CE/CB and
   current-score coefficient on the SAME admitted parameter box.

The selected successor assumes actual Clifford-well minimizer convergence
and the source-owned strong-convexity law, but needs a quantitative certified
quadrature residual, not the N5 samples. No further coupling or physical
state hypothesis is introduced. The conditional loop theorem here remains
an analytic derivation with finite exact-consumer evidence, not an externally
accepted campaign claim.

## 8. Physical-regime fork: near-Dirac uses the existing non-relative cone

Parent's later decision input changes applicability/selection, not the truth
of the anchored result. Section7 ALREADY uses a non-relative beta cone: no
new kappa-removal theorem is claimed by this section. The previous proposed
metric-admission continuation is now owned independently by NativeMetricAdmission;
this pair does not duplicate it. DiracPhotonEnvelope independently owns the
physical/link-unit photon family and packet consequences. This pair supplies
the primitive hatted kernel and actual finite-cutoff native current consumer.

### 8.1. Exact applicability algebra and what Gordon g means

With isotropic G=diag(g0,gs,gs,gs), spacing a, physical m,r,v fixed, the native
Wilson mass and odd spatial operator are

    M=sqrt(g0)[m+r L_A/(2a)],
    Dv=(v/a)sum_j Phi_j S_j,
    delta_W(a)=a m r g0/(v^2 gs),
    kappa_v^2=3v^2 gs/(2mr a g0)=3/(2delta_W(a)).

Thus relative kappa<1 requires delta_W>1.5. At the carrier-level Gordon
nonrelativistic boundary the scalar Wilson inertia changes the kinetic mass
but not the Dirac spin-Pauli numerator, giving

    g_Gordon=2/(1+delta_W)<.8.

This is NOT a measured physical mechanical-spin g, nor proof of the open
physical spatial-spin join. Near g=2 therefore cannot be advertised as the
reach of the uniform relative-Neumann method. In a tilted Gram define the
trace comparator delta_trace=3mr a s/(v^2 Tr Gss); the exact relative spatial
bound still satisfies kappa_v^2=3/(2delta_trace), and its additional mass tilt
only tightens relative admission. The scalar Gordon formula is explicitly
an isotropic comparator, not an exact anisotropic/tilted observable.

`bulk_cutoff_matching.py` independently defines dimensionless bare mass am,
physical stiffness2/a, reconstructed m_eff=m sqrt(g0),r_eff=r sqrt(g0),
v_eff=v sqrt(gs), and its Wilson debt a m_eff r_eff/v_eff^2. Its near-cutoff
trajectory is an explicitly different bare family; its free-sea Gaussian
matching is not an all-background phase proof. Reported a=.001 debt about
.000741 consequently has kappa_v about45, not an admitted relative series.
The source's own limits exclude fixed-nonzero-coupling UV closure. This work
does not rerun or retroactively reinterpret that reported matching consumer.

### 8.2. Reusable primitive kernel in one unit system

Use t_hat=t/a, h_hat=a h_physical, b_hat=a a0, spatial link angle theta.
The public helper `primitive_cone_envelope` takes keyword-only

    m_hat,r,v,s_lower,g00_upper,
    spatial_diagonal_upper=(g11_upper,g22_upper,g33_upper),
    tilt_root_upper,xi_fraction.

Here m_hat=am_physical, and tilt_root_upper bounds sqrt(G00-s), NOT the
dimensionless relative tilt sqrt((G00-s)/s). Every bound is on the true full
Gram in every claimed well; the helper neither replaces G nor certifies
global metric membership. The native Wilson/odd endpoint formula is the
same as section7 with m replaced by m_hat. Choose0<xi_fraction<.5.

Returned scalar keys are:

    m_hat,r,v,mu,Ch,odd_row,odd_jet,native_jet,xi,
    alpha,gamma,cone_radius,cone_angle_inverse,point_kernel_bound,T0,P,H,
    triangle,native_contact,anchor_first,anchor_contact,
    CE_bulk,CB_bulk,log_Rstar,
    mass_only_jet_squared,mass_only_contact,xi_fraction.

`odd_row` is d, `odd_jet` the whole anchor row A, `native_jet` J,
`cone_radius` R and `cone_angle_inverse`1/c_beta. Alpha is a lattice-site
distance rate; all time/energy/current moments are HATTED. The helper returns
no chosen N0, physical UE/K or state/packet certificate. The existing a1
`even_loop_family` now consumes this SAME primitive; there is no independent
duplicate kernel formula for DiracPhotonEnvelope to maintain.

It returns the source triangle/native contact and BOTH insertion-jet moments
in H, and the original full spatial/circle seam prefactor in logarithmic form.
The additional ordered mass-only thermal Hessian tail uses its Lmass^2 and
cmass values. This is precisely the information the physical family owner
needs before any canonical/Haar/harmonic mixture and field-unit conversion.
Full conditional/canonical ownership is still `noncompact_maxwell_phase.md`
sections1–5, not an unprojected, harmonic-frozen determinant.

For clarity, in physical time the SAME cone has

    mu_phys=m sqrt(s), xi_phys=xi_hat/a,
    d_phys=d_hat/a, Ch_phys=Ch_hat/a,
    alpha_physical=alpha_site/a, gamma_physical=gamma_hat/a.

The actual kernel's finite-circle prefactor is invariant under t=a t_hat.
As a tends to zero, d_phys/mu_phys grows roughly1/a, the cone approaches its
boundary and the moment constants deteriorate. Survival at every fixed a>0
does not yield a uniform fixed-coupling UV result. DiracPhotonEnvelope owns
the exact conversion of raw site moments into physical-volume photon/current
coefficients; no a1 UE/K or tight a1 metric box is borrowed at a=.001.

### 8.3. Native finite-cutoff kernel/current consumer and adverse

Run `--cutoff-cone --spacings .01 .001 --size 5`. This mode skips both earlier
a1 consumers. It constructs each spacing's actual full-ten-coordinate
clamped native centre123 G with physical stiffness2/a, genuine500-mode
three-axis endpoints, arbitrary real curved phases and scalar physical a0
histories. The new public primitive is called with m_hat=.2a and each true
anisotropic diagonal/Schur/tilt bound. Physical durations are a times
[.31,.43,.37]; the ordered Green remains exact continuous segment evolution.

The consumer verifies the spacing-correct delta/kappa algebra; it demands
kappa>1 for the near-Dirac branch while the weighted cone remains R<1.
On the actual flat native static mass congruence it computes the skew odd
spectral radius. It must exceed one and make an incorrectly reused20-term
relative Neumann series fail the actual inverse boundary. The correct ordered
curved-history offsite Green must lie below the finiteT cone envelope.
Native physical current/contact are independently differentiated from the
same physical endpoints at every spacing. No on-shell/physical g, continuum
phase coefficient, global metric admission or cutoff packet claim follows
from these finite-matrix observations.

Real Pyright/AST capability was shared with both independent owners:
`local://issue222-real-pyright-probe.py` uses the existing1.1.414 stdio server,
actual project-root configuration and scientific interpreter, real symbols
and diagnostics. No reinstallation, OMP restart or repository LSP config is
needed. Ownership/source bindings, not repeated baseline validation, join
the primitive and physical-family consumers.

### 8.4. Observed finite-cutoff applicability and released interface

The new native500-mode two-spacing consumer exited0 in5.0194068s with empty
stderr and unchanged source
`ce9e5e1dcc6168daf0503d836da284f4990378be449c7696c131cbe829140c50`.
At a=.01 the true finite clamped Schur was.2892442859, delta_trace=.0074612662,
kappa_total=14.1800038, cone R=.9981461467 and the isotropic carrier comparator
g=1.985187984. At a=.001 they were.2878904084,.0007396162,45.0353978,
.9998152672 and1.998521861. These are actual finite-G input/comparator
observations, not a physical mechanical-spin measurement.

The actual flat static relative spectral radii3.59227172/3.79431377 exceed
one; the incorrectly reused20-term relative series produces inverse errors
1.23360577e11/3.69898524e11. The cone nevertheless bounds the actual ordered
curved-history offsite Green norms2.64878225e-4/2.65183177e-4. Its conservative
finite-circle point envelopes2.79118437e7/8.91411625e9 and hatted anchored H
1.01099489e48/3.31949672e60 show the cost explicitly, rather than implying
useful uniform UV bounds. Physical spatial rate alpha_site/a stays about
.05566/.05553. Native same-current finite-difference errors were about
2.66e-11; second-contact errors1.60e-6/1.55e-6 passed the declared2e-6 scale.

The reusable helper and native kernel/current implementation were stable
through this execution. Subsequent output-only repair moved nested metadata
into its construction for Pyright, and corrected the cutoff mode's recorded
seed to its actual2221071. The initial result and all source/input hashes are
retained; final changed-source real diagnostics returned zero. Primitive
function arguments/returned units and kernel calculations did not change.
The final frozen consumer receipt supplies the resulting exact source hash.

NativeMetricAdmission independently reports its a1 all64 box certificate
N_metric=25 in `local://issue222-native-metric-state-packet.json`; the sole
publisher now selects the stronger a1 tight-loop family at its existing
N0=108999. This is the other owner's admission result, not a second
certificate produced here. It removes the previous a1 metric-membership
hypothesis without making the strong-Wilson relative method near-Dirac.
The physical-regime fork and its actual primitive consumer are shared with
DiracPhotonEnvelope; that owner alone selects/scales the cutoff photon family.

Two material next constructions are (A) a source-marked Wilson-energy/heat
moment envelope retaining spatial quadratic-form coercivity instead of the
global odd-row cone angle and exponential-distance P²P4; or (B) consume the
admitted a1 supplier in the actual finite-width two-receiver current/recoil
join already independently owned by PhotonPacketSupplier. Select A for this
kernel branch: it must use the exact non-relative inverse and contact-complete
anchored lambda source, not a forbidden kappa<1 premise at near-Dirac.
The observable payoff is a smaller hatted H/current-score moment and less
conservative same-cutoff CE/CB, with physical consequence selection remaining
DiracPhotonEnvelope's responsibility. No more a1 metric work is owned here.

## 9. Wilson-energy/source-marked heat replaces the global cone angle

The selected successor retains the actual Wilson spatial quadratic form in
the source evaluator. It leaves `primitive_cone_envelope`'s implementation
and return contract unchanged and adds `primitive_wilson_energy_envelope`.
No new relative-Neumann premise, physical family selection or metric
certificate is introduced.

### 9.1. The marked instantaneous Wilson inverse has a 3D heat bound

At every time M=sqrt(s)[m_hat+r L_A/2] is scalar in the four Clifford components.
Its instantaneous inverse diagonal is dominated by the free Wilson heat
return, because every phase path has unit modulus. Write sigma=r sqrt(s_lower)
and mu=m_hat sqrt(s_lower). On an N-axis circle,

    p_N(t;0,0)=(1/N)sum_k exp[-sigma t(1-cos(2pi k/N))]
       <=1/N+sqrt(pi/(8 sigma t)).

Here 1-cos(2pi k/N)>=8k^2/N^2 for wrapped |k|<=N/2; extend the positive
Gaussian sum to all integers and dominate its nonzero terms by an integral.
The three independent axis returns multiply. Since (x+y)^3<=4(x^3+y^3),
and a heat return is also <=1, split its resolvent integral at

    t_heat=4^(2/3) pi/(8r sqrt(s_lower)).

The short part costs at most t_heat, the long t^(-3/2) part2t_heat, and
the finiteN constant part4/(N^3 mu). Therefore, for N>=N_floor>=3,

    max_x <x|M_A^(-1)|x>
       <=C_M=3t_heat+4/(N_floor^3 mu).

The second term is mandatory. Spatially uniform light phases cancelling the
central seam yield the genuine free Wilson zero mode. A finite torus cannot
borrow only the infinite-volume inverse diagonal; the consumer below makes
that omission fail on the actual native operator.

Positive-kernel Schwarz gives |(M^-1)_xy|<=C_M. Hence for a spin-block vector f,

    ||M^-1/2 f||^2 <=C_M (sum_x ||f_x||_4)^2.

This is a source-marked spatial heat estimate, not a global1/mu bound and
not a replacement of the matter operator by a Gaussian photon vacuum.

### 9.2. Two native source windows and exact mass-energy coercivity

Conjugate D_lambda by a periodic spatial/time Lipschitz weight w. Its
spatial rate alpha satisfies Ch expm1(alpha)=xi; its temporal derivative
has magnitude <=gamma. The SAME scalar AP/holonomy boundary is retained.
The exact coercive form is

    Re(beta D_lambda,w)>=M-(xi+gamma)I>=eta M,
    eta=1-(xi+gamma)/mu>0.

Thus ||M^1/2 D_lambda,w^-1 beta M^1/2||<=1/eta. This uses the entire
two-sided mass congruence; time-varying M produces no omitted derivative.
The original beta/chirality canonical positivity remains available before
conditioning/projection. No stable-graph angle is needed for this inverse.

A short source transport U_w(u,s) is generated by the actual weighted
native h_lambda and scalar temporal connection. Its block-row and block-column
l1 norms are <=exp(B_w |u-s|), where

    B_w=(m_hat+3r)sqrt(g00_upper)+Ch exp(alpha)+gamma.

The scalar temporal source is removed by its actual onsite unitary gauge;
its magnitude or Brownian derivative is not bounded. Every offsite endpoint
and seam is retained. Choose delta<=min(T0/8,1/(2B_w)).

For a point source f_x, average U_w(u,s)f_x/delta over a length-delta source
window. Its squared M^-1/2 spacetime norm is <=
C_M exp(2B_w delta)/delta. Do the SAME with the adjoint native target
transport over a target window. Keep their cyclic order on the chosen Green
branch, not just disjoint supports. In a chart with t>=s, use u<s,v>t when
t-s<=T/2; use u>s,v<t when t-s>T/2. Rotate the chart for the other order.
With delta<=T/8 these pairs never cross v-u=0 modulo T. Blind outward
windows near opposite chart seams DO cross that jump and do not reproduce
the point evaluator. Actual transports, not frozen point vectors, are required.

Away from a source jump, the Green composition identities make their
bilinear average EXACTLY the point Green matrix element. One-sided source
windows recover the prescribed plus/minus boundary value at the jump.
Periodic lifts account for AP/holonomy signs/unitaries without changing norms.
Applying the mass-energy inverse bound to both averages proves

    ||K_lambda(x,t;y,s)|| <=C_energy exp[-alpha d_X-gamma d_T],
    C_energy=exp(2B_w delta) C_M/(delta eta)
             <=e C_M/(delta eta).

This is the new point evaluator. The earlier cone constant can still be
chosen if smaller; both are honest bounds on the SAME operator. The heat
source estimate avoids the global odd-row cone-angle loss, while the
current and contact remain those of the original native determinant.

### 9.3. Reusable moments, finite-volume hypothesis and physical meaning

`primitive_wilson_energy_envelope` has the original primitive's keyword
inputs plus `size_floor`. It uses the same alpha/gamma budget and hatted
units, returns the same triangle/native-contact/anchor-first/anchor-contact
moments with C replaced by min(C_cone,C_energy), and additionally returns

    size_floor,Wilson_heat_split_time,
    mass_diagonal_infinite_part,mass_diagonal_zero_mode_part,
    mass_inverse_diagonal_upper,source_transport_B,source_interval,
    energy_relative_coercivity,energy_point_candidate,
    cone_point_candidate,H_gain.

The existing four moment terms scale in C with powers3,2,2,1. H,CE_bulk,
CB_bulk and log_Rstar are updated by those exact positive-polynomial
scalings; no source/contact is deleted. The mass-only thermal seam is
UNCHANGED. The family owner must retain N>=N_floor as well as its existing
metric/spatial/temporal onset. For a chosen N_floor64 the already enormous
wm29 onset implies this condition, so no previously admitted thermodynamic
domain is silently lost.

DiracPhotonEnvelope owns new physical UE/K/C_J selection. A smaller H at
the SAME old rotor coefficients shrinks the score part of current variance
quadratically; it does not change the static Wick contribution. If instead
CE/CB and rotor coefficients are retuned at the SAME alpha/gamma/metric
box, H->H/F entails UE_hat->F UE_hat, score C_J->C_J/F, so the matched
product UE_hat C_J and normalized physical packet defect are approximately
INVARIANT. The primary retuned payoff is stronger admitted coupling, not an
invented improvement of that product. No fixed-coupling UV claim follows.

### 9.4. Actual native marked-source consumer

`--energy-marked --spacings .001 --size 5` skips all older consumers. At the
actual full-ten clamped native G it constructs the allowed harmonic history
cancelling central link phases, identifies its Wilson mass with the true
free torus mass, and checks its inverse diagonal against C_M. Dropping the
finite-volume zero-mode term must fail that SAME native inverse boundary.

On three curved/noncommuting native histories it independently checks every
instantaneous mass inverse diagonal, then constructs the continuous ordered
Green. Two source/target windows use the actual forward/backward/adjoint
native transports; their point-evaluator identity must reach the actual
offsite Green. Frozen window vectors must fail the same identity. The actual
point kernel must fit the finiteT energy envelope; short periods use
delta<=T/8 explicitly, not the large-family T0 interval. The anchored H gain
is computed on the SAME box/alpha/gamma as the earlier cone primitive.
The near-seam pair additionally preserves the chosen AP branch by placing
its windows inward. Blind outward windows with the actual periodic lifts
must fail the SAME point-evaluator identity, so AP signs alone are not
mistaken for cyclic-order preservation.
This is execution of the marked source mechanism, not another baseline
phase/Pauli/metric check.

### 9.5. Observed consumer and unchanged-cutoff consequence

The stable consumer naturally exited0 in7.679891818s with Python3.12.3,
NumPy2.5.3, SciPy1.18.1 and source
`4b9f10c87d9dfb055a2a718e25cfad23ffc4261d6b8ad48a211b1ddf412957ca`.
It used actual500 modes, central123, a=.001, seed2221072 and durations_hat
(.31,.43,.37); all earlier baseline consumers were skipped. Exact receipts:
`local://issue222-wilson-energy-final-receipt.json` and
`local://issue222-wilson-energy-final-stdout.json`.

The actual free Wilson inverse diagonal81.0073218901 exceeds the infinite
part46.1062312361; retaining the finite zero mode gives344.305369976.
Three curved histories give diagonals6.6006173061,6.5519643557,6.6229193868.
Actual native mass/free-Wilson identity error is1.85087e-16. The transported
point evaluator error2.50714e-15 contrasts with frozen-window error.09611745.
At the AP seam cyclic-order preserving windows give1.99834e-15, whereas
blind outward windows—even with exact AP lifts—give.01062505. Actual
offsite Green norm.000444881307 fits the finiteT envelope7982.622014.
The SAME native box/alpha/gamma anchored H improves104985848.38fold.

Separately executing the public helper on DiracPhotonEnvelope's universal
a=.001 box with N_floor64, xi_fraction.35 gives C3651.61012483 instead
of221998188.7545; H4.69993620550e58 instead of1.05605624412e73, a factor
224695867761579.12. This is a derived same-box moment improvement, not
a numerical coupled-ground observation or a physical UE/K selection.
The independent physical-family owner retains its metric admission,
N>=max(64,N0), unchanged mass thermal remainder and complete seam payment.
The old primitive-cone body and API are unchanged; historical source-bound
receipts remain evidence of their actual versions. Real Pyright557symbols
and zero diagnostics are recorded in
`local://issue222-wilson-energy-language-final.json`.

Selected next construction: use the same mass-energy point/extended-source
estimate for weighted spatial l2 Green columns, then contract the positive
anchored triangle before its Ward moments. This targets the two independent
pointwise spatial volume sums rather than another improvement of C alone.
An alternative is physical-cutoff interval metric admission replacing the
loose universal Schur/diagonal box; that belongs to the metric supplier.
Neither selection is already an improved photon/charged packet theorem.
