# MinSet exact centres and the local full-normal correlated trial

Governing plan: issue-222-fullfield-closedrotation-continuation-20261006,
revision 7954ba4c18ed7d90fdc3e6cbd3bf75c3dc80d568cae4781d73dadd61e041a718;
general template 89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe.
This helper does not modify the shared Hamiltonian/state or the Noether consumer.
The campaign remains open. A quantum ground is not used.

## 1. The discriminating colour-centre question

Use the original endpoint hopping, not an unsymmetrized vertex. For a positive
finite-N ten-Gram minimizer G, let Phi=G^(1/2) Gamma and
h(k)=sum_mu x_mu(k) Phi_mu, E(k)^2=x(k)^T G x(k). The actual heavy occupation
is the whole top 2V cluster of h (the Hamiltonian is minus h). Thus
P+(k)=(I+h(k)/E(k))/2, including its identity term. For a nearest positive-axis
path F_j and generator T with Tr T=0, the actual link derivative is

    d_j h = i/2 [ exp(ik_j){F_j,T} - exp(-ik_j){F_j^dagger,T} ].

The h-dependent trace is zero POINTWISE:
Tr h{F,T}=Tr {h,F}T=0, since both h and F belong to the SAME four-dimensional
Clifford span and their anticommutator is scalar. The identity-part trace is
not generally zero pointwise. Its sum is zero because sum_k exp(ik_j)=0 on
every real-shifted N-grid, N>=2. This last step matters: there is no residual
exp(-cN) alias divided by E, and no Lambda times discarded alias force.
All45 uniform colour first variations are EXACTLY zero for every such Gram,
not merely at a cubic/symmetric Gram. Translation invariance makes the single
edge first variation the same total divided by V, so nonuniform link first
variations also vanish. Alignment and plaquette first variations are zero
because their squared residuals vanish at constant Phi and central flat U.

Consequently the material joint centre is already (Phi(G_N(m,s)), U_central),
with actual finite-N G_N and actual central holonomy. No joint colour shift
or Lambda change is needed. This conclusion would be false for the deliberately
wrong one-sided hopping vertex: its trace retains a nontrivial sea denominator.
The skew N5 theta001 quarter-centre is the positive adverse control; the zero
shift is a symmetric control. These are native finite-volume controls, not
approximations to the astronomical Gram or its MinSet ordering.

For the field first derivative, strict Gram stationarity cancels the metric
part. The Clifford spectral-defect first derivative is zero. Hence the complete
60-coordinate field first derivative vanishes as well.

## 2. All nonuniform normals, not a uniform sweep

Let g=1/20, Gbar=1/2 be the existing actual all-real-shift N>=25 Gram enclosure.
The original coefficients are x0=1/5+(3/25)sum(1-cos k_j), xj=(3/10)sin k_j.
Define Cx^2=(23/25)^2+3(3/10)^2, Emax=sqrt(Gbar Cx^2),
Delta=2(1/5)sqrt(g). The exact Laurent design D for unordered products x_mu x_nu
has degree at most four per coordinate. Its shifted-grid mean equals its
integral at every N>=5. The rational eta=1/Tr(D^-1)>0 is therefore uniform in
all shifts and all legal N, not a selected momentum sample.

The independently executed integer330x60 Gram/trace-free-Clifford certificate
has kernel15, exactly the gauge image, and positive spectral gap2. Using this
WEAKER gap rather than changing to the older gap4 normalization yields

    alpha = 2 g^2/Gbar min(2, eta/(2 Emax^3)).

It bounds the complete45 uniform physical normals, including the35 non-Gram
normals. The factor g^2/Gbar follows by transporting both tensor legs through
G^(1/2), then taking distance to the transported gauge orbit before choosing a
slice. The heavy spectral defect Hessian is positive; no occupation constraint
on separate k blocks is imposed away from the centre.

At each Fourier q the alignment map is

    L_j = ( (1-exp(iq_j)) I_60, -D_Phi on colour link j ).

Generators satisfy Tr Ta Tb=delta_ab/2. D_Phi^dagger D_Phi>=g/2.
The field mass is1 and colour mass is1/.6, because the lawful kinetic term is
.3 chi E4^2, not .15 chi E4^2. In mass-normal coordinates the nonzero singular
squares of L lie above min(sum_j|1-exp(iq_j)|^2, .3g). For nonzero periodic q,
sum_j|1-exp(iq_j)|^2>=16/N^2. Thus

    b = (1/5) min(16/N^2, .3g)

is a conservative alignment bound on every stiff normal. At q=0 its kernel
has45 physical field normals; at every other q its only kernel is the15 gauge
directions. After all15V Gauss equations the dimension is90V, split45 soft and
90V-45 stiff. Plaquettes only improve this bound.

The whole original endpoint sea Hessian/contact plus local quartic Hessian
has operator norm at most the existing rational L=leading_upper. It follows
from cross-cluster denominators >=Delta and endpoint vertex Hilbert-Schmidt
bounds; it is independent of q and V in normalized Fourier coordinates.
The q=0 soft block is >=alpha Lambda. Mixed soft/stiff blocks cost at most
L Lambda. Young/Schur, and the nonzero-q alignment bound, give

    K_actual >= (alpha/2) Lambda P_soft + (b/2) Lambda^(5/4) P_stiff

provided Lambda^(1/4)>=4(L+2L^2/alpha)/b. This is an all-q analytic argument.
The optional all125-q N5 native probe is a separate adverse/Ward control.
It cannot establish the legal-N quantifier.

## 3. Full Haar quotient and an actual local coordinate domain

Take the gauge slice D_Phi^dagger deltaPhi_x=0 at every site. Its45V field
coordinates and45V links give90V coordinates. The orbit derivative contains
both i[alpha_x,Phi] and the link difference alpha_x-alpha_y. Its positive
mass inertia is >=g/2, even at q=0. In these raw slice coordinates the quotient
metric is NOT the flat diagonal field/link metric. If M is the original mass
and O is the complete orbit map, its covariant metric is the pullback of

    M - M O (O^dagger M O)^-1 O^dagger M.

Use its positive square root at the centre to mass-normalize the slice. This
constant linear normalization includes all gauge/link correlations. Its45
uniform soft directions are a tangent subspace; the actual curved chart need
not map the entire uniform-flat manifold to that linear subspace. The kinetic
cometric at its origin is identity. In the current two-bank state heavy plus
light plus Ref number is4V+4 in BOTH banks, so the residual SU4 centre acts
trivially. Full equivariance, including site Haar factors, implements Gauss.
Charged/reference fibres retain their fixed `qRef=nRef-4`.

For quantitative control set d=105V, rho=sqrt(4VGbar),
s=min(1/4,sqrt(g/2),sqrt(b),Delta/8), h=2^20 d^2(1+rho+Cx),
Jchart=(h/s)^256 and

    r=min(1/64, min(alpha,b,Delta)/(2^20 d C), Delta/(64Vh)),

where C is the finite comparison constant defined below. This further shrinking
pays nonlinear potential curvature, not just a cometric/density comparison.

The primitive maps are product link exponentials, the gauge-slice implicit
inverse, the inverse orbit-inertia matrix and the constant mass normalization.
Exponential derivatives through order8 are bounded by q! 2^q d(1+rho).
The inverse differentiated recurrence is
D^q F^-1=-F^-1 sum_(nontrivial ordered partitions) product(D^j F F^-1),
with at most q! 2^(q^2) terms at q<=8. The least orbit/slice singular value is
at least s; matrix inverse singular values are at least s^2. Neumann expansion
on radius<=s/(64h) preserves those bounds. Applying the recurrence to the
implicit gauge map, inverse inertia, and their finite products is bounded by
q! (h/s)^(256(q+1)). The displayed radius lies inside these Neumann discs.
The complete log density has at most d eigenvalue/determinant contributions;
its variation is <=d Jchart r<2^-20. Half-density conjugation therefore retains
all Haar/coarea factors, with density ratio in[1/2,2], and contracted jet bounds
absorbed below. The cometric lies in[1/4,4] on this explicitly local domain.
These statements concern a regular tube; no global Gribov chart is claimed.

The entire fixed-number heavy projector has a gapDelta/2 on this tube. Its
rank-one Slater section is obtained by actual Kato transport of the WHOLE top
2V projector. Contour/Neumann differentiation bounds its derivatives by the
existing whole_heavy_projector_jets. The frozen normal certificate used the
matter representation bound4V+3. The current two-bank construction uses the
safe bound4V+8; this costs less than2^8 in its eighth power. The unchanged
contracted residual headroom2^64 below covers it. Quotient density, Born-Huang,
Berry and angular terms are retained, not set to zero. The frozen comparison
constant is `C=2^32 d^8 Jchart^8 (1+1/alpha+1/b)^8 (4V+3)^8`;
it is not retagged as a newly executed certificate.
The radius is chosen BEFORE the trial, independently of its quantum width.

The nonlinear local floor is proved without asserting a Hessian bound on a
fixed-radius tube multiplied by Lambda^(5/4). On the uniform-flat manifold,
V0(z,0)>=v+alpha|z|^2/4. The off-manifold first derivative of V0 is <=C|z|,
and its stiff Hessian is >=-C. The nonnegative alignment/plaquette potential
satisfies V1(z,xi)>=b|xi|^2/4 throughout the displayed small tube. Therefore
Young's inequality gives

    Lambda V0+Lambda^(5/4)V1-Lambda v
       >= alpha Lambda |z|^2/8+b Lambda^(5/4)|xi|^2/8,

when Lambda^(1/4)>=16(C+2C^2/alpha)/b. The helper separately pays this
nonlinear onset; it is not substituted by the weaker centre Schur onset.

## 4. Correlated Gaussian, higher moments and cutoff

In mass-normal coordinates the kinetic is (chi/2)p^2 at the origin. The
general original cometric G0 INCLUDES chi. K is the true full90V ORIGINAL
heavy/scalar/colour Hessian: LOCAL+SEA+CONTACT at Lambda, and alignment plus
the `.7 Lambda^(5/4)` colour plaquette. Bounded light/Ref, Coulomb and
connection operators remain remainders, not a claimed scalar whole-H ground.
The positive correlated precision is

    A=G0^-1/2 (G0^1/2 K G0^1/2)^1/2 G0^-1/2,
    A G0 A=K, Sigma=(1/2)A^-1.

The state is
psi proportional to exp(-y^T A y/2). Equivalently put G0=chi I after the mass
normalization. Its exact quadratic energy is
(1/2)Tr[(G0^1/2 K G0^1/2)^1/2], bounded by
Z Lambda^(3/8), Z=90V sqrt(L+10+Gbar). No diagonal covariance is substituted
for the correlated covariance. The CLI checks a noncommuting three-coordinate
Riccati example and rejects the more expensive diagonal precision.

Operator monotonicity of inverse square roots and the Hessian lower bound give
soft variance <=2 alpha^-1/2 Lambda^-3/4 and stiff variance
<=2 b^-1/2 Lambda^-7/8. Set M=4d(1+alpha^-1/2+b^-1/2).
Gaussian Wick gives
E||z||^(2k)<= (2k-1)!! M^k Lambda^-3k/4 and
E||xi||^(2k)<= (2k-1)!! M^k Lambda^-7k/8;
mixed moments follow Cauchy-Schwarz. Odd centered cubic polynomials vanish,
including correlated ones. All higher even moments and momentum insertions
are therefore computable before any cutoff.

The stiff potential vanishes for every uniform field quadruple and central
flat link, including non-Clifford quadruples. Its first stiff derivative
vanishes there. TWO-stiff-leg Taylor divisibility holds in coordinates
adapted to that entire manifold, not automatically in the actual raw curved
chart. In the latter the cubic `Lambda^(5/4) xi z²` has operator residual
`CN Lambda^1/16`; the associated pure-soft quartic has residual
`CN Lambda^-1/4`. The geometric flat identity is retained without erasing
these actual-chart words. A symmetric radial cutoff kills odd Gaussian
expectations, not their operator norms. Metric, density, frame and all
momentum contractions remain in the contracted majorant.

Take cutoff1 inside r/3, zero outside2r/3, symmetric in y, with first two
derivatives <=2^12/r^2. The Gaussian tail is bounded by
exp[-r^2 Lambda^(3/4)/(512M)]. To obtain this dimension-uniform tail, use
the Gaussian exponential moment at parameter1/(4 Tr Sigma), bounded by exp(1/2),
and pay the interior r/3 threshold before removing that prefactor. The exported
j=16 inverse-exponential bound
16! x^-16 pays cutoff derivatives, normalization, the Lambda^(5/4) potential
and all displayed finite polynomial moments. No Gaussian is asserted to be
compactly supported before this operation.

A conservative NEUTRAL trial coefficient is

    EexCoeff=16[ Z+C(1+M)^4+VN^2+1 ], Eex<=EexCoeff Lambda^(3/8).

The VN^2 allowance covers neutral light/Coulomb/coexact zero-point/harmonic
costs only with A's exact photon/harmonic construction and coefficients; the
helper separately exports the whole-light norm bound rather than inferring
actual creator normalization. A must use its own light/ref/SOURCE contribution.
The source-created trial coefficient is NOT established by a neutral energy
bound. The corrected field construction here is an actual correlated quasimode,
not a request for a quantum ground or a bare existence descriptor.

## 5. Lawful Root floor and what this does not establish

On the union of these actual local wells the heavy/scalar/stiff potential is
bounded below by Lambda min_s v(m,s), with the tubes shrunk by the Hessian and
finite-jet onset. Whole light is bounded below by minus the exported local
whole-CAR norm, and total beta norm is1/16. Thus

    e_m=Lambda min_s v(m,s)-C_light-1/16

is the lawful fullRoot local Dirichlet floor. ALL512 histories remain in H,
SOURCE and the neutral right action. Initial m amplitudes are1/8; only exact
finite-N minimizer-set s receive normalized equal tie weights. There is no
Lambda-weighted excited-history population, no s superselection, no floor
averaging and no N5 choice of the astronomical minimizer ordering.

The helper compares its explicit required log2 Lambda against the conservative
unchanged selected-tower lower2^315. It also pays
8 EexCoeff Lambda^(3/8)<=V Lambda. Since the original Root Utrial exceeds
12V Lambda, the neutral local Dirichlet energy window is below that unchanged
Root separation, subject to A supplying its exact remaining sectors. This does
not retune Delta or beta. It is not a proof that the cutoff state is already a
whole-H spectral vector.

Unresolved consumer obligations are explicit: the actual source-created trial
and Hermitian raw-quartet creator Gram; higher SOURCE/fullQ graphs; full-domain
exterior/union Agmon and Dirichlet-to-whole-H leakage; physical Noether/stress
and NEW scattering transfer. No global coercivity of the unbounded field
space, global ground, nonuniform Haar integral computation or Born accuracy
is inferred from the native controls. The prior global-floor alias shortcut
remains FAILED. Its failure is not used to erase any actual finite-N offset.

## 6. Successor: actual union shells without spectral counting

This extension is under governing revision
f2ef2b878e04b7b1cca51e9f4dff5c8f6009e73d7e9c2ca55c00ce338533d9d3.
Sections1–5 and their original executed receipt remain frozen evidence for
their earlier scope. This successor adds the following operator theorem;
it does not rerun or retag the unchanged all125-q result.

### Actual centres, union domain, and colour gluing

The ordinary heavy symbol is x(k+theta/N), with
theta=pi(m/2+s). Thus its distributed per-link phase is pi(m/2+s)/N.
The source's ordinary per-edge centre phase is separate data and is not divided
by N by this theorem. The same field/Root law and every actual offset remain.

For fixed m put Omega_m=union_s B(c_m,s,R_D), R_D=3r/4, with c_m,s the actual finite-N
field/colour quotient centre. No pairwise s separation is used. For each
flat Root history the scalar restoration is measured from its own c_m,s;
since v_m,s>=min_t v_m,t, it implies the weaker bound measured from distance
to the UNION, provided Omega_m is inside each relevant regular local chart.
This last inclusion needs justification; it cannot be inferred merely from
equal labels or small unmultiplied energy offsets.

It is paid by a finite-grid analytic bound. For every G in the real positive
enclosure [g I,Gbar I], E(k)^2=x(k)^T G x(k) has a zero-free complex strip of
width2^-20: its real value is >=g/25, and changing k by that imaginary amount
changes it by less than half that bound, by the elementary sine/cosine
derivative bounds from Cx. The Gram-gradient integrand x_mu x_nu/E is analytic
there and bounded by2^16. Shifted-grid Fourier aliases therefore change the
gradient between any two shifts by at most2^22 exp(-2^-20 N). Strong Gram
monotonicity with kappa=2, and the positive square-root lift, bound the
centre difference in normalized coordinates by

    2^32 V Jchart exp(-2^-20 N).

At the selected N, exp(-x)<=p^p/x^p with p=2^20 gives a binary log upper
below -60 million, while the regular radius has log about -1.79 million.
Thus all eight centres for one m are within r/8 of one another, although
their differences need NOT be smaller than the INTERIOR cutoff radius
R_out=r0 Lambda^-1/4. Their offsets are not equated or multiplied-away.
The regular Dirichlet union radius R_D=3r/4 contains the actual field trial
cutoff support<=2r/3 and, with the centre spread<=r/8, lies inside every
relevant regular r tube. Choose r0<=r/128 so the thin R_out cutoff is strictly
inside that regular domain. The actual matrix lower follows on the SAME
regular Dirichlet domain from each history's local bound; it is not narrowed
to a shrinking domain by this theorem.

The off-code input has a direct pointwise proof, stronger than merely citing
a spectral Q-gap. The authoritative Root quartic derivation is
bosonic_original_law_join.md45–57 and
quartic_vacuum_coefficients_from_l1(V,kappa,C_H,C_L). At every Phi/U/A,
all auxiliary matter blocks have the entrywise coefficient norm lower
-C sum_x r_x; Young's inequality and the quartic leave a nonnegative
half-quartic and a remainder -B_N(Lambda). No field, photon or harmonic
kinetic energy is used to obtain this bound. Every nonflat contractible
Root syndrome costs at least Delta, hence the off-code POTENTIAL is

    Q V0 Q >= (Delta-B_N)Q
       = (Utrial+window_old+beta+G)Q.

The flat-code projector is coordinate independent; the full even/gauge
connection preserves its cycle constraints, so retaining all normal kinetic
energy is legitimate. The original Delta already uses the actual heavy/light
coefficient caps. A must bind its changed-light C_L to those SAME caps; a
bare-law cap cannot silently stand in for the improved light law.
Since e_m<=4V Lambda and Utrial>=300V Lambda, this off-code block dominates
kappa Lambda d_union^2 on the displayed local union. On flat code the
individual history bounds above apply. Finally V_beta+beta I>=0 on the
FULL auxiliary space, including its P/Q cross terms. Thus the full matrix
local form lower follows without hard projection or a new Root parameter.
The pointwise calculation, not spectral separation alone, pays this step.

For ordinary m gluing, the actual Z4^3 holonomy classes are distinguished by
Wilson traces Tr W_j=4 i^(m_j). A word of length N has a telescoping derivative
bound <=2^16 N Jchart in the local normalized quotient metric. Consequently
different m gauge orbits are at distance >=1/(2^16 N Jchart). Our supported
r0 tubes are smaller than one eighth of this proven lower bound. Holonomy
cutoffs can therefore make the64 initial m supports disjoint; this is a
derived separation, not an assumption about s or a phase-label assertion.
Within one m, initial different s components are orthogonal by the actual
Root projectors P_s even when their physical field tubes overlap. The raw
m coefficients1/8 and exact normalized MinSet tie vectors consequently have
norm1 before filtering. Filtering and interior cutoff still use ONE final
normalization; actual unequal filter norms are not reset per branch.

### The full local form input

Let A bind the actual full neutral or charged local form and floor so that

    H_D-e_m >= (chi/2) D_y^* g(y) D_y + kappa Lambda d_union(y)^2,
    0<g(y)<=4, chi=Lambda^-1/2,

on the SAME union domain, with kappa=min(alpha,b)/16. All actual matrix Berry,
Gauss, Haar and heavy/light/reference connections remain in D_y. Root beta
has been spent only through its norm in e_m. Coexact EM, Coulomb and harmonic
forms are retained. If another sector has a negative bounded term, its exact
bound must be included in e_m BEFORE using the positive interval [0,Ecut].
The arithmetic API does not certify this input by accepting a number.

Take P=1_[0,Ecut](H_D-e_m), Ecut<=Ccut Lambda^3/8, with the ACTUAL combined
sector/trial coefficient. A's actual filtered vector f(H_D-e_m)tau is in ran P
because f has that spectral support. A neutral energy estimate does not imply
the source-created vector belongs to the charged P. That transition is
handled separately below.

### Noncommuting spectral-window IMS inequality

Let eta be a scalar Lipschitz cutoff supported where the barrier is >=B.
The covariant IMS identity gives, for psi=P v with ||v||=1,

    B ||eta psi||^2
       <= Re <psi,eta^2 H_E psi>
          +(chi/2) <psi,|d eta|_g^2 psi>,

where H_E=(H_D-e_m) restricted to ran P. Do NOT commute eta and H_E.
Do NOT replace the first term pointwise by Ecut||eta psi||^2.
Instead use the OPERATOR norms

    |<eta P v, eta P H_E v>|
       <= ||eta P|| ||eta P H_E|| <= Ecut ||eta P||^2,

then take the supremum over v in the barrier inequality. This yields

    (B-Ecut)||eta P||^2 <= (chi/2)|| |d eta|_g P ||^2.

Nested cutoffs eta_j have their gradient supported where eta_(j-1)=1;
therefore the right side is bounded by
2 chi ||d eta_j||_infinity^2 ||eta_(j-1)P||^2.
No spectral rank, eigenvector basis, compact resolvent, heat kernel trace,
Jensen step or ground-state premise enters this argument.

Use at most M=512 centre cutoffs, and form their PRODUCT for the exterior
of the union. A cutoff transition of width w has derivative <=4M/w.
Starting outside radius R_out/8 and ending by R_out/4 leaves space for n transitions.
There B>=kappa Lambda R_out^2/2^10. Choose

    n=floor(a Lambda^1/4), a=2^-16 sqrt(kappa) r0^2/M.

The recurrence contracts each operator norm by at least1/4 once Ecut<=B/2.
A conservative shell conclusion, including floor/integer loss, is

    ||1_shell P|| <= 2^(-floor(a Lambda^1/4)).

Replacing nonsmooth characteristic functions by the nested Lipschitz cutoffs
only weakens this conclusion. Overlapping or coincident centres cause no
singularity: products, not a nearest-centre Hessian, define the cutoffs.

This OPERATOR theorem applies to arbitrary tensor spectators. In particular
there are2(V-1) coexact oscillator modes, each with infinite occupation, the
unbounded real harmonic direct integral and every Bloch kappa fibre, all
light/reference configurations, and the entire auxiliary Root space. Their
infinite spectral multiplicity requires no counting prefactor. Uniformity in
kappa follows from the same local form bound, not from setting kappa=0.

### Actual localized kinetic graph and boundary commutator

For a smooth shell cutoff zeta with slightly wider support, the same
noncommuting-window bound applied to the localized IMS form gives

    ||sqrt(chi) zeta D_y P||^2
       <= [2 Ecut+C chi/R_out^2] epsilon_shell^2.

Here both ||zeta P|| and ||d zeta P|| are bounded by the previous wider
shell estimate. Matrix connection terms are not discarded; the leading
commutator of a scalar cutoff depends on the actual cometric alone.

Define a smooth interior union cutoff by
gamma=1-product_j(1-gamma_j), with gamma_j=1 on the inner third and0 outside
the outer two-thirds of its centre tube. This is valid for coincident and
overlapping centres. The finite chart jets bound its gradient and metric
Laplacian; the helper exports conservative constants with the actual512,
full dimension and Jchart included. On the whole microscopic local form,

    [H,gamma]=-(chi/2)[Delta_g gamma+2(d gamma)^* g D_y].

Thus

    ||[H,gamma]P||
       <= C_gamma [chi/R_out^2+sqrt(chi)/R_out sqrt(Ecut+Cchi/R_out^2)]
                   epsilon_shell
       <= C_boundary Lambda^3/16 epsilon_shell.

Extension by zero of gamma psi has support strictly inside the Dirichlet
domain and belongs to the corresponding whole-H operator domain when psi
has its spectral graph regularity. All field and matter matrix terms agree
there; no alternative microscopic Hamiltonian is introduced. Duhamel gives

    ||exp(-itH) gamma psi - gamma exp(-itH_D)psi||
       <= |t| C_boundary Lambda^3/16 epsilon_shell,

uniformly for psi in that actual window. The post-cutoff norm is at least
sqrt(1-epsilon_shell^2); normalizing multiplies errors by its reciprocal.
This is an actual Dirichlet-to-whole-H theorem, conditional ONLY on binding
the actual full local form/window inputs above, not a separate Agmon premise.

### SOURCE, photon and higher graphs

For the actual coordinate SOURCE B, let M0 bound its operator norm and M1
its COVARIANT normal first jet between neutral and charged Gauss fibres.
M1 is not an ordinary45-normal coefficient derivative with the matter
connection omitted. It can instead be paid conservatively BEFORE Gauss:
sum all60V Cartesian derivative norms squared plus .6 times all45V colour
left-Lie derivative norms squared. The full SOURCE is Gauss invariant and
maps the complete light/reference modules, so projection to the physical
quotient contracts this mass-weighted ambient gradient. A directional
Frechet bound must spend105V when converted into a summed gradient bound.
Both endpoint Phi frames, nonlinear whitening/eigenvector/polar derivatives,
every signed colour edge, full Ref4 and the original SCREENED denominator
belong in that ambient bound. A prototype quaternion/default unity source
does not establish it.

The literal screened_marks source has no coexact EM dependence: its EM phase
is Green div A plus the separately owned harmonic phase. On the exact initial
product trial tau, B tau therefore retains the full coexact vacuum, and
a_g^dagger B tau is exactly a one-photon factor with the TRUE B norm.
No ambient degree-one field cutoff is imposed on that source-created state.

An actual source-trial upper, with Mtheta the harmonic theta=N A_h derivative
bound and Ztau=||B tau||, follows directly by the product kinetic form:

    Eex(B tau/Ztau) <=
      [2 M0^2 Eex(tau)+2 chi M1^2+2(UE/N) Mtheta^2
       +2 M0^2(C_L,q+beta+C_Coul)]/Ztau^2.

Here Mtheta<=4 M0 for the legal3N+2 path, C_Coul can conservatively be
9 UE V N^2, and the allowed reference-number penalties vanish on the actual
neutral/charged reference fibres. Heavy/scalar/alignment/colour plaquette
and Root-face penalty terms commute with the even-light coordinate SOURCE;
the noncommuting beta cost is bounded, not erased. Photon birth adds at most
4c; bounded light expectation is already included. For Lambda>=1 every
displayed fixed coefficient is absorbed into a real Csource Lambda^3/8.
This gives A a material source-trial construction once its actual jets and
TRIAL norm are bound. The original screened legal-word centre Gram lower
9/32 I and the normal variation M1 sqrt(E||y||^2) can pay Ztau. They do not
automatically pay ZXi after the full-H filter: trial, filtered neutral, and
actual raw-quartet creator norms are different quantities.

The corresponding unfiltered B energy commutator includes the harmonic term
2 sqrt((UE/N)E) Mtheta, its second derivative, the actual covariant normal
M1 sqrt(2 chi E)+chi M2/2, and the bounded light/reference/Coulomb/beta
commutators. There is no coexact momentum term for THIS source. Its use in a
photon-created or higher graph still requires the actual mixed-state graph,
not a product of unrelated first moments.
Then, without falsely commuting beta and B,

    [H_q,gamma] B P_n
       = B[H_n,gamma]P_n-chi (d gamma)^*g(D_y B)P_n

on the common local chart. It is bounded by the same shell exponential with
coefficient M0 C_boundary+M1 C_gradient; normalization divides by the actual
source norm Z_B. If a photon creator is inserted, its actual source-first
graph norm must multiply this estimate; it is not a bounded unit multiplier.
The scalar gamma commutes with pure EM/harmonic coordinate/momentum operators,
but their unbounded creator graph still requires the actual state bound.

### Historical fixed-m ordinary SOURCE and true-photon control

The following NEW owner receipt supplied FIXED-m local graph coefficients:
`local://issue222-minset-full-sector-receipt.json`, stdout SHA256
`7ba57e84d11689072effa13aa4ec2bcf578379eccd2cbbadb1bd415bfcbe205b`.
Its SAME worst-light floor is `e_m=Lambda min_s v(m,s)-CL-beta` in both
neutral and charged fibres. It retains the photon zero-point energy in
`H_D-e_m`; that energy is counted ONCE in the neutral trial excess, not
subtracted again when bounding the photon number graph. Its actual common
coefficient is `C*=2^1787662` and ONE preparation scale is
`L=4 C* Lambda^(3/8)`. Normalized neutral, SOURCE-created and true
photon-created trial excess coefficients are respectively at most
`2^1787349`, `2^1787660`, and `2^1787661`; the proof intermediate `a†tau`
is included. Its claimed global coherent64 admission was WITHDRAWN by Main:
per-m energy floors cannot be substituted for the ONE physical global floor.
The equations below remain fixed-m trial controls and proof-graph identities,
NOT acceptance of the all64 once-prepared physical SOURCE.

Let `Kn=HD,n-e_m`, `Kq=HD,q-e_m`, and `fn=f(Kn/L)`, `fq=f(Kq/L)`.
Here f is the existing smooth preparation function, equals1 belowL and
vanishes above2L. `fq` is a mathematical comparator only. The physical
state remains `a†_g B gamma fn tau` followed by its true global norm:
there is NO charged postfilter or raw-quartet whitening. The same e_m on
both sides cancels in the intertwining commutator without removing any
actual m-dependent physical evolution phase. B preserves the colour class;
the fixed-m graph constants are uniform in m and actual minimizing s ties.
That uniformity does NOT turn separate physical fn,m into the ONE global fn.
It does not require beta to commute with B. Section8 gives the global
weighted-shadow replacement and distinguishes local proof graphs from filters.

Fourier Duhamel, on the actual trial rather than an arbitrary low-window
vector, gives

    ||B fn tau-fq B tau||
       <= Cf,1/L sup_s ||(HD,q B-B HD,n) exp(-is Kn/L) tau||.

The actual full ambient first jet, harmonic first jet, Laplacian second
jet and bounded light/Ref/Coulomb/beta commutators bound this right side.
The normal term is bounded by `sqrt(2 chi E_tau) ||g^(1/2) D B||`
(equivalently `2 sqrt(chi E_tau) M1` in Guard's HALF-cometric convention);
the harmonic momentum factor is at most `sqrt((UE/N) E_tau)`.
E_tau is conserved under Kn.
Neither factor uses a ground-state, global neutral variance, projected
Dirac sea, scalar EndMark or even-complex-source assumption. The corrected
Guard packet SHA256 `46a7f18f20f2919940357ab162b38ebef9ff1a65b6104eb89f393ee3a41e9685`
has raw q0..q6 logarithmic bounds `[152,288,425,563,700,838,975]`,
FULL ambient canonical M1-squared log876 and harmonic Mtheta-squared
log307. Its DIFFERENT comparison-frame covariant M1-squared log890938
may conservatively be used in the boundary insertion; these conventions
must not be interchanged. The withdrawn trace-score2eta raw proof is not
used or retagged.

The owner proves normalized RAW-trial `Z_tau=||B tau||>=1/2`, from the
full stochastic-Ref/raw-quartet centre identity, all legal signed words,
and the normalized correlated Gaussian variation. This is a TRIAL lower,
not an assumed filtered creator lower. Since the actual normalized Btau
energy is at mostL/4, Markov's positive-energy inequality gives
`||fq Btau||>=sqrt(3)/2 Z_tau`. Consequently

    ||B gamma fn tau||
       >= sqrt(3)/2 Z_tau-epsilon_B-M0 epsilon_shell.

Division by `||gamma fn tau||<=1` cannot reduce this lower. The unchanged
tower pays both errors, yielding the owner's actual filtered SOURCE norm
lower1/4. The full-Ref trace has no spurious factor1/4 averaging.

For an unbounded TRUE coexact creator, commute only the bounded B and gamma,
which are coexact independent, and use the EXACT order

    a† B fn tau-fq B a†tau
       = (B fn-fq B)a†tau+B[a†,fn]tau.

The first error is proved DIRECTLY on the exact initial one-photon trial
`a†tau`; it is never obtained by multiplying a neutral norm error by a
bounded-unit substitute for a†. The second error is Fourier Duhamel for
the actual creator graph. With normalized mode g,

    [H,a†_g]=a†_(omega g)+J_g,
    ||a†_(omega g) Un(s)tau||
       <=omega_max sqrt(1+E_tau/omega_min),
    omega_min>=4c/N, omega_max<4c,
    H_D-e_m>=H_ph>=omega_min N_ph.

This positive full-oscillator energy form controls the actual N+1 graph.
No photon-count cutoff or spectral rank is used. On the regular tube,
the original one/two-link Weyl phases have unit norm, and their canonical
mode leg has l2 norm at most `sqrt(UE/omega_min)`. The actual light l1
coefficient cap thus gives
`||J_g||<=4 CL sqrt(UE/omega_min)`. The unchanged exact UE makes this
finite leg particularly small; it is not a replacement of UE or c.
This argument retains beta, whole Root space and arbitrary photon
occupations during Un evolution. It assumes ONLY the actual regular
Dirichlet form/domain already bound by the owner, not the driven flow.

Combining the direct one-photon SOURCE error and the creator commutator
error admits the physical `a† B gamma fn tau` into the charged ordinary
window up to its quantified tail. CCR also gives
`||a†_g B gamma fn tau||>=||B gamma fn tau||`, so its TRUE global
normalization has the same lower1/4. This does not make a† a bounded
operator and does not pay higher Noether/Born-LIGHT graphs by a product
of separate first moments. The final whole-H Duhamel theorem adds the
charged ordinary-window tail to the union shell/commutator error.
Driven rotation, SOURCE-relative Born-LIGHT comparison and calibrated
nonforward scattering remain separate owner obligations.

The historical m-local common coefficient, corrected covariant source
boundary jet `M0log152/M1log445469`, local lower1/4 and target4938bits return
required `log2 Lambda>=71487408`, below the unchanged selected lower2^315.
The capsule28094 and owner7ba remain unchanged historical controls; their
global admission claim is not served as current truth. The helper's generic
`actual_sector_inputs_bound=False` remains false. Global once-prepared
normalization and SOURCE admission are supplied only by the successor owner.

For an ARBITRARY neutral low-window vector (not the actual trial route just
proved), if B P_n is not already in the charged low window, a separate charged spectral tail is required. With a genuine separation
d=Ecut_q-Ecut_n+(e_q-e_n)>0, the Sylvester estimate is

    ||P_(>Ecut_q,q) B P_(<=Ecut_n,n)|| <= ||[H,B]P_n||/d.

Higher-source graph bounds can improve the tail. They must use the ACTUAL
normal/EM/harmonic/Root commutator and raw creator normalization. The normal
shell theorem cannot pay this charged transition by itself. Normal graph
insertions P(H_D-e_m)^j gain only Ecut^j; genuinely charged or mixed graph
insertions require A's finite jets and off-code/source graph, not a trace bound.

For the actual long rotation clock
T=1008124871316563/2, the ordinary theorem yields a separate time factor.
It does NOT establish the driven trajectory H_dr=H-omega(t)G_phys:
that trajectory can boost normal momenta far outside the ordinary window.
B must supply its actual driven shell/kinetic graph or a SOURCE-relative
comparison cancelling the common large phase. In particular [G_phys,P_s]
and [G_phys,gamma] are additional actual operators; neither commutation nor
a driven Root gap is inferred from the ordinary result.

### Quantified unchanged tower and native consumer

Write a>=2^-A and a finite time/source/normalization prefactor <=2^F Lambda^p.
The elementary inequalities exp(-x)<=2^-x and 2^(L/8)>=L^2 for L>=128 show
that the shell/commutator error is below2^-b_target whenever

    L=log2 Lambda >=max(128,8(A+2),16(F+p+b_target+4)),

and the separate barrier condition Ecut<=B/2 holds. The helper includes that
condition, T6 or the exact long clock, source amplification, and the default
absolute target UE/(V2^40) with4096 further amplitude bits. The unchanged
selected lower L>=2^315 pays this explicit finite selector. There is no
allocation of2^(constant*V), no photon spectral count and no new Lambda.

The new --barrier-normal-receipt CLI executes a one-normal ORIGINAL N5
heavy/scalar potential curve with TWO actual reflection-related finite-N
Gram centres, overlapping centre tubes and a nonzero1/16 history coupling.
It filters an explicit trial with the actual matrix H_D, forms the union
cutoff, measures the actual low-window shell and commutator OPERATOR norms,
and evolves the corresponding larger-domain SAME potential for unscaledT6.
A sharp inner cutoff is an adverse boundary control. This finite-difference
one-normal execution is not the astronomical90V state, formal fullRoot, or
an infinite EM/harmonic execution. Those spectators are covered by the
operator theorem only after A binds the actual full form and sector window.

## 7. Actual massive-normal momentum driver and zero-ended return

This is support for B's SAME physical clock, not a replacement Hamiltonian,
an added SU2 rotor, a GU pulse, or ordinary-window inheritance. B still owns
the complete physical generator, EM/current exchange, coupled Root consumer,
SOURCE-relative ordered path and final neutral right action. The construction
below pays the FAST Phi/colour normal part using the original mass metric.
Its comparison eigensection is not imposed as the physical preparation or
called a ground of the whole photon/light Hamiltonian.

### 7.1 Derive the matter fraction from the complete Gauss mass

At the actual cubic finite-N Clifford centre, take the physical Cartesian
row tangent `dotPhi1=-Phi2`, `dotPhi2=Phi1`. Solve ALL15 SU4 orbit columns D
for c with `D c=dotPhi`. The solution is the actual S3, not an assigned
Pauli matrix. With `Tr(Ta Tb)=delta_ab/2`, the cubic solution has
`||c||²=2` and `Iphi=||D c||²=2gs`. In a normalized Fourier profile q,
the colour orbit columns are `(1-exp(iq_j)) c`; therefore

    Ilink=sum_j |1-exp(iq_j)|² ||c||²/0.6,
    a=Iphi/(Iphi+Ilink), Ired=Iphi Ilink/(Iphi+Ilink).

The original mass is `diag(I60,I45/0.6)/chi`. For raw field-only tangent
`v=(D c,0)` and FULL orbit `O=(D,Lq)`, its mass-horizontal projection is

    r=v-O(O* M O)^-1 O* M v=((1-a)D c,-a Lq c),
    r* M r=Ired.

M here omits the common factor1/chi. The field-only Gauss problem would
instead project v to zero. It loses the actual physical link normal.
No generator GU has been added: the link component is a derived projection
of the SAME field generator.

For relative angle u=phi_angle-link_angle, let Qg be the matter charge
in the Gauss convention `p_phi+p_link+Qg=0`. The exact Routh kinetic form is

    Hkin=chi (p_u+a Qg)^2/(2Ired)+chi Qg²/(2(Iphi+Ilink)),
    Gphi=p_u=(p_u+a Qg)-a Qg=P_u+a Qbody,
    Qbody=-Qg.

A local radial fibre gauge removes the constant connection in P_u. Thus
the physical driver has BOTH `-Omega P_u` and `-Omega a Qbody`.
The latter is derived from Gauss; it is not additional matter-spin hardware.
At q=0 the link inertia vanishes and there is no physical normal: the
remaining body coefficient is1. At hard q it is strictly less than1.
Consequently a hard mode does NOT automatically have a2pi minus.
For a smooth spatial alpha at the cubic centre the same derivation gives

    a alpha=(gs-Laplacian/0.6)^-1 gs alpha,
    ||a alpha-alpha||infinity
       <=||(-Laplacian) alpha||infinity/(0.6 gmin).

Positivity of the discrete screened resolvent proves this bound. The
physical body error is therefore a smooth-profile derivative cost, not
an assumed exact sign for every Fourier component. Actual noncubic G_N
uses the full positive15x15 orbit inertia; the scalar formula is the
native cubic control only. All finite-N alias coefficients are retained.

### 7.2 Solve the TRUE momentum-forced Gaussian

In mass-normal coordinates diagonalize the actual joint Hessian, NOT a
field-only Hessian. Let frequencies be omega_j and actual projected force
components b_j. The exact quadratic field driver is

    Hfast(t)=sum_j [chi p_j²/2+omega_j² y_j²/(2chi)
                    -Omega(t)b_j p_j].

For `theta=2pi(10u³-15u⁴+6u⁵)`, `u=t/T`,
`Omega=(2pi/T) w(u)`, `w=30u²(1-u)²`. BOTH Omega and its first derivative
vanish at both endpoints. The exact coherent displacement is

    alpha_j(t)=-b_j sqrt(omega_j/(2chi)) exp(-i omega_j t)
                   integral_0^t Omega(s) exp(i omega_j s) ds.

For z=omega T the complete endpoint integral is

    integral_0^T Omega(t) exp(iomega t)dt
      =2pi [60(exp(iz)-1)/(iz)^3
             -360(exp(iz)+1)/(iz)^4
             +720(exp(iz)-1)/(iz)^5].

Three integrations by parts, using endpoint w=w'=0 and
`|w''(0)|+|w''(1)|+integral|w'''|=300`, give

    ||alpha(T)|| <=600pi/(sqrt(2chi) T³)
                       sqrt(sum_j |b_j|²/omega_j^5).

The state returns to its original Gaussian up to its ACTUAL common phase
and a norm error at most this coherent norm. The legal normal bounds are
`omega_soft>=csoft Lambda^1/4` and
`omega_stiff>=cstiff Lambda^3/8`; endpoint errors accordingly have powers
`Lambda^-3/8` and `Lambda^-11/16`. This result never replaces the literal
momentum drive by an angle-centre drive.

At midclock, however, `pmean approximately Omega b/chi`. Its ordinary
kinetic energy is `Omega² ||b||²/(2chi)`, of order `Lambda^1/2`,
NOT the ordinary preparation's `Lambda^3/8`. The instantaneous driven
floor is the opposite quantity. Its leading scalar phase is

    Phi_inertia=||b||²/(2chi) integral Omega²
               =20pi² ||b||²/(7chi T).

It can be enormous and is not discarded. The exact scalar phase also has
the normal-acceleration term
`sum_j b_j²/(2chi omega_j²) integral (Omega')²`, plus the explicitly
bounded endpoint oscillatory and fourth-derivative terms. The soft
acceleration term has Lambda power0, not a negative power. These are
common field phases, not an imposed neutral invariance condition.

The CLI evaluates the exact polynomial integral and these phase terms
with110-digit arithmetic on the explicitly declared native mode inputs.
Legal N uses the bounds above, not a huge-phase floating extrapolation.
Its Gaussian second-graph adverse is also physical:
for `Gphi=pnormal+aQbody`, mean mu and variance sigma²,

    ||Gphi psi||²=mu²+sigma²,
    ||Gphi² psi||=sqrt(mu^4+6mu² sigma²+3sigma^4).

The latter is strictly larger than the former for sigma>0. Thus a q2
graph cannot be capped by the square of q1 on this ACTUAL normal Gaussian.
This is not a requirement that neutral variance vanish.

### 7.3 Branch-local normal resolvent, not T times a growing cubic residual

Write the accepted local normal quadratic as H2. Its zero-point scalar is
at most `C0 Lambda^3/8`, and its least excitation is at least
`csoft Lambda^1/4`. Gaussian Wick contractions give, uniformly over every
actual centre,

    <|y_soft|²><=C Lambda^-3/4,
    <|y_stiff|²><=C Lambda^-7/8.

In the actual raw chart the worst cubic is
`Lambda^(5/4) y_stiff y_soft²`, of order`Lambda^1/16`.
The flat-adapted two-stiff-leg word and the metric word
`chi y_soft p_stiff²` have order1. Odd Gaussian means vanish, but the
operator residual does not. Neither `T CN Lambda^1/16` nor a stale
`T CN` estimate proves nonlinear return.

An intermediate core has radius `epsilon Lambda^-9/32`, inside the unchanged
regular Dirichlet domain. Cauchy-Schwarz against the soft/stiff quadratic
forms bounds the one-stiff/two-soft cubic relatively by
`C Lambda^1/8 |y| <=C epsilon Lambda^-5/32`. On the first two normal
levels this gives `C Lambda^7/32`, below their `Lambda^8/32` separation.
The exterior barrier has power7/16, above the zero point3/8; nested IMS
costs power1/16. With the already stated local coercivity and chart bounds,
min-max isolates the branch-local normal lowest cluster and gives

    R=Qnormal(Hnormal_ms-Enormal_ms)^-1 Qnormal,
    ||R||<=2/(csoft Lambda^1/4).

This is a FIELD-normal branch comparator, not a gap of the coupled Root
matrix, the light sector, photons, or the physical whole Hamiltonian.
R tensors spectator identities. No actual preparation is restricted to
this cluster. Resolving the explicit raw residual gives dressing
`O(C Lambda^-3/16)`. For its odd cubic alone, the first-order energy is
zero and the second-order resolvent energy is bounded by
`C Lambda^(2/16-1/4)=C Lambda^-1/8`; this does not erase other retained
diagonal phases or prove the full physical clock.

The whole heavy occupied Slater band is treated by its own original
`Delta Lambda` resolvent. Heavy-body off-band words are bounded and cost
`O(C/Lambda)`; they are not mistaken for normal or photon gaps. Its
connections and full Haar/coarea half-density remain. The source commutes
the even heavy algebra and its Kato transport on the FULL auxiliary space,
as proved by the separate ambient1792-commutator capsule.

### 7.4 Curved metric, actual DY and accelerated quasimode

Let g(y) be the original quotient cometric and b(y) the ACTUAL projected
Phi/colour vector field. Before projecting, this field contains the
Cartesian row action, covariant nearest differences and the original
four-link colour clovers. Differentiating the projection includes
`DY-Dprime lambda`; neither the colour partner nor Dprime is omitted.
Product exponentials, the15V orbit-inertia inverse and the accepted
implicit chart give actual jets through8

    ||D^q b||<=q! CY^(q+1),
    CY=2^32 d³(N+1)(1+rho) Jchart^4.

This follows directly from at most four links per clover, nearest
incidence norm<=2, velocity norm<=N, and the existing chart inverse jets.
It is an all-mode bound on the original vector field, not a fitted
single-plane coefficient. The code records its logarithmic upper.

Complete the driven kinetic square. The leading scalar boost is
`exp(i Omega S0(y)/chi)`, with `dS0=M0 b0`. Nonzero curl of the remaining
one-form is NOT silently removed. After the boost its linear DY remainder
is a true quadratic y-p word. Its worst Gaussian matrix element is
`C Omega Lambda^1/16`; dividing by the normal gap gives a correction
`O(C Omega Lambda^-3/16)`. The corresponding virtual energy is
`O(C Omega² Lambda^-1/8)`.

The geometry-dependent inertial potential is
`-Omega² b(y)*M(y)b(y)/(2chi)`. Its linear force is of order
`Omega² Lambda^1/2`. Solve the actual nearby stationary centre by the
implicit function theorem:

    ycentre_soft=O(C Omega² Lambda^-1/2),
    ycentre_stiff=O(C Omega² Lambda^-3/4).

These are smaller than the respective Gaussian widths. The induced soft
centre-energy correction has Lambda power0, proportional to Omega^4;
it is retained in Efast(t), NOT bounded by a nonexistent negative power.
The shifted quadratic, including the actual noncommuting DY matrix, stays
positive when `C Omega² Lambda^-1/2` is small. Its least frequency remains
at least half csoft Lambda^1/4. All remaining cubic/density terms use the
reduced normal resolvent just established.

For a fixed branch comparator, a boosted/recentered instantaneous normal
cluster can be used in a superadiabatic construction, keeping its exact
scalar energy and Berry/acceleration phases. Boost differentiation costs
at most`Lambda^1/8`, while each normal resolvent removes`Lambda^1/4`.
This supplies a candidate three-step envelope

    Cfast (1+T+1/T)^8 Csource Lambda^-1/32,
    Cfast=2^512 d^32 CY^32 Ccompare^32
              (1+alpha^-1+b^-1)^64.

The CLI evaluates its finite arithmetic selector. This is NOT an
admission certificate for the actual nonlinear, Root-coupled physical
clock: its complete instantaneous-cluster derivative/diagonal remainder
and SOURCE-relative common-field cancellation belong to B's consumer.
The frozen exact quadratic forced return is a component control only.
In particular this ledger must never replace the actual growing raw
cubic by CN0, multiply it by T, and declare it paid.

### 7.5 SOURCE-relative and spectator boundary

The scalar normal boost commutes the ORIGINAL coordinate SOURCE and true
coexact creator. BEFORE estimating momentum, its mean cancels in the
relative driven commutator:

    chi g0 DB pboost-Omega b0.DB=0.

Here `pboost=Omega M0 b0/chi` and `g0 M0=I`. For the ACTUAL curved metric
the remainder is `Omega [g(y) M0 b0-b(y)].DB`, not zero. It has vacuum
fluctuations and `Omega O(y) DB`, NOT a falsely capped midclock momentum.
The original second ordered source jet includes DYY; q1² is not substituted
for it. Guard's corrected q0..q6 conventions are retained. The normal error
envelope budgets inverse32, including the two-bank1/sqrt2 SOURCE factor,
but A must bind the actual ONE-global-filter SOURCE norm. Neither the
withdrawn7ba claim nor RAW norms are filtered acceptance.

The leading spectator Hamiltonian contains the complete photon/harmonic
operators and the bounded actual light/Ref/body terms. Pure geometry
commutes Hph, Hharm and the colour-invariant Coulomb charge operator.
Normal variation of the light Weyl words is bounded by `CL |y|` for EVERY
coexact A, because the Weyl factors are unitary. Normal variation of the
body coefficient is likewise `C |y| Qmatter`, with finite full-CAR norm.
Its time integral is small by the actual Gaussian widths. These terms
need no photon gap. The photon/current-energy exchange must be bounded
from ITS own equation, not from total driven energy of orderLambda^1/2.
B's accepted EM positive-form factor and actual bounded current provide
that separate graph interface; higher photon/physical-Noether graphs remain
the actual B consumer's responsibility.

All branch-local e_m and Efast(m,s,t) phases are retained. Source cycle
preservation permits same-branch cancellation; it does not make beta
commute or erase Lambda times an exp(-cN) alias. Different actual minimizer
ties keep their phases and weights. No hypothetical global equal floor
or Hamiltonian superselection has been introduced.

### 7.6 Driven common-worst floor, interior barrier and Root room

A common WORST driven floor incurs a `Cdrive Lambda^1/2` offset. Calling
it an ordinary Lambda^3/8 excess is wrong. The earlier thin ordinary Gamma
has a barrier of orderLambda^1/2 and cannot generically dominate this NEW
offset by a power of Lambda. For the proof use a wider INTERIOR comparison
cutoff `r0 Lambda^-3/16`. Its barrier is
`kappa r0² Lambda^5/8/2^10`, above the driven common excess. This changes
NEITHER the regular H_D union3r/4 domain NOR the once-prepared state's
original Gamma. Initial comparison between the two cutoffs is paid by
the already concentrated Gaussian/fast-cluster tail.

The full finite-N shifts in b, M, e_m and Efast are kept in these coefficients.
The total physical energy boost is still only Cdrive Lambda^1/2, below
V Lambda at the unchanged selected tower. Original Utrial>=300V Lambda
therefore leaves a large Root-Q margin above the normal trajectory, without
changing Delta. Completing the RAW Phi/colour Noether square costs only
this Lambda^1/2 term. Its field derivatives and even matter Gauss operators
preserve the encoded flat faces; betaTOTAL1/16 remains the noncommuting
off-code perturbation. The NORMAL-part Q-versus-trajectory margin can use
orderV Lambda, not merely the fixed historical G. B must still bind the
complete physical Gphys/EM/current/Root producer and the associated
time-dependent exchange in its final SAME-state consumer.

### 7.7 Native and acceptance scopes

The NEW driver CLI solves actual original G_N at N5/kappa000, builds the
full105 Hessian/contact, projects ALL15 Gauss with massdiag(1,1/.6), and
realifies the actual +/-q pair to180 normal oscillators. The colour
plaquette is `.7 Lambda^5/4` in EVERY frequency. An unscaled .7 plaquette
is a different positive-control quadratic and is not reused as canonical.
Positive/adverse observations cover physical mass projection, exact
Riccati covariance, momentum-forced zero-ended return, common floor/phase,
the midclock ordinary-window failure and q1² graph failure. The hard-mode
body fraction rejects an automatic2pi minus; the uniform rest control
derives its minus from the actual S3 eigenvalues.

This native is an explicitly declared quadratic Fourier restriction with
clock6, not a full90V nonlinear evolution or the physical SOFT long-clock
consumer. The successor analytic normal lemma uses B's newly doubled
`T=1008124871316564`, `Rin=1342177280000`, `Rout=2684354560000` and complete
local90V jets. B's exact Arb ceiling replaces the under-ceiled predecessor
1008124871316563; its executed native and ledger receipts are not retagged.
The65/256 moment-proof envelope covers both clocks, while B owns exact c/4.
Full physical closed rotation, calibrated Born/nonforward
probability and complete coupled Root/EM/source-relative acceptance remain
with their designated owners.

## 8. SAME raw Gaussian quasimode and ONE global weighted shadow

`raw_correlated_trial_ledger` exports the exact all90V construction and
finite constants for A's successor. It does not redefine the field Gaussian.
Raw m amplitudes are1/8 on all64 m, with normalized actual finite-N min_s
ties. The physical state has two banks divided bysqrt2: full negative sea
with Ref4, and that sea plus four orthogonal positive spectators with Ref0.
Both keep `qRef=nRef-4`; spectator O(1) energies and phases enter their
actual scalar references. The Gaussian uses the full correlated Riccati
precision; cutoff1 inside r/3, support2r/3, includes quotient half-density.
No physical ground, G_infinity, or binary64 argmin is inserted.

Let Ps and Ph be the45 uniform soft and90V-45 stiff mass-normal projectors.
The exact matrix inequality K>=alpha Lambda Ps/2+b Lambda^(5/4) Ph/2 and
operator monotonicity give

    Sigma_y=(sqrt(chi)/2) K^-1/2
      <=(2alpha)^-1/2 Lambda^-3/4 Ps
        +(2b)^-1/2 Lambda^-7/8 Ph.

The scalar Gaussian momentum covariance is
Sigma_p=K^1/2/(2sqrt(chi)). Compression concavity bounds its soft trace by
45 sqrt(Lambda L)/[2sqrt(chi)], and its stiff trace by Z Lambda^7/8/2.
Actual cutoff derivatives add the separately paid Gaussian tail. The
exported momentum bounds concern this SCALAR half-density Gaussian, not
a false claim that Berry/matter covariant momentum has no connection.
Those connections are retained in the residual majorant.

For k<=4 the normalized cut position moments are at most
`4(2k-1)!! (Tr Sigma)^k`; mixed contractions are kept. In particular the
uncut Gaussian has `<y_i p_j>=i delta_ij/2` even when K is non-diagonal.
Sixth Wick moments bound the ACTUAL-chart cubic
`Lambda^(5/4) xi z²` by`CN Lambda^1/16`. The flat-adapted
`Lambda^(5/4) xi² z` and metric`chi z p_xi²` words have power0.
Eighth moments bound quartic norms, including the actual-chart soft quartic
with power-1/4. Heavy cubic costs power-1/8 and Berry p_xi power-1/16.

Writing `Mbar>=1+Mposition+Mmomentum` and Ccompare for the contracted
metric/density/frame jet bound, one explicit normal-field operator bound is

    ||[Hnormal-(Lambda v_ms+E0_ms)] tau_ms||
      <=(2^64 Ccompare Mbar^4+1) Lambda^1/16 =: CN Lambda^1/16,

where E0_ms is the exact full90V quadratic zero point. The1 pays cutoff
operator tails. The ledger exports the conservative residual logarithm,
and separately bounds that tail by `2^Ccut Lambda^-2` after the unchanged
selected tower condition. A adds its actual bounded light/Ref/Coulomb,
photon zero point, harmonic and beta residuals and scalar reference; this
normal-field equation is NOT a whole-H eigenvector claim.

The physical preparation has ONE floor

    e_global=Lambda min_(m,s) v_ms-CLcommon-beta,
    fn=f((H_D-e_global)/L), L=4 Cstar Lambda^3/8.

It does not preserve equal64 prepared amplitudes. If mu_ms is A's complete
scalar quasimode reference, its mathematical shadow coefficients are

    w_ms,b=(1/8)/sqrt(2*tiecount_m) * f((mu_ms,b-e_global)/L).

Every Lambda(v_ms-v_global), exact E0_ms, spectator energy and branch
evolution phase is retained. Each minimizing-bank tie aggregate has
amplitude1/(8sqrt2); their combined aggregate has amplitude1/8 when BOTH
actual references lie on the unchanged plateau. Fourier calculus gives

    ||fn tau-tau_shadow|| <=32 F1 Ctotal Lambda^1/16/L,
    F1=int |t| |fhat(t)| dt,

using at most1024 normalized branch/bank vectors, with the bank factor
included rather than assuming their beta-coupled errors stay orthogonal.
It gives decay Lambda^-5/16, subject to A binding Ctotal and its actual
cutoff/source remainders. The normal SOURCE budget uses inverse32.
Uniform RAW Source19/SOFT Gram bounds extend to a weighted shadow after
the explicit bank factor, not automatically to the physical filtered vector.

The union IMS operator theorem itself survives the clean global cutover:

    H_D(m)-e_global=(H_D(m)-e_m)+(e_m-e_global)
      >= kinetic+kappa Lambda distance_to_union²,

because the finite-N floor offset is nonnegative (take the same worst CL).
For commutator and N+1 proofs, `direct_sum_m[H_D(m)-e_m]` can be a conserved
PROOF graph on the disjoint Dirichlet components, while the physical fn
still uses only e_global. This does not declare whole-H superselection:
the whole-domain boundary theorem pays the actual inter-chart leakage.
No exponentially small finite-N alias is dropped before multiplying Lambda.


## 9. Actual unweighted fast graph and whole-field moment bridge

The successor CLI has a ledger-only path; it does NOT rerun the frozen
native180 control. Two separate operator arguments replace an illicit
whole-neutral Gaussian-return gate or unweighted-error moment transfer.

### 9.1 One scalar floor and the SECOND fast graph

Inside the original regular union, define

    Kfast(t)=Hfast-Omega(t)Gfast-e_global+Cdrive Lambda^1/2,
    Kfast>=Lambda^1/2+(chi/4)Pboost*Pboost
                         +kappa Lambda distance_to_union².

Hfast includes the actual heavy algebra, full Root operator and beta.
Any heavy orbital generator belongs inside Gfast; otherwise its commutator
with Lambda Hheavy cannot be called a bounded light current. The common
Cdrive pays the original Phi/colour kinetic completion, body terms and
the worst boost floor, without removing any finite-N m/s offset.
Do NOT subtract an S-dependent diagonal classical Lambda v_s: beta need
not commute it. The physical filter still has ONE scalar e_global.

The Phi/colour primitive and local SU4 body operators preserve local
charge, heavy is EM-neutral, and betaZ preserves charge. Therefore Kfast
commutes H_EM and G_EM, including Coulomb and harmonic operators. At
Omega=0, positivity and strong commutation imply, for every prepared
global-filter vector u=fn tau,

    ||Kfast u|| <=||(Kfast+H_EM)u||
       =||(H_D-e_global-hLight+Cdrive Lambda^1/2)u||
       <=(2L+CL+Cdrive Lambda^1/2)||u||.

This is an actual second fast graph, not a projection estimate on a
photon-weighted vector. Multiplying the initial Gamma adds only its
chi*dGamma*P and chi*d²Gamma commutators. Applying SOURCE adds
chi*DB*P, chi*D²B, Omega*DYB and the retained [beta,B].
The corrected jets and actual normalized SOURCE lower1/32 pay these
finite constants; no beta/SOURCE commutation is imposed.

Let hslow=hLight-Omega GorbLight. All its normal derivatives are bounded
on the regular union, uniformly in coexact A, because Peierls words are
unitary. Heavy/light even algebras commute on the full auxiliary space.
Consequently

    Dfast=partial_t Kfast+i[hslow,Kfast]
         =-Omega_prime Gfast+i[hslow,Kfast],
    ||Dfast psi|| <=Cslow (|Omega_prime| Lambda^1/4+Lambda^-1/4)
                              ||Kfast^1/2 psi||
             +Cslow (|Omega Omega_prime| Lambda^1/2+1+|Omega|).

This retains the bounded beta current. The completed kinetic form yields
the Gfast bound; chi*P=chi*Pboost+Omega*Y yields the actual field-current
bound. With F=||Kfast psi||/Lambda^1/2>=1, differentiation of its norm and
`||Kfast^1/2 psi||<=||Kfast psi||^1/2` give

    |d sqrt(F)| <=Cslow (|Omega_prime|+|Omega Omega_prime|
                                  +(1+|Omega|)Lambda^-1/2).

Integrate the actual quintic total variations2Omega_peak andOmega_peak².
This is polynomial growth, not exp(Cslow*T), and proves
`||Kfast psi(t)||<=Cgraph Lambda^1/2` along the coupled Dirichlet flow.
It needs neither an isolated coupled-Root band nor frozen branch phases.

Choose a PROOF cutoff1 up to r/2 and zero outside2r/3, within the unchanged
3r/4 domain. Its fixed shell barrier gives amplitude`C Lambda^-1/4`.
Magnetic IMS/Caccioppoli, with eta supported on that shell, gives

    (chi/4)||Pboost eta psi||²
        <=||eta² psi|| ||Kfast psi||+C chi ||deta psi||²,
    ||chi Pboost eta psi||<=C Lambda^-1/8.

Here Pboost=P-Omega*g(y)^-1*b(y)/chi is the FULL curved one-form, not an
assumed exact scalar gradient. In [Hdrive,gamma], its mean cancellation is
exact with g(y), not only g0. The resulting unweighted whole/Dirichlet
Duhamel error is`T Cboundary Lambda^-1/8`. The ledger gives its finite
selector and retains the actual preparation Gamma unchanged.
No photon cubic weight is moved through fn, SOURCE, or this boundary.

### 9.2 GLOBAL Phi moment, not an inherited moment

For the whole domain let rho²=sum Tr(Phi_x²)/4. Reuse the actual heavy
bare and selected improved light coefficient l1 caps and the global quartic
supplier with kappa1:

    H_R>=Tphi+Tcolour+H_EM+Lambda rho^4/(8V)-B_N-beta,
    B_N=a Lambda+b+c/Lambda.

Alignment is retained in its original positive form. Any genuine
Phi-independent scalar added by preparation must be subtracted in BOTH
initial energy and evolution. H_R itself has no such added scalar.
No ordinary J term, history alias, or absolute Lambda^5/4 trial floor is
silently changed into a Lambda V first-energy bound.

For the raw Cartesian/covariant/clover primitive the global bounds use
CR=2^16(1+Rout):

    |Gphi|<=CR rho sqrt(2Tphi/chi),
    |Gcolour|<=CR sqrt(V) sqrt(2Tcolour/chi),
    |Gorb|<=CR V,   |GEM|<=sqrt2 Rout H_EM/c.

These precede Gauss reduction, so singular inverse body inertia outside
the chart is not invoked. With the actual quintic peak bounded by
165/(14T), v/c<=65/256 is a slightly wider rational PROOF guard than the
physical c/4 guard. It leaves at least5/8 of H_EM. Young completion gives

    H_R-Omega G >=Tphi/2+Tcolour/2+(5/8)H_EM
                       +Lambda rho^4/(16V)-Bhat Lambda V,
    Bhat=(a+b+c)/V+2^80.

The2^80 pays Phi/colour boost absorption, bounded orbital terms and beta
when Omega_peak CR<2^16. Set S=<H_R-Omega G>+Bhat Lambda V, F=S/(Lambda V).
Exact energy differentiation gives S'=-Omega'<G>, without a rotational
symmetry assumption, hence

    |F'|<=|Omega'| [4CR Lambda^-1/4 F^3/4
                   +3CR Lambda^-1/4 F^1/2+CR/Lambda
                   +sqrt2 Rout F/((5/8)c)].

If Omega_peak CR Lambda^-1/4<=1/512, the integrated EM exponent is below
1.17 and the field contribution below.03. Gronwall and the quartic yield

    S(t)<=8(C_E0+Bhat+1)Lambda V,
    sup_t <sum Tr Phi_x²><=64 sqrt(C_E0+Bhat+1)V.

C_E0 MUST be A's actual normalized SOURCE initial relative-energy
coefficient. The CLI accepts it explicitly; absent that binding it exports
the proved formula and does not claim prepared-state acceptance. This is a
whole-trajectory form estimate, never multiplication of an unweighted
leakage norm by an unbounded Phi moment. It supplies no weighted photon
cubic graph or calibrated Born acceptance.

For the executed coherent-bank-v2 A receipt
`local://issue222-global-filter-execution.json`, the actual absolute
initial SOURCE/true-creator first energy is at most8V Lambda.
Pass `--initial-energy-coefficient-log2 3 --driver-energy-receipt PATH`
with the selected legal clock1008124871316564 andRout2684354560000.
The CLI checks its identity, selected tower, executed onset and absolute
coefficient contract and records the receipt SHA. This gives the explicit
whole Phi trace upper`2^48 V`. The original Root/source primitive binding
remains the transport integration owner’s requirement.

The historical native-stable receipt/source cbde642d and doc f1a7a603 are
kept at their executed hashes. Its native quadratic controls remain valid;
its CN0 and inverse16 annotations are WITHDRAWN, not retagged as this
corrected ledger. The current raw exponent is1/16, global filter error
exponent-5/16, branch resolvent dressing-3/16, and source inverse32.


