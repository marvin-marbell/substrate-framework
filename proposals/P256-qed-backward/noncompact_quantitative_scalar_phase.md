# Quantitative original whole-heavy scalar phase at a1

Governing shared plan: `issue-222-synthesis-continuation-20261006`, assignment
revision `168949e5f2df57656471db8e62d4ae809ab290b1d09ea28c620f52349324a549`,
pinned general template `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`.
This pair supplies scalar admission, not campaign completion. The original
slow theorem's already proved eventual regular-well/global-minimum argument is
used; it is made quantitative by the exact rational polynomial law below,
not by treating native N5 normal positivity as globality.

## Result and executable interface

For the original a1 parameters m=1/5, r=3/25, v=3/10, kappa=2, J=1/5,
K4=7/10, the sufficient integer threshold is

\[
 N_{\rm phase}=2^{96}=79228162514264337593543950336.
\]

Every flat global minimum at N>=Nphase is a regular metric-Clifford orbit with
central SU4 holonomies. All64 centre classes retain their actual shifted full
10-coordinate Gram minimizers; no offdiagonal entry, anisotropy or scalar-well
offset is deleted. Every reducible flat competitor, including every noncentral
flat holonomy competitor, lies at least V*2^-89 above the flat minimum.

`phase_bounds(N)` constructs the following exact, finite full-domain constants
without allocating an N^3 grid. Set V=N^3, M=100000 V^2, A=100 M^2, and
B=2^(2^A). Then

\[
 \Lambda_{\rm phase}=2^{8B},\qquad c_{\rm phase}=2^{-8B^2},
 \qquad r_{\rm reg}=\frac1{2^{20}V^3}.
\]

These are explicit integers/rationals, not numerical estimates or implicit
optimization thresholds. The Python representation stores their exact power
expressions; materializing their digits would be pointless. For every
Lambda>=Lambda_phase,

\[
 U_\Lambda(y)-v_*\ge c_{\rm phase}\,d(y,Z_N)^2
 \quad\text{on the entire field/link domain},                 \tag{A}
\]

where U_Lambda=V0+Lambda^(1/4)V1, vstar is the actual flat global minimum,
and Z_N is the entire gauge-invariant set of flat global minima. Distance uses
field HS/4 and link HS norms; minimizing over Z_N includes the entire colour
orbit. Equation(A), not a hard restriction to Z_N, is the global exterior
restoration. Outside epsilon cores its unscaled barrier is c_phase epsilon^2;
the physical potential barrier is Lambda c_phase epsilon^2. Replacing Z_N by
the union of all64 native centre orbits only weakens this lower bound. The
quantum scalar energies E_theta(Lambda) remain the exact source-defined
localized energies, not this classical lower bound or harmonic replacements.

The enormous height majorant is selected because it closes the global
nonuniform obligation with one completely specified rational law. A tighter
analytic chart/normal budget can reduce the onset but is not necessary for
existence of legal finite parameters. It is not claimed to be an efficient
numerical many-body calculation.

## 1. Exact compactness and quantitative Cauchy slack

Use the full flat/covariantly uniform class of
`bulk_flat_holonomy_bound.md`, including all reducible block twists and the
actual whole NH=2V filling. Let F(G), B(Phi), D_N(G) have that source's meaning.
The whole-half Ky Fan lower bound is F+B-D_N, not a forced two-per-momentum
occupation. The identity Clifford trial is a legal upper comparison, not a
restriction on holonomies.

The rational majorants Cx<=11/10, ||G||F<=3, CD<=10 follow directly from the
source quartic compactness: Cx^2=(23/25)^2+3(3/10)^2<(11/10)^2,
(2 sqrt(2) Cx)^2<3^3, and
4(22/7)^2(9r^2+3v^2)*6<100. Thus every flat minimizer has

\[
 \|G-G_*\|_F^2+B(\Phi)\le20/N.                           \tag{1}
\]

The same compact bound holds for every flat tuple whose actual energy is<=0,
and for every continuum tuple with F+B<=0. The rank-two temporal trial
Phi0=Gamma0/2, Phi1=Phi2=Phi3=0 has energy per site
1/16-14/25=-199/400<-1/4 on every central shifted N>=3 grid.
Competitors of positive energy therefore already have gap>1/4; compactness
need only be used on the nonpositive sublevel. No large-field competitor is
silently omitted from the separation claim.

The NativeMetric Arb128 continuum certificate supplies spatial eigenvalue
>0.06, temporal eigenvalue>0.4. Its independent strong-convexity/analytic-strip
proof also supplies every real shifted native full10 minimizer at N>=25 with
lambda_min>0.06 (paying cross terms by its certified Frobenius error). This
N25 is used only for native metric bounds, never as the global phase onset.
In the present proof ||G-Gstar||F<=1/100 implies G>=I/20.

For h=sum x_mu Phi_mu, compactness gives ||h||op<=||h||HS<=6. Put q=Tr h^2/4
and s_i=|lambda_i(h)|. Strong concavity of sqrt on [0,36] gives

\[
 2\sqrt q-\tfrac12\sum_i s_i
 \ge\frac{\|h^2-qI\|_{HS}^2}{16\,6^3}
 \ge\frac{\|h^2-qI\|_{HS}^2}{4096}.                      \tag{2}
\]

The bound remains valid at a zero eigenvalue by continuity. Indeed the
Jensen remainder for sqrt is at least variance(s_i^2)/(8*6^3), and the
left side is twice that Jensen remainder. Define S^2=int||h^2-qI||HS^2.
Then S^2<=4096 B(Phi). The denominator16, rather than an unproved denominator8,
is deliberately paid here.

## 2. Rational Fourier richness, not a sampled Hessian

Set D_munu={Phi_mu,Phi_nu}/2-G_munu I, including D_mumu=Phi_mu^2-G_mumu I.
Then h^2-qI=sum x_mu^2 D_mumu+2 sum_(mu<nu) x_mu x_nu D_munu.
Writing x0=c-r sum cos_j with c=14/25 and xj=v sin_j, distinct Fourier
characters give the following HS bounds by Parseval:

* A cos_i cos_j coefficient 2r^2 D00 gives ||D00||<=S/r^2.
* The cos(2k_i) coefficient (r^2 D00-v^2 Dii)/2 gives
  ||Dii||<4S/v^2, paying the already extracted D00.
* A sin_i sin_j coefficient 2v^2 Dij gives ||Dij||<=S/v^2.
* The sin_i coefficient 2vc D0i gives ||D0i||<=S/(sqrt(2)vc).

No other term has these characters after the stated subtraction. Paying both
ordered copies of offdiagonal defects,

\[
 \sum_{\mu,\nu}\|D_{\mu\nu}\|_{HS}^2
 \le\left(r^{-4}+48v^{-4}+6v^{-4}+\frac3{v^2c^2}\right)S^2
 <2^{16}S^2.                                              \tag{3}
\]

These are continuum Fourier inequalities, uniform in every flat twist. The
same quadratic richness is exact on every shifted N>=5 grid; this does not
require numerical rank tests or enumeration of the admitted gigantic graph.

## 3. Constructive irreducibility without an approximate-representation black box

For G>=I/20 normalize A_mu=sum (G^-1/2)_munu Phi_nu. Equation(3) gives

\[
 \sum_{\mu,\nu}\|\{A_\mu,A_\nu\}-2\delta_{\mu\nu}I\|_{HS}^2
 \le2^{27}S^2.                                           \tag{4}
\]

Suppose every relation has operator norm<=delta=2^-24. Form the16 ordered
monomials W_S=product_(i in S) A_i, S subset {0,1,2,3}, including I. Their
HS/4 Gram matrix is positive definite. Here is a finite error count, avoiding
an existence-only approximate-Clifford theorem:

* A_i^2 differs from I by at most delta/2 and ||A_i||<=sqrt(1+delta/2).
  Every word of length<=8 consequently has norm<2.
* Reduce W_S^* W_T by moving letters and cancelling repeated adjacent letters.
  At most28 swaps and4 cancellations are needed. Each replacement error has
  norm<=2delta, hence the total is<=64delta; using128delta is safe.
* For any nonempty ordered word W_R choose a generator j which anticommutes
  with it in the exact Clifford algebra: for odd |R|<4 choose j outside R;
  for even |R| choose j in R. At most4 swaps and one cancellation show
  A_j W_R A_j=-W_R+E, ||E||<=16delta. Cyclicity and A_j^2=I+E_j then give
  |Tr W_R|/4<=20delta.

Thus every entry of Gram(W)-I has absolute value<=256delta; its operator
norm is<=16*256delta=4096delta<1/2. The16 monomials are linearly independent
and span Mat4. Any simultaneous invariant subspace for the Hermitian A_i
would therefore be invariant for all Mat4, which is impossible. Irreducibility
is proved constructively, without assuming an isolated charged pole or
removing reducible flat competitors.

For a reducible tuple either ||G-Gstar||F>1/100, costing >1/10000 by strong
convexity, or equation(4) must fail the delta admission. In the latter case
S^2>2^-48/2^27 and B>2^-87. Consequently every reducible flat tuple costs at
least 2^-88 over F(Gstar). Pay both actual finite-size comparison debts:

\[
 e_{\rm reducible,N}-e_{\min,N}
 \ge2^{-88}-20/N\ge2^{-89}\quad(N\ge2^{96}).               \tag{5}
\]

A noncentral holonomy commuting with all Phi has a proper invariant eigenspace,
so equation(5) applies to every noncentral flat competitor. Every flat
minimum is irreducible and has central holonomies. In each central class the
exact full-PSD strongly-convex Gram functional has its positive native Gtheta;
metric-Clifford fields attain it. Every minimizer must attain both Gram and
sampled Cauchy equalities. Shifted N>=5 quadratic richness then forces the
exact Clifford relations, giving precisely the regular centre-class orbits.
No claim fixes which of the64 classes realizes vstar.

## 4. Actual local Gauss and whole-heavy regularity

With the executing generators Ta=Ba/sqrt(8), Tr TaTb=delta_ab/2, the canonical
Clifford field orbit Gram in HS/4 has nonzero spectrum 1/2,1,3/2,2.
This is also obtained directly by expanding a traceless generator in the15
nonempty Clifford words and counting which of the four generators anticommute
with each word. A full10 metric G>=I/20 therefore gives

\[
 D_{\rm field}^*D_{\rm field}\ge1/40
\]

at every site. Links add a nonnegative form. Thus all15V Gauss tangents are
independent and the continuous stabilizer is absent; the only global
stabilizer is Z4. The scalar factor4gfloor sometimes used with ||T||HS^2/4
is NOT the source's Euclidean Ta-coordinate normalization.

Each metric-Clifford h(k) has radii>=m sqrt(gfloor). The actual entire whole
NH=2V Slater gap is twice the smallest radius, hence >1/20. This is not an
occupation imposed per momentum at a nonuniform field. The source sign gap
only proves that the whole positive cluster is smooth in this tube.

For ambient distance d, the change of the field gauge derivative has operator
norm<=4d: use ||[T,E]||HS<=2||T||op||E||HS and Tr T^2=|t|^2/2, independently
at each site. Native endpoint row/HS bounds give ||delta h||op<=10Vd for d<=1;
both endpoints and link variations are paid, using ||Phi_mu||op<=2sqrt(gmax)
and gmax<1. The radius rreg=1/(2^20 V^3) therefore preserves gauge Gram>=1/160
and whole-heavy gap>=1/40 by singular-value perturbation and Weyl. Compact
orbit tubular coordinates are legitimate on a sufficiently small sub-tube;
this explicit no-rank-loss/no-band-crossing radius is not asserted to be a
complete sharp normal-chart injectivity radius or a finite-jet bound.

## 5. Full-domain rational polynomial law and explicit height majorant

This is the quantitative replacement for the original slow theorem's
compactness-only Lambda onset. It includes every nonuniform field and link,
not just a fixed Clifford shape or a toron-restricted potential.

Use raw real and imaginary entries of the Hermitian traceless Phi_mu(x), and
raw entries of U_e with U_e^*U_e=I, det U_e=1. There are at most64V field
coordinates and96V link coordinates; no irrational basis coefficient is
needed. Add a raw Hermitian 4V-by-4V matrix Q satisfying

\[
 Q^2=Q,\qquad\operatorname{Tr}Q=2V.
\]

Hermiticity and idempotence make Q exactly a rank2V orthogonal projector, with
at most32V^2 real variables. The exact whole-heavy potential is the minimum
over this complete Grassmannian of the rational polynomial

\[
 f_0(y,Q)=\sum_x\|G_x\|_F^2-\operatorname{Tr}(Qh_{\rm native}(y)),\qquad
 f_t=f_0+t\left(W/40+7P/10\right),\quad t=\Lambda^{1/4}.   \tag{6}
\]

Ky Fan's variational principle makes (6) EXACT for the whole fixed-NH ground
energy. It does not replace it by Tr|h| or constrain Q to translation blocks.
The executing endpoint law has degree2 in Phi,U; f0 has degree4. Alignment
||Phi_x-U Phi_y U^*||HS^2 has degree6; the positive plaquette term can be written
4-ReTr(U1 U2 U3^* U4^*), of degree4 on the SU4 constraint. Thus total degrees
including t are<=7; use the conservative initial degree8. Exact rational
coefficients come from the declared decimals, never from floating eigenvalues.

The declared exact decimal rationals and the executing binary64 parameters
are distinct rational coefficient choices. The interval NativeMetric supplier
encloses both. The rational Cx/compactness/CD/Fourier-richness majorants
persist with m in [.199,.201], r in [.119,.121], v in [.299,.301];
the producer checks these adverse endpoints exactly. The metric floor is
still supplied only for the much smaller certified parameter intervals,
not inferred for that entire larger box. Positive J/K4 binary roundoff
preserves the original eventual-well argument. The polynomial height
calculation pays the exact binary rational denominators as well (at most64
coefficient bits before the displayed slack). Thus the floating consumer is
not asserted to be bitwise the decimal rational law.

Here is the precise first-order formula whose sign cells are bounded; it is
not left as an unspecified optimization. Define Domain(y,Q) by the above
Hermitian/traceless/SU4/projector equations. Define

```
FlatMin(y,Q,m) := Domain(y,Q) AND V1(y)=0 AND f0(y,Q)=m
                 AND FOR ALL (z,R):
                     [Domain(z,R) AND V1(z)=0] IMPLIES f0(z,R)>=m.
Admission(t,c) := t>0 AND c>0 AND
                 EXISTS (y0,Q0,m): FlatMin(y0,Q0,m) AND
                 FOR ALL (x,P): Domain(x,P) IMPLIES
                     EXISTS (y,Q): FlatMin(y,Q,m) AND
                        ft(x,P)-m >= c*d(x,y)^2.
```

Distance is the rational raw-field HS/4 plus raw-link HS squared distance.
The repeated FlatMin universal variables are distinct and all quantified.
Expansion uses fewer than100000V^2=M real variables, including t,c,m; fewer
than2M^4 atomic polynomials; degrees<=8. After individually clearing positive
rational denominators, integer coefficient bits are<=100M. This generously
pays trace sums, determinant expansion, projector products, and repeated
constraints; it does not clear denominators by multiplying the entire formula
and accidentally changing signs.

The original slow theorem proves the necessary nonemptiness of Admission:
for sufficiently large t, global minimizers are exactly flat regular wells;
local adapted positive soft/stiff restoration plus compact-exterior separation
and quartic infinity coercivity give a positive GLOBAL quadratic distance
constant. Equation(5) supplies its formerly existence-only N hypothesis.
This is the only qualitative theorem used in this height step. No bound on
an unproved principal sea minimizer is borrowed from NativeMetric. Monotonicity
of V1>=0 makes Admission upward closed in t and downward closed in positive c.

For completeness the elimination-height estimate is specified rather than
invoking an unspecified CAD constant. Perform full rational cylindrical
projection, including all coefficients, all reducta and derivatives, their
discriminants and every principal subresultant coefficient; this is full
projection, not a well-orientedness-assuming reduced projection. If total degree, coefficient
bit size and polynomial count at one stage are d,l,s, conservative determinant
bounds for the Sylvester/principal-subresultant matrices give

\[
 d'\le2d^2,\quad
 l'\le16d^2(l+\log_2s+M+1),\quad
 \log_2s'\le4\log_2s+16d^2.                              \tag{7}
\]

For the coefficient estimate, a matrix has at most2d rows; expand its
determinant, paying its factorial and the (d+1)^M monomial count in each
multivariate coefficient product. The displayed16d^2 is a slack upper on
that count in bit units. Subresultants only decrease matrix size. Projection
nullification is handled by the full coefficient/sign stratification, not
silently ignored. Starting at d=8,l=100M,s<=2M^4, iterate (7) at mostM times.
Taking logarithms bounds their maximum by a recurrence a'<=4a+20+log2(M+1);
induction gives every final degree, coefficient bit size and log2 count
strictly below B=2^(2^(100M^2)). This intentionally loose closed expression
also dominates every lower-dimensional projection polynomial.

Project c last, so the one-dimensional base is t. Nonzero integer polynomials
in t have all roots of magnitude<=1+2^B by Cauchy's bound. Since Admission is
nonempty for all sufficiently large t and is constant on the final unbounded
base cell, take the explicit integer t0=2^(2B). At that t, coefficients of
any c-polynomial have bit size<=B+2B^2+log2(B+1)<4B^2. After stripping zero
powers of c, reciprocal Cauchy bounds put its smallest nonzero root at least
2^(-4B^2-1). Nonemptiness and downward closure put the entire first positive
c-cell in Admission. Therefore c0=2^(-8B^2) is admitted. Monotonicity in t
extends it to all t>=t0. Converting t^4 gives Lambda_phase=2^(8B) and proves
(A). No CAD run on the gigantic graph, guessed quantifier witness, floating
Hessian or artificial field/occupation cutoff is needed: (7) bounds the
exact rational microscopic formula itself.

This technique is less efficient than extracting sharp compact-exterior
geometry, but it is a positive computable finite admission, not another
acceptance gate. The regular chart finite-jet constants needed for quantitative
BO residuals are a separate normal-budget supplier; this pair does not
mislabel a gauge-rank radius as those constants.

## 6. Unbounded fields and actual light-sector domination

Put rho^2=sum_x TrG_x. The original endpoint bound gives, on the ENTIRE field
and link domain, ||h||op<=2Cx rho. Hence with NH=2V,

\[
 V_0\ge A_4\rho^4-B_H\rho,\quad A_4=1/(4V),\quad B_H=22V/5.
\]

The integer rho0=1+36V^2 satisfies rho0^3>=2BH/A4. Because vstar<0 and V1>=0,
rho>=rho0 implies U-vstar>=A4 rho^4/2. For the original arbitrary fixed-NL
light/reference sector, BL=2NL*(11/10) bounds ||Hlight||<=BL rho uniformly in
all EM phases; consequently C0=BL rho0 and C1=2BL/A4 give
||Hlight||<=C0+C1(U-vstar). This is not a fictitious global bounded light norm.
Photon kinetic/magnetic forms stay positive and are not modified by admission.

## 7. Consumer evidence and decision-changing successors

Run the producer with the project Python and an explicit output path. It
executes the NativeMetric Arb128 interval certificate and checks the rational
analytic inequalities above. `phase_bounds(2^96)` exercises exact legal large-N
handling without creating an N^3 array; `phase_bounds(25)` must reject the
metric-as-global shortcut.

`analytic_bloch_metric_enclosure(N,cert)` also exercises the admitted huge-N
actual native metric law, not a grid relabelled from N3. Its sigma=1/2 strip
bound uses t=exp(-N/2)<=2^-floor(N/2). For N>=256,
((1+t)/(1-t))^3-1<=32t and the certified gradient majorant is<64, so a
conservative native tail2^-100 is sufficient. Full-matrix strong convexity
then bounds each exact G_Ntheta within2^-38 in Frobenius norm of the certified
decimal trial, uniformly over every real shift. The producer emits all64
N-owned shift descriptors, full diagonal intervals and nonzero-width
offdiagonal intervals, with zero allocated grid points. These intervals
enclose, but do not substitute for, the exact full10 native minimizers or
any actual scalar offset. They likewise do not transplant a different
finite-N harmonic scattering coefficient to this legal N.

Separately, at native N3 it calls the actual whole-heavy endpoint matrix for
all64 full10 shifted metric frames, including the offdiagonal adversary. It
runs reducible2+2 noncentral flat and Cartan occupation-transfer competitors
through the source's native seam/Bloch/covariance consumer. A further control
perturbs all site fields and all links away from the uniform/Clifford locus,
compares the full rank2V projector polynomial energy with the actual whole
spectrum, and exercises positive V1 and quartic coercivity. These floating
consumer checks are diagnostics of the executing law, not promoted into the
analytic proof constants. Source hashes bind the run; natural process exit
and the output artifact are the execution evidence.

Achievement: explicit Nphase and finite Lambda_phase/global exterior core
restoration replace the original existence-only thresholds. Unlock: the full
original scalar source is admitted at legal finite parameters, retaining the
entire field/link/heavy domain and every centre offset. Decision delta: neither
N5 local positivity nor N25 metric admission is a prerequisite falsely sold
as global phase.

Two successors are (i) plug this exact scalar admission into the quantitative
normal/source-energy suppliers, or (ii) sharpen the conservative full-domain
height onset by explicit nonuniform chart/exterior estimates. Select (i): the
height onset is already a legal finite existence bound, while joined source
energy and same-current scattering need actual normal/energy constants. Only
limitations that change that decision are retained: these enormous constants
are not efficient simulation parameters, and scalar admission itself is not
a particle/photon/scattering construction.
