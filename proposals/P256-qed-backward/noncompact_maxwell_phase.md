# Native noncompact Maxwell-comparable zero-temperature photon response

Governing plan `issue-222-continuing-backward-20260930`, assignment revision
`0affb6059cb724fb5eaba4ec284bd2fb07cfdc25f045eff80327b48dbd624655`, pinned
general `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`,
work `campaign`, requirements `outcome/verification`. The campaign remains
active. This is a different **real-link, quadratic-curl** electromagnetic root,
not a compact-field phase theorem. Original quantum CAR/gauge quantization is
an explicit assumption. Scientific execution and review belong to the parent.

## 1. The construction and its exact hypotheses

Fix lattice spacing **a=1**, nearest-axis genuine three-dimensional tori
`N>=5`, `V=N^3`, original `m=.2,r=.12,v=.3,kappa=2`. Retain the original
whole `NL=2V` CAR sector, background two, and all64 central colour/Gram wells
with their actual scalar offsets. There are no selector particles. The native
endpoint is

\[
 F_j=-(r\Phi_0+i v\Phi_j)/2,\qquad
 h_{x,x+e_j}=F_j U_{xj}e^{iA_{xj}},\qquad
 h_{xx}=(m+3r)\Phi_0 .
\]

In each retained well, the full positive Gram and its true Clifford fields
are used, not an isotropic replacement. Independent noncentral colour paths
and noncommuting temporal SU4 sources are NOT admitted by this fixed-body
proof. Colour equivariance and all15V Gauss equations return through the
existing full-field lift in section 8.

Let `D` be incidence, `C` the actual periodic curl, `L=DD^T`, and let `A_T`
be coexact and `A_h` harmonic real coordinates. The physical neutral operator
is exactly

\[
 H=UE(p_T^2+p_h^2)+\frac K2\|CA_T\|^2
    +UE(n-2)^TL^+(n-2)+d\Gamma h(A_T+A_h),\quad UE,K>0.             \tag{1}
\]

Longitudinal Gauss is reduced/rigged, not imposed as a nonexistent invariant
vector in unreduced longitudinal L2. Every fixed-number occupation remains.
Harmonic coordinates remain real. Their microscopic period lattice gives a
Bloch direct integral; use the periodic neutral fibre, whose spectral infimum
is the full-real neutral infimum by the positive winding argument in
`noncompact_charge_threshold.md` section2. Cell-fibre vectors are not
normalizable full-real ground vectors; narrow Bloch packets approximate them.

The **metric-family hypothesis** is

\[
 s_{N\theta}=G_{00}-G_{0s}G_{ss}^{-1}G_{s0}\ge s_*>0,
 \qquad \|G_{N\theta}\|_F\le t_*=(2\sqrt2 C_x)^{2/3},
 \quad C_x=\sqrt{(m+6r)^2+3v^2}.                                \tag{2}
\]

The upper bound is the source-owned quartic/whole-half trace-norm bound. The
all64 large-N Gram convergence in `noncompact_microscopic_charge_window.md`
section6 supplies an existence-only admission N1 and, for example,
`s_*=g0/2>0`. The CLI's explicit default `s_*=.01` is a DECLARED parameter
hypothesis, not a numerical proof that every N has that floor. Its all64
finite-N check does not certify N1. Everything below holds on any admitted
family satisfying (2); changing the floor changes the explicit coefficients.
No UV limit is taken.

**Result.** There is an explicit fixed strictly positive bare family, section4,
for which the actual positive physical measure has uniformly convex
**nonzero-spatial-momentum** conditional photon coordinates. After retaining,
not freezing, its canonical, temporal and harmonic integrations, its centered
transverse covariance satisfies (12). Thermodynamic zero-temperature limit
points consequently have gapless photon spectral response, (15), with the
same native unit-charge current normalization. This proves a Maxwell-
comparable phase/IR response, not a unique sharp photon pole, an exact
renormalized tensor/speed, Lorentz symmetry, Compton closure or a compact
monopole theorem.

## 2. Actual physical measure; the first route fails and is replaced

A density with uniformly positive Hessian on *every* photon coordinate is
impossible here: bare Maxwell vanishes on constant flat torons and global
temporal holonomy, whereas the native determinant has their genuine finite-
volume/temperature reaction. Nor may one first integrate holonomy and apply a
conditional logdet Hessian bound to the log of that integral: the latter has
an additional score covariance. That attempted route is rejected.

Instead retain the following exact measure. The positive coexact oscillator
heat kernel, periodic-fibre harmonic heat kernel and quadratic magnetic
potential of (1) give their real Brownian weight. All harmonic winding classes
are summed. For the Coulomb term use its actual Hubbard identity

\[
 e^{-dt UE\rho^TL^+\rho}
   =\mathbb E\exp(i\sqrt{2dt UE}\,Z\cdot(L^+)^{1/2}\rho),
 \qquad\rho=n-2.                                               \tag{3}
\]

The temporal source has zero spatial mean, plus global holonomy `alpha/T`.
Retain the global Haar integral

\[
 \int_0^{2\pi}\frac{d\alpha}{2\pi}
 e^{-i(2\int\sum_xa_{0x}+2V\alpha)}
 \det(I+e^{i\alpha}U[A,a_0]),                                  \tag{4}
\]

with the SAME source, interfaces and closing seam. It selects precisely the
full `2V` exterior coefficient. The original beta-coercivity and chirality
homotopy prove that each normalized integrand is strictly positive. An
unnormalized determinant, or a Gaussian over range(L+) without (4), is not
the physical canonical measure. Formulae (3),(4), positive heat kernels and
Hamiltonian Trotter convergence construct the actual measure of (1), without
a temporal Wilson operator. Bounded native h allows the continuous-time limit.

### The exact positivity lemma at the canonical boundary

An instantaneous one-body gap is NOT sufficient for positivity of an
arbitrary noncommuting ordered heat trace. The extra native structure is
essential. With the full Schur beta, the admitted body Hamiltonian is
`h=beta M+Dodd`, where `M>=mu`, `[beta,M]=0`, `{beta,Dodd}=0`, and all operators
are Hermitian. Therefore on the AP circle with the SAME scalar site-U1
interfaces and closing holonomy,

\[
 \beta\mathcal D=M+\mathcal A,\qquad
 \mathcal A=\beta\partial_t-i\beta a_0+\beta D_{odd},
 \quad\mathcal A^\dagger=-\mathcal A.                         \tag{4a}
\]

The derivative's boundary term vanishes because every admitted interface
and the global twist commute with beta. No spatial h at different times
has been assumed to commute. Coercivity excludes a nonzero homogeneous AP
solution for every alpha; the finite-spatial first-order index-zero boundary
problem then implies `det(I+exp(i alpha)U)!=0`.

There is also a fixed Hermitian unitary chirality J, anticommuting with every
native h and commuting with the scalar temporal sources/interfaces.
For the ACTUAL ordered generator `X=-h+i a0`,
`X^dagger J+J X=0`. Hence `U^dagger J U=J`, and `tr h=0` gives
`det U=exp(4i integral sum a0)`, including the genuine kick phases.
Consequently

\[
 [\det(I+e^{i\alpha}U)]^*
   =e^{-i(4\int\sum_xa_{0x}+4V\alpha)}
          \det(I+e^{i\alpha}U).                              \tag{4b}
\]

The centred factor in (4) is real and never zero. Homotopy
`h_lambda=(1-lambda)h+lambda mu beta` preserves (4a), trace zero and J at
every time/interface. At lambda=1 its centred determinant is explicitly

\[
 \prod_x[2\cosh(\mu T)+2\cos(\theta_x+\alpha)]^2>0,             \tag{4c}
\]

where theta_x includes all actual temporal U1 kicks. Continuity fixes the
positive sign for lambda=0. This proves the positive density needed for
conditional BL at each actual global-Haar alpha, not merely positivity
after taking a trace. Multiplication by the positive Gaussian/heat factors
in (3) completes the conditional density. The alpha marginal is its actual
normalizing integral, not a replacement canonical occupation rule.

An arbitrary temporal SU4 twist does NOT obey this lemma. For example, J
itself is an SU4 matrix and anticommutes with beta. For the native gapped
onsite h4, `U4=J exp(-T h4)` has `U4^2=I` and a -1 eigenvalue, so
`det(I+U4)=0`. Such a twist alone is not a compatible fixed-body field
history; a legitimate full-field Haar path must transform Phi and heavy
data too. It is an explicit adverse to blind gap-implies-positivity or to
silently reusing (4a) for generic colour twists. No positive microscopic
SU4-twisted determinant measure is being assumed.

`native_canonical_boundary` now checks this boundary on the actual N5
noncommuting native histories: every slice's beta coercivity/chirality,
ordered pseudounitarity, all4V Floquet eigenvalues, and the full `4V+1`
root-of-unity rule selecting the complete2V canonical coefficient. Nine
direct determinants independently identify those all-mode factors.
It retains the centred background and exhibits the forbidden native SU4
twist against a compatible scalar-twist positive control. These are
parent-run controls; the analytic lemma, not finite sampling, proves
positivity at all alpha.

Condition on the **entire spatially uniform harmonic history**, including
its initial period-cell point and its winding, on alpha, and on a centre well.
These data are denoted zeta; their distribution is the actual interacting
one. They are never given independent Gaussian or uniform weights. The
remaining variables have spatial `k!=0`. Quotient their exact gauge directions
using the spacetime orthogonal complement of
`(eta_A,eta_0)=(d chi,partial_t chi)`. This is a linear physical quotient, not
an extra local number projection. Its Maxwell form has no zero mode:

\[
 \|eta\|_2^2\le q_{\min}^{-2}
        (\|E_eta\|_2^2+\|B_eta\|_2^2),
 \quad q_{\min}=2\sin(\pi/N),
 \quad E_eta=\partial_t eta_A-d eta_0,
 \quad B_eta=C eta_A.                                         \tag{5}
\]

The eigenvalues are `omega^2+q(k)^2` on this quotient, with
`q(k)^2=sum_j|1-exp(ik_j)|^2`. In particular the static nonconstant a0 source
is paid by E; no site-holonomy reaction has been assumed zero.

For a literal finite-dimensional construction, retain finitely many temporal
Fourier coefficients in the Gaussian photon and Coulomb fields, evaluate
**exact** continuous native evolution on these backgrounds, and use the density
`exp(-S_Maxwell+W)` on the quotient. Increasing this photon-source regulator
converges to (3)'s commuting Gaussian kicks and the oscillator Brownian paths;
Wong-Zakai unitary temporal interfaces use the original background phase.
Uniform inequalities below pass by Gaussian domination. This is a regulator
of (1), not a finite-dt temporal-Wilson fermion endpoint. Bare electric/magnetic
quadratic forms fix the Gaussian normalization, including Coulomb.

## 3. Explicit bulk envelope and paid torus/circle seams

Use AppendixA of `original_massive_photon_locality.md`, including its actual
positive measure `nu_(N,T)` and every native contact atom. Its exponentially
weighted anchored density-Hessian mass is H. The following uniform upper
inputs are sufficient:

\[
 \mu=m\sqrt{s_*},\quad C_h=3(r+|v|)\sqrt{t_*},
 \quad B=\sqrt{t_*}(m+6r+3|v|),
 \quad\gamma=\xi=\mu/4,\quad\alpha=\log(1+\xi/C_h).
\]

For the tilted onsite odd part take `d0=(m+3r)sqrt(t*)`,
`c_upper=(m+3r)sqrt(t*)`, `v_alpha=d0+Ch exp(alpha)`. With T0=1 take

\[
 q_t=\max(c_{upper},\log2,\gamma+8v_\alpha,\gamma+4),\quad
 M_t^{upper}=q_t,
\]
\[
 \delta=\min(1/8,[4(B+M_t^{upper}+\gamma+\xi)]^{-1}),\quad
 \bar C=e/[\delta(\mu-\gamma-\xi)],
\]
\[
 P={16\over\gamma}[1+2/\operatorname{expm1}(\alpha/2)]^3,
 \quad L_v=\max((r+|v|)\sqrt{t_*}/2,1),
\]
\[
 H_{low}=16(2\bar C)^3L_v^2 e^{6\alpha}P^2
            +8(2\bar C)^2L_v e^{4\alpha}P,
\quad
 H=M_t^{upper}H_{low}
        +{16384v_\alpha^2+2048\over q_t-\gamma}.                 \tag{6}
\]

Using a larger low-mass interval than the true `q_t-c` is conservative. These
constants depend on the true metric bounds, not on background amplitude,
harmonic history, winding class, alpha or centre weights.

Here is a finite-torus completion of AppendixA.4; it deliberately spends large
constants rather than dropping a seam. Take `C_area=64`,
`P4=sup_(u>=0)(1+u)^4 exp(-alpha u)` and set

\[
 C_{E0}=8H/(e^2\gamma^2),\qquad C_{B0}=2C_{area}^2HP4.
\]

For every smooth `k!=0` quotient source on an admitted T>=1 circle,

\[
 |W''[eta,eta]|\le2C_{E0}\|E_eta\|^2+2C_{B0}\|B_eta\|^2
                   +R_{N,T}(\|eta\|^2+\|E_eta\|^2+\|B_eta\|^2),
\]
\[
 R_{N,T}=R_*[e^{-\alpha N/64}+e^{-\gamma T/64}],\quad
 R_*=2^{20}H e^{\alpha/2}P_{12}(\alpha/4)P_{12}(\gamma/4),
 \quad P_{12}(b)=\sup_{x\ge0}(1+x)^{12}e^{-bx}.                \tag{7}
\]

**Why (7) pays the seams.** At each mass-density anchor use two independent
periodic cutoff gauge functions, one for each differentiated source slot.
On a box of spatial radius at most N/8 and a time interval of radius at most
T/8 they equal the temporal-cut/rooted-spatial-tree gauges of AppendixA.4.
They are extended periodically to zero outside twice that box. The inner
residual is the actual plaquette filling plus integral of E. A rectangular
three-axis filling uses at most `64(1+r)^2` oriented plaquettes; its translated
multiplicity is paid a second time. Summing anchors before Schwarz gives the
displayed CE0,CB0. Young's inequality to separate inner and outer residuals
doubles these local constants.

The remaining Hessian terms have at least one source outside the inner box.
Separate the spatial wrapping and temporal-circle cases by their union bound.
The positive diameter envelope respectively costs a spatial radius at least
`(N-16)_+/32` or a time radius at least `T/32`. The minus16 allows the discrete
tree, plaquette endpoints and contact support, including small N. In the outer
residual retain the source itself. After translation the path Schwarz cost,
rectangular filling and its multiplicity, and the cutoff/product-rule costs
are bounded by `2^20(1+r)^12(1+|u|)^12`; here r and |u| are the actual relative
source offsets, NOT the full circle/torus sizes. One may get this deliberately
loose bound by the path cost `8(1+r+|u|)^2`, filling/multiplicity cost
`64(1+r+|u|)^4`, and product-rule/line-integral cost
`2^11(1+r+|u|)^6`, and using `1+r+|u|<=(1+r)(1+|u|)`.
Cutoffs have slopes at most8. Time primitives are integrated before anchor
integration; spatial-line time derivatives contribute E and are retained.

Spend one quarter of the available spatial and temporal envelope exponents
on these polynomial costs. Their suprema are the explicit P12 values in (7).
Spend a further half exponent on the respective wrapping distance; the
remaining envelope is bounded by its anchored weighted mass H. This gives the
separate exponentials displayed in (7), with `exp(alpha/2)` paying minus16.
It avoids the false growth of a spatial-winding error with the WHOLE time
circle. It is valid for all T>=1, including T tending to infinity at fixed N.
No point-evaluation bound on Brownian derivatives or bound on the conditioned
background harmonic history is used. Contact atoms retain their support and
unit collapsed-time mass. Positive domination and Tonelli pay mixed
inner/outer terms as well.

This estimate is conservative and is for the *conditional* W only. The
canonical logarithm's covariance terms have not been relabelled as W''.

## 4. Fixed positive coefficients, no coupling sent to zero

Choose

\[
 \tau=\max(1,(\alpha+1)/\gamma),\quad T_N=\tau N^2,
 \quad\beta_s=\alpha/64,\quad\beta_t=\gamma/64,
 \quad X=\max(5,2/\beta_s),
\]
\[
 C_s={R_*\over8}[X^2e^{-\beta_s X}+1/(e\beta_t\tau)],
 \quad C_E=2C_{E0}+C_s,\quad C_B=2C_{B0}+C_s.                 \tag{8}
\]

Indeed `qmin^2>=16/N^2` and `1+qmin^-2<=N^2/8` for N>=5.
The spatial tail is maximized by `X^2 exp(-beta_s X)`. For EVERY
`T>=tau N^2`, the temporal term obeys
`N^2 exp(-beta_t T)<=T exp(-beta_t T)/tau<=1/(e beta_t tau)`.
Thus (5),(7) give `R_(N,T)(1+qmin^-2)<=Cs` uniformly for ALL these T,
not just T=T_N. Spatial winding and temporal-circle remnants are paid before
either limit. They tend to zero in the iterated zeroT thenN limit or the stated
diagonal exhaustion. A first-order temporal holonomy reaction is not assumed
zero: it remains in zeta with its true weight. The original Floquet bound also
gives neutral canonical/global-point error
`eta_can<=4V log[(1+exp(-mu T))/(1-exp(-mu T))]->0`; the construction does
not replace (4) by a point evaluation at finite N.

For ANY chosen bare speed c>0 set

\[
 h_E=16\max(C_E,C_B/c^2,1),\quad
 \boxed{UE=1/(2h_E)>0,\qquad K=c^2h_E>0},
 \quad\rho=\max(C_E/h_E,C_B/K)\le1/16.                         \tag{9}
\]

These are **fixed**, not N-dependent, coefficients. They can be extremely
conservative. `phase_constants(schur_lower,speed)` evaluates all formulae,
including logarithmic seam constants, and refuses nonfinite floating values.
The formulas themselves remain mathematical real constants. No agreement
with the previous `.37/.9` or `UE=1e-4/K=28.3614` finite-graph cases is claimed.
The default declared family is `s*=.01,c=.1`; the owner's CLI prints its actual
UE,K,CE,CB. Scientific constants are not outward-rounded interval certificates.

There are no integer magnetic monopole sectors: C acts on real cochains and
`dC A=0` identically. Harmonic Bloch/winding sectors remain. Absence of magnetic
monopoles is not itself the proof; (7)-(9) control the matter back reaction.
The **bare** speed is preserved algebraically, `sqrt(2UE K)=c`. No statement
that radiative renormalization leaves the physical speed exactly c follows.

## 5. Derived convexity and two-sided covariance on the true measure

On each finite-dimensional conditional quotient, with zeta fixed, let

\[
 S_\zeta(X)=\tfrac12 X^TMX-W_\zeta(X),\quad
 X^TMX=h_E\|E_X\|^2+K\|B_X\|^2.
\]

By (7)-(9), every background X obeys

\[
 (1-\rho)M\le S_\zeta''(X)\le(1+\rho)M.                       \tag{10}
\]

This is an all-background result, not a perturbative expansion at X=0.
Bounded native h and the positive determinant ensure the density is real;
(10) supplies quadratic confinement and normalizability on the physical
quotient. Constants harmonic/holonomy directions were not included in M.

For completeness, the covariance comparison is derived, not assumed.
For `f=j.X-<j.X>` solve `L u=f`, where
`L=-Delta+grad S.grad` on the strictly log-concave measure. Integration by
parts gives `<f^2>=<grad f.grad u>`. Differentiating this identity gives the
Bochner identity
`<(Lu)^2>=<||Hess u||^2+grad u.S''.grad u>`.
Cauchy-Schwarz with `(1-rho)M` therefore yields
`Var(j.X)<=j.[(1-rho)M]^-1.j`. Smooth finite-volume approximants and quadratic
confinement justify the integrations and the inverse on mean-zero functions.

For the other direction IBP gives
`<(j.X-<j.X>)(v.grad S)>=j.v` and
`<(v.grad S)^2>=<v.S''.v><=v.[(1+rho)M].v`. Optimizing the Schwarz lower
bound over v yields

\[
 {j^TM^{-1}j\over1+\rho}\le\operatorname{Var}_\zeta(j.X)
                  \le {j^TM^{-1}j\over1-\rho}.                \tag{11}
\]

**The ensemble join matters.** At every fixed harmonic history, winding,
alpha and centre, the conditional measure is spatially translation invariant.
Central seams can be distributed by their actual scalar gauge transform;
translations on gauge-invariant field strengths are ordinary translations.
Every spatial k!=0 linear field-strength/transverse source consequently has
EXACT conditional mean zero. This uses spatial, not time, translations: a
conditioned winding harmonic history need not be time-translation invariant.
Thus the law of total covariance has zero between-zeta term for these sources.
Integrating (11) against the TRUE, interacting distribution of zeta, including
(4), harmonic windings and all64 offsets, gives

\[
 \boxed{\frac1{1+\rho}M^{-1}\ \le\ C_{physical}\ \le\
                         \frac1{1-\rho}M^{-1}\quad(k\ne0).}   \tag{12}
\]

This does not claim that the full mixture is uniformly log-concave. In
particular zero-spatial-momentum/harmonic response is not assigned (12).
At any scalar source the same convexity proof applied to an exponential tilt
gives `<exp(t j.X)> <=exp[t^2 j.M^-1.j/(2(1-rho))]`. The zero conditional
means make this subGaussian bound survive the mixture. Uniform moments and
source-tail bounds follow; there is no assumption of an uncorrelated photon
Gaussian vacuum. Photon cutoff removal uses these uniform bounds.

## 6. Zero temperature, genuine 3D gapless response, and normalization

At k!=0 the transverse bare tensor is

\[
 G^0_{ij}(\omega,k)={2UE\,P^T_{ij}(k)\over\omega^2+c^2q(k)^2},
 \quad P^T=I-dd^*/q^2,\quad d_j=1-e^{ik_j}.                   \tag{13}
\]

Equation (12) compares the physical tensor in quadratic-form order, not just
one chosen axis pole. Apply the actual field-strength maps to both sides to
obtain two-sided E/B and mixed-source quadratic-form bounds. The primitive
current is exactly `delta h/delta A`: unit link charge, reverse endpoint
`-i h_yx`, and local contact `-h_xy/-h_yx`. Hence (13)'s residue `2UE` and
all response sources share the microscopic normalization; no independent
renormalized charge is inserted. The canonical integration's extra score
covariances are part of Cphysical, not deleted from it.

Take the thermodynamic/zero-temperature exhaustion T_N=tau N^2. Tightness of
smeared field strengths follows from (12) and the subGaussian bound. In three
spatial dimensions the equal-time transverse infrared integral
`integral_(|k|<1) d^3k/|k|` is finite, so the transverse potential also has
local smeared tightness; longitudinal gauge directions were quotiented.
Finite-volume translations, positivity and the canonical normalization pass
to limit points. Reflection positivity is that of the actual selfadjoint
Hamiltonian (1), not of a guessed logdet action. Since T_N tends to infinity,
the limiting local KMS states are ground states: pass their positive-frequency
spectral/KMS relation to fixed local bounded Weyl observables; negative-energy
weight is suppressed by `exp(-T_N |E|)`. Unbounded linear sources are obtained
by the subGaussian truncation. No uniform finite-N harmonic spectral gap is
required for this local ground-state limit. The nonzero-momentum bounds pass
on smeared momentum sets and then almost everywhere in k.

For a unit transverse component the zeroT spectral measure is positive and

\[
 C(\omega,k)=\int_0^\infty{2E\over\omega^2+E^2}d\sigma_k(E),
\quad C(0,k)\ge{1\over(1+\rho)Kq^2},
\quad C(t=0,k)\le{1\over(1-\rho)2\sqrt{h_EK}\,q}.             \tag{14}
\]

If the source spectral support were entirely above m, it would obey
`C(0,k)<=2 C(t=0,k)/m`. Therefore

\[
 \boxed{\inf\operatorname{supp}\sigma_k
       \le {1+\rho\over1-\rho}\,c\,q(k)\longrightarrow0.}     \tag{15}
\]

The lower covariance also excludes a vanishing source spectral weight. This
is an actual gapless coupled-photon response on three-dimensional infinite
space, not a renamed finite oscillator pole. It does not establish a delta
function pole, asymptotic Fock photons, uniqueness of the limiting state, an
exact renormalized speed/tensor or a scattering amplitude. Those materially
stronger constructions remain separate joined-campaign work.

## 7. Native consumer and its adverse controls

`noncompact_maxwell_phase.py` obtains the actual full-ten-coordinate centre123
N5 Gram, fullPhi fields and retained central seams from `original_frame`.
It calls `BulkQuantumLaw.onebody` ONLY for the original endpoint law; the
helper's selector Hamiltonian is never evaluated. Its 500-mode, three-axis
native matrices use arbitrary REAL unwrapped phases, unequal continuous-time
segments and a resolved commuting temporal/background-two source.

The independent `ContinuousGreen` source jump/AP identity is checked before
response bounds. An upper triangular exponential differentiates the ACTUAL
ordered segment evolution twice, including `X''/2`; a separate native
Frechet insertion isolates the local contact. Product-rule differentiation
forms the true monodromy derivatives. `tr(Q U''-Q U' Q U')`, with the same
Q supplied by the actual evolution, is the source Hessian. It is not a
Gaussian forwarding operation or a static equal-time replacement.

Independently perturbing the original `onebody` sources by +/-epsilon and
rerunning ordered evolution differentiates the exact physical FIRST response
with native Frechet insertions. Its centered difference is compared with the
second jet in matter-curvature units; a Ward source uses its contact scale.
The extensive logdet second difference is reported separately, not used as
the acceptance oracle: cancellation and an absolute unit floor can otherwise
make a small nonzero matter curvature indistinguishable from zero.
The static spatial plane source has nonzero B; the nonconstant temporal plane
source has nonzero E. Their actual matter curvature is compared with the PAID
finite-circle envelope and the declared positive bare Maxwell Hessian. The
signed tiny relative correction is retained separately and logarithmically:
subtracting it from 1 at these conservative coefficients would lose it in
float64. The linearized inverse correction is reported as such; it is not
labelled the integrated physical covariance. A Gaussian-only zero matter
response must fail the same independent differentiated-source assertion.

A pure spatial gradient is independently differentiated at the same native
background. Its full second Ward derivative must vanish, while its
bubble-only derivative must fail that SAME assertion. Dropping actual spatial
seam vertices must fail the independent native commutator assertion. The
reused piecewise-gauge consumer also drops temporal kicks/closing seam and
must fail actual monodromy covariance. Source normalization, holonomy phase,
full Green and all seams remain tied to the same native matrices. The separate
complete256-CAR/global-Haar control checks the full neutral exterior sector,
not a grandcanonical substitution. No Monte Carlo measure sample is claimed.

Owner command, from this proposal directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
 noncompact_maxwell_phase.py --size 5 --centre 1 2 3 \
 --schur-lower .01 --bare-speed .1 --epsilon .002 \
 --output /tmp/issue222-noncompact-maxwell-phase.json
```

For the cheaper actual noncommuting-history/canonical-density discrimination,
add `--density-only` and choose a separate output path. This retains native
geometry, Green/source/seam identification, all-mode canonical Haar nodes,
direct determinant controls and the forbidden-SU4 boundary witness; it skips
the current-Hessian slice and all64 metric samples. JSON explicitly reports
`mode=canonical_density_only` and `phase_curvature_exercised=false`; a green
result from this mode is not evidence that the omitted curvature probes ran.

All imported source/proof hashes and the new pair are bound before/after the
run, and native inputs have their own hashes. Source identity alone is not
science evidence. No science CLI, build, tests or formatter was run by this
worker; the owner's result must supply natural-exit evidence. Actual Pyright
symbols/definition/representative diagnostics and a positive native `expm`
AST query were exercised before implementation; the mounted LSP device was
unavailable and the real installed JSON-RPC server was used instead.

## 8. Full original fields through a legal Lambda(N) state/dynamics join

Use precisely the EM-independent full-Gauss W of
`noncompact_microscopic_charge_window.md` sections2-4. It retains all60V scalar
coordinates, original whole-heavy NH=2V, SU4 links/Gauss, every light state and
all64 scalar offsets. On the closed domain it intertwines all photon coordinate
multipliers, electric derivatives, every local n_x and the unbounded quadratic
photon/Coulomb form. Its residual is d_N(Lambda)->0 and its complement lower
g_N(Lambda)->infinity at each admitted fixed N. There is no fixed-Lambda
uniform-N theorem.

**Heavy/field Haar boundary.** The BL density in sections2-5 is the admitted
body effective operator's U1-canonical density, not an unrestricted
microscopic colour-Haar determinant. For its full-field preparation the
heavy/Cartesian/colour variables are integrated BEFORE identifying photon
observables. In the actual orbit chart W has the normal factor u_theta(y),
the actual complete-whole-heavy Slater line Omega_H(y), and the colour
equivariant entire light fibre, with the original orbit Jacobian and Haar
normalization. All colour transforms are unitary, and the heavy line has
unit norm. Thus the literal positive Hilbert-space preparation satisfies

\[
 \int_{\mathrm{original\ fields/heavy/Haar}}
       (W\psi)^\dagger O_{\rm photon}(W\psi)
       =\psi^\dagger O_{\rm photon}\psi,
 \qquad W^\dagger W=I.                                      \tag{16a}
\]

This integration retains the true normal/heavy/Haar weight through W's
normalized measure; it does not assign that measure an arbitrary constant
fermionic determinant or a generic SU4-twist positivity. The full heavy
Hamiltonian still has its entire fixed-number Hilbert space; its complement
is paid by g_N, not discarded from the law. The equality is exact for the
specified lifted preparation, with the finite-Lambda dynamical error below.
It is NOT equality with an unproved microscopic Gibbs marginal.

There is a direct constructive state extension that does NOT infer covariance
of an exact microscopic ground from norm-resolvent convergence. Choose an
actual effective ground/near-infimum state in the periodic neutral fibre and
a sufficiently narrow symmetric real harmonic Bloch packet; let its excess
be epsilon_N. Choose a finite Lambda(N) with

\[
 g_N>T_0(N),\qquad d_N\le N^{-4}/(1+\log N),\qquad
 \epsilon_N+\epsilon_{packet,N}\le N^{-4}.                     \tag{16}
\]

The source-owned fixed-N d/g asymptotics guarantee existence, not a numerical
recipe or invented constants. The literal full-field physical state W rho W*
has identical photon/Coulomb expectations and is within
`4d_N+epsilon_N+epsilon_packet,N` of the true microscopic neutral infimum by
the block theorem. Narrow Bloch packets recover local fibre expectations.
These are physical near-ground states of the full law along a legal trajectory,
not asserted exact finite-Lambda vacuum eigenvectors.

For bounded photon Weyl sources O, `OW=WO`, and the actual residual Duhamel
identity gives
`||exp(-it K)W-W exp(-it H_eff)||<=|t|d_N`.
Thus their full-field correlations differ from the effective ones by at most
`2|t|d_N ||O||^2`. To extend to a linear smeared transverse photon source F,
truncate it at `R=sigma sqrt(32 log N)`, where
`sigma^2=j.M^-1.j/(1-rho)`. The section5 subGaussian bound pays both discarded
L2 tails. The normalized correlator error is then bounded by a constant times
`|t|d_N log N+N^-4`. This vanishes even on the IR photon observation times
`t=O(N/c)` by (16); weighted sources are not silently assumed bounded.
Magnetic multiplication and electric Weyl sources intertwine as well;
field-strength distributions are obtained by smearing and truncation, not
by assigning the white temporal Hubbard field a Hamiltonian observable.

Consequently the effective gapless Maxwell-comparable response has a full-
field low-energy state/dynamics realization on this declared joint trajectory.
This does NOT prove that every exact microscopic ground has those covariances,
that a fixed finite Lambda full-field measure is convex, or that unbounded
source moments follow from an energy comparison alone. The explicit lift,
weighted tails, time error and order of limits are the available BO extension.
