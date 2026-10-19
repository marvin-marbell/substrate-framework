# Same-current harmonic scattering fibers

This supplier owns the conditional **matter** coefficient, not the photon
phase and not a guessed neutral harmonic distribution. It retains the true
fixed-body Gram, all `4V` fermion modes, all `2V` occupied columns, the
unfiltered creator and all four opposite-reference colors on the RIGHT.
It imports the actual current/contact and Euclidean preparation interface
from `noncompact_inclusive_scattering_join.py` without modifying that owner.

The governing plan is `issue-222-synthesis-continuation-20261006`, pinned
revision `ad6a94bfefcf9a2e3d10d60a51d344c6ac559f67a5dc41d46d7ae9f87be9f9ff`;
template `general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`.

## 1. Physical source, sea, and relative reference

For a fixed native body frame, put `q=theta/N+a_h`. The real baseline is
`theta/N-theta*1_seam`, with the actual central SU4 seam retained. Mobile's
source constructor takes **a_h**, not q: its longitudinal phase already
provides `exp[-i(theta/N)(X-X_R)]`. The complete matrix source is

`M_a(x)=eta(x) exp[-iq.(X-X_R)] V_alpha(x) W_tree(x,R)/sqrt2`.

No occupied components are removed at preparation. The whole-sea CAR gives
`||Q_a M_a||_HS=1`, while `||M_a||_HS=sqrt2`; `P_a=1_{h0(a)<0}` and
`Q_a=1-P_a`. These are conditional free-sea quantities, not assertions that
the coupled neutral ground is a Slater. The recoil detector uses the SAME
phase, tree, four references and occupied projector. Exact CT/quaternion
normalization is checked separately in every executed fiber.

The endpoint phase is dimensionless. For the model convention
`H_ph=UE p^2+omega^2 A^2/(4UE)`,
`<0|A|1>=sqrt(UE/omega)`. Beam fields here contain `1/sqrt(V omega)` and the
**conditional physical candidate** two-photon amplitude is `UE*C(a)`, with
square `UE^2*|C(a)|^2`. There is no extra factor 1/2 or charge rescaling;
actual receiver probabilities also require their declared Gram/reference.

The scalar `Esea(a)=sum_{occupied} E_n(a)` is retained in
`H_h=UE/V p_a^2+Esea(a)`. A standalone canonical-sea amplitude has a scalar
factor `exp[-it Esea(a)]`. Its derivative includes `-it Esea'(a)`; its
second derivative includes `-it Esea''(a)-t^2 Esea'(a)^2` and the source
cross derivative. The consumer checks the full first scalar jet against
literal shifted whole-sea spectra. It rejects deleting Esea or X_R from
that jet. In the relative charged/neutral source history this scalar
**cancels against the same neutral reference**, exactly. It must neither
be deleted from the harmonic Hamiltonian nor be added again as a spurious
extensive phase in the relative coefficient derivative. The motion bound
below is for the relative/covariant coefficient after this cancellation.

## 2. Exact prepared coefficient, including the negative-source boundary

Let `O` be the complete occupied isometry, `P=OOdag`, and write the
interaction-picture literal Peierls propagator as
`A(x,y)=I+x a+y b+xy c+...`, where x is the incoming and y the outgoing
canonical photon coordinate without its common sqrtUE leg. Then

- `a=-i integral J_in(t) dt`, `b=-i integral J_out(t) dt`;
- `c=-i integral C_in,out(t) dt`
  `-integral_{t>s}[J_out(t)J_in(s)+J_in(t)J_out(s)] ds dt`;
- `X=-Q J_out(0)P/(E_plus-E_minus+omega_out)` is the initial
  full particle-hole preparation tangent, summed over actual photon modes;
- `Ydag` is the final tangent obtained by taking the dagger of the supplier's
  `preparation_tangent(annihilation_packet(incoming),T)`.

The quotient of full charged and neutral exterior determinants has output

`R(x,y)M=[A-F S^-1 Gdag A]M`,
`F=A(O+y X O)`, `G=O+x Y O`, `S=Gdag F`.

All occupied columns occur in S. The code represents inverse jets in the
ambient P-subspace; its zeroth inverse is P, not the full-space identity.
The polynomial has the following exact jets:

`F0=P, Fx=aP, Fy=bP+X, Fxy=cP+aX`,
`H0=P, Hx=Pa+Ydag, Hy=Pb, Hxy=Pc+Ydag b`,
`Sx=PaP, Sy=PbP`,
`Sxy=PcP+PaX+Ydag bP+Ydag X`,
`(S^-1)xy=Sx Sy+Sy Sx-Sxy`.

Thus `Rxy=c-(F S^-1 H)xy`. This identity does **not** replace M by QM.
`P M` contributes through the preparation boundary and is kept. For X=Y=0
and positive source/detector, this reduces to the supplier's fullCAR
open-line `J2 Q J1-J1 P J2`, both photon histories and contact. For nonzero
X/Y the unfiltered negative-source boundary is independently discriminated.
The final Euclidean detector is also **unfiltered**: the scalar is
`<D_full,Rxy M_full>`, not `<Q D_full,Rxy M_full>`. The latter is separately
reported as a positive final-chart probe. Deleting the final negative-source
boundary is rejected, just as deleting the initial one is rejected.
The native consumer compares this exact jet with literal endpoint
exponentials and the whole occupied bordered exterior determinant.

For `f_i,f_o` the sum of mode coefficient magnitudes times maximum component
polarization, divided by `sqrt(V omega)`, set

`j_i=6 tau f_i`, `j_o=6 tau f_o`, `k=6 tau f_i f_o`,
`a0=T j_i`, `b0=T j_o`, `c0=T k+T^2 j_i j_o`,
`x0=j_o/(2mu+omega_o,min)`, `y0=j_i/(2mu+omega_i,min)`.

Here tau bounds the **complete Wilson plus odd** endpoint spectral norm;
`mu=.2 sqrt(Schur(G))` is the uniform full Bloch gap in this body frame.
The photon frequencies remain the actual multiband frequencies. Triangle
inequality on the displayed polynomial gives

`||Rxy|| <= 4c0+8a0b0+4a0x0+4b0y0+2x0y0`.

The code sums its explicit words instead of assuming cancellations.
Therefore `|C(a)|=|<D_a,Rxy M_a>| <=2 ||Rxy||`, uniformly in a;
both `||Rxy M_a||^2` and `||Q_a Rxy M_a||^2` are at most `2||Rxy||^2`. These bounds include the initial/final
jets and their exchange cross term. They are deliberately conservative.

## 3. Gauge covariance stops at the actual periodic seams

Let `D_q(x)=exp[-iq.(X-X_R)]`. In the local coordinate chart,
`D_qdag h0(a) D_q=h_*+Delta_seam(q)`, where `h_*` is h0 at the
**compensating** harmonic value `a_*=-theta/N`. It has the same body Gram,
not another well. The source/detector become exactly the same compact
matrices. Literal Peierls current and mixed contact transform with h;
there is no leftover bulk current phase. Only wrap edges differ.

`||Delta_seam|| <=12 tau` for every real q. This is not small because a is
in a small-looking chart. Large-gauge gluing shifts a by `2pi/N`; the light
gauge factor is `exp[-i(2pi/N)X]` and the opposite reference contributes
`exp[+i(2pi/N)X_R]`. The consumer checks H and the complete prepared scalar
coefficient at both x-cell faces. Source phase-only equality without the
seam cost is explicitly rejected.

For compact source support S, set `w(x)=exp[kappa dist(x,S)]`,
`kappa=log(1+mu/(24tau))`, `g=6tau(exp(kappa)-1)=mu/4`.
Let d be the periodic graph distance from S to vertices incident on any
coordinate seam. For the whole spectral bound `B=||onsite||+6tau`, take
negative/positive rectangular contours at distance at least mu/2 and
perimeter at most `4(B+mu)`. All real-harmonic spectra lie in `[-B,B]`
with the same gap. The conjugated hopping perturbation has norm at most g,
so contour resolvents have weighted norm at most `4/mu`, ordinary norm
at most `2/mu`. Put `Cp=8(B+mu)/(pi mu)`.

For every primitive L in the Rxy polynomial the code supplies

`||L||<=n_L`, `||w L w^-1||<=l_L`,
`||(L_a-L_*)v||<=d_L exp(-kappa d) ||w v||`.

For P, `n_P=1`, `l_P=Cp`, `d_P=4 Cp*(12tau)/mu`.
Real propagation has `l_U<=exp(g|t|)` and
`d_U<=(12tau)|t|exp(g|t|)`. A current has
`l_J<=exp(kappa) j`, `d_J<=(12tau) f`; contact uses f_i f_o.
These give, for a/b,

`l_a<=T exp(2gT) exp(kappa) j_i`,
`d_a<=T exp(2gT)[12tau f_i+2(12tau)T exp(kappa)j_i]`,

and the same outgoing formula for b. Contact and the two ordered histories give
the explicit c formulas in `uniform_enclosure`; both histories are paid.

For X/Y use the two full spectral contours with denominator
`z_plus-z_minus+omega`, whose real part is at least `mu+omega`. Then

`l_X<=Cp^2 exp(kappa)j_o/(mu+omega_o,min)`,
`d_X<=Cp^2[12tau f_o+8(12tau)exp(kappa)j_o/mu]/(mu+omega_o,min)`.

The same formulas with the incoming packet apply to Ydag. They bound the
**infinite Euclidean preparation integration**, not a finite Euclidean
cutoff. Thus neither a preparation seam nor an occupied projector seam is
silently unpaid.

For a word `L1...Ln`, telescoping gives its coefficient difference bound

`exp(-kappa d) sum_r d_Lr product_{s<r} n_Ls product_{s>r} l_Ls`.

Summing all exact words gives D_R. The optional positive output Q adds a
projector-seam term; retaining that nonnegative extra debt also bounds the
unfiltered full output. The full source and detector each have norm sqrt2;
`|C(a)-C_*| <= E_seam=2 D_QR exp(-kappa d)` for **every** harmonic value.
The code prints a finite distance needed for E_seam<=.001. This proof is
localization/finite-flight plus massive-Bloch contours; it has no seam area,
volume, arbitrary harmonic sample count or assumed marginal density factor.
The raw massive tail is also intersected with the ordinary uniform triangle
bound `|C(a)-C_*|<=2*conditional_amplitude_absolute_upper`. This prevents
an unadmitted small-volume exponential estimate from weakening a proven
ordinary enclosure.

This is an enclosure, not a claim that N3 or another small grid has an
admitted small seam error. When `|C_*|>E_seam+other_paid_coefficient_debts`,
a nonzero amplitude lower bound follows for every actual marginal. The
current small-volume consumer does not certify that inequality. No
nonzero coupled-ground lower bound is inferred from a uniform upper bound.

## 4. Integrating the actual marginal, without specifying it

For any actual normalized harmonic probability measure nu (continuous,
discrete, singular or correlated after disintegration), the uniform result
implies

`|integral C(a) dnu-C_*|<=E_seam`,
`integral ||Q_a Rxy M_a||^2 dnu <=2||Rxy||^2`,
`integral |C(a)|^2 dnu >=max(0,|C_*|-E_seam)^2`.

`marginal_enclosure(C_*, coefficient_upper, uniform_error)` returns this
disk and probability interval without taking or manufacturing a measure.
Its small-volume lower endpoint is explicitly zero whenever the uniform
radius contains the reference coefficient.

For inclusive outputs, compare gauge-transformed vectors before integrating
or use the same uniform norm difference. For an off-diagonal harmonic
kernel or a general entangled photon-mode/source receiver block, its actual
exchange Gram and whitening are still required; a scalar marginal average
is not a substitute for that block. PhotonPacketSupplier owns that debt.
The executed fibers are positive controls, never quadrature points used to
invent nu or prove uniformity. The uniform proof is the contour/telescoping
argument above. Replacing the actual neutral matter conditional state with
its free sea remains a separate SameCurrent preparation/comparison debt.

## 5. Harmonic motion is paid, not declared higher order

For any relative coefficient/source map K(a), harmonic kinetic gives

`[UE/V p_a^2,K]Omega=-(UE/V)[Delta_a K Omega+2 grad_a K.grad_a Omega]`.

The actual scalar-paid harmonic energy-density moment EH implies
`||p_a Omega||<=V sqrt(EH/UE)`. This is Mobile's moment, retained with
Esea and its scalar variational comparison; deleting Esea does not supply
this estimate. Hence finite-time Duhamel costs
The native receipt takes EH from Mobile's `theorem_bounds` scalar-paid
energy-density field and carries that supplier's admission requirements.
The actual-ground harmonic moment remains conditional on those hypotheses;
the native small-grid coefficient execution is not an actual-ground test.

`T[2sqrt(UE EH) D1+UE/V D2]`,

where D1 bounds the combined source/coefficient first derivative and D2
its Laplacian. `uniform_enclosure` computes explicit D1,D2 for **Rxy M**,
conservatively also paying an optional Q projection, not only the free source.
For each axis `||h'||,||h''||<=2tau`; let
`u1=1+2T(2tau)`, `u2=1+6T(2tau)+4T^2(2tau)^2`.
Real-current primitives have first/second bounds `T j u1,T j u2`.
The c bounds include both derivative placements:
`c1<=T k u1+2T^2 j_i j_o u1`,
`c2<=T k u2+T^2 j_i j_o(2u2+2u1^2)`.
Contour projector bounds are
`P1<=Cp*4(2tau)/mu`,
`P2<=Cp[4(2tau)/mu+32(2tau)^2/mu^2]`.
The double contour gives
`X1<=Cp^2 j_o[1+8(2tau)/mu]/(mu+omega_o,min)`,
`X2<=Cp^2 j_o[1+24(2tau)/mu+96(2tau)^2/mu^2]/(mu+omega_o,min)`.
Y has the incoming analog. All primitive source/reference jets are obtained
by the product rule. The source moments are
`M1_j=sqrt(2 sum|eta|^2(X_j-X_R,j)^2)`,
`M2_j=sqrt(2 sum|eta|^2(X_j-X_R,j)^4)`.
The consumer includes the extra measured Q projector derivatives.

Multiplying by the physical two-photon factor UE yields a motion amplitude
error `UE*T[2sqrt(UE EH)D1+UE/V D2]`: at **fixed** T, source geometry, gap,
photon band and finite coefficient jets, with bounded scalar-paid EH, this
is `O(UE^(3/2))+O(UE^2/V)`, whereas the prepared leading amplitude is O(UE).
This is a derived ordering, not valid if N, T, inverse gap, jet norms or EH
are allowed to diverge with UE. In particular the initial photon ground
preparation is itself O(UE) and is already in Rxy; it cannot be discarded
under this motion argument. Actual coupled neutral photon/charged graph
moments must satisfy their own source comparison bounds before claiming a
closed coupled scattering result.

## 6. Native consumer and honest boundaries

Run the project Python on this executable with `--size 3 --time 2 --nodes 8
--exterior-steps 32 --output /chosen/result.json`. It evaluates an arbitrary
three-axis fiber, both opposite large-gauge cell faces, and the compensating
zero-light-holonomy fiber. For every fiber it retains the canonical sea and
actual phases and reports the FULL current/contact/preparation scalar,
the separate positive-output inclusive chart probe and full contour HS norm,
quadrature error, Esea and four-reference receipts. It checks:

- exact local source covariance AND prepared coefficient rejection of omitted theta/N or a_h;
- complete-reference and BOTH initial/final unfiltered-negative-source preparation boundaries;
- actual large-gauge H and prepared coefficient gluing;
- independent literal Peierls full-exterior versus polynomial contour;
- source/reference/whole-sea scalar jets against finite harmonic shifts;
- the conditional coefficient's harmonic kinetic product-rule cross term
  with a periodic Fourier probe, rejecting omission of `2 grad K.grad phi`;
- omission of contact and either real-time current history;
- analytic all-fiber norm/seam enclosure and rejection of unpaid seam equality.

The JSON binds exact local source hashes, input, Python/numpy/scipy runtime
and the plan/template. Natural process exit is recorded by the runner.
This is a finite conditional free-sea preparation/current/contact consumer,
not a simulated coupled neutral ground, not a replacement marginal and not
a certified nonzero actual coupled probability. The constructive next
integration is the supplier's actual neutral preparation replacement bound,
using these uniform coefficient jets and the scalar-paid harmonic motion
budget; the alternative is to sharpen the local weighted words to admit a
reference coefficient enclosure at useful N without a huge-grid simulation.

## 7. Sparse compensated Bloch coefficient and a positive SAME-N guard

The selected continuation keeps the declared observable. PhotonPacketSupplier
confirmed that its receiver is GLOBAL and coherent:
`x_i=a_dagger(d_i) B_eta_i Omega/sqrt2`, `Y=X G_actual^(-1/2)`,
`E_i=|Y_i><Y_i|`. The bar4 reference is inside each correlated vector.
It is neither four separate outcomes nor an identity harmonic detector.
Accordingly the bound remains on the coherent scalar `integral C(a)dnu`.
Replacing it by `integral ||C_vec(a)||^2 dnu` would change the observable and
is not used to manufacture positivity.

### Finite-N normalization, not a transplanted continuum amplitude

For N a multiple of5, retain the SAME two photon mode numbers `N/5,N/5+1`,
weights `1,.25` normalized by `sqrt(1+.25^2)`, native frequencies
`omega=c|1-exp(ik)|`, incoming x-cone and outgoing y-cone/z-polarization.
The second mode separation is `2pi/N`; the frequency band narrows with N.
Keep charge width.8/carrier[1.2,0,0], all19 compact seed sites and the
recoil detector `Delta=2pi/N*(1,-1,0)`. Thus charge recoil also changes with
N exactly as the existing comparator specifies. No fixed finite bandwidth
or recoil is secretly asserted in the N limit.

Each normalized photon supplies `V^(-1/2)`, and each compact charge Fourier
source/detector supplies `V^(-1/2)`. Consequently

`C_connected,N = V^(-1) * mean_{p in nativeBZ_N} f_N(p)`,
`A_physical,N = UE/V * mean f_N`.

`f_N` still depends on N through the actual mode frequencies, polarization,
recoil and photon phases. The moderate BZ integral evaluated by the code is
an intermediate enclosure of the SAME finite-N sum, not a fixed C_infinity
promoted to every N. Only the two-photon coordinate V factor is stripped.
Photon packet centers cancel the common compact charge/source origin phase;
the outgoing displacement[0,1,0] and the opposite-reference phases remain.
The square of this unwhitened candidate has its `1/V^2` factor.
No actual receiver probability, cross-section or enlarged receiver is inferred.

### Full4x4 occupied/empty calculus and initial cloud subtraction

The actual fixed-frame Clifford symbol obeys `h(p)^2=E(p)^2 I`, with
the full occupied projector `P=(I-h/E)/2` and `Q=(I+h/E)/2`. No band
eigenvector is selected and no occupied source component is removed.
For a photon shifting momentum by k, the exact interaction current is

`J_I(t,p)=exp[it h(p+k)] J(p,k) exp[-it h(p)] exp[-is omega t]`.

Resolve both full spectral clusters. First-current integrals are entire
functions `F(delta)=integral_0^T exp(i delta t)dt`. Both real histories use
`I(alpha,beta)=integral_0^Tdt integral_0^t ds exp(i alpha t+i beta s)`.
The mixed contact is the literal second Peierls endpoint derivative.
Euclidean initial/final shifts are exactly

`X(p)= -Q(p+k_o) J_o(p) P(p)/(E(p+k_o)+E(p)+omega_o)`,
`Y_T(p)= -P(p+k_i) J_i(p) Q(p) exp[-iT(Eout+Ein+omega_i)]`
`          /(Eout+Ein+omega_i)`.

The verified reduced full-source operator is

`Rxy=Q(c-aPb-bPa)Q-QaX-XaQ-QbY-YbQ-XY+YX`.

The code subtracts its exact zero-time charged cloud `-X Y_0+Y_0 X`,
not zero, not a neutral cloud and not a tiny-time substitute. It computes
the full detector trace of `Rxy(T)-Rxy(0)`, preserving both initial and
final negative-source boundaries. No giant occupied matrix, charged Fock
matrix, native N^3 Hamiltonian or ground simulation is formed.

An independent native N5 bridge using the actual G5 and the same packets
gave `V(C6-C0)=2.031761736767851-.20692415213154264i`, differing from
SameCurrent's executed full-sea reference by `2.66815e-15`. This bridge
fixes both the sparse algebra and the V normalization; its value is not
the coefficient used at the selected larger N.

### Certified moderate quadrature, native Gram and roundoff debts

The NativeMetricAdmission certificate supplies the trial
`Gtrial=diag(.4187253908699568,.0632221974274214,... )`, full-ten residual,
analytic strip sigma.5 and strong-convexity constant2. Every real shifted
native G_N satisfies

`||G_N-Gtrial||_F <= [residual+continuum_gradient_error`
`      +M_gradient(((1+exp(-sigma N))/(1-exp(-sigma N)))^3-1)]/2`.

The principal square-root frame is used on both sides. Its Lipschitz
constant is `1/(sqrt(gs)+sqrt(gs-distance))`. The source quaternion labels
do NOT require a guessed field-independent source:
`G^(-1/2) sqrt(G) SELECTED=SELECTED` proves their exact constancy in this
actual canonical frame. The unfiltered source and tree are otherwise the
same. Primitive full4x4 real-time/projector/Euclidean perturbation bounds,
followed by telescoping the reduced words, pay the actual G_N-trial error.
This is not a replacement of the full-ten minimizer by a diagonal ansatz.

For complex `|Im p_j|<=sigma`, beta-accretivity gives
`Re E>=cbar>0`. Therefore
`||P||,||Q|| <= (1+B_complex/cbar)/2`. Since the Hermitian imaginary part
of h has norm at most
`b_im=[3r sqrt(g0)+v sqrt(3gs)] sinh(sigma)`,
Gronwall gives `||exp(+-it h)||<=exp(b_im|t|)`. The two infinite Euclidean
denominators have real part at least `2cbar+omega`.
These bounds give an explicit holomorphic majorant M for **f_N**, including
source Fourier tails and all full-source/cloud-subtracted words. For a
q^3 trapezoid rule,

`error(q) <= M[((1+exp(-sigma q))/(1-exp(-sigma q)))^3-1]`.

The SAME bound at N pays the finite native sum versus this intermediate
continuum integral. All scalar strip/error computations use the existing
Arb128 runtime and outward safety margins. Source/input hashes bind the
exact NativeMetric certificate, not an operator-selected unverified trial.

The numerical integrand does not borrow an unproved vendor transcendental
accuracy assumption. `bounded_phase` reduces |argument|<=64 to[-pi,pi] and
uses degree36/37 cosine/sine Horner polynomials. `bounded_sinc` uses the
degree80 even Taylor polynomial on |x|<=8. Arb encloses Taylor remainders,
coefficient rounding, argument reduction and the standard IEEE gamma_n
arithmetic bound; they fit the explicit elementary error `1e-10`.
Small ordered denominators are evaluated by the entire moment series, with
`|beta T|<=.25`: eleven beta terms and forty small-alpha terms. For
`|alpha T|>.5`, moment recurrence errors are weighted by
`|beta/alpha|<=.5`; thus a small-denominator cancellation is not suppressed.
The code pays `300 T^2*elementary_error` for that kernel plus its energy
argument sensitivity. Full spectral projector, current, contact, both
histories and X/Y error placements are propagated through the reduced
words; source Fourier and final positive-weight sum rounding are paid too.
The proof surface is IEEE binary64 basic arithmetic/sqrt, 0<T<=6, and
the explicitly enforced phase/sinc domains.

### Positive coherent conditional lower and receiver boundary

Let z be the computed volume-rescaled SAME-N compensated coefficient, and
`e_ref=e_quadrature+e_native_periodization+e_native_G+e_arithmetic`.
Independently pay BOTH T and T0 full prepared harmonic seams:
`e_seam=V*(E_seam,T+E_seam,0)`. Then

`L_N=|z|-e_ref-e_seam >0`

certifies, for every normalized actual harmonic marginal,

`|integral C_connected,N(a)dnu| >= L_N/V`.

This defeats phase winding by a uniform coherent disk, without changing
the POVM. The native consumer chooses finite N2000 and a moderate80^3 BZ
rule, rather than simulating the enormous native lattice. It prints each
debt, the exact cut distance `N/2-2` and the positive SAME-N guard.

The observed N2000/q80 consumer gives
`z=1.3785256963806265-.48619783538721856i`,
`e_ref=.03134522679540843`, `e_seam=1.6048932178673353e-17`,
and `L_N=1.4304076355985422`. Thus every actual normalized harmonic marginal
has unwhitened connected coefficient modulus at least
`1.7880095444981778e-10`. The **conditional candidate** physical amplitude
lower is `2.758799058299553e-158`, square `7.61097223e-316`, using the unchanged
N5-anchored positive UE/K family. The N-dependent frequencies are
`.1475079277813101,.14782666012015402`; the photon modes are400/401.
None of these small physical numbers is a volume-independent scattering
probability or an already-normalized actual receiver effect.

The entire-kernel small-denominator/boundary probe agrees with independent
64-node time quadrature to `5.68434e-14`. The N5 physical-normalization bridge
agrees to `2.10693e-17`; deleting the photon V factor gives a defect
`2.02593344458`, rather than another equivalent convention.

The corresponding `UE*L_N/V` and its square are explicitly labeled
**unwhitened conditional** quantities. They are not promoted to actual
`E_i=|Y_i><Y_i|` probabilities: actual-neutral replacement, leading actual
global Gram/off-Gram control and the complete same-current amputation/
finite-flight/legal-Lambda remainders must survive the positive margin.
In particular whitening can contribute at leading UE and is not bounded
by merely reporting that a global Gram exists. SameCurrent and Photon
retain that consumer contract.

The declared receiver is static at readout T, while the physical incoming
state is `q=a_dagger(g) B_eta Omega/sqrt2`, normalized by its actual norm n,
not an input-whitened vector. Its amplitude is therefore
`A=G_out^(-1/2) m(T)/n`. Bare `m_y0=0`, but `m_x0` is generally not one:
photon centers differ and charged postselection remains static. If
`G_out=I+UE*g1+o(UE)`, the leading y component is
`UE*(m_y1-.5*g1_yx*m_x0)/n0`. Thus the positive `C_T-C_0` disk is still
only a candidate until both the static reference and this leading cloud
mixing are matched. A symmetric expression
`G^(-1/2)(M-G)G^(-1/2)` would require a different, actually declared
symmetric reference/input contract; it is not the present receiver.

Use `--bloch-lower --target-size 2000 --quadrature 80 --time 6
--metric-certificate /the/existing/native-metric-state-packet.json
--output /chosen/bloch-result.json`. The original small-grid literal
consumer remains available without `--bloch-lower`.

