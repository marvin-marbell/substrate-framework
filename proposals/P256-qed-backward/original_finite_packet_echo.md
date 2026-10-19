# Guarded spin plateau: an entire-orbital finite-width coefficient echo

## 1. Binding, law, and the concrete decision

Shared plan `issue-222-continuing-backward-20260930`, revision `a406dca7a4cc642318250c028a763daa96fa66a5999da5dae0c96e53910c5c91`, pinned `general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`, work item campaign, requirements outcome/verification. Main owns scientific execution, receipts, shared-plan/world-model integration and public claims. Only this proof and `original_finite_packet_echo.py` are new files. No existing source or world-model was edited.

**Decision:** use a bounded transport support with its entire one-edge halo strictly inside a larger constant-spin plateau. Do not attempt another point-row echo or tie transport amplitude to the inverse-screened torque. On the isometric central BODY this materially repairs the finite-width obstruction: the global orbital/connection commutator is zero, so reverse transport removes the entire ordered orbital evolution, not just one row. Occupation conditioning moves during the pulse and returns at its endpoint. This is an exact source-only, BODY-relative finite-width result. Its actual-ground/full-H transfer remains conditional; it is not the mechanical spin of a proved mobile electron.

Retain the native incumbent law: m=.2, r=.12, v=.3, kappa=2/a, chi=Lambda^(-1/2), field kinetic chi/2, colour electric .3chi, J=.2Lambda^(5/4), K4=.7Lambda^(5/4). Native photon coefficients UE,K are unspecified fixed positive symbols. Physical photons keep the cosine law, not a Gaussian/Villain Hamiltonian. Keep all60V fields, 3V SU4 and compact U1 links, their cotangents, whole NH=2V/fixed-NL CAR, all15V Gauss/background2, the opposite four/bar-four and U1 reference at the SAME R, every64 true ten-Gram/weight/offset, all periodic packets, torons, images and harmonics. Admission is N>=max(5,N0(a)); N0 is existence-only. A runnable N9 diagnostic is not a numerical admission certificate.

Sources actually read: current `world-model/modules/spin.md`; archived localized construction CA86 §§3–7; complete executed 90F7 source; full normal construction6867 §§2–6 and actual helpers; full original-heat E802 §§1–5 and finite-B §11; regional E1A4 actual heat/path helpers; native `original_total_angular_transfer.py`, `original_charged_principal.py`, `original_uniform_charge_gap.py`, and `original_slow_kinetic_bo.md`. These retain the quantum-statistics assumption; no Euler derivation is claimed.

## 2. Same original primitive, independently specified controllers

For axis i, use exactly

```
YPhi = -v_j D_j Phi + f R_i Phi,
YU   = -ia sum_j vbar_j B4bar_ji U,
YA   = -a sum_j vbar_j B1bar_ji,
J(v,f) = -i(Y.partial + divY/2) + dGamma_L(o(U,A;v))
                                  + dGamma_H(o(U,0;v)).
```

The actual four oriented clovers are anchored at each vertex; reverse seam edges use inverse links, electromagnetic flux uses sine, and the scalar orbital edge is `vbar_j U_xy exp(iA_xy)/(2ia)` with its Hermitian reverse. No band spin matrix, Gauss charge, hidden coupler or endpoint apparatus matrix is added to J. The two permitted source inputs are

```
J1 = J(v_i,f),  J2 = J(-v_i,0),
v_i(x) = g_x e_i cross(x-x0),
f = b_spin + (c/lambda)L b_spin,  c=1/.6.
```

At the isometric central BODY, `If s_i=lambda s_i`, `Df^T r_i=If s_i`; therefore the complete inverse-screened equation `(If+cL)ell=f Df^T r_i` gives `ell_x=b_spin(x) S_i`. The spatial S_i is derived from normalized actual Cartesian fields: `S_i=-i[Gamma_j,Gamma_k]/4`. On the plateau `exp(2pi iS_i)=-I`. R is outside the plateau and buffer, so the ideal connection there is zero. In a tilted true Gram solve the actual15V equation; do not infer ell_R=0 merely from f_R=0.

Choose support C of g and require that **C and every endpoint of every one-edge neighbour of C lie strictly inside b_spin=1**. This stronger condition covers an edge with only one velocity endpoint in C. The source/orbital graph is defined by actual nonzero edge coefficients, not by a visual radius. The supplied consumer uses cube1 support, cube2 halo, cube3 spin plateau on odd N>=9, g=.25 in cube1 and zero elsewhere, with the same profiles for all3 axes. The finite lattice samples admit smooth continuum interpolation; none is a domain boundary for the microscopic fields/CAR. The velocity is bounded independently of Lambda and of f's signed compensation shell.

## 3. Global commutation and reversal of the entire orbital propagator

Write `q=diag_x(b_spin(x) S_i)`. For every nonzero original orbital edge both endpoints have b_spin=1. A central colour link is a scalar centre element, and the U1 factor is scalar. Thus

```
o_xy q_y - q_x o_xy = 0
```

on every nonzero forward or reverse edge. On zero edges the identity is automatic. On all diagonal blocks the scalar orbital is zero. Consequently

```
[o(A),q]=0 GLOBALLY for every compact A,
[dGamma(o(A)),dGamma(q)]=0 in EVERY occupation sector.
```

This is stronger than support-local commutation on the initial packet. There is no annular orbital edge crossing into a changing q. It includes any relevant reverse seam. Outside the orbital graph q may vary and the intrinsic/normal spectators may evolve; nothing asserts an inverse full annular microscopic state.

Use source angle theta, observer photon characteristic `A'=-YA(v,A)`. Let V solve `V'=i o(A(theta))V`, V(0)=I on all4V rows. The complete compact photon ODE is autonomous, smooth and linear in v. Controller2 has the opposite vector field at **every** A and is therefore its inverse full characteristic, including torons. Its actual half-density Jacobian cancels the first. Along the reverse path its orbital coefficient is exactly `-o(A(2pi-theta))`. Ordered differentiation, not a commute-between-times assumption, gives V2=V1^(-1). Individual o at distinct times need not commute.

Since q commutes with o at every time and is independent of A, controller1 factorizes on the BODY fibre as `V1 exp(i theta q)`. The torqueless second controller has ell2=0 at BODY and removes V1, leaving

```
E_echo = exp(2pi i q).
```

The same statement holds for whole light/heavy exterior functors. Heavy motion is common to the two preparation histories and is retained. The full neutral RIGHT is retained, not replaced by identity. SU4 normal restoring/vertical holonomy is **not** reversed by J2: that is why the intrinsic sign survives. This is the BODY-effective source obtained from full normal/pc cancellation, not the free bare SU4 characteristic without restoring forces.

**Failure and repair relative to CA86 §7:** if the halo touches a changing plateau, or a colour link on a nonzero edge does not commute with S_i, this factorization fails. Ordinary photon reversal alone does not remove the ordered orbital/connection commutator. Merely making the initial packet small is insufficient. The guard-collar protocol repairs it globally. Noncentral links which happen to lie in the spin centralizer are not adverse; the consumer deliberately selects a noncommuting SU4 perturbation.

## 4. The new finite-B state and its whole-CAR projector endpoint

Use a new finite compact packet eta supported inside the plateau, with at least two nonzero rows,

```
M_alpha(x,A) = eta_x F_xR(A) V_alpha(Phi_x) W_xR(U),
B_alpha = C_Malpha P_B,
P_B = product_(x in B) 1_(n_L(x)=2),
F_xR = I_(delta_x-delta_R)/I0.
```

The SAME opposite reference is part of C_M. The original everywhere-nonzero Gaussian preparation is unchanged and is not identified with this new state. B may include any finite sites inside the plateau, not only the two diagnostic rows. All occupations and all original hopping terms remain in H; P_B is initial conditioning, never P_B H P_B.

On the entire CAR, `C_M^dagger C_N=Tr(M^dagger N)-dGamma(NM^dagger)`. For distinct x,y in B, compression by P_B kills `c_x^dagger T c_y`, because it changes (2,2) to (3,1). Each diagonal label has `P_B C_Malpha^dagger C_Malpha P_B=2s P_B`, `s=sum_B |eta_x F_xR|^2`. The original invariant-ground quaternion/C or T scalarization supplies the mixed entries on aligned BODY, hence

```
Z_B=<s P_B>,  G_initial=2 Z_B I4.
```

Use the original positive canonical/temporal-Haar multivariate midpoint supplier, not generic C/Gauss alone: `Z_B>=3^(-|B|)<s>`. At each fixed admitted N the actual groundspace minimum of the analytic nonzero s is positive; the inherited mixture gives Z_B>0. A uniform thermodynamic capacity lower bound would still need the original HC supplier. Finite-B conditioning costs the explicit 3^(-|B|) floor.

Let E(theta) be the complete charged row evolution along the first/second BODY source path. Its actual pushed projector is

```
P_B(theta)=Gamma(E(theta)) P_B Gamma(E(theta))^dagger,
P_B' = i[dGamma(o+q),P_B]                  [first leg],
P_B' = -i[dGamma(o),P_B]                    [second leg].
```

The connection q commutes with every local number projector, but o generally does not. During both legs P_B(theta) is a whole-CAR operator and is generally not a product of on-site projectors. The row's changing packet subspace is not silently frozen. At the endpoint E_echo is site-block diagonal, so **P_B(4pi)=P_B exactly**. On every packet row E_echo=-I; ell_R=0 and the full photon characteristic/Jacobian has returned. Therefore, as an operator intertwiner between full neutral and charged-plus-SAME-reference spaces,

```
U_echo,+ B_alpha = - B_alpha U_echo,0.              (FW-ECHO)
```

This holds for arbitrary finite eta, arbitrary smooth complete-heat multiplier, and every complex quartet. It restores the entire initial coefficient/preparation map and P_B, not just a point coefficient. Every individual orbital row can move on the way. It does not claim that the entire neutral state, all normal/annular fields or their cotangents return.

Taking adjoints of FW-ECHO and using unitarity also gives `[U_echo,0,B_alpha^dagger B_beta]=0`. Thus on the SAME initial neutral purification the two unnormalized history Grams are both G_initial and their cross matrix is **-G_initial=-2Z_B I4**. Only the initial inverse Gram is used. There is no postnormalization, no removal of photon/preparation loss by redefining the final preparation, and no assumption U_echo,0=I. The all-angle response still contains real orbital and projector motion; it is not asserted to be a pure spin matrix.

This local two-controller operation is not a pure global SO3 operation. J2 is a torqueless countertransport source, not a second rigid-space rotation; nonuniform f/g annuli are not a closed SO3 action. The previous microscopic global2pi=+1, including wholeNH/fixedNL and SAME reference factors, is unchanged.

## 5. Actual-normal transfer on this SAME finite-B state

The primitive-to-BODY step is required, not a new source law. In product-mass quotient RNC retain all90V physical normals and all15V Gauss rows:

```
M0=diag(I60V,c I45V), Gamma0=Df^T Df+cL,
ell_j=Gamma0^(-1) Df^T Yj_field,
c_j=Yj(0)-G0 ell_j,
B_j=P0(DYj(0)-dG[.]ell_j),
chi p_c,j = omega_j(c_j+B_j y_c,j)+y_c,j'.
```

For j1 the same f generates the nonzero field/link affine residual; for j2, ell2=c2=0 on central BODY, but **B2 generally is nonzero**. The complete covariant advection, clover and RNC/BCH link jets survive. The consumer constructs both actual all-coordinate JVPs on the full105V raw space, projects the full15V mass Gauss equations, and compares each with a finite difference of the real primitive in RNC coordinates. It retains all90V directions; it does not select an oscillator. The selected isometric diagnostic frame/all3 axes are not an all64 quantum-ground computation. The exact all64 geometric solves are a distinct output.

The original kinetic product rule is

```
[chi p_y^2/2,C_M] = -i chi C_(partial_y M).p_y -chi C_(Delta_y M)/2.
```

Combining the actual centroid with this rule, including Wilson derivatives, supplies `ell_x M-M ell_R+oM-i YA.partial_A M`, with large pc rather than p=0. The squeezed covariance and common metaplectic/centroid action, scalar offsets, Riesz/heavy/Born–Huang/Haar/contact/divergence terms remain in BOTH histories. Each leg has duration T=Lambda^(-1/32), with smooth source amplitude of total angle2pi. The entire normal evolution starts its second leg with the first leg's actual squeezed state/centroid, not a fresh vacuum. The general normal estimate applies on that transported finite first-Sobolev family with the actual covariance majorant.

For this finite-B preparation, the original P_B has no coordinate derivative. Its pushed version has derivatives which must **not** be erased. If K_j is the full fibre generator,

```
partial_r P' = i[K_j,partial_r P]+i[partial_r K_j,P],
partial_rs P' = i[K_j,partial_rs P]
               +i[partial_r K_j,partial_s P]+i[partial_s K_j,partial_r P]
               +i[partial_rs K_j,P].
```

The transported coefficient has the corresponding first/second variational equations. On every fixed finite graph the full fibre is finite, the source is smooth and bounded on the regular tube, and these ordered derivatives have finite bounds; for example `||dGamma(partial_r k)||<=NL ||partial_r k||`. This is a legitimate fixed-N majorant, not a uniform volume theorem. Alternatively carry the original constant P_B on the RIGHT of the exact propagated fibre and differentiate that full intertwiner; the same terms reappear. Its whole normal jet is

```
partial_r(C_M P)=C_(partial_r M)P+C_M partial_r P,
Delta(C_M P)=C_(Delta M)P+2 sum_r C_(partial_r M)partial_r P+C_M Delta P.
```

Thus the safe preparation bound on EACH of the two complete normal paths is

```
Eprep,j <= C1,j/sqrt(2Z_B) [T sup|y_c,j'|
              +2pi ||B_j|| sup|y_c,j|+chi T sqrt(Tr Pcov,j)]
            +chi T C2,j/sqrt(2Z_B).
```

C1,j,C2,j are the actual Frechet/Sylvester/colour-path/whole-projector derivative majorants just described, not the old point constants. Include the initial actual all64 ground norm AND electric-FORM transfer, scalar-well-to-complete-normal-vacuum excess/gap transfer, both source affine/nonlinear remainders, offBODY/smooth-extension defects and matching cutoff tails. The reviewed full-normal fixed-N estimates give soft/stiff variances O_N(Lambda^(-3/4))/O_N(Lambda^(-7/8)), fluctuation momentum payment O_N(Lambda^(-3/32)), centroid O_N(Lambda^(-7/16)), metric pc^2 payment O_N(Lambda^(-3/32)), scalar-to-vacuum norm O_N(Lambda^(-3/80)), and stiff mixed-cubic integrated O_N(T). These are asymptotic supplier statements with changed finite preparation constants, not numerical payments in the new receipt.

## 6. True all64 geometry and full-H residual before division

For true well theta and its actual inherited density rho_theta, let

```
D_theta,alpha = U_theta,+^source B_alpha + B_alpha U_theta,0^source,
R_theta,alpha beta = Tr rho_theta D_theta,alpha^dagger D_theta,beta,
Z_theta = Tr rho_theta s P_B,
K_theta = R_theta/(2Z_theta).
```

Use the invariant actual ground ensemble, including offset crossings and all64. The matrix error is

```
Egeo,B^2 = lambda_max[ (sum_theta p_theta Z_theta K_theta)
                                      /(sum_theta p_theta Z_theta) ].
```

**This is the complete matrix weighted before division.** A geometric maximum is legal; replacing the inherited weights with equal1/64, treating a small point residual as this full finite-B matrix, or inserting a second propagated Gram inverse is not. If individual well symmetry is not independently available, use the full actual initial G and `lambda_max(G^(-1/2) sum p_theta R_theta G^(-1/2))`; the aggregate invariant BODY supplier gives G=2(sum pZ)I. Nothing in the code assigns native p_theta or Z_theta.

There is an explicit conservative source comparison available without pretending to know rho_theta: set `delta_theta=max_x ||ell_theta,x-b_spin(x) S_theta,i||`, `r_theta=||ell_theta,R||`. Both are computed from the real full15V solve, using true ten-Grams. Whole exterior-sector unitary Duhamel gives

```
d_theta <= min(4, 2pi[(4V+1)delta_theta+r_theta]).
```

This is a charged-plus-neutral **unitary comparison**, not a normalized native marked residual. With the four-label preparation bound `||B||<=4` (|F|<=1, sum|eta|^2=1, ||P_B||=1), a safe marked bound is

```
R_theta <= 16 d_theta^2 I4,
Egeo,B^2 <= [sum p_theta 16 d_theta^2]/[2 sum p_theta Z_theta].
```

This deliberately loose fixed-N bound includes full neutral RIGHT, reference leakage and occupation motion. The exact matrix above is preferable. It exposes why the old point Egeo cannot simply be reused. The code prints d_theta, the global commutator and true Gram/offset; it does NOT print a claimed paid Egeo. Native geometric convergence needs a volume/preparation-weight argument or actual R_theta, not a conjectured harmless factorV. At fixed support the limiting-Gram mechanism suggests refinement, but the displayed global bound alone does not establish uniform-N smallness.

Both real legs use the SAME unreversed original H in H-omega_j J_j. On the actual same-ground purification the slow Duhamel residual contains

```
[Hph,C_M P] = -UE C_(Delta_A M)P -2i UE C_(grad_A M)P.E
              -UE[2 C_(grad_A M).grad_A P+C_M Delta_A P]
              -2i UE C_M grad_A P.E,
[HL,C_M P] = C_(hM)P+C_M[HL,P],
[Jorb,C_M P] = C_(oM)P+C_M[Jorb,P].
```

For the original P_B the A-derivative terms of P are zero; for the co-moving P(theta) they generally are not. The alternative constant-initial-projector representation does not delete these terms; it moves them into the evolving fibre and its photon jets. Compact magnetic energy commutes with a fibrewise multiplication mark, but its SAME full cosine force/radiation still appears in the slow unitary and ordered residual. The actual reference/connection and whole-heavy/common RIGHT are retained. Neutral electric momentum is not zero. Every original I0 first/second derivative is present. Native actual-ground form control, not a Gaussian trial, supplies the electric legs.

A sufficient error envelope on the SAME initially normalized map is

```
eps_B,N(Lambda) = Egeo,B,N
  + delta_ground,B,N(Lambda)/sqrt(2Z_B,N)
  + Enormal,1,B,N(Lambda)+Enormal,2,B,N(Lambda)
  + 2T Cslow,B,N(Lambda)/sqrt(2Z_B,N)
  + EoffBODY,B,N(Lambda).
```

Here Cslow is the full two-path residual norm after the exact product rules above; the preparation's moving-projector and current contacts are included, not independently asserted zero. Delta_ground uses the required electric FORM component for unbounded terms. At fixed admitted N, smooth compact sources/full heat jets and the actual-ground form supplier give finite constants, and the existing normal transfer supplies asymptotic payments. No uniform-N/separation bound or finite native numeric epsilon is supplied. Increasing N must control the true weighted Egeo, then Lambda pays fixed-N dynamical error; the printed global geometry upper alone is not that existence proof.

If eps<1 is actually supplied, define X=U+ B rho^(1/2)G^(-1/2), Y=B U0 rho^(1/2)G^(-1/2). Then X^dagger X=I, ||X+Y||<=eps, and the physical raw amplitude has negative real part for every nonzero complex quartet:

```
||X^dagger Y+I||<=eps,
Re(u^dagger X^dagger Y u)<=-(1-eps)||u||^2,
||Y^dagger Y-I||<=2eps+eps^2,
visibility_u >= 2(1-eps)/(2+2eps+eps^2)>0.
```

The after-history norm is reported, not removed. Full-H phase can be complex; exact minus and exact raw2Z belong to the source-only isometric theorem.

## 7. Complete parent-runnable discrimination and evidence boundary

From the scientific root, with the supplied project interpreter:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
  original_finite_packet_echo.py --heat 1.2
```

Defaults point to the real immutable CA86/6867/E1A4/90F7 archive and the parent's original native input. Actual helper definitions are AST-selected from complete fences; no supplier main/science/fallback is executed. Original input owner hashes, all64 cardinality and the zero-class isometric connection are checked. Native input here is explicitly the inherited a=.6 family; no CLI spacing relabel is allowed. Size must be odd N>=9; the default two-site packet, complete velocity graph and guard plateau fit without seam ambiguity. The proof covers arbitrary compact eta/B inside the plateau, but the numerical default exercises this specific nonzero complex two-site packet, not all possible preparations.

Specifically B has offsets (-2,1,1) and (-1,1,1) from x0, weights .6/.4 and relative phase .37. The first row lies in the velocity halo and the second in its support, both strictly inside the spin plateau. Every axis has a genuinely nonzero whole orbital row on this SAME packet; one row may be exactly isolated for an axis, and that isolated row is retained. Their actual x-directed edge is nonzero for the z-axis source. The complete-CAR bond comparison uses these SAME two sites, eta, evaluated full-heat values and Wilson paths, not a second ad hoc packet.

The source comparison uses isometric zero-class frame with genuine central theta123 links as an explicitly labelled BODY algebra/geometry comparison; that pair is not claimed to be the theta123 minimizing native well. Separate all64/all3 actual geometry outputs use each row's true ten-Gram, actual central toron links, scalar offset and full15V solve. No isotropic fit, equal-weight mixture or N9/N5 admission is made.

The complete photon ODE uses every3V link and all12V anchored clovers, independently compared with actual `primitive_flow`. Its generic compact input includes loop/toron components. The entire nonzero orbital block is propagated as a full scalar matrix, tensor the four colour rows, plus exactly isolated rows; the original full operator has zero edges to its complement, so this block decomposition omits no coupled row and is not a field/Fock cutoff. The original full4V orbital operator is also used to select a real nonzero edge for the separate complete-CAR comparison. Entire8-mode CAR has all9 occupation sectors (total256), neutral70 and charged56 times the real4 reference; all native whole-fibre statements are exterior-functor identities, not an assertion that this small comparison is native ground.

Same-assertion adverse controls include: a changing halo and noncommuting central-link perturbation rejected at the global commutator assertion for all3 axes; omitted second transport rejected at the ENTIRE propagator reversal assertion; halo/noncentral complete-CAR protocols rejected at the SAME finite-B full-RIGHT endpoint; frozen neutral RIGHT rejected at its propagated intertwiner; omitted `[Jorb,P_B]` and `[HL,P_B]` rejected at the FULL commutator assertions; deleted B_j rejected at each FULL normal RNC jet. The genuine intermediate P_B/subspace motion is reported. Raw2Z, raw minus cross, endpoint P_B and actual RIGHT are measured without postnormalization in the separate full-CAR comparison.

Full original graph-heat/I0 first and Laplacian jets are evaluated by the existing complete-image Haar quadrature, with omitted-image bounds, effective samples and independent scramble disagreement. A finite-difference first-quotient-jet assertion rejects deleted I0 reaction at the SAME assertion. **Site-Haar quadrature is uncertified and is not an actual-ground photon expectation, a norm capacity proof or a finite-H payment.** The exact source endpoint is independent of the particular F values, and its proof retains the full exact Iq/I0, not this finite quadrature as a new physical preparation law.

Worker verification: actual root Pyright-language-server definition/documentSymbol resolved the original scalar_orbital owner and primitive_flow; positive mounted Python AST query exercised the real scalar_orbital calls. The worker LSP device is unavailable and reported; direct stdio navigation is not represented as completed diagnostics. Syntax compilation/structural verification are non-scientific. Per assignment, **Main runs the composed consumer; no scientific run or runtime residual is preclaimed here**.

The new consumer's ordered_echo, normal_controller_jets and whole_occupation symbols were also resolved with the actual root language server. Full-module AST compilation was exercised without importing or running its scientific consumers.

## 8. What this changes and what remains

The new finite-width mechanism removes the point-row and orbital/occupation-endpoint obstruction on isometric source-only BODY by an actual guarded-support controller. It makes a physically discriminating prediction: arbitrary finite packet coefficients and endpoint conditioning return with an intrinsic minus, while intermediate finite packet/current/occupation motion is real; breaking the halo or noncommuting a link spoils the same endpoint. This is more than another penalty bound.

The concrete next physical decision is to use this guarded protocol as the finite-B preparation channel and compute the **true weighted** all64 source residual / marked current on the actual initial-ground density, rather than seek more point-only signs. The global upper printed by the consumer may be too loose and cannot stand in for that calculation. Actual finite-H epsilon, ground/electric-FORM constants, finite-energy normalized charge/reference bounds, the new finite-B actual excess/current and its moments after transport, thermodynamic capacity/phase, and membership in a genuine mobile charged branch remain unpaid. Initial finite-width coefficient support is not itself compact support of the interacting charge excess. This protocol deliberately reverses orbital transport at its endpoint, so it does not demonstrate translational mobility. Physical global SO3, photon/current/Compton and continuum/UV obligations remain independent. No isolated charged pole is required or assumed.
