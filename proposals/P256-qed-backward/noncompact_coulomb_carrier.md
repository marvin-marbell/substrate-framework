# A genuinely different noncompact Coulomb carrier

Issue222 remains ACTIVE. Governing campaign plan is `issue-222-continuing-backward-20260930`, assigned revision `a406dca7a4cc642318250c028a763daa96fa66a5999da5dae0c96e53910c5c91`, general template `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`. This is an additional branch, not a proof of the native compact law. The integration owner runs the accompanying consumer and owns receipts/world-model integration.

## 1. Material law and quantization

Replace ONLY the electromagnetic coordinate/restoring law by

\[
 A\in\mathbb R^{3V},\quad E=-i\partial_A,\qquad
 H_{EM}=U_E\|E\|^2+\frac K2\|C A\|^2,\qquad U_E,K>0.
\]

The coefficient convention is important: there is **no** factor1/2 multiplying the electric term. With `UE E²/2`, every electric formula below is halved. `C` is the actual lattice curl, `D` the site-edge incidence with `DC†=0`, and `L=DD†`. The microscopic light hops still contain `exp(iA_e)`, not their small-A expansion. Quantum postulates are the complete CAR, Schrödinger noncompact photon representation, the whole fixed-number matter sectors, selfadjoint lower-bounded Friedrichs forms and the complete local Gauss constraints. This is a specified quantum root, not emergent quantization from classical Euler equations.

Keep the native matter/heavy endpoint and full field potentials: `{m=.2,r=.12,v=.3,kappa=2/a}`, `NH=2V`, retained entire light fixed-NL CAR, all60V Cartesian fields, all3V SU4 links and15V colour Gauss equations, background2, complete static opposite charge plus **all four** anti-fundamental components at the same fixed R, all64 true centre Gram minima, weights and offsets. The slow-field branch uses `chi=Lambda^-1/2`, `u4=.3chi`, `J=.2Lambda^5/4`, `K4=.7Lambda^5/4`. Electromagnetic UE,K are independent fixed positive symbols; no comparison value chooses native couplers. Arbitrary complex quartets, all BODY orientations and field configurations remain in the microscopic form domain. The regular tube is only a supplier construction, never a permanent domain cut.

After U1 gauge reduction, both physical transverse coordinates and all harmonic coordinates remain. Quadratic curl restores only transverse nonzero-curl modes. Harmonic coordinates are free in the photon law and periodic in the matter hops; the full operator has Bloch bands, not an assumed normalizable finite-graph ground. A compact torus or Villain model would be a different law again.

## 2. Legal source and exact complete-reference norm

For x different from R set

\[
 q=\delta_x-\delta_R,\quad j=D^\dagger L^+q,\quad r=\|j\|^2=q^\dagger L^+q,
 \qquad F=\exp(i j\cdot A).
\]

`Dj=q`, `Cj=0`, and j is orthogonal to physical harmonic and transverse directions. On R-valued coordinates F is a legal globally defined unit-modulus multiplier. The fractional j generally makes F **illegal on the compact link torus**: `F(A+2pi e)=exp(2pi i j_e)F(A)`. Thus this construction really removes a compact normalized-capacity premise rather than asserting that a compact capacity changed its name.

Let V be a smooth unitary colour label, gauge-covariant from x to the static anti-fundamental reference at R and EM-neutral (no A dependence). It may depend on the entire field configuration and a colour Wilson transport; it is not silently frozen. Define the source into the charged-plus-reference fibre by

\[
 T=F B,\qquad B\psi=\sum_{b,a}|\bar b\rangle_R V_{ba}c^\dagger_{x,a}\psi.
\]

The anti-fundamental/transposition convention is selected so this contracts the fundamental creator; a reversed convention requires conjugating V, not dropping R. Completeness and unitarity give pointwise

\[
 T^\dagger T=B^\dagger B=\sum_a c_{x,a}c^\dagger_{x,a}=4-n_x=:w.
\]

For a complete C-symmetric half-neutral state, or its C/translation-averaged density matrix, `mean(n_x)=2`, hence `Tr rho T†T=2` EXACTLY. This works even when probability(n_x=2)=0. The equally weighted vacuum/full local mixture is C-invariant, has mean2 and norm2 but P2=0; imposing a P2 floor or an occupation filter is neither necessary nor justified. Averaging is a physical mixed trial; at least one pure component attains no greater average Rayleigh quotient. On a fixed-total neutral lattice, mixture of configurations with one region empty and its complement full supplies the same local adversary without changing NL.

The mean2 property of the actual infimum supplier must still be established or constructed by symmetry. No inference from arbitrary non-C states is made.

## 3. Exact electric commutator, including the neutral current

In the neutral full-Gauss domain `DE=(n-2)`, since the microscopic heavy species is EM-neutral. Therefore the retained operator identity is

\[
 b:=j\cdot E=q^\dagger L^+(n-2).
\]

There is **no** replacement of the neutral photon current b by zero as an operator. `E F=F(E+j)` and V has no EM derivative, so

\[
 T^\dagger[H_{EM},T]=U_E w(2b+r).
\]

Quadratic curl commutes with T exactly; this does not discard its neutral vacuum energy. Since `mean(b)=0`, `mean(w)=2`, and `mean(wb)=-Cov(n_x,b)`, the normalized source commutator is

\[
 \boxed{\frac{\langle T^\dagger[H_{EM},T]\rangle}{2}
 =U_E\,[r-\operatorname{Cov}(n_x,q^\dagger L^+(n-2))].}
\]

This negative cross is the entire point of the construction. It does not follow from independent photons or a Gaussian neutral matter state.

In a translation-invariant neutral density matrix define `delta n_k=sum_y exp(-ik.y)(n_y-2)` and `S_n(k)=V^-1 mean(delta n_k†delta n_k)`. Equal-time densities commute, so translation invariance makes S real, even and nonnegative. Expanding `L+` in Fourier modes, excluding only its constant kernel, gives

\[
 \operatorname{Cov}(n_x,b)
 =\frac1V\sum_{k\ne0}\frac{S_n(k)(1-\cos(k\cdot(x-R)))}{\lambda_k}\ge0,
 \quad \lambda_k=\sum_{i=1}^3(2-2\cos k_i).
\]

Thus the interacting full-Gauss electric commutator is at most UE r, with no density floor, gap, occupation Gaussian, photon Gaussian, HC or S1 source premise.

### Uniform 3D fractional-gradient bound

For a cubic N torus,

\[
 r=\frac1{N^3}\sum_{k\ne0}\frac{2(1-\cos(k\cdot(x-R)))}{\lambda_k}.
\]

Choose centred integer modes m with `|m_i|<=N/2`. The elementary sine chord inequality yields `lambda>=16|m|²/N²`. The sup-norm shell of radius s contains at most `24s²+2<=26s²` modes and `|m|²>=s²`; the numerator is at most4. Consequently

\[
 r\le\frac1{4N}\sum_{m\ne0}|m|^{-2}
 \le\frac{26\lfloor N/2\rfloor}{4N}\le\frac{13}4.
\]

The constant is deliberately coarse, independent of separation and N. The consumer evaluates the exact finite Fourier resistance, not this shell estimate alone. The result is fixed lattice-spacing a; rescaling physical electric units requires an explicit law conversion.

## 4. Full Rayleigh transfer is NOT photon energy alone

For any normalized neutral vector psi, write E0 for the neutral infimum and eta=(H0-E0)psi. Then

\[
 \frac{\langle T\psi,H_+T\psi\rangle}{2}-E_0
 =\frac{\langle\psi,w\eta\rangle}{2}
  +\frac{\langle T\psi,(H_+T-TH_0)\psi\rangle}{2}.
\]

A spectral approximate-infimum vector supported in `[E0,E0+epsilon]` has `||eta||<=epsilon`; since `0<=w<=4`, the first term has absolute value at most2epsilon. Such vectors exist by the spectral theorem even without a ground eigenvector; averaging their density matrix over C, translations and the required T symmetry preserves energy support provided these are actual symmetries of the selected law. Do not assume time reversal of a chosen asymmetric background.

For the electromagnetic energy *alone*, there is an additional occupation-reweighting term `mean(w H_EM)/2-mean(H_EM)`. It is generally nonzero. Only the full neutral approximate-eigenstate gives the full weighted cancellation. The consumer reports the photon-alone difference separately and directly compares full Rayleigh transfer with the electric plus light commutators.

For the light Hamiltonian, retain the exact expression

\[
 L_T=\frac12\langle B^\dagger(H_{L,+}B-BH_{L,0})\rangle.
\]

At a fixed-field effective well V changes only the external reference labels; it commutes with the light Hamiltonian and cancels after summing the complete reference. For `H_L=dGamma(h)`, `[H_L,c†_a]=sum_i h_ia c†_i` and the numerator is `sum_a,i h_ia mean(c_a c†_i)`. This is not set to zero: it contains onsite and all endpoint hopping contributions, including the actual neutral light current. A onebody row bound with finite native endpoint degree gives a constant M_L(a) uniformly in N for bounded native Gram fields, without a light spectral-gap assumption. Arbitrary full fields are unbounded and require the BO moment/exterior form supplier rather than this frozen-row bound.

### Field, colour and Wilson-reference derivatives

With each real/body or group derivative D_I, let `Gamma_I=B†D_I B` and `Z_I=(D_I B)†D_I B`. Since w is independent of field/link coordinates, differentiating `B†B=w` retains the anti-Hermitian connection identity; the kinetic commutator includes both derivative-square and signed `Gamma_I D_I psi` terms. In quadratic-form notation for kinetic coefficient k_I,

\[
 \Delta_I=\frac{k_I}{2}
 \left[\langle\psi,Z_I\psi\rangle
 +2\Re\langle(D_I B)\psi,B D_I\psi\rangle\right]
\]

when k_I multiplies `D_I²` (the denominator2 is source norm). The displayed expression is the integrated commutator; derivative/divergence/density terms are absorbed by the actual covariant form measure, not omitted. In a nonconstant body cometric use its full mixed form instead of adding diagonal coefficients. Colour Wilson path derivatives, full60V field derivatives, heavy Berry/normal Born-Huang, light/reference connections, Casimir/reference-charge terms and all15V Gauss directions stay in these supplier bounds. A path of growing length has growing derivative constants; no uniform separated result follows by ignoring them. Freezing V with arbitrary BODY would violate Gauss covariance.

## 5. Supplier-to-microscopic upper transfer and its precise debt

The existing owner `original_slow_kinetic_bo.md` §§3–7 provides the conditional full-domain slow-normal construction at fixed a,N and regular sufficiently-large-N native minima: full heavy Slaterline, all normal directions, full-Gauss isometries W, norm residuals d0,d+, and complementary lower bound g. Its §5 explicitly uses W independent of A. Therefore replacing the compact magnetic multiplication by the unbounded positive quadratic-curl multiplication still intertwines the **entire** photon form: `H_EM W=W H_EM` on the appropriate form/operator domains, not just a low-energy photon window. The fixed-frame matter matrix remains bounded uniformly in all real A because its phases have modulus1. The exterior proof still retains positive photon energy. This is the actual reusable part of that supplier, not an assertion that its compact determinant lower-gap theorem automatically transfers.

Operator-domain extension for the unbounded quadratic multiplier needs the W photon-form isometry/domain argument; W's A independence and bounded normal residual supply it. The full magnetic zero set is different, but the colour/heavy normal wells were not selected by an EM-dependent heavy sea. There is no new EM-dependent heavy projector.

If the effective neutral infimum supplier is C/translation symmetric and M_L(a) is uniformly bounded over **all64 retained native centre Grams**, the effective bound is

\[
 E_{eff,+}(x,R)-E_{eff,0}\le \frac{13}4 U_E+M_L(a)
\]

by sending spectral approximate-infimum and harmonic packet errors to zero. No source capacity appears. Retain offsets: use the same minimizing neutral well or infimum mixture and its corresponding charged trial; do not choose a different charged zero-offset well or erase weights.

For finite approximants, retain `2epsilon`, harmonic Bloch packet energy, reference derivative/current bound and BO errors. A microscopic spectral/block transfer gives

\[
 E_{micro,+}-E_{micro,0}
 \le \frac{13}4U_E+M_L(a)+B_{ref}+C_{ref}
       +d_+ +3d_0+\epsilon_{packet}+2\epsilon,
\]

once g0 is above the relevant effective energy window. The neutral lower bound here uses the complementary gap and `3d0`, not merely the existence of a neutral trial. The Python `supplier_upper` accepts the already-combined neutral and charged error bounds; callers must pass `3d0` and `d+`, not confuse them with plain d0,d+.

At each N, the slow-normal suppliers have finite N-dependent derivative constants tending to zero with Lambda. Along an existence-only trajectory choose finite Lambda(N) large enough to make `d++3d0`, Wilson/field reference costs and current errors less than a fixed chosen tolerance, while retaining every reference path and all64 wells. Also widen the harmonic approximate-infimum packet so its **retained** kinetic excess goes to zero. This establishes a conditional uniform *upper compensated energy* at fixed a along legal Lambda(N), not fixed-Lambda uniform-N control. Need `N>=max(5,N0(a))` and regular-native supplier thresholds; N5 is diagnostic, never admission.

**Still unexplained/conditional:** actual noncompact neutral C/translation-symmetric infimum supplier for the native coupled law including all centre offsets; uniform effective local row/moment bound under its arbitrary BODY and native Gram family; revision-correct full-domain BO estimates for this changed photon form; quantitative N-dependent reference-current bounds or replacing the explicit Wilson source by the full-Gauss W source with an exact norm identity. The arbitrary full-field distribution is not localized from mean2; localization follows only from the heavy scalar normal/exterior supplier. We have removed an actual HC/P2 normalized-capacity premise but not solved these suppliers by assertion. The compact charged **lower** theorem is not imported. No mobile band, isolated pole, source-independent electron, Lorentz invariance or Compton amplitude follows from the upper bound.

## 6. Complete additional actual quantum comparator

`noncompact_coulomb_carrier.py` is runnable with the project interpreter. It implements the whole8-mode canonical neutral sector (dimension70) and whole charged sector (dimension56), with all4 static colour-reference components (56x4), no occupation filter. This is a light/photon discriminator, **not** the numerical full-heavy/full-field/all15Gauss consumer.

The explicit two-site graph has3 parallel directed hops, `D=[[1,1,1],[-1,-1,-1]]`, curl `(1,-1,0)`, transverse coordinate `(1,-1,0)/sqrt2`, and retained harmonic `(1,1,-2)/sqrt6`. Choosing rankone curl is part of this diagnostic graph law, not pretending that an ordinary cubic cell has only one transverse mode. Then `j=(1,1,1)/3`, `r=1/3`, `b=(n0-2)/3`. The exact neutral Coulomb energy is `UE(n0-2)^2/3`; in the charged sector the opposite static reference at site1 makes rho1=-rho0. Consequently electric source transfer equals `UE[1-Var(n0)]/3` at mean2. This is a direct coupled interacting check of the negative covariance, not free-photon energy subtraction.

The quadratic transverse oscillator is `UE p_z²+K z²`, with the **true quadratic restoring form**; exp(i t_e z) uses exact Hermite-Galerkin displacement matrix elements evaluated by associated Laguerre polynomials, never exp(i truncated z) as a fake continuous phase. The harmonic coordinate has period `2pi sqrt6` only in the matter coefficients; y itself is R. Bloch labels m in a finite Fourier range represent its periodic cell, physical momentum `(m+theta)/sqrt6`; photon energy `UE(m+theta)^2/6` is kept explicitly. Three hopping shifts `(1,1,-2)` act on all entire occupations. The finite basis is a stated Galerkin approximation, not a fixative ground flag or a full Hilbert-space ground.

The CLI solves actual coupled sparse H in several Bloch fibres, residual-checks each canonical eigenpair, selects only the sampled lowest fibre (explicitly **not** a certified global infimum), applies the complete source, and checks exact norm/density/electric/full-Rayleigh assertions. It also reports photon-alone energy, which need not equal the commutator. The charged ground is not needed to prove this variational transfer; the actual full charged Hamiltonian acts on the complete source.

The selected neutral cell vector is also zero-embedded into a true larger basis with two extra oscillator modes and two extra Fourier labels at each edge. Its enlarged-H residual and oscillator/Fourier edge probabilities are reported independently of the tiny within-basis eigensolver residual. Neither is called a full-space residual certificate; a small eigensolver residual must not conceal photon truncation.

The independent light check is the exact stationary CAR identity `L_T=-mean(H_light)/4` for this hopping-only two-site law: the complete creator commutator is minus one directed-hop expectation divided by norm2; actual neutral number-current stationarity makes the directed expectation half the total. The current is measured, not erased. The packet consumer integrates actual harmonic fibre-H differences using normalized uniform Bloch density on a narrow interval; eight-node Legendre quadrature is exact for this quadratic momentum dependence. It reports the integrated linear current, measured excess and normalization1, then rejects deletion of that same measured excess.

For the analytic infimum, use the Bloch direct integral of the **full** operator. At a minimizing Bloch fibre, take its approximate ground cell vector and a normalized symmetric narrow distribution of nearby physical momenta. Hold that cell vector fixed. Expanding the actual harmonic kinetic quadratic gives a linear current term plus `UE delta p²`; integrate the symmetric momentum density to cancel the linear term, leaving exact `UE Var(p)` error. If the minimizing fibre itself only approaches the infimum, add its explicit spectral error. These normalized full-R wavepackets approach the real infimum without assuming any L2 ground or deleting a harmonic direction. C/T/translation symmetrize the density matrix of these packets before using norm2. The consumer exposes and retains the same wavepacket excess; it does not falsely certify its sampled theta as the actual minimizing theta.

Adverse actual consumers replace the quadratic oscillator potential by a periodic cosine, omit three reference components, omit the negative covariance, remove hopping, remove harmonic kinetic, or erase packet error. Each must reject the same norm/electric/full primitive-action or packet-energy assertion used by the correct branch. The C-invariant vacuum/full mixture separately rejects a P2 conclusion. Oscillator/Fourier cutoffs and theta samples remain numerical convergence debts, plainly reported.

## Concrete next decision

Select this **changed noncompact photon root** for the backward charged-upper branch because it supplies a globally legal unit-modulus fractional Coulomb source and the exact norm2/negative-covariance mechanism. Do not transplant it into the native cosine model. The next physical decision is whether the full native matter/colour/normal supplier actually produces the required symmetry-preserving neutral infimum and uniform local commutator bound for this root; use the parent's independently coupled comparator and full supplier estimates to decide that, not an HC map or a sampled harmonic ground flag.
