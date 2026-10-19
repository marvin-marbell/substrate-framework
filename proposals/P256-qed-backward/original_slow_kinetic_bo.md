# Original whole-heavy slow-kinetic low-energy theorem

This is a mathematical fixed-cutoff, sufficiently-large-finite-graph construction, with native source diagnostics. It does not assert a numerical infinite-dimensional ground calculation, native N5 globality, a thermodynamic photon phase or localized mechanical half-spin. The complete complement/domain argument was independently reviewed in `local://issue222-slow-kinetic-BO-derivation.md` revision85C6. The native normal theorem is separately reviewed in `original_normal_gap_bound.md`. Neither review is campaign completion.

Governing plan: `issue-222-continuing-backward-20260930`, historical assignment revision381993ecb362c8c67dbc334aebbcb6093b64a7b78111b5a9e785d4746b9bbe3b; pinned general template89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe; workcampaign, outcome/verification. Fundamental heavy/light CAR and compact gauge quantization remain microscopic assumptions; this is not an Euler derivation of quantum statistics.

## 1. Exact microscopic law and theorem

Fix a>0, barem=.2,r=.12,v=.3, kappa=2/a, and a connected periodic N^3 graph, V=N^3. Retain the full domains R^(60V), SU4^(3V), U1^(3V), original fixed-NH heavy CAR with NH=2V, and the entire fixed-NL light CAR sector. Neutral NL=2V. A charged NL=2V+q, q=±1, has an explicit static opposite SU4 reference (bar4/4) and opposite U1 charge at a specified site. No rotor/angular/Fock/normal cutoff defines the Hamiltonian. All15V colour Gauss equations and complete U1 Gauss/background-two equations are retained; total periodic charge is zero including the reference.

Use the actual unimproved endpoint average of `BulkQuantumLaw.onebody`, not its added27-flavour selector. Each Phi_mu is traceless Hermitian4x4. Its local Gram is G_mu,nu=Tr(Phi_mu Phi_nu)/4, quartic kappa||G||²/2. Heavy has Hamiltonian -Lambda dGamma(h_H), actual whole2V filling among4V states. Light coefficient is1 and couples to the same endpoint plus compact EM phases. The explicit **different slow bare trajectory** is

```
field kinetic = Lambda^(-1/2) p_Phi^2/2,
colour electric u4=.3 Lambda^(-1/2),
alignment J=.2 Lambda^(5/4), colour plaquette K4=.7 Lambda^(5/4).
```

It is not the former sqrt(Lambda) field law or fixed-u4 result. Pure photon electric/magnetic coefficients are positive and fixed at eacha,N. The full physical operator is the Friedrichs restriction of this gauge-invariant closed semibounded form. Quartic field growth, compact rotors and finite fermion fibres ensure its lower boundedness and closure.

**Theorem.** There exists an existence-only N0(a) such that at every fixed N>=max(5,N0(a)), and sufficientlylarge Lambda, one can construct an exact full-Gauss isometry W_Lambda from a direct sum of64 native light/reference–compact-EM physical spaces, with common scalar well energies E_theta(Lambda), E_min=minE_theta. Let

```
K_Lambda=H_Lambda-E_min,
H_eff,Lambda=directsum_theta[(E_theta-E_min)+H_EM+h_light,native(G_theta,Utheta,A)].
```

Every offset is retained; centre labels are not conserved finite-Lambda sectors. There are d_Lambda->0 and g_Lambda->infinity with

```
||K_Lambda W-W H_eff,Lambda|| <=d_Lambda,
(1-WW†)K_Lambda(1-WW†) >=g_Lambda(1-WW†),
d_Lambda <=C Lambda^(-1/16)+C Lambda^(-5/16)+poly(Lambda) exp(-c Lambda^(7/20)),
g_Lambda >=c Lambda^(1/4)-C Lambda^(7/40)-C.
```

The first residual extends boundedly over the entire effective Hilbert space, initially defined on D(H_EM). For fixed nonrealz,

```
||(K_Lambda-z)^(-1)-W(H_eff,Lambda-z)^(-1)W†||
 <=3d_Lambda/|Imz|²+1/(g_Lambda-|Rez|)
```

for sufficientlylargeLambda. Native light current/contact have corresponding bounded image differences tendingzero. This is generalized norm-resolvent approximation of a Lambda-dependent family, not a fixed-sector limit, uniform-a/N estimate or numerical quantum spectrum.

## 2. Regular flat minima: existence threshold, not N5 certification

Put V0=quartic+lowest original whole-heavy energy, V1=the positive alignment/colour magnetic form without its Lambda powers, U_Lambda=V0+Lambda^(1/4)V1. ZeroV1 implies covariantly uniform fields and flat commuting SU4 holonomies after tree gauge fixing; reducible and noncentral competitors are retained.

`bulk_flat_holonomy_bound.md` proves the actual whole-half trace-norm lower bound on every commuting flat class, by the continuum Gram functional plus nonnegative spectral slack and one explicit D_N=O(1/N) atfixeda. A Clifford identity toron trial gives the opposite finite-size upper bound. Quartic coercivity bounds the Gram of any flat minimizer. Thus along every bad sequence N->infinity, compactness gives a limiting Gram minimizer with zero continuum slack. Strict Gram convexity and full quadratic coefficient richness make that limit a full-rank Clifford orbit. Reducible tuples are closed and disjoint from this compact irreducible orbit, so for sufficientlylargeN every flat global minimizer is irreducible. Its commuting holonomies are consequently centre Z4, not a presumed identity.

For each of64centre triples theta, the exact finite shifted-grid Gram functional

```
F_Ntheta(G)=kappa ||G||²/2-2 average_k sqrt(x(k+theta/(Na))^T G x(k+theta/(Na)))
```

is stronglyconvex. Its minimizer is positive interior for sufficientlylargeN by convergence to the positive continuum minimizer. The metric-Clifford Phi_mu=sum_nu sqrt(G)_mu,nu Gamma_nu attains the whole-half trace bound and all slack equalities. At N>=5, degree2 Fourier polynomials have no grid aliasing for any shift. Pair-axis cos_i cos_j, sin_i sin_j, cos_i sin_j/sin_i cos_j isolate the x0², xi xj, x0xi/x0xj coefficients; axis-frequency2 isolates xi². Hence the ten x_mu x_nu functions span Sym4. Zero sampled slack forces every anticommutator2G_mu,nu I. All flat global minima are therefore regular metric-Clifford centre-class orbits, at some source-owned subset of64classes. Finite offdiagonal or anisotropic G entries are not erased.

This establishes an existence-only N0(a), not a computed numerical threshold. Native N5 all-centre normal positivity/Gram stationarity does not certify N5 belongs to the globality range. Which theta minimizes energy is not fixed in advance.

At an exact metric-Clifford Gram minimum, field gradients vanish. A uniform traceless colour-link variation gives the actual endpoint derivative dh={T,partial_kh}/2. Since {h,partial_kh}=partial_k(x^TGx)I and TrT=0, its projector contribution vanishes, as does the full periodic trace contribution. Translation symmetry gives zero gradient at every individual link. This is why no generic Lambda^(-1/4) scalar displacement is used; the earlier unsymmetrized-vertex proposal was incorrect.

Write e_theta=V F_Ntheta(G_theta), v*=min_theta e_theta<0. In flat-adapted local coordinates (z,xi), where z spans nearby uniform irreducible tuples modulo commonSU4 and xi the remaining normal bundle, dimensionz=45. Exact flatness gives

```
V1(z,0)=partial_xi V1(z,0)=0,
V1(z,xi)=xi^T B(z,xi)xi/2 >=b|xi|².
```

Uniform positive restoration and stationarity give V0(z,0)-e_theta>=c|z|², partial_xiV0(0,0)=0. Smoothness and Young's inequality yield throughout a fixed small tube

```
U_Lambda-e_theta >=c'|z|²+b'Lambda^(1/4)|xi|²
```

for sufficientlylargeLambda. No Lambda-independent full-Hessian convexity in arbitrary stiff coordinates is assumed. Every global U_Lambda minimizer is bounded and approaches a flatglobalminimum by compactness/V1->0. The strict local lower bound then makes it exactly a centre-class flatpoint. Thus minU_Lambda=v* for all sufficientlylargeLambda. Outside radius-epsilon cores around all64classes, U_Lambda-v*>=c epsilon²: regular tubes use this lower bound; compact exterior excludes flatglobalminima; large fields use coercivity.

## 3. Full physical normal positivity and quadratic budget

For traceless h0 with h0²=r²I, the exact sea-Cauchy slack second derivative is

```
B''=[Tr(dh++_0²)+Tr(dh--_0²)]/(2r),
```

where0 removes the two-dimensional band-block trace. Its zero condition is {h0,dh}=(Tr(h0dh)/2)I. Full quadratic richness makes this the differentiated Clifford relation. Subtract the10 metric deformations; remaining derivations of complexCl4=Mat4 are inner, with precisely15 traceless anti-Hermitian SU4 generators. Thus the60 uniform fields have35 strictly positive slack normals,10 Gram normals,15 gauge zero modes:45 physical uniform normals.

Positive alignment/plaquette zero modes are covariantly constant field variation and flat link variation. A harmonic colour Lie toron must commute with all four irreducible fields, hence iszero; other link variations are gauge exact. After all15V Gauss directions are removed, the stiff Hessian has kernel45 and90V-45 positive normals. No continuous stabilizer or toron mode is dropped.

Both cometrics carry chi=Lambda^(-1/2). At each Fourier momentum use the complete mass-horizontal basis Z of ker((sqrt(C0^-1)G_gauge)†), C0=diag(I60,.6I45). The actual full quadratic normal Hamiltonian uses LambdaH0+Lambda^(5/4)H1 and this metric, not guessed separated frequencies or a Euclidean complement offstationarity. `original_normal_gap_bound.md` proves an all-q H0 upperL from actual whole-half denominators/contact/elastic terms, and a Schur criterion with positive uniformalpha and stiffb. It gives smallest frequencycLambda^(1/4), stiff frequenciescLambda^(3/8), and total90V-real-mode zero-point budgetO(Lambda^(3/8)). Numerical alpha/b/onset estimates in that consumer are not interval certificates of exact minima.

## 4. Scalar well, adapted coordinates and exact vacuum subtraction

Near a regular orbit the physical quotient is smooth: global stabilizerZ4 is discrete and all15V Gauss tangents independent. Its metric is chi times a Lambda-independent smooth uniformlyelliptic cometric. The auxiliary scalar normal operator S_theta includes the actual projected heavyBerry line, normal BornHuang, quotient density and orderchi heavy-only angular terms. It is defined in a bodychart and need not itself be a physical heavy-only Gauss sector. Light/reference connections stay in the microscopic operator and are compared below; neither their means nor variances are zeroed.

Let Q_Lambda be the complete positive quadratic normal operator. On an epsilon=Lambda^(-1/5) core, exact flat-adapted V1=xi B(z,xi)xi/2 bounds every stiff Taylor remainder Cepsilon-relative to its stiff quadratic. This would fail in an arbitrary curved chart generating pure-soft stiff quartics. The V0 Taylor remainder is likewise relative to the full positive quadratic; metric/density/Berry terms give

```
(1-Cepsilon)Q-Cchi/epsilon <=S_theta-Lambda e_theta
 <=(1+Cepsilon)Q+Cchi/epsilon.
```

Mixed sigma*tau² potential and sigma*p_tau² kinetic terms can be O1 on a product Gaussian; they are not individually discarded. The form comparison discounts their relative effect, not an uncomputed second-order energy coefficient.

The core boundary barrier is cLambda epsilon²=cLambda^(3/5), above the O(Lambda^(3/8)) vacuum excess. Full-domain IMS costs chi/epsilon²=Lambda^(-1/10). Minmax with cutoff quadratic states and the exterior barrier gives a simple first Dirichlet state phi_theta in a fixed regular tube,

```
E_theta-Lambda e_theta=O(Lambda^(3/8)),
g_theta >=cLambda^(1/4)-CLambda^(7/40).
```

The errorepsilonTrQ=O(Lambda^(7/40)) is smaller than the smallest normal gap. Agmon action to the core boundary is at least cLambda^(3/4)epsilon²=cLambda^(7/20), so smoothing/cutoff errors are polynomial-times-exponential. The exact localized E_theta is subtracted; no claimed harmonic energy or second-order correction replaces it.

Positive kinetic and the vacuum excess give ||p phi||<=CLambda^(7/16). Weighted potential lower bounds give ||xi phi||<=CLambda^(-7/16), and the sufficient coarse ||z phi||<=CLambda^(-5/16). No sharp uniformLambda^(-3/8) moment is assumed. All wells/offsets remain; finite-Lambda tunnelling is only exponentially small, not forbidden by a Hamiltonian domain cut.

## 5. Full-Gauss isometry and entire-photon bounded residual

The actual original wholeNH pointwise heavy ground is a smooth rankone Slaterline Omega_H(y) near each orbit. At arbitrary nonuniform y it is the actual whole cluster, not two occupied states imposed per momentum. Its many-body excited gap is at leastDelta_theta>0 before the factorLambda. The projector and firsttwo derivatives are bounded atfixeda,N. A smooth Kato unitary trivializes its line/complement on the contractible normal tube.

Cut off phi_theta smoothly within2epsilon, equal1 withinepsilon, and normalize in the actual quotient density to obtain u_theta. Supports of the64regularcores are disjoint for largeLambda, and scalar/cutoff residuals are exponentially small. The fullcolourorbit section Omega_H tensor arbitrary raw light/reference vector descends along the complete SU4^V orbit: stabilizer character is i^(NH+NL) timesreferencecharacter, equal1 for neutral4V or charged±1 with opposite reference. It is constant on the ENTIRE fixed-NL Fock fibre, not a chosen Slater. Full orbit integration with its real measure yields exact isometries W_theta; disjoint cores give W†W=I exactly, without near-identity Gram inversion or angle restriction.

W preserves complete U1 Gauss. Heavy/normal factors are EM-neutral and independentA; colour transformations preserve every n_light(x); referencecharge remains. Thus both unbounded purephoton electric and bounded compactmagnetic operators intertwine exactly. H_eff is selfadjoint onD(H_EM), since fixed-frame light/reference fibres are finite and its multiplication matrix is bounded uniformlyallA. No light spectral gap or light adiabatic projection is used in this BO step.

In the bodychart, the physical kinetic operator is chi times smooth normal derivatives and bounded heavy/light/reference matrixconnections. Heavy mean/BornHuang are retained in S_theta; adding light/reference representation yields firstorderCchi p and boundedCchi terms. Acting on u_theta these have normO(Lambda^(-1/16))+O(chi). Off-heavy kinetic channels obey the same bounded derivative-projector estimate. Native light matrix/current/contact are Lipschitz in normal fields/links uniformly allcompactA, giving O(Lambda^(-5/16)) image differences. Therefore R=KW-WH_eff extends boundedly on the entire effective space with normd_Lambda as stated. Photon energy windows are unnecessary for this residual: no A-derivative hitsW. An image statement alone is insufficient; K=Heff directsum0 is the counterexample excluded next.

## 6. Full complementary lower bound, including unbounded exterior light

LetP=WW†, C=1-P. Choose a gauge-invariant IMS partition into slightly enlarged shrinkingcores, equal1 on eachu_theta support, and allremainingfullspace. Its derivativecost is atmostCchi/epsilon². Quotient charts are used only on regularcores; exterior uses the unreduced fullphysical form.

ForPsi inranC, its localheavyground component is orthogonal to u_theta almosteveryA and componentwise in the FULL light/reference fibre. The overlap is an admissible effectiveU1Gauss vector; W†Psi=0 excludes it, not merely one lightstate or one photon window. The overlap with the true scalarwell ground is exponentially small because u_theta and phi_theta differ exponentially.

After smooth heavy-line/complement trivialization, Cauchy-Schwarz controls all heavy-offband and light/reference connectioncross terms by epsilon times positive scalar-relative kinetic/normalform plusCchi/epsilon. Keep the potential at fullstrength. Only the nonnegative S_theta-Lambda e_theta is discounted; the negative classical Lambda e_theta baseline is NEVER multiplied by1-epsilon. Light is uniformlybounded on the compact core. The PH-ground lower bound is

```
(1-Cepsilon)[E_theta-Lambda e_theta+g_theta]
 -(E_min-Lambda e_theta)-Cchi/epsilon-C
 >=cLambda^(1/4)-CLambda^(7/40)-C.
```

Use E_theta>=E_min and epsilon times the O(Lambda^(3/8)) relative vacuum excess. Heavyexcited components instead have cLambda-CLambda^(3/8)-C. All ground-overlap losses are polynomial-times-exponential.

For the exterior, define rho²=sum_x,mu TrPhi_mu²/4,

```
Cx=sqrt((|m|+6|r|/a)²+3(v/a)²),
A4=kappa/(8V), B_H=2NH Cx, B_L=2NL Cx.
```

The actual nearest-axis endpoint row/Frobenius bound gives ||h_onebody||<=2Cx rho, V0>=A4rho⁴-B_Hrho and ||H_light||<=B_Lrho. For rho>=rho0=max(1,(2B_H/A4)^(1/3)), U_Lambda-v*>=A4rho⁴/2. Globally,

```
||H_light||<=C0+C1(U_Lambda-v*),
C0=B_Lrho0, C1=2B_L/A4.
```

This covers singular/reducible and unbounded fieldregions; a fictitious global constant light norm is not used. The full heavy matrix is bounded below by its actual pointwise lowest fixedNH energy, and photon/kinetic forms are positive. ExteriorK is consequently bounded below by

```
(Lambda-C1)(U_Lambda-v*)-(E_min-Lambda v*)-C0
 >=cLambda^(3/5)-CLambda^(3/8)-C,
```

above the normal gap. Cores, exterior and IMS give CKC>=g_Lambda C on the full complementary form domain. There is no permanent well-domain truncation.

## 7. Domains and generalized norm resolvent

W mapsD(H_eff) intoD(K) by its bounded residual. Forxi inD(K), eta inD(H_eff), duality gives

```
<H_eff eta,W†xi>=<eta,W†Kxi-R†xi>.
```

Selfadjointness implies W†xi belongsD(H_eff), henceP/C preserveD(K). OffdiagonalCKP=C R W† is bounded byd_Lambda. Subtracting the bounded offdiagonal gives the selfadjoint block operatorD_Lambda=W H_eff W† directsum K_C; its compressed complement has lowerg_Lambda and theP block error is atmostd_Lambda. Thus ||K-D_Lambda||<=3d_Lambda, giving the stated resolvent inequality by the resolventidentity and ||(K_C-z)^-1||<=1/(g_Lambda-|Rez|).

Every competing centre offset remains. No fixed effective-sector limit is inferred without separate offset limits. Bounded/smoothed spectral observables transfer; sharp projection endpoints require an actual isolated window, not an arbitrary discontinuous cutoff.

## 8. Charged thresholds and the native metric family

`original_uniform_charge_gap.md` independently proves the sharp native effective-form inequality E_eff,±-E_eff,0>=mu_min=m min_theta sqrt(s_theta), at **any fixed positive** compact electric/magnetic coefficients. It uses actual spatial Gram-dual beta coercivity, a genuinely pointwise Euclidean Green estimate, the strictly positive background-two Hamiltonian determinant, exact canonical/local Gauss projection and dense full-CAR/Wilson charged range. Every scalar offset is common to neutral/charged; all64wells remain. Its repaired actual-kernel validator and independent whole256CAR correlation regression are separately reviewed. The previous extensive Born-Huang sufficient bound in `original_compact_charge_bound.md` remains valid, but its weak electric choice is no longer required for this transfer.

Here the source-to-effective remainder is bounded separately in each full-Gauss sector and tendszero. Each effective charged ground infimum has a fixed finite upper witness: a local charged creator/hole with opposite reference, compact Wilson string and a smooth physical EM vector; at a zero-offset well its energy is fixeda,N finite. For largeLambda, the complementary g_Lambda is above that bound. Block comparison gives E_micro,±>=E_min+E_eff,±-3d_± and the transferred neutral trial gives E_micro,0<=E_min+E_eff,0+d_0. Consequently

```
E_micro,±-E_micro,0 >=mu_min-3d_±-d_0.
```

A strict positive finite-volume full-microscopic compensated charged-sector lower therefore holds for sufficientlylargeLambda at every fixed positive pair of electromagnetic coefficients. These remain source-defined charged-plus-static-reference sectors, not an isolated electron or reference-free thermodynamic state. This lower theorem does not determine a charged upper bound, photon phase or speed; none is inferred from the determinant or Gauss embedding.

### 8.1 The actual cutoff metric has a positive large-volume Schur floor

Fixa>0. Let F_infinity(G)=kappa||G||²/2-2 integral_BZ sqrt(x(k)^T G x(k)) with actual x0=m+(r/a)sum(1-cos k_j), xj=(v/a)sin k_j and kappa=2/a. Quartic coercivity gives a minimizer over PSD Sym4; strict Gram convexity makes it unique. Proper-cube covariance, including individual sign flips paired into proper rotations, makes this unique G_infinity=diag(g0,gs,gs,gs). At G=0 a positive ray has a negative sqrt(t) leading term, so zero is not minimal. If g0=0,gs>0, adding positiveg0 has strictly negative directional derivative on any region away from the sine zeros, since x0>=m>0; if gs=0,g0>0, adding positivegs has derivative -integral(sum xj²)/(sqrt(g0)x0)<0, sincev!=0. Thus g0>0 andgs>0 without a floating positivity assumption.

Coercivity bounds all native centre minima in one fixed compact Gram set. The source-owned periodic Riemann estimate is uniform over its64centre shifts: sup|F_Ntheta-F_infinity|<=D_N(a)->0. Let eta_N=2sqrt(D_N/kappa). Strong convexity and the two comparison errors give ||G_Ntheta-G_infinity||_F<=eta_N, uniformlytheta. For eta_N<=gs/2, the actual spatial block is at leastgs/2 and

```
s_Ntheta >=g0-eta_N-2eta_N²/gs.
```

For eta_N<=min(gs/2,g0/4,sqrt(g0 gs/8)), this is at leastg0/2. Consequently the **native** family, not merely a fixed-metric surrogate, has

```
inf_(N>=N1(a),theta) mu_Ntheta >=m sqrt(g0/2)>0.
```

Combine with the existence-only globality rangeN0(a). At each N>=max(N0,N1,5), choose a finite sufficientlylargeLambda(N) so3d_±(N,Lambda)+d_0(N,Lambda)<m sqrt(g0/2)/2. The corresponding original full-domain trajectory has a uniform positive compensated lower at unchanged positive electromagnetic coefficients. This is an existence construction along an N-dependent heavy/alignment/slow-kinetic trajectory, **not** a uniform-BO estimate at fixedLambda, a numericalN0/N1 certificate, a reference-independent finite mass upper or a thermodynamic photon/deconfinement theorem.

## 9. Native diagnostic boundary and remaining campaign debt

`original_slow_kinetic_bo.py` exercises actual full4V heavy matrices and whole2V cluster derivatives in mass-horizontal uniform/nonuniform field+colour directions. It independently compares full Slater sea/contact/quartic curvature with the original Fourier105 H0, retains nonzero heavy colour means/variances, and checks actual nonlinear normal current/contact with arbitrary compact EM phases. Deterministic shrinking directions are not quantum vacuum samples. Incorrectly coupling the neutral heavy projector to EM changes its actual section and would invalidate exact photon intertwining.

The CLI does not numerically compute the infinite-dimensional scalarvacuum/isometry/resolvent, nor certify nativeN5 globality. The theorem is analytic at the stated existence-only sufficientlylargeN range. It does not solve thermodynamic Coulomb/Z4 phases, finite isolated-charge energy/reference removal, localized mechanical half-spin, radiative scattering, uniform Lorentz/cutoff control or UV completion. The original campaign remains open while those constructions are actionable.
