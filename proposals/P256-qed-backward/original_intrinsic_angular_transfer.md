# Original Cartesian intrinsic angular momentum: Gauss cloud and low-energy transfer

This construction advances the **mechanical field operator**, not just a binary internal cocycle. It does not prove a localized electron, a full microscopic continuous Noether generator, deconfinement/reference removal or a radiative scattering theory. Governing plan `issue-222-continuing-backward-20260930`, assignment revision54e3034987e0918a06b1734de362727e1c0aeff914790ef8f21e68b6c6024810, pinned general template89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe, workcampaign/outcome/verification. Quantum CAR and compact canonical quantization remain assumptions; no27-flavour selector is used.

## 1. Actual operator and precise theorem

Use the entire original60V Cartesian fields, SU4^(3V), compact EM links, whole NH=2V and fixed NL=2V+q CAR, with the actual opposite4/bar4 and U1 reference for q=±1. Work atfixeda and N>=max(5,N0(a)), in the explicit slow trajectory of `original_slow_kinetic_bo.md`: chi=Lambda^-1/2, u4=.3chi, J/K4 proportionalLambda5/4. All15V colour and complete U1 Gauss constraints remain. N0 is existence-only; nativeN5 is not globally certified.

Phi_i has the **physical vector index** verified by the native position/vector/oriented-link operation in `original_physical_rotation_lift.md`. In source field coordinates Phi_mu=sum_a phi_mu,a X_a, Tr(X_a X_b)/4=delta_ab, the intrinsic Cartesian angular component, for real fixed site weightsf, is

```
J_Phi,i(f)=sum_x,a f_x [phi_j,a(x) p_k,a(x)-phi_k,a(x) p_j,a(x)],
p=-i partial_phi, (j,k)=(2,3),(3,1),(1,2).
```

This is the actual coordinate angular differential operator of the three Phi vectors, not a matrix assigned to scalar matter. It commutes with all colour/U1 Gauss constraints because colour rotates the a-index and the physical operation rotates the i-index. Its usual selfadjoint coordinate-flow realization on the full field domain is retained. It is **not separately conserved** by the finite cubic Hamiltonian; there is no arbitrary continuous spatial-rotation symmetry on that graph.

At a native centreclassc, let G_c be the exact full10Gram and Phi^c its metric-Clifford representative. In the source mass coordinates, M=diag(I60,(1/.6)I45). For periodic momentumq define the actual full105-by15 gauge tangentG_c(q), and let V_i^c(q;f) be the actual Cartesian vector-index tangent `(deltaPhi_j=-f Phi_k, deltaPhi_k=f Phi_j, deltaU=0)`. Define the colour-orbit mechanical connection

```
ell_i^c(q;f)=[G_c(q)† M G_c(q)]^-1 G_c(q)† M V_i^c(q;f).
```

No generator or normalization is postulated before this operation. The actual field mass D_c†D_c is strictlypositive on all15 colour generators, by irreducibility; link mass also enters. Fourier transformell back to sites, and set the bounded entire-light/reference operator

```
S_Phi,i^c(f)=sum_x,a ell_i,x,a^c(f)[Q_L,a(x)+Q_ref,a(x)].
```

Q_ref is at its specified site, `-T_a*` for the bar4 reference and T_a for4; a charged sector without that reference is not introduced. Every64centre block and common scalar offset is retained.

Let K_Lambda, W_Lambda and H_eff,Lambda be the complete full-Gauss construction of the BO proof. For any real eta in C_c^infinity(R), the actual full-physical bounded low-energy angular observable obeys

```
||eta(K) J_Phi,i(f) eta(K)
 -W eta(H_eff) [directsum_c S_Phi,i^c(f)] eta(H_eff) W†||
 <=C_eta,f Lambda^(-3/40)+poly(Lambda) exp(-c Lambda^(7/20)).
```

Constants depend onfixeda,N,couplings,sector andf. No uniform-volume/cutoff estimate or numerical quantum normalvacuum is claimed. This is a smoothed spectral compression of **one mechanical component**, not a claim that exponentiating this component gives the full physical rotation. The proof below controls its unbounded normal momenta; norm-resolvent convergence alone would not suffice.

## 2. Field and colour-link geometry: screened mechanical density

At an isometric source minimum G=diag(g0,gs,gs,gs), recovering Gamma_mu=(sqrt(G)^-1)_mu,nu Phi_nu gives the actual Clifford frame. The physical field tangent satisfies

```
deltaPhi_j=-Phi_k, deltaPhi_k=Phi_j = i[S_i,Phi_(j,k)],
S_i=-i Gamma_j Gamma_k/2.
```

The coefficient1/2 is forced by `[Gamma_j Gamma_k,Gamma_j]=-2Gamma_k`; it is not assigned to raw matter. With T_a=X_a/sqrt8, s_i,a=2Tr(T_a S_i), one has D†D s_i=gs s_i. Actual link gauge tangents are `(1-exp(iq_axis))I15`. Therefore

```
ell_i(q;f)=s_i * gs/[gs+|1-exp(iq)|²/.6] * f_hat(q).
```

All15 generators are solved before selecting this bivector channel. If f=1, this is the entire light/reference spin generator. If f is a spatial window, it is the Yukawa-smoothed f, with cell screening length1/sqrt(.6gs). The missing part is the colour-link contribution required by complete Gauss; it cannot be removed by using a field-only or Euclidean slice. A local spin reading does not automatically exclude the reference: its coefficient is the smoothed value atR, which must be controlled at the actual separation.

For arbitrary full native G, the same matrix connection remains the result. Vector-index rotations need not equal a pure body bivector of standard normalisation; no isotropic replacement is made. Their RHS lives in gradeone, so the solution lies in the six Clifford bivectors: the other nine SU4 generators have orthogonal derivative grades. The reviewed metric-family theorem gives G_Nc->diag(g0,gsI) uniformlyc atfixeda. Thus the standard screened expression is recovered in that family, not asserted exactly for a tilted/anisotropic finite nativeframe.

At the isometric rest band beta=Gamma0, the two-dimensional beta-positive fibre and the conjugate beta-negative hole fibre each carry sum_i S_i²=3/4. This identifies the rest **field doublet** of the mechanical image. It is not a claim that an arbitrary interacting charged ground or its photon cloud has totalspin1/2. Scalar raw matter remains scalar under microscopic position transport.

## 3. Whole-heavy point character, not an even-filling shortcut

Even NH alone proves only an ordinary representation. Here the actual vacuum character is computable. Fix ONE common chiral pairC+/C- of the recovered Gamma5=Gamma0Gamma1Gamma2Gamma3 for allclasses. At each centre, actual h is chiral, invertible and has2V positive modes. Write Q=C+†h C-, and letU be its unitary polar factor. The actual complete positive-heavy occupied frame is

```
Z_H(h)=(C+ +C- U†)/sqrt2.
```

Its projector is the full pointwise2V ground cluster of -dGamma(h), not per-cell or per-Bloch occupations imposed by hand. For a physically restored native cube endpoint E_R, chirality is preserved. Exact covariance gives

```
E_R Z_H(h_c)=Z_H(h_Rc) E_+,
E_+=C+†E_R C+.
```

Consequently its occupied **Slater vector** transforms bydetE_+. This determinant, not a density expectation, owns the phase. Native E_R=diag(i^k_R) S_R† P_R. Its two-dimensional local chiral spin block has determinantone, and the site permutation contributesdet(P_R)^2=1. The remaining phase is `(-1)^(sum_x k_R(x))`.

For a reversed sourceaxis with centrelabele, the rooted seam cochain is `-e 1_(y_targetaxis!=0)` modulo4; unreversed/permuted axes need none. Each summand has total `-e N²(N-1)`, which is even for every integerN. Adding them modulo4 does not change parity. Hence detE_+=1 for all64classes and24properrotations. This uses the actual seam, and is not a claim that the entangled Slater state has exactlytwo particles at eachsite.

A practical phase error was discriminated before this construction: independently diagonalizing the degenerate chiral operator at source and target gave false whole-Slater phase-minus observations, while its positive-cluster projector stayed correct. A common anchored chiral fibre removed them; the actual phase residuals became4.34e-14 or less at a1/.001. Casewise basis rotations are not physical vacuum characters. The producer requires a shared chiral convention.

## 4. Auxiliary well character and bounded mechanical image

Choose the normal chart and cutoff tubes equivariantly under the finite proper-cube group, using the actual quotient metric. The full heavy Kato line can be chosen along these equivariant radial paths; its point characters areone by section3. Source symmetry transports S_c and its positive quadratic Q_c to the targetclasses. For a class stabilizer the normalized real quadratic vacuum and symmetric cutoff trial have trivialcharacter.

The BO form comparison gives trial Rayleigh excess over the actual simple scalarwell ground atmostC Lambda7/40, while its firstgap is at leastc Lambda1/4. Its overlap with that trial is therefore nonzero for sufficientlylargeLambda. An eigenvector with nontrivial stabilizercharacter would be orthogonal to the trivial trial. Thus the actual simple auxiliary wellground has trivialstabilizercharacter. One can choose allclass phases consistently under24properrotations; the full W intertwines the cube action with the whole light/actual-reference and EM action. This is an analytic eigenline result, not a numerical vacuum computation or arbitrary continuous Noether embedding.

In a mass-horizontal bodychart the exact physical Cartesian operator is

```
J_Phi=-i[X(y) partial_y+div_J X/2]
       +ell(y)·[Q_H+Q_L+Q_ref].
```

The firstterm includes the actual quotient density, and is selfadjoint in that measure. X andell are smooth bounded on regularcores, with ell(y)-ell(0)=O(|y|). It leavesA and the unbounded pure photon Hamiltonian unchanged.

At the starpoint ell(0) is a Clifford bivector. The actual flat full-heavy local density is I/2 plus a Clifford-vector combination, so its contraction with any suchbivector iszero. This does **not** remove nonzero heavy means in the other colour generators, any charge variance, Berry derivatives or angular kinetic energy.

Kato radial gauge makes the normal Berry form zero at the starpoint. The mechanical normal–colour cometric cross is alsozero there, because the normal tangent is exactly mass-horizontal. Hence the auxiliary scalar normal connection is O(|y|), even though heavy angular means in otherchannels need not vanish.

To bound the scalar angular mean, compare S_c with its real scalar counterpart retaining its actual potential, metric, density and real BornHuang/angularterms. Its ground is realpositive. A selfadjoint real-coefficient firstorder flow has zero expectation in that realstate. On the shrinking supported core |y|<=2epsilon, epsilon=Lambda^-1/5, the magnetic connection perturbation satisfies

```
||R_mag u_real|| <=C chi(epsilon ||p u_real||+1)
                 <=C Lambda^(-21/80).
```

Here ||p u_real||<=C Lambda7/16 follows from the full zero-point budget. No illicit product of separate y- and p-moments is used. The momentum-resolvent estimate proved below gives a scalar angular differenceO(Lambda^-3/40). Heavymean/Berry contributions are O(||y u||)<=C Lambda^-5/16. Thus

```
W†J_Phi W =directsum_c S_Phi^c(f)+O(Lambda^-3/40).
```

This image alone is not the full spectral statement; the next two sections control the last unbounded boundary.

## 5. Why the existing norm BO estimate is insufficient

On the entire realnormal axis take

```
H=chi(p-aS)^2/2+Lambda^(5/4)x²/2, a=.7,S=.5.
```

Its exactground is e^(iaSx)u0. Withomega=Lambda3/8, variancex=chi/(2omega), its normdistance fromu0 is

```
sqrt[-2 expm1(-(aS)^2 variancex/2)] ->0,
```

but its canonical pmean=aS=.35, whereas the undressedu0 mean iszero. A small bounded Hamiltonian residual and normclose eigenstates cannot transfer an unbounded momentum. This counterexample must not be “repaired” by declaring its Berry connectionzero.

The actual source has a demonstrably different property: the entire15-generator mass-horizontal normal chart has C_yg(0)=0, and the Kato heavy normalBerry form hasA_H(0)=0. Thus a **light/reference ground-heavy normal** firstorder coefficient is O(y), not an arbitrary constant connection. The constant heavy derivative of its projector instead enters the excited-heavy channel whosegap is orderLambda. These are the two facts that improve the momentum comparison.

## 6. Graded residual and momentum resolvents

Use the fullBO scalar vacuum/isometry, no permanent normal-domain cut. Near each regularcore split the complement into heavy-ground scalar-normal excitations C_N and actual heavyexcited C_H. The already reviewed exterior barrier and fullphysical Agmon/IMS comparison control allremainingregions; cutoffs are auxiliary comparison maps only. Offtube sources fromW are polynomial-times-exp(-c Lambda7/20).

On the source u_cOmega_H tensor the ENTIRE light/reference/EM fibre, the heavy-ground normal residual has three contributions: native light deviationO(||y u||); light/reference firstorderconnection chiO(y)p; boundedchi angularterms. The scalarpart uses its exactE_c. Thus

```
r_N<=C[Lambda^-5/16+chi epsilon Lambda7/16+chi]
    <=C Lambda^-21/80.
```

The heavyexcited residual comes from the constant derivative of the actual projector and is boundedr_H<=Cchi Lambda7/16=C Lambda^-1/16. PureEM terms commute exactly withW; neither residual contains an unbounded Aderivative. No lightgap is assumed.

The fullcomplement lower gives g_N>=c Lambda1/4-CLambda7/40-C, and on excited-heavycores g_H>=cLambda-CLambda3/8-C. Positive kinetic and the exterior quartic/light domination give, for the corresponding resolvents at any fixed low complexz,

```
||p (K_N-z)^-1|| <=C chi^-1/2 sqrt[g_N^-1+E_ZP g_N^-2]
                  <=C Lambda3/16,
||p (K_H-z)^-1|| <=C chi^-1/2 sqrt[g_H^-1+E_ZP g_H^-2]
                  <=C Lambda^-1/4,
E_ZP=O(Lambda3/8).
```

This follows by applying the positive kineticform to the resolvent output and using its norm bound; the negative classical Lambdae_c baseline is never discounted. Photon positivity and the global bound ||H_light||<=C0+C1(U_Lambda-v*) cover the full domain. Matrix/light/reference covariant connections are bounded oncores; outside, fullphysical barrier localization supplies the weighted coordinate bound.

Heavy–normal offdiagonal kineticterms have form T=chi B1p+chi B0 with boundedcoefficients. Eliminating the excited-heavyblock discounts positive normal kinetic by atmostO(chi/Lambda)=O(Lambda^-3/2); its vacuum-budget cost is negligible relative tog_N. Its RHS backreaction satisfies

```
||T(K_H-z)^-1R_H|| <=C chi Lambda^-1/4 r_H
                    <=C Lambda^-13/16.
```

Normal sources remain dominated byr_N. Weighted resolvent errors are consequently

```
normal: Lambda3/16 r_N <=C Lambda^-3/40,
heavy: Lambda^-1/4 r_H <=C Lambda^-5/16.
```

Regular quotient metrics are uniformlyequivalent oncores. These estimates control the actual canonical normal momenta of low states, not just their norms. Replacing the scalar auxiliary magneticground by its real counterpart uses the same normalestimate withr_mag and proves section4's scalar meanbound. Heavy angular charges are finite matrices atfixedNH,NL,N and cause no additional unbounded direction.

## 7. Smoothed physical angular comparison

Apply resolvent block identities and smooth functional calculus toeta, with itsfixed compact low-energy support. The growing entirecomplement gap makes the C_N/C_H/outside parts ofeta(K) small in the preceding weighted momentum norms. The Pblock ofeta(K) differs from Weta(H_eff)W† by a bounded error tendingzero, and the exact P J_Phi P block is bounded by section4.

A naive bound using ||pW|| times a Pblock error could grow; it is not used. In cross terms put the selfadjoint firstorder operator on the small Ccomponent:

```
<Cpsi,J_Phi Wf>=<J_Phi Cpsi,Wf>.
```

Its bounded corecoefficients, bounded matter/reference angularmatrices and the weighted momentum estimates give O(Lambda^-3/40). Exterior coordinate/derivative terms are exponentiallysmall by the fullphysical barrier/Agmon estimate. The C–C term is controlled the sameway with an extra small normfactor. The P–P term uses the bounded compressedS_Phi, not an unbounded pW. Combining gives the section1 estimate. All centre offsets remain in the smooth filters; no empty class is selected as the charged ground.

## 8. Same-current gyro and the actual remaining spin boundary

At an isometric source minimum define mu=m sqrt(g0), ce=v sqrt(gs), r_eff=r sqrt(g0). The actual same-field endpoint law has long-wavebody

```
h=Gamma0[mu+(a r_eff/2)pi²]+ce Gamma_i pi_i+O(a²p³).
```

Its free Foldy positive-band operator gives kineticcoefficient(ce²+mu a r_eff)/(2mu), spinmagneticcoefficientce²/(2mu), with the spinSigma fixed by section2's mechanical image. Therefore its free same-current gyro is

```
g_current=2ce²/(ce²+mu a r_eff).
```

It approaches2 when that Wilson inertialdebt vanishes. This is not a radiative coupledmodel gyro or a physical photon normalisation prediction; those must use the native photon fieldleg and current. The producer reports this expression only at invariant isometric nativeclasses, not for generaltiltedGram.

The actual Cartesian intrinsic component now has a source-derived Gauss cloud, a whole-heavy/auxiliary character construction and a momentum-controlled low-physical spectral transfer. What remains is the **total** spatial orbital/vector/link/photon generator in a controlled spacetime limit, localization of the mobilecharged excitation with reference/boundary ownership, its finite separated energy in the compact Coulomb/Z4 phases, and the same scattering consumer. Compressing one component or observing its restdoublet does not complete those obligations.

## 9. Native consumer and evidence boundary

`original_intrinsic_angular_transfer.py` independently selects all64 original full10Gram frames and uses all15 field/link Gauss columns. Native cases000,100,123,222 retain general fullmatrices, actual wholeheavy4V/2V and the physicalPhi-index tangents. At000/222 it compares everyperiodicq against the screened analytic mechanicalkernel and rejects using a Euclideanmetric or omittingcolour-link flux. At a samplednonzeroq it checks the entire90-direction masshorizontal normalspace and the zero base mixedconnection. It computes allsix heavybivector angularmeans from the actual fulloccupied projector; othercolourmeans/variance are not declaredzero.

The source checks the common-chiral wholepositive heavySlater against its actual250-mode projector and physical quarterturn transport, plus all64-by24 pointcharacters and rooted seamparity. A nativeprototype using this shared convention exitedzero atN5/a1,.001: mechanicalconnection errors<=2.86e-15, Euclideanwrongdefects.0416..0657, omittedlinkfluxdefects about2.44; all64pointcharacters<=7.78e-13, actualwholeSlaterphasetransport<=4.34e-14, occupiedprojectors<=2.34e-13. These are floating native point/tangent observations, not intervalcertificates or infinite-dimensional quantumvacuum calculations.

The companion complete analytic proof—not finite diagnostics—supplies the scalar well, graded momentum and smoothed-spectral statements at the specified existence-largefiniteN scope. Final CLI/types and independent theorem review remain integration-owner evidence. No full localized mechanicalhalfspin or campaignclosure is asserted.
