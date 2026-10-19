# Original-field local bosonic statistics join

Governing plan `issue-222-synthesis-continuation-20261006`, statistics work item, pinned general template `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`. This is a constructive finite-penalty successor to the original bare law, not an assertion that the bare law was already microscopic bosonic. `bosonic_statistics.py` is the primitive algebra supplier; its occupation-sign implementation is used only in the comparison isometry. Original endpoint coefficients come from `BulkQuantumLaw.onebody` and `scalar_paths`, not imported Clifford/Dirac endpoint matrices.

## Local graph and exact colour action

At every periodic cubic site use independent four-mode heavy and light **trees**, plus one heavy/light anchor bridge. Connect nearest-neighbour anchors separately in each species. Attach an actual four-mode reference tree to a light anchor. The mode graph has `8V+4` vertices, `13V+4` edges and `5V+1` independent cycles; maximum degree 11 (the reference attachment is included). The code exposes only symbolic counts at `Nstar=5*2^96`; it never allocates that lattice or an exponential edge matrix.

The primitive operators are exactly the existing Pauli words `B_i`, oriented `A_ij`, `n_i=(1-B_i)/2`, and the existing path bilinear `T_ij=-i path_A(j,...,i) n_j(1-n_i)`. Onsite bilinears use unique internal-tree paths. Their complete even matrix algebra is exact in **every spatial-loop sector**. Using onsite K4 edges without fixing its internal loops would instead break SU4 Lie closure in extra sectors; that tempting implementation is rejected. Each local gauge generator is the onsite tree-even lift of an actual traceless colour generator, summed over H/L and the attached reference. Add the original adjoint Phi derivative and the left/right SU4 electric generators to obtain the full colour Gauss operator. EM Gauss retains divergence, light density, background-two and reference charge.

A spatial Pauli `Z_e` commutes with every onsite tree bilinear: those bilinears have X support only on internal edges, with at most Z tails on spatial edges. Thus a finite field-independent sum `D=sum epsilon_e Z_e` preserves **exact** SU4 gauge action, heavy/light number, reference charge and total parity, but anticommutes with cycles through e. This is genuine microscopic extra-sector dynamics, not an exact reencoding. Couplings are nonzero and finite. `||D||<=sum |epsilon_e|=beta` is a bounded defect estimate, not a bounded norm for the unbounded original Hamiltonian.

Use local elementary H/L spatial plaquette penalties and local rectangles made from a heavy edge, two onsite H-L bridges and a reversed light edge. Rectangles are necessary: spatial plaquettes alone miss interspecies cycles. The contracted cell complex is `T3 x interval`, hence there are **three**, not six, auxiliary statistical holonomies and eight flat auxiliary spin structures. These are independent of the **64 original SU4 colour-centre wells**, all of which remain in the original field law. The low family retains the full eight-sector auxiliary direct sum over the original colour histories. Native GF2 positive controls on N3/N4 give cycle/local-constraint ranks136/133 and321/318; omitting bridge rectangles falsely leaves32/69 holonomies. **No nonlocal winding-loop penalty is represented as local.** The prepared + source history is compared inside the retained family. Local defects may alter winding history through excited flux; their finite-time Schur remainder is bounded, not declared zero.

## Full original field Hamiltonian

Let Phi be arbitrary Cartesian traceless Hermitian four by four fields, not selected endpoint matrices. The original coefficient law is

`h[x,x]=onsite*Phi_0(x)`,

`h[x,y]=(F_x U_xy+U_xy F_y)/2`, `F_x=sum_mu scalar_axis,mu Phi_mu(x)`.

Light adds `exp(i A_xy)` with real noncompact A; heavy carries scale `-Lambda`. `OriginalBosonicLaw.local_matter_coefficients` lazily supplies every colour component and Hermitian-conjugate direction for one site, using field/link callbacks. Every coefficient multiplies a primitive Pauli path bilinear along that actual link. Light current and contact are the first and second derivatives of **that same** exponential and coefficient. There is no separate imposed current endpoint.

The full closed physical form is the sum of these local matter terms and the original slow trajectory:

- `Lambda^-1/2 p_Phi^2/2` on the complete `R^(60V)`;
- `.3 Lambda^-1/2 E4^2/2` on complete `SU4^(3V)`;
- `Lambda (2/a) sum ||G_x||_F^2/2`, `G_mu,nu=Tr(Phi_mu Phi_nu)/4`;
- `.2 Lambda^5/4 sum ||Phi_x-U Phi_y Udagger||_F^2/8`;
- `.7 Lambda^5/4 sum (4-ReTr U_face)`;
- the original positive real quadratic curl/coexact photon, Coulomb and harmonic electric forms (not sin magnetic energy and not a compact EM replacement);
- finite local loop penalties and D, plus finite gauge-invariant unwanted-reference penalties.

`full_form_contract` exports this specification. `field_potential_at` evaluates the original quartic/alignment normalization directly. The noncompact EM coefficients are owned by the original photon supplier and are not retuned by this statistics construction. The Friedrichs restriction to the complete joint Gauss domain defines the operator; the code isometry is independent of Phi and EM and exists solely for exact-code comparison/transfer. The original bare branch remains unchanged in its source files.

## Actual reference and source

Four reference tree modes are essential, not an odd boundary spectator. For a plus charged leg, the neutral reference is its filled n=4 singlet. Projected light-create/ref-annihilate maps it to n=3, whose actual SU4 representation is `wedge^3 4 = bar4`. Its charge is `n_ref-4=-1`. For a minus leg the neutral n=0 singlet maps to n=1 fundamental, charge `n_ref=+1`. The two branches are exchanged by the admitted original PH/Phi-parity transformation; reference offsets/penalties are paired accordingly. A finite occupation-polynomial penalty suppresses unwanted reference occupations while preserving colour and charge. It does not discard the full microscopic reference fibre.

The four marks are the four projected primitive bilinears. `reference_source` retains fixed quartet `[1,.3i,-.2+.4i,.6]` without source-column whitening or changed norm. Dressing inserts the actual field/SU4 source matrices between these indices. Scalar contraction is exactly colour invariant off code; generic fixed g alone is not gauge invariant. Actual wedge3 generators match `-T^T` simultaneously in one signed hole basis, not merely by eigenvalues. The antiunitary combines onsite paired path-A products, the **essential bipartite density phase**, and conjugation. Density complement alone is not the correct PH action on off-diagonal bilinears. The proper map sends `T_ij` to `delta_ij-T_ji`, maps plus source at g to minus source at conjugate g with one common sign, and preserves spatial Z defects and local cycle penalties. The actual original distinct-site field bond changes sign under this map; combined with global Phi parity it is invariant. All original onsite colour coefficients are traceless and spatial coefficients join distinct vertices, so there is no microscopic matter trace remainder. At Nstar the local cycles have length4 and winding length is divisible4; their admitted flat family is preserved. This does not assert an unrestricted harmonic ground: kappa0 or explicitly paired histories remain required.

## Unbounded quartic lower bound and finite-sector vacuum advantage

Set `r_x^2=sum_mu ||Phi_mu(x)||_F^2`. Since `Tr G_x=r_x^2/4` and `||G_x||_F^2>=(Tr G_x)^2/4`, the quartic is at least `c sum r_x^4`, with `c=Lambda*kappa/128`, `kappa=2/a`.

An encoded directed bilinear has norm at most one even in extra sectors. Four by four entrywise coefficient sums are at most four Frobenius norms. Hence every matter sector, without a CAR representation or full-H bound, obeys

`H_matter >= -C sum r_x`,

`C=(Lambda+1)[4|onsite|+8 sum_axis ||scalar_axis||_1]`.

The exact rational implementation conservatively includes both real and imaginary scalar parts. Young's inequality gives

`C r <= c r^4/2 + C^2/(2c) + c/8`,

so `H0 >= -V[C^2/(2c)+c/8] = L_N` after retaining all positive field/gauge/EM terms. This bound is `O_N(Lambda)` on the chosen original slow trajectory. It is uniform across all loop sectors and all unbounded Phi configurations. The actual code vacuum upper `U_N` must be obtained from an admissible full-Gauss trial or supplied vacuum packet; it is **not** inferred from a frozen finite matrix. The auxiliary vacuum advantage is bounded by `U_N-L_N`.

An explicit admissible upper is now supplied, rather than left to a frozen-matrix inference. In the original Cartesian coordinates `Tr(B_a B_b)=4 delta_ab`, take the normalized radial Gaussian `exp(-sum C_mu_a^2/2)`, constant normalized Haar wave on all SU4 links, and the original coexact photon Fock vacuum in the admitted kappa0 fibre. Put `(NH,NL)=(2,2)` at each site in its SU4 singlet and the neutral reference singlet. In the PLUS charged trial, the reference site instead carries `(H2,L3,ref3)` in the singlet of `wedge2(4) tensor bar4 tensor bar4`; MINUS uses `(H2,L1,ref1)` in the singlet of `wedge2(4) tensor4 tensor4`. Both have exactly zero total local EM rho, with the physical reference charge retained. Fixed local occupations make every spatial matter hopping expectation zero; radial field mean zero removes onsite matter expectation. Gaussian, Haar and matter singlets satisfy the FULL joint Gauss constraint, not merely matter parity. Choosing a locally flat auxiliary trial also gives zero local penalty and defect expectation.

Wick moments give `E||Gram||^2=300`, `E W_link=240`, field kinetic `15 V Lambda^-1/2`, alignment `18 V Lambda^5/4` and colour magnetic `(42/5)V Lambda^5/4`; Haar electric Casimir is zero. The original photon frequencies obey `omega<4c`, with exact unchanged `c=1130202988249777/9007199254740992`, so an admissible vacuum upper in either branch is

`U_trial=V[300 Lambda+(132/5)Lambda^5/4+15 Lambda^-1/2+4c]`.

This intentionally loose positive bound is NOT a claim that the selected ground is that Gaussian or that the physical source is co-located with the reference. It is used only for a variational upper. Primitive-Pauli simultaneous compression of all15 generators verifies the three trial singlets, with Gauss residuals below `3.44e-15`, normalized norms1, and correct Gaussian moments300/240. The full-sector lower is `-B_N`, with exact rational `B_N=a_N Lambda+b_N+c_N/Lambda`. `quartic_vacuum_coefficients_from_l1(V,kappa,C_H,C_L)` also supplies the bound for explicitly changed heavy/light hopping coefficients without borrowing bare-law values.

The final selected auxiliary gap is **not** blindly `2^131072`: `G=max(2^131072,B_N(Lambda)+U_trial(Lambda)+E+beta+1)`, and `Delta=B_N+U_trial+E+beta+G`. Consequently `Q(H-E_low)Q>=G` throughout `E_low<=actual_code_vacuum+E`. This dominates both the full-Phi `O_N Lambda` advantage and the explicit trial cost at the ACTUAL selected finite Lambda tower, however far its value exceeds the source-floor exponent131072. The analogous prepared/source-only selectors use their own dyadic floor in the same max. There is no no-advantage assumption, no whole-H operator norm and no giant Lambda/native-grid allocation.

Let P include the entire locally flat low space (all torons), Q=1-P, and E be the true selected low-energy window above the actual code vacuum. If

`g=Delta+L_N-U_N-E-beta >0`,

then `Q(H-Etotal)Q >= g Q`. All microscopic auxiliary sectors are retained. The coupling norm is at most beta because D alone changes cycles and original encoded even terms commute with them. Thus

`||Q psi|| <= beta/g ||P psi||`,

`||P D Q (Q(H-Etotal)Q)^-1 Q D P|| <= beta^2/g`.

These are genuine gap/Schur estimates on the unbounded field law. `finite_gap_budget` implements them using caller-supplied actual vacuum/window data. The code isometry does not manufacture that gap. At fixed nonreal z the usual block resolvent estimate adds `beta/g` image error and `beta^2/g` effective correction. A conservative finite-time amplitude bound includes four undressed input/output mixing costs, `4 beta/g+|T| beta^2/g`; source preparation and unbounded observable graph costs are additional, not hidden.

For genuinely bounded observables such as density/link phases the image error is bounded by their operator coefficients times `beta/g`. The actual original Phi-dependent current/contact and differential Noether are **not** bounded full-H operators. They require the weighted source/current graphs, and relative Noether uses the literal `Jc B-B Jn` Lie derivative before estimating. The primitive sin(curlA)=sinc(curlA)curlA observable is retained where the supplier uses it; it is not substituted for the real quadratic photon Hamiltonian. `weighted_transfer_constants` implements the bounded-observable contract only; the distinct quartic 3/4-power API controls the actual Phi/source-Lie images.

## Large-family quantitative selector and actual admission boundary

The actual comparison target is `epsilon=UE/(V*2^40)`, with `UE=Fraction.from_float(1.5429442570866347e-148)/2^4096` and `V=Nstar^3`. For any certified finite source/current/Noether graph constant `M<=2^m`, choose finite positive total beta=1, local spatial defect coefficient `1/(6V)`, and a finite penalty with excess gap

`g>=2^(m-floor(log2 epsilon)+16+ceil(log2(2+|T|)))`.

Then `2M beta/g < epsilon/2^14`, and the T6 Schur-time error is smaller still. The actual finite G and Delta use the explicit max/dominance construction above; the dyadic selector is only a LOWER FLOOR. This is a constructive nonzero local-dynamics family below the actual UE/V margin, not a generic small percentage of the body amplitude. It is deliberately expensive and introduces new finite penalty parameters; no naturalness or parameter closure is claimed.

The default selector pays only the actual bounded source-value contribution, with certified `m=2856` for multiplier `3.6186143592462693787e859`. At T6 its required gap FLOOR is `2^7798` and source-value error upper `2^-4941`, below target log floor -4923. This is not an unbounded Noether graph. The corrected selected normal packet states a source-value Lambda threshold180375 (the earlier contract180370 is superseded); its ACTUAL selected full finite Lambda is a much larger explicit tower and is retained unchanged. The full-Phi lower and admissible upper are paid in both actual-G dominance and Delta, never confused with a source-envelope floor.
 
The physical supplier now binds the actual selected normal receipt `dc7d6277001394ea4c7b96aa6d8874e889e50b830fea47d6a4ca22db8ebbafcf`, source jets `[4,427,850,1274,1698,2122,2547]`, `A4=1/(4V)`, CD and Hvac. The combined all-auxiliary quartic fourth-root coefficient is at most `2^199`; the actual positive source metric, source/current/Lie jets and retained prepared-Gamma electric boundary give combined graph upper `2^3972`. This controls the **relative** observable `Jc B-B Jn`, not a bounded norm of the differential J and not a neutral-variance deletion.

For `z=Q(H_Q-E)^-1 D P psi`, `||z||<=beta/G` and `<z,(H_Q-E)z><=beta^2/G`. The retained positive half-quartic and actual complementary shift give `||rho z||<=C beta/G^(3/4)`, not the false bounded-operator `G^-1` rate. `quartic_weighted_transfer_budget` uses beta1/16, nonzero local coefficient `1/(96V)`, minimum gap FLOOR `2^11892` and eight source/observable mixing costs. Its conservative dyadic weighted error is at most `2^-4948`, plus a much smaller ordinary T6 term; the exact target ratio is below `1.476e-8`. The physical supplier's tighter rational arithmetic gives `8.12958e-9`. Actual G and Delta are the stronger full-Phi/trial-cost construction above, so the floor-based estimate remains conservative. No giant allocation is needed.

The retained coexact EM orbital multiplier has the DIFFERENT energy-graph power `G^-1/2`, not `G^-3/4`. Original photon coercivity gives `C_A^2<=UE*Nstar^2/(8c^2)<1`, using curl `lambda_min>=16/Nstar^2`. Since actual G additionally dominates `U_trial>=Lambda^5/4` and the actual selected `log2 Lambda>=180375`, the EM graph uses the certified larger gap lower exponent225468. Even the conservative original source envelope `M=2^65536` then gives eight EM image costs at most `2^-47199`; these are included in the exact rational total below. No zero-frequency harmonic-oscillator coercivity is asserted: this is the coexact branch in the admitted kappa0/paired-history domain. The physical scalar-EM/coexact-orbital baseline remains a retained observable (supplier bound log2 -1482.87148), not a discarded probability error.

The m3972/G11892 result above is retained as a prepared-Gamma comparison, not promoted to an interacting-ground graph. The stronger **original microscopic source19** construction below pays its own state and source graphs. Neither calculation changes the two fixed joint B vectors, fullRef4, unwhitened input or global2x2 Loewdin map; the physical transfer pair owns their coupled consumer.

### Original microscopic ground, actual source19 and post-source photon boundary

The final original microscopic graph receipt `48b45ba330635169bde84020f16b52805faf995d2767cbc7c7121b83f4ae5fdc` distinguishes original Omega, response `U_g`, bare photon creation and **after-B normalized** charged input. Its underlying response is `U_g=-sum g_k/sqrt(omega_k) (Q0+omega_k)^-1 J_k Omega_original`, uniformly for any unit-l2 complex coexact g over all `2(V-1)` modes. It is not `a_dagger[g] Omega` and not a Gamma-response surrogate.

The receipt's RAW nativeN41 profile is not relabelled source19. PhysicalTransfer reconstructs source19 from its OWN q6 ceilings, actual19 support, tree word `D+2<=D+60` and scalar-orbital coefficient upper192. This gives angular1/2 logs1494/2184 and mixed two-current/two-angular log3187. The actual normal chart tube prefactor27868 remains multiplied by `Lambda^-1/5`; it is not erased or mistaken for a mixed graph. Combining these with positive metric2856 and actual q1 produces a conservative finite envelope `2^65536`. The selected original Lambda tower already pays its field/current/source-normal remainder.

The physical input is `X=B a_dagger[g] Omega_original`, with fixed norm lower `1/sqrt8` (a conservative .25 is used in contractions). Its **normalized after-B** photon half-energy graph is at most `2^600`. The bare-created bound150 and response-U bound950 are separate and are not substituted for600. The physical scalar-EM/coexact-orbital boundary remains an observable contribution, not a discarded probability error or a total-J half-spin claim.

`original_microscopic_loop_budget` selects finite nonzero beta1/16, local coefficient `1/(96V)` and minimum auxiliary-gap FLOOR `2^131072`. Its exported actual G is the max of that floor and the explicit full-Phi/trial-cost dominance term, with Delta paying `B_N+U_trial+E+beta+G`. Eight Phi-weighted source-image costs use `G^-3/4`, the separate coexact EM images use the paid `G^-1/2` estimate, and ordinary source/output/T6 costs use G^-1. The conservative FLOOR-based total target ratio has strict positive log2 upper -27846. The JSON emits null for its underflowing floating display **with the explicit log bound and positivity**, never fake zero. This is the large-family **loop** source-graph admission, not an assertion that a2^65536 source envelope dominates the full field Hamiltonian. PhysicalTransfer composes Noether, same current/contact and the fixed receiver. Fundamental quantum-law, parameters and changed propagation-branch debt remain explicit.

## Native execution and discriminating controls

Run `OPENBLAS_NUM_THREADS=1 /home/administrator/substrate-framework/.venv/bin/python bosonic_original_law_join.py`. The eight-edge native ring uses bulk and reference internal trees, a second spatial loop path and arbitrary full15 Cartesian Phi components. Independent quantum three-level field/colour and photon factors give dimension2304. The field-coordinate jet varies ALL four Phi matrices and the actual SU4 link `exp(i(.19+.09 theta) T)`; its potential is evaluated from the original quartic/alignment form. The real noncompact photon factor has its own electric/oscillator dynamics and cloud. The folded spatial-cell Galerkin retains the **actual endpoint coefficient** multiplied by its loop operator, so different auxiliary sectors genuinely have different field/current dynamics. These are explicitly native Galerkin restrictions, not cutoff definitions of the full law or a full Haar-gauge vacuum calculation. The reference is static apart from the source and finite unwanted-sector penalty.

The same Hamiltonian contains the endpoint exponential whose first/second derivatives are checked. Source preparation includes a nontrivial photon cloud and all four reference colours. Finite coupling .13, penalty31, time .7 produce nonzero loop leakage. **All105 generator pairs at BOTH onsite four-mode trees** close SU4 on the FULL auxiliary Hilbert (maximum residual `5.0242958677880805e-15`), not just the desired code. A two-distinct-site original-field Ward probe includes the alternate spatial loop, the actual nonzero defect/penalty and original projected photon phase: full-sector H/current/contact covariance residuals are `7.213136344911301e-14`, `5.011146222559555e-16`, `5.011146222559555e-16`, with NO code projection. Field/link covariance supplies the corresponding field/electric Gauss action. Gauge/charge defect commutators, simultaneous bar4 matrix comparison, source norm and path-transfer comparator are also checked. Missing reference colours, erased colour boundary, frozen current, wrong loop sector and undressed hopping discriminate. The existing primitive exchange witness returns -1 in the code and +1 for false-sector/undressed controls. No exchange phase is inferred from a supplied Dirac spin matrix.

The independent-field native run with exact projected photon phases observed loop leakage `6.875498319661058e-5`, full all-code-band gap `9.413685895096707`, Schur residual bound `.001795258540419612`, actual low-band spectral deviation `.0005490726448194039` and finite-time state deviation `.008300480231835964` within its gap-transfer bound. The same-link first derivative error was `1.0409452925253178e-8` and second derivative error `5.79498056014075e-8`. Local heavy/light coefficient emission separately agrees with the original endpoint rule to `7.757919228897728e-18`. Photon phases reuse exact `phase_galerkin` matrix elements, NOT exp(i compressed A); full p² compression retains the upper-level term and the photon oscillator has its actual projected spectrum. The finite projection's prepared input norm/tail and exp-compressed-A adversary are reported without source-column renormalization. These are finite native evidence, not constants substituted into the unbounded full-field theorem.

A coupled spatial exchange panel also evolves finite beta .13 and penalty31 with actual original Phi endpoint entries, independent quantum field/photon coordinates and the SAME neutral background history. Its primitive pair source is used for the two-charge exchange only; fullRef4 is the separate charged-source panel. The code exchange amplitude is `-.9980883595803892-.007197699553558038i`, while false loop sector gives `+.9980901606009985+.007197974769443168i`, over actual pulse duration291.5604588214709. This is genuine finite-dynamics exchange, not just the zero-defect -1 identity.

An additional PH boundary control matters: the folded single spatial-cell Galerkin can lose the **global** zero-trace cancellation and has trace remainder1.7626601453769728. It is explicitly NOT a full PH-ground witness. The actual distinct-site original bond has PH residual4.505474626922391e-16, source and loop PH residuals0, and the complete large law has no trace remainder because original spatial endpoints are distinct and onsite Phi is traceless. No field-dependent centering counterterm is inserted to hide this native projection limitation.

The JSON report includes source hashes and an input hash and exits naturally only if all controls pass. It is evidence of the finite coupled mechanism and exact local algebra, not a full unrestricted many-field ground computation or a physical-spin measurement. The source-bound physical transfer pair supplies that separate consumer boundary.

## Full native improved-law microscopic orbit consumer (2026-10-07)

`microscopic_native_orbit_lift.py` and `microscopic_native_orbit_consumer.py`
now instantiate the **full N3 graph**, rather than the folded eight-edge
comparison: 220 heavy/light/Ref modes, 355 edge qubits, 136 independent
cycles, 243 local plaquette/bridge constraints of rank133, and all eight flat
histories. Literal tensor-product Pauli words use the existing
`primitive_pauli_words`; their occupation/cycle stabilizers have rank355.
The exact compressed representation therefore specifies an actual code
isometry without allocating or independently evolving a `2^355` matrix.
Cached heavy54/light54/light58 orbitals are prepared with the encoded
quadratic generator `dGamma(log U)`, not a new sea diagonalization.

ABI: `NativePauliGraph(law,reference_site)`;
`path_terms(law,phi,links,a,species,first=None,second=None)` consumes
**ImprovedPhiLaw.terms directly**, retaining each ordered one/two-link colour
product and summed U1 phase. `lift_matrix(matrix,modes)` lifts arbitrary
heavy/light/Ref even matrices. Each `EvenTerm` contains its physical
coefficient, full graph path and explicit Pauli polynomial.
`decode(terms,history=(1,1,1),defect_edge=None)` evaluates the central-cycle
algebra; it does not fit CAR signs. Modes0:108 are heavy,108:216 light,
216:220 Ref. `source_terms(graph,S,T)` lifts the full108-by4 and4-by108
SOURCE marks, including the original Ref attachment. Physical Wilson marks
and statistical auxiliary paths are kept distinct. Photon creation is the
same original full52 field action, tensor identity on the edge qubits.

Run from this directory:

```sh
python microscopic_native_orbit_consumer.py --cache-npz CACHE --cache-manifest MANIFEST --output OBSERVATION
```

The sealed observation `local://issue222-fullgraph-microscopic-orbit-observation.json`
has SHA256 `8d936b2e4a8de3a97b8b94048d57e48812069b193f02821e9855b48044b5c5a4`.
It naturally exited0; scientific runtime104.844778647s, command107.22s.
Cache/manifest are the original `10d61adb...` / `90a6d7f3...` pair.
Supplier SHA256 is `dda1c13e64434e8492420729310266e760daefa42b220cfb640f723eb643b3ce`;
consumer SHA256 is `ea2ce94dee14c479549ffc9f3f3b364c4210ebd61c2317f6b4f5c211ce62e34e`.
Loaded source epochs were stable. Scoped Pyright and actual configured
Pyright-language-server diagnostics both returned zero errors; real
symbols/references and positive structural AST calls exercised the exports.
The OMP LSP device was unavailable; the installed server was driven directly.

Observed heavy/light primitive lift errors0, positive originalJ error
`9.69972e-20`, contact error `8.06791e-23`. Each profile exercised32 actual
SOURCE/current path pairs through literal Pauli-polynomial commutators,
with error0; every current polynomial's actual Z conjugation also agreed.
Original frozen FUNCTION errors were `5.32165e-16` (SOURCE19) and
`5.97019e-16` (SOFT), retaining their single globalZ. Actual relative
G-image errors were `6.23593e-17` / `6.93949e-28`; norms squared remained
`.1440429640808187` / `.07686598600070062`. The original three photon
creator images agreed within `8.63e-17`, with all52/55 inputs retained.

The defect is **not zero**:162 spatial `Z_e` terms, coefficient1/2592,
total beta1/16, distinct nonzero auxiliary syndromes, and
`||D P psi_unit||^2=2.411265432098765e-5`.
Actual `[D,B]` norms squared were `1.35610e-6` / `1.79840e-6`;
`[D,J]` on the actual SOURCE has norm squared `4.90351e-14` /
`4.87393e-14`. These are auxiliaryQ images, not PH or field-outside
failures. Current second products on the SOURCE retain PH before compression.
Dropping range-two paths changed the original current's action on the
unchanged neutral state by squared norm `1.86209e-9`. A wrong winding
history changed it by `1.16883e-7`; removing neutral RIGHT caused
SOURCE19 error `.06449404474322405`. Setting D to zero erases the measured
auxiliary current/source images and the following positive bound.

The usable controlled construction is a **bounded selected-state auxiliary
syndrome bundle**, not a claim to solve full interacting H. For
`H_aux=Delta sum_local(1-S)/2+D`, `tau=Delta*t=.1`, the actual first
Duhamel coefficient is
`a_e=epsilon*(exp(-i m_e tau)-1)/(m_e Delta)`, with native `m_e=5`.
The exact remainder is at most
`beta^2*t^2*||psi||/2`. Reporting `Delta*a_e` avoids silently underflowing
the nonzero dynamics. The same actual selector remains
`Delta=B_N+U_trial+window+beta+max(2^GAP_LOG2,B_N+U_trial+window+beta+1)`;
`GAP_LOG2=132096` is a **floor, not the chosen Delta**. No penalty20
or alternate state/filter is substituted. The observed rigorous bounds are
`Delta*||Q_aux psi(t)|| >= .0004590121213861545` (SOURCE19) and
`.0004579379554092435` (SOFT). Scaled current bundle norms squared
`Delta^2||J psi_first,Q||^2` are `2.41484e-14` / `2.41882e-14`;
their finite point-current error is also bounded, not erased.

The all-aux full-field quartic lower was renewed from the actual bare-heavy
and improved-light path coefficients: `H0>=-768240124841.8785` at the
unchanged native Lambda. The original Gaussian/Haar onsite-singlet vacuum
trial upper is `901258621760.7988`; it is **not the prepared FUNCTION**.
Original-state and driven SOURCE spectral-window/tail bindings remain unpaid,
so `fullfield_transfer_admitted=False`. Phi/current weighted control needs
the quartic gap^-3/4 graph, coexact coordinates need the distinct gap^-1/2
graph, and the free harmonic coordinates have no oscillator position
coercivity. This bounded auxiliary calculation does not establish full
field/matter propagation, an interacting charged pole, physical spatial
half-spin, scattering or spacetime closure. The campaign remains OPEN.

## Actual LEGAL driven auxiliary admission and original current transfer

The preceding native point receipt remains exactly as recorded. A subsequent
**LEGAL source-bound construction** now pays full-field driven auxiliary
state/effect transfer and the original Phi-linear current/contact images;
it does not infer those conclusions from the native point gap.
`microscopic_legal_auxiliary_admission.py` consumes the unchanged actual
filter receipt `9fdf6a3f...`, whole Noether/work receipt `13305584...`,
fixed-penalty correction `2d3a0176...`, and the fullgraph observation above.
It does not repeat their SCI, retune Delta, replace the initial state by a
Gaussian, change the filter/globalZ, or relabel the physical compact drive
as a static coordinate transformation.

The decisive step is to **retain the positive cycle penalty** dropped by the
original whole-field energy proof. All field derivatives and encoded EVEN
Noether/matter/current terms commute the field-independent all-eight-flat
projector P; only the actual D mixes P and Q. The original coefficient-l1
quartic and Noether Young absorptions never spend the positive penalty.
Consequently, with the original physical `Hdrive=H_R-Omega G`,

```text
Hdrive+Bhat Lambda V >= (Delta/2) Q_aux
    + Tphi/2 + Tcolour/2 + (5/8)H_EM + Lambda rho^4/(16V).
S(t)=<Hdrive>+Bhat Lambda V <= 8(C_E0+Bhat+1)Lambda V.
```

The native full-edge Pauli cycles are involutions, but inherited wider Root
vacancy/double sectors can have S_gamma=0. The LEGAL admission therefore
uses the conservative **Delta/2 minimum penalty**, not parameter Delta
as a gap. The original fixed selector itself is unchanged.

The original normalized neutral, both SOURCE maps/coherent quartet and true
incoming-creator states have the actual initial `C_E0=8` witness, including
the noncommuting beta/source contribution. The SAME fixed selector has
`Delta>=2U_trial+2beta >=(264/5)V Lambda^(5/4)+2beta`.
Hence `||Q_aux Psi(t)|| <= sqrt[(5/132)*8(C_E0+Bhat+1)] Lambda^(-1/8)`
through the actual physical clock. No instantaneous rotating-generator
global gap or massless-photon gap is invented.

Initial auxiliary Q is **not assumed zero**: RAWtau may be flat before the
original full-H filter acts. The proof comparator is
`v(t)=U_P(t,0)P Psi_original(0)`, without physical renormalization.
The exact projected evolution has forcing `P D Q Psi`, so
`||Psi(t)-v(t)|| <=(1+T beta) sup_s||Q_aux Psi(s)||`.
P projection is proof only; the original FUNCTION remains untouched.
Every bounded physical group/effect observation transfers with
`2||O||*||Psi-v||`. The separate first-FORM spectral comparator
`K=H0+B_N>=0`, `w=Lambda^(9/8)` gives initial tail norm
`sqrt(a_N+b_N+c_N+8V) Lambda^(-1/16)`;
`Emax=w-B_N` and `g>=Delta/2-w-beta>=Delta/4`;
the original trial has `U_trial>=2w+beta` for Lambda>=1.
This is not a newly applied physical filter or an inferred strong H norm.

The original positive J and bilinear contact now have a genuine auxiliary
**image** transfer as well. The retained quartic term supplies
`||rho^2 Psi||<=4V sqrt(C_E)` on the actual full driven state.
The proof code flow has its own same work bound with initial coefficient9:
projection removes at most bounded D cross terms, and the fixed selector
makes the Q non-D ordinary-energy block nonnegative.
Thus `||rho^2 v||<=4V sqrt(C_E_code)`. Cauchy then gives

```text
||rho(Psi-v)||^2 <= ||Psi-v||*(||rho^2Psi||+||rho^2v||),
```

an image error of order Lambda^(-1/16). This uses **two independent actual
quartic graphs**, not norm-closeness alone. Actual improved scalar paths
supply `||J_f Psi||<=16 sqrt(2)V Cpaths ||f|| rho-graph` and
`||C_fg Psi||<=32V Cpaths ||f||||g|| rho-graph`.
The canonical photon leg is the original `EMlegs*g/sqrt2`;
for a unit coexact packet its squared field norm is at most
`UE*N/(4c)` from `omega_min>=4c/N`. There is no guessed Dirac vertex,
new photon/volume scale or heavy EM contribution.

The bounded original SOURCE value is renewed independently: `psi1/psi0`
is a convex average of four unit-modulus centres; every screened path
factor has modulus at most1, original frames are unitary, and quaternion
matrices q/2 have HS orthogonality. With the unchanged raw quartet
norm-squared33/20 and inherited profile-l2 majorant33/32, both map
coefficient-l1 norms squared are bounded by
`64*(33/20)*(33/32)^2*V`, within the original full-aux CB bound.
Actual Pauli transfers have norm at most1 on P+Q. This pays
`||[D,B Gamma]||<=2beta||B Gamma||`, with the **original Gamma**
supported source domain retained. The separate rho-weighted defect/source
bound uses the actual whole Phi graph because rho commutes both D and the
SOURCE coefficient; differential source/guard jets are not promoted.

Final corrected consumer SHA256:
`bae32cf6a7e7b0cf6afcb852f4c460a27389a9a2ff7c73035c500384ea2ef880`.
Observation `local://issue222-LEGAL-driven-auxiliary-halfpenalty-admission.json`
SHA256 `306a7608bce161a4c83ae94baf5254185de5d17b048fd36f3e17d4ea2bcff1c2`
naturally exited0 in1.55s (proof arithmetic `.007719286s`).
The earlier `d7f7368d...` observation remains immutable as a narrower
pure-Pauli Delta-gap construction; it is superseded for inherited S_gamma=0
coverage by this half-penalty receipt, without any law/state/filter retune.
Its bounded unit-effect error log2 upper is approximately `-8.343699359e93`;
the original target `UE/(V*2^40)` has log2 floor `-4922`.
Original J/contact image log2 uppers are approximately `-4.171849680e93`,
also strictly below that target. Replacing actual Delta by its floor
2^132096 fails the evaluated energy/window admission; this failure does
not assert an observed large Q norm from a failed upper envelope.
Scoped Pyright and current-version actual LSP diagnostics returned zero;
positive AST queries exercised the true consumer.

Accordingly the NEW LEGAL receipt reports
`actual_driven_state_transfer_admitted=True`,
`fullfield_transfer_admitted=True` and
`original_J_contact_auxiliary_images_admitted=True` in precisely those
state/current scopes. Differential G, rG/guard mixed graphs, late
photon-weighted boundary/scattering and physical half-spin/pole/spacetime
closure remain separate obligations. No whole-particle or campaign closure
follows from this admission.

## Decision and debt

Select this explicit original-coefficient bosonic successor, with exact all-sector local SU4 action, actual reference embeddings, bounded nonzero auxiliary flux dynamics, uniform quartic vacuum lower bound and finite low/high gap. Retain all flat holonomies and bound mixing rather than using nonlocal penalties or false topological invariance. Reject onsite K4 gauge generators in unfixed local flux sectors, the full-Phi operator-norm shortcut, and a parity-only reference.

Remaining debt is precisely: fundamental complex tensor-product quantum law; chosen microscopic coefficients and very strong finite penalties; downstream full physical massive/photon/Noether/source receiver synthesis. The original bare law and original microscopic source19/photon boundaries are preserved. The physical transfer supplier constructs an explicitly changed range-two/near-Dirac propagation successor rather than falsely claiming exact bare-Wilson g=2; its legal-volume vacuum/phase proof is separate. The original microscopic loop allocation is quantified, not a planned future graph, but neither it nor finite code identity relabels internal j as physical half-spin or arbitrary total-J² as3/4. No campaign/issue closure, merge or deployment is authorized by this pair.

## NEW induced-law primitive/full-Phi transfer (2026-10-08)

`induced_root_transfer.py` is a new no-bare-curl branch, **not** a retag of
CURRENT427 or the old photon vacuum bound. `Ntotal=64` means all filled
negative Dirac bands, including the tagged active flavour; there are 63
spectator flavours, one active flavour, one EM-neutral original heavy
species and the actual four-mode Ref. Quantum tensor-product, Pauli,
canonical field postulates and microscopic coefficients remain inputs.

The primitive graph is `T3 × species-tree`, with a four-colour onsite tree
for every species, spatial anchor edges, nearest-species bridges and one
attached Ref tree. For `S=Ntotal+1` and `V=N^3`,
`vertices=4SV+4`, `edges=(7S-1)V+4`, and
`independent_cycles=(3S-1)V+1`. The native `N=3` execution has 7,024 modes,
12,262 edge qubits and 5,239 independent cycles. Actual binary Pauli ranks
give 5,236 independent local constraints and hence **three common
statistical torons** and eight flat histories; eight is derived anew,
not copied from the two-species model. Statistical Z2 histories are not
EM U1 flux spaces or SU4 centre wells. Omitting the bridge rectangles
leaves 1,856 additional unfixed cycles. An executed active-flavour cut
sheet preserves every species plaquette but violates 18 bridge
constraints and changes the physical current image by 8.10842 at `a=.25`.

H, current and contact come from the SAME full Cartesian-Phi range-one/
range-two endpoint coefficients, SU4 path products and Peierls phases.
No CAR endpoint Hamiltonian constructs them. Actual Pauli path reduction
at arbitrary independent SU4/EM links agrees with the direct full-Phi
matrix: H/current errors zero, contact error `8.75e-16`, and current
finite-derivative error `5.37e-10` in the `a=.25` witness. Onsite colour
tree even algebra is exact, including in nonflat spatial sectors.
Wrong cycle and different statistical-toron current defects are 6.22579
and 8.44160. Endpoint CAR is **not asserted on all Q**.

The two Coulomb-constrained induced photon modes are consumed with
`A_link=a A_phys` and `sqrt(Z/(2 omega))`. For the actual `UEphys=1`,
`Ntotal=64`, `a=.25`, order-four BZ witness, frequencies are
`.0095585831/.0095586357`, residues `.0405896410/.0405895660`, and the
actual charged code-Slater current-image norms are `1.30071/1.30513`.
The filled native negative doublet and tagged positive particle are
retained; this fixed-field native code state is not a full-field
Gauss-prepared valence sea. The local polarized current jet is not
claimed to be an exact `N=3` plane wave at the supplier's external q.
Dropping residue normalization changes the images by `.89835/.89956`.

An exact full bosonic cyclic-cell/real-photon Galerkin evolves the
Peierls Hamiltonian with **bare electric kinetic only**. Its one-link
coefficient is an actual component of the same Phi law. Current and
contact are derivatives of that coupled H (errors `5.07e-9/5.61e-8`).
The finite charge-preserving `Z_e` defect has `beta=1/16`, nonzero
`[D,H]` and `[D,J]` norms `.190205`. At time `.4`, gap 128 gives observed
Q amplitude `.00089291` and current-image error `.00054418`, below their
bounds `.00196533/.00172338`. Removing the penalty increases Q amplitude
to `.0249340`; commuting-statistics current differs by `1.24241`.

### Renewed unbounded-Phi and actual Gauss upper

With `r_x^2=sum_mu ||Phi_mu(x)||_F^2`,
`c0=(2/a)/128` and `K=Lambda h_H+Ntotal h_L`, actual path coefficient-l1
sums give the uniform all-statistical-sector form inequality

`H0 >= (c0 Lambda/2) sum_x r_x^4
       - V [K^2/(2 c0 Lambda)+K/4]`.

It follows from the Gram quartic and
`r <= r^2+1/4`, then Young's quadratic inequality. Current/contact
coefficients count every flavour and actual one/two-link path lengths;
the four-component Ref source has operator cap four on P and Q.

The explicit alternate energy-upper trial is deliberately **not the
filled-negative-band Slater sea**. It is a tensor product of actual
pairwise `wedge2 × wedge2` SU4 singlets with **exact occupation two in
every charged flavour at every cell**, heavy empty and Ref wedge4
(plus branch) or wedge0 (minus branch). Hence rho annihilates this trial
cell by cell, not merely in expectation. Scalar Cartesian Gaussian Phi
(variance `.25`) and constant-Haar SU4 links are gauge invariant;
their field kinetic, quartic, alignment and magnetic energies are paid.
The primitive Ref scalar source commutes with SU4 and EM Gauss, has
executed norm-squared two in either branch, and gives a colocated charged
trial with the same field upper. Singlet residual is `2.18e-16`.
This upper does not remove valence-sea charge fluctuations, sea-fibre
derivatives or Casimir costs from a different prepared state.

All coexact **and harmonic** real photon coordinates are retained in a
normalized width-one Gaussian. Its actual electric energy upper is
`(UEphys/a)(2V+1)/4`, equal to 55 in the `a=.25`, `V=27` witness, not
the old `4cV`. With `Lambda=256`, the exact total trial upper is
`276373633/40`. No global induced oscillator ground or A-coordinate
coercivity is asserted.

### Statistical gap scope and evidence

`Q_stat` means nonflat statistical Pauli cycle sectors. Sea particle-hole
excitations remain in `P_stat`; `Q_sea`/interband gaps are **not** this
Schur gap. With the renewed full-Phi lower, the exact alternate Gauss
upper and window one, the selected penalty is
`Delta=B+Utrial+window+beta+2^80`. Its gap lower is `2^80`, leakage bound
`5.17e-26`, Schur residual `3.24e-27`, and the actual Phi/current/contact/
Ref/two-polarization weighted image bound is `6.96e-14` for the `a=.25`
family. The Phi multiplication resolvent graph has gap power `3/4`;
the bounded defect is paid in its energy remainder. Replacing Delta by
128 fails the full-field advantage upper by `8.97e7`. This is a finite
energy-window auxiliary transfer, not an arbitrary-time prepared-sea
or full interacting QED ground/pole admission.

Source-bound natural-exit capsules:
`/tmp/induced-root-transfer-a025-final.json` (116.77 s) and the separate
diagnostic `/tmp/induced-root-transfer-repaired-final.json` (121.86 s).
Loaded Root SHA256:
`19d73d1d6fec14ed60d5707b7e6a7fe2fa56b8a1a725ed275116374cef663a09`;
loaded photon supplier:
`2ba6981926f1d623fa5dd16bd7f0d25d87dddbd9d2186668f5cefaaad1728ac7`.
Each capsule records all loaded dependencies, input digest and scientific
Python runtime; source changes during calculation are rejected.
The `a=.0625, UEphys=1e-16` run exercises the repaired dimensionless pole
root, but is **diagnostic only**, not the joined UE1 family. Actual Pyright
1.1.414 push diagnostics for the final Root file are empty; positive AST
queries and real symbol/definition/reference navigation were exercised.

### Executed mass-resolved UE1 successor and full constrained vertex

The historical equal-mass and tiny-inertia observations above are not
retagged. The next **executed** Root family is `a=.015625`, `UEphys=1`,
`Ntotal=64`, `species=((1,.2),(63,2.0))`, with order-twelve actual mixed
band response. The active flavour is one of these 64 filled species.
The authoritative neutral heavy descriptor is
`RotationBand(a=.015625,m=.2,r=.12,v=.3,improved=False)` with coefficient
`-Lambda`, `Lambda=256`; it is not the old heavy `a=1` band or an
unannounced `m=1,v=1` heavy repair. The full-Phi quartic is
`Lambda(2/a)||G||²/2`, with zero preferred Gram. The imported selected
Gram remains a prepared reference, not a demonstrated new minimum.

Every actual species mass enters the coefficient ledger:
`K=Lambda h_H+sum_species count*h_species`.
The charged matter l1 sum is
`855444290449274477/17592186044416`;
using active mass for all flavours underpays it by approximately 453.6.
Representative actual Pauli H/direct-Phi comparisons are zero for both
mass groups; replacing auxiliary mass two by active mass `.2` changes
the native H by `12.28889`. The graph still has 65 species plus Ref4,
and the statistical topology is unchanged. Local SU4 and Ref-source
Gauss commutators are now evaluated as **raw full-edge Pauli
polynomials without any cycle quotient**: their residuals are zero.

The physical photon consumer retains the induced scalar polarization:
`V = epsilon0*rho - epsilon·J`, with the SAME onsite primitive density,
Peierls current, active sea background and Ref compensation. A real
restriction of the supplied mode at q `(.05,.035,.02)` defines the
native density/link profiles; it is explicitly not an exact periodic
`N=3` plane wave. Both modes have omega approximately `.00870436617`,
residue `.0323500005`, and amputator `1.36318115`. Their actual active
code-Slater vertex-image norms are `1.25188/1.24784`, with direct
primitive/full-Phi vertex comparison error zero. Omitting A0 changes
the native vertex by `6.00e-9/1.01e-8`; omitting the residue changes it
by `.71467/.71376`.

The native coupled-cell repair also makes field normalization and Gauss
explicit: `A_link=e_trans*x`, `||e_trans||=1`, bare kinetic
`UElink*p²`, and actual longitudinal electric field
`E_L=Incidence^T Lplus*rho`. Its Coulomb term is
`UElink*rho^T Lplus*rho`; rho uses the declared neutral number-two
sector. The executed EM Gauss residual is `1.05e-15`. The native trial
actually pays electric energy 32 and Coulomb energy 16, not a fake
zero from mean background cancellation. Its current/contact derivative
errors are `1.02e-8/9.57e-7`; nonzero `[D,H]`/`[D,J]` norms are
`3.04328/1.52164`. At time `.4`, observed Q amplitude `.00042357` and
current-image error `.00218392` are below bounds `.00196533/.01381220`.
Omitting the statistical penalty increases Q amplitude to `.02183265`.
This exact finite quantum-field panel does not assert an unrestricted
three-dimensional field ground or pole.

The alternate full-Gauss pair-occupation upper remains a different state
from the valence Slater sea. For this actual cutoff its bare photon upper
is 880 and total exact upper is `4133302633/40`; retained quartic
coercivity is 128. The exact all-sector lower uses the actual mass sum,
not a compatible-looking active-only formula. At selected gap `2^80`,
the current/contact/Ref/two-polarization image bound is `7.14e-15`.
The physical current pays its factor a, contact pays a² and **two**
mode legs, and A0/density has its own bounded number/Ref graph
(image bound `4.18e-30`). Source cap four is paid separately. This
continues to control Q_stat only; Q_sea remains in retained dynamics.

Reproduction:

```sh
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/induced_root_transfer.py \
  --a .015625 --flavors 64 --species 1:0.2,63:2 \
  --inertia 1 --order 12 \
  --output /tmp/induced-root-transfer-mass-resolved-final.json
```

This scientific CLI naturally exited0 in59.27s. Capsule loaded Root
SHA256 is
`32bbf87a5539de6269217ac4f399282fcff53cb7923ee0eb0e7a6d7aa0f8f30b`;
loaded photon SHA256 is
`01f70abc8fafe4944aa35ab0127ca4bbb4d2eb70f26b363ce14f7349dabb6596`;
input digest is
`611abc55725c2e467add69ae8238a1dca245813184a2350e12a3c433f8d8cb7b`.
The capsule also binds the Python executable digest, runtime, RNG/
native-field inputs, full dependency hashes and assignment epoch.
Actual final Pyright1.1.414 reopened-buffer diagnostics are empty
(`/tmp/induced-root-mass-resolved-lsp.json`); positive final AST probes
reach the unquotiented algebra and primitive density vertex.

Remaining physical debts are not removed by this transfer: the supplied
Gram's new full-Phi backreaction/quantum preparation, the interacting
collective sea/photon dynamics, self-energy/vertex corrections, physical
spin/Noether transport, and common-cone/spacetime/scattering completion.
The parent-observed mass-resolved common-cone failure remains a failure;
neither this finite gap nor a positive RPA pole repairs it.

