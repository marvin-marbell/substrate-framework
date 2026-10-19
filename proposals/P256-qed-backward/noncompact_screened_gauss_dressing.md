# Smooth screened full-SU4/Phi source and complete physical angular payment

This pair replaces a **source**, not the original full-SU4/Cartesian-field Hamiltonian. It establishes a working equivariant smooth-source capability, its actual CAR/current/Noether consumer, and a branch-specific obstruction to identifying that source with an ordinary local physical half-spin. It does not claim a finite-energy electron, a microscopic SO3 action on a cubic grid, or a total angular Casimir.

## 1. Native frame chart and source

Let Phi have positive scalar Gram `G_mn=Tr(Phi_m Phi_n)/4`. Whiten all ten Gram entries. Form the actual intertwiner operator `A(U)_m=gamma_m U-U Gamma_m`; take its least right singular vector, its polar factor, and remove one fourth of its determinant phase. This reconstructs `u(Phi)` up to Z4. For an exact irreducible Clifford frame the stabilizer in SU4 is Z4. For off-Clifford fields we retain the least-vector/polar construction on a **simple-singular-vector, nonsingular-polar guard**; we do not pretend that every field is Clifford or discard its normal coordinates.

Under gauge conjugation the intertwiner operator is unitarily transformed on its left colour index. Thus the least vector and polar chart obey `u(g Phi g†)=g u(Phi) z`. The code records the Gram lower bound, squared singular gap, polar lower bound and non-Clifford residual. Chart derivatives are meaningful only within this domain; a globally finite-energy polar source additionally requires a smooth guard cutoff and its gradient payment.

Use the ORIGINAL `original_centre_charge.packet` and `packet_derivatives`, not a Z4 rotor law. With `z=i`, define

`psi_q(V)=(1/4) sum_k z^(kq) exp[eta ReTr(z^(-k)V)]`,

`s_eta(V)=psi_1(V)/psi_0(V)`.

The implementation's common `exp(-4eta)` and factor4 cancel in the ratio. Jensen gives the unshifted `psi_0>=1`; the implementation's shifted denominator is positive, **not >=1**. Positivity and `|s|<=1` are global. `s(zV)=z s(V)`. Its exact full15 derivatives are `(dpsi1 psi0-psi1 dpsi0)/psi0²`. The canonical generator normalization remains `Tr(Ta Tb)=delta_ab/2`, `sum Ta²=15I/8`.

For an oriented path from x to outside R, let `V_xy=u_x† U_xy u_y`. With the primitive quaternion marks `q=(I,2iS1,2iS2,2iS3)`, the actual source is

`M_a,xR=eta_x exp(i[chi_x-chi_R]) exp(-i(x-R).a_h) u_x q_a [prod_path s_eta(V)] u_R†`.

Here `chi=G_MP div A` is the original real-link Coulomb phase; the zero mode is quotiented. The optional harmonic argument owns its **separate** `exp(-i(x-R).a_h)` factor and is not also inserted in Coulomb A. SU4 centre holonomy remains only in U. No photon theta is silently interpreted as an additional colour centre phase. Reverse path edges use U† and exchange endpoint charts.

Independent chart replacements `u_y->u_y z_y` multiply each scalar by `z_x* z_y`, telescope along the full path and cancel the endpoint factors. Gauge equivariance is `M->g_x M g_R†`; the complete four-component opposite reference is retained.

## 2. Actual metric, CAR and energy input

The source convention is **q, not q/2**. On a CT-invariant neutral comparator, the actual complete-reference CAR metric is `G0_ab=(1/2)sum_x Tr(M_a,xR† M_b,xR)`. Multiplying every source by1/2 scales G0 by1/4 but leaves frozen-initial-metric derivative forms invariant. Never assume `G0=2I`: scalar attenuation changes it. The executable exercises full exterior CAR for all 12 microscopic modes and all four RIGHT components and independently checks the exact whole-sea Slater identity on the native 3D law. Its Phi-sign CT-even source is directly checked on non-Clifford fields.

`native_consumer` computes the frozen initial inverse once and provides:

- `all_path_and_RIGHT_Cartesian_Phi_kinetic_matrix`: every one of the 60 Cartesian derivatives at every source/path/RIGHT site;
- `all_path_full15_colour_kinetic_matrix`: all15 canonical left derivatives on every path link;
- `angular_horizontal_initial_metric_matrix`: the complete retained horizontal angular derivative, including electric flux and RIGHT;
- `original_slow_extra_energy_per_chi=.5 PhiForm+.3 LinkForm`.

These are pointwise derivative forms, not a neutral expectation or total energy certificate. For the declared original slow trajectory `chi=Lambda^(-1/2)`, use its actual coefficients `chi/2` and `.3chi`, not a fast `sqrt(Lambda)` or invented rotor inertia. Cartesian coordinates use the native B basis with `Tr(Ba Bb)/4=delta_ab`; canonical link generators are `Ta=Ba/sqrt8`. Neutral kinetic product cross terms, smooth guard cutoff derivatives, photon/harmonic expectations, sea and potential changes remain explicitly unpaid. CentreSourceEnergy owns the integrated original-law budget and the guard-free successor below.

## 3. Complete local physical Noether derivative

For the ordinary plateau f=1 on the charged cloud and f=0 at RIGHT, physically rotate spatial Phi indices. Decompose its **actual** tangent using all15 Gauss columns and the original field/link mass metric `diag(I60,I45/.6)`:

`Y=G ell+Y_horizontal`.

The source derivative satisfies

`d_phys M=i ell_x M-i M ell_R+d_horizontal M`.

The executable differentiates the actual non-Clifford Phi and noncentral SU4 links, including the opposite electric-link tangent in `Y_horizontal`, and checks this equality. There is no appended microscopic matter spin. The angular graph norm includes the Cartesian fields, links, orbital source, real photon vector/electric tangent and reference; omitting any one is not an angular theorem.

For a relative-link tangent `dV=i Da V`, canonical coefficient norm `||d||²=2Tr(D²)`, the scalar angular term has the computable bound

`|d prod s(Ve)| <= sum_e b_eta(Ve)||d_e|| prod_(j!=e)|s(Vj)|`,

where `b_eta=min(2eta sqrt30,4eta sqrt30 sum_wrong exp[-eta score_gap])`.

This is exponentially small on a fixed near-centre tube, but **not along an entire physical rotation loop**. Bad-link/guard probabilities and their angular correlations must be paid with the original neutral law. In particular an ordinary local2pi history cannot remain inside the same tube everywhere.

## 4. Exact ordinary-loop failure: seam, not word length

At identity links the plateau has one boundary relative link `V(theta)=exp(-i theta Sz)`; all interior links are identity. Its trace is `4 cos(theta/2)`. At theta=pi every centre seed has equal weight, hence `s=0` exactly. At2pi `u_x=-I` and the boundary scalar also changes sign. The complete smooth source returns **+I**, not -I. The intrinsic-spin shortcut drops precisely this compensating seam.

The scalar total variation along the loop is at least `2|s(I)|`; after the one initial normalization it is at least2. Adding 8,80 or800 interior factors does not remove that obligation. Unlike the failed thin word, the new scalar gradient is exponentially small near the endpoints and concentrated at the seam; nevertheless it is not uniformly small and the source metric collapses at the seam. No final inverse or reset is permitted. The executable also remeasures the thin word's same full-quartet CT horizontal norm squared2 for all three lengths, and reports the screened source attenuation separately.

This is a counterexample to this ordinary local-spin construction, **not a total no-go** for dynamically noncontractible charged cloud sectors. The positive band/orbital chart remains usable, but is not evidence of the missing local source angular intertwiner.

## 5. Materially different cloud routes actually exercised

First solve the original mechanical screened response and move both Phi and gauge links with it, keeping its endpoint and outside reference. This is a real screened field/electric-link cloud, not a thin-word change. The executable measures its link mass norm, field endpoint, guard-link endpoint, complete source covariance and relative source-minus defect. It fails as a physical spin mechanism: the corotating part is vertical Gauss motion, its endpoint is not the same ordinary world and its complete neutral history cannot be discarded.

**Selected successor:** CentreSourceEnergy's normalized whole-Cartesian covariant orientation cloud. For each path site integrate independent full-Haar orientations g with posterior

`p_alpha(g|Phi) proportional exp[2alpha <Phi,g Phi_star g†>_Cartesian]`.

Integrate the complete primitive source integrand `g_x q_a prod s_eta(g_y† U_yz g_z) g_R†` against the normalized positive posterior. This is an entire source on the full Cartesian domain, including Phi=0; it removes the polar-chart guard rather than assigning a new end spin. Its field derivatives are posterior covariances and its kinetic coefficients remain those of the original law. The finite-node consumer uses genuine fullSU4 Haar product samples, matched transformed nodes for gauge identities, complete CAR, and exact Cartesian covariance derivatives. Matched-node covariance is **not** Haar convergence or a nonperturbative angular-holonomy proof. This different smooth cloud still returns + on a closed ordinary physical2pi loop.

The other successor is a genuinely horizontal dynamically stabilized texture with a noncontractible physical configuration loop, followed by quantization relative the same neutral history. Its existence, topology and stability in the original Cartesian law are not established here. Neither successor is another global-reference calibration.

## 6. Reproducible scoped consumer and observed result

Run from this proposal directory using the project Python:

`/home/administrator/substrate-framework/.venv/bin/python noncompact_screened_gauss_dressing.py --size 3 --screen-eta 4 --loop-samples 400`

At the executed N3 native non-Clifford/noncentral input: gauge error5.11e-15, centre branch error0, full15 scalar finite-difference error2.03e-11, actual exterior-CAR error5.54e-16, G0 eigenvalues1.99999703264, source CT error6.56e-15, complete Gauss error1.88e-10, physicalNoether decomposition error2.85e-10. The horizontal source norm is1.91507. All240 Cartesian path/RIGHT and45 link derivatives are retained; the extra source kinetic matrix is approximately27.7774chiI, with small nonzero off-diagonal terms. The actual new-source orbital evolution error is9.21e-15; the same-world real photon/electric rotation error is0.

The local loop has + endpoint error2.45e-16, seam source norm2.65e-15, scalar variation1.99999955 and full15 maximum gradient2.82843. The - endpoint defect is16 in raw single-mark HS convention, or8 in half-HS CT convention. The material corotating cloud has guard-link endpoint defect0.09066 and field endpoint defect0.81484; its outside response is0.0362749 rather than0. Discarding it would change the world, not prove spin.

The corotating cloud is also passed through actual 12-mode exterior neutral/charged CAR evolution and the complete bar4 reference, then the required Gauss identification. It returns the SAME initial source state with error8.95e-15; an imposed source-end-only minus has defect5.60027. Thus the negative decision is made at the full neutral-history consumer, not a coefficient sign.

The selected orientation-cloud consumer calls `noncompact_centre_source_energy.orientation_cloud` directly; no second posterior convention is installed. Initial unpaired512-node diagnostics are retained in the earlier receipt. The physical CT consumer now pairs every configuration with `g Gamma5` at **all** path vertices, yielding1024 total nodes. Since Gamma5 anticommutes with every native template Phi_mu and commutes with q, this pairs the finite rule exactly under Phi sign. Source CT error5.80e-16, Cartesian derivative error2.51e-11, original-link derivative error1.73e-10 and actual complete-CAR error4.18e-17 are observed. Its remeasured G0 eigenvalues are0.00194454,0.00299381,0.00412147,0.07814147, **not2**. Supplier full120Cartesian and15link jets/Laplacians are consumed into frozen-initial-metric energy matrices. Finite sampling is not a lower bound on the continuum Haar source metric. The supplier's separate actual-neutral CT/moment/source-error theorem may supply that lower bound under its declared original-well hypotheses; numerical node rank alone does not.

The actual radial positive-band consumer separately pays the cubic-lattice remainder: at momentum radius0.02 its orbital-aware spin intertwining defect3.51e-7 and radial J² defect1.44e-6 remain nonzero. These are continuum approximations to L+S, not microscopic SO3 or total localNoether/J². Real Pyright document symbols/diagnostics and positive Python structural AST were exercised before implementation; changed-file diagnostics are scoped separately.

## 7. Physical cube extension and the localization boundary

The closed ordinary Phi-only loop above is not the definition of a separated symmetry-localization operator. The native Clifford vacuum has the exact extension `Gtilde={(R,g):g Gamma g†=diag(1,R)Gamma}`, with kernel Z4. Orthogonal pi rotations commute on spatial vectors, but their **derived** Clifford compensators anticommute. An abstract extension alone would not prove physical spin, so `physical_cocycle_and_flux_consumer` executes the actual physical operations first: spatial sites, vector Phi indices, oriented gauge links, real photon/electric vectors and the outside reference. It then uses `make_lift`/`restore_links` for the actual Gauss body restoration.

On the executed108-mode native Hamiltonian, four physical pi operations close the complete world to4.42e-14. Their one-body cocycle is the central -I. The whole54-occupied-mode neutral exterior determinant is +; the charged-minus-SAME-neutral complete CAR overlap is -I to8.74e-14. The complete bar4 reference contributes the other minus and the physical global pair overlap is +I to8.94e-14. These are actual cubic group controls, not continuous SO3 or individual localized physical generators.

Original Gauss law supplies an additional exact, independently checked localization ingredient. A region C with the charge inside and RIGHT outside has centre gauge value -I in C and +I outside. The adjoint Phi fields are unchanged, but every crossing SU4 link is translated by the actual centre. Centre-equivariant scalar transport changes the source by -1. In a Gauss-constrained state, exponentiating the regional Gauss constraint relates this **boundary electric centre-flux operator** to the matter centre character `(-1)^N_C`; the adjoint field centre character is trivial. The source adds one CAR mode inside C, so it changes that character by -1.

The executable checks this through actual shifted native links and the whole exterior operator Ward identity `B_flux P_neutral=P_charged B`. The residual centre source character and fullCAR flux–charge Ward errors are both0. Full Gauss transformation remains gauge identity, not a physical minus. Keeping the boundary flux character is essential; simply keeping a source-end gauge sign repeats the failed shortcut.

**The boundary that is still unpaid:** an exact global extension plus exact regional Gauss charge does not construct individual localized physical rotation operators. A flux-only operation changes the native H by operator norm0.732539 and costs0.155266 in the same fixed neutral sea at N3. Thus it cannot be declared an independent low-energy symmetry merely because it carries q1. On the original charged low sector, one must actually bound the regional Noether/current/field/photon and boundary residuals, with source support and RIGHT separation. The all64 normal-gap and centre-plaquette barrier supplied by CentreSourceEnergy are useful ingredients, but do not by themselves prove fixed-law clustering or this factorization.

For clarity, the full source differential identity remains `-i d_phys M=ell_x M-M ell_R-i d_horizontal M`, after all orbital and photon terms are retained. In the vacuum core the physical vector tangent is a Clifford commutator, fixing S rather than appending it. Away from that core, its actual Cartesian normal, link flux, orbital and photon contributions must still be bounded in the SAME state. The code explicitly sets `localized_individual_physical_Noether_operators_admitted=False`. The selected physical successor is this original-law symmetry-localization construction, supported by the exact source/energy and flux-charge capabilities; it is not another global-reference calibration.

## 8. Different dynamical Berry/rotation protocol actually run

Because flux algebra does not pay individual localization, the pair also implements `dynamical_local_frame_protocol`: physically drive the original Phi spatial indices in the charged region through a closed2pi loop, keep the actual SU4 links, photon coordinates and RIGHT fixed, and transport the **initial prepared state** rather than rebuilding the smooth source at each angle. The entire time-dependent native H is used for both charged and SAME-neutral evolution. This is an externally prescribed native-field protocol, not an Euler solution or a spatial SO3 action on a cubic grid.

The geometric comparison uses the actual positive spectral cluster and polar Kato steps; no spin factor is appended. Independently, midpoint native Schrodinger propagators are multiplied in time. The actual complete filled-sea charged-minus-neutral overlap is evaluated by the neutral determinant and its one-added-particle Schur complement, retaining all four reference components. No final source metric is inverted.

At meshes32 and64, the native gap stays at least0.129680; the full Phi/H world closes to2.48e-17. The driven native unitary defects are below3.30e-13. Kato mesh change is1.94e-4 and native dynamic mesh change2.45e-6. This simple screened-cloud drive does **not** produce the desired half-spin: the source Berry plus defect is0.04005 and minus defect5.65671. The charged relative Berry overlap is near the identity with a small genuine rotation, not an imposed minus.

The actual domain payment is also visible: the original adjoint alignment W reaches10.93418, costing `0.273354 Lambda^1.25` with the original J coefficient, while the instantaneous same-neutral sea shift reaches0.0255005. At duration0.4, the unprojected original Cartesian field drive kinetic energy satisfies `E_drive chi=14.05156`. This is explicitly the unprojected externally imposed field velocity, not a Gauss-constrained cloud energy certificate. Coupled field/photon backreaction and the charged Gauss momentum solution remain unpaid. The result is a real failure of this drive, not a no-go for a dynamically noncontractible sector.

## 9. Source derivatives through order6, not a total J graph norm

`orientation_cloud_derivative_majorants(L,eta,alpha,B²,G0lower,max_order=6)` exports conservative ordered whole-Cartesian and all15 link derivative bounds. With q/2 unit-HS marks,

`C_q <= q! [4 alpha sqrt(B²(L+1)) + 2 sqrt(15L)(1+4eta)]^q`.

Here `B²=sum_mu Tr(Phi_star_mu²)/4`. Every ordered canonical generator has operator norm below1, so a k-th centre-seed score derivative is bounded by4eta. The exponential derivative Bell polynomial is at most `k!(1+4eta)^k`; differentiating the positive-denominator ratio recursively gives the factor2^k. The field posterior score is linear in B coordinates, with gradient norm `2alpha B sqrt(L+1)`; its normalized expectation obeys the analogous quotient bound. Combining field/link tensors gives the displayed conservative sum. Four q-convention source marks give at most `4 C_q/sqrt(G0lower)` after the one frozen initial normalization.

These constants are useful to QuantitativeNormalBudget but are **source derivatives**, not a physical Noether theorem. Contracting with the actual Phi-dependent vectorfield, its Lie derivatives, orbital generator, photon/electric coordinates and reference/boundary localization requires the corresponding actual-state moments. The export says `total_physical_Noether_bound=False`; it must not be relabelled as a total J^6 bound.

## 10. True full-spectral mixed graph: the first-moment shortcut is not used

Write the ORIGINAL physical operator, before quotient coordinates, as `H_R=T+V`, after retaining the negative classical baseline `Lambda v*` and adding the actual exterior light constant. For `Lambda>C1`, the original inequality `||H_light||<=C0+C1(U_Lambda-v*)` makes `V>=0`. The Cartesian, full15 link-Casimir and noncompact photon kinetic summands of T are positive and strongly commuting; each preserves the full Gauss subspace. No projected Hamiltonian replaces this operator.

On its smooth physical form core, integration by parts gives

`||T psi||^2+||V psi||^2 <= ||H_R psi||^2 + <sum_i t_i D_i^2 V>`.

Indeed the omitted cross term is `2 sum_i t_i <D_i psi,V D_i psi>`, which is nonnegative even for the actual matrix-valued CAR potential. The identity extends by the closed graph/Friedrichs approximation. This is stronger than a kinetic first moment and is the needed discriminator for an unbounded Noether insertion.

The coefficients are from the original Cartesian law. For X of shape4 by15, `rho²=||X||²`, and quartic `kappa ||XX^T||²/2`, the complete Cartesian Laplacian is `40 kappa rho²`. The raw heavy and light endpoint matrices are linear in Phi, so their Cartesian Laplacians vanish. With the actual alignment coefficient `.2 Lambda^1.25/8`, the full Cartesian alignment Laplacian is `72 V Lambda^1.25`; its all15 colour Laplacian is at most `2.4 Lambda^1.25 rho²`. Summing the four fundamental-Casimir15/8 derivatives of every plaquette gives at most `90 V K4`, with `K4=.7 Lambda^1.25`. The kinetic cometrics are `chi/2` and `.3 chi`; these yield the coefficients `36 V Lambda^.75`, `.72 Lambda^.75 rho²` and `18.9 V Lambda^.75`. The quartic contribution is `20 kappa Lambda^.5 rho²`.

The heavy colour Laplacian is bounded by its actual endpoint row bound times `CF Lambda rho`; the light colour and real-Peierls photon Laplacians are bounded by constants times rho. The latter is not merely a constant photon remainder. The real quadratic magnetic Laplacian is its actual electric cometric times the finite curl trace. The matched Coulomb operator `UE rho_charge G rho_charge` has no Phi/U/A derivative; harmonic electric energy is `UE/V p²`, not an invented half. Thus the total cometric potential Laplacian is bounded by `C_N Lambda^.75(1+rho²)` for Lambda at least1.

Coercivity `V>=a_Lambda rho^4-c_Lambda`, with `a_Lambda>=c_N Lambda` and `c_Lambda<=C_N Lambda`, gives on the TRUE spectral window `<rho^4><=C_N`, since the shifted physical energy is at most `C_N Lambda^3/8+E+C0`. The squared-form bound gives `||T psi||<=C_N Lambda^3/8`, hence `||Pi_Phi² psi||<=C_N Lambda^7/8`. Finally, the exact mixed identity

`||rho Pi_Phi psi||² = Re<Pi_Phi² psi,rho² psi>+60 V ||psi||²`

gives `||Phi Pi_Phi psi||<=C_N Lambda^7/16` on the true full spectral window. A product of unrelated Phi and Pi first moments would not prove this. `true_spectral_field_graph` exports this calculation with literal shifted-energy/coercivity/Laplacian coefficients.

For the photon graph, use the original colour/heavy operator `K_col=H-H_light-H_EM`. It has no A and preserves each light site occupation, so it strongly commutes with the actual Coulomb/coexact/harmonic H_EM, also after full Gauss restriction. Repeating the colour-only normal/complement argument gives `K_col-Emin>=-1` once its actual residual is paid; this bound is not obtained by subtracting H_EM from a full-H inequality. Therefore positive commuting `A=K_col-Emin+1` and `B=H_EM` imply, on a true `H-Emin` energy window, `||H_EM psi||<=E+1+BL||rho psi||=O_N(E+1)`. The negative classical baseline is not dropped; the `H-Lambda v*` window would still contain the large vacuum energy and would not give this result. Strong graph/domain closure follows from the squared-form core argument and the commuting positive sum. Coexact angular A-Pi products are then controlled by the actual oscillator energy and lowest coexact frequency. Harmonic A-Pi is **not** controlled by electric energy: its literal prepared-packet moment remains an input, and a sharp energy projection may destroy its A-domain.

## 11. Actual Gauss-derived core Noether and the retained electric boundary

The microscopic angular observable is the symmetrized generator of the original Cartesian-vector, covariant spatial/orbital, full colour electric and real-photon flow, plus its actual scalar CAR orbital generator and reference action. It contains no bare inserted spin. The complete vectorfield grows at most linearly in Phi/A, has smooth compact-colour coefficients and a finite CAR transport matrix; its unitary flow defines the physical differential operator on the full Gauss domain. The half-divergence term is retained where necessary. Source commutators do not lose it: a multiplicative divergence term commutes with the source.

In the actual mass-horizontal bodychart, let `I=G†MG`, `A=I^-1 G†M dy`, with all15 Gauss columns and `M=diag(I60,(1/.6)I45)` at a uniform core. Physical vector tangents have `dPhi_j=-Phi_k,dPhi_k=Phi_j`. Solving `I ell=G†M Y` gives their colour generators. The executable independently solves all three, then checks that their commutators really produce those Cartesian vector tangents. Their half factor is a consequence of this solve.

On the full-Gauss isometry W, the normal scalar ground factor is common to the charged and neutral histories. Its real symmetrized horizontal first derivative has zero compressed mean by integration by parts, not by dropping its variance. The heavy scalar Berry/background term is also common. The relative observable is the actual operator connection `J_charged B-B J_neutral`, not the generally wrong replacement `J_charged B-<J_neutral>B`: neutral covariance is not silently discarded. The Gauss constraint turns the vertical component into the light/reference charge action; the horizontal field/link normal and photon/orbital contributions remain until their graph bounds are paid.

For an isometric Clifford core and an ordinary rigid plateau f, the spin Gauss sector is the actual massive discrete equation

`(m+Delta) a=m f`, with `m=.6 gs`.

This is the Schur/mechanical connection of the original Phi/link cometrics, not a Z4 Hamiltonian. At the charged point x and outside reference R, the relative core matrix is

`N_i=a(x) L_i-a(R) R_i`.

Here `L_i` and `R_i` are the LEFT and RIGHT actions on the actual four source marks, obtained by full-CAR/initial-G0 contraction. The second term is the reference/boundary electric contribution; omitting it is not permitted. LEFT and RIGHT commute, `sum L_i²=sum R_i²=3/4`, and RIGHT has the reversed multiplication commutator. Therefore `||N_i-L_i||<= (|1-a(x)|+|a(R)|)/2`. The discrete Green kernel bounds both errors exponentially with `kappa=acosh(1+m/4)`. The producer computes the actual Fourier response on graphs17,33,65,129,257, and evaluates the Noether matrices, SU2 commutators, J² and2pi directly. These graphs use the stated native-frame coefficient; they are not silently asserted to have its minimizing Gram.

The executable also contracts the final three matrices through the entire12-mode exterior CAR and complete bar4 RIGHT with the SAME neutral/particle-hole pair and one frozen initial metric. The real photon contribution is zero at A=E=0 in this specific core probe, and the actual covariant scalar-orbital matrix annihilates its centred point source. This does not remove orbital/photon contributions from a mobile or irradiated packet.

## 12. Weighted physical transfer: why the normal connection matters

The generic Hamiltonian residual `d=O(Lambda^-1/16)` does not bound an unbounded J. Multiplying it by a momentum graph would even grow. A different derivation uses the actual RNC normal connection. At the centre of the mass-horizontal normal slice, `G0†M0 Z=0`, hence the light/reference mechanical normal connection has `A_L(0)=0` exactly. Its first jet and all higher required jets are controlled by the actual inverse Gauss matrix, chart and complete projector jets; the executable checks this zero and computes all90 first jets at the native core. The vertical term `chi rho†I^-1 rho` remains nonzero and bounded; it is not eliminated by this choice of section.

On the original shrinking core `epsilon=Lambda^-1/5`, `A_L=O(epsilon)` improves the light/reference first-order normal channel to

`R_N<=C_RN Lambda^-21/80+C_L Lambda^-5/16+C_B Lambda^-1/2+cutoff_tail`.

This uses the actual `||p u||<=C Lambda^7/16`, not a guessed sharp normal width. Off-heavy Kato derivatives still give `R_H<=C_RH Lambda^-1/16+C_B Lambda^-1/2+tail`. They belong to a different channel, with a core heavy gap proportional to Lambda.

The channel coupling is an unbounded first-order differential form `chi B_alpha p_alpha+chi B0`. It is **not** declared a bounded block operator. Smooth Kato/chart jets and the positive kinetic form control its Schur complement; integration by parts and the closed form inverse define the coupling. In particular, the weighted resolvent is estimated from positive kinetic/coercive potential graphs, not from the norm of d. In the normal heavy-ground channel, gap g and the retained vacuum excess give

`||J (D_N-E)^-1|| <= C chi^-1/2 sqrt(g+C Lambda^3/8)/(g-E) <= C Lambda^3/16`.

Thus the leading normal mixing error is `Lambda^(3/16-21/80)=Lambda^-3/40`. On a localized heavy-excited core, the same estimate with its true gap proportional to Lambda is `C Lambda^-1/4`; its mixing error is `Lambda^-5/16`. A **global** heavy gap proportional to Lambda cannot be borrowed outside regular cores. One can instead combine the heavy core and the original globally forbidden exterior using IMS/Agmon: its safe gap is `c_phase Lambda^3/5`, yielding a weighted resolvent `C(c_phase) Lambda^-1/20` and mixing `C(c_phase) Lambda^-9/80`. The actual all64 full-domain phase coercivity and interface derivatives must be paid in the Lambda selector. The exterior Agmon action is proportional to `Lambda^7/20`; polynomial current/derivative losses therefore remain exponentially controlled. Heavy-to-normal feedback carries an extra chi times the weighted momentum resolvent and is of strictly smaller order. None of these statements assumes an invariant compressed normal complement under the full dynamics.

`refined_noether_transfer_selector` reuses the literal `NormalBudget` constants. Its local power Lambda=t^80 pays the leading normal term as `C/t^6`, localized heavy term as `C/t^25`, and the other displayed terms with smaller powers. The globally exterior-safe heavy coefficient carries the explicit `c_phase^-2` cost and exponent `-9/80`; the actual supplier's symbolic global selector must pay it. The function intentionally does not certify the whole physical packet from these local terms alone.

For the temporal observation use the SAME `SmoothPulse(order=8)` on the actual full H, with `tau=Lambda^-1/32`. Let P be the true resolved energy projector and Q the true spectral projector above g, not a noninvariant compressed normal C. Eight integrations by parts followed by the separated-spectrum Sylvester/Laplace inverse give

`||Q J_tau P|| <= ||w^(8)||_1 [(g-E)tau]^-8 ||J P||`.

The true mixed graph in §10 gives the field/link contribution `Lambda^7/16`; since `g tau` grows as `Lambda^7/32`, its high-spectral remainder is `Lambda^-21/16`. `physical_spectral_filter_bound` uses the actual polynomial derivative majorant. Time smearing does not delete low-to-low virtual-normal terms; those are the weighted RNC/Schur terms just derived. Photon/orbital/reference graphs and higher energy-tail graphs are still literal packet inputs. An energy-probability tail alone is not a J-weighted tail.

## 13. The materially different smooth Haar source also closes the physical core

The finite-alpha orientation source need not exactly map its four marks to an invariant Spin subspace. It would be incorrect to repin its metric to2I or assume exact closure from covariance. A separate full-Haar concentration estimate pays this defect while preserving the globally smooth source.

For the true Clifford core Phi*, write `B²=sum_mu Tr(Phi*_mu²)/4`, and let g_min be the full Gram minimum. Its posterior deficit is

`D(u)=B²-score(u)=sum_mu ||[u,Phi*_mu]||_HS²/8`.

The full Clifford commutator spectrum has smallest nonzero eigenvalue4, so `D(u)>=g_min ||u-Tr(u)I/4||²/2`. Unitarity and detu=1 give the conservative `dist(u,Z4)²<=50 ||u-Tr(u)I/4||²`; hence `D>=g_min dist(u,Z4)²/100`. These distances partition a proof only: the source does **not** evaluate a discontinuous nearest-centre function.

An explicit normalized-Haar ball lower bound needs no unverified sampling convergence. Push a standard4 by4 complex Gaussian through positive-diagonal QR and determinant-quarter normalization, which is exactly SU4-Haar by equivariance. The Gaussian box with HS displacement at most `r/4096` maps into the SU4 ball of radius r. Elementary Gram-Schmidt/determinant bounds give this inclusion for r at most1. The32-real-coordinate box density is at least `pi^-16 exp(-9)`, giving

`Haar(B_r)>=c_H r^32`, `c_H=pi^-16 exp(-9)[2/(4096 sqrt32)]^32`.

Since `D<=2B²r²` in that ball, setting r=alpha^-1/2 gives `Z'>=c_H alpha^-16 exp(-2B²)`. Therefore a node outside centre distance d has posterior probability at most `c_H^-1 alpha^16 exp(2B²-alpha g_min d²/100)`. Its squared distance moment is at most `d²+16 p_tail`.

At every centre node the full-group scalar is `s_eta(z)=z tanh(2eta)`. All centre phases telescope through the complete bar4 reference. For L edges the exact core mark is `c_eta^L q_i/2`. The original smooth ratio has a conservative full-group Lipschitz bound32eta; telescoping the L scalar factors and two endpoints gives whole-four-mark perturbation

`delta<=2 sqrt(L+1)(1+64eta L) sqrt(d²+16p_tail)`.

Thus the actual CT initial metric obeys `G0>= (c_eta^L-delta)² I/2` in the supplier q/2 convention, or four times this in the q convention. The initial polar/metric identification is fixed once. The physical core projected generators differ from the Gauss-derived LEFT half-spin representation by at most `4delta/(c_eta^L-delta)`. Source-spin leakage is bounded at the same scale. `orientation_core_halfspin_bound` chooses a finite dyadic alpha and proves these inequalities numerically using logarithmic tails. It does not claim the existing1024-node diagnostic integrates that large-alpha posterior.

Normal field/link displacement changes the posterior by the literal score bound `2alpha B (L+1) C_chart epsilon`, and changes scalar transport by the literal full15 link bound. These are finite constants times epsilon once alpha is selected. Alpha is fixed FIRST, then Lambda is selected to pay these terms and the original full source jets/kinetic/current graph; no delta-function source or zero-energy alpha infinity limit is substituted.

For a prepared mobile/photonic packet, add the actual orbital/band-cubic, photon/harmonic, reference/boundary, phase-Gram, weighted energy-tail, RNC/Schur and high-spectral terms to the Haar/core screening error. If this total is e, three Hermitian projected matrices satisfy the conservative bounds `||J²-3/4||<=3e(1+e)`, `||[J_i,J_j]-i eps_ijk J_k||<=3e+2e²`, and `||exp(-2pi iJ_i)+I||<=2pi e`, in the one fixed initial chart. This is a finite-resolution physical-Noether construction with explicit inputs, not the raw closed-Phi coefficient loop of §4 or §8. The latter remains PLUS and does not measure this spectrally resolved coupled-medium observable. The actual finite cubic group/cocycle remains exact; no exact microscopic continuous SO3 symmetry on the cubic lattice is claimed.

**Decision:** there is now a constructive original-law route to the local Noether join: the true full-spectral mixed graph, actual mass-horizontal zero/first jets, Gauss-derived three core generators with their RIGHT electric term, full-CAR matrices, and finite smooth-Haar source concentration. This producer supplies literal source/normal/EM/orbital/localization inputs; the physical-transfer and source-flight consumers compose them with the actual receiver and common history. Successor1 is that actual physical receiver contraction, selected. Successor2 is a directly evolved self-consistent field/photon cloud if the selected construction's explicit costs fail.

## 14. Scoped observations and exact consumer inputs

The actual native N3 consumer gives full15 inertia minimum0.02847442384, mass-horizontal connection origin error3.72e-15 and complete90-normal first-jet Frobenius norm330.990690. The three independently solved physical vector-tangent errors are0,0,1.47e-16. The original CT source metric is `1.99999729916 I` in the q convention, not an installed2I. Entire-CAR metric error is6.27e-16 and the final screened Noether matrix contraction error1.96e-16.

On screening graphs17,33,65,129,257, the directly computed generator errors are respectively0.384605,0.226329,0.0613182,0.00357267,1.00579e-5. At257 the actual J² error is3.01735e-5, SU2 error1.00577e-5 and2pi-minus error6.31959e-5. This is a measured scale-dependent LEFT/RIGHT matrix with the original Gauss Green coefficient, not a source-end sign.

`--normal-noether` contracts the ordinary physical Phi vector tangent into **all2430** native N3 normal modes:45 soft plus45 stiff at q0 and2340 nonzero-q stiff. It retains the full H0-coupled q0 block. At Lambda2^64 and2^80, the actual minimum frequencies are4317.879 and69086.072. The unsmeared physical horizontal one-phonon norms are6.57480e7 and8.41574e9; the SAME compact r8 clock at durations0.25 and0.176777 gives5.61494e-37 and1.92526e-44. Conservative operator bounds are0.0130826 and6.23826e-9. Mass-Gauss errors are below3.84e-15 and source reconstructions below1.47e-16. The producer calculates norms from the supplier's squared coefficients; it does not invent their phases. These are genuine full-axis quadratic-normal observations, not a simulation of the nonlinear full quantum state or its quadratic pair-creation terms.

An earlier Lambda2^16 run failed the supplier's16-step Riccati fixed point, with residual1.89e61. This was a noncontractive numerical block solver, not a negative physical Hessian or a reason to discard axes. The larger-Lambda successor reached the complete block assertions and natural exit0. The failure receipt is retained.

The finite-Haar concentration selector on the actual short Γ-core pathL6 chooses alpha2^79 for requested generator tolerance1e-3; its proved source-core bound is6.25010e-5 and q/2 CT Gram lower0.499983. This is an analytic full-Haar posterior bound, not a claimed high-alpha Monte Carlo observation. For the legal far-RIGHT maximum word `L=5*2^95+2`, fixed eta4 loses the source norm and is rejected. Choose `eta=log(8L)/4=17.3844654`; then `c_eta^L=0.778800783`, and the finite selector chooses alpha2^372, again giving core generator bound6.25010e-5 and CT Gram lower0.303256. The stable evaluation is `log c_eta=log1p(-2/(exp(4eta)+1))`: using `log(tanh(2eta))` rounds to0 and incorrectly erases an order-one accumulated far-word loss. This source norm precedes any particular photon/rest projection; its receiver must use that projection's actual frozen input Gram, not overwrite it with the continuum CT bound.

The exact rational local weighted-Noether selector, using the admitted N25 native Gram enclosure, source norm upper3, field/link coefficient1, photon/orbital/reference coefficient1 and energy window1, selects Lambda`2^468160` for its **local terms only** at target1e-3. Global phase/exterior, large-alpha source jets, actual far-RIGHT harmonic moments and the actual incoming/scattering packet must still enlarge this expression as needed. The native original60-coordinate quartic finite-difference Laplacian reached the independent `40 kappa rho²` assertion with error below1e-6, before an intentionally retained failed N5 default-Gram admission; the corrected N25 arithmetic/source receipt exits0.

Reproduction: `python noncompact_screened_gauss_dressing.py --size 3 --screen-eta 4 --loop-samples 400 --normal-noether`. This reaches the actual source, centre seams, finite normal derivatives, CT CAR metric, physical cubic/flux consumers, separate failed Berry route, three Gauss-derived screened Noether matrices and full native normal-clock observations. Permanent tests or alternate Hamiltonian domains were not added.

## 15. Linear real-Cartan comparison: harmonic null does not mean photon zero

This section is the **linear comparison**, not the literal original photon primitive. The existing `mechanical_flow` real-Cartan witness uses `Y_A=-v·clover(curl A)`, with compact periodic `v=f Omega×(x-x0)`. Its real clover is linear and annihilates uniform harmonic A exactly. The original `primitive_flow` instead uses the sine of each oriented plaquette; §16 consumes that unchanged law. For the linear witness, the ordinary discrete Cartan lift differs by `grad(lambda)`, with the centred local contraction `lambda_x=(1/2) sum_i v_xi(A_xi+A_(x-i)i)`. The associated matter/reference lift differs by the COMPLETE U1 Gauss charge action. Its field-dependent ordering trace is

`Tr[grad d(lambda)/dA]=(1/2)sum_(x,i)(v_xi-v_(x+i)i)=0`

on the actual periodic graph. Thus the gauge-equivalent symmetrized lift is legitimate on the physical domain, not a discarded ordering anomaly. This is a local lattice Cartan lift; the physical continuous/cubic distinction and paid band discretization errors remain. A global rigid vector rotation has a different nonperiodic/seam boundary and must retain its harmonic A-Pi term.

For the local flow, `J_ph=(E·Y_A+Y_A·E)/2` needs coexact A-E, harmonic `p_h` times coexact curl A, and the longitudinal Coulomb E contribution. It does not need an energy-only bound on `A_h p_h`. The harmonic cross graph is controlled by the commuting positive harmonic and coexact Hamiltonians: `H_h V_mag<=H_h H_coexact<=(H_h+H_coexact)^2/4`. The coexact quadratic oscillator graph is controlled by `H_coexact/omega_min+mode_count`. Together with §10's actual H_EM graph and the bounded finite-CAR Coulomb charge matrix, these give finite local Noether graphs. Global/scattering harmonic source derivatives proportional to `D^q` are not removed.

The linear comparison is exercised at real nonzero A, noncentral full-SU4 links and all-Cartesian non-Clifford fields, with the outside reference fixed. Linear flow norm is0.482609, harmonic-shift null error7.73e-17, and the literal compact-sine flow differs by0.256577. The native endpoint-current derivative error is6.31e-11; the separate gradient-lift U1 ordering trace is0. These do not replace the original sine flow or its generally nonzero ordering divergence.

Crucially, the actual Coulomb/harmonic source contribution is **not zero**: its measured scalar angular coefficient is `-0.0341872`, and the independently differentiated source agrees to8.11e-11. This term acts on the photon/orbital factor, not as an extra quaternion spin matrix. Its low-energy Coulomb/boundary part cannot be removed by the normal high-frequency clock. The receiver must retain its physical orbital/photon connection and source-current graph, or prove its actual projected tail small. The bare A=0 core probe is not evidence for vanishing in a radiating packet.

The legal far-reference source export additionally selects integer eta18 and finite alpha2^371 at `L=5*2^95+2`, using all64-safe Gram minimum1/40 and B² upper2. It gives `c_eta^L=0.9789124062`, four-mark perturbation1.52955e-5, core q/2 CT Gram lower0.479120 and core generator error6.25010e-5. After paying the normal score/link/exterior debt, the conservative requirements `G_B>=1/4`, `G_X>=1/8` make the normalized photon source perturbation at most4.32622e-5. The logarithmic q6 export is finite: normalized power-of-two upper exponents are `[4,427,850,1274,1698,2122,2547]`. These are full Cartesian/all15 source jets, not a total J graph or a replacement for the receiver's actual frozen projected Gram.

This refinement removes a false local harmonic-A obstruction without hiding the genuine low-photon/Coulomb boundary. The selected successor remains the actual receiver's original-law physical connection/current/tail contraction. A failure of that low-photon boundary estimate would require a genuinely different coexact photon cloud, not another end-spin calibration.

The whole-Haar source diagnostic also contracts the supplier's actual120 Cartesian jets with all THREE original physical Phi vector tangents, holding the outside field/reference fixed. It independently finite-differences those physical source directions and consumes the resulting Lie derivatives through the full exterior CAR at the one actual initial metric. Its exported matrices are explicitly labelled **primitive source coefficient derivatives**, not `P_W J_phys P_W` or a total angular certificate. They supply the missing literal source-flow input to the Noether contractor; no cross-axis phonon phases or bare spin factors are invented.

## 16. Literal source-valued SAME-neutral physical Noether contraction

For a RAW CAR source `B=M(Phi,U,A)c†`, compare the charged and neutral histories with the **same** actual neutral state. The differential operator identity is

`J_c B - B J_n = creation(Y M + L_A M + retained Gauss/harmonic scalar boundary)`.

Every canonical field/matter derivative on the neutral state, the E–A angular noise and its symmetrized ordering term cancels operatorially. Its variance has not been dropped, and `J_c` alone is not replaced by a coupling-independent bounded matrix. The remaining harmonic momentum shift and longitudinal end-charge insertion are nonzero scalars multiplying the coexact clover field. Exact coexact commutation belongs to this RAW source, whose real electromagnetic phase depends only on Coulomb and harmonic coordinates. An independently filtered `F_h M` has the supplier's nonzero coexact filter derivative; it is a localization comparison, not the frozen RAW hardware.

The literal native witness uses full non-Clifford Phi, noncentral SU4 links, all12 CAR modes on three sites, the complete bar4 reference, and charge sectors6/7. Its SAME neutral state retains an actual native Peierls exterior propagator depending on coexact and harmonic A, the complete Gauss phase, finite-energy coexact/harmonic Gaussians and a one-photon factor. It uses the original `primitive_flow` sine plaquettes and retains its finite-differenced ordering divergence on BOTH legs. At UE1,1/4,1/16, the source identity errors are9.65e-11,7.11e-11,6.31e-11; neutral angular noise norms are0.074525,0.219967,0.259668, and the nonzero scalar boundaries are−0.0144405,−0.0374481,−0.0234418. Coexact source-phase errors are below2.61e-17. This is an operator-identity/finite-state witness, not an interacting-ground spectral theorem.

For bounding only, `sin(theta)=sinc(theta) theta`, `|sinc|<=1`, hence `K ||curl A psi||²<=<psi,H_EM psi>`, `K=c²/(2UE)`, pays the original sine clover without replacing it. `relative_photon_noether_graph` retains sqrt(UE) logarithmically and uses the ACTUAL prepared photon graph. With rotation speed boundR, phase vectorw, spacinga, frozen normn0 and positive coexact curl singular lowerq_min, its coefficient is

`[8 R ||w||/(a c n0) +12 sqrt(2) R/(a c n0 q_min)] sqrt(E_ph) sqrt(UE)`.

The second term is the genuine scalar-orbital coexact A correction. Coulomb/harmonic source and reference terms have not been declared zero. No soft photons are filtered out.
The large harmonic coordinate of a normalized broad R³ packet is not assumed small. On every support/halo edge, the RAW phase `exp(i chi_x−i r_x Ah)` cancels the original Peierls harmonic/longitudinal connection EXACTLY: `r_y−r_x=e_j`, `A_L=chi_x−chi_y`, so `exp(i(A_L+Ah))*phase_y/phase_x=1`. The common far-reference phase is left symbolic rather than separately approximated. Thus only AT appears in the orbital correction, while the genuine harmonic momentum shift remains inw. The embedded consumer exercises the covariant equality and rejects the missing-harmonic-phase control.


The original mixed weighted graph is supplied by `weighted_photon_curl_graph`. Put `rho=sqrt(1+sum Phi²)`, actual `Q=H−E_min`, and `|H_light|<=B_L rho`. The separately proved original colour lower bound and phase IMS give

`||rho curl A psi||² <= (2UE/c²)[||rho²psi|| ||Qpsi|| +<rho²> +B_L<rho³> +(chi/2)||psi||²]`.

The true H graph and weighted moments are inputs, not products of unrelated first moments. The full-OriginalMicPhoton consumer supplies these for its normalized paired-Bloch photon state; a fictitious normalizable ground state on all of harmonic R³ is not inferred.

## 17. Physical Phi jets, whole-Haar source and original weighted bad tube

On the admitted Clifford core, the ordinary Cartesian physical-vector tangent is independently solved from all15 mass Gauss coefficients: `D Phi=i[S,Phi]`. No matter spin was appended. The native consumer now tests all three tangents against the actual one-body Phi current, full-band divided-difference filter derivative and fixed-outside-RIGHT raw-source derivative. Errors are respectively at most2.37e-16,7.27e-15,8.68e-10. The filter Ward witness is global; it is not a certificate that its local reference tail vanished.

For the actual smooth whole-Haar source, change integration variables by the physical Spin rotation. The derivative of `s_eta` in every traceless direction is zero at a centre, whereas that of a thin fundamental link is not. A whole-group scalar Hessian majorant is `4 eta+48 eta²`. A radial compact physical profile has total variation along a coordinate path at most2. Consequently `physical_orientation_lie_majorant` controls its boundary Lie jet by `(4+8 TV*(4eta+48eta²))*rms_node`; it does not accumulate an unsuppressed fundamental derivative at every far-word link. For screeneta18/alpha2^371, the actual legal radial maximum word is `D+60`, not the scattering word `D+2`. Its all64-safe point constants remain Gram_min1/40 and B²upper2. The routine reselects alpha and retains the exact far-word attenuation through `log1p`, rather than rounding `tanh` to1.

`orientation_core_point_source_bound` additionally uses the WHOLE-group `|Ds_eta|<=4eta` HS bound. Centre characters telescope before distances are estimated. For the canonical q/2 matrix with HS norm1, its source-value error is `(2+8eta L)*rms_node`, and normalized state comparison is at most twice that value divided by the ACTUAL frozen source norm (with its actual creation factor). This is a source-vector bound, not a crossed two-detector Gram Loewner claim. The harder physical-Noether derivative, normal displacement and mixed photon graph remain separate.
At legalD+60 the executed source-value bound is2.1479991e-21 in canonical HS convention (8.5919961e-21 for the four primitive marks), normalized vector comparison8.7899587e-21 usingn0=0.4892165624 and creation factor1025/1024. The independently derived physical first-Lie upper is3.8520678e-47. The old6.25010e-5 core chart bound remains conservative; it is not mislabelled a full physical Casimir. In particular the centre's SECOND angular derivative is nonzero: `s_eta(exp(it S_i))=tanh(2eta cos(t/2))`, hence `s''(0)=−eta/2 sech²(2eta)`. Its link `sum(df)²` contribution must be retained together with the true two-Lie/orbital/EM graph.


`physical_raw_noether_geometry_majorant(64,1)` supplies an explicit `CY=2^64` for q0–q2 of the **unprojected** original local physical vector field. Its active bound includes60 Phi,45 compact-link and3 electromagnetic coordinates per site in a145³ halo box. Each output uses fewer than4096 inputs (64 plaquette links times15,12 transports times15, seven Phi blocks times60); two jets cost at most the square of that count. Matrix products have length at most4, generators have norm at most1, sine derivatives have modulus at most1, admitted core Phi components have HS norm at most2, and the velocity is bounded by64. The explicit arithmetic coefficient `16 ceil(sqrt(108*145³))*4096²*ceil(4096*(1+64))` is below2^61; the selected2^64 includes reserve. This bounds coefficient growth, NOT total angular-noise variance. No inverse-cometric/Gauss projection or smaller Hamiltonian domain is used.

Thus the source-valued good-tube first and second angular costs are `(M2+M1) CY*(2J) Lambda^(-1/5)` and `(M3+M2+M1) CY³*(2J) Lambda^(-1/5)`. The bad second word costs `M2 CY² ||(1+rho²)P_bad psi||`, with harmless numerical reserves covered byCY. The original normal owner pays them using its GLOBAL quartic FORM,

`||rho²u||² <=(2/A4)[Lambda^-1 ||Q^.5u||² +(CD+2)||u||²]`.

For its actual neutral ground/fibre state, `z=(1−WW†)Omega`, `||z||<=d/G`, `||Q^.5z||²<=d²/G`. Choose a gauge-invariant bad multiplication OUTSIDE the complete2epsilon support ofW; then `P_bad W=0` and rho commutes withP_bad. For Lambda>=G, the weighted bad bound is `(1+sqrt_upper(2(CD+3)/A4))*d/G`. A bad partition cutting into the cutoff annulus must separately pay its normal Agmon tail. This proof does not multiply first moments or assume a rho4 operator norm. Photon-created/propagated source states require their actual weighted graph/source-tail consumer, not automatic inheritance of the ground formula.

## 18. SAME legal-N radial source comparison; unchanged scattering hardware

`--embedded-radial-receiver` materializes the supplier's RAW41³ coefficient table: sigma0.25, bandwidth2.4, radial guard radii10/17, normalized before any guard. All68921 coefficients are retained and embedded around `(D,D,D)` on legal `Nstar=5*2^96`, reference `(0,D,D)`. Support means Chebyshev RADIUS20, not20 sites. Its word upper isD+60. A53³ literal-edge halo measures the original scalar orbital operator after the cut/wrap embedding: all three `||L_i eta||` are4.25838e-9; `||sum L_i²eta||` is7.60384e-8; the SU2 orbital algebra defects are2.30318e-10. No native41 normal/BO constant was transferred. All six derivative halos have Euclidean radius at mostsqrt(3)*26=45.034, inside physical plateau48, outer64. A conservative six-orbital-jet ceiling is2^50.

The RAW eta removed by the supplier's guard has norm9.23029e-5. This is measured coverage, not the filtered carrier's separate0.000202405 guard error. The unchanged primitive beta ratios are `[1,.3i,−.2+.4i,.6]`; their one overall matrix normalization gives canonical `Q=sum beta_i q_i/(2||beta||)`, HS norm1. Complete bar4 stays inside the source. No beta ratio, detector coefficient, outcome or rank is changed.

The auxiliary129³ Bloch contraction uses the fixed admitted Gamma trialg0=0.4187253908699568, gs=0.0632221974274214 and the FULL particle projector at every momentum. Bandwidth2.4 prepares eta; it is not a Hamiltonian cutoff. Its actual fixed-quartet particle-band norm squared is0.5. Tiny exterior probabilities are squared from an SVD factor of the actual16-component all-reference state, not a cancellation-prone sum of signed Gram terms. Analytic strip0.5/projector majorant8 pay all nonzero periodic images: local-state alias bound9.49850e-8. The finite-nativeGamma/root discrepancy, normal transfer, source-value comparison and actual source-flight propagation must also be contracted before a true interacting localization probability is claimed.
The positive-band first-energy comparison is computed from the original local mass/Wilson and sin stencils applied to the embeddedeta. Since `Tr(Q† Gamma_mu Q)=0`, its normalized particle energy RMS equals that ofeta, and the actual all-toron metric enclosure `||G||<1/2` gives0.173445 as an upper comparison; adding the selected incoming photon frequency below0.15 gives0.323445. These are Gamma comparisons, NOT automatically the full-original SOURCE Q graph: the source-flight consumer pays the original commutator, normal/high and occupation costs. The SVD exterior probability is1.99857e-32; the periodic-image state error, not that tiny sampled tail alone, controls legal-N localization.


This comparison characterizes the SAME source family on the SAME legal N. It does not replace the canonical fixed19-site scattering preparation, actual GLOBAL2x2 G0, its whitening/cloud debt, or its two frozen detector vectors. There is no nativeNstar FFT. Static C-real source halfHS requires the actual pairedκ domain; a fixed movingκ is not silently made C-real. For a scalar/individual photon input, CCR plus RAW coexact commutation gives its positive norm lower bound; for crossed detector indices the cloud matrix need not be positive and the existing off-Gram debt stays.

The new scoped command is `python noncompact_screened_gauss_dressing.py --size 3 --screen-eta 4 --loop-samples 400 --normal-noether --embedded-radial-receiver`. It consumes the literal full-CAR SAME-neutral identity, original Phi Ward jets, explicit original vector-field geometry, recomputed legal far-word Haar core, actual embedded orbital halo/guard and Bloch localization comparison. The stable topic binds its natural exit, source/input hashes and real selected-file Pyright result. Selected successor is the physical/source-flight consumer's common-history original-law contraction; the alternate is an explicitly self-consistent altered cloud only if these paid costs fail.

The graph helpers accept arbitrarily large exact integer/rational bounds and combine positive terms logarithmically. Full-current inputs such as2^2001 energy or2^1198 weighted moment previously raised Python float-conversion overflow; the repaired path retains finite log bounds and an extra-bit dyadic upper exponent. A linear-value field is `None` when outside IEEE range, with its logarithmic field supplied; it is never a fabricated zero graph. UE remains explicit through sqrt(UE), and RHS ordering/chi terms are retained.

