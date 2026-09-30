# Independent quantum/constraint integration review

## Revision and coverage

Governing work item: integration, plan issue-222-qed-backward-20260930 at cc4ce18061fac9fcbd05a1e4ba70d2616b78ebe4766316992133bd001d4dc864; template general at 89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe. Branch research/222-qed-backward, HEAD/baseline 14e0b08f6099698b6d117840e65f86cbe70d4efb. The new proposal files are additions outside that committed baseline. Read issue #222 and the complete constraint.py, constraint.md, quantum_supplier.py and quantum_supplier.md; read lattice_compton.md as the joined construction summary only. Read integrate.py's original four-CLI dispatch and the scoped constraint/quantum results. This report does not independently review compton.py, lattice_compton.py, reference.md, later spin-transfer artifacts, or later integration revisions.

Reviewed artifact identities recorded by the parent-run results.json:

|Artifact|SHA256|
|---|---|
|constraint.py|dc20f1c15ecffb3fc5a117652d9f0a3c5932d75375909478e36c123820c2c687|
|constraint.md|ba4a168e62faf894f4c4ae7bb5d96e068fb72f02a887e08c478bc0e2770d3ac7|
|quantum_supplier.py|204726434b0e9dd695c1fd1412c6babcfc46480e51fb9e4a2408a0be82e47695|
|quantum_supplier.md|f884e3de2f46586e88a512fadf37dc59a7dfe6edc80a13825e10848e5f115e9d|
|lattice_compton.md (summary only)|6d8019e3b328c349525984f105d41474338ec722cd6c974128e0593f2b0ec985|

These hashes identify the recorded smoke snapshot, not endorsement of unreviewed siblings. Parent reports natural exit 0 in 5.25 s for four real CLIs. results.json records Python 3.12.3, numpy 2.5.3, scipy 1.18.1, sympy 1.14.0 and the scientific-environment executable. I inspected its scoped outputs; I did not rerun that integrated calculation, a suite, builds, lint or formatting. An AST positive control located both many_body_report consumers in report(). No LSP device was available in this read-only review, and no editor/source mutation was attempted. Live integration gained an additional CLI during review; that later sibling is explicitly outside coverage.

## Resolved finding (P2 at original smoke snapshot)

quantum_supplier.py, photon_modes(): the absolute 1e-14 transverse-eigenvalue threshold drops legitimate long-wavelength oscillators. Trigger: default positive coefficients and momentum [1e-8,0,0]. Independently executed the permitted small scientific falsifier using the parent scientific Python, without running report(): 4 sin(k/2)^2 = 1.0000000000000001e-16, but photon_modes returns physical_rank=0 and frequencies=[]. Positive control k=0.05 returns rank=2 and frequencies 0.049994791829424665 twice. The exact matrix at the failing input is diag(0,d²,d²); both modes are physical, not numerical longitudinal noise. Consequence: the callable violates the nonzero-momentum two-mode claim precisely on approach to the soft/continuum limit. The existing radii 0.05, 0.5 and 1.5 all pass and cannot exclude this failure. Smallest correction: separate exactly d=0 and use a scale-relative spectral criterion, or construct the transverse subspace from nonzero d, without an absolute eigenvalue floor. This does not refute the displayed analytic rank-two theorem, nor invalidate the recorded finite-radius values; it corrects their executable extension. Parent owns correction and its scoped confirmation; any fixed revision needs an updated hash, not silent inheritance of this report's snapshot verdict.

## Constraint derivation witnesses and earned scope

- The unusual {x,p}=-delta sign is explicit and consistent: p=i partial, exp(-ip.x), [p-qA,p-qA]=-iqF, and {pi,pi}=-qF. Conventional c=-p is a coordinate change, not a charge reversal.
- For minimal Q=pi.psi-m psi5, the bosonic momentum bracket contributes -q F psi psi; odd brackets contribute -i pi²+i m². Their sum is -2iH with the displayed spin coefficient. The symbolic constant-field computation compares this to an independently written H, rather than declaring H=QQ. Its missing-spin residual has -2q F01 psi0 psi1 and all five analogous components; QH vanishes whereas the omitted-spin QH has nonzero pi.F components. Arbitrary smooth F needs the stated graded-Jacobi/Bianchi argument; the executable itself verifies constant backgrounds only.
- Clifford matrices give {psi_mu,psi_nu}=eta_mu_nu, {psi5,psi5}=-1 and vanishing mixed anticommutators. Qhat=i gamma5(D-m)/sqrt(2) squares to [P²-m²-q sigma.F/2]/2. The differential Landau-gauge polynomial probe retains the nonzero field commutator and derivatives, so it is stronger than squaring a constant momentum matrix. Q=0 remains necessary: H=0 alone would overcount physical first-order solutions.
- With F12=-B, the same square gives E²=m²+Pi²-q B Sigma_z and g=2 in its minimal, low-field nonrelativistic magnetic limit. The recorded same-orbital expectations E+=sqrt(6/5), E-=1 and omitted-spin sqrt(11/10) agree with the derivation. The code constructs those energies from the analytic Landau formula; it does not independently solve a Landau eigenproblem. This is a conditional minimal-ansatz coefficient, not a symmetry-only determination, measured magnetic anomaly or fundamental g.
- The cubic deformation quantizes consistently: f maps to -i sigma.F/2, so -ia f psi5 maps to the claimed Pauli T with kappa q/(4m). Classical residuals include the pi.F psi psi5 terms and the quartic coefficient kappa²q²(F01F23-F02F13+F03F12)/m². They cannot be recovered by simply multiplying the minimal spin term by 1+kappa. At operator level [T,D]-2mT-T² is necessary. The constant-field differential probe checks that full square; its field-gradient term for variable F is analytic, not numerically exercised. Quantizing Q first avoids falsely equating naive quartic substitution with the quantum Hamiltonian.
- The minimal first/second-order resolvent map retains both K12={V1,V2} and variation of the B numerator. LS=1, or the displayed VG0-S0{D0,V}G0=-S0VS0 identity, derives the two first-order orderings with external Q-state selection and imported boundary prescription. The recorded rational-matrix contact is -I, so it is not silently zero: the full mixed coefficient agrees, while deletion of either contact or endpoint fails. This probe is off-shell and contains no photon spacetime gradients or actual Compton kinematics; the general differential/amputation extension is the analytic derivation, not its numerical scope. For the anomaly K=B L and S=K^-1 B have the correct noncommuting order and changed contact/right-endpoint terms.

## Quantum supplier witnesses and earned scope

- On the tree, fixed N=2 and sum r=2 make every physical flux uniquely r_j-n_j. Exactly six occupation/flux states result, |E|<=1. Allowed covariant forward and reverse hops stay inside them. The larger 750-state auxiliary space uses noncyclic truncated shifts, with both unitarity defects correctly reported as 1. Full-space legal commutators and physical-outgoing leakage are actually computed, and physical boundary-clipping count is zero; this establishes the exact selected invariant sector, not arbitrary truncated-rotor unitarity or plaquette closure.
- Legal hopping shifts flux and occupation together. Bare hopping is evolved unprojected at E=0, so the adverse Gauss violation is not hidden by projection. Recorded legal departure probabilities are about 0.672 with zero leakage; bare <sum G²> is 1.3566596699059545 (fermions) and 1.378441115539942 (hard-core bosons). The continuity calculation evaluates full sparse commutators: n_dot=div E_dot, consistent with J=it(T-Tdag), n_dot=-div J and E_dot=-J on this plaquette-free tree. It is closely related to Gauss conservation, not an independent derivation of the statistics.
- The six-move exchange product uses actual transition matrix elements and returns occupations AND flux to their initial values. Its CAR -1 and hard-core-boson +1 are closed-path, basis-rephasing-independent signs. Full 16-state CAR identities verify the imported representation; they do not explain why that representation exists.
- M=d²I-dd^T has a longitudinal null vector and two degenerate transverse eigenvalues for d!=0. Canonical oscillator quantization then gives omega=sqrt(U_E K)|d| and theta variance U_E/(2 omega). Nine finite-radius outputs genuinely diagonalize M and exhibit high-radius anisotropy: at radius 1.5 the axis frequency is 1.3632775200466682 while the diagonal is about 1.4535624963809635. They are not numerical evidence that a compact phase exists. The numerical soft-rank defect above must be fixed separately.
- Continuum transfer theta=q a A_cov and E_rotor=a² Pi/q preserves the symplectic term, makes Gauss div Pi=q rho_number, and yields coefficients U_E a/q² and K q² a. Canonical Maxwell matching therefore requires U_E=q²/a and K=1/(q²a); A_cont=sqrt(a) theta_lattice/q has variance 1/(2 omega) under that same matching. This is a derived conditional normalization requirement, not a measured residue, emergent quantum postulate or independently derived charge.
- The neutral 8³ inverse-Laplacian response reaches its Gauss assertion (maximum residual 6.765421556309548e-17) and two energy forms agree at 0.224841852063406. The asymptotic cross coefficient U_E a/(4 pi)=q²/(4 pi) follows analytically under Maxwell matching; it is not measured by that one finite-volume calculation. Continuous real-flux minimization, compact integer eigenstates, monopole/confinement physics and Coulomb-phase stability are expressly distinguished.

## Strongest false positive and the actual join

The decisive false positive is a charged hard-core bosonic system: it preserves precisely the same Gauss law and current continuity and can use the same two harmonic photon modes and Coulomb coefficient. Its spinless dilute band also has a Thomson-like charge/inertial-mass response. Those successes cannot identify an electron. Its closed exchange gives +1 instead of -1; the CAR version fixes that sign only by importing CAR and still has no intrinsic spin, Dirac antiparticle sector or magnetic moment. Independently, graded worldline closure supplies single-particle spin algebra only after importing odd variables, their brackets and Clifford quantization; it neither supplies interparticle CAR nor quantizes a photon.

The joined lattice summary does not claim otherwise. Same-link Peierls band/vertex/contact consistency and the inertial/rest-mass distinction are substantive conditional interfaces, stronger than juxtaposed analogies. Altering a Wilson mass while altering its one- and two-link interactions together is a material within-model construction, not just renaming an input. Nevertheless the scalar rotor matter has not been shown to generate the imported spinor band; the compact lattice has not been shown to generate its deconfined harmonic photon endpoint. The constraint slice explicitly calls itself a Dirac reencoding. Thus the acceptable result is a coupled calculation under declared quantum/CAR/Clifford/gauge/parameter assumptions, with actual conditional maps and explicit debts—not a microscopic electron/photon origin or an independent explanation of QED from an unselected deeper law. This review reads the joined derivation summary; it does not certify its scattering code, rates, cross sections or runtime output.

## Exact acceptable claims after the rank correction

1. Given the stated minimal graded algebra, Q ansatz, external U(1) field and Clifford quantization, closure and the first-order state condition bind the Dirac numerator/minimal vertex/leading g=2 magnetic term, and the resolvent identity binds second-order contact and endpoints to the first-order channels.
2. Given rotor and site Hilbert algebras, the specified tree Hamiltonian has an exact six-state quantum physical sector with Gauss-preserving correlated hopping, derived continuity and nontrivial dynamics; the actual closed exchange distinguishes imported CAR from commuting-site hard-core bosons.
3. Given the noncompact harmonic and canonical quantum approximation, nonzero physical momenta have two transverse oscillators, a controlled anisotropic lattice dispersion, and the displayed conditional Maxwell residue/Coulomb normalization. Compact phase, helicity beyond the rotational continuum limit and interacting charged-state dressing remain open.
4. These are partial interfaces useful to a declared conditional Compton construction. None generates CAR, quantum postulates, the spinor matter band, measured mass/charge, a common matter/photon light cone, radiative corrections or a microscopic origin.

Final scoped verdict: the one reproducible soft-mode defect was corrected and specifically exercised by the parent. No outstanding scientific inconsistency was established in the assigned derivations under their explicit conditional scopes. The original smoke hash and numerical record remain historical; the corrected source is identified in the addendum below. No merge, issue closure, ontology endorsement or scientific promotion authority is implied.

## Correction addendum

After the original review, the parent removed the absolute spectral floor, added soft nonzero/zero-mode controls, and exercised the failing path. Reviewer inspected only changed photon selection and report controls. Reported parent observation: rank2 and omega=[1e-8,1e-8] at k=[1e-8,0,0]; rank0 at zero. This resolves the finding without changing the analytic photon theorem or broader input debt. Updated quantum_supplier.py SHA256: be51fae71eb793ef07f386de04db6c6aae5ad6511acb05ade8a9a54e03a833fd. No updated integrated-suite execution is claimed by this addendum.

## Narrow typing-equivalence addendum

Parent final LSP diagnostics exposed annotation/stub issues in `constraint.py`.
Reviewer inspected only the invalidated lines: `psi: list[Grass]` leaves the
five singleton exterior monomials identical; `sp.S.NegativeOne` is exactly the
same `-1` multiplication; the local zero-Matrix accumulator iterates the same
ordered four `gamma[mu]*momentum(mu,vector)` terms as the former `sum` without
modifying gamma or the input vector. These are mathematically equivalent typing
corrections; no new scientific finding or reviewer runtime validation is claimed.
The parent observed zero diagnostics on the corrected explicit file path.
Updated `constraint.py` SHA256:
`ce60f9e0c2abeacf7f1692fe22521f3a40ea3a05f3c54a70e9c1a7e0188302de`.
The original hash stays historical; the final integrated receipt supplies
current runtime evidence. The prior scoped scientific verdict remains correct.
