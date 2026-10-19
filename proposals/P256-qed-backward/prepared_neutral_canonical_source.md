# Original SOURCE on the shared prepared native state

`CanonicalPreparedSpatialChargeTransfer(prepared, charge)` consumes the ONE previously prepared `PreparedSpatialGaussQuantumSlice`. It changes the source, not that owner's Hamiltonian or neutral preparation. The plus view uses the same neutral array; the minus view uses the existing full PH/Hodge map, actual Ref0→1 and bulk4→3, not a fresh preparation. Historical cached source bytes and signatures remain immutable controls.

## Authority and clean new regime

The authority is `noncompact_screened_gauss_dressing.screened_marks`, specifically primitive labels `I, Gamma2 Gamma3, Gamma3 Gamma1, Gamma1 Gamma2`, its signed `source_paths` and its product of per-edge centre marks. The new producer calls that original function rather than reconstructing a fitted matrix. Contract the four primitive marks with the fixed raw quartet `[1,.3i,-.2+.4i,.6]` and divide globally by2. No normalization of the quartet, per-column rescaling or output-derived adjustment is used.

For each path edge y→t, its scalar is `psi1(F_y† U_y→t F_t)/psi0(...)`. Negative steps use `U_target†`, in the original fixed axis order, and all interior frames participate. The scalar of the whole tree is a different off-core functional. The native harmonic Weyl factor is `exp[-i theta.(Birth-Ref)/N]`; its displacement is unwrapped, so replacing the signed colour word by a modulo-N winding is not valid. The Coulomb source has zero coexact derivative.

Historical A controls used a whole-word scalar and the boost-like `Gamma0 Gammaj` quartet, plus an incorrectly oriented modulo z-word. Their state preparation and Hamiltonian observations remain their own exact finite scopes; their source/Gram results are NOT observations of this original-source regime. The new class overrides literal/source operators, Fourier words and creator Gram. `source_from_descriptor` retains its explicitly native N3 body projection, not legal-N geometry.

The native chart still has only two Phi normals, two colour normals, complete52 coexact+3 harmonic modes with EM total degree0/1, two retained light orbitals and six heavy states. The source is evaluated by the original field producer BEFORE compression. This does not turn that chart into full Haar/Gauss/MinSet or supply the omitted guarded mode. `prepared_guard_colour_chart.target_source_jets` exposes the corresponding original product-per-edge first/second jets, with independent endpoint/interior fields.

## Genuine source norm: comparator rather than arbitrary low energy

A neutral low-energy Markov estimate alone cannot freeze the light sea: the large field window admits an entire light band. Instead compare the actual filter with a heavy/field filter acting on an EXPLICIT centre negative Slater sea. Bounded light/reference/Coulomb terms contribute `C_N/L`; the matter-Gauss/Berry first-order terms additionally require actual comparison-window momentum and graph bounds. For a fixed smooth compact f, Duhamel/Fourier comparison gives

`||(f((H-e)/L)-f((H_field-e)/L)) psi|| <= C_f,1/L * sup_t ||(H-H_field) exp[it(H_field-e)/L] psi||`,

with the appropriate weighted integral if the graph bound depends on t. It is the explicit trial and these actual graph bounds, not an assertion about every low-window ket, that retain the comparator sea. Phi/colour normal localization then controls the difference between the actual source and its centre functional through the actual per-edge jets. FullRoot beta, charged source tails, [H,gamma] and exterior leakage remain in that comparison.

At a Clifford centre the actual light onebody h anticommutes with the chirality matrix. The four ORIGINAL quaternion labels commute with it, including their full spatial embedding and complex eta. Thus, for the four source maps T_a and the full positive projector, `Tr(T_a† P_+ T_b)=Tr(T_a† T_b)/2`. This uses the actual complete Ref4 trace, not Spin3 alone. Cold tree attenuation is retained: each centre ratio has modulus below1, and the complete path product must be bounded at the actual legal word length. The CLI's local Clifford trace with no path is only the bare point comparator, not that global attenuation proof.

For actual normalized filtered states let V contain the four source columns and V0 their true comparator columns. If `||V-V0||<=epsilon`, then `||G-G0||<=2||V0|| epsilon+epsilon²`. This is a quantitative route to a genuine raw-quartet lower, after the filter/source/normalization comparisons are proved. It is not currently an admitted MinSet bound.

Because the canonical source commutes with all coexact creators, `G_creator=G_source+G_Ng` is positive-semidefinite above the SOURCE Gram. Actual raw normalization is `c†G c`, and the two-excitation components outside native EM degree0/1 are retained in the true creator norm. No scalar Gram, frozen photon vacuum after filtering, or input whitening is assumed. Ordinary occupation bounds, mixed harmonic-Pi and driven hardware, all-normal Eex/coercivity and union Agmon/exterior proof remain separate obligations.

## Runtime scope

The actual CLI loads the frozen raw-state NPZ, constructs both new source charge views, measures their true Hermitian four-basis Gram and current/contact, and checks that the owner's historical operator signature is unchanged. The execution receipt records its actual exit, source hashes, input state, wall time and stdout/stderr. Neither this scoped run nor the local Clifford trace is a new MinSet state, physical SO3 closure or Source19 Born probability. Campaign and fundamental quantum-law/parameter debt remain OPEN.

## Current public before-projection API

The current class overrides `source_coordinate_mark(fields, links, theta, profile=None, quartet=None, a=None)`. Inputs are the FULL arrays `[V,4,4,4]` and `[V,3,4,4]`, `theta=N A_h`, optional complex birth profile/raw quartet and the noncompact `[V,3]` Coulomb argument. It returns all `[V,4,4]` birth/reference marks. The old two-endpoint/whole-tree-only signature is intentionally not retained: it cannot determine the original interior per-edge product. No historical boost functional remains behind this current entry point.

`source_operator` is inherited from the charge-view base and calls `self.literal_source_operator` virtually; that override is canonical. The inherited descriptor method calls `self.source_operator`, hence is canonical without an instance replacement. The minus constructor likewise invokes the canonical literal builder virtually. The plus constructor installs its canonical B explicitly. Four-basis Gram compares against the view's actual source state, never the historical owner's B.

The changed current provider CLI `--descriptor` exercises that actual dispatch with the frozen19-complex-profile descriptor, both charges, raw Hermitian Gram, full-coordinate producer, current/contact and true photon-creator normalization. Its natural-exit receipt is `local://issue222-canonical-source-current-api-receipt.json`; the earlier caf22 default-source receipt remains a different immutable source-byte scope. The spatial/degree-two defects are not erased by this API correction.
