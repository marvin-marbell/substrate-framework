# Ambient weak Hamiltonian and one once-prepared neutral state

`prepared_neutral_ambient_weak_state.py` supplies an actual positive-form native weak Hamiltonian and literal SOURCE-created field functions, rather than a point source matrix renamed a field state. Its completed first execution is a finite quadrature/Galerkin construction, **not** the legal full-H state, an actual ground state, exact Gauss-kernel dynamics or a closed physical rotation.

## Exact native construction

The executed inputs are `N=3`, `Lambda=2^24`, `chi=Lambda^(-1/2)`, `UE=2^-20`, photon `c=1/8`, `K=c^2/(2UE)`, normal Gaussian precision `sigma=2^-13`, compact normal radius `.012`, and FAR `(0,1,1)`. The mean is the selected original-heavy-frame/light-improved band, with colour links identity. The initial diagnostic SOURCE profile is the normalized even `exp(-|x-(1,1,1)|^2/.65^2)` profile. It is **not** the original SOURCE19 profile or the selected narrow SOFT profile. Native photon `c`, mode IDs, radius and volume are diagnostic inputs; they are not a retuning or delivery of the legal family.

The field chart includes all `90V=2430` normal coordinates and `15V=405` gauge coordinates, with the full raw `105V=2835` Jacobian, metric, inverse, Haar density, normal/gauge cross terms and vertical quadratic charge. The wave has `J^(-1/2)` and the common normal Gaussian with a smooth radial cutoff. Its density is integrated against `J dq Haar`: exactly one density cancellation is used. Seven normal polynomials, constant plus six fixed full-normal directions, multiply each of the two neutral banks. No unused normal coordinate is frozen.

All `2(V-1)=52` real coexact coordinates and the separate three harmonic coordinates are sampled. The real coexact legs are `e_r sqrt(2UE/omega_r)` with `omega_r=c sqrt(lambda_r)`; a normalized creation wave on the vacuum is `sqrt(2) g.y`. The harmonics are `theta=N mean A` with free kinetic `(UE/N) p_theta^2`, not an invented oscillator potential. The positive coexact kinetic plus magnetic potential already includes the vacuum zero point; it is not added a second time.

The massive sector uses the original **bare**, EM-neutral `h_F(Phi,Ucolour)`, with its fixed positive occupied54 cluster in `-Lambda h_F`. It is not the improved light operator. The light sector uses the full improved108-mode microscopic operator and the once-selected negative54 cluster plus the fixed positive R4 quartet. The two neutral banks are sea54/Ref4 and sea54+R4/Ref0, with the same light+Ref number58. The physical reference charge is `Nref-4` in both channels and its static onsite energy remains zero. The charged fibre has all216 PLUS and232 MINUS vectors; the original quaternion/global2 SOURCE is evaluated on the actual fields and harmonics with the **initial fixed orbitals**. Initial SOURCE functions are held fixed under a moving mean frame; the current physical `B_t` is a separate observable.

The Hamiltonian retains the original quartic, both `Lambda^1.25` alignment and `.7` colour plaquette stiffness, massive sea energy, full light bilinear, full Coulomb `UE rho Green rho` formed **before** projection, and full-cometric positive momentum form. Wick products restore omitted particle-hole/multipair charge norm before compression, including the massive Slater variance. The native Root slice is explicitly the fixed flat-code `P` compression with `P V_beta P=0`; the full coupled Q sector, all eight legal holonomy sectors and encoded odd-CAR/Gauss dynamics are not declared closed. No energy-origin subtraction is introduced (`C=0`).

## Once-prepared state and norm units

Four full-cometric stochastic trace directions per node and fourteen antithetic full-dimensional nodes produce the native14-by14 weak forms. The actual generalized eigensystem of this neutral `H14_MC` defines one smooth compact spectral selector, using its explicitly native spectrum-defined floor/window. The selector is applied **once**, followed by **one global neutral normalization**. There is no charged filter, no per-bank normalization and no pointwise SOURCE recapture.

The completed run reports:

- RAW filtered neutral functional norm squared, **before** the one neutral normalization: `.8920522472031723`.
- `||B_Gaussian Psi_normalized||^2`, **after** that neutral normalization but **before** SOURCE normalization: `.8718600287632787`.
- Prepared neutral bank weights `.5000000067738677` and `.4999999932261322`, not reset to exact halves.

The NPZ `neutral_state` is a coefficient vector with physical native `neutral_Gram` norm one. Its `source_state` is `neutral_state/sqrt(.8718600287632787)`, in the literal created-function charged basis, with charged metric norm one. The `.871860` value belongs specifically to the initial diagnostic Gaussian profile: it is not a SOURCE19/SOFT normalizer or a legal lower bound. Literal `B(q,theta) psi_i(q)` functions define the charged labels; their identity coefficient map is a definition, not a polynomial fit or a proof of dynamical closure.

## Actual scientific execution

Command, proposal-directory cwd, Python runtime:

`/home/administrator/substrate-framework/.venv/bin/python prepared_neutral_ambient_weak_state.py --root-lemma <session-local>/issue222-prepared-MinSet-root-source-lemma.json --output-npz <session-local>/issue222-ambient-weak-state.npz --nodes 14 --kinetic-directions 4`

`local://issue222-ambient-weak-state-execution.json` records natural exit0, wall `224.18710944405757s`, empty stderr, unchanged loaded sources and no unbound source. Source SHA256 is `2f571e66f67c486efbb2fe92969dc689c1a48a1ea7d8559fd0d867ed4b776f47`; stdout SHA256 is `db9ae39105c1489869e73313d5f429068260c910cf2b361fe20a22eebf926c78`. The frozen source archive SHA256 is `f18f2c8cc791b2e9d35dee67040c8d125ce613acd78f68c822bad16c7337059e`; NPZ SHA256 is `eb283954e7ebb13e1b5368deb058214443d9a839b8319161c0306afb2c8b53b6`.

Actual all-heavy/light/Ref number is112, hence SU4 nality zero. The heavy centre phase compensates the light/Ref phase: fibre centre errors are at most `8.32668e-15` and actual SOURCE covariance error `9.65097e-17`. This removes the missing-heavy centre obstruction on this representation; it is only a necessary global-stabilizer consistency witness, not a proof of the full group-averaged state. The actual massive EM finite-difference current is zero, the light current norm `.3741964013501603`, and the bare-heavy versus improved-light onebody defect `1.4543536820431096` rejects borrowing the light operator for the massive sector.

## Failed vacuum-only photon premise and shared successor

Despite integrating all52 coexact coordinates, these fourteen basis functions have **only the coexact vacuum factor**. Exact Gaussian integration gives `P14 a_g^dagger P14=0`; random finite-node correlations cannot be called a captured photon input. The shared successor must therefore retain the exact frozen neutral coefficient vector while adding actual creator functions, individual original `I/Gamma23/Gamma31/Gamma12` SOURCE-word functions, and physical Noether EM partners. Its preparation remains `f(H14_MC) tau`, not silently `f(H_extended_exactEM) tau`. Exact Gaussian/Weyl integration can change the old stochastic vacuum Hamiltonian block; that difference must be observed and reported without refiltering the frozen state.

The separate hard-Fock and quartet-function pairs derive these actual functions and adverse branches. Their operator/function proofs are not an intrinsic-only physical full-loop result. The shared Hamiltonian, true Noether consumer and same-current static binary scattering consumer retain independent source-bound execution obligations.

## Remaining genuinely unpaid premises

Finite normal/harmonic quadrature and stochastic trace error; native SOURCE finite-difference remainders; field-polynomial and full light/heavy particle-hole/multipair dynamical complements; full Root Q/all512 auxiliary histories and encoded odd-CAR control; actual Gauss-kernel construction beyond the centre witness; the shared physical driven flow and legal closed-rotation/source-relative costs; native-to-legal hard-mode/profile/parameter differences; fundamental quantum-law and microscopic-parameter debt. None is erased by this fourteen-dimensional preparation or its native current/nality witnesses.

## Executed-source versus current typing epoch

The scientific execution above remains bound to its exact `2f571e66...` archive and frozen NPZ; it is not re-executed to backdate current bytes. The later A9–17 current-source pass adds only explicit `typing.cast` identities and heterogeneous-container annotations for real Pyright. `local://issue222-prepared-before-typing-sources.zip` SHA256 `f2ad3950193b00f4fa147f49163d0fded5877953fe25568b6022a68afb43c40e` preserves the before bytes. `local://issue222-prepared-current-typing-AST-bridge.json` records each full before/current source hash and the complete runtime AST equality after erasing only typing imports, annotations/type comments and identity `cast(T,x)` wrappers. All physical expressions, operators, constants, output keys and control branches remain in that comparison; no numerical tolerance, profile, state/filter, source law or quantum-law premise is changed.

The actual source-bound current Pyright receipt is `local://issue222-prepared-owned-current-real-lsp.json`; the failing before-typing diagnostics are preserved separately as `local://issue222-prepared-owned-before-typing-real-lsp.json`. These must be read independently from scientific receipts. An equal runtime AST bridges typing-only bytes; it is not an assertion that a scientific run used the later bytes or that an interrupted source epoch was uninterrupted. The first new shared-extension SCI ended at its metadata-seal guard without a sealed NPZ, and its unpersisted epoch discrepancy has no established cause; no successful shared-H runtime is inferred from the original A17 run.

Final current-source observation: actual Pyright stdio received diagnostics for all nine A9–17 files, zero diagnostics in each,1369 document symbols in total, definition/reference RPC and positive parsed AST, server exit0 and source hashes stable. Receipt SHA256 `fc47e6bca0870d52588d371a556b8f8cf31ce3e63f7c110e08f81d3833d47f00`; full runtime-AST bridge SHA256 `b0f3431119263535cd8f3b4d28d3181910cc13776648d0d72d032fe361116925`, all ten selected files equal (A9–17 plus unchanged common-boost module). Current A17 source is `ac16f7a26c7edddc137d9a68f985321d01c369884a36e49dc9d5c3606270503d` and has316 symbols. Direct real Pyright supplies this evidence; the xdLSP device remains unavailable. Neither the original scientific runtime nor the frozen NPZ was rerun or rewritten.
