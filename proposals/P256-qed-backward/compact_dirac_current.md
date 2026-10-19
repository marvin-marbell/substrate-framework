# Compact one-flavour current and path-dressed light-electron consumer

## Source, regulator and scope

`compact_dirac_current.py` extends the actual `compact_dirac_measure.py` measure, not a partition-only determinant diagnostic. It inverts the same Wilson Dirac matrix, inserts its exact link current and contact, and consumes path-dressed open fermion blocks on changing compact photon configurations sampled with the full single-flavour determinant. No determinant absolute value, second flavour, square, root or quenched replacement is used.

The governing plan is `issue-222-continuing-backward-20260930`, revision `bbff677d7b2d3b5ddb68187ca868a67f676af68a0b51e8d2eb49b797d967b2e9`; the pinned general template is `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`. Parent owns integration and scientific acceptance. Existing producers and proofs are read-only suppliers.

The default input is the actual successful `bulk_cutoff_matching_receipt.json`, point **a=.2** in its explicitly changed bare family kappa(a)=2/a, primitive m=.2/r=.12/v=.3, g²=.1. The receipt gives

- selected band m=.11265226021197168, r=.067591356127183, v=.09211559694033435;
- Hamiltonian rotor UE1=.04623528922363761, K1=4.588013299810326;
- spacing a=.2; no substitution of the older a=1 UE1=.4/K1=.7 point.

The consumer reconstructs the actual selected fields from the receipt Gram and the supplier's `SELECTED` matrices, reconstructs their unitary/band with `bulk_frame_compton.reconstruct_frame`, and compares with the receipt band. Relevant actual producer SHA256s and input-receipt bytes are bound. The exact input row, field coordinates, reconstructed frame and executed parameters are included in the output.

The default lattice is **2 x 3 x 3 x 3**, dt=.2, with periodic spatial/photon boundaries and antiperiodic temporal fermion seam. Three spatial sites give sin(2pi/3) != 0 rather than an identically zero sine kinetic term. At a=.2 the hopping scale is manageable relative to the selected mass and dt, unlike the stiff finest-cutoff hopping. This is a finite point, not an infrared spatial box or a continuum extrapolation. The dt refinements below quantify rather than assume the continuous-time correspondence.

The photon action is precisely the supplier's Haar-normalized positive temporal rotor heat kernel with parameter UE1*dt and spatial cosine with coefficient K1*dt. These are declared Gaussian-matching inputs. The nonlinear compact ensemble's speed/coupling are **not** reidentified with the original Gaussian estimates.

## Exact variations of the same Dirac inverse

Let l=(x,mu), y=x+mu, and include the antiperiodic seam sign in u_l. With w_0=d_0=1/dt, w_i=r/a, d_i=v/a, define the forward and reverse blocks

\[
 F_l=u_l(d_\mu\Gamma_\mu-w_\mu I)/2,\qquad
 B_l=u_l^*(-d_\mu\Gamma_\mu-w_\mu I)/2.
\]

They occupy D_xy and D_yx. **B_l is not F_l dagger.** All overlapping blocks on a short periodic direction are added, never overwritten. With link phase theta_l,

\[
 V_l=\partial_l D:\quad (V_l)_{xy}=iF_l,\ (V_l)_{yx}=-iB_l,
\]
\[
 C_l=\partial_l^2D:\quad (C_l)_{xy}=-F_l,\ (C_l)_{yx}=-B_l.
\]

Mixed derivatives of D on distinct link coordinates vanish. The consumer solves S=D^-1 and checks D S=I. The supplier's positive-mass gamma5 Hermiticity proof guarantees det D>0 for every compact configuration; the actual complex slogdet sign is checked +1 and never altered.

For L=log(det D),

\[
 j_l=\partial_l L=\mathrm{Tr}(S V_l),\qquad
 K_{lk}=\partial_l\partial_k L
 =\delta_{lk}\mathrm{Tr}(S C_l)-\mathrm{Tr}(S V_k S V_l).
\]

These derivatives are of log weight, so their sign is opposite the fermion effective-action derivatives if that action is defined as -log det D. For a prescribed source b_l the measured source current is b.j and the determinant Hessian is b.K.b. This is a configuration response; an equilibrium ensemble derivative would additionally involve the appropriate fluctuations and photon/source law. The code does not mislabel the finite raw sample mean as that ensemble response.

Spatial variations satisfy the supplier's actual raw-path identity V=beta*partial_theta h, C=beta*partial_theta² h, with theta=e*a*A and e=1. The existing raw source current/contact and unitary controls execute at this selected receipt point. Physical derivatives are e*a times link-phase first derivatives and (e*a)² times second derivatives. Temporal variations retain **both** the Wilson and Dirac terms of the specified regulator.

## Nonlinear configuration-level Ward identity and wrong contact

For arbitrary site phases chi_x set g_l=chi_x-chi_y. This is the finite local U1 law U'_l=exp(i chi_x) U_l exp(-i chi_y), not a small-photon expansion. It implies D'=G D G dagger and

\[
 V_g=\sum_l g_l V_l=i[\chi,D],\qquad
 \partial_g S=-S V_g S=i[\chi,S].
\]

The exact determinant is invariant on this gauge orbit, so its link current has zero site divergence. Differentiating with respect to every independent theta_l gives

\[
 \boxed{(K g)_l=g_l\mathrm{Tr}(S C_l)
                  -\mathrm{Tr}(S V_g S V_l)=0.}
\]

The program checks the entire link-vector contraction, not only a scalar double contraction or a free momentum Ward component. No symmetrization, projection or cancellation fix is applied. This holds at each retained nonzero compact sample as well as the initial configuration, and supplies a full nonlinear finite-regulator matter response independent of a Gaussian sea approximation.

The adverse computation omits the contact and passes the remaining bubble through the **same Ward assertion and tolerance**. It must actually raise; it is not merely recorded as a different partition value. The sampled probe must have a nonzero reached adverse defect. This lower bound serves only to select a discriminating wrong-law probe, not as a physical lower-effect or coupling gate.

## Dressed open light-electron block and current insertion

Choose a two-link path x --time--> z --axis1--> y, with W=U_0(x) U_1(z). The open block S_xy transforms as exp(i chi_x) S_xy exp(-i chi_y), while W transforms with the identical endpoint phases. Thus

\[
 \mathcal G_{xy}=W^* S_{xy}
\]

is a gauge-invariant **path-dressed conditional light-fermion** observable. The photon Wilson path does not contain the antiperiodic fermion minus sign; that boundary condition belongs in D. Let p.b be the sum of b_l along the two-link path. Its exact source insertion is

\[
 \partial_b\mathcal G_{xy}=W^*[-(S V_b S)_{xy}-i(p.b) S_{xy}].
\]

The second term differentiates the dressing. Along the gauge direction p.g=chi_x-chi_y the insertion vanishes. The wrong undressed insertion omits that term and must fail the identical gauge-insertion assertion. A finite, arbitrary site gauge transformation also checks the inverse covariance and the dressed block's invariance, independently of the infinitesimal identity. Central finite differences independently reach the link vertex, contact, dressed source insertion and determinant source Hessian on initial and final configurations.

This gauge-invariant bilocal is neutral as a whole, as it must be on a periodic gauge-constrained volume; it is not proof of an isolated charged primitive electron. It is neither the neutral hidden Pauli-code marker used by the forward consumer nor a physical external SU4 **B4/bar4** carrier. Those three objects remain distinct. A dressed conditional light-fermion propagator does not establish primitive mechanical spin, a separated compensating charge, or the campaign's full charged-particle construction.

## Exact neutral transfer and quantified finite-step current debt

The inherited neutral-transfer controls execute with the same selected band, lattice and dt, including nontrivial static and time-dependent spatial links and site-dependent temporal holonomies. They retain

\[
 \det D=dt^{-4V}\prod_t\det B_t\prod_x z_x^{-2}\det(I+ZP),
 \qquad z_x=\Omega_x^*,\quad Q_x=n_{light}(x)-2.
\]

The inverse holonomy, n-2 background, finite Wilson normalization and antiperiodic seam are already present in det D; no extra background factor is appended to sampling. Omission controls expose the wrong canonical n trace and removed normalization. These controls are not an assertion that the primitive unfrozen heavy/colour/normal-field Gauss trace has been joined.

For static nonzero spatial links, group the beta blocks of the actual source h as [[W,F],[F dagger,-W]], and construct

\[
 T=\begin{pmatrix}B^{-1}&-B^{-1}C\\-C^\dagger B^{-1}&B+C^\dagger B^{-1}C\end{pmatrix},
 \qquad B=I+dt W,\ C=dt F,\ H_{dt}=-\log T/dt.
\]

This is positive transfer on the same one-flavour one-particle space. Its logarithm is not h at finite dt. The current extension independently compares the first and second uniform spatial-axis link-phase derivatives of H_dt with the source h derivatives at a nonzero compact background. It records relative current/contact discrepancies at dt, dt/2, dt/4, dt/8, and requires monotone decrease on that probe. These are numerical central differences with declared phase increment 5e-4, not certified interval derivatives. Source link/current/contact identities themselves are exact apart from the supplied frame reconstruction tolerance and floating arithmetic. The finite-step discrepancies must remain visible rather than being relabelled exact microscopic-Hamiltonian equality.

## Actual coupled consumer and reproduction

Run from the proposal directory with the project scientific environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/administrator/substrate-framework/.venv/bin/python compact_dirac_current.py --input bulk_cutoff_matching_receipt.json --output compact_dirac_current_receipt.json
```

Defaults: seed=22273, burn=3 bounded sweeps, samples=8, updates=16 links per sweep, phase proposal width=.08. Proposals use the full inherited joint log weight photon_log+log det D, including all transfer/heat-kernel normalizations. Each retained row records actual angles, photon weight terms, the matrix-valued dressed block/insertion, all link currents, and nonlinear Ward/adverse results. The source has nonzero spatial curl, independently checked, and is not secretly a pure gauge variation. Accepted changes and distinct retained compact configurations are required. A raw correlated matrix mean and its finite configuration spread are labelled exactly that; no mixing or error-bar claim is made.

Parent runs final default scientific acceptance and native type diagnostics after all slices land. A local discriminating numerical probe, if supplied separately, is not a substitute for that final receipt. Worker AST navigation positively exercised the actual `dirac_matrix` definition; the worker's LSP device is unavailable, so no worker LSP success is claimed.

### Observed discriminating local probe

`compact_dirac_current_probe.json` records a short run on the exact historical a=.2 row, input receipt SHA256 `9c1dcd2d9456fc073bcacf4f03be811f257ee5ec0b8fdb49fa4d582514d45413`, seed=22273, burn=1, samples=2, updates=4. It accepted **9/12** compact updates; retained dressed blocks had finite spread `3.8620012614e-8`. Across initial and retained configurations, the Hessian Ward norm was at most `2.7231e-15`; the omitted-contact defect was at least `8.6311`, and the undressed-insertion defect at least `.00176268`. Both wrong laws actually failed the same assertions. Finite inverse covariance residuals were at most `1.087e-15`; dressed finite gauge-invariance residuals were below `6e-18`. The source current/frame reconstruction probe separately observed current correspondence error `6.701e-16` and unitary intertwiner error `6.651e-15`.

The measured H_dt/source-current relative debts were **.19853/.10966/.05788/.02978** at dt=.2/.1/.05/.025; the contact debts were **.23824/.13343/.07093/.03662**. Thus finite-dt current differences are substantial and visible, not silently equated to the microscopic source. The inherited nonzero-holonomy neutral transfer controls also completed in this scoped probe.

The bound CLI correctly refused this historical receipt after parent changed a shared producer, with `Cutoff receipt source changed: bulk_frame_compton.py`. The direct numerical probe used its declared historical row and current consumer functions, not a weakened acceptance binding. Parent is regenerating the supplier receipt by its exact scientific CLI; **no final refreshed-input default acceptance is claimed here**. Reproduce the short numerical computation with the same declared row by calling `finite_step_current((3,3,3), band, .2, 22275)`, `neutral_transfer_controls((2,3,3,3), band, .2, 22274)`, and `coupled_chain((2,3,3,3), band, .2, UE1, K1, 22273, 1, 2, 4, .08)`, where band/UE1/K1 come from that actual receipt row. The stricter CLI above is the final reproducible acceptance route once supplier bindings are fresh.

## Limits

This removes Gaussian-only matter response reliance **at the specified finite compact regulator** by a real inverse/current/dressed-electron consumer with Ward controls. It does not prove an interacting equilibrium phase, nonlinear photon velocity matching, radiative Compton scattering, a nonzero-coupling UV continuum, or the full primitive quantum joining. The source-selected uniform fields and colour frame remain frozen; U1 photons are genuinely sampled. Heavy orbitals, SU4 dynamics, full Cartesian/Haar normal-field kinetics, primitive Pauli-code/centre constraints and their conditional measure/Jacobian are absent. No commit, publication, merge or issue closure is authorized by this receipt.
