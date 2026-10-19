# Same-law restored photon band to charged dynamical response

Issue222 remains active. Shared governing plan: `issue-222-continuing-backward-20260930`, revision `6b16fc5fa2204be1c2e567e06313425b16e3253daa91a041e857e12cca47af37`, general template revision `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`; work item `campaign`, requirements `outcome` and `verification`. The integration owner executes and owns scientific receipts. This revision is a runnable construction, not an already observed numerical result.

## Exact changed-root consumer and command

```sh
OPENBLAS_NUM_THREADS=1 /home/administrator/substrate-framework/.venv/bin/python /home/administrator/substrate-222/proposals/P256-qed-backward/noncompact_photon_response.py --n 7 --M 4 --UE .37 --K .9 --zeta 0 --eta .5 --neutral-states 16 --krylov 100 --intervals 1024 --photon-bands 1
```

For a bounded first exact finite calculation use `--n 3 --M 2 --krylov 40 --intervals 1024`. Run both complete consumers with `--cutoff-grid 3:2 5:3`, or enlarged `--cutoff-grid 7:4 9:5`; empirical changes in dominant energy, residue and fraction are printed, without declaring infinite-law convergence. Each run compares full-reorthogonalized Krylov sizes `krylov//2` and `krylov`; use `--krylov 40`, then `80` or `160` to vary the quadrature candidates independently of the complete pole supplier. `--photon-bands 2` joins the two largest-residue complete clusters separately to the actual charged time response. `--spectral-max-dimension 3000` bounds the complete source-sector dense solve; larger required dimensions fail explicitly instead of silently substituting incomplete Ritz poles. These are comparison photon coefficients, not native-coupler retuning. Output binds the response, companion spectrum solver, proof, engine and changed-root proof by SHA256.

The graph is precisely two vertices and three parallel directed edges with incidence rows `(1,1,1)` and its negative, rank-one curl `(1,-1,0)`, restored coordinate direction `f=(1,-1,0)/sqrt(2)`, and retained harmonic direction `h=(1,1,-2)/sqrt(6)`. The reduced Gauss fibre includes the exact longitudinal Coulomb operator `UE(n_x-2)^2/3`; the charged sector includes the opposite static reference at the other vertex. Literal longitudinal-invariant vectors in unreduced `L2(R^E)` are not asserted to exist.

The neutral sector has all 70 four-particle occupations and the charged sector all 56 five-particle occupations. All four external anti-fundamental reference components are retained in the source. The native N5 frame is only an actual source-selected frozen comparison, not a native admitted quantum ground or replacement for the heavy/full-field/colour Gauss supplier. The oscillator/Fourier basis is finite; the harmonic coordinate remains real and the displayed state belongs to one Bloch fibre, not a normalizable full-real-coordinate ground.

## 1. The photon leg is supplied by the actual interacting neutral state

With the engine's convention, `H_EM=UE p_Q^2+K Q^2` and `omega_bare=2 sqrt(UE K)`. The physical coordinate operator is

\[
Q_f=\sqrt{U_E/\omega_{bare}}(a+a^\dagger).
\]

This equation defines the coordinate, **not an interacting pole or residue**. The hopping phases are exact precompression Hermite/Laguerre matrix elements of `exp(i f_e Q_f)` with the actual harmonic shifts `(1,1,-2)`. They are not exponentials of a truncated coordinate matrix.

For the same actual neutral ground vector `psi`, the connected source is `q=(P_n Q_f P_n-<Q_f>)psi`. The exact output outside the oscillator cutoff is a single level:

\[
\|(1-P_n)Q_f\psi\|=\sqrt{U_E n/\omega_{bare}}\,\|\psi_{n-1}\|.
\]

Its square is the exact missing contribution to `||Q_f psi||^2`. This is an operator-output leakage calculation, not a convergence certificate for the unbounded photon source or its microscopic normal-domain transfer.

The prior parent consumer at n7,M4,UE=.37,K=.9,eta=.5 (EXIT0, 669 seconds, source prefix c5d30136) found the lowest bright low16 pole at .6405195001 with residue .0006487878, about .002026 of connected Q variance .3202841032. The unresolved low16 weight .3196353154 makes that pole a **real minor feature**, not an established dominant photon leg; the bare oscillator frequency 1.154123 is not a substitute. This receipt motivates a material supplier correction, not a numerical claim for the revised code.

The companion `noncompact_photon_spectrum.py` assembles the same entire finite H from exact phases, onsite and Coulomb matrices, checks its action against the independent engine, and checks that the Q-source triality projector has zero coupling to its complement. It then diagonalizes **every state** in that exact Q-accessible sector. This is not a Krylov subspace or a changed Hamiltonian: Q commutes with triality, and the other sectors are Q-dark for the exact sector state. Measured roundoff source leakage into them is retained. Degeneracies with dark sectors do not change the complete Q-accessible projector residue. All sector eigenvalues/eigenclusters and their Q weights are printed; the low poles remain separately visible.

For the complete eigenbasis X, the code measures `R=HX-X diag(E)` and `||X*X-I||`, uses the Frobenius residual as an operator-norm upper bound with a polar-orthogonalization correction, and brackets each eigencluster against the **entire complementary finite spectrum**. A gap smaller than twice the backward error fails. Cluster residual/gap bounds control projector error, hence residue and spectral-fraction error. These are a posteriori floating-point finite-matrix certificates for the supplied q and supplied E0, not interval arithmetic, exact real-number arithmetic, or an unbounded-H/state-domain theorem. Excitation intervals include cluster spread and the finite diagonalization backward error; the independently measured ground residual remains separately reported. The highest-residue complete excited cluster must have a positive lower-minus-competitor-upper residue margin; this is a numerical ordering certificate, **not a minimum physical residue/effect theorem**. Selected full-H eigenvectors also have whole-engine residuals and enlarged-basis tail residuals.

Positive full-reorthogonalized Krylov measures are retained only as **candidate quadrature** with size-dependent candidate energies, weights and whole-H Ritz residuals. A deliberately incomplete four-vector candidate must fail the identical whole-H eigencluster residual assertion; no raw unconverged node supplies a pole. The independently soluble finite oscillator positive control checks its exact omega, UE/omega coordinate residue, and independent resolvent (including a wrong-sign adversary). It validates the solver on a case where the bare measure really is correct, without assuming that measure for interacting H.

At each complex probe `z`, the Krylov spectral transform is checked against an independent full finite-H GMRES solution of `(H_0-E0-z)x=q`. If the compressed vector has residual `r`, selfadjointness gives

\[
|\langle q,x-x_{Krylov}\rangle|\le\|q\|\|r\|/|\Im z|.
\]

The complete eigencluster spectral transform is separately checked against the independent full-H GMRES resolvent, with its finite backward error, GMRES residual and measured complementary-source leakage in the error bound. Substituting the bare oscillator measure `(UE/omega_bare)/(omega_bare-z)` must fail this **same complete-measure assertion**, not a physical threshold or a loose unconverged-Krylov tolerance; nondiscriminating inputs fail explicitly. The unbounded-Q output leakage remains separate from both finite-H certificates and enlarged-basis displacement/Fourier residuals.

## 2. The creator actually reaches the charged state used by the response

The complete microscopic reference operators satisfy `sum_b B_b^dagger B_b=4-n_x`. All four actual source labels act on `psi`, giving the complex quartet Gram and full Hermitian Rayleigh matrix, including off-diagonal terms. The code never equates equal diagonal energies with a scalar complex-source matrix.

The prepared source selects a spectral sector. Every hopping harmonic shift `(1,1,-2)` equals the corresponding change of n_x modulo3, so **t=(ell−n_x) mod3 commutes with the entire H**, every Peierls current/contact, and Q_f. Derive t from the actual four-reference source, verify its projector commutator, and solve the invariant source-addressed block, with eigenpair residuals checked in the **whole original H**. This is spectral decomposition of an exactly conserved quantity, not an occupation/P_B cutoff or a replacement law; every other charged block remains in the full resolvents.

The first actual consumer rejected its intended global-ground join. At n7,M4,UE=.37,K=.9,zeta0 the neutral state is t1 and the point creator is t0. Unconditioned charged ground is t1, E−E0=.3380392786, with source overlap about4e−29. The actual source-reachable t0 lowest pair is E−E0=.3765941592 with each basis-source overlap .31457078776 per eigenvector. The unrelated global-ground Ward checks survive only as global-ground checks, **not** proof about this prepared carrier. The repaired consumer retains this wrong-ground failure at the same nonzero-source-weight assertion. It does not modify the source or delete harmonics to manufacture overlap.

For the declared normalized complex quartet `alpha`, define the four-column normalized created source `S=(sum_a alpha_a B_a psi)/sqrt(2)`. Let `G` be an orthonormal basis of the complete **source-addressed invariant-sector** lowest eigenspace, bracketed by a strictly higher eigenvalue in that same sector. Ec and the reported gap refer to this sector, not the global charged infimum or a reference-free mass. The actual spectral source weight and stationary charged density are

\[
A=G^\dagger S,\qquad w_c=\operatorname{Tr}(AA^\dagger),\qquad
\rho_c=AA^\dagger/w_c.
\]

The CLI requires a nonzero measured `w_c`. Thus the charged density is an actual source-conditioned stationary sector-cluster density from the **same neutral state and four references**, not a synthetic spectral array or unrelated electron input. Conditioning is declared: it is not a proof that the full unprojected created vector is a charged quasiparticle or that its weight remains nonzero in a volume limit. The omitted-reference adversary reaches and fails the same exact source norm-operator assertion.

## 3. Two ordered finite-frequency channels, contact, and an independent consumer

External edge sources shift the *same* photon Peierls phases by `alpha a_e+beta b_e`, so `J_a=partial_alpha H` and `C_ab=partial_alpha partial_beta H` are exactly the engine's current and contact. Actual shifted Hamiltonians independently finite-difference the restored-direction current and contact. Freezing hopping must fail the identical current-jet assertion.

Let `Delta=H_+-Ec`. For complete charged cluster columns `G`, the matrix response is

\[
\Pi_{ab}(z)=G^\dagger C_{ab}G
 -(J_aG)^\dagger(\Delta-z)^{-1}J_bG
 -(J_bG)^\dagger(\Delta+z)^{-1}J_aG.
\]

Both full resolvents include the complete finite Hilbert space and the degenerate cluster; neither channel is replaced by twice one ordered matrix element. Each selected highest-residue complete neutral eigencluster supplies its own measured energy and residue, with probes `z=Omega_band+i eta` and `z=.6 Omega_band+i eta`. A degenerate numerical cluster's center is a probe with its printed excitation interval, not a claim that numerically split energies are one exact pole. Both inverses are safely bounded by `1/eta`. The response is generally complex at these frequencies; it is not a purported imaginary static Hermitian effect.

The independent actual quantum-time consumer uses `exp(+/- i Delta t)` from the same engine, not the frequency-space solutions:

\[
\operatorname{Tr}(\rho_c\Pi_{ab}(z))
=\operatorname{Tr}(\rho_c G^\dagger C_{ab}G)
-i\int_0^\infty e^{izt}\operatorname{Tr}\rho_c
 \left[(J_aG)^\dagger e^{-i\Delta t}J_bG
 -(J_bG)^\dagger e^{+i\Delta t}J_aG\right]dt.
\]

The integral stops at `T=16/eta`; its rigorous omitted-time bound is `2 ||J_a G||_F ||J_b G||_F exp(-eta T)/eta`. Fine/coarse Simpson differences are reported as a numerical diagnostic, **not a rigorous quadrature certificate**. The same time-versus-frequency assertion rejects omitted second channel, omitted contact, and frozen hopping. It also requires actual nonzero response, because zero is explicitly an adversary to that very assertion. No minimum effect is claimed for arbitrary parameters; inputs which fail to discriminate raise rather than silently weaken an assertion.

The physical sources are `a=b=f` and `a=f,b=(f+h)/sqrt(2)`, with real independent edge directions. The second is not a one-sided static f/h term. Pure h static translation zero is not invoked as a nonforward Compton amplitude. For each selected actual complete neutral band, the finite external-leg diagnostic matrix is `Z_band (f.a)(f.b) Pi_ab`; its output names that band index, energy, residue and fraction. The same complete-reference source-addressed charged density, both ordered channels, actual contact, and independent quantum-time omission assertions are retained unchanged. This is finite photon-band response, not an LSZ or on-shell S-matrix assertion: no spatial momentum, asymptotic flux or scattering normalization is inferred on a two-vertex graph.

## 4. Density/scalar signs are checked rather than assumed

For `N=n_x`, and the longitudinal edge direction `g=(1,1,1)`, the exact microscopic convention is `J_g=i[N,H]`. Consequently `J_g G=-i Delta N G`. Define `S_NN=(NG)^dagger[(Delta-z)^-1+(Delta+z)^-1]NG`. The code checks full cluster matrices, not only their real or averaged part:

\[
\Pi_{gg}=-z^2 S_{NN},\qquad
\Pi_{fg}=iz\big[(J_fG)^\dagger(\Delta-z)^{-1}NG
 +(NG)^\dagger(\Delta+z)^{-1}J_fG\big].
\]

Thus the signed density kernel is `Pi_00=-S_NN` and `Pi_gg=z^2 Pi_00`. These identities retain the actual contact and both channel signs. They concern scalar/temporal Ward response on this graph, not a relativistic four-vector photon identification.

## Remaining explanatory debts

This correction replaces a tiny lowest-resolved-band supplier by a complete same-law Q-accessible finite spectral measure, with dominant-band ordering and projector-residue errors, while retaining the earlier low pole as a minor feature. It does **not** discharge unbounded-Q source/operator-domain and normal/Born-Huang transfer; oscillator/harmonic and time quadrature convergence; global Bloch-infimum/normalizable packet transfer; native full-field/heavy/colour Gauss and separated-reference bounds; a thermodynamic nonlinear phase or isolated mobile charged sector; global/reference-free mass; 3D continuum, physical spin/Lorentz, zero-broadening, radiation and asymptotic scattering limits. Quantum postulates and the explicit noncompact root remain declared premises. The parent scientific run, including adverse controls, is required before revised numerical claims.
