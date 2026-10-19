# All-Haar, all-phase, full-coordinate Gauss packet

`full_haar_field_packet.py` removes the scalar colour and electromagnetic support tubes. It imports the unchanged `FullBandGaussSection`, compact bump, primitive point controls and source identities from the accepted full-band packet. The Hamiltonian quadratic form and its normalization are exactly those in `full_band_packet_bound.md`: 120 Cartesian field coordinates, one genuine SU(4) link and two compact parallel EM paths. This is an analytic trial construction, not a solved ground or phase.

The shared defaults remain `m=.2,r=.12,v=.3,kappa=2,J=.2,ue4=.3,ue1=.4,K=.7`, unimproved, whole-heavy scale one. Coefficients are read from the owner's `onsite` and `paths`; there is no second implementation of the one-body law. The field anchor is the same size-three selected frame with temporal amplitude approximately `.64839832515` and spatial amplitudes `.23863957694`. The Cartesian field kinetic **replaces**, rather than adds to, selected-orbit inertia. All imported files remain unchanged.

## First discriminator and material repair

Let `c_min=min_s onsite_s>0`, `c_max=max_s onsite_s`, `p=max_s ||path_s||2`, and let both temporal anchor matrices have eigenvalues `(-b,-b,b,b)`. Define

\[
 F_* = \max_{s,x}\sum_\mu |path_{s\mu}|\,\|\Phi^*_{x\mu}\|_{op}.
\]

The triangle bound reads the actual path. At the shared anchor it is approximately `.07469983605`, while `c_min b` is approximately `.20748746405`. Thus the onsite positive gap dominates hopping **uniformly in the colour link and both phases**, not just near the identity. The actual light factor `(exp(iA0)+exp(-iA1))/2` has modulus at most one.

For the full 120-coordinate field ball `||u-u*||2<R`, set

\[
 b_-=b-2R,\quad F_R=F_*+\sqrt2 pR,\quad
 m_0=c_{min}b_- -F_R,\quad
 E_{max}=c_{max}(b+2R)+F_R.
\]

Each temporal perturbation has operator norm at most twice its coefficient norm. Each endpoint path changes by at most `2p r_x`; the endpoint average is bounded by `p(r_0+r_1)<=sqrt(2)pR`. There is no small-link or small-phase approximation. Outward `1e-10`-scaled numerical margins are used in the receipt's anchor constants; the mathematical formulas concern exact norms of the specified finite matrices, not interval-enclosed U/A extrema.

The first requested radius is `.2`. The all-Haar anticommutator inequality then fails and is rejected explicitly. The repair changes the **field** support, not the law, scalar link envelope or scalar phase envelopes. The automatic route compares 76 deterministic radii `.005,.006,...,.080`, accepts each only after the analytic chart and negative-leading-energy inequalities, and chooses the smallest resulting threshold. It does not claim a globally optimal radius. An explicit `--field-radius` must pass the same inequalities; it cannot silently recover by narrowing gauge support. If no candidate passes, the CLI fails without a negative-energy receipt.

## Exact normalized scalar packet

Let

\[
 p_0(\Phi)=Z_F^{-1}\left(1-\|u-u_*\|^2/R^2\right)_+^4,
 \qquad Z_F=\pi^{60}R^{120}\Gamma(5)/\Gamma(65).
\]

Average this **field-only density** over normalized `SU4_0 x SU4_1` Haar:

\[
 p(\Phi)=\int_{SU4^2}p_0(g^{-1}\Phi g)\,dg,
 \qquad f(\Phi)=\sqrt{p(\Phi)},\qquad
 \Psi(\Phi,U,A)=f(\Phi)\Omega(\Phi,U,A).
\]

Here `Omega` is the actual 4900-state Pauli `FullBandGaussSection.state`, with `NH=NL=4`, heavy top-four occupied under the whole-negative law and light bottom-four occupied. Scalar link and phase amplitudes are **exactly one** on normalized Haar and the two normalized circles. Hence their scalar Fisher costs are exactly zero, not merely small. The support includes every `U in SU4` and every `A in T^2`, and a compact union of gauge images of the full field ball. No field coordinate is frozen. Fubini and invariant field measure give `int p=1`; pointwise `||Omega||=1` then gives `||Psi||=1` exactly. The integral defines the packet without a numerical Haar quadrature surrogate. `log_field_seed_density` evaluates the normalized seed; the averaged density is the displayed compact integral, not that seed alone.

Scalar gauge invariance combined with the unchanged section covariance gives all thirty primitive colour and both light-only EM Gauss equations as vector equations. Heavy is EM neutral. The compact boundary power yields a form-domain H1 amplitude, and the estimates below keep the section uniformly smooth on its support.

## Global chart and quantitative principal angles

For every field in the seed ball, let

\[
 B=\operatorname{diag}(\operatorname{sign}\Phi_{00},
                       \operatorname{sign}\Phi_{10}).
\]

The local matrices have two positive and two negative eigenvalues, with absolute eigenvalues at least `b_-` and local rank-two gaps at least `g_L=2b_-`. For either species' actual one-body matrix, the onsite contribution to the anticommutator is `c_s |Phi_0|`. Its off-diagonal block is `(T_s B_1+B_0 T_s)/2`, whose operator norm is at most `||T_s||<=F_R`. Therefore

\[
 A_s=\{h_s,B\}/2\succeq m_0I>0,\qquad \|h_s\|_{op}\le E_{max}.
\]

For an eigenvector of eigenvalue `lambda`, `lambda <B> >=m_0`, so `|lambda|>=m_0`. Scaling hopping continuously to zero preserves this strict inequality, so the inertia is exactly four positive/four negative eigenvalues. The middle-band gap is at least `g_B=2m_0` for **all** U and A.

The positive plane has an orthonormal frame `V`, positive restriction `S=V†hV`, and compressed beta form `X=V†BV`. It solves

\[
 SX+XS=2V^\dagger A_sV,\qquad
 X=2\int_0^\infty e^{-tS}(V^\dagger A_sV)e^{-tS}\,dt
 \succeq m_0S^{-1}\succeq (m_0/E_{max})I.
\]

For the negative plane, replace `(h,B)` by `(-h,-B)` and obtain the identical bound. The Gram matrix of projection of the positive h-plane onto the local B-positive plane is `(I+X)/2`; analogously the negative plane projects to B-negative. Thus every principal cosine obeys

\[
 \cos^2\theta_j\ge q=\frac{1+m_0/E_{max}}2>\tfrac12.
\]

This is quantitative, not a mere transversality claim.

### Why the actual Pauli number polynomial has this overlap

No Slater signs are supplied to build a state. The owner's matrix units are `PrimitiveMatter.transfer` using `PauliCode.transfer`, whose amplitudes are matrix elements of the actual two-word `S=+1` projected edge states. Multiplication uses the exact binary Pauli-word law; closure of paths has the actual `S=+1` holonomy. These give

\[
 E_{ij}^\dagger=E_{ji},\quad
 [E_{ij},E_{k\ell}]=\delta_{jk}E_{i\ell}-\delta_{i\ell}E_{kj},\quad
 [E^H_{ij},E^L_{k\ell}]=0.
\]

The diagonal units are actual zero/one occupations, their species sums are four, and every allowed transfer connects its two occupation masks with unit amplitude. Their joint weight basis is therefore the fixed-four exterior representation (dimension 70 per species), up to phases **derived** from those same Pauli words. Equivalently, the Lie algebra exponentiates the one-body change of basis on that finite representation. This representation identification is a theorem about the primitive builders, not substitution of canonical CAR matrices into the Hamiltonian.

For a rank-four one-body projector P, `N(P)=sum P_ij E_ij` has integer occupations `0,1,2,3,4`. Consequently the owner's actual polynomial

\[
 \Pi_P=\prod_{j=0}^3 (N(P)-j)/4!
\]

is precisely the filled-plane orthogonal projector. The paired epsilon singlet is invariant under the actual local SU4 representation. Its six complementary local two-particle terms have equal modulus; the owner's `sqrt(6)` local projection onto the light temporal-negative plane produces a normalized light-filled negative plane and heavy-filled complementary positive plane. At two sites their product is exactly the local initial filled B-negative light/B-positive heavy state, with its phase fixed by the actual singlet and occupation polynomials.

For completeness, the exterior overlap modulus follows without choosing supplied signs: apply the induced unitary to four mutually orthonormal occupied creators; their Gram norm is the determinant of the one-body projection Gram matrix. Every eigenvalue of that Gram matrix is at least q. Each species' squared filled overlap is at least `q^4`; the commuting species projections give the **actual** joint raw weight

\[
 w=\|\Pi_{P_H}\Pi_{P_L}\,initial\|^2\ge q^8>0.
\]

There is no cancellation of alternative singlet charts or a frozen-heavy state. The global source section chart is certified on the full seed support, all gauge images and all U/A. The CLI independently exercises all 8192 same-species matrix-unit commutators, all 4096 cross-species commutators and all 128 adjoint relations on the full actual 4900-dimensional representation. For this algebra certificate, it re-evaluates each actual projected Pauli word using exact fourth-root phases and the exact two-term normalization factor `1/2`; it does not round measured transfers or import wedge signs. Every algebra residual is required to be **exactly zero**. The exact-word units are independently compared against the owner's floating-point units. Initial complementary-filled occupation and actual number-polynomial idempotence/filled occupation are then checked on the unchanged owner representation. This finite exhaustive algebra certificate establishes the primitive representation; the uniform U/A inequalities come from the analytic argument above.

## Kinetic cost, including section Berry phase

For separated spectra the positive Sylvester integral gives `||dP||op<=||dh||op/gap`. For the rank-n filled projector its derivative norm is the square root of the sum of squared occupied-empty mixing singular values, hence at most `sqrt(n)||dP||op`. In particular each global rank-four filled projector costs at most `2||dP||op`, not an invented Slater-sign derivative.

For one site's paired initial epsilon state, the complementary heavy/light filled determinants give

\[
 \|d\chi\|^2=\operatorname{Tr}(dP)^2\le4\|dP\|_{op}^2.
\]

Its phase connection vanishes because the complementary trace connections sum to the traceless local SU4 connection. This proves only the initial paired identity; it does **not** assume cancellation of the full hopping section's Berry phase. A field coordinate tangent is one basis matrix of operator norm at most two. If

\[
 d_\mu=2c_{max}\delta_{\mu0}+\max_s|path_{s\mu}|,
\]

then the actual endpoint-averaged one-body tangent has norm at most `d_mu`. Differentiating the actual raw projector product and normalization gives

\[
 D_{F,\mu}=q^{-4}\left(4\delta_{\mu0}/g_L+4d_\mu/g_B\right),\quad
 D_U=q^{-4}\,4F_R/(\sqrt2 g_B),\quad
 D_A=q^{-4}\,F_R/g_B.
\]

Normalization subtracts the real parallel component, which cannot increase derivative norm; no extra factor two is needed. There are 30 field directions per component, 15 colour tangents and two phase directions. The section metric upper bounds are `30 sum_mu D_F,mu^2`, `15 D_U^2`, `2 D_A^2`. Components absent from the actual x-path and onsite operator really have zero **section** derivative; their scalar field-envelope derivatives remain present.

Fisher convexity under the field gauge average preserves the compact 120-ball envelope cost `5120/R^2`. For the real amplitude f and normalized complex section, `Re<Omega,dOmega>=0`, so the scalar/section kinetic cross term vanishes. The complete `||dOmega||^2` remains, including its Berry part. The field/link/phase costs are thus finite without any scalar gauge-localization Fisher term.

## Honest all-link leading energy and amplitude bounds

Let `rho_*=||Phi_*||HS/2` and `G_*` be the two-block Gram vector. The full-field bounds are

\[
 Q\le\frac\kappa2(\|G_*\|_F+2\rho_*R+R^2)^2,\quad
 W\le2\sum_{x,\mu}\operatorname{Tr}\Phi_{x\mu}^2
   \le8(\rho_*+R)^2.
\]

At identical endpoints the first W expression is exactly `4 sum_mu Tr Phi_mu^2`. This is a full W upper bound on arbitrary links, **not** the selected centre alignment value. The factor two in the sum over both endpoints follows directly from `||X-Y||^2<=2||X||^2+2||Y||^2`.

Ky Fan tests the actual h-heavy top-four sum on the **onsite** positive four-plane. Hopping has zero trace on that plane, regardless of U. For a traceless temporal perturbation, `|Tr(P_rank2 dPhi)|<=||dPhi||HS` because `||P-I/2||HS=1`. Thus

\[
 -\sum_{top4}\lambda(h_H)\le -c_H(4b-2\sqrt2 R)=H_{KF}.
\]

The analogous negative onsite test gives the upper bound `L_KF=-c_L(4b-2sqrt(2)R)` on the light bottom-four sum. The certified leading coefficient is

\[
 C=Q+JW_{upper}/8+H_{KF}<0.
\]

Neither the centre heavy energy nor the centre alignment is reused. Since scalar phase density is uniform, `||Omega||=1` and scalar field density is phase-independent, the magnetic packet expectation is **exactly zero**. Keeping a conservative nonnegative constant,

\[
 T=\tfrac12(5120/R^2+30\sum_\mu D_{F,\mu}^2),\quad
 B=u_4\,15D_U^2+\tfrac{u_1}{2}\,2D_A^2+\max(0,L_{KF}),
\]

\[
 E_0\le E_{trial}\le C\Lambda+T\sqrt\Lambda+B,\quad
 \Lambda_* =\max\left(1,
 [ (T+\sqrt{T^2+4(-C)B})/(2(-C)) ]^2\right).
\]

For `Lambda>Lambda_*` the trial energy is strictly negative. The receipt evaluates the bound at `4 Lambda_*`. Trial support has `sum_x rho_x<=sqrt(2)(rho_*+R)` and the scalar amplitude is at most `Z_F^(-1/2)`, reported in logarithmic form. These are finite compact-packet amplitude bounds, not local order.

The owner one-body estimate `|dGamma_s(h_s)|<=c sum_x rho_x`, with `c=8(o+p)` and `rho_x=sqrt(sum_mu Tr Phi_xmu^2/4)`, gives

\[
 H\ge\Lambda\kappa\sum_x\rho_x^4/8-(\Lambda+1)c\sum_x\rho_x-K.
\]

The Gram inequality used here is `||G_x||F^2>=Tr(G_x)^2/4=rho_x^4/4`. Put `alpha=Lambda kappa/8`, `d=(Lambda+1)c`. Young minimization gives `alpha r^4-dr>=-Y(alpha)`, with `Y(a)=3d^(4/3)/(4^(4/3)a^(1/3))`. Thus `E0>=-2Y(alpha)-K`, a finite lower energy bound. Splitting the quartic in half gives a finite fourth-moment upper bound on a ground or energy-controlled minimizing sequence,

\[
 \sum_x\langle\rho_x^4\rangle\le
 [E_{upper}+2Y(\alpha/2)+K]/(\alpha/2).
\]

The negative trial also implies

\[
 \sum_x\langle\rho_x\rangle\ge
 \max(0,[-E_{upper}-K]/[(\Lambda+1)c]),\quad
 \liminf_{\Lambda\to\infty}\sum_x\langle\rho_x\rangle\ge-C/c.
\]

These are invariant volume-sum amplitude consequences, not a charged lower gap, local Clifford selection or phase theorem.

## Portable parent-owned execution

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/full_haar_field_packet.py \
  --output proposals/P256-qed-backward/full_haar_field_packet_receipt.json
```

The parent owns this final CLI and composed validation. The receipt captures own/source/proof hashes before constructing the law, checks them again before output, and emits strict finite JSON. It independently probes non-Clifford deformed fields, a colour link far outside the old `.001` tube and phases `(2.137,-1.081)` using the actual primitive Gauss/whole-energy consumer. Principal-angle observations and exhaustive finite Pauli algebra controls are explicitly separated from the uniform certificate. No Haar/phase truncation or sampled-extrema assumption is used.

### Scoped scientific probe, not the final CLI

A project-Python API smoke run selected `R=.033` after rejecting `.2` at the positive-anticommutator inequality. Of the 76 scalar candidate inequalities, 47 passed. It observed these computed uniform constants:

| Quantity | Certified value |
|---|---:|
| `m0`, middle-band gap lower | `.10412801233607`, `.20825602467214` |
| each principal cosine squared lower | `.667490824352296` |
| actual joint projection weight lower | `.0394060043063307` |
| `C`, `T`, `B` | `-.0817356988215544`, `2462389.18194732`, `144.047009612302` |
| strict negativity threshold | `9.07590264801556e14` |
| upper energy at four times threshold | `-1.48365049074534e14` |

The threshold is approximately 68.1 times smaller than the earlier narrow-gauge `R=.006` threshold `6.183138628248723e16`, despite replacing the centre heavy/alignment values by the more conservative all-link Ky Fan/full-W bounds. The constant term drops from approximately `2.17e14` to `144.05`; the remaining threshold is dominated by genuine field kinetic cost, not hidden scalar gauge localization.

At the deformed primitive point the link distance from identity was `2.889651169931746`, phases were `(2.137,-1.081)`, and actual projection weight was `.987828030938253`. Colour/EM Gauss residual maxima were `9.74e-16`/`3.92e-16`; heavy/light full-primitive occupied-energy residual maximum was `7.22e-16`. The minimum observed principal cosine squared was `.99763348`, not used to strengthen the uniform lower bound.

A separate scoped call of the exact-word representation consumer returned **zero** for every exhaustive same-species/cross-species/adjoint algebra residual. Exact words versus the actual floating owner units differed in Frobenius norm by at most `8.31e-15`; initial complementary occupations and actual number-polynomial controls were below `1.76e-15`. These are observed primitive consumer results, not the parent final receipt. Native root-launched Pyright 1.1.414 symbols/diagnostics and a positive Python AST call query were exercised; final native diagnostics were empty, and scoped CLI diagnostics reported zero errors/warnings.

At `4 Lambda_*`, the invariant total-amplitude mean lower bound is `.0106083038659064`, its asymptotic lower bound is `.0212166077317923`, and the finite total fourth-moment upper bound is `90.9803967686891`. These are deliberately loose average bounds, not assertions about a local ordered phase.

This remains a two-cell, one-link, two-path variational theorem. It supplies no three-dimensional phase, lower charged gap, mechanical half-spin, Lorentz symmetry, canonical/full QED or issue-closure conclusion.

## Actual parent full-Haar consumer

The parent subsequently executed the actual CLI into
`full_haar_field_packet_receipt.json`, naturally exiting zero. The executing
code was `26da553446f9a631c82e6987e105ffc5ae8acd5e67fb41dc379671b8c205b2c0`.
The receipt reproduces the uniform C/T/B and threshold above, all 120 field
coordinates, complete colour Haar link and both full phase circles, the
non-Clifford/noncentral primitive point and exhaustive exact-word algebra
controls. A scoped proof/source/receipt review accepted this variational
construction; numerical displays remain conservative binary64 evaluations,
not a formal interval/proof-assistant artifact.

This earns an admissible quantum energy upper and nonzero invariant
amplitude mean bound without freezing those field coordinates. It does not
derive the selected mean frame as the unrestricted ground. The separate
unrestricted-field competition consumer must decide that question on its
actual one-axis and three-axis geometries; local/global phase and the joined
charged sector are not inferred from the packet.

## Same-law stationary-pair anchor, not a restricted field packet

The actual producer accepts `--anchor-frame pair-stationary`. It obtains the
balanced two-field reference from `unrestricted_field_minimum.py`, using the
requested quartic stiffness rather than silently retaining that provider's
bare default. With the actual one-axis onsite coefficient \(c\), imaginary
spatial path coefficient \(t\), real Wilson path coefficient \(d\), and
positive stiffness \(\kappa\), the reference is

\[
R=[(c^4+t^4)/\kappa]^{1/3},\qquad
b^2=c^2/(\kappa R),\quad s^2=t^2/(\kappa R),\quad
\bar\Phi=(b\Gamma_0,s\Gamma_1,0,0).
\]

For the stated positive cluster regime \(R>db\), its restricted classical
energy is \(-3R\). This is an anchor choice, not a global field minimum or
an assertion that the two absent mean components must stay zero. The
normalized field packet still varies **all 120 Cartesian coordinates**;
both colour Haar factors and both electromagnetic circles remain complete.
The uniform all-link inequalities above need identical endpoints and a
symmetric rank-two temporal mean, not four nonzero Clifford components.

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/full_haar_field_packet.py \
  --anchor-frame pair-stationary \
  --output proposals/P256-qed-backward/full_haar_pair_reference_receipt.json
```

The first actual parent CLI finished in 18.40 seconds. At the same bare law,
the selected radius is `.046`, the uniform actual joint projection weight
lower is `.019202627555553388`, and the energy upper is

\[
E_0\le -.24076162233613418\,\Lambda
+1796509.1391696879\sqrt{\Lambda}+797.1501879466975.
\]

Strict negativity follows above `5.567809140503005e13`, about 16.3 times
below the environmental-anchor all-Haar threshold. At its actual
non-Clifford/noncentral point, the projection weight is
`.9801234249488937`; colour and electromagnetic Gauss residual maxima are
`1.3642e-15` and `4.5753e-16`, and both full primitive occupied-energy
residuals are below `6.34e-16`. Exhaustive exact-word same-species,
cross-species and adjoint controls are zero; actual degree-four projector
and occupation residuals are below `1.11e-15`.

The executing manifest now inherits the reference provider's complete owner
map as well as this producer/proof; each receipt records the mean matrices,
reference formula and actual stiffness. These results improve a legal
quantum upper, not a phase, bulk ground selection, charged gap or physical
spin theorem. In particular, the pair's forty-one classical Hessian flats
and residual stabilizer do not disappear because this trial has a better
upper bound.

