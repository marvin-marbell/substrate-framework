# Cartesian-derived angular principal form and its actual scalar gauge quotient

## Scope, laws and source ownership

This replaces the independently declared scalar `I=2` **at tangential principal-form level**, not by asserting a full Cartesian Born–Oppenheimer theorem. The law is the actual two-endpoint, 120-real-coordinate law of `full_band_packet_bound.md` and `full_haar_field_packet.md`:

\[
 \mathfrak t[\Psi]=\frac{\sqrt\Lambda}{2}\sum_{x=0,1;\mu=0}^3\sum_{b=1}^{15}
 \|D_{u_{x\mu b}}\Psi\|^2+u_4\sum_{a=1}^{15}\|E_a\Psi\|^2.
\]

Here `Phi[x,mu]=sum_b u[x,mu,b] B_b`; the existing owner `sea_selection.matrix_unit_basis` gives `Tr(B_a B_b)=4 delta_ab`, and `FullBandGaussSection.generators` gives `T_a=B_a/sqrt(8)`, `Tr(T_a T_b)=delta_ab/2`. The source-owned Casimir is one left **or** one right Casimir, not their sum: `C2(4)=15/8`. Its coefficient is `u4=ue4`. The endpoint field kinetic replaces the old selected-orbit kinetic; adding an independent rotor counts the same field tangent twice. No inertia is inferred from a potential Hessian.

Bare values remain `a=1,m=.2,r=.12,v=.3,improved=False,kappa=2,g0=0,J=.2`. `RotationBand`, `uniform_data` and `pair_clifford_reference(model, stiffness=2)` execute their inherited owners. Comparisons are:

* the size-three environmental full-four-Clifford frame, not a demonstrated Cartesian ground;
* the balanced two-field, actual pair stationary reference, not a bulk minimizer;
* all 60 uniform Cartesian coordinate bytes of the existing open-cube candidate, not a 480-coordinate or periodic-bulk ground.

The last frame is **not** silently refitted, projected onto an exact commutant or replaced by the environmental frame. The input receipt is hashed byte-for-byte. Its previous source hashes and differences from executing owners are separately emitted: old candidate stationarity evidence is not laundered into a new contemporaneous classical proof. Own source/proof, all inherited executing owners, the Cartesian kinetic proofs, and the consulted charged-boundary owner/proof are hashed before construction and checked unchanged before output. Parent owns scientific execution and joined verification.

## 1. Actual conjugation differential, metric and inertia

Fix one mean frame `Phi=(Phi_0,...,Phi_3)` and equal endpoint means. For `q(t)=exp(i t x_a T_a)`,

\[
 \delta\Phi_\mu=i[x_aT_a,\Phi_\mu],\qquad
 D_{(\mu b),a}=\frac14\operatorname{Tr}\{B_b i[T_a,\Phi_\mu]\},
\]

\[
 \boxed{G_{ab}=(D^TD)_{ab}
 =\frac14\sum_\mu\operatorname{Re}\operatorname{Tr}
 \{i[T_a,\Phi_\mu]i[T_b,\Phi_\mu]\}.}
\]

This is the pullback of the actual Cartesian metric `sum du²=HS²/4`, not the HS metric without its factor four. The ordinary classical Legendre dual of the given quantum principal symbol assigns one endpoint

\[
 I_f=G/\sqrt\Lambda,\qquad
 L_{f,\mathrm{principal}}=\tfrac12 x^T I_f x.
\]

A link velocity `dot U=i z_a T_a U` has the owner coordinate metric `2 Tr(Z²)=sum z_a²`. Since its Hamiltonian is `u4 sum E_a²`,

\[
 \boxed{I_\ell=1/(2u_4),\qquad L_\ell=\tfrac12 I_\ell\|z\|^2.}
\]

There is no additional link inertia to choose. The positive endpoint eigenvalues are divided by `sqrt(Lambda)`, while the link inertia is tied to the already existing `ue4`.

Let `H={h in SU4: h Phi_mu h†=Phi_mu for every mu}`. Its Lie algebra is exactly `ker D=ker G`; the field orbit is `SU4/H`, of dimension `rank D`. The normal Cartesian dimension per endpoint is `60-rank D`, not zero.

For weighted Clifford fields `Phi_mu=b_mu Gamma_mu`, the Hermitian-normalized nonempty Clifford monomials diagonalize G. If `S` has grade `k`, it anticommutes with `Gamma_mu` when `k` is even and `mu in S`, or `k` is odd and `mu notin S`. Consequently

\[
 g_S=\frac12\sum_{\mu:\,\{\Gamma_S,\Gamma_\mu\}=0}b_\mu^2
 =\ell_S/8,
 \qquad \ell_S=4\sum_{\mathrm{anticommuting}\ \mu}b_\mu^2.
\]

The factor `1/8` follows from `Tr(T_S²)=1/2` and the field `HS/4` metric. It is checked against the full matrix G in the CLI.

* Four nonzero Clifford amplitudes give rank 15, zero Lie kernel and the exact group stabilizer `H=Z4`.
* Two nonzero anticommuting fields generate `M2(C) tensor I2`. Their complex commutant has dimension four and their connected SU4 stabilizer is SU2, of Lie dimension three. The full stabilizer has the determinant-constrained U2 components, including the centre; discarding disconnected elements is not permitted. Orbit rank is 12.
* The cube receipt resolves a one-dimensional Lie kernel at its stated HS commutator tolerance `1e-7`. Its finite coordinate bytes have a small **nonzero** discarded tangent singular value. The CLI displays the raw G, all singular values, raw field inertia, resolved G, numerical kernel and discarded tangent norm separately. The rank-14/U1 calculation is conditional on this resolved germ, **not an exact algebraic U1 theorem for those bytes**. It introduces no epsilon floor, regularized inverse or invented U1 charge. Keeping a tiny true direction would instead produce a very large inverse and a different exact stratum; the receipt does not justify hiding that distinction.

## 2. Gauge quotient, invariant measure and principal Dirichlet form

Represent endpoint fields by `q_x Phi q_x†`, with `q_x in SU4/H`, and keep the actual link U. Independent local gauge transformations act as

\[
 q_0\mapsto g_0q_0,\quad q_1\mapsto g_1q_1,
 \quad U\mapsto g_0Ug_1^\dagger.
\]

The relative link is

\[
 V=q_0^\dagger Uq_1.
\]

Changing orientation representatives `q_x -> q_x h_x` leaves fields unchanged and gives `V -> h_0† V h_1`. Thus the **scalar** physical quotient is

\[
 \boxed{\big((SU4/H)^2\times SU4\big)/(SU4_0\times SU4_1)
 \simeq H\backslash SU4/H.}
\]

At fixed orbit the field measure is the invariant homogeneous-orbit measure: ambient Cartesian conjugation is an isometry, so the orbit-volume factor is constant along that orbit. Normalize it using the pushforward of Haar. Invariant product measure on both endpoint orbits and on the link reduces exactly to the pushforward of normalized Haar `dmu_H(V)` on the double coset. Explicitly, for any residual-biinvariant scalar F,

\[
 \int dq_0\,dq_1\,dU\,|F(q_0^\dagger Uq_1)|^2
 =\int d\mu_H(V)\,|F(V)|^2.
\]

This is Fubini plus left/right Haar invariance, not a finite quadrature or a choice of a new volume. It is a fixed-orbit disintegration; the fixed orbit has zero ambient 120-dimensional measure and is **not** by itself a normalized Cartesian BO wavefunction.

At representatives `q0=q1=1,U=V`, write `R=Ad_V` in the owner generator basis. Endpoint body velocities `x0,x1` and link left velocity z obey

\[
 w=-x_0+z+Rx_1.
\]

In the full-rank case the product inertia is `M=diag(I_f,I_f,I_l Id)` and the relative map is `A=(-Id,R,Id)`. Constrained minimization, rather than summing inertias, gives

\[
 \min_{Av=w}\tfrac12 v^TMv
 =\tfrac12 w^T(AM^{-1}A^T)^{-1}w,
\]

\[
 \boxed{Q(V)=AM^{-1}A^T
 =\sqrt\Lambda\{G^{-1}+RG^{-1}R^T\}+2u_4\,\mathrm{Id},
 \qquad I_{\mathrm{quotient}}(V)=Q(V)^{-1}.}
\]

For example if `G=g Id`, then `Iquot=(2 sqrt(Lambda)/g+2u4)^-1`, **not** `2g/sqrt(Lambda)+1/(2u4)`. Velocity inertias combine through a constrained minimum (parallel inverse sum), not an additive mechanical guess.

The scalar principal Dirichlet form, on smooth residual-invariant functions and then its natural form closure, is

\[
 \boxed{\mathfrak t_{\rm tan}[F]
 =\frac12\int d\mu_H(V)\,(d_LF)^*Q(V)d_LF.}
\]

Equivalently, let `X_La F=d/dt F(exp(i t T_a)V)` and use the right vector fields with either consistent sign. Then

\[
 \mathfrak t_{\rm tan}[F]
 =\frac{\sqrt\Lambda}{2}\int
 \big\{(X_LF)^*G^{-1}(X_LF)+(X_RF)^*G^{-1}(X_RF)\big\}\,d\mu_H
 +u_4\int\sum_a|X_{La}F|^2\,d\mu_H.
\]

The symmetric coefficient matrices give anisotropic left/right rotors, each with coefficient `sqrt(Lambda) G^-1/2`, plus the existing link Casimir **once**. These Haar vector fields have zero Haar divergence. In coordinate charts the operator is the divergence form belonging to this measure and form. The volume of the quotient metric `Q(V)^-1` is not substituted for Haar: in general it differs by a variable determinant density. Replacing it would change drift and, after conjugating measures, scalar potential terms. On a double coset, the Haar pushforward also records varying orbit-volume factors near singular strata; those are not discarded.

Common endpoint conjugation is gauge. For any x, simultaneous `x0=x1=x` and `z=x-Rx` give `w=0`. The minimum quotient cost is zero, even though this particular vertical product velocity has positive unreduced kinetic cost. Gauge-invariant scalar functions have zero differential along that velocity. It is not a second physical rotor. The CLI checks both this tangent identity and a finite actual two-endpoint SU4 transformation.

### The scalar centre consequence is not optional

For the full-four-Clifford frame `H=Z4`, residual invariance is `F(zV)=F(V)`, so the scalar domain is `L²(SU4/Z4)`, not unrestricted `L²(SU4)`. All four centre links are the **same scalar gauge orbit**. Scalar Haar four-well trial functions in different centre-character sectors therefore do not automatically form four physical scalar states. One may use a single chart on this quotient, or residual-invariant sums of equivalent charts, but may not count its four lifted copies as a physical quartet.

## 3. Nontrivial stabilizers, horizontal covectors and singular strata

Let K contain orthonormal basis columns for `h=ker G`, and let `G+` be the inverse on `h-perp`, zero on h. This is an exact Moore–Penrose inverse when the stabilizer is exact, not `(G+epsilon Id)^-1`. At V, a scalar bi-H-invariant differential must satisfy

\[
 K^T\xi=0,\qquad (RK)^T\xi=0.
\]

Let S be an orthonormal basis of `(h+Ad_V h)-perp`. On these horizontal covectors,

\[
 Q(V)=\sqrt\Lambda(G^++RG^+R^T)+2u_4\,\mathrm{Id},\qquad
 Q_h=S^TQS,\qquad I_h=Q_h^{-1}.
\]

The inverse is taken only on the true horizontal block, not across null orientation directions. Endpoint velocity coordinates are only `x=B y`, where B spans `h-perp`; the independent minimum uses `M=diag(B^TGB/sqrt(Lambda), B^TGB/sqrt(Lambda), Id/(2u4))` and constraint `S^T(-B y0+R B y1+z)=w_h`. It minimizes over the vertical freedom of the relative velocity, rather than falsely fixing a nonexistent global double-coset chart. The dual formula is also checked directly by minimum Cartesian gradients solving `D^T grad=p` for the two horizontal endpoint covectors.

At a regular point the horizontal dimension is

\[
 15-\dim(h+Ad_Vh)=15-2\dim h+\dim(h\cap Ad_Vh).
\]

At a centre `Ad_V=Id`, it is `15-dim h`: 12 for the pair, and 14 for the numerically resolved cube. This is a **horizontal slice dimension**, not the dimension of a smooth quotient manifold at that point. The point isotropy `H intersect VHV†` acts on the slice; the local quotient is its orbit/cone quotient. A differential of a smooth invariant scalar must additionally be fixed by that isotropy. The CLI computes connected infinitesimal isotropy and the fixed first-covector subspace. Disconnected invariance is still required; a finite Lie calculation does not settle it. Enhanced-isotropy points cannot be patched with epsilon inverses or treated as ordinary flat rotor coordinates. A stabilizer SU2 or U1 is a gauge isotropy group, not evidence of mechanical spin or electromagnetic charge.

## 4. Isotropic form envelopes and permissible BO uses

For full rank, let `gmin>0,gmax` be the extreme G eigenvalues. Uniformly on all V,

\[
 q_-\,\mathrm{Id}\le Q(V)\le q_+\,\mathrm{Id},\qquad
 q_-={2\sqrt\Lambda\over g_{\max}}+2u_4,
 \quad q_+={2\sqrt\Lambda\over g_{\min}}+2u_4.
\]

Consequently, on the **same Haar measure and residual scalar domain**,

\[
 \kappa_-\int|dF|^2\le\mathfrak t_{\rm tan}[F]
 \le\kappa_+\int|dF|^2,\qquad
 \kappa_\pm=q_\pm/2,
\]

and the quotient inertia eigenvalues lie in `[1/q+,1/q-]`. For an exact nontrivial stabilizer the same bounds, using extreme **positive** G eigenvalues, hold on horizontal covectors: both xi and `R^T xi` lie in the endpoint support. These are form envelopes on residual-invariant functions, not a claim of a regular manifold. The cube envelope is conditional on its resolved germ.

What these bounds can safely do, **after matching potential and domain**:

* bound scalar angular Rayleigh quotients of legitimate residual-invariant trial functions;
* bracket ordered scalar eigenvalues with isotropic comparison operators on that same compact quotient/form domain, by the min–max principle;
* bound local angular kinetic costs in a centre-distance localization trial. A four-centre lift has one scalar quotient well, not four independent scalar multiplicities;
* provide coefficients for rederived angular harmonic comparisons where coordinate/Haar bounds and potential curvature/coercivity have independently been supplied.

What they do **not** inherit from fixed `I=2` heavy-BO receipts:

* full 120-coordinate normal stability or confinement, an isometric normal ground bundle, a uniform normal excitation gap, control of angular/normal cross terms, or the normal Jacobian/drift/geometric potential;
* Born/Berry costs of a moving heavy or light section, matter connections or charged residual equivariance;
* old numeric BO constants, quartet isolation, charged-gap inequalities, phase/thermodynamic conclusions or a particle/spin interpretation.

There is also a material scaling change. With a fixed frame and `ue4` fixed, the angular coefficient now grows as `sqrt(Lambda)` rather than staying constant. For a nondegenerate scalar harmonic potential of order Lambda, conditional angular oscillator energy scales as `Lambda^(3/4)` and width as `Lambda^(-1/8)`, not the old fixed-inertia `sqrt(Lambda)` and `Lambda^(-1/4)`. This is a scaling calculation, not a normal/charged BO theorem. It changes the trial energy and any heavy-complement Schur estimates; one cannot re-label the old receipt by inserting an effective I. Positivity/coercivity facts of the unchanged potential remain facts only for their original frame and domain. Their localization consequences require new legal trial upper bounds.

## 5. Charged matter sections and actual boundary data

For a matter-valued wavefunction, scalar double-coset invariance is replaced by the actual residual representation. Gauge reduction ties orientation derivatives to endpoint matter generators; generally the endpoint forms become anisotropic **covariant** derivatives, not derivatives of an arbitrary scalar F. The principal scalar symbol may remain the expression above on an appropriate bundle, but this does not determine its connection, Born term, stabilizer-allowed fibre or physical domain. Those must come from the source-owned matter embedding and Gauss equations. No extra rotor is added.

The existing `gauss_heavy_bo.py` charged construction is concrete but conditional on its declared angular law: `NH=4,NL=4+q`, `q=0,-1,+1`, and an explicit right external representation `B=1,4,bar4`, respectively. The total centre condition is `(8+q+b)%4=0`; removing B fails in the charged sectors. Its right covariant connection subtracts the actual boundary generators. This is a retained external charged fibre, **not** an odd-total-number singlet on a closed graph, and its four colour indices are not four physical flux vacua. The Cartesian quotient supplier neither erases B nor manufactures it from G. Deriving B as a real exterior dressing, selecting boundary versus total-centre sectors, and proving the charged Gauss/BO operator on that domain belong to the joined consumer. A scalar metric alone cannot establish them.

For the pair SU2 stabilizer or resolved cube U1 stabilizer, the actual restriction of the microscopic matter/boundary representation to the **full** stabilizer must be imposed, including its disconnected centre components. The scalar kernel dimension does not specify that representation or identify physical spin.

## 6. Executable controls, input provenance and next join interface

Parent can execute from the repository root:

```sh
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/cartesian_angular_quotient.py \
  --Lambda 1000000 --ue4 .3 \
  --input-receipt proposals/P256-qed-backward/unrestricted_field_minimum_receipt.json \
  --output /tmp/issue222-cartesian-angular-quotient.json
```

Dependencies are existing NumPy/SciPy and the inherited proposal modules/proofs; no new scientific package or changed shared file is needed. The CLI builds the actual owner model, emits finite strict JSON, and checks source/proof/input bytes unchanged during execution. It evaluates all three frames at the identity centre and the actual SU4 matrix `exp(i sum_a .71 sin(.37a+.29) T_a)`, verifying determinant/unitarity and a nonzero distance from every centre. No scientific CLI was run by the delegated author mid-flight; parent owns observed results.

Controls are designed to reject plausible wrong implementations:

1. actual generator trace norm and fundamental Casimir, plus central differences of actual matrix conjugation with quadratically shrinking error;
2. independent Cartesian and HS/4 G constructions, algebraic Clifford spectra, and failure of using B instead of the normalized T;
3. independent KKT constrained velocity minimum, inverse-metric dual and direct minimum ambient Cartesian-gradient form;
4. wrong additive-inertia inverse, and the exact 75% endpoint-cometric error caused by forgetting HS/4;
5. finite independent endpoint gauge transformation, common-conjugation zero relative velocity versus its nonzero unreduced cost, and residual-centre gauge equivalence;
6. horizontal null spaces, point isotropy and connected smooth first-covector restrictions at centre and noncentral points; the cube discarded tangent is printed, not hidden;
7. isotropic cometric envelope comparisons at both points. The all-V envelope is the algebraic inequality above, not inferred from these two probes.

Concrete join API: `tangent_metric(frame, owner.basis, owner.generators)` returns raw G, D, support/kernel and the resolved inverse; `quotient_principal(metric, V, owner.generators, Lambda=..., ue4=...)` returns ambient Q, horizontal basis, `horizontal_cometric`, `horizontal_inertia`, and isotropy/covector data as NumPy arrays. The parent supplies its actual periodic supplier frame or actual pair frame; it must label exact versus numerical stabilizers. A joined scalar consumer uses the Haar residual-invariant form and envelopes directly. A joined matter consumer must replace its old endpoint `1/(2I)` weights by `sqrt(Lambda) G+ /2` on the allowed support, retain the link coefficient ue4 once, and independently derive/project its actual Gauss connections, Born/normal terms and boundary representation. This is the principal-symbol interface, not a completed charged implementation or permission to transfer old constants.
