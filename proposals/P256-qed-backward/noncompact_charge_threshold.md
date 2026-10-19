# Noncompact Coulomb Gaussian gives a massive threshold across harmonic fibres

## Explicit operator and boundary

Continuing issue222 campaign, work campaign/outcome/verification, pinned general `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`. This is a construction on the **different real-link quadratic-curl electromagnetic law**, not a proof of compact native PH-01. It reuses the actual fixed metric-Clifford native light endpoint and full CAR. It does not import a Dirac propagator.

On a finite connected graph, eliminate the **real longitudinal Gauss orbit**. Let `L=DD^T`, `G=L^+`, background two, `n_x` the complete local number, and reference charge `-q` at R. With every `NL=2V+q` state retained,

\[
 H_q=UE(p_T^2+p_h^2)+\frac K2\|CA_T\|^2
       +UE(n-2-q\delta_R)^T G(n-2-q\delta_R)+d\Gamma(h(A_T,A_h)).
\]

This is the physical noncompact quotient, not an empty literal invariant kernel in unreduced longitudinal L2. `K,UE>0`, fixed full positive metric G_field, `m>0,r>=0`, all actual central colour holonomies retained. No reference self energy is subtracted. The body colour/reference equivariant lift is not numerical full colour-field dynamics.

Require the curl quadratic form to be positive on the retained coexact subspace. Its kernel defines the harmonic space. Choose a full real harmonic period lattice on which all microscopic link phases repeat. Integer D and C provide a rational kernel and such a lattice; choosing a nonminimal period cell is harmless. Harmonic coordinates remain R and the full physical operator is a direct integral over boundary twists theta. For each fibre its coexact oscillator confinement and compact harmonic cell give compact resolvent. The coexact space is not compactified.

Define

\[
 \beta=\frac{\Phi_0-b_j\Phi_j}{\sqrt s},\quad
 b=G_{sp}^{-1}G_{sp,0},\quad
 s=G_{00}-G_{0,sp}G_{sp}^{-1}G_{sp,0},\quad\mu=m\sqrt s.
\]

The original fixed-metric endpoint has `h=beta M+Dodd`, `M>=mu`, `{beta,Dodd}=0`, and a fixed chirality anticommuting with h. See `original_uniform_charge_gap.md` §§3–5 for the **derived** arbitrary-temporal-background inverse and positive determinant, including its identity source jump and canonical background factor. Those arguments require bounded h, not compact spatial angles; `exp(iA_e)` keeps h bounded on real links.

## 1. Exact reduced Coulomb temporal factor

For every complete neutral or charged occupation and `dt>0`,

\[
 e^{-dt UE\rho^T G\rho}
 =\mathbb E_Z e^{i\sqrt{2dtUE}\,(G^{1/2}Z)\cdot\rho},
 \qquad\rho=n-2-q\delta_R.
\]

This finite-dimensional Gaussian identity is exact. Temporal phases are unitary and commute with beta. The neutral background contributes `exp(-2i integral sum a0)`; the static reference during a charged interval contributes `exp(-iq integral a0_R)`, a unit-modulus factor. The charged electric cost is present, not removed by that factor.

The physical canonical neutral trace still requires the constant-site holonomy Fourier integral `z^(-2V) det(I+zU)`; a Gaussian in the range of L+ alone does **not** project NL. The full normalized determinant is strictly positive for every global holonomy and temporal Gaussian realization by the original coercivity/homotopy proof. Thus the canonical neutral fermion/photon integrand is nonnegative.

The positive coexact oscillator heat kernel and periodic-zero-twist harmonic heat kernel, with the real quadratic potential, provide the remaining neutral Brownian weight. Matter and Coulomb are bounded on a fixed graph; harmonic oscillator forms and bounded perturbations give the actual Hamiltonian Trotter/Feynman–Kac limit. No cosine potential or longitudinal gauge-volume trace is substituted.

## 2. Neutral global infimum is the zero harmonic twist

For a harmonic twist theta, resolve the neutral heat path into harmonic winding classes. Relative to the zero-twist path integrand, each class gains only `exp(i theta.winding)`. All zero-twist weights, **including the canonical normalized determinant**, are nonnegative. Hence

\[
 0<Z_{0,\theta}(T)\le Z_{0,0}(T).
\]

The left side is the actual trace of a selfadjoint fibre Hamiltonian. Taking its low-temperature spectral exponent gives `E_{0,theta}>=E_{0,0}`. This is an interacting statement derived from the positive neutral measure; it does not follow from a scalar-photon diamagnetic analogy alone. Consequently the full noncompact neutral infimum is `E0=E_{0,0}`. A fibre ground is not a normalizable full-R ground; narrow normalizable Bloch packets approximate the infimum as in `noncompact_neutral_supplier.md`.

## 3. Charge can change harmonic fibre without changing Gauss

For each theta take a bounded smooth charged preparation `B_theta` from the periodic neutral fibre into the theta charged fibre. In a harmonic-cell trivialization it includes the unit-modulus section `exp(i theta.y/period)` multiplying a periodic full-CAR/Wilson preparation. This is legal on noncompact links. It is a change of harmonic boundary fibre, not a fictitious periodic fractional Fourier state or a replacement of the actual point-source sector.

In the finite-temperature correlation, the charged interval's winding phase and the section endpoint factors combine into a unit-modulus factor. The underlying normalized denominator and Brownian/temporal determinant measure remain those of the **periodic neutral fibre**. Fermionic insertion Wick lines use the same arbitrary-background inverse. A net charge one forces a line to cross the interval; its derived bound is `C_gamma exp(-gamma t)` for every `gamma<mu`, uniformly in backgrounds and twists. Other finite CAR contractions and bounded coefficient/reference transports contribute a time-independent prefactor. Therefore

\[
 \frac{\operatorname{Tr}_0[e^{-(T-t)H_{0,0}}B_\theta^\dagger
               e^{-tH_{q,\theta}}B_\theta]}{Z_{0,0}(T)}
 \le C_{B,\gamma}e^{-\gamma d_T(t,0)}.
\]

This avoids dividing by the possibly smaller twisted-neutral trace. A proof performed in an unrelated charged sector, or a bound using `Z0(theta)` and silently dropping its ratio, would not establish the all-fibre result.

## 4. Spectral conclusion and dense range

At fixed graph the periodic neutral ground eigenspace is finite dimensional. Take T to infinity first. The resulting source spectral measure relative to `E_{0,0}` is positive; its Euclidean exponential bound excludes support below gamma.

The oscillator/quadratic and endpoint/Coulomb coefficients are analytic, uniformly elliptic with finite CAR matrix dimension, on `R^coexact x T^harmonic`. Analytic elliptic regularity makes each nonzero neutral ground vector nonzero almost everywhere (choose a nonzero analytic component). Smooth bounded covariant coefficient maps, with compactly supported coexact cutoffs and a theta section, approximate arbitrary charged vectors divided by that neutral vector. The **entire** original finite CAR algebra supplies every neutral-to-charged matrix unit with net charge q, and all four actual reference components. Real longitudinal dressing supplies any total-zero local character; colour Wilson transport supplies the actual reference covariance. This is a dense-range proof, not an assumption that a single point creator reaches the lowest charged state.

The source bounds therefore force the full charged spectral projection below `E_{0,0}+gamma` to vanish in each theta fibre. Taking gamma up to mu and then the direct-integral infimum gives the effective untruncated target

\[
 \boxed{\inf\sigma(H_{\pm1})-\inf\sigma(H_0)\ge m\sqrt s.}
\]

An analytic common scalar potential with the same form confinement can be retained in the positive path weight. A non-scalar colour/light angular Hamiltonian is **not** such a scalar; restoring full microscopic fields requires the actual lower/neutral trial BO transfer, not this theorem's renaming. If the actual original transfer errors extend to this positive quadratic photon form, well by well the threshold loses their stated errors and then the minimum actual Schur mass is taken with common offsets. That transfer is a distinct obligation here.

## 5. Joining to the noncompact upper construction

`noncompact_coulomb_carrier.md` and `noncompact_neutral_supplier.md` already derive an unfiltered complete-reference compensated point upper bounded by `(13/4)UE+B*(a)` along their admitted Lambda(N) family, with actual state/domain/BO errors retained. This threshold construction provides the missing effective **positive lower**, including all harmonic fibres. Together they are a candidate positive finite compensated charge window, uniform in reference separation on the 3D graph when the metric-family and full microscopic transfers are actually supplied. They are not a mobile isolated particle, a total-spin eigenspace or a photon S-matrix.

For a fixed finite-width packet, positivity of `Hq-E0` would additionally permit Cauchy–Schwarz control of cross-site point source energy forms rather than silently assigning the point upper to every complex packet. For K marked components the actual initial Gram still belongs to `noncompact_marked_metric.md`; source-mode/current/derivative and long reference costs remain explicit.

## 6. Actual consumer and limitations

`noncompact_charge_threshold.py` executes arbitrary **real unwrapped** native N5 time backgrounds at centre123 with all ten Gram entries retained, the reused source-jump/positive-det/canonical controls, and all64 Schur formulas. It separately checks the exact reduced-Coulomb Gaussian factor on the whole70 neutral sector; omitted background two fails that same assertion.

Its whole quantum two-vertex diagnostic scans neutral and both charged numbers across harmonic twists. It retains the complete reference, exact nonlinear phases, oscillator and harmonic Galerkin space, actual source-addressed sector/residue, and actual full-H Euclidean source heat. Finite Galerkin charged-minus-neutral Ritz differences are **observations**, not certificates of the infinite-H gap; their enlarged residuals and truncated positivity limitation remain visible. Source addressing is not inferred from the global charged ground.

Parent invocation:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/administrator/substrate-framework/.venv/bin/python \
 noncompact_charge_threshold.py --n 5 --M 3 --zetas -.4 0 .4
```

Scientific outputs require the parent's exact-source receipt. Independent review must specifically check the charged-interval twist phases, dense range and unbounded-form limits; the compact proof alone is not review of this extension. The continuing electron/photon outcome remains active.
