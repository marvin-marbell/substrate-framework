# A compact full-coordinate packet of the actual full occupied sea

`full_band_packet_bound.py` supplies a normalized quantum trial in the actual two-cell field Hilbert space, not just a local section or selected-orbit rotor. It retains all 120 independent real canonical field coordinates, one genuine SU(4) Haar link and two untruncated compact EM phases. The matter factor is the unchanged 4900-state primitive Pauli `FullBandGaussSection`; no CAR replacement, frozen heavy state, gauge fixing or mean-charge substitution occurs.

## Law and first discriminating probe

The scope is the existing one-x-bond/two-parallel-EM-path owner, not a cubic lattice extension. Write `Phi[x,mu]=sum_a u[x,mu,a] B_a`, with `Tr B_a B_b=4 delta_ab`, and `T_a=B_a/sqrt(8)`. Its Hamiltonian quadratic form is

\[
 H_\Lambda=\frac{\sqrt\Lambda}{2}\sum_{120}\Pi^2
 +\Lambda\frac\kappa2\sum_x\|G_x\|_F^2
 +\frac{\Lambda J}{8}W-\Lambda\,d\Gamma_H(h_H)
 +d\Gamma_L(h_L)+u_4\sum_{15}E_a^2
 +\frac{u_1}{2}(e_0^2+e_1^2)-K\cos(a_0+a_1),
\]

where `G[x,mu,nu]=Tr(Phi[x,mu] Phi[x,nu])/4`, and **exactly**
`W=sum_mu ||Phi[0,mu]-U Phi[1,mu] U†||HS²`. The onsite and path coefficients are read from `FullBandGaussSection.onebody`, including its one-axis onsite adjustment and light-only two-phase factor. Heavy polarity is the whole `-Lambda h_H`, not just an onsite sign. The full Cartesian field kinetic term replaces the selected-orbit inertia rotor: adding that rotor separately would count field motion twice. There is no colour plaquette in this one-link geometry. The field stiffness is the bare zero-preferred-Gram quartic, not the selected-frame restoring Hessian.

Defaults are the shared `m=.2,r=.12,v=.3,kappa=2,J=.2,u4=.3,u1=.4,K=.7`. The anchor uses the existing size-three selected frame, **but** the actual link is
`U*=exp(i sum_a .13 sin(.37a+.29) T_a)` and the actual phases are `(.137,-.081)`. The first cheap calculation evaluates the full quartic, exact alignment, and the occupied top-four sum of the entire heavy one-body operator. It gives

- quartic plus alignment: `0.3753875937442592`;
- whole-heavy coefficient: `-0.8420197863824028`;
- leading sum: `-0.46663219263814365`.

This is an observed finite anchor calculation, not assumed selected-frame negativity. It survives, so no coefficient change is needed. The CLI also contains a materially different onsite-only anchor route if a user's coefficients make this first sum nonnegative; that route must independently pass the same uniform inequalities or fail loudly. No negative-energy receipt is emitted when a certified inequality fails.

## Exact normalized packet, without Gaussian tails

Let `R,eta,tau>0`, with `eta,tau<.1`. Define a real compact amplitude `b(s)=(1-s)_+²`. The seed amplitude is the product of

1. `b(||u-u*||²/R²)` on the **full** 120-dimensional canonical ball;
2. `b(||U-U*||HS²/eta²)` on the Haar link;
3. `b(wrap(a_j-a_j*)²/tau²)` on each of the two circles.

The respective density normalizers are exactly

\[
 Z_F=\pi^{60}R^{120}\frac{\Gamma(5)}{\Gamma(65)},\qquad
 Z_{A}=\frac{\tau B(1/2,5)}{2\pi},
\]

and the finite positive Haar integral `Z_U=int_SU4 b(||U-I||HS²/eta²)² dU`. An explicit computable three-angle representation is

\[
 Z_U=\frac1{24(2\pi)^3}\int_{[-L,L]^3}
 \left(1-\frac{\sum_{i=1}^4 4\sin^2(\theta_i/2)}{\eta^2}\right)_+^4
 \prod_{i<j}4\sin^2\frac{\theta_i-\theta_j}{2}\,d^3\theta,
 \quad L=2\arcsin(\eta/2),\quad\theta_4=-\theta_1-\theta_2-\theta_3.
\]

For `eta<.1`, all support eigenangles are near zero and their sum is zero, so there is no omitted determinant branch. `haar_link_normalizer` reports diagnostic tensor Gauss-Legendre estimates, **not** a replacement for this exact integral or a certified cubature error. `log_seed_density` evaluates the seed in log form given the exact normalizer; the log avoids underflow of `R^120`. The seed alone is not physical.

Let `p0` be that normalized product density and average over the compact **physical gauge action** `G=SU4_0 x SU4_1 x U1_0 x U1_1`:

\[
 p(q)=\int_G p_0(g^{-1}\cdot q)\,dg,\qquad
 f(q)=\sqrt{p(q)},\qquad \Psi(q)=f(q)\Omega(q).
\]

Explicitly, `Phi_x -> g_x Phi_x g_x†`, `U -> g_0 U g_1†`, and `a -> a+(theta_0-theta_1)(1,-1)`. All measures are invariant; hence Fubini gives `int p=1` **exactly**, and `||Omega(q)||=1` on the certified support below. This defines an actual normalized scalar packet by compact finite integrals, not a finite orbit sample, a fixed representative or an unnormalized projection. It is unnecessary to numerically integrate the gauge average to establish its normalization or variational bound. All 120 coordinates have an open positive-density neighborhood; none is frozen. Its support is the compact union of gauge images of the seed support. It never extends beyond that support into a crossing or zero-overlap locus.

The true section's finite covariance gives `Omega(gq)=R(g)Omega(q)`. Since `f(gq)=f(q)`, both colour Gauss equations and both light-only EM Gauss equations hold as vector equations. The heavy remains EM neutral. Gauge averaging is a scalar operation on full-coordinate probability, not replacing physical Gauss by gauge fixing. The boundary powers give an H1 packet, sufficient for the Hamiltonian form domain; no unjustified global analyticity is claimed.

## Uniform gap and overlap certificate

Put `o=max_s |onsite_s|`, `p=max_s ||path_s||2`, `A=max_(s,x)||sum_mu path[s,mu] Phi*[x,mu]||HS`, and `dF=2(o+p)`. On the complete seed support,

\[
 \|h_s(q)-h_s(q_*)\|_{op}\le\delta=d_F R+A(\eta+\tau).
\]

This uses the actual endpoint average, unitary norm, `|light_phase_factor|<=1` and its Lipschitz bound; nonlinear perturbed-field/link terms are retained by splitting the difference with an exact unitary endpoint. Each local temporal matrix changes in operator norm by at most `2R`. Consequently Weyl gives uniform gaps

\[
 g_L=g_{L,*}-4R>0,\qquad g_B=g_{B,*}-2\delta>0.
\]

The separated rank projector satisfies `||dP||op<=2||dh||op/g`. Its rank-n exterior filled projector satisfies `||dPi||op<=n||dP||op`; the code's occupation polynomial is precisely this projector on the relevant fixed-number sector. For the local paired initial state the two `sqrt(6)` factors give the deliberately conservative product norm six. Telescoping the actual local and global projector products therefore gives

\[
 \|raw(q)-raw(q_*)\|\le D=96R/g_L+16\delta/g_B,
 \qquad w(q)\ge w_{min}=(\sqrt{w_*}-D)^2,
 \quad \sqrt{w_*}>D.
\]

These inequalities are deterministic operator estimates, **not** extrema inferred from finite samples. They certify the section chart on the entire support and all its gauge images. The implementation rejects a failed Weyl or positive-overlap inequality; it never swaps to a frozen-heavy or arbitrary-chart fallback.

## Actual kinetic costs, including Berry terms

For a normalized complex section, `Re<Omega,dOmega>=0`. The real scalar envelope therefore has **zero cross term** in the norm of `d(f Omega)`, even when `Im<Omega,dOmega>` is nonzero. No paired Berry cancellation from the simpler local bundle is imported: the entire `||dOmega||²`, including its Berry part, is bounded here.

For the normalized seed, the 120-coordinate ball Fisher cost is exactly `5120/R²`. More generally a D-ball amplitude `(1-r²/R²)^p` gives `p D(D+4p)/(2(2p-1)R²)`; here `D=120,p=2`. Each one-dimensional phase cost is exactly `3/tau²`, hence both cost `6/tau²`.

For the link bump, the Haar metric uses `Tr T_a T_b=delta_ab/2`. If `q=||U-I||HS²`, then `sum_a (E_a q)²<=2q`. Thus the unnormalized bump derivative is at most `8/eta²`. To bound its normalization ratio, exponential HS coordinates have Jacobian
`prod_(i<j) sinc((lambda_i-lambda_j)/2)²`. On the HS ball of radius `eta/2` it is at least `j=(sin(eta)/eta)^12`, and the bump density is at least `(3/4)^4`. Every chordal-support point has principal-log HS distance at most `pi eta/2`; the exponential Jacobian is at most one. Cancelling the common Euclidean volume constant gives the explicit bound

\[
 K_U\le\frac{8\pi^{15}}{\eta^2(3/4)^4 j}.
\]

This is intentionally loose but finite, computable and independent of uncertified quadrature. The normalization constant of Haar measure cancels in the volume ratio. Fisher convexity (Cauchy-Schwarz under the gauge integral) implies `int |grad sqrt(int p0(gq)dg)|² <= int_G int |grad sqrt(p0(gq))|²`. Gauge maps are isometries of each field/link/phase metric, so the same seed bounds apply to `f` separately in each kinetic sector.

For the actual full section use `A_s=A+2pR`, which includes the endpoint perturbation, and the certified gaps above. Differentiating both local projectors, both whole-band projectors and the actual normalization gives the following **per-direction** upper bounds:

\[
 D_F=\frac2{\sqrt{w_{min}}}(48/g_L+16d_F/g_B),\quad
 D_U=\frac2{\sqrt{w_{min}}}\frac{16A_s}{\sqrt2 g_B},\quad
 D_A=\frac2{\sqrt{w_{min}}}\frac{4A_s}{g_B}.
\]

The colour tangent has `||T_a||op<=1/sqrt(2)` and each light phase derivative has half the endpoint coefficient. Thus field, link and phase metric costs are respectively `120 D_F²`, `15 D_U²`, and `2 D_A²`. No finite derivative sample is used to certify these upper bounds.

## Full-field potential and quantitative energy bound

Let `rho*=||u*||2=||Phi*||HS/2`, and regard `G*` as the two-block Gram vector. Full field, not selected-frame, estimates give

\[
 Q=\frac\kappa2(\|G_*\|_F+2\rho_*R+R^2)^2,
 \quad L_W=\sqrt{W_*}+2\sqrt2R+2\|\Phi_{1,*}\|_{HS}\eta,
 \quad V_W=JL_W^2/8.
\]

The first inequality follows by bounding `d(uu^T)` in Frobenius norm. The second uses both endpoint changes and both conjugation changes; it bounds the exact `W`, not `PUP` products. Weyl bounds the occupied four-level heavy energy by `H*= -sum_top4 eig(h_H*)+4delta`; the light energy similarly by `L*=sum_bottom4 eig(h_L*)+4delta`. Define

\[
 C=Q+V_W+H_*,\quad
 T=(5120/R^2+120D_F^2)/2,
\]
\[
 B=u_4(K_U+15D_U^2)+\frac{u_1}{2}(6/\tau^2+2D_A^2)
 +K+\max(0,L_*).
\]

The magnetic term is bounded above by `K`; no unjustified zero mean is assumed for this phase-localized packet. Gauge invariance propagates every estimate to the complete physical support. Therefore

\[
 E_0\le E_{trial}\le C\Lambda+T\sqrt\Lambda+B,\qquad\Lambda\ge1.
\]

If `C<0`, the explicit threshold is

\[
 \Lambda_* =\max\left(1,
 \left[\frac{T+\sqrt{T^2+4(-C)B}}{2(-C)}\right]^2\right),
 \qquad \Lambda>\Lambda_*\Longrightarrow E_{trial}<0.
\]

A small discriminating calculation of the analytic constants at `R=eta=tau=.001` observed `w_min=.7080872858957358`, `g_L=1.2927966503006638`, `g_B=.3361841761874783`, `C=-.4591787304049673`, `T=2562333057.1144714`, `B=217377780197342.12`, and `Lambda*=3.114015987812689e19`. The bound at `4 Lambda*` is approximately `-2.8598015532672205e19`. This large conservative threshold is a genuine finite variational result, not a prediction that the Lambda-one shared finite EFT has the same full-field ground.

For an invariant-amplitude consequence, the crude endpoint-path operator estimate gives `|dGamma_s(h_s)|<=c sum_x rho_x`, where `c=8(o+p)` is safe for this one-bond geometry and `rho_x=||Phi_x||HS/2`. The nonnegative kinetic/quartic/alignment terms and magnetic lower bound imply `H >= -(Lambda+1)c sum_x rho_x-K`. Hence a ground (or minimizing sequence with the same energy control) obeys

\[
 \sum_x\langle\rho_x\rangle\ge
 \max\left(0,\frac{-C\Lambda-T\sqrt\Lambda-B-K}{(\Lambda+1)c}\right),
 \quad \liminf_{\Lambda\to\infty}\sum_x\langle\rho_x\rangle\ge -C/c.
\]

This is an average-amplitude consequence of a trial upper bound, **not** local amplitude order, selection of a Clifford phase, a solved full ground wavefunction, a charged lower gap, physical mechanical half-spin, Lorentz symmetry, a three-dimensional photon or scattering.

## Execution and evidence boundary

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python proposals/P256-qed-backward/full_band_packet_bound.py \
  --output proposals/P256-qed-backward/full_band_packet_bound_receipt.json
```

The parent executes this final CLI at the composed validation boundary. It captures supplier and own Python/proof hashes before constructing the law, checks them again before writing strict portable JSON, emits all analytic constants, and exercises the actual section on a non-Clifford point inside the packet. `packet_point_controls` independently checks all thirty primitive colour Gauss equations, both EM Gauss equations and both whole occupied-energy residuals. A scoped scientific calculation observed maxima `9.97e-16`, `6.01e-16`, and `8.88e-16`, respectively, and rejected a wider `R=.1` packet at the precise positive-overlap inequality. These finite controls support the executable implementation, not the uniform proof. The separately accepted section supplier owns finite covariance, current/contact and frozen/missing-heavy adverse controls; this new CLI does not rerun those unrelated controls. Native Pyright symbols/definition and AST spectral-call discovery were exercised before implementation; the new source's post-edit native diagnostic report is empty. The Haar normalizer quadrature is labelled diagnostic and cannot be mistaken for a chart or energy certificate. There is no asserted full-coordinate spectrum computation.

## Parent-owned execution and material packet refinement

The final default command completed naturally in 12.44 s. Its receipt binds
source `9b8e6f0d4c20d875389bd38d76d273be8d431a18cbe0331f9925daa72a8450d9`
and this proof's execution-time hash
`73d5549537456a22bcc94045ac5c235c9f795e93b6c4a36ff699f467efcdc953`.
`ReviewFullBandPacket` independently accepted the source, uniform argument
and subsequently inspected parent receipt; it did not run the CLI.

Changing only the **packet support**, not the bare law, lowers the conservative
threshold. Link and phase radii remain `.001`:

| Field radius | Certified coefficient C | Field kinetic coefficient T | Positive overlap lower | Strict negativity threshold |
|---:|---:|---:|---:|---:|
| .001 | -.4591787304049673 | 2562333057.1144714 | .7080872858957358 | 3.114015987812689e19 |
| .004 | -.4394261640811597 | 167525305.69966444 | .22565252292738233 | 1.4632899008663104e17 |
| .006 | -.42619579164976595 | 105103295.06278944 | .050893409768409184 | 6.183138628248723e16 |

The `.004` and `.006` CLIs completed naturally in 12.30 s and 11.99 s,
respectively; their receipts are `full_band_packet_bound_r004_receipt.json`
and `full_band_packet_bound_r006_receipt.json`, with the same source/proof
hashes. The `.006` uniform local/global gap lower bounds remain positive,
`1.2727966503006638` and `.3265530773031976`. Its constant term is
`B=217377780201991.1`; the bound at four times its threshold is
`-5.292193103113624e16`. Wider support reduces the envelope Fisher cost while
weakening the overlap bound, rather than pretending the derivative cost vanished.

The exact Haar normalizer remains the proved integral. The default order-12
and order-20 estimates, `3.811111326065381e-55` and `3.682760328656989e-55`,
are diagnostic cubature, not normalization or uniform-bound certificates.
All full-phase, lower-gap and mechanical-spin exclusions above remain open.
