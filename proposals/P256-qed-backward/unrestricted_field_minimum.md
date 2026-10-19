# Actual Cartesian field competition, without an assumed selected ground

This is a same-law competition consumer, **not** a full-field ground/phase
certificate. `unrestricted_field_minimum.py` owns no imported physics. Its
scientific execution is reserved for the parent integration boundary; numerical
stationarity, Hessian and quantum-packet values are earned only by its JSON
receipt, not by the formulas or the existence of this file.

## Domain and source ownership

The authoritative basis is `sea_selection.matrix_unit_basis`: fifteen
**traceless Hermitian** matrices, `Tr(B_a B_b)=4 delta_ab`. Thus each of the four
orbital labels has fifteen unrestricted real coordinates; a site has sixty.
Identity matrices are not legal competitors. There is no fixed Gram budget,
orthonormal-frame constraint, imposed Clifford algebra or additional field term.
The CLI verifies traces, Hermiticity and normalization before constructing a
competitor. `FullBandGaussSection.onebody` independently rejects a field outside
this domain.

The local potential is precisely the one in
`full_band_packet_bound.anchor_probe` and its companion derivation:

\[
 V_F=\frac\kappa2\sum_x\|G_x\|_F^2+
 \frac J8\sum_{(x,y),\mu}\|\Phi_{x\mu}-U_{xy}\Phi_{y\mu}U_{xy}^\dagger\|_{HS}^2,
 \qquad G_{x,\mu\nu}=\frac14\operatorname{Tr}(\Phi_{x\mu}\Phi_{x\nu}).
\]

Here `kappa=2`, `J=.2`, zero preferred Gram, `a=1,m=.2,r=.12,v=.3`,
and the unimproved law are fixed in the consumer. No optimizer can retune them.
The field kinetic term is the declared full-Cartesian
`sqrt(Lambda)/2 sum Pi^2` law of `full_band_packet_bound.md`; this file does not
add a selected-orbit inertia rotor, infer its metric from a Hessian or identify
it with the independently declared isotropic angular EFT inertia.

The two-cell consumer uses the exact `FullBandGaussSection.onebody` with its
one-axis onsite `m+r=.32`, the actual x path and its adjoint, zero compact phases,
and identity colour link. **Both** endpoints' four matrices are varied:
120 independent coordinates. Its leading functional is

\[
 F_2(\Phi)=V_F(\Phi,I)-\sum_{j=5}^{8}\lambda_j(h_H(\Phi,I)).
\]

The three-axis comparison uses `centre_flux_sea.onebody` on the open
`2 x 2 x 2` cube, all eight sites, all twelve edges and onsite
`m+3r=.56`. It varies every one of the sixty coordinates of a uniform frame,
not every one of the cube's 480 independent site coordinates:

\[
 F_8(\Phi)=8\frac\kappa2\|G\|_F^2-
 \sum_{j=17}^{32}\lambda_j(h_H(\Phi_x=\Phi,U_e=I)).
\]

The vanishing alignment here is a value of the **same** potential at uniform
fields and identity links, not removal of its term. The code also evaluates
that potential after independent local gauge transformations, when the
transformed cube fields are no longer uniform. There is no four-band momentum
replacement, absolute-value occupation or assumed +/- eigenvalue pairing.
The distinct onsite coefficients are deliberate geometry semantics, not a
coupling change.

`source_sha256` binds this note, the consumer and each named physics/proof owner
at both ends of an execution. A source change during execution rejects the
receipt. Source owners are `FullBandGaussSection.onebody`,
`centre_flux_sea.onebody`, `full_field_charge_bound.primitive_edges`,
`local_orbital_stability.scalar_paths`, `RotationBand.couplings`, and the
primitive `PauliCode.element`/`EdgeWords` operations. The governing shared plan
is `issue-222-continuing-backward-20260930`, assignment revision
`d7cfa8de60401582b6b901673280c7baeb7ca13556cfcf4f4ee7b80016cb94f1`,
pinned general template
`89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`.

## An exact failure of the environmental frame in the pair

`uniform_data(band,3,2,0,basis)` selects a trial for a periodic three-axis
problem. Write its frame as `(b Gamma_0,s Gamma_1,s Gamma_2,s Gamma_3)`.
The one-axis pair heavy operator depends only on `Phi_0,Phi_1`; **all four**
labels still enter its actual local quartic. Consequently the admissible
competitor `(b Gamma_0,s Gamma_1,0,0)` has exactly the same whole-heavy energy
and, at identical endpoints and identity link,

\[
 F_2(\text{selected})-F_2(\text{trimmed})=2\kappa s^4>0.
\]

The actual gradient at each endpoint in either omitted spatial label is
`2 kappa s^3 Gamma_mu` in canonical coordinates. The joint omitted-label
coordinate norm is `4 kappa s^3`. This is an exact same-potential descent,
not exclusion of those directions from the domain. The CLI reports the full
120-coordinate selected gradient and checks its spectral derivative. On the
cube all three spatial labels really enter the heavy operator; the deletion is
only another competitor there, **not** this obstruction or a three-dimensional
verdict.

## An explicit improved, fully stationary pair reference

Use the legal uniform two-field Clifford trial
`Phi_0=b Gamma_0,Phi_1=s Gamma_1,Phi_2=Phi_3=0`. Read, rather than prescribe,
`c=.32,d=.06,t=.15` from the pair owner's onsite and x path. Its heavy operator
is the sum of the commuting operators

\[
 A=c b\,1\otimes\Gamma_0+t s\,\sigma_y\otimes\Gamma_1,
 \qquad B=-d b\,\sigma_x\otimes\Gamma_0.
\]

`A^2=R^2`, `R=sqrt(c^2 b^2+t^2 s^2)`, and `c>d` implies `R>d|b|`
for the nonzero optimum. The eight levels are `+/- R +/- d b`, each twice;
the actual top-four sum is `4R`, **including** the Wilson hopping. This spectral
identity is restricted to this derived reference, never assumed for competitors.
Its same-law energy is

\[
 F_{2,\mathrm{Cl}}=\kappa(b^4+s^4)-4R.
\]

In `g_0=b^2,g_1=s^2`, this is a strictly convex quadratic minus a square root;
its interior equations give the exact restricted global minimum

\[
 R_*^3=\frac{c^4+t^4}{\kappa}=
 \frac{1099201}{200000000},\qquad
 b_*^2=\frac{32}{625R_*},\qquad s_*^2=\frac9{800R_*},\qquad
 F_{2,\mathrm{Cl},*}=-3\left(\frac{1099201}{200000000}\right)^{1/3}.
\]

It is strictly below the optimum of the onsite-only balanced trial because
`t>0`. Moreover `P_top=(1+A/R_*)/2`: contraction with **every** source-owned
Cartesian derivative leaves only the displayed `Gamma_0,Gamma_1` gradients,
which the quartic stationarity equations cancel. Thus this reference is
stationary in the full 120-coordinate pair functional, not only in the two
amplitudes. The CLI explicitly checks that residual; its analytic Hessian,
not the amplitude derivation alone, decides transverse stability.

Two nonzero anticommuting irreducible two-label factors in four dimensions
have complex common commutant `M_2(C)`, dimension four, with connected SU(4)
stabilizer of dimension three. The absent environmental labels cannot secretly
complete the frame. This stabilizer is not exactly the four-element centre;
no exactly-four-centre Higgs, vacuum or angular isolation statement is inherited.
The current consumer reports common commutant rank/singular values for every
refined candidate as well as this reference.

## Full-direction refinement, derivatives and stability receipts

For linear source-owned `h(u)=sum_a u_a D_a`, the separated top-half projector
`P` gives the exact derivative `-Re Tr(P D_a)`. The sea Hessian is

\[
 (H_H)_{ab}=-2\operatorname{Re}\sum_{e\in\mathrm{empty},o\in\mathrm{top}}
 \frac{\overline{\langle e|D_a|o\rangle}\langle e|D_b|o\rangle}
 {\lambda_o-\lambda_e}.
\]

The code adds the full quartic product-rule Hessian and actual endpoint
alignment Hessian. Occupied-occupied degeneracy is harmless; a nonpositive
occupied-empty gap explicitly rejects the Hessian. Two deterministic
all-coordinate derivative directions check both the gradient and Hessian
against independent central differences at the selected and best frames.

Six explicit deterministic seeds include the environmental frame, the derived
Clifford trial, onsite-only, commuting traceless matrices and two sine-pattern
perturbations in **all** permitted coordinates (including asymmetric endpoint
perturbations for the pair). L-BFGS supplies only an estimate. The consumer then
repairs the actual gradient with a Hessian-based descent and independent
residual criterion. A materially negative Hessian direction triggers bounded
deterministic descent and another residual repair. Stalled repairs are visible
rejections, never optimizer-success claims.

Receipts include energy decomposition, coordinates, Gram, middle-cluster gap,
stationarity residual, common SU(4) stabilizer/complex commutant dimensions,
full Hessian, Hessian normal to the actual common gauge orbit, and negative/flat
counts. A stationary saddle stays a stationary saddle if the bounded escape
search does not remove it. Zero transverse Hessian directions can require a
quartic analysis; they are not a positive gap. A list of converged seeds proves
neither a classical global minimum nor a quantum phase.
Numerical commutant dimensions are tolerance-bound singular-value observations;
the exact two-field reference has the separately derived dimension above.

There is also a genuine unrestricted **coercive lower comparison**. If
`rho=||u||`, `Tr h(u)^2=u^T K u <= L rho^2`, and `N=dim h`, the trace-zero
half-filled Ky Fan inequality gives `sum_top lambda <= sqrt(N L/4) rho`.
Since a four-label Gram obeys `||G||_F^2 >= (Tr G)^2/4`, the quartic is bounded
below by `a rho^4`, where `a=kappa/(8V)` for independently varied pair sites
and `a=V kappa/8` for a uniform frame. Nonnegative alignment can be discarded
in this **lower** comparison. Hence

\[
 F\ge a\rho^4-b\rho\ge-3a\left(\frac b{4a}\right)^{4/3},
 \qquad b=\sqrt{NL/4}.
\]

The JSON evaluates an absolute-row-sum upper for `K` with a floating margin.
The displayed algebraic inequality is exact; its numerical row-sum evaluation
is not advertised as a rational/outward-interval global-minimum certificate.
No equality or sharp minimizer is inferred from this deliberately broad bound.

## Primitive and physical quantum checks

At the selected and best pair frames the consumer builds the **whole**
four-of-eight heavy Hamiltonian from actual `PauliCode.element` S=+ word
amplitudes (70 states), including every nonzero onsite and hopping coefficient.
It compares its ground energy/eigenvector with minus the actual top-four sum.
No reference-chart overlap is required, so failure of the supplier's preferred
section chart cannot exclude an admissible classical field or heavy state.

For the cube, actual `EdgeWords` on a connected 33-vertex tree with its real
boundary parity marker build complete one- and two-particle primitive spectra
(dimensions 32 and 496). They check the primitive coefficient/exterior law
against actual eigenvalues of the full 32-band matrix. Top-sixteen filling uses
that faithful exterior representation; the enormous sixteen-particle sector
is **not** claimed to have been directly diagonalized. Independent endpoint
SU(4) conjugations check the actual onebody and full scalar-potential covariance
at the best pair and cube frames.

Only the explicit improved **pair Clifford reference** is promoted to a
quantum energy upper. The numerical cube candidates and the best unconstrained
pair candidate are not promoted just because an optimizer returned them.
`quantum_reference` reuses `full_band_packet_bound.packet_bound` at the new
materially derived anchor, checks equality of its leading anchor functional,
and chooses a radius only by that supplier's uniform gap/overlap inequalities.
It also runs the actual 4900-state section Gauss/current controls at this anchor,
including all fifteen colour directions at both sites, both light-only EM
Gauss equations, finite covariance and missing-heavy/frozen-section adverse
controls. A chart rejection here is rejection of this **packet**, not of the
classical competitor.

Explicitly the trial is `Psi=sqrt(p) Omega`, where `p` is the exact normalized
compact physical `SU4^2 x U1^2` gauge average of the full 120-ball, Haar-link and
two-phase bump density, and `Omega` is the normalized primitive full occupied
heavy/light section. The exact gauge integral, not finite orbit samples,
makes the scalar amplitude invariant; the section's covariance makes `Psi`
locally equivariant. Uniform gap/positive-overlap, Haar measure and Fisher
bounds in `full_band_packet_bound.md` propagate to the entire gauge orbit of
support. The resulting finite constants certify

\[
 E_0\le E_{\rm trial}\le C\Lambda+T\sqrt\Lambda+B,\qquad\Lambda\ge1.
\]

`C,T,B`, the positive support gaps, chart weight and finite negative-energy
threshold are supplied by the executed JSON. This does not project light/B/EM
out of the actual operator, impose zero orientation momentum on physical
states, count four translated scalar wells as physical vacua or prove a
full-field quantum ground/phase.

## Parent-owned execution and earned scope

Run after the shared source owners finish editing:

```sh
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/unrestricted_field_minimum.py \
  --output /tmp/issue222-unrestricted-field-minimum.json
```

The consumer has no changed-bare-coupling switches. `--maxiter` and
`--stationarity-tolerance` control deterministic numerical refinement only.
It rejects mixed-revision source bindings and failed derivative/primitive/
physical checks. Scientific values have **not** been claimed from worker-side
execution; that execution is deliberately parent-owned under the shared
contract.

The immediate earned analytic outcome is that the environmental selected frame
is not the actual one-axis pair minimum, with an explicit same-law descent and
an exact fully stationary two-field reference. The cube is a separate actual
three-axis comparison, whose finite stationary/Hessian evidence must come
from the executable consumer. Neither the pair failure nor finite cube seeds
settle the 3D unrestricted field phase. The joined electron/photon spectrum,
nonuniform 480-coordinate cube minimum, thermodynamic limit and genuine
quantum phase remain outside this result, not silently solved by the reference
or imported four-centre claims.

## Actual parent field competition and earned discriminator

The first complete parent calculation reached its result but naturally failed
at strict JSON serialization because the lattice-displacement metadata retained
NumPy integers. Explicit native integer conversion at that discrete offset
owner repairs the receipt without a generic serializer fallback. Source
binding now inherits both packet and cube owner manifests, rather than a
second incomplete list: the executed `gauss_charge_intertwiner.py` boundary
word owner and the associated proof providers are included at both ends.

The corrected actual CLI naturally exited zero into
`unrestricted_field_minimum_receipt.json` at source
`958f8973bce7c1db037ac5943315f528a72d470205f86b1bc75ed39e6089f6a8`.
Its pair selected energy is `-.46924476180770286`; deleting the absent
axes gives `-.4822174468171629`, a same-law descent of `.012972685009460028`.
The exact two-field reference reaches `-.5294240034098797`, full-coordinate
gradient norm `7.3431e-16`, and occupied/empty gap `.28831325111077327`.
Its common complex commutant dimension is four, connected stabilizer Lie
dimension three, and full Hessian has **41 numerical flat modes**, no
resolved negative modes, and normal minimum about `-3.2e-16`. This is not
a transversely isolated or fully Higgsed pair ground.

The actual uniform sixty-coordinate open cube has a lower stationary
candidate at `-4.419900402084378`, compared with selected `-4.403817306473285`.
Its gradient norm is `1.4161e-9`, half-cluster gap `.5031771923977721`,
common commutant dimension two, connected stabilizer Lie dimension one,
fourteen gauge-orbit flats, and normal Hessian minimum `.0005347893777355077`.
Those rank/stability values are tolerance-bound numerical observations,
not an exact stabilizer theorem, unrestricted 480-coordinate minimum or
thermodynamic phase. In particular, full four-centre Higgs claims cannot
be inherited merely because all four orbital labels have nonzero Gram entries.

Actual pair primitive residuals were below `7e-16`; cube N=1/N=2 spectra
matched the existing exterior word representation within `2.02e-15` and
`1.596e-14`, respectively. The N=16 energy uses that faithful representation,
not an unperformed enormous-sector diagonalization. Gauss/current/contact
and adverse physical controls passed in the saved receipt.

Only the explicit pair reference is promoted to the gauge-admissible all-120
quantum trial. Its local-support upper has
`C=-.523356963077517`, `T=2.56364853567676e9`,
`B=2.1737778019745916e14`, support gap `.28598873112984863` and
projection weight lower `.6395815326709595`; negativity follows above
`2.399584189871119e19`. Its better classical/leading energy does **not**
make this narrow-support threshold better than the separate all-Haar packet.
No cube candidate is presented as a quantum upper.

The parent native project-interpreter Pyright found and repaired array dtype,
native-index and heterogeneous-record typing errors that an empty worker LSP
push did not expose; the repaired consumer has zero native errors/warnings.
Scoped source/proof/actual-receipt re-review accepted the scientific
discriminators and complete manifest. The ensuing design decision is to
compare the finite open-cube stabilizer with the actual periodic/bulk sea and
derive the actual Cartesian quotient kinetic law, not assume either the
environmental frame or fixed isotropic angular EFT is that joined ground.

## Reusing the actual reference in the all-Haar quantum consumer

`pair_clifford_reference(model, *, stiffness)` now takes the caller's actual
positive quartic stiffness explicitly. The same bare competition CLI passes
`STIFFNESS`; the full-Haar producer passes its requested law coefficient.
The general reference remains \(R^3=(c^4+t^4)/\kappa\),
\(b^2=c^2/(\kappa R)\), \(s^2=t^2/(\kappa R)\), with restricted
minimum \(-3R\) only in \(R>db\). No implicit preferred-Gram term or
different alignment law is introduced.

The parent exercised `full_haar_field_packet.py --anchor-frame
pair-stationary` through its actual primitive Gauss and number-polynomial
consumer. Its all-link/all-phase, all-120-coordinate bound has
`C=-.24076162233613418`, `T=1796509.1391696879`, `B=797.1501879466975`
and strict negativity above `5.567809140503005e13`. The saved
`full_haar_pair_reference_receipt.json` is a wider, substantially stronger
trial than the narrow-support reference above; it does not turn the pair's
flat directions into an isolated ground.

A separate parent midpoint-periodic probe, after recovering the primitive
Fourier coefficients at residual `1.11e-16`, found at a \(24^3\) mesh
energies per cell `-.5618431153915654` for the environmental three-axis
frame, `-.5250279523137149` for this pair reference, and
`-.5587672674100455` for the open-cube candidate. Thus the cube's boundary
improvement does not establish a better periodic frame. These are finite
quadrature comparisons of fixed candidates, not a thermodynamic or
all-sixty-coordinate stationarity theorem; the separate periodic supplier
must resolve that distinction.

