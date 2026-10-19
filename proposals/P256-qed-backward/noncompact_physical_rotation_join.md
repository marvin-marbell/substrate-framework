# Physical spatial rotation joined to a charged packet boundary

Governing plan `issue-222-synthesis-continuation-20261006`, template
`general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`,
work item `campaign`, requirements `outcome/verification`. This new pair does not
change any supplier's law or existing evidence. Source/current units are those of
`RotationBand(a,m=.2,r=.12,v=.3,improved=False)` and the actual original full-ten
Gram. No isolated charged pole is required by the construction.

## Decision and usable interface

The surviving route is **global physical spatial pullback about fixed RIGHT,
followed by its complete Gauss body-chart restoration and reference-calibrated
boundary factorization**. It is not an achieved closed local intrinsic-window
rotation. Two attempted local-window routes are discriminated below. The raw
matter action is still a spatial scalar. `physical_spin_intertwiner(frame,R)`
returns the derived body action D; `packet_rotation(frame,R,p,wave)` actually
evaluates D(R)wave(R^-1 p). `positive_spin_frame(frame,p,band)` returns the actual
native positive cluster in one anchored rest convention, a chart for external
spin observables rather than a filter of the unfiltered CT creator or H.

There are two distinct observables:

* The full source, its true Wilson/electric flux and complete RIGHT bar4 reference
  have a **linear** physical rotation. Their 2pi endpoint is +I.
* The reference-calibrated source-end factor in that completed body action is
  spinorial, D(2pi)=-I. Its partner is the actual conjugate reference action, not
  an inverse seam manufactured to cancel a sign. Spatial separation alone does
  NOT prove that a closed local physical Noether curve implements that factor.

A radial rest packet has centered spatial J² approaching 3/4. A moving packet
family has spin-1/2 labels while its centre and carrier rotate as orbital data.
It is incorrect to assign J²=3/4 about the distant fixed reference to an arbitrary
translated, anisotropic or boosted packet. The consumer retains that distinction.

## 1. Inspecting the physical law fixes the lift

`physical_spin_boundary.physical_cubic` rotates positions, the three Phi vector
indices, oriented SU4 links and real electromagnetic links. Reversal stores U†
and minus A at the displaced endpoint. Matter receives only the site permutation
P_R. Photon electric tangents obey the same oriented-link transformation. For a
proper cube rotation,

    h[rotated Phi,U,A] = P_R h[Phi,U,A] P_R†.

This identity includes the native endpoint-average hopping and its independent
Peierls current. The real quadratic-curl photon energy is invariant under the
same transformation of A and its electric tangent. Four physical quarter turns
are +I. This is an exact discrete symmetry, not continuous SO3 on a finite cube.

At an isometric Clifford star recover Gamma from the ACTUAL Gram, and set
S_i=-i Gamma_j Gamma_k/2. The physical index tangent is
(delta Phi_j,delta Phi_k)=(-Phi_k,Phi_j)=i[S_i,Phi]. The coefficient 1/2 is forced
by this equality. Solving the physical vector tangent against all fifteen gauge
generators precedes its comparison with S. The SU4 transformation restoring the
physically rotated body is D(R)=exp(-i theta omega.S). Thus a body-chart packet
obeys D(R)psi(R^-1 x), and a reference bar4 obeys the conjugate D. Neither D nor
an internal Berry mark replaces P_R or the actual photon/field-index operation.

At a tilted finite native Gram, an arbitrary continuous rotation does not preserve
that same star. The exported D uses its whitened Clifford axes, but the finite
same-frame mismatch is explicitly paid below. Exact cube covariance compares the
rotated star/centre, with the actual Z4 seam cochain of
`original_physical_rotation_lift.make_lift`; it does not isotropize c123.

## 2. Physical Gauss, electric flux and normal remainder

Use the true field/link mass matrix M=diag(I60,I45/.6). At each Fourier q let
G(q) be `OriginalSpatial.gauge_columns(q)` and V_i be the physical index tangent
with zero link component, as supplied by `field_rotation_tangents`. For a real
window f, the mechanical connection and its remaining normal tangent are

    ell_i(q) = [G† M G]^-1 G† M V_i f_hat(q),
    Y_horizontal = V_i f_hat-G ell_i.

All15 generators are solved; no HomCasimir, field-only metric or selected normal
mode is substituted. G†M Y_horizontal=0. The electric-link part of G ell is
nonzero for a nonconstant window, and Y_horizontal contains its opposite.
This is the boundary flux contribution, not an error to erase.

On equivariant sections the exact full Gauss equation is

    -i D_G = Q_H+Q_L+Q_ref.

Consequently the physical Cartesian first-order generator becomes the
mass-horizontal first-order normal operator, with its measure half divergence,
plus ell.(Q_H+Q_L+Q_ref). This is the owner derivation in
`original_intrinsic_angular_transfer.md`, not an assumption that the heavy charge
vanishes. The normal momentum terms are controlled by that owner's graded
momentum/resolvent bounds; a normal norm distance alone is insufficient.
`original_total_angular_transfer.md` adds the actual scalar CAR orbital and
photon/colour stress. It retains the full Ward remainder and surface terms.

For the genuine source matrix M_a(x)=eta(x) label_a(Phi(x)) W(x,R) times its
Coulomb phase, a gauge tangent obeys the exact pointwise identity

    delta M_a(x) = i ell(x) M_a(x)-i M_a(x) ell(R).

W is the actual ordered SU4 word. The derivative of W is the electric-flux
boundary action. The consumer differentiates the complete source on noncentral
SU4 links and non-Clifford normal fields. Holding its links fixed or dropping the
right-reference term fails that SAME identity. Thus a spin factor is not being
inferred from the internal algebra of the quaternion labels.

The label quartet by itself has full adjoint action 1+3 when both endpoints are
rotated. The source-side left action has two complex doublets. Only after paying
flux and reference localization is that latter action a local physical boundary
observable. A bare HomCasimir or integer adjoint mark would not establish it.

## 3. Failed driven-loop route and materially different repair

The existing mechanical torque f+Delta f/(.6gs) produces an exact body plateau.
It is a useful driven intrinsic supplier, but its exterior torque is generally
noninteger. A 2pi field-index curve then does NOT close on the exterior fields.
Its two-controller orbital/photon echo is not a single D(R)psi(R^-1 x) curve.
More quadrature, its minus endpoint or an internal Spin3 label cannot fix this.

The second attempted repair uses an **ordinary rigid plateau**, with no inverse
screening torque. At the isometric star the all15 solve reduces, in this one
derived bivector channel, to

    sigma=(m_s²+Delta)^-1 m_s² f,    m_s²=.6gs,
    ell_i(x)=sigma(x) S_i.

This is the actual electric-flux screening law. For a cubic plateau whose edge is
at least d cells from the source support, and with R at least d cells outside it,

    |1-sigma(source)| <=12 exp(-kappa d),
    |sigma(R)|       <=12 exp(-kappa d),
    kappa=acosh(1+m_s²/4).

Proof: use the positive nearest-neighbour resolvent/random-walk kernel. Its
one-coordinate exponential moment is m_s²/[m_s²-2(cosh(kappa)-1)]=2.
Apply the two signs on each of three coordinates and Markov's inequality. The
same proof holds on the periodic graph using the lifted walk and torus distance;
images do not improve the bound by being discarded. The COLOUR-CONNECTION
PART of the normalized source CAR commutator has coefficient error at most
(|1-sigma|+|sigma(R)|)/2. Its exponent pays 2pi times that error. This is NOT
yet a bound on the full physical Noether/source intertwiner: its horizontal
Wilson-flux derivative is not included by screening coefficients alone.

Finite-size screening is computed at N17/33/65/321 with bounded-memory exact
separable Fourier sums. Its coefficient is the measured N5 isometric frame:
the larger screening graphs are not mislabeled as their minimizing scalar stars.
The analytic guard is also returned. Small boxes are not silently declared
local-spin admitted. For a tilted Gram retain the full matrix connection; its
source/reference defects and actual Gram convergence error must be added. The
uniform metric supplier gives a quantitative true-Gram route for the admitted
family; arbitrary nonuniform field isotropy is not asserted.

### Decisive Wilson-flux failure and the third route

The actual source/Noether consumer retains a horizontal derivative norm1.3858.
A second, independently executed genuine SU4 Wilson-word probe makes the
nonlocal boundary explicit. For identity links and ell decreasing from1 at the
source to0 at the complete reference along a true word of lengthL,

    delta W =i sum_edges (ell_x-ell_y) S =iS,
    delta_horizontal M_a=-i label_a S,
    delta_physical label_a=i[S,label_a]
       =iS label_a-i label_a S.

At L8/80/800 each link tangent decreases .125/.0125/.00125, but the horizontal
source derivative norm stays EXACTLY2. The derivative identity error is4.72e-11.
The actual physical Phi-index source loop is +I (error6.93e-16); claiming -I
after discarding that flux gives defect8. Thus a distant reference, tiny local
link gradient, small screened coefficient or large Lambda does not alone erase
the accumulated Wilson term. The ordinary-plateau estimate is retained as a
useful connection-coefficient estimate, NOT promoted to achieved local physical
half-spin. Equal normal/heavy RIGHT histories cancel common state evolution,
not the source's nonzero coordinate derivative.

The materially different selected third route is the GLOBAL completed spatial
rotation about the fixed reference. The producer executes four actual spatial
quarter turns with every moved vector/link/site, then each actual body Gauss
restoration and the conjugate bar4 action. That full curve is linear, while its
source-end boundary factor is spinorial. The orbital-aware continuous packet
intertwiner below supplies its controlled low-momentum extension and retains
the reference as a calibrated factor, rather than claiming a closed intrinsic
window operation. No additional microscopic matter spin law is introduced.

In the total-angular owner's effective image the reference-calibrated spin
observable subtracts the measured orbital/photon and actual reference angular
pieces from the SAME charged-minus-neutral physical J. Normal/heavy and
colour-flux terms must first be transferred or measured there; they are not
dropped. The finite positive-packet chart provides its concrete comparison
observable. A coupled microscopic local J²/Noether realization still needs that
weighted source-angular boundary, not another internal Casimir or pulse echo.

## 4. Orbital-aware actual-band preparation and continuum error payment

Let Xi be the actual beta-positive rest frame, diagonalizing its derived S_z
once. For every p use the full native h_Phi(p), its actual positive projector P+,
and

    U(p)=P+(p) Xi [Xi†P+(p)Xi]^-1/2.

Every eigenmode of h is retained; a singular chart is an explicit error, not a
fallback. The CT source and its occupied components are NOT replaced by U.
U only gives the spin observable and the positive/hole CAR comparator components.
One initial G is used; no final metric renormalization hides source loss.

At the isometric quadratic comparison band,

    h2(p)=Gamma0[mu+a r_eff |p|²/2]+c Gamma.p,
    U2(p)=sqrt((E+M)/(2E))[Xi+c Gamma.p Xi/(E+M)],
    D(R)U2(R^-1 p)=U2(p) D_rest(R).

Here mu=m sqrt(g0), c=v sqrt(gs), r_eff=r sqrt(g0), all from the same star.
The orbital derivative is essential:
J_i U2=(S_i-i(e_i cross p).partial_p)U2=U2 S_rest,i. Thus a radial rest envelope
has total centered J²=3/4. The consumer evaluates both first angular derivatives
and the actual second derivative J² on the native positive cluster; it does not
merely re-evaluate an internal Casimir. Moving packets carry
exp(-i p.X) exp(-|p-p0|²/(2sigma²)), and the physical action rotates X AND p0.
The internal-only matrix action fails the same packet-intertwining surface.

For the exact native isometric symbol,

    ||h_a-h2|| <=rho(p)=c a²|p|³/6+r_eff a³|p|⁴/24,
    ||h_a(Rp)-D h_a(p)D†|| <=2rho(p).

For full native Gram B=sqrt(G), compare B_iso=diag(sqrt(G00),sqrt(gs)I).
The producer uses the conservative frame Frobenius distance delta_B, which bounds
its operator distance. The paid finite same-frame covariance bound is

    2rho(p)+2delta_B[m+r a|p|²/2+v|p|].

The measured cubic-star and tilted-star native covariance must satisfy this bound.
For unit Gaussian radial density exp(-p²/sigma²),
M_n=sigma^n Gamma((n+3)/2)/Gamma(3/2), so the all-momentum RMS lattice remainder is
c a² sqrt(M6)/6+r_eff a³ sqrt(M8)/24. The displayed Gaussian is not a momentum
hard cut; the Brillouin boundary probability is separately bounded by
GammaUpper(3/2,(pi/(a sigma))²)/Gamma(3/2). The packet kinetic energy is finite,
bounded by mu+a r_eff M2/2+c sqrt(M2). A compact-box interpolation additionally
pays its actual spatial seam/tail; exact cube permutations need no interpolation.

For a general moving/boosted source replace M_n by its actual absolute momentum
moments, not moments about the carrier. This producer's Gaussian RMS row is the
radial REST envelope, while its moving-family adverse probe is separately labelled.
No rest moment is reused as a bound on that displaced Gaussian.

## 5. Same-Omega, full-H and fixed-energy admission contract

The native3D consumer retains all4V light modes and the whole2V neutral sea.
It evolves the complete unfiltered, genuine-tree, holonomy/harmonic-dressed CT
source under the unreversed actual h, then rotates its orbital envelope and every
field/link index. It also executes real photon coordinate/electric rotation and
the SAME native Peierls current covariance. Internal-only source rotation and a
missing reference fail the consumer's actual surfaces.

The coupled large-volume admission is inherited from
`noncompact_charged_packet_transport`: compare source-created ACTUAL Omega and
actual neutral RIGHT, not unrelated free vacua. If its normalized same-H packet
error is epsilon_src(t), apply that estimate to both rotated branches. For the
bounded rotation observable the triangle/Duhamel bound adds

    epsilon_rotation <=epsilon_src+epsilon_rotated_src
       + |t| RMS(covariance remainder)
       +2pi epsilon_screen+epsilon_Gram+epsilon_normal+epsilon_spatial_seam.

There is no inference from a free numerical sea to a coupled ground. The real
quadratic-curl photon packet rotates its momentum/polarization as a vector;
its same-current preparation error and finite photon band energy are those of
`noncompact_photon_packet_supplier`/the fixed-cutoff family. Photon vector and
orbital angular momentum are not assigned spin1/2. Their common neutral action
and the source-relative remainders remain in the above comparison.

The actual moving-source consumer also executes all FOUR spatial quarter turns
on its positions, fields, true oriented links and wavepacket. In parallel it
executes body restoration with each actual centre/seam and the conjugate RIGHT
reference. Raw microscopic and completed body loops are +I, while the
source-end-only body factor is -I. This is not just exp(-2pi i S) checked on an
internal matrix, and the reference-omission adversary reaches that same loop.


`shared_history_contract` consumes the exact worker-owned shared-clock receipt,
checks its executable SHA, and attaches it rather than rerunning or rebranding
its histories. Mechanical normal, actual Lambda-heavy, light, photon and complete
bar4 RIGHT all use tau=Lambda^-1/32. The whole-heavy common factor includes all924
exterior states and actual huge energy/source phases; normal has all11250 modes
and its full affine B majorant. Those phases cancel between the same histories by
unitarity, not by being small. The producer does not call their two-leg echo a
single spatial SO3 curve.

For the microscopic slow-variable transfer the owners provide fixed-N weighted
normal/heavy/angular error powers. Their constants are fixed-N and their nonlinear
scalar N0/tube onset is not numerically explicit. Choose N for the photon, Gram,
spatial tail and screening budgets first, then finite Lambda(N) for those weighted
bounds. This is a controlled diagonal construction, not a uniform fixed-Lambda
volume theorem. An unbounded interacting total J² measurement additionally needs
that branch's angular graph-norm transfer: this bounded spin-rotation capability
must not be relabelled as an already computed full3D interacting J² spectrum.

### Quantitatively nonempty effective bounded-rotation budget

`admitted_rotation_budget` uses the actual a1 `tight-paid` positive family,
not the finite mechanical interface's UE=.37/K=.9. It consumes a fresh Arb128
Q64 native-metric certificate and evaluates the uniformly all-shift/all64
Gram distance at N=max(Nmetric,Nphoton), rounded upward to odd N. The underflowed
alias and seam exponents are retained as logarithmic upper bounds; the seam
numerical bound is rounded UP to exp(-700), not set to zero.

The source is a broad sigma=.02 Gaussian with carrier norm .02, compactly
approximated only in POSITION at L=ceil(8/sigma) cells. Its unfiltered quaternion
source retains all4 bar4 components and initial CT norm²2. For the normalized
discrete Gaussian, Poisson summation gives Z1>=sqrt(pi)/sigma and monotone
integration bounds its L1 by
[(1+sqrt(2pi)/sigma)/(pi^(1/4)/sqrt(sigma))]^3. Removing the spatial tail
multiplies that by 1/sqrt(1-tail); tail is bounded by
3[2sigma exp(-(sigma L)²)/sqrt(pi)+erfc(sigma L)]. The bounded-rotation
comparison pays FOUR times the norm tail. No momentum/H filter is applied.
Minkowski bounds the absolute moment root by the radial moment root PLUS |p0|;
rest moments are not reused for the moving source.

The same-H residual is the owner's packet force/electric estimate with these
L1 and harmonic first/second derivative upper bounds. Actual reference distance
is bounded by4L per coordinate. Native source h0(a_h) and actual scalar toron
potential are retained; its fixed-frame replacement pays the owner's CT seam
tail. The rigid plateau has a further paid guard to that source support, with R
outside by the same guard. N is checked against ALL source/guard/reference
geometry, not only against the photon onset.

For finite true-Gram geometry write deltaPhi<=deltaG/sqrt(lambda_min(G)).
The Cartesian gauge matrix obeys deltaD<=sqrt(30)deltaPhi, hence
deltaA<=(2||D0||+deltaD)deltaD and
delta(D†V)<=||D0||deltaPhi+deltaD(||V0||+deltaPhi).
Because A=D†D+Delta/.6 diagonalizes into fifteen positive scalar resolvents,
its l-infinity matrix-kernel bound is15/lambda_min(D†D). Applying the resolvent
identity to the bounded ordinary window f yields the explicit connection
perturbation in the code, WITHOUT a volume factor or omission of link flux.
The source-plus-reference spin norm pays sqrt(2) times that coefficient error,
and its loop pays2pi. The certificate therefore controls the same true Grams,
not just an isometric comparison. The output sums two same-H source errors,
two seam errors, full15 geometry, rigid-screening/reference, native orbital
lattice remainder and spatial source tail, and requires their sum below .01.

The output key `explicit_effective_bounded_rotation_error` is the SUM OF THE
EXPLICIT COEFFICIENT/SOURCE/ORBITAL DEBTS, not a complete physical local-rotation
certificate. The unsuppressed horizontal Wilson/source term just demonstrated
must be retained in any attempted ordinary-window Noether realization; its
vanishing is not implied by a Lambda(N) choice. The global reference-calibrated
body/fundamental comparison separately requires the owners' fixed-N weighted
normal/heavy and angular/current transfer on the actual interacting source.
No numerical microscopic N0/Lambda or coupled local total-J² certificate is
claimed. Finite UE=.37 histories are attached with their OWN parameters, not
relabelled as simulations of this very weak admitted family.

The new all-state native current capability from `PhotonPacketSupplier` supplies
the photon leg on the SAME effective actual-Omega charged path. For a normalized
physical a1 coexact link-volume mode (sum_xj |f|²=1), its full-CAR bound is
CJ=192 tau_endpoint² N³. The uniform admitted true-Gram endpoint is
tau_endpoint=(.12sqrt(.43)+.3sqrt(.065))/2, including every unitary colour and
real-photon/harmonic phase. The code retains the volume factor; this is not a
thermodynamic uniform variance claim or an all-scalar-normal operator bound.
At a nonzero band [omega_min,omega_max]=[.005,.01] it pays
D=UE CJ/omega_min³, physical leg sqrt(UE), initial photon-on-charge depletion2D,
normalized insertion error2|t|sqrt(UE CJ/omega_min), and finite photon energy
omega_max(1+D)+sqrt(UE CJ/omega_min). Tiny Gram corrections are stored as an
offset, never rounded into an allegedly exact upper1.

This insertion compares to the EXACT charged H path, with actual neutral/current
and full bar4 RIGHT retained. It is deliberately NOT added to a generic
source-to-free norm error multiplied by the unbounded creation ladder: that
composition still needs its weighted ladder/current/normal jets. Different
photon/source receiver labels require their actual exchange Gram and initial
whitening; a quartet CCR lower bound is not a universal lowerI on that matrix.


For a continuum photon mode with 0<|k| and |ak|<=1, the centered real-curl
symbol is kappa_j=2sin(ak_j/2)/a. Thus
|kappa-k|<=a²|k|³/24. A native transverse projection after vector/spatial
rotation differs from the continuum transverse projector by at most
a²|k|²/6 (the conservative small-angle projector bound); the frequency error is
at most c_ph a²|k|³/24. The true primitive photon/current preparation residual,
its absolute spectral moments and field angular graph norms remain supplier
inputs. The exact cube photon/current consumer does not establish those
non-cubic continuum moment assumptions for an arbitrary rough coupled state.


## 6. Reproducible consumer

From this directory, using the framework Python and one BLAS thread:

```sh
/home/administrator/substrate-framework/.venv/bin/python \
 noncompact_physical_rotation_join.py --size 5 --spacing 1 --time .5 \
 --screen-sizes 17 33 65 321
```

The output binds source/proof dependencies, exact inputs, all Gauss/flux/current
controls, finite packet/energy/2pi bounds, actual native orbital-aware spin rows,
screening guard and the worker-owned shared-clock receipt. LSP is real Pyright
stdio; AST is the installed Python ast-grep parser. The finite execution proves
its stated identities and estimates. It does not exhaust the coupled N0/T or
microscopic Lambda admission and does not close the campaign.

## 7. Executed evidence and decision delta

The final executable SHA256 is
`b950319832910770f6ec2777b01ce8abaa6606d2be5de999f5dea82940c23bd6`.
The complete CLI naturally exited0 in6.465107185 seconds; stderr was empty.
Its executed proof SHA256 was
`213cda588180342b3f440bd2eaf5cabb7eb68cc810dd33f79e6c86991f2e1496`.
Exact source/proof/stdout/stderr/receipt are archived under
`/tmp/issue222-physical-rotation-join-20261006/final-*`. This evidence appendix
and the Wilson-flux interpretation correction postdate that run; the executable
and released spin-helper bodies are unchanged.

Observed actual moving-source four-quarter errors: raw physical loop0, completed
Gauss/bar4 loop6.9346e-15, source-end-only body minus3.6465e-15, full vector/link
closure1.1312e-15. Same unreversed-H source covariance1.2461e-14, true Gauss
source derivative1.2816e-10 and actual physical Noether decomposition2.3538e-10.
Missing Wilson flux .0661348, missing reference .401052 and internal-only
moving source6.9455 all failed their same consumer surfaces.

At the isometric native star and momentum radius .16/.08/.04/.02, the actual
orbital-aware positive-packet J² errors were7.6101e-4/9.5815e-5/1.2016e-5/
1.5193e-6. Full orbital-spin Ward errors decreased as radius³, while intrinsic
spin-only errors were linear. The tilted c123 frame retained its8.5166e-4
isotropy distance; all64 actual Gram/connection/normal remainders were returned,
with maximum frame-isotropy distance .00167267 and no well weights invented.

N321 ordinary-window point-source screening produced coefficient .99999951356,
reference2.04997e-13 and COLOUR-CONNECTION loop error1.5282e-6. The separate
Wilson-word execution above proves why that is not the full physical local loop.
At the actual positive a1 family N108999, UE1.82004e-35/K2.74719e32,
T>=315257516790.59, the explicitly computed source/Gram/coefficient/orbital budget
was .00096572708. Its source error per branch7.6684e-14, all-shift Gram distance
1.5352e-12, full15 connection perturbation1.4540e-5 and guard82 are retained.
Same-Omega exact charged-path photon insertion at band .005-.01 has error
4.6683e-9, annihilation debt2.1793e-13 and finite photon energy .0100000023342.
It is not multiplied through an unbounded creation ladder by a generic charge
norm error.

Real Pyright1.1.414 stdio returned403 symbols and EMPTY diagnostics on the
final executable. Positive Python ast-grep queries matched the actual physical
vector-index operation and exported positive-spin-frame consumers;
ast-grep0.45.3 was installed. The unavailable xd://lsp adapter was reported and
replaced by the real server, not a syntax-only substitute.

The materially different checked output is now a genuine completed spatial
body/reference rotation plus an orbital-aware boundary spin chart; the attempted
local-window half-spin shortcut is withdrawn by exact flux evidence. Two
concrete successors are (selected) the SAME-current actual-Omega,
reference-calibrated total-angular source/ladder graph-norm transfer on this
orbital-aware packet, and (alternative) a dynamically screened colour-neutral
reference/cloud preparation changing the Wilson boundary state rather than
merely its coefficient window. Neither is a claim of campaign completion.

