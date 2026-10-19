# All-bosonic internal-colour frame torque

This is a **new source protocol on the same microscopic law**, not a repair of
the original field-only Cartesian torque by notation. It adds the actual SU4
link-electric generator and replaces the off-orbit scalar velocity. It is not
pure spatial rotation, global mechanical spin, or a completed electron/photon
construction. Canonical quantization and CAR remain assumptions.

Governing contract: `issue-222-continuing-backward-20260930`, revision
`6b16fc5fa2204be1c2e567e06313425b16e3253daa91a041e857e12cca47af37`,
`general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`.
Scientific runs belong to the parent; this document gives exact arguments and
runnable discriminators, not numerical results from an unexecuted consumer.

## 1. A frame derived from the actual four scalar matrices

Write `G_mn=Tr(Phi_m Phi_n)/4` and `A=G^(-1/2) Phi`, on the open set `G>0`.
The labels m=0,1,2,3 are the microscopic scalar/vector labels, not an arbitrarily
selected colour eigenbasis. Define in that order

```
B_0=A_0,                 Gamma_0=sgn(B_0),
B_m=prod_(n<m) P_n A_m,  P_n X=(X-Gamma_n X Gamma_n)/2,
Gamma_m=sgn(B_m),        m=1,2,3.
```

Restrict first to the regular tube where every B_m is invertible and Gamma_0
has signature (2,2). The latter follows locally from the Clifford core. Earlier
conjugation maps commute because earlier Gammas anticommute. Thus B_m
anticommutes with every earlier Gamma_n. Odd functional calculus preserves
that property, while `sgn(B_m)^2=I`. This constructs an **exact** Clifford tuple
from arbitrary non-Clifford scalar tuples in the tube. All operations are
conjugation-equivariant, including matrix square roots and signs; no fixed
Pauli/gamma matrix or colour diagonalization convention is supplied.

The actual frame generators are

```
S_i(Phi)=-i [Gamma_j(Phi),Gamma_k(Phi)]/4,
(j,k)=(2,3),(3,1),(1,2).
```

They are Hermitian/traceless, obey `[S_i,S_j]=i epsilon_ijk S_k`, and
`sum_i S_i^2=3I/4`. At every exact Clifford core they recover the whitened
scalar Clifford frame, including nonisometric finite-N true Gram matrices.
This is a frame algebra, not an interacting charged-state multiplet theorem.

**Sign parity:** `G(-Phi)=G(Phi)`, `A(-Phi)=-A(Phi)`. Inductively every B_m and
Gamma_m changes sign; every projection P_n is unchanged. Consequently
`S_i(-Phi)=S_i(Phi)` exactly. The producer checks this at all selected true
Gram centres and at full nonuniform scalar configurations. This parity is
available to the parent's independent CT marked-metric construction; no CT
invariance of a selected pure quantum ground is assumed here.

**Cutoff without a field truncation.** Let chi be a smooth, SU4-invariant,
sign-even function supported within the regular tube, equal to one near the
selected cores. The implementation uses a C-infinity scalar step, the smooth
radial invariant `Tr G` (zero for `Tr G<=.01`, plateau for `Tr G>=.02`),
Gram invariant `det G/(Tr G)^4`, the whitened Clifford defect, and the smooth
flattening invariant `1/Tr(B_m^-2)`. The last cutoff is zero on a neighbourhood
of a singular B_m; evaluating Gamma_m there is unnecessary. The Gram cutoff
similarly vanishes before the whitening singularity. The norm in the defect
is only used away from zero or on a constant plateau. Hence the extended
`chi S_i`, set to zero off the support, is smooth. The code returns this
cut generator, and reports chi separately. **Its Casimir and exact Spin3
algebra hold only on chi=1**; multiplication by chi does not preserve the
unit Spin3 normalization in the transition region. No scalar state, normal
mode, field value or colour-link Hilbert space has been deleted.

There is no smooth normalized Spin3 frame assigned to all scalar tuples:
Phi=0 has full SU4 stabilizer, and a conjugation-equivariant traceless
matrix there must vanish. The cutoff is therefore physical source
localization, not a secretly global frame. There is no lift ambiguity in the
matrix-valued frame itself. A putative colour basis realizing it has centre
ambiguity; centre characters cannot be removed from Wilson/reference transport.

## 2. The actual bosonic source and exact zero normal velocity

For externally specified real bounded b_x, define `g_x=b_x chi_x S_i(Phi_x)`.
The geometric tangent, with a common orientation convention, is

```
Y Phi_xm = i[g_x,Phi_xm],
Y U_xy   = i(g_x U_xy-U_xy g_y),
Y A_xy   = 0.
```

The source operator contains **no CAR or reference term**:

```
K_bos = 1/2 sum_xa {g_x^a, Q_phi,x^a + (div E4)_x^a}.
```

Take the wavefunction convention used by the existing Gauss section:
`psi(Omega Phi Omega^dagger, Omega U Omega^dagger)=Gamma_matter(Omega)psi`.
Then `Q_bos=iY` on the configuration argument and the full constraint is
`Q_bos+Q_H+Q_L+Q_ref=0`. On that physical domain,

```
K_bos = -sum_xa g_x^a (Q_H,x^a+Q_L,x^a+Q_ref,x^a).
```

The sign in a displayed classical velocity versus a pullback operator changes
with the common orientation convention; the producer explicitly uses the
positive geometric tangent above. For its pulse `exp(-i theta K_bos)`, the
BODY physical matter evolution is `exp(+i theta sum g Q_matter)`. This fixes
the source/pulse signs rather than mixing the negative-Lie electric convention
of `quantum_su4_gauss.py` with a positive-Lie CAR convention.

On b=chi=1, the field-dependent geometric vector fields satisfy
`[Y_i,Y_j]=-epsilon_ijk Y_k`; hence
`[K_i,K_j]=-i epsilon_ijk K_k`. The standard positive-Lie Spin3 generators
are `I_i=-K_i`, consistent with the reduced positive-Lie CAR source. This
orientation distinction does not change Casimir or the centre character.

For all15V Gauss columns G and any positive bosonic mass M,

```
Y=G g, ell=(G^* M G)^(-1) G^* M Y=g,
c=Y-G ell=0.
```

This is an identity at **every** regular scalar/link configuration, not merely
BODY; therefore the complete physical-normal JVP of c is also zero. At
singular configurations use the supported Gauss inverse: Gg is still exactly
vertical. No field-only normal complement is substituted. The producer uses
all60V scalar and45V genuine link coordinates on native N3, all15V Gauss rows,
noncentral links and non-Clifford scalar cases. It projects its perturbation
to the actual mass-horizontal normal space before taking the JVP. The
same-boundary adverse source is the original `primitive_flow` with zero
orbital velocity and `f=b+L b/(.6 lambda)`, **without** the new link torque.
That source has a nonzero quotient coefficient and normal derivative.

**Ordering and divergence.** Each fixed-generator Gauss vector field is
divergence-free for Cartesian scalar measure times link Haar measure.
Equivariance implies `X_a g^b=f_ab^c g^c` up to the common orientation sign,
so `sum_a X_a g^a=0` by antisymmetry. An invariant chi contributes zero to
this contraction; b is external and independent of the fields. Thus the
symmetric ordering has no residual half-divergence term. Each local
self-flow leaves g fixed along its orbit, since `[g,g]=0`. Its unitary
configuration flow exists for all times, even on the cutoff plateau
boundary. The product measure is preserved. Pullback along this complete
flow gives the selfadjoint generator on its Stone domain, with the smooth
cylindrical finite-CAR domain as a common form domain. The coefficient is
bounded, and the scalar conjugation does not increase scalar norms. No
claim about arbitrary noninvariant cutoffs or incomplete flow domains follows.

## 3. A genuine Wilson-dressed complete-reference preparation

For every packet site x take an actual oriented SU4 Wilson word W_xR from the
same outside reference R, including actual inverse links for reversed edges.
It is unitary on the **uncompressed** colour-link configuration space:
`W_xR -> Omega_x W_xR Omega_R^dagger`. A Peter-Weyl compression PUP is not a
unitary replacement. For local quaternion labels

```
V_0(x)=I, V_j(x)=2i S_j(Phi_x),
B_a = sum_x eta_x c_L,x^dagger V_a(x) W_xR F_xR r_R^dagger.
```

The external reference r is a real bar4 colour representation with opposite
EM charge. All four reference components are retained. F_xR is the
noncompact Coulomb dressing: longitudinal electric momentum is shifted by
`D^T (D D^T)^+ (delta_x-delta_R)`. The zero sum is essential. This is a
Weyl-unitary translation before reduction, not a frozen scalar photon phase.
In the GraphQuantum representation the longitudinal component is solved
exactly and appears as the unchanged Coulomb term; oscillator and harmonic
coordinates still propagate quantum mechanically.

On b=1, chi=1 at every packet site and b_R=0, the bosonic derivative of
`V_a W_xR` is

```
Y_i(V_a W_xR) = i[S_i,V_a] W_xR+i V_a S_i W_xR
              = i S_i V_a W_xR.
```

All internal Wilson-endpoint derivatives cancel, regardless of noncentral
links or mismatched local frames. Left multiplication closes on
`(I,2iS_1,2iS_2,2iS_3)` with the same coefficients at all x. This is the
additional protocol's exact complete-reference **relative preparation**
Spin3 action, proved on full fields, not merely inferred from a quaternion
marker at BODY. A common smooth mark cutoff is even under Phi sign and
preserves CT parity; its effect on norms belongs to the parent's marked-metric
proof. Exact finite-angle unit-normalized Spin3 requires the *torque* plateau.

At theta=2pi, `exp(i theta S_i)=-I`. On the local source the scalar fields
return by adjoint action but the crossing links/Wilson word acquire -1.
Thus this is **not** a literal closed bosonic configuration path. The
relative creator obtains -1 while the complete neutral RIGHT evolves too.
For b=1 globally, including the bar4 reference, the two endpoint centre
characters multiply to +1. All Z4 characters remain present: fundamental
and bar4 characters are z and z^-1, respectively. The producer separately
checks relative minus, global-reference plus and Wilson unitarity, rather
than treating a bosonic gauge path as an automatic physical normal loop.

**Mechanical interpretation is falsified off the core.** At an isometric
Clifford core, `i[S_i,Phi_m]` agrees with the primitive Cartesian vector-index
tangent. A generic non-Clifford scalar perturbation within chi=1 does not.
Even exact native nonisometric true Gram frames require whitening and can
fail the primitive unwhitened vector-index equality. The scalar/link source
above is therefore an **internal colour-frame torque**, not an exact
primitive mechanical rotation of the Cartesian vector fields. The repair
materially changes the source, rather than pretending to pay the original
normal-transfer debt. A spatial stress/current construction still has its
own microscopic transfer obligation.

## 4. Whole heavy CAR and finite-strength relative pulses

On the fixed BODY used by the existing three-site twelve-mode quantum source
engine, the heavy species is EM neutral. Retain the entire `N_H=6`
`wedge^6 C^12`, dimension924. Its Hamiltonian is the actual scalar endpoint
onebody lift `H_H=-Lambda dGamma(h_BODY)`, with genuine heavy transport on
all four graph edges, **not** a local six-mode or two-particles-per-site cut.
The light sectors are the entire dimensions924/792 (`N_L=6/7`), including
noncompact quantum phases, Coulomb energy and the complete neutral cluster.

For the localized plateau q=diag(1,1,0) tensor S, `Q_H=dGamma(q)` does not
commute with H_H. Heavy intermediate states can leave the whole positive
Slater ground. The producer reports this loss and the actual Lambda-scaled
midpoint and final heavy excitation energies for every requested scale.

Nevertheless, on this **separated fixedBODY** Hamiltonian the full pulse is
exactly

```
U_n = U_H tensor U_light,n,
U_c = U_H tensor U_light,c tensor U_ref,
U_H=exp(-i T H_H+i theta Q_H).
```

A creator does not change the heavy factor. If the full neutral RIGHT uses
the same evolved U_H, the heavy tensor contraction is exactly
`<U_H h,U_H h>=1` for any whole-heavy initial h, any Lambda and pulse strength.
It cancels even when the midpoint excitation costs O(Lambda). **No heavy
invariance or Hamiltonian reversal is assumed.** The numerical producer
propagates the dimension924 factor itself and computes all full-tensor history
inner products from factors; it never allocates heavy times light times
photon times reference. Repeating the same heavy propagation twice would
supply no independent evidence, so it reuses the analytically identical
factor. Dropping heavy RIGHT instead leaves `<h,U_H h>`; that adverse
full-history cross form is reported without final renormalization.

The remaining light/photon relative finite-pulse error is actually computed
from the same H and the same initial Gram, for requested durations. It is
not assumed zero from the 2pi preparation sign. Source-only relative minus
and global-reference plus are separate exact checks.

**Boundary of the cancellation:** the full unfrozen law has
`H_H(Phi,U)` and `g(Phi)` and generally entangles heavy, scalar and link
coordinates. It need not factor as above. The dimension924 calculation does
not bound its normal-dependent pulse commutator, its Lambda-weighted heavy
leakage, or its quantum-colour vacuum. Exact verticality removes the
first-order bosonic quotient coefficient, but does **not** by itself prove
those state/energy estimates. There is no suppression or omission of Lambda
cost, and no claim of uniform microscopic finite-pulse cancellation from
this BODY calculation.

## 5. Parent runnable calculation and remaining joined obligations

```
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/bosonic_frame_torque.py
```

Default: native N3; **all64 actual full-ten true Gram minima**, three
noncentral/all-normal all15V cases, full-field unitary Wilson/frame checks,
whole-heavy924 relative pulses at Lambda1/8/32 and T=.04/.02; oscillator3,
harmonic range[-2,2]. The packet BODY uses the same native N5 frame supplier
as `noncompact_finite_packet_spin.py`, independently of the native N3 survey;
`--packet-frame-grid` exposes that provenance. `--centres selected` reduces
only the Gram-star survey. `--levels`, `--harmonics`, `--eigenpairs`,
`--durations` and `--heavy-scales` are explicit refinements. Imported
engine/source hashes bind the resulting
JSON. `BulkQuantumLaw` is used only for native original endpoint geometry;
its added-selector energy is not used or attributed to the original law.
No shared reference file is modified.

The deliverable distinguishes: exact field-derived relative Spin3;
exact bosonic normal-velocity cancellation; actual fixedBODY complete-CAR
pulse; and a **failed mechanical interpretation**. It does not establish a
charged-state invariant spin multiplet, remove the external reference,
prepare the full quantum-scalar/colour normal vacuum, prove native compact
photon admission, supply interacting current/spatial/Lorentz transfer, or
finish the electron/photon construction. Those are substantive surviving
scientific debts, not consequences of a quaternion label or a 2pi sign.
