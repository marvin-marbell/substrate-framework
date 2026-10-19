# Physical charged trial cost without freezing the full field ground

## Shared Hamiltonian and precise charged extension

Use exactly the full-coordinate neutral Hamiltonian of
`pair_full_field_concentration.md` on R120×SU4×T² with NH=NL=4 and all4900
primitive Pauli configurations:

```
H_N = H_common + H_light,
H_common = sqrt(Lambda)/2 p_fields² + ue4 C_colour
         + ue1/2 (p_A0²+p_A1²) - K cos(A0+A1)
         + Lambda [V_kappa + J W/8 - dGamma(h_heavy)],
H_light = dGamma(h_light).
```

The charged operators use **the same coordinate Hamiltonian**, NH4 and NL3
with external fundamental4 at the right site (q=-1), or NL5 with external
antifundamental bar4 (q=+1). All3920 matter states times4 boundary components
are retained. The external boundary contributes to right colour Gauss and
to the right EM source offset, not a new Hamiltonian term. This is an explicit
charged extension of the full two-phase owner; it is not an assumption that
an arbitrary charged boundary apparatus has zero energy. In particular, do
not import the extra q² tree-background term from the older restricted
`PhysicalHamiltonian` benchmark into this different full two-phase operator.
An additional prescribed boundary energy would have to be added explicitly.

Here q is the light-number departure relative to the neutral sector, with a
**fixed compensating colour/EM reference at R**. The closed graph's total EM
Gauss charge is still zero: the light source has net q and the external offset
has net -q. A_R inserts the light excitation at the same reference site; its
state need not have a strictly site-local density after Fermi correlations.
This bound does not establish an isolated charged particle, a separated
charge/reference energy, a long-range Coulomb dressing or a mobile packet.
Those require a spatial construction with the reference held separately.

Let c_Ri be the actual directed marker-word annihilators of
`gauss_charge_intertwiner.rectangular_ladder`. Define constant fibre maps

```
A_- = sum_i c_Ri tensor |i>,
A_+ = sum_i c_Ri† tensor |bar i>.
```

Their native CAR and Gauss intertwining were established by the existing
primitive source. Here they are recomputed on the complete4900→15680 fibres,
without a finite Haar representation or field/angular cutoff. Their exact
norm operators are `A_-†A_-=n_R` and `A_+†A_+=4-n_R`, not I.

The target colour generators are Q_R+T for q=-1, Q_R-T^T for q=+1. At the
right EM site replace n_R-2 by n_R-2-q. Left generators are unchanged.
All30 colour and both EM source operator identities hold. Since A is
coordinate-independent, the field/link/phase derivative part of Gauss also
intertwines. Thus A maps the genuine neutral physical space and form domain
to the appropriate charged physical space. Its bounded norm (at most2)
preserves differentiability; finite quartic moments preserve the unbounded
multiplication terms' ground-state expectations.

**The entire common operator intertwines exactly:**

```
H_common,q A_q = A_q H_common,N.
```

Raw Cartesian and Haar/phase kinetics commute with constant A; heavy
bilinears act in the other flavour and commute; quartic, alignment and
magnetic multiplication commute. No scalar inertia, adiabatic projection,
local harmonic well, light occupation cut or chosen mean is used. Intertwining
operators does NOT by itself equate normalized unweighted common-energy
expectations; the exact neutral eigenstate identity below is essential.

## Exact neutral-ground symmetry, not an assumed occupation

The full one-axis two-cell neutral law has the following reflection:

- interchange the two cells and their actual Pauli modes;
- send Phi_1,x to -Phi_1,other-cell, and interchange the other field components;
- invert the colour link V→V† and interchange A0,A1.

The endpoint matrix becomes its adjoint, and
`(exp(iA0)+exp(-iA1))/2` becomes its conjugate. The native one-body Hamiltonians
are conjugated by cell interchange. All Gram quartic and transported alignment
norms, full field metric, Haar Casimir and torus kinetic/magnetic terms are
invariant. This unitary exchanges colour/EM Gauss sites and preserves NH4/NL4.
The implementation lifts cell interchange with actual marker-word Fock phases,
not a guessed bit-permutation sign, and checks the complete native identity at
arbitrary all120 fields/noncentral links/phases.

The proved compact-resolvent physical neutral Hamiltonian has a finite-rank
nonzero ground eigenspace. Take the normalized ground projector rho_N; it is
reflection invariant, including when the ground is degenerate. Consequently

```
Tr(rho_N n_R)=2,
Tr(rho_N dGamma(P_R h_light)) = Tr(rho_N H_light)/2,
Tr(rho_N dGamma(h_light P_R)) = Tr(rho_N H_light)/2.
```

The last two quantities are real. They also follow from reflection and
stationarity: `[H_common,n_R]=0`, so the neutral-ground expectation of
`[H_light,n_R]` vanishes. No numerical quantum ground or per-configuration
right occupation2 is assumed.

The physical charged trial density is
`rho_q = A_q rho_N A_q† / 2`. It is normalized, positive, and remains on the
full original coordinate domains. Ground-state stationarity gives

```
Tr(H_q rho_q)-E_N = Tr(rho_N A_q† [H,A_q])/2.
```

Only the light commutator remains. CAR gives

```
A_-†[H_light,A_-] = -dGamma(P_R h_light),
A_+†[H_light,A_+] = Tr(P_R h_light)-dGamma(h_light P_R).
```

`Tr(P_R h_light)=c_light Tr(Phi_0,R)=0` on the actual traceless field domain.
Therefore **both** physical charged trial costs obey the exact identity

```
E_trial,+ - E_N = E_trial,- - E_N = -<H_light>_N/4.
E_ground,q - E_N <= -<H_light>_N/4.
```

The charged domains are nonempty because both trial norms equal2. Quartic
coercivity also makes each charged physical operator bounded below and gives
compact resolvent. This identity is an upper-cost result, not a lower mass
gap: it does not claim that the right-hand side or the actual charged-minus-
neutral ground difference is positive, or that every charged state is in A's
image. A† is not a complete inverse on the enlarged charged fibres.

## Joining the already proved full-field moment bound

The source-owned pointwise light lower bound is

```
H_light >= -a_u sqrt(u) -a_v sqrt(v),
a_u=8sqrt(2)|c_light|+8|d_light|, a_v=8|t_light|,
u=(TrPhi_0,L²+TrPhi_0,R²)/8,
v=(TrPhi_1,L²+TrPhi_1,R²)/8.
```

The accepted neutral ground theorem gives
`<(u-u*)²+(v-v*)²> <= s(Lambda)/kappa`,
`s(Lambda)=O(Lambda^-1/6)`. Cauchy and Jensen yield

```
<u> <= u* +sqrt(s/kappa), <v> <= v* +sqrt(s/kappa),
E_ground,q-E_N <= C_q(Lambda)
 = [a_u sqrt(u*+sqrt(s/kappa))
    +a_v sqrt(v*+sqrt(s/kappa))]/4.
```

The same bound holds for both q. It tends to the **finite** limit
`[a_u sqrt(u*)+a_v sqrt(v*)]/4=0.62789737501526` at the actual kappa2 bare
point. The correction is O(Lambda^-1/12), since u*,v*>0. Unlike a trial-to-
trial subtraction, the comparison is to the exact full neutral ground.
It avoids importing a scalar inertia or paying a guessed divergent common
kinetic cost. Constants are evaluated in floating arithmetic, not interval
certificates or a numerical solution of either full quantum ground.

## Actual native consumer and adverse controls

The executable recomputes complete norm, colour/EM and common-heavy
intertwiners and the actual light commutators at eight arbitrary all120-field,
noncentral-colour/compact-phase points. Wrong external boundary sign and
omission must fail. It computes the source-owner's complete frozen Pauli
anchor eigenstate, at a common heavy coefficient1e6, and measures both
correctly normalized charged trial costs. These equal
`-E_light/4=R*=0.176474667803293` in that **frozen pointwise** control; that
number is not asserted to be the full-field quantum charge gap. Omitting the
actual norm squared2 must create a common-scale error of order1e6. The
full-domain energy bound uses the analytic neutral ground theorem, not this
frozen control state.

```
python proposals/P256-qed-backward/pair_full_field_charge_transfer.py \
  --output proposals/P256-qed-backward/pair_full_field_charge_transfer_receipt.json
```

Still open: positive charged lower mass gap, charged packet propagation and
same-current scattering, moving normal/geometric dynamics beyond this pair,
full spatial/thermodynamic and compact-photon phases, mechanical spin,
radiative/Lorentz closure, and the original joined campaign outcome.
