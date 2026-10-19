# Original full-field local-reference charged upper and two-sided threshold

Governing plan `issue-222-continuing-backward-20260930`, assignment revision `fdd8d586e9506148da75d9a41626f3bbadec4a4c88133582afc041e903bbe8a6`, pinned general template `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`, work `campaign`, requirements `outcome/verification`. This construction generalizes the existing pair local-reference transfer to the unchanged three-dimensional original full-field quantum law. Fundamental CAR/gauge quantization is still assumed. No added selector, mean replacement or field/rotor/Fock cutoff is used.

## 1. Exact sector and reference ownership

Use the complete original Hamiltonian and positive slow trajectory defined in `original_slow_kinetic_bo.md` section1: R^(60V), SU4^(3V), U1^(3V), NH=2V, neutral NL=2V, original endpoint-average hH/hL, field quartic kappa sum_x ||G_x||F²/2, kappa=2/a, positive alignment/colour plaquette and electromagnetic electric/magnetic forms. Magnetic forms use their common nonnegative constant shift. Reference sectors have NL=2V+q, q=±1, the actual static opposite SU4 bar4/4 representation and opposite U1 charge at R, no extra reference Hamiltonian. All fifteen/site colour and original background-two U1 Gauss constraints remain.

The reference is at the **same site as this trial insertion**. The resulting energy upper is a coincident compensated-sector threshold; it is not a separated-charge upper or deconfinement result. The reference can bind the carrier, and the trial does not establish a mobile electron or half-spin. Moving or adding an apparatus energy to that reference changes the contract.

## 2. Exact full-coordinate insertion

With the existing CAR convention define

```
B_R,- = sum_alpha c_R,alpha tensor |alpha_ref>,
B_R,+ = sum_alpha c_R,alpha† tensor |bar alpha_ref>.
B_-†B_-=n_R; B_+†B_+=4-n_R.
```

These are bounded, coordinate-independent fibre maps, not normalized isometries. For q=+ use reference generator -T^T, and for q=- use T. They intertwine all colour generators, with the reference only at R, and n_x-2-q delta_xR at the electromagnetic source boundary. In particular the source at R is unchanged after insertion. Raw Cartesian, Haar and electromagnetic derivatives commute with B; they are not replaced by scalar inertias. Heavy bilinears act in the other CAR flavour and commute with these maps. Field quartic/alignment/magnetic multiplication commutes. Consequently the entire common part satisfies

```
H_common,q B_R,q = B_R,q H_common,0.
```

The bounded map preserves the full physical form domain; the unbounded field multipliers below are controlled by the quartic moment, not norm closeness of a BO vector. Odd light maps are legitimate intertwiners between the stated sectors, not observables creating an unreferenced charge on a periodic graph. Neutral and charged total centre characters are respectively z^(4V) and z^(4V+q-q), both trivial.

CAR gives, on the full coordinate core,

```
B_-†[H_light,B_-] = -dGamma(P_R h_L),
B_+†[H_light,B_+] = Tr(P_R h_L)-dGamma(h_L P_R).
```

The onsite trace is zero on the actual traceless field domain. These are the same row/column identities as the existing pair supplier, with no two-cell reflection assumption.

## 3. Authoritative neutral ground and translation

The full confining finite-graph physical operator has compact resolvent. Take its normalized ground projector rho0, not an assumed product state or per-configuration occupation. Translation acts on positions, all fields, directed colour/EM links and actual CAR modes with their Fock signs. It preserves the microscopic neutral Hamiltonian, Gauss constraints, NH/NL and the full coordinate domains. Thus rho0 is translation invariant even if the neutral ground is degenerate or the finite-Lambda state is spread over the64 centre wells. No arbitrary centre is made a conserved sector.

Summed NL=2V gives Tr(rho0 n_R)=2. Translation and sum_R P_R=I give

```
Tr(rho0 dGamma(P_R h_L)) = Tr(rho0 H_light)/V,
Tr(rho0 dGamma(h_L P_R)) = Tr(rho0 H_light)/V.
```

They are real: [H_common,n_R]=0 and ground stationarity makes the expectation of [H_light,n_R] vanish. These statements use the actual full Hamiltonian, not a frozen inhomogeneous h for which translation could be false.

The physical charged trial rho_q=B_R,q rho0 B_R,q†/2 is normalized. Eigenstate stationarity, rather than cancellation of unweighted common energies, gives

```
Tr(H_q rho_q)-E0 = -Tr(rho0 H_light)/(2V).
E_q-E0 <= -<H_light>_0/(2V).
```

The density proof also implies at least one admissible pure trial with no larger energy. Reference position R is arbitrary by translation, but the carrier/reference separation is not prescribed.

## 4. Uniform field coercivity, with actual endpoint factors

Set t0=m+3r/a and

```
Q2=(1/V) sum_x Tr G_x
  =(1/(4V)) sum_x,mu Tr Phi_mu,x²,
C_a=max(t0²+3r²/(2a²), v²/(2a²)).
```

For N>=5 each native directed edge has the distinct block

```
h_xy = -[r(Phi_0,x U+U Phi_0,y)
           +i v(Phi_j,x U+U Phi_j,y)] exp(iA_e)/(4a).
```

This is the actual endpoint average. Frobenius invariance under U, Hermiticity and trace of the commutator give

```
||h_xy||F² <= [r²(||Phi_0,x||F²+||Phi_0,y||F²)
               +v²(||Phi_j,x||F²+||Phi_j,y||F²)]/(8a²).
```

The cross term between the two Hermitian transported endpoint sums has zero trace. Counting both h_xy/h_yx blocks, six incidences for Phi0 and two for each Phij yields

```
Tr h² <= (t0²+3r²/(2a²)) sum_x Tr Phi_0,x²
               +(v²/(2a²)) sum_x,j Tr Phi_j,x²
         <=4 V C_a Q2.
```

This holds for arbitrary non-Clifford fields and noncentral colour links, both with and without compact EM phases. It does not invoke a beta coercivity valid only at fixed metric.

Every h is Hermitian and traceless. Its entire CAR second quantization in any retained number sector has norm at most Tr|h|/2; Cauchy–Schwarz then gives

```
||dGamma(h)|| <= sqrt(V Tr h²) <=2 V sqrt(C_a Q2).
```

The local Gram is PSD, so ||G_x||F²>= (Tr G_x)²/4. The actual quartic is consequently at least kappa V Q2²/8. All other bosonic terms are nonnegative in the declared common shift. Hence, on the **unshifted full** microscopic form,

```
H_Lambda >= Lambda kappa V Q2²/8
                  -2(Lambda+1) V sqrt(C_a Q2).
```

No negative classical well baseline is discarded. If E0<=0 and Lambda>=1, taking its true ground expectation, using convexity of Q2² and concavity of sqrt(Q2), gives for q2=<Q2>0>0

```
kappa q2²/8 <=2(1+1/Lambda) sqrt(C_a) sqrt(q2),
q2 <= [16(1+1/Lambda) sqrt(C_a)/kappa]^(2/3)
   <= [32 sqrt(C_a)/kappa]^(2/3).
```

The local insertion cost is bounded using the true field moment:

```
-<H_light>0/(2V) <=sqrt(C_a) sqrt(q2)
 <= U_a := sqrt(C_a)[32 sqrt(C_a)/kappa]^(1/3).
```

U_a is independent of N, Lambda and the fixed positive photon coefficients under the stated E0<=0 premise. This is a finite fixed-cutoff bound; with kappa=2/a it can diverge as1/a and is not a finite interacting UV mass estimate.

## 5. The E0<=0 premise and two-sided original trajectory

At the existence-largefiniteN scope of the reviewed original BO theorem, the actual classical heavy/field minimum v_N is strictly negative. A positive Gram ray gives a negative sqrt(t) term before its positive t² quartic; the native Clifford/flat representative attains that variational value. For each fixed suchN, the full-domain BO neutral trial has total energy

```
E0 <=Lambda v_N +O_N(Lambda^(3/8))+O_N(1),
```

including the entire normal zero-point and fixed electromagnetic energy. The common nonnegative colour/photon magnetic shift is already part of this Hamiltonian. Thus E0<=0 for sufficientlylarge finiteLambda at eachN. No numerical N5 vacuum or globality certificate is inferred.

The reviewed sharp effective threshold plus full BO lower transfer and actual native large-N Schur floor imply, at all sufficientlylargeN and a finite sufficientlylarge choice Lambda(N),

```
L_a := m sqrt(g0,infinity/2)/2 >0,
L_a <= E_micro,+/- - E_micro,0 <= U_a <infinity.
```

Choose Lambda(N) large enough for both E0<=0 and the previously proved charged-lower error requirement. There is no uniform-BO estimate at fixedLambda; photon coefficients remain arbitrary fixed positive values. This joins a uniform-in-N positive lower and finite upper of the same **original full-coordinate compensated sectors** along the explicit trajectory. It does not establish convergence of those thresholds, a separated reference, a thermodynamic state, Coulomb/Z4 phase, localized mechanical spin, photon pole or scattering.

## 6. Native producer and the next physical boundary

`original_local_charge_upper.py` uses four actual full10Gram native frames and arbitrary full-field/noncentral-colour/compact-EM perturbations of the original periodic endpoint owner. It checks trace-quadratic and Gram-quartic bounds without beta/isotropy replacements. Its whole8-mode tree CAR diagnostics retain all neutral/charged configurations and all four reference components, test both references/sites and all15 colour/U1 source identities, and compare full charged Hamiltonian commutators at nonzero electric coefficients. Dropping the reference offset or using the wrong colour representation must reach and fail those same equations. Tree rotor elimination is exact by Gauss; it is not a periodic rotor truncation.

The finite-tree quantum calculation is not a numerical full periodic ground or a proof of the full graph's translation-invariant vacuum. The exact analytic full-state argument owns that claim. Next decisive boundary remains a carrier constrained away from its reference with a uniform energy bound and compact Coulomb/Z4 photon phases; a coincident trial deliberately does not hide that debt.

## 7. Exact photon marginal: co-located and separated single-path packets

The original unreduced antiunitary is
`C=I_Phi P K`, where `P=(-1)^Ntotal product_j(c_j+c_j†)` acts on BOTH
existing CAR flavours, K conjugates wavefunction values at fixed U,A,
and I_Phi reverses all60V Cartesian coordinates. Native Hermitian
traceless onebodies are odd in Phi: PK sends dGamma(h) to its negative,
and I_Phi restores H. C sends pPhi to +pPhi and all colour/photon electric
generators to their negative; both complete Gauss generators reverse.
The background-two charge and both half-filled sectors are essential.
No U->U* or A->-A coordinate pullback, Clifford restriction or selector
is used. This symmetry does not require positive bare m.

For an ACTUAL C-invariant neutral density rho, including the normalized
true ground projector or Gibbs state, full internal integration gives
`w_nL(A)=2w0(A)`. This is not an occupation claim at fixed Phi,U.
Equalities are almost everywhere, pointwise wherever kernels are continuous.
Marginal/norm identities require no energy moment. Electric means, second
moments and current identities below require finite Tr(rho sum_e E_e²).
The phase-energy preparation requires finite FULL physical form energy;
the ground/Gibbs examples satisfy these assumptions.

For any oriented path gamma from insertion L to the static reference R,
let W be its actual unitary SU4 transporter and ell its INTEGER EM cochain,
`div ell=deltaL-deltaR`. Existing remote maps from
`pair_separated_charge_transfer.md` give

```
(B_+ psi)_b = exp(+i ell.A) sum_a W_ab c_La† psi,
(B_- psi)_b = exp(-i ell.A) sum_a (W†)_ba c_La psi.
B_-†B_-=n_L=N_-; B_+†B_+=4-n_L=N_+.
rho_gamma,q=B_q rho B_q†/2.
```

They intertwine all full original Gauss constraints: opposite bar4/4
and EM reference at R, with interior colour/EM cancellation. Smooth
bounded multipliers preserve the full form domain. Cartesian angular
terms and whole heavy CAR remain; they are not replaced by inertias.
The empty/full local null occupations are retained.

For every scalar F(A), `B_q†F(A)B_q=F(A)N_q`, so

```
w_gamma,q(A)=w0(A);
M_(w_gamma,q,lambda)=M_(w0,lambda).
```

Thus EVERY shell and cross-mode element of the normalized periodic
gauge/harmonic-invariant phase response is identical, even at arbitrary
carrier/reference separation. Neutral translation covariance of M is
legitimate without translating the reference or charged matter density.
Incoherent convex path mixtures retain this equality. Arbitrary same-site
coherent multipath sums do not; canonical distinct-site profiles do (§8).
A neutral shell bound is still required; equality does not invent it.

The full reduced photon kernel is more precise:

```
sigma_gamma,q(A,A')=exp[iq ell.(A-A')] sigma_q^(0)(A,A'),
sigma_q^(0)(A,A')=(1/2)Tr_internal[N_q rho(A,A')].
```

Colour W cancels under this full partial trace. In general the weighted
kernel differs from the neutral kernel OFF diagonal. Define
`v_q^(0)=(1/2)Tr rho N_q E`, `K0=Tr rho sum_e E_e²`. Exact product rules give

```
<E>_gamma,q=q ell+v_q^(0),
<sum E²>_gamma,q=K0+2q ell.v_q^(0)+||ell||²,
J_gamma,q(S)=(1/4)Tr rho N_q{E,gradS}
              +q<ell.gradS>_w0.
```

C-even E² fixes the weighted second moment, not the charged current.
The neutral scalar current is zero, but the first term in J_gamma,q
need not vanish. The nonstationary two-Riesz preparation in
`original_massive_photon_locality.md`, Appendix C, must use this actual
packet current. Its phases preserve the reference/Gauss sector and
cannot remove the longitudinal source by a nonperiodic electric shift.

This constructs a genuine separated packet, NOT a uniform-energy or
mobile electron. Its photon kinetic change is
`ue[2q ell.v_q^(0)+||ell||²]`; a simple path has ||ell||²=length.
Colour kinetic cross terms likewise remain, including the Cartesian
angular contribution if Gauss is used. The section5 two-sided threshold
still belongs to the COINCIDENT trial only. No deconfinement, renormalized
mass, velocity, Noether half-spin or scattering follows from marginal equality.

The producer now exercises two actual noncentral N5 links and compact
phases, complete endpoint NL4 dimension70 and NL3/5-reference dimension224,
all30 endpoint colour directions, interior colour cancellation and all
three EM source derivatives. Heavy/intermediate factors remain whole
spectators; this is a local operator diagnostic, not a full periodic
density or a rotor truncation. Missing colour and EM dressings are
rejected at the same physical Gauss assertions. No numerical periodic
ground or separated uniform energy bound is claimed.

## 8. Canonical coherent distinct-site packets

Choose one actual unitary path W_L and integer EM cochain ell_L per
distinct insertion site L, all ending at the SAME fixed reference R.
The identity path is allowed for L=R. For arbitrary coordinate-independent
complex coefficients f_L define the four modes

```
d_f,b† = sum_(L,a) f_L exp(i ell_L.A) W_L,ab c_La†,
d_f,b = (d_f,b†)†.
{d_f,b,d_g,c†} = <f,g> delta_bc.
```

Different-site CAR remove cross-path terms; W_L†W_L=I removes the remaining
colour matrices. The maps `(B_f,+ psi)_b=d_f,b†psi` and
`(B_f,- psi)_b=d_f,b psi` intertwine the SAME complete charged/reference
Gauss equations. With n_f=sum_b d_f,b†d_f,b,

```
B_f,-†B_f,- = n_f,
B_f,+†B_f,+ = 4||f||²-n_f,
C n_f C^-1 = 4||f||²-n_f.
Tr rho n_f F(A) = 2||f||² Tr rho F(A).
rho_f,q = B_f,q rho B_f,q†/(2||f||²),
w_f,q(A) = w0(A),      M_(w_f,q,lambda) = M_(w0,lambda).
```

These exact identities hold for every actual C-invariant neutral density
and every real diagonal photon multiplier F. Polarization gives
`Tr rho B_f,+†F B_g,+=2<f,g>Tr rho F`. Thus a true C-real pure neutral ground
defines an ACTUAL linear isometry `f -> B_f,+Psi0/sqrt2` into the original
charged physical Hilbert space. A C-real ground exists because the original
8V-mode C has square+1 and preserves the ground space. Translation
covariance is inherited only if the chosen neutral marginal has it.

This corrects the overbroad previous exclusion of coherent spatial
profiles. At ONE site a sum of paths instead has coefficient matrix
V=sum_gamma a_gamma exp(i ell_gamma.A)W_gamma; generally V†V is not a
constant frame. Such valid physical states must retain their actual
Wilson-loop Gram and do not automatically inherit the neutral marginal.

The producer consumes both charge signs, complete70→224 endpoint/reference
CAR, two native noncentral paths with the same reference, all30 endpoint
colour generators and the actual U1 derivatives. Ordered particle-hole
plus value conjugation checks the norm relation. The same-site two-path
counterexample reaches and rejects the SAME canonical-complement equation.
No full periodic quantum density is numerically sampled or approximated.

Canonical normalization is not an energy or dynamics theorem. The full
trial form is `T_LL'=(1/2)<B_LPsi0,(H_q-E0)B_L'Psi0>`; it includes actual
colour/photon path kinetic cross terms and light current. It is not a
proven invariant particle Hamiltonian or uniform-separation upper.
Photon OFF-diagonal kernels and currents need not match the neutral ones;
the single-path formulas of §7 cannot be reused with one effective ell.
Finite electric second moments and full form energy have the same
requirements as §7. Nonstationary phase preparation uses each packet's
OWN current. C reverses time, so initial marginal equality is not an
automatic equality for every evolved charged packet.

The coherent preparation/isometry gap is removed. Neutral deep-IR
capacity, deconfined moving charged dynamics, interacting mass/charge,
localized total Noether half-spin and scattering remain open suppliers.

## 9. Even constant-trace reference orientation

The photon identity and true-ground profile isometry need less than four
canonical individual modes. A smooth reference matrix M_R suffices if it
is colour covariant, EVEN under full Phi inversion, and
`Tr(M_R†M_R)=4` on the ENTIRE Cartesian domain. Replace W_L by W_L M_R.
Then

```
{d_f,b,d_g,c†}=<f,g>(M_R†M_R)_bc,
B_f,-†B_f,-=n_f,
B_f,+†B_f,+=4||f||²-n_f,
C n_f C^-1=4||f||²-n_f.
```

The same C pairing/polarization therefore gives the EXACT neutral photon
marginal and site-profile isometry. The individual mode Gram is generally
NOT identity; a summed-complement pass must not be called canonical CAR.
No occupation, field or reference component is removed from the law.

A globally smooth concrete orientation filter is

```
H_R=-i[Phi_1,R,Phi_2,R],
F_R=(I+H_R(H_R²+epsilon² I)^(-1/2))/2,
M_R=2F_R/sqrt(Tr F_R²),       epsilon>0.
```

H is traceless Hermitian/even/covariant; at least one of its eigenvalues
is nonnegative, so `Tr F_R²>=1/4`. There is no zero normalization or excluded
field. M is bounded and its Cartesian derivatives grow at most linearly
with Phi at fixed epsilon; the original quartic moment gives finite full
form energy. Noncommuting Frechet derivatives and kinetic mixed terms
remain. At an isometric Clifford body the filter orients the source-derived
spin-bivector correlation, not an imported spin species.

The producer checks actual reference-field inversion and all15 finite
Cartesian colour derivatives. At R, `-W T M+W[T,M]=-W M T`; dropping the
field term violates the same full endpoint Gauss boundary. The shared
complete70→224 consumer executes canonical identity and oriented frames,
both references, all30 colour/U1 identities and ordered PH+K. The oriented
case rejects the distinct WRONG assertion that its four modes are
canonical, while its true trace complement passes.

This is an oriented PREPARATION with the same neutral photon response,
not localized total Noether half-spin. A disjoint carrier intrinsic
generator still commutes with a reference-only filter; full regional
field/link/photon stress, screening, annular flux and leakage remain.
The new Cartesian cost prevents borrowing the old coincident Ua unchanged.
No uniform separated energy, mobile band, current transfer or scattering
is supplied by this constant-trace construction.
