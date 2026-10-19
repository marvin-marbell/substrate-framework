# Same-current static scattering: charged/neutral flight and full-Phi opposed-leg transfer

Governing plan: `issue-222-synthesis-continuation-20261006`, pinned general template
`89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`.
This new pair does not edit or replace the frozen inclusive-join/receiver suppliers.
The executable records their actual source hashes before and after execution.

## 1. Selected physical problem and units

The selected descriptor is `issue222-legal-volume-far-RIGHT-global-Gram-b4096-descriptor.json`:

- N = 5 times 2^96, V = N^3, D = N/2;
- source origin (D,D,D), genuine colour-tree RIGHT (0,D,D);
- maximum source word D+2 and source harmonic displacement (D+1,1,1);
- exact binary64 canonical c = 1130202988249777/9007199254740992;
- UE = UE0/2^4096, K = c^2/(2UE), both strictly positive.

4096 is the TOTAL division exponent, not an extra division after the older p527
experiment. The latter short-source experiment is not admission of this geometry.
Native endpoint floats receive an explicit rational slack from the supplied
full-ten Gram interval, including a binary roundoff allowance. Hard-mode weights,
frequencies, source coordinates and V are not replaced by large-N floating values.
Every square root is rounded upward by integer arithmetic; serialized rationals
retain numerator/denominator and both outward decimal endpoints.

The existing `source_matrix` is a KERNEL of HS squared 4. The creation matrix is
`source_normalized_creation_matrix = M/sqrt(2)`; the raw q/2 matrix is M/2. Their
normalizations are not interchangeable. The physical screened core has source
Gram s/2, photon Gram s/4, and input norm sqrt(s/4), approximately .4894562031,
not one. A common CONSTANT chart rescales B by sqrt(s/2) and both photon/source
vectors by sqrt(s/4). This leaves the physical normalized input and global output
projectors unchanged. The flight code conservatively uses the global screened
q0 upper 16, despite the tighter positive-Haar/Jensen operator upper 2.

The physical source is the mobile Coulomb U1 dressing, harmonic phase, and a
SU4-only tree. There is NO uncompensated coexact electromagnetic Wilson string.
For the supplied transverse hard packets, div(g)=0 and the harmonic component is
zero. Therefore [a_g,B]=0 exactly, even on an entangled microscopic state. This is
an operator identity, not a differentiation of Omega or a factorization of fields.

## 2. The actual receiver and the static bridge

The input remains raw q = a_g^dagger B_eta Omega/sqrt(2), divided ONLY by its actual
norm. At the readout T the two receiver vectors x_F,x_R are STATIC physical vectors
in the complete Hilbert space. Set G = X^dagger X, Y = X G^(-1/2), and use the two
coherent effects |Y_i><Y_i| and their complement. The four bar-reference entries
are internal entangled components; they are not four additional outcomes.
Neither input whitening nor a reference trace is substituted.

In the fixed normalized-core chart the leading scalar is

    mean(C_static) - .5 mean(g1_RF) mean(m_F0).

The product is taken AFTER the actual coherent integration, not fibre by fibre.
Finite-alpha changes of global Born G0 and input n are retained by the supplier's
global disk. For general diagonal Born G0, the exact inverse-square-root divided
difference replaces -.5. This executable pays additional ground/microscopic
changes with |f'|<=4 and |f''|<=24 on eigenvalues >=1/4; it never resets actual G0
or n to identity/one. The small offGram still carries UE. In particular the
cloud cross entries are swapped-source contractions, not a presumed positive
2-by-2 cloud addition.

To use the frozen interaction-picture native helper for a STATIC detector:
backpropagate the FULL detector by free time -T and multiply each outgoing
annihilation coefficient by exp(-i omega T). The T0 offGram uses the FORWARD
receiver photon centered at origin+(1,0,0), not the original incoming g. The bare
forward amplitude contains both the photon overlap and the complete-CAR electron
autocorrelation. One must not subtract C_0 from C_T or discard the negative final
detector component.

`static_native_bridge` compares this three-kernel identity with independent
bordered exterior determinants, retaining all 250 negative occupied columns at
N5 and all four reference rows. It uses the actual axial far-RIGHT primitive
source at this finite N, with no positive-band source filter. The legal-volume
screened value is supplied by the separate interval/coherent-current evaluator,
not inferred from the N5 value.

## 3. Full-CAR occupation payment before photon amputation

Let P be the entire negative projector of h0(a_h), and Esea its actual scalar
trace. For any CAR state its onebody density R obeys 0<=R<=I and

    ||R-P||_HS^2 <= Tr[(1-P)R + P(1-R)] = n_exc.

The spectral gap and Hilbert-Schmidt Young inequality give the OPERATOR form

    dGamma(h0+D)-Esea >= (mu/2)n_exc + Tr(PD) - ||D||_HS^2/mu.

No positive-particle restriction or fermionic-sector truncation is used. On a
native coexact zero-mean field the linear trace vanishes because each directional
bond trace is spatially constant. The literal endpoint exponential supplies

    |Tr P(D-J_linear)| <= 4 tau sum_e A_e^2,
    ||D||_HS^2 <= 8 tau^2 sum_e A_e^2.

The native consumer minimizes the resulting quadratic form over ALL 2^500
occupations by the exact onebody negative-part spectral reduction. It retains
Esea and rejects dropping the Young term.

Compare the actual neutral near-infimum state with the filled-sea/coexact-vacuum
trial using the SAME scalar harmonic operator, rather than dropping the varying
sea scalar. Its determinant-line polar kinetic cost is explicitly paid. With
S_N from section 5, H_EM<=Q+S_N and ||W^dagger Omega||^2>=1/2 give the conservative
point variance sigma^2 <= (4 g_star/K)S_N, g_star=13/16. The code obtains an O(UE)
excitation density (with finite-N coefficients), then pays BOTH the occupied-sea
projection leakage and the actual marginal's excited mass. It does not call the
projected state an exact effective ground.

For a smooth bounded body map F_s with q0..q6 phase jets,

    R_F = i Fdot - (H_target F - F H_neutral)

contains the literal coexact Peierls error, the ENTIRE neutral/charged Coulomb
operators, and harmonic electric cross and second-derivative terms. Onebody
Touchard/Cauchy/positive-particle-hole heat bounds control derivatives; no extensive
manybody differentiated propagator is substituted. Unitary Duhamel bounds F_s
on source states and on current/contact/cloud-source states. Positive band
resolvents pay R_B U_g in its actual positive energy graph, not by multiplying
||R_B Omega|| and ||a_g Omega||.

The raw bound includes both real ordered histories, the single contact, outgoing
KET PRE cloud, incoming BRA POST cloud, and the swapped-source cloud exchange.
It also supplies the static bare forward and offGram errors before global
output-only normalization.

## 4. The full-Phi word closure: move the currents to opposite source legs

Write either real history as

    < J_h U_+^dagger(T-t2) B_R Omega,
      U_+(t2-t1) J_g U_+(t1) B_F Omega >.

This placement is decisive. It needs one current applied to each evolved SOURCE
state, not a claim that an unbounded current preserves a half-energy graph.
PRE/POST similarly use B U_d Omega as one source leg; exchange has no current.

Put Q_q=H_q-E0_neutral. The compensated charged Q_q is strictly positive by the
same all64 untruncated mass transfer. For a source and any u in D(Q_0),

    ||Q_q^(1/2) B u||^2
      <= B0^2 ||u|| ||Q_0 u|| + B0 ||u|| ||(Q_q B-B Q_0)u||.

If a free-flow residual RB is used instead, i Bdot MUST be added. The executable
bounds the stationary commutator directly and also includes an explicit Bdot
allowance. It does not assume the previously unsupplied RB_U half-energy graph.

The commutator contains bounded full-Haar Cartesian/native15-link first/second
source derivatives, weighted normal momenta, light rho multiplication, Coulomb,
and the actual far harmonic jets. For chi=Lambda^(-1/2), on ANY physical form-domain
u (not only a normal projection), the retained original kinetic form obeys

    chi [||Pi_Phi u||^2 + .6 ||Pi_U u||^2]
      <= 2 [ e_u + (Hvac Lambda^(3/8)+U0+d0+C0)||u||^2 ].

Consequently chi Pi source terms are bounded uniformly for Lambda>=1, since
chi Lambda^(3/8)=Lambda^(-1/8). This use of the scalar normal vacuum is legitimate
for the normal momentum bound; it is NOT a fictitious photon energy subtraction.
The p1/p2 source bounds include the physical metric conversion and the actual
D+2 colour word. They are not bounds on the physical Noether generator.

The global quartic form, with rho^2=sum Tr(Phi^2)/4, gives

    rho^4 <= a_N I + b_N Lambda^(-1) Q_q.

Apply the operator-monotone square root (on the closed forms), then scalar
Cauchy-Schwarz in the spectral measure. For any finite half-energy v,

    ||rho v|| <= C_rho ||v||
       + C_Q Lambda^(-1/4) ||Q_q^(1/2)v||^(1/2) ||v||^(1/2).

A source propagation defect has norm O(d+source-image-error) by the bounded W
residual, while its half-energy is bounded by the sum of the two SOURCE energies.
This interpolated rho bound therefore controls the current on that defect.
The current/contact W images are the existing literal endpoint images for
ARBITRARY electromagnetic phases and the whole CAR fibre. The central propagator
then telescopes on bounded effective-current states. There is no [Q,J] assumption,
no factorized excited-state moments, and no need to close a higher-current graph.

The code constructs an exact finite polynomial sandwich multiplier for both real
histories, contact, PRE/POST, exchange, bare metric, cross metric, and global
whitening. It chooses a smaller normal residual, a larger legal Lambda, and an
additional exact t^(-20) inequality (Lambda=t^80). The actual source photon birth
has finite Q graph from [Q,a_g^dagger]=omega_g a_g^dagger+sqrt(UE)J_g^dagger; source
application then gives finite input and receiver half-energy bounds. Thus the
real prepared original states, not just conditional diagrams, have finite energy.

## 5. Photon half-graph and the spectral state

N is even; NH=2V is divisible by four. The pure-colour NL0 construction is therefore
available on this actual common N (it is not inferred from odd N5). Remove the
light/photon terms in the normal proof, not by subtracting an operator inequality:
Kcol-Emin+delta_col>=0, retaining every exact all64 Etheta. Kcol strongly commutes
H_EM. The pointwise light bound and global phase coercivity give H_EM<=Q+S_N once

    L_H^2/(Lambda c_phase)<=1,
    L_H^2 Hvac/(c_phase Lambda^(5/8))<=1,
    d0<=1/4, delta_col<=1/2.

These are explicit logarithmic choices in the executable. The conservative shift
is S_N=16V+UE V N^2/4+3UE/(2V). No Lambda times classical minimum is used as the
photon vacuum. The original negative vstar is retained in the normal momentum
form and only dropped where doing so is an UPPER bound on the quartic moment.

Choose Omega from a nonzero smooth spectral filter of Q_0 with support [0,epsilon]
using a dense physical trial family. Select a C-real neutral kappa0 fibre, or an
explicit kappa/-kappa pair, under the original full-field antiunitary. The normal
Kato/Slater W is C-equivariant. A moving arbitrary fixed kappa is not declared
C-real, and a time-free B_t is not declared Phi-even. The static bare source metric
is one half of its ACTUAL field-HS expectation; the photon-created norm retains
its positive cloud term. No generic matrix Loewner positivity of the crosscloud
is assumed.

Epsilon is chosen AFTER the finite sandwich multiplier, and below the body
amputation width. The hard-band pull-through remainder has actual Q graph and
is included. This produces a normalizable finite-energy state even if one works
on a noncompact harmonic direct integral. No isolated charged pole, and no tiny
Gamma ground uniqueness gap, enters this construction.

## 6. Executable evidence and decision boundary

Run the scientific venv Python with this script, the exact descriptor, T=6,
--n 3 --M 8 --finite-time 1 --nodes 4 --native-steps 64 and an output JSON path.
The output preserves before/after source hashes, exact input SHA, interpreter and
argv. --bounds-only executes the exact body/full-Phi/global-receiver construction
without pretending that it performed the finite numerical consumers.

The complete run additionally exercises:

1. The two-site COUPLED model with three coexact photon levels, seventeen harmonic
   levels, complete 70/56 neutral/charged exterior sectors and all four references:
   direct charged/neutral current, cloud and contact-state propagation versus exact
   Duhamel. It rejects dropping Coulomb/reference, harmonic electric, or coexact
   Peierls flight. This is not a positive-band truncation.
2. The native N5 500-dimensional onebody/full-occupation Young form, retaining all
   250 occupied columns and the actual scalar sea energy.
3. The actual axial far-RIGHT N5 static source/receiver, independent complete
   bordered exterior derivatives, T0 offGram and bare forward kernel. The moving
   C_T-C_0 target and removal of the negative final detector fail.

The legal-volume finite-parameter bound is a derived original-state theorem;
these finite experiments exercise its identities and adverse alternatives, not
an enormous-volume numerical simulation. The receiver effect contains the whole
photon Hilbert space; no exclusive one-photon occupation is imposed. Its specified
operational measurement is exactly the original coherent two-vector POVM.
Physical half-spin/Noether pulse and charged transport are independently owned
joins; this pair does not alone declare LAW-05 closed.

Two concrete continuations: (A) join these actual finite-energy receiver/source
bounds at the selected common Lambda with the physical rotation and mobile packet
receipts; (B) replace the coarse whole-energy point variance by a sharper same-state
variance to test whether a materially larger UE retains the SAME global detector.
Select A: it changes the joined physical decision without changing the current,
source, receiver, normal-domain proof or hard-photon legs.
