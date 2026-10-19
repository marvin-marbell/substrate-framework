# Compton dependencies → coupled spinning constraint (conditional attempt)

**Status:** conditional mathematical interface, not a proposed fundamental ontology,
not scientific fulfillment of #222. This artifact starts from the observable and
tries to supply a *coupled* propagation–vertex–spin-response relationship. It does
not derive photons, many-electron statistics, measured parameters or quantum
postulates. The executable is `constraint.py`; no numerical run is asserted here.
Parent integration owns the smoke evidence and the ordinary Compton reference.

## 1. Why this interface, rather than three independent suppliers?

At leading order, with metric (+---), ℏ=c=1 and q=−e, the Compton tensor uses

\[
S_0(r)=\frac{\not r+m}{r^2-m^2+i0},\qquad
\Gamma^\mu_0=\gamma^\mu,
\]
\[
\mathcal M=q^2\bar u(p')\left[
\not\epsilon'^* S_0(p+k)\not\epsilon+
\not\epsilon S_0(p-k')\not\epsilon'^*
\right]u(p).
\]

A convention-dependent common amplitude phase does not affect the probabilities.
The two channels, the numerator, vertex and external Dirac equations are essential:
contracting with k replaces a vertex by the difference of inverse propagators,
\(k_\mu\Gamma^\mu_0=S_0^{-1}(r+k)-S_0^{-1}(r)\), after removing the displayed
propagator's overall Feynman i. Adjacent inverse propagators collapse, endpoint
terms annihilate the external states, and the channels cancel. At the interacting
level this becomes the Ward–Takahashi identity for consistently renormalized
propagator and vertex. A chosen propagator and an unrelated current generally
cannot pass it. Matching only the scalar denominator or the Thomson coefficient
would not test this coupling.

The Ward identity fixes the *longitudinal* compatibility, not the entire vertex.
The spin response is a transverse discriminator: a Pauli vertex is gauge
compatible but changes magnetic splitting and subleading Compton spin response.
Thus the observable suggests testing whether one constraint can supply a common
first-order numerator, minimal charge vertex and its associated magnetic term.
It does **not** single out worldlines, an ontology, or a unique fundamental g.

## 2. Complete reduced variables and bracket convention

Use commuting \(x^\mu,p_\mu\), four odd \(\psi^\mu\), one odd \(\psi_5\), an
external real U(1) potential \(A_\mu(x)\), and real constants m>0 and q. Define
\(F_{\mu\nu}=\partial_\mu A_\nu-\partial_\nu A_\mu\) and
\(\pi_\mu=p_\mu-qA_\mu\). The nonzero reduced brackets are

\[
\{x^\mu,p_\nu\}=-\delta^\mu_\nu,\qquad
\{\psi^\mu,\psi^\nu\}=-i\eta^{\mu\nu},\qquad
\{\psi_5,\psi_5\}=i.
\]

All mixed brackets and \(\{x,x\},\{p,p\}\) vanish. The bracket is even,
graded antisymmetric and obeys the graded Leibniz and Jacobi identities. This
explicit symplectic sign choice permits \(\hat p_\mu=i\partial_\mu\), physical
plane waves exp(−ip·x), and \([\hat f,\hat g]_{\rm graded}=i\widehat{\{f,g\}}\).
Equivalently the conventional canonical variable is \(c_\mu=-p_\mu\), with
\(\{x,c\}=+\delta\); no physical charge is reversed by that change of variables.
In particular,

\[
\{\pi_\mu,\pi_\nu\}=-qF_{\mu\nu},\quad
\{\pi_\mu,f(x)\}=\partial_\mu f,\quad
[\hat\pi_\mu,\hat\pi_\nu]=-iqF_{\mu\nu}.
\]

The brackets are reduced after the odd canonical second-class constraints; odd
canonical momenta are not additional degrees of freedom. An even multiplier N
and an odd multiplier χ enforce H=0 and Q=0 in the total generator NH+χQ; their
momenta are nondynamical constraints. They are not extra photon variables.

Take the *minimal linear ansatz*

\[
Q=\pi_\mu\psi^\mu-m\psi_5,\qquad
H=\tfrac12[\pi^2-m^2-iqF_{\mu\nu}\psi^\mu\psi^\nu].
\]

Direct use of the brackets gives

\[
\{Q,Q\}=-i\pi^2+im^2-qF_{\mu\nu}\psi^\mu\psi^\nu=-2iH,
\quad \{Q,H\}=0,\quad\{H,H\}=0.
\]

The QH relation also follows from graded Jacobi; the covariant-momentum Jacobi
identity requires dF=0, already guaranteed by F=dA. These are first-class
constraints, not an arbitrary spin force glued to a scalar mass shell. Omitting
the spin term while leaving Q unchanged produces the nonzero closure residual
\(-qF_{\mu\nu}\psi^\mu\psi^\nu\), even for a constant nonzero field.
`classical_probe()` computes QQ and QH in the exterior algebra for all six
independent constant F components, and exhibits both wrong-case residuals.
Constant-field verification is not a numerical proof for arbitrary backgrounds;
the arbitrary smooth-background extension here is the displayed Jacobi argument.

## 3. Clifford quantization, mass sign and the g=2 condition

Assume canonical graded quantization and an irreducible four-component Clifford
representation. With \(\gamma_5=i\gamma^0\gamma^1\gamma^2\gamma^3\), use

\[
\hat\psi^\mu=\frac{i\gamma_5\gamma^\mu}{\sqrt2},\qquad
\hat\psi_5=\frac{i\gamma_5}{\sqrt2},\qquad
\sigma^{\mu\nu}=\frac i2[\gamma^\mu,\gamma^\nu].
\]

Their anticommutators are \(\eta^{\mu\nu}\), −1, and zero respectively. Let
\(P_\mu=i\partial_\mu-qA_\mu\), \(D=\gamma^\mu P_\mu\). Then

\[
\hat Q=\frac{i\gamma_5}{\sqrt2}(D-m),\qquad
\hat Q^2=\hat H=\tfrac12\left[P^2-m^2-\tfrac q2\sigma^{\mu\nu}F_{\mu\nu}\right].
\]

The sign −m in Q therefore gives the standard positive-mass Dirac equation
\((D-m)\Psi=0\), not a tachyonic mass constraint. In the square,
\(D^2=P^2+\frac14[\gamma^\mu,\gamma^\nu][P_\mu,P_\nu]
=P^2-\frac q2\sigma^{\mu\nu}F_{\mu\nu}\).
The zero constraint alone on H admits more solutions than Q=0; retain Q when
mapping physical spinor states. Squaring without that endpoint restriction is
not equivalent to the first-order electron theory.

For a magnetic field B in the z direction, covariant \(F_{12}=-B\), and
\(\sigma^{\mu\nu}F_{\mu\nu}=-2B\Sigma_z\). Thus

\[
E^2=m^2+\boldsymbol\Pi^2-qB\Sigma_z,
\qquad H_{\rm NR}=m+\frac{\boldsymbol\Pi^2}{2m}
-\frac{q}{2m}\boldsymbol\sigma\cdot\mathbf B+\cdots.
\]

With \(\mathbf S=\boldsymbol\sigma/2\) and
\(H_{\rm spin}=-gq\mathbf S\cdot\mathbf B/(2m)\), this is **g=2**.
This coefficient follows *conditionally* from the same minimal linear Q,
minimal covariant momentum, closure, Clifford quantization and first-order
physical-state constraint. It is not selected by U(1) symmetry alone or proven
fundamental. Representation, quantization, m and q were assumed, not explained.

`operator_probe()` independently realizes P as differential operators in the
Landau gauge \(A_2=-Bx\). It squares Q on a four-component nonconstant polynomial,
retaining derivatives and the field commutator, compares to H, and checks the
missing-spin case. It also computes exact same-orbital Landau spin energies,

\[
E_{n,s}^2=m^2+p_z^2+|qB|(2n+1)-qBs,
\quad n\ge0,\quad s=\pm1.
\]

For its declared inputs m=e=1, q=−1, B=1/10, n=0 and pz=0, the derived energies
are \(E_{+}=\sqrt{6/5}\), \(E_{-}=1\), whereas the missing-Pauli square gives
both \(\sqrt{11/10}\). These are analytic expectations pending the parent run,
not reported numerical observations. Comparing the same oscillator index matters:
optimizing orbital states independently could obscure the intended spin test.

## 4. Gauge-compatible anomaly changes the ansatz, not gauge invariance

Add the gauge-invariant effective interaction

\[
\mathcal L_{\rm Pauli}=-\frac{\kappa q}{4m}\bar\Psi\sigma^{\mu\nu}F_{\mu\nu}\Psi,
\quad T=\frac{\kappa q}{4m}\sigma^{\mu\nu}F_{\mu\nu},\quad L_\kappa=D-m-T.
\]

For an incoming photon exp(−ik·x), the stripped vertex is
\(\Gamma^\mu(k)=\gamma^\mu+i\kappa\sigma^{\mu\nu}k_\nu/(2m)\): the same
κ convention as the first-order Compton API, with transfer k incoming and −k′
outgoing. Its added term contracts to zero with k, so the Ward identity remains
compatible with the unchanged free propagator. The low-field nonrelativistic
magnetic response has \(g=2(1+\kappa)\). This parameter is an allowed independent
low-energy coefficient; radiative QED corrections are not computed here.

A corresponding cubic odd constraint is

\[
f=F_{\mu\nu}\psi^\mu\psi^\nu,\qquad a=\frac{\kappa q}{2m},\qquad
Q_\kappa=Q-iaf\psi_5.
\]

For constant F its exact classical closure gives

\[
H_\kappa=\tfrac12(\pi^2-m^2)-\tfrac i2q(1+\kappa)f
+\tfrac12a^2 f^2-2ia\pi^\mu F_{\mu\nu}\psi^\nu\psi_5.
\]

For variable F, the additional term
\(a\partial_\rho F_{\mu\nu}\psi^\rho\psi^\mu\psi^\nu\psi_5\) vanishes when
dF=0. The executable computes the deformed bracket directly and reports the
terms missed by merely rescaling the minimal spin coefficient. A new closed
constraint is permitted; **minimal** Q with an arbitrarily altered H is not.

At operator level the unambiguous prescription is to quantize Qκ first and square:

\[
\hat Q_\kappa=\frac{i\gamma_5}{\sqrt2}L_\kappa,\quad
2\hat H_\kappa=(D+m+T)(D-m-T)
=D^2-m^2+[T,D]-2mT-T^2.
\]

Here \([T,D]=[T,\gamma^\mu]P_\mu-i\gamma^\mu\partial_\mu T\), so field,
spin–momentum and field-gradient effects accompany the anomaly; T² also remains.
Naive substitution of Clifford matrices into a classical quartic polynomial need
not equal this quantum operator because contractions/ordering can contribute.
The executable's differential operator check keeps the full commutator and T².
No equality between naive classical-H quantization and this prescription is
claimed. A generalized worldline construction can accommodate such terms; it
must not discard them while claiming minimal closure or a unique g.

## 5. Exact map to the first-order Compton consumer: contact AND endpoints

For κ=0 set L=D−m, B=D+m, K=LB=BL=D²−m². With compatible boundary conditions,

\[
S=L^{-1}=BK^{-1},\qquad
S_0(r)=(\not r+m)/(r^2-m^2+i0).
\]

The Feynman prescription and quantum propagator interpretation are imported.
The numerator map, its m sign and its physical-state restriction are supplied by
this constraint representation. Coupling A through P gives δL=−qγ·δA, precisely
the charge vertex used by the ordinary reference. Multiplication by the usual
Feynman i's recovers its diagram convention.

A second-order calculation is **not** just two second-order single-photon
vertices. Write \(D=D_0+V\), \(V=-q\gamma\cdot A\),
\(G_0=(D_0^2-m^2)^{-1}\), \(B_0=D_0+m\). Then

\[
K=K_0+K_1+K_2,\quad K_1=\{D_0,V\},\quad K_2=V^2=q^2A^2\mathbf1.
\]

The differential K1 contains both the orbital current and the field-strength
spin vertex. For two photon fields V1,V2, the **contact** insertion is
\(K_{12}=\{V_1,V_2\}=2q^2\epsilon_1\cdot\epsilon_2\mathbf1\), including
plane-wave factors. It is the spinor second-order seagull; first-order QED has
no independent two-photon electron vertex. The mixed second-order coefficient
of the *full first-order propagator* is

\[
\begin{split}
S_{12}={}&B_0G_0(K_1^{(1)}G_0K_1^{(2)}+K_1^{(2)}G_0K_1^{(1)})G_0\\
&-B_0G_0K_{12}G_0
-(V_1G_0K_1^{(2)}+V_2G_0K_1^{(1)})G_0.
\end{split}
\]

The second line has both the contact and the **endpoint numerator variation**
δB=V. To verify the map without any appeal to a vanishing seagull, use
\(S_0=B_0G_0\) and the operator identity

\[
VG_0-S_0\{D_0,V\}G_0=-S_0VS_0,
\]

which follows from S0D0=1+mS0. Expanding BK⁻¹ to mixed order, or multiplying
LS=1 and matching that order, then gives exactly

\[
S_{12}=S_0V_1S_0V_2S_0+S_0V_2S_0V_1S_0.
\]

This identity permits noncommuting differential operators; derivatives must act
on the photon fields as well as the next wavefunction. On amputating the two
external S0 poles and selecting external states satisfying (slash p−m)u=0,
\(\bar u u=2m\), the right side gives the displayed two Compton channels, with
r=p+k or p−k′ and outgoing polarization conjugated. This is the analytic map to
the parent's first-order API. Endpoint/contact contributions on the left combine
to that right side; they have not been separately set to zero on shell.
`endpoint_probe()` checks this resolvent identity at exact rational off-shell
matrix inputs with nonzero photon-polarization scalar product. Removing either
contact or endpoint fails its equality. It is an adverse identity check, not a
numerical evaluation of second-order Compton kinematics.

For κ≠0 use \(B_\kappa=D+m+T\),
\(K_\kappa=B_\kappa L_\kappa=2\hat H_\kappa\), and
\(S_\kappa=K_\kappa^{-1}B_\kappa\). Unlike κ=0, the order matters. For
δL=V−δT and δB=V+δT,
\(\delta K=(\delta B)L+B\delta L\) and the mixed coefficient also includes
\((\delta B_1)(\delta L_2)+(\delta B_2)(\delta L_1)\).
The identity L⁻¹=K⁻¹B retains these contacts and the right endpoint; expansion
therefore yields the first-order Pauli vertices above. Merely rescaling a
second-order spin vertex and reusing the minimal contacts/endpoints is wrong.

## 6. Input/debt ledger and the next constructive bridge

No parameters are fitted to Compton in this slice. Two constants m and q are
imported (reference m=e=1 is a normalization, not a physical derivation); κ is
an optional third coefficient, not derived. Six structural input blocks remain:

1. Lorentzian spacetime and relativistic phase-space kinematics.
2. Odd variables, their symplectic brackets, and the linear minimal Q ansatz.
3. First-class constraint interpretation and physical Q-state selection.
4. Canonical quantum postulates, Clifford representation, complex amplitudes,
   positive-energy state normalization and causal/Feynman prescription.
5. A U(1) potential, its gauge law and minimal charge coupling. This slice uses an
   external background; Maxwell dynamics and quantum helicity states are absent.
6. The many-body/field completion, exchange statistics, charged-state dressing,
   renormalization and observable infrared resolution beyond the tree reference.

What is gained conditionally: one Q and its closure lock propagation, the minimal
vertex and the leading magnetic coefficient together; one resolvent identity
locks their second-order contacts and external endpoints to the two-channel
Compton consumer. That reduces independent *matching choices*, not explanatory
input debt. Counting six blocks is bookkeeping, not a measure of explanatory
complexity. No microscopic degrees of freedom, stable finite-energy electron,
fermionic exchange law, photon quantization, electron mass/charge values or unique
fundamental g have been derived. Grassmann variables alone describe spin algebra;
they do not imply exchange antisymmetry of separated particles. This is currently
**a conditional interface/reencoding of the Dirac sector**, not a deeper ontology.

The next constructive bridge is to derive rather than declare a low-energy odd
charge Q_eff from a candidate with specified microscopic variables and dynamics.
It must simultaneously supply its graded/symplectic algebra, gauge action and
physical-state map. Project the candidate onto one shared state space, compute
QQ and the current variation in a nonzero background, then independently compute
same-orbital spin splitting and the Compton polarization/spin amplitude. If the
projection supplies an extra transverse coefficient κ, retain it and predict it
from that same construction instead of tuning it separately. Failure of closure,
wrong endpoint normalization, an unexplained imposed Clifford algebra, or an
independently fitted κ exposes exactly where the bridge remains conditional.
This specifies an operational missing interface without preselecting Euler,
rotor networks, worldlines or any other microscopic root.

## Sources actually read and their limited use

- [Issue #222](https://github.com/vantasnerdan/substrate-framework/issues/222):
  observable-first target and explicit explanatory obligations; not evidence
  that this attempted supplier establishes them.
- D. G. Gagné, [*Worldline Supersymmetry and Dimensional Reduction*,
  hep-th/9604149v1, introduction and §2](https://arxiv.org/html/hep-th/9604149v1):
  identifies Dirac supercharge/Hamiltonian-squared constructions and warns that
  higher-form couplings require terms missed by an overly simple superworldline
  action. That work is a field-theory-derived reformulation, not a substrate
  derivation. Its Euclidean conventions are not copied into the Minkowski
  equations above, which are derived explicitly here. Literature suggests this
  conditional interface; it does not prove its transfer to a microscopic root.
