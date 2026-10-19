# Backward construction: the band, its charge-current vertex and its contact term

**Status: conditional construction, not a derived electron or photon ontology.**
The calculation is selected by Compton's coupled requirements: pole residues,
physical transverse quantum states, charge-current compatibility and a soft
response normalized by the electron's mass. It joins the supplier interfaces
at the scattering amplitude, rather than collecting separate likenesses.
Lattice Hamiltonians are standard research tools; this is not a claim to have
invented Wilson fermions or a new fundamental theory.

## 1. Which target relationship is uncertain?

A proposed massive carrier band and a proposed transverse photon band do not
specify their interaction. Assigning an arbitrary current after choosing their
propagation can violate the Ward identities. Conversely, satisfying gauge
identities does not force the relativistic dispersion, the inertial/rest mass
relation or fermionic statistics. The first constructive question is:

> Can one explicit quantum-link coupling produce the pole terms and the contact
> contribution required by its own charged band, and approach the reference
> Compton amplitude without silently replacing that band by the Dirac one?

The target fixes the probes before the candidate: incoming and outgoing Ward
contractions; the Thomson limit; finite-energy spin/polarization sums;
orientation dependence; and the species content invisible to a one-electron
low-momentum fit. `lattice_compton.py` implements the following candidate.

## 2. Declared construction and imports

Use continuous time, a cubic spatial lattice of spacing `a`, four-component
fermionic site operators with canonical anticommutation relations, and link
quantum gauge variables. The weak-field, noncompact quadratic photon sector is
used for this tree calculation; transferring it from a compact rotor's Coulomb
phase is a separate obligation. The CAR, complex quantum Hilbert space,
Clifford matrices, gauge constraint and parameters are **inputs**. This is a
conditional mathematical supplier, not an emergence proof for those inputs.

The free single-fermion Hamiltonian is

\[
h_a(\mathbf p)=v\sum_i\alpha_i\frac{\sin(ap_i)}a+\beta M_a(\mathbf p),\qquad
M_a=m+\frac r a\sum_i(1-\cos ap_i),
\]

with `alpha_i=gamma^0 gamma^i`, `beta=gamma^0`, `m>0`, `r>=0`.
Its energies are `±E_a`,

\[
E_a(\mathbf p)^2=v^2\sum_i\frac{\sin^2(ap_i)}{a^2}+M_a(\mathbf p)^2.
\]

The vacuum interpretation requires filling the negative band and normal
ordering the charge; the calculation uses external positive-band electrons and
the **full** four-component resolvent for internal lines. It does not discard
the negative band or derive Fock quantization from one-particle mechanics.
The positive-energy spinor is normalized to `u_a†u_a=1`:

\[
u_a(\mathbf p,s)=\begin{pmatrix}
\sqrt{(E_a+M_a)/(2E_a)}\,\chi_s\\
v\,\boldsymbol\sigma\cdot\sin(a\mathbf p)\,\chi_s/
[a\sqrt{2E_a(E_a+M_a)}]
\end{pmatrix}.
\]

Each hopping operator is gauged with the **same** link
`U_i(x)=exp[-iq a A_i(x+a e_i/2)]`, `q=-e`. In this convention a constant
vector potential shifts `p -> p-qA`. The scalar coupling is `q A_0`.
Changing a band term requires changing its link paths as well; copying a
continuum `q gamma^mu` vertex onto a different band is not this construction.

The photon dispersion and physical polarization condition are

\[
\widehat k_i=\frac{2\sin(ak_i/2)}a,\qquad
\omega_\gamma=c_\gamma|\widehat{\mathbf k}|,\qquad
\widehat{\mathbf k}\cdot\boldsymbol\epsilon=0.
\]

The two transverse harmonic oscillators become photon quantum states only
because quantum canonical commutation relations and Fock quantization are
accepted at this scope. Small momentum recovers transverse helicity `±1`;
finite lattice spacing does not have full rotational or Lorentz symmetry.
Setting `v=c_gamma=1` is imported speed matching, not a derived attractor.

## 3. Derive the vertices from that band

Write `b_i=a(p'_i+p_i)/2`. Expanding the gauged hopping gives the contracted
one-photon operator

\[
V(p',p;\epsilon)=q\left[\epsilon^0 I-
\sum_i\epsilon^i(v\alpha_i\cos b_i+r\beta\sin b_i)\right].
\]

Expanding to second order gives the two-photon contact operator

\[
W(p',p;\epsilon,\epsilon'^*)=q^2a\sum_i\epsilon^i\epsilon'^{i*}
[-v\alpha_i\sin b_i+r\beta\cos b_i].
\]

The factor `1/2` in the Taylor expansion cancels the two external-photon
contractions. `W` is not an optional force or a fitted correction. It follows
from the same link exponential as `V` and `h_a`.

For a photon entering a vertex with spatial momentum `k=p'-p`,

\[
\sum_i\widehat k_i(v\alpha_i\cos b_i+r\beta\sin b_i)
=h_a(p')-h_a(p).
\]

Thus a gauge polarization `(omega,khat)` yields
`V=q[omega I-h_a(p')+h_a(p)]`. This is the exact lattice difference identity,
not the continuum identity with `khat` quietly replaced by `k`.

## 4. Joined tree scattering calculation

For an initially stationary electron, choose incoming spatial momentum `k`
and outgoing direction `n'`. Solve the **candidate's** energy equation

\[
m+\omega_\gamma(k)=E_a(k-k')+\omega_\gamma(k'),\qquad k'=t n'.
\]

The code selects the low-Brillouin-zone root in `[0,2|k|]` for its stated
inputs. It makes no high-momentum, Umklapp or multiple-root claim. Define
`G(z,p)=[zI-h_a(p)]^{-1}`. The reduced kernel is

\[
T_a=u_a(p')^\dagger\big[
 V_{\rm out}G(E_i+\omega,k)V_{\rm in}
 +V_{\rm in}G(E_i-\omega',-k')V_{\rm out}+W
\big]u_a(p),\quad p=0,\ p'=k-k'.
\]

Every intermediate vertex uses its actual endpoint momenta. When an incoming
polarization is replaced by `(omega,khat)`, the first pole reduces to
`q u_f† V_out,s u_i` and the second to `-q u_f† V_out,u u_i`. Their difference
is precisely canceled by the contact term, using the link trigonometric
identity. The outgoing contraction follows in the same way. **Deleting `W`
is a concrete wrong construction**, despite preserving the free bands and
the single-photon vertex.

For comparison only, define `M_a=2 sqrt(E_i E_f) T_a`. In the continuum,
`G(z,p) gamma^0=(slash P+m)/(P^2-m^2)` and
`gamma^0 slash epsilon=-alpha·epsilon` for physical temporal-gauge photons.
Consequently this normalization approaches the reference covariant amplitude
up to its immaterial common overall sign. The code averages its squared
modulus over initial spin and polarization and sums final spin and
polarization exactly as in the reference.

**At finite `a`, this squared matrix element is not itself a cross section.**
For the declared canonical photon Hamiltonian
`H_gamma=(E^2+c_gamma^2 B^2)/2`, each external photon field contributes
`1/sqrt(2 omega V)`. Golden-rule state counting then gives a model-defined
tree cross section per outgoing **wave-vector** solid angle:

\[
\frac{d\sigma_a}{d\Omega_{k'}}=
\frac{\overline{|T_a|^2}\,|k'|^2}
{16\pi^2\omega\omega'\,|\mathbf v_{\gamma,\mathrm{in}}|\,
 |(\mathbf v_{\gamma,\mathrm{out}}-\mathbf v_{f,\mathrm{out}})\cdot n'|}.
\]

Here `v_gamma=grad_k omega` and `v_f=grad_p E_a`. The incident photon beam
flux is its group speed, not a silently imported continuum value. The final
radial denominator comes from the candidate energy-conservation delta
function. The electron is initially at rest. At finite spacing, wave-vector
angle need not equal the outgoing ray angle; this is not an experimental
angular distribution until that map is specified.

In the continuum, the radial denominator is
`m omega/(omega' E_f)`. With `|M|^2=4mE_f |T|^2`, the formula reduces to
`|M|^2 (omega'/omega)^2/(64 pi^2 m^2)`, the reference laboratory cross
section. `cross_section` computes the finite-model rate with these flux and
density factors; its convergence to the QED rate is a separate consumer check.
No claim is made that the finite lattice is nature or that canonical photon
normalization has been explained rather than imported.

## 5. Failure, partial repair, and a materially different repair

Expand the charged energy around rest:

\[
E_a(p)=m+\frac12\left(\frac{v^2}{m}+ra\right)|p|^2+O(|p|^4).
\]

The soft Compton response samples the inverse **inertial** mass,
`1/m_kin=v^2/m+ra`, not the rest gap `m` by itself. In reference-normalized
units, the Thomson squared-amplitude ratio is

\[
\frac{\overline{|M_a|^2}_{\rm soft}}
 {2e^4(1+\cos^2\theta)}=(v^2+ram)^2.
\]

Thus a gauge-compatible Wilson band with `v=1` fails this relation at finite
`a`. The failure is not a failure of the Ward identity. Tuning
`v^2=1-ram` repairs the soft mass relation at that spacing, but does not force
the finite-energy Compton shape or common limiting photon/matter speed.
The code explicitly distinguishes this single-observable repair from the
full interaction target.

A second repair changes the gauge-covariant operator, not merely a number:

\[
M_a^{(2)}(p)=m+\frac r a\sum_i(1-\cos ap_i)^2.
\]

This uses nearest and straight two-link paths since
`(1-cos x)^2=2(1-cos x)-(1-cos 2x)/2`. The quadratic rest curvature vanishes,
while corner gaps remain heavy: a corner with `n` components at `pi/a` has
`M=m+4rn/a` rather than the nearest Wilson value `m+2rn/a`.
Naive `r=0` also removes the soft curvature but leaves eight light spatial
Dirac cones. That is an unacceptable unmentioned species change, not a
successful electron derivation.

The two-link path sums the two link fields before exponentiation. Its
one-photon factor is `2 cos(ak_i/2)` and its two-photon factor is the product
of the incoming/outgoing path factors. Accordingly the beta part of `V/q`
inside the spatial minus sign becomes

\[
r\,[2\sin b_i-\cos(a(p'_i-p_i)/2)\sin(2b_i)],
\]

and the beta part of `W/q^2a` becomes

\[
r\,[2\cos b_i-2\cos(ak_i/2)\cos(ak'_i/2)\cos(2b_i)].
\]

These are **not** obtained by substituting the new mass into the old vertex.
They preserve the discrete Ward identity and its two-photon cancellation.
For `v=1` the soft mass relation is repaired without speed tuning. The
remaining dispersion errors start at `O(a^2)` in the alpha/photon bands and
`O(a^3 p^4)` in this mass term. Whether this improves the actual finite-energy
observable is checked by the matrix calculation, not inferred from those
orders alone. Numerical results and convergence are recorded in `results.json`
and the campaign README after execution.

## 6. What this earns, and what remains open

This supplies one explicit **coupled** conditional interface:
charged band -> same-link current -> mandatory two-photon contact -> tree
scattering, joined to a two-transverse-mode photon band. It derives which
parts have to change together and exposes a repair discriminated by the
physical target. It does not explain CAR, the quantum postulates, why this
Clifford band exists, the observed electron mass/charge, a microscopic origin
of gauge constraints, radiative corrections, infrared dressing or a compact
lattice Coulomb phase. Nor does a rest-frame orientation probe prove Lorentz
invariance; that survives only as the controlled matched-speed continuum
limit of the declared bands and kernel.

The next backward relationships are those imported structures, not more
polishing of a Thomson fit. `constraint.md` investigates why a graded
constraint can bind propagation and magnetic response but does not generate
exchange statistics. `quantum_supplier.md` explicitly checks Gauss-compatible
quantum hopping and the distinct CAR input. A deeper supplier must replace
an input with a demonstrated map without breaking this joined consumer.

## Source provenance

The construction and trigonometric identities above are derived directly in
this record. [Baaquie, *Lattice gauge theory: Hamiltonian, Wilson fermions, and
action* (1986)](https://doi.org/10.1103/PhysRevD.33.2367) was read at the
abstract level only; it establishes historical context, **not** any detailed
formula used here. Its full text was access-restricted. Our continuous-time
Hamiltonian is explicitly specified rather than asserted to be that paper's
transfer-matrix Hamiltonian. The reference QED formulas and source access are
recorded separately in `reference.md`.
