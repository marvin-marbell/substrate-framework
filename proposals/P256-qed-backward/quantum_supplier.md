# A quantum supplier selected backward from Compton requirements

**Work item:** `quantum-supplier`, governing plan `issue-222-qed-backward-20260930`, initial pinned revision `17c444b342123ed2d3cc243358ec5e687ef78b1337a7eaf41238273966535b6d`, refreshed joining revision `5062d77da98952dff6fe9da61b2ed705195a1b67fb947ab099b83e2ee395998b`, template `general` revision `89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`.

**Claim level:** conditional construction, not an electron mechanism or a QED match. The actual quantum many-body calculation is implemented in `quantum_supplier.py`; its numerical execution is reserved for the parent's integrated smoke. Exact algebraic predictions below are derivations, not fabricated run output. The code prints matrices' spectra, dynamical probabilities, exchange products, cubic photon modes and source-response residuals as JSON and exits nonzero if its discriminating checks fail. No closure, promotion or merge is implied.

## 1. Observable first, supplier second

The parent reference is leading-order spin/polarization-resolved Compton scattering, with its Thomson limit, in QED with metric (+---), hbar=c=1 and electron q=-e. Its backward map demands, among other coupled structures:

1. Two **quantum** physical transverse external photon polarizations, with the correct energy and external-state normalization, not three unconstrained vector oscillators or classical waves.
2. The **same charge** in the constraint's source, conserved matter current and matter's gauge-covariant propagation/vertex. Removing a longitudinal state must not remove its electrostatic source response.
3. A fermionic many-body charged sector: exchanging two electron excitations changes amplitude sign. A charge label and transverse wave do not supply that sign.
4. The reference's massive spin-1/2 propagator, both Compton time orderings, their vertex compatibility and the resulting finite-energy amplitude. This slice does not yet supply these endpoints.

A constrained link network is therefore attempted as a supplier of items 1–3's **coupled gauge/current structure**. It was not selected because a rotor already resembles an electron. Quantum states, CAR, an integer electric-flux algebra, hopping coefficients and local constraints are explicit inputs. This is an attempted explanatory interface, not a derivation of those postulates from a deeper root.

A relevant primary lead is [Hermele–Fisher–Balents, cond-mat/0305401v3](https://arxiv.org/html/cond-mat/0305401). Sections II.3 and III.1 construct rotor/gauge descriptions and a Coulomb-phase Gaussian effective theory; IV.1 explicitly says the discussed electric charges and monopoles are **bosonic**, with possible fermionic dyons conditional on binding. That last distinction is essential: their result suggests the gauge supplier but does not transfer an electron, derive our imported CAR, or establish our chosen compact model's phase. The equations below are derived for the stated Hamiltonian, not claimed as numerical reproduction of that spin model.

## 2. Local quantum algebra and one coupled source/current

Orient every link l from x to y. Its exact infinite Hilbert space is l2(Z), with

\[
E_l|r\rangle=r|r\rangle,\qquad U_l|r\rangle=|r+1\rangle,
\qquad [E_l,U_{l'}]=\delta_{ll'}U_l,\quad U_l^\dagger U_l=1.
\]

One may write U=exp(i theta), where theta is compact. The unitary shift algebra is exact; a global self-adjoint angle obeying a naive [theta,E]=i identity is not needed. Canonical real theta and E are used **locally in the noncompact harmonic approximation**, not asserted globally on a compact rotor.

At sites, import `{c_x,c_y^dagger}=delta_xy`, `{c_x,c_y}=0` and n_x=c_x^dagger c_x. They commute with link operators. A static compensating background r_x fixes the neutral sector. Define

\[
G_x=({\rm div}E)_x-n_x+r_x,\qquad G_x|\Psi_{\rm phys}\rangle=0,
\]

\[
T_l=c_y^\dagger U_l^\dagger c_x,\qquad
H={U_E\over2}\sum_l E_l^2+K\sum_p(1-\cos\Theta_p)
-t\sum_l(T_l+T_l^\dagger),\quad
\Theta_p=\sum_{l\in\partial p}s_{pl}\theta_l.
\]

Here U_E denotes the **energy coefficient**, distinct from the shift operator U_l. In code it is `electric_stiffness`. For an x->y hop, n_x decreases, n_y increases and E_l decreases. At both endpoints the constraint remains unchanged: `[G_x,T_l]=[G_y,T_l]=0`. Bare `c_y^dagger c_x` instead changes G_x by +1 and G_y by -1. The plaquette shift has zero divergence and also commutes with all G. Thus the gauge-covariant hop is selected by the target source/current condition, not appended as an arbitrary force.

Under exp(i sum alpha_x G_x), c_x maps to exp(i alpha_x)c_x and U_xy maps to exp(i(alpha_x-alpha_y))U_xy. All terms above are invariant. With

\[
J_l=it(T_l-T_l^\dagger),\qquad
\dot n_x=-({\rm div}J)_x,\qquad
\dot E_l=-J_l+\text{divergence-free plaquette contribution},
\]

we obtain `div E_dot = n_dot`. Therefore `G_dot=0` and the hopping current is derived from the **same operator and t** as matter propagation. Multiplying the number density/current by q=-e yields the physical charge density/current; no unrelated charge constant is attached to another force. The star calculation explicitly evaluates the commutators and continuity equation as sparse matrices.

The bosonic substitution uses commuting-site hard-core operators b_x, with identical occupations, link shifts and T_l=b_y^dagger U_l^dagger b_x. The same Gauss/current proof holds. The sign of exchanging excitations does not.

## 3. A finite quantum calculation without fake rotor unitarity

### Exact selected physical sector

Use a four-site star with oriented edges 0->1, 0->2, 0->3, two particles, background `(0,1,1,0)` and no external flux. There are no plaquettes. The leaf constraints determine every flux uniquely:

\[
E_{0j}=r_j-n_j\quad(j=1,2,3),\qquad
\sum_j E_{0j}=n_0-r_0.
\]

The center equation follows from the fixed total number sum n=sum r=2. Hence the physical sector has exactly choose(4,2)=6 states. Every flux is -1, 0 or +1, and all allowed forward/reverse matter hops remain inside these six states. This is an exact invariant subspace of the **infinite-rotor hopping plus electric-energy Hamiltonian on this tree**. It is not a cutoff approximation to its physical dynamics. Adding plaquettes, relaxing the particle sector, or changing background invalidates this particular finite proof.

For a genuine forbidden-operation control the code also constructs an auxiliary 750-dimensional space: six occupations times five flux values per link, E=-2,...,+2. A raising shift at E=+2 is zero, not wrapped to -2. Thus its `[E,U]=U` is exact but

\[
U^\dagger U=1-|+2\rangle\langle+2|,\qquad
UU^\dagger=1-|-2\rangle\langle-2|.
\]

Each unitarity defect has norm 1. These defects are **disclosed**, not treated as exact rotor quantization. The complete physical sector and every physical outgoing hop stay away from these boundaries; code counts clipped physical transitions and evaluates `(1-P)H_hop P`. Bare hopping stays at fixed E=0, so its forbidden evolution also does not touch a shift boundary. The unrestricted auxiliary spectrum is never advertised as the compact theory's spectrum.

### Discriminating controls and dynamics

Initial state: particles on leaves 1 and 2, all E=0. The implementation diagonalizes the six-state physical Hamiltonian with t=U_E=1 and evolves to time 0.7. It reports all energies and probabilities, a nonzero departure probability, zero physical leakage and the sparse full-space Gauss commutator. This is a many-body quantum amplitude calculation, not classical occupation updates.

For a bare-hop control, evolve the six occupations with all E held at zero, using the **unprojected** bare hopping matrix. Report `<sum_x G_x^2>`; this must be nonzero. Projecting the illegal hop away and announcing conservation would be a false positive. At small time tau, each of the two initially allowed leaf->center hops produces `sum G^2=2`, so the forbidden expectation begins at `4 t^2 tau^2`. Legal hopping instead correlates each matter move with a link-flux change.

A six-move legal exchange is

\[
1\to0\to3,\qquad 2\to0\to1,\qquad 3\to0\to2.
\]

It returns both occupations and all three fluxes to the initial state, but the two original particles have exchanged. The ordered product of the actual transition matrix elements (without the common Hamiltonian -t factors) is **-1 for CAR fermions and +1 for commuting-site hard-core bosons**. These are analytically predicted signs; the numerical code evaluates rather than hardcodes them. The site ordering is 0,1,2,3 and the CAR sign is `(-1)^(occupied lower sites)` at each creation/annihilation. Rephasing basis states cancels on the closed path. The extra empty branch is necessary: a one-dimensional hard-core chain would block the exchange and would not be an informative statistics control.

The full 16-state matter Fock operators also check CAR directly. **Verifying an imported representation is not deriving CAR.** The star has no propagating photons and no continuum spin. Its role is the local quantum charge-current/constraint and statistics discriminator; the cubic calculation below is a distinct conditional sector of the same link Hamiltonian, not a single numerically simulated combined lattice scattering experiment.

## 4. Three-dimensional transverse quantum modes and finite-scale mismatch

At a small-flux background, expand the compact plaquette energy to quadratic order:

\[
H_\gamma^{(2)}={U_E\over2}\sum E^2+{K\over2}\sum\Theta_p^2.
\]

This discards compact large-flux events and integer-flux effects in the low-energy description; it is not proof of a Coulomb phase. In charge-free nonzero momentum sectors, use midpoint link Fourier components. Put

\[
d_i(\mathbf k)=2\sin(k_i a/2),\quad
M_{ij}=d^2\delta_{ij}-d_i d_j,
\quad H_\gamma(\mathbf k)={U_E\over2}|E|^2+{K\over2}\theta^\dagger M\theta.
\]

The midpoint phases cancel in curl-dagger-curl. For d!=0, M d=0 and M restricted to d-perpendicular is d^2 times the identity. The Gauss equation d dot E=0 and gauge quotient remove the longitudinal canonical pair; they do not make it a third physical zero-energy photon. Exactly **two** physical oscillator polarizations remain. Their dispersion is

\[
\omega^2=U_E K\,4\sum_i\sin^2(k_i a/2),\qquad
c_\gamma=a\sqrt{U_EK}.
\]

With canonical oscillator quantization, each transverse mode has `H=omega(a^dagger a+1/2)` and ground variance `<|theta_lambda|^2>=U_E/(2 omega)` in orthonormal lattice Fourier conventions. Quantum state structure is imported; this is nevertheless a quantized harmonic mode, not just a classical frequency. Circular combinations of the two real polarization vectors become helicity ±1 in the rotational continuum limit. At finite a, lattice rotations and transversality to d replace continuous rotations and transversality to k; exact Lorentz helicity is not claimed. At k=0 there are global zero modes/topological sectors, not the rank-two nonzero-momentum argument.

The continuum expansion is

\[
\omega^2=c_\gamma^2\left[k^2-{a^2\over12}\sum_i k_i^4+O(a^4k^6)\right].
\]

At equal physical |k|=s, the axis and body-diagonal predictions are

\[
\omega_{\rm axis}=2\sqrt{U_EK}\sin(as/2),\qquad
\omega_{\rm diagonal}=2\sqrt{3U_EK}\sin(as/(2\sqrt3)).
\]

Their fractional continuum errors begin at `-a^2 s^2/24` and `-a^2 s^2/72`. This direction-dependent error cannot be absorbed into one photon-speed normalization. The code diagonalizes M for each of |k|=0.05,0.5,1.5 (a=1), on axis, diagonal and direction `(1,2,3)/sqrt(14)`, reporting its full spectrum, transverse vectors, frequencies, variances and continuum ratios. The high-momentum axis/diagonal difference is an adverse control. These default harmonic values do not establish that the compact U_E=K=1 model is in a deconfined phase.

### Photon normalization and charge transfer are linked

Take spatial A_i to be **covariant** components (A_i=-A_physical-vector,i), so E_physical,i=dot A_i-partial_i A_0 and D_i=partial_i+iq A_i. A consistent continuum map is

\[
\theta_i=q a A_i,\qquad E_{\rm rotor,i}={a^2\over q}\Pi_i,
\qquad n_x-r_x=a^3\rho_{\rm number},\qquad q=-e.
\]

The symplectic term sum E_rotor dot theta becomes integral Pi dot A. The Hamiltonian becomes

\[
H_\gamma\to\int d^3x\left[{U_Ea\over2q^2}\Pi^2+{Kq^2a\over2}B^2\right].
\]

The Gauss constraint becomes `div Pi=q rho_number`, and matter hopping carries exp(-i q a A_i). Thus one q enters both source and coupling; its magnitude is not independently adjustable after photon normalization. For canonical Maxwell coefficients and c=1,

\[
U_E=q^2/a,\qquad K=1/(q^2a).
\]

These are **transfer requirements**, not derived microscopic parameters. With N sites and V=N a^3, continuum-normalized Fourier A is `sqrt(a) theta_lattice/q`; therefore its ground variance is `a U_E/(2 q^2 omega)=1/(2 omega)` only under the stated coefficient matching. Mere frequency agreement without this residue would not normalize the Compton photon legs or the charge correctly. Continuum Maxwell Lorentz covariance holds in this free quadratic limit after c=1; no matter/photon common light cone follows from it.

## 5. Source response survives the gauge quotient

For neutral static number source rho on a periodic lattice, let D be the oriented incidence/divergence matrix and L=D D^T the positive lattice Laplacian. Minimizing electric energy subject to D E=rho gives

\[
\phi=L^+\rho,\qquad E_L=D^T\phi,\qquad
V[\rho]={U_E\over2}\rho^T L^+\rho.
\]

The zero Fourier eigenvalue is removed only because sum rho=0. Nonneutral periodic data would be inconsistent without external flux or compensation, not repaired by silently discarding their charge. L(k)=4 sum sin^2(k_i a/2), exactly the same d^2 as the transverse photon dispersion.

The code uses an 8^3 periodic cube with +1 at `(0,0,0)` and -1 at `(4,0,0)`. An FFT applies the actual inverse Laplacian, reconstructs `E_i(x)=phi(x)-phi(x+e_i)`, evaluates `div E-rho`, and compares electric-field energy to `U_E rho dot phi/2`. This is a harmonic **real-flux minimizer**, not an exact eigenstate of compact integer E and not a determination of nonperturbative confinement.

In the infinite-volume long-distance limit, `L^+(R)~1/(4 pi |R|)` for dimensionless separation R=x/a. The pair cross energy is `U_E a rho_1 rho_2/(4 pi |x|)=q^2 rho_1 rho_2/(4 pi |x|)` under Maxwell matching. Self energies are cutoff dependent. This ties the Coulomb coefficient to the same photon/matter normalization rather than fitting it separately.

## 6. Concrete bosonic false positive and the charged-sector mismatch

The hard-core bosonic substitution passes the same gauge-constraint, continuity, photon rank-two and harmonic Coulomb constructions but has exchange +1. A spin label was never included in either matter representation. In the dilute single-particle sector on the cubic extension of this hopping law,

\[
\varepsilon(\mathbf k)=\mu-2t\sum_i\cos(k_i a)
=(\mu-6t)+t a^2 k^2-{t a^4\over12}\sum_i k_i^4+\cdots,
\quad m_{\rm kin}={1\over2t a^2}.
\]

This dispersion and its link-generated response are identical for single fermions and single hard-core bosons. In a slowly varying field the leading matter Hamiltonian is `(p-q A_physical)^2/(2 m_kin)`, including its `q^2 A^2/(2 m_kin)` seagull and current term. It therefore has a low-energy **Thomson-like charge/mass response** regardless of the exchange sign. This is an analytic dilute/continuum comparison; the star is not a Thomson scattering calculation and no bosonic Compton cross section has been numerically substituted for the parent result.

The mismatch is structural, not a tiny lattice correction:

- The implemented matter is spinless, nonrelativistic and number conserving. It has no four-component Dirac Clifford structure, negative-energy/antiparticle sector, intrinsic magnetic moment or spin-resolved Compton vertices.
- Its independently adjustable band bottom mu-6t is not linked to its kinetic mass. Setting the bottom equal to m and the curvature equal to 1/(2m) is tuning two parameters, not deriving relativistic dispersion.
- Its k^4 coefficient at a fixed direction is `-t a^4 sum k_i^4/12`; a massive relativistic branch has isotropic `-k^4/(8m^3)`. Matching k^2 and photon speed is insufficient.
- Photon and matter can have different limiting speeds; finite lattice anisotropy and the preferred frame remain visible.
- The Coulomb-background star states are quantum flux/occupation states, not the infrared-dressed scattering states of an interacting electron.

Thus gauge structure plus a two-polarization photon plus an apparent Thomson coefficient would be a **false positive** for an electron explanation. The closed exchange path rejects that identification even before the missing spin/Dirac comparison. CAR passes the exchange discriminator only because it was imported.

## 7. Target-driven repair and joining the constraint slice

The material repair of bare hopping is already constructed: replace it by the flux-changing covariant T rather than project away the violation. The next charged-dispersion mismatch selects a different repair: a spinor-valued link hopping with a mass term, not relabeling the scalar band's mu as electron mass. A concrete conditional candidate is a Wilson-type matrix band

\[
h(\mathbf k)=\sum_i\alpha_i{\sin(k_i a)\over a}
+\beta\left[m_0+{r_W\over a}\sum_i(1-\cos(k_i a))\right],
\qquad \{\alpha_i,\alpha_j\}=2\delta_{ij},\quad
\{\alpha_i,\beta\}=0,\quad\beta^2=1.
\]

Its links must carry the **same** U_l, with a matrix-valued hopping replacing scalar t; Gauss charge becomes the sum over components. This is the joining contract with `constraint-interface`, whose Clifford/propagation/vertex construction is complementary. It is not implemented in `quantum_supplier.py` or claimed as an emergent endpoint here. The parent has now implemented the conditional joined band and Compton kernel in [`lattice_compton.py`](lattice_compton.py), with its derivation in [`lattice_compton.md`](lattice_compton.md). Importing alpha/beta and CAR is still conditional input; deriving them needs another supplier. Wilson's extra term changes the vertex and contact terms, so dropping it from the interaction while retaining it in propagation fails the very compatibility sought. A naive sin-only replacement instead supplies extra light lattice species and does not select one electron.
 
The parent's target-driven correction is stronger than assembling independent likenesses: its same-link Peierls expansion supplies one-photon V **and mandatory two-photon W**, then evaluates both pole orderings and W against Compton. The nearest-link Wilson band's inverse inertial mass differs from its rest gap: the derived soft squared-amplitude ratio is `(v^2+r a m)^2`. Speed tuning repairs that soft coefficient alone. A material repair replaces the Wilson mass contribution by `(r/a) sum_i(1-cos(a p_i))^2`; nearest and straight two-link paths are gauged together, so V and W change together while the quadratic mass curvature vanishes and corner species remain heavy. These formulas and implementation are read from the parent's derivation; numerical success is reserved for integrated execution. They do not derive CAR, Clifford matrices or the compact Coulomb phase from our star sector.

The vector-potential convention also must be transferred rather than guessed: our covariant spatial A_i is minus the physical three-vector used in the parent's `p-qA` band convention. Its Peierls unitary exp(-i q a A_physical) equals our U_l; writing a hop in the opposite orientation uses its adjoint. The invariant object is the oriented covariant hopping and its derived current, not the literal name U.

The following records this slice's local result and the requirements for its **conditional joined** Compton transfer. The parent's constructed tree kernel is a consumer calculation, not proof that microscopic rotor matter has become its spinor endpoint:

| Required interface | Present result | Remaining discriminant/repair |
|---|---|---|
| Physical photon states | Two quantized harmonic transverse modes at k!=0 | Continuum residue and helicity map with controlled a|k| error; compact-phase stability |
| Source/current coupling | Exact local Gauss-preserving T and derived continuity; parent constructs same-link spinor V/W | Transfer charge/residue consistently; compact-to-effective matching and integrated Ward controls |
| Quantum charged statistics | CAR sign verified; bosonic exchange control differs | Origin of CAR or a genuinely derived composite exchange mechanism |
| Electron propagation/spin | Scalar band fails Dirac requirement; parent explicitly imports a spinor band | Derive rather than import the charged Clifford/spin sector; preserve species control and common light cone |
| Mass and charge | U_E,K,t,mu imported; q fixed only after Maxwell normalization | Renormalized pole/threshold mass and charge, parameter fixing, independent tests |
| Scattering | No star-graph scattering; parent implements joined lattice tree Compton | Integrated pole/contact/kinematics comparison; microscopic transfer and dressing/IR convention beyond tree level |

Lattice scales a, 1/a and U_E,K must separate from m and external omega/momentum; target energies must obey a|k|,a m <<1 with a controlled error estimate. Matching c_gamma=1 does not prove this hierarchy or set an experimental upper bound on a. Compact monopoles, flux sectors, confinement, matter screening and interactions can invalidate the harmonic endpoint; a small-flux expansion alone does not choose the nonperturbative phase. Even a deconfined phase does not imply a one-electron species or the measured e and m. The parent now supplies an explicit conditional charged-state/dispersion/spin map for its joined scattering comparison; the still-missing implication is that this rotor construction supplies that map, not merely a generic lattice photon resemblance.

**Inputs/debt:** Hilbert-space quantum postulates, rotor shift algebra, CAR or bosonic site algebra, static background, coefficients and cubic geometry are imported. Local Gauss invariance, continuity, finite exact star sector, exchange signs, harmonic rank/dispersion, finite-scale anisotropy and inverse-Laplacian response are derived at the stated scopes. Compact phase, emergence of CAR/spin/Dirac structure, mass origin, radiative/infrared physics and microscopic transfer into the parent's conditional Compton consumer are unresolved. Literature is a mechanism lead, not proof that any of those endpoints transfer.

## Reproducible calculation contract

From the worktree root:

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/quantum_supplier.py
```

The parent may import `report()`, `many_body_report(fermionic)`, `photon_modes(momentum, a, electric_stiffness, magnetic_stiffness)` and `coulomb_report()`. Output is actual deterministic finite matrix/FFT calculations, not mocks; `report()['checks']` includes adverse bare-hop, exchange-statistics and finite-lattice-anisotropy controls. Default a,U_E,K,t=1 are calculation units, **not** derived electron mass/charge values or a claim about the compact model's phase. Numerical success proves these scoped algebra/mode calculations, not the unresolved target interfaces above.
