# Neutral heavy sea, zero-origin orbital selection, and the same light photon consumer

This construction changes the material supplier of the orbital shape: a neutral
heavy occupied sea, rather than a prescribed nonzero orbital norm or a large light
hopping coefficient. `heavy_orbital_join.py` joins that supplier to the existing
`ContinuousSea` → `axis_scattering` → primitive-current leading-soft inclusive
consumer. It computes a **uniform two-species saddle**, not a continuous SU(4)
Gauss-projected ground state or a quantum three-dimensional Higgs phase. The
hierarchy below is a finite-volume low-energy expansion with explicit hypotheses.
No numerical result is claimed here before the CLI is run and its receipt reviewed.

## 1. Microscopic law and the changed supplier

At each site let four initially unassigned, spatially scalar orbital matrices be

\[
 A_\mu(x)=\sum_{a=1}^{15}y_{\mu a}(x)B_a,\qquad
 \operatorname{tr}(B_aB_b)=4\delta_{ab},\quad y_{\mu a}\in\mathbb R.
\]

These are sixty unconstrained real adjoint fields, with
`matrix_unit_basis()` supplying all traceless Hermitian directions. Their Gram
matrix is \(G_{\mu\nu}=\operatorname{tr}(A_\mu A_\nu)/4\). Neither
orthogonality nor an anticommutator constraint belongs to the microscopic law.
Use the same local SU(4) law as `orbital_gauge_locking.py`:

\[
 A_\mu(x)\mapsto V_xA_\mu(x)V_x^\dagger,\qquad
 U^{(4)}_{xy}\mapsto V_xU^{(4)}_{xy}V_y^\dagger,
\]
\[
 h^{(s)}_{xy}=u_{xy}^{q_s}\sum_\mu c^{(s)}_{xy,\mu}
       \frac{A_\mu(x)U^{(4)}_{xy}+U^{(4)}_{xy}A_\mu(y)}2,
\quad
 E_{\rm grad}=\frac J8\sum_{xy,\mu}
 \|A_\mu(x)-U^{(4)}_{xy}A_\mu(y)U^{(4)\dagger}_{xy}\|_F^2.
\]

For improved long links, the SU(4) and charged U(1) products follow the actual
primitive path, including repeated elementary links. The scalar coefficients and
onsite law are precisely `RotationBand.couplings`, with the orbital matrices
reframed by `h_from_couplings`. The expression is locally covariant for arbitrary
Hermitian shapes, not just on the selected orbit.

There are two four-flavour primitive matter species, both SU(4) fundamentals,
both microscopically spatial scalars, and both half filled. The light species has
U(1) charge −1 and base \(ma_L=.4\) or \(.2\), \(r_L=v_L=1\). The heavy
species has charge **zero**, base \(ma_H=.8\), \(r_H=v_H=1\), and its **entire**
onsite and hopping law is multiplied by \(\Lambda>0\). Both use the improved
mass law in lattice units \(a=1\). The CLI uses `e=0` for the heavy base band;
it never passes that band to the photon-response constructor.

In a **fixed local occupation** \(n_H=n_L=2\) sector, each species transforms
as \(\wedge^2 4\), so the formal centre charges sum to \(2+2=0\pmod4\).
That bookkeeping does not establish the local condition in the actual hopping
sea. Half filling fixes global numbers and mean local numbers, not each cell's
occupation. For each species the local Gaussian density matrix is
\(C_s=\tfrac12(I-b_s\Gamma_0)\), where
\(b_s=\langle d_{s0}/|d_s|\rangle\). Its number variance is
\(1-b_s^2>0\) for the rich grid and nonzero vector hopping. The receipt
computes the actual local occupation probabilities and convolves the two
species, explicitly exposing non-neutral local centre sectors.

Consequently no external centre-two reference is needed for **fixed**
two-plus-two representation bookkeeping or global total-number neutrality;
the product sea itself is not locally Z4-projected. Electric flux/Higgs
dressing and the continuous SU(4) Gauss constraint still have to be supplied.
Even a centre-neutral local representation need not be a singlet:
\(6\otimes6=1\oplus15\oplus20'\).

The bare orbital potential is

\[
 V_{\rm bare}=\frac{\Lambda\kappa}{2}\|G\|_F^2,\qquad \kappa>0.
\]

Because \(G=yy^T\), its unique bare minimum is all sixty fields zero. There is
no preferred nonzero Gram, no bare Clifford penalty, and no fiat stiffening of
the light band. In particular `select_gram(..., preferred=0)` is essential.

## 2. Exact unrestricted uniform saddle

Use one common cubic BZ set with strictly positive normalized weights \(w_p\).
For \(s=H,L\), let \(x_s(p)\) be the four scalar coefficient functions of the
actual base local band, and \(R_s(p;G)=\sqrt{x_s(p)^TGx_s(p)}\). The occupied
energy bound gives the uniform functional

\[
 F(G)=-2\Lambda\sum_pw_pR_H(p;G)-2\sum_pw_pR_L(p;G)
                    +\frac{\Lambda\kappa}{2}\|G\|_F^2. \tag{1}
\]

The implementation concatenates \(x_H,x_L\) and uses weights
\(\Lambda w_p/(\Lambda+1),w_p/(\Lambda+1)\), stiffness
\(\kappa\Lambda/(\Lambda+1)\), zero preference, then multiplies the objective
by \(\Lambda+1\). Midpoint quadrature's implicit equal weights are made
explicit for this concatenation. Split-Gauss weights are retained, not replaced
by a uniform average. Each species sees the same normalized BZ measure.

Here is why the reduction does not assume the conclusion:

1. For every traceless Hermitian four-level block, the sum of its two lowest
   eigenvalues is at least \(-\sqrt{\operatorname{tr}h^2}=-2R\), with equality
   exactly at \((-R,-R,R,R)\). The total-number half-filled ground is also
   bounded this way: the unconstrained sum of negative levels is no smaller
   than \(-\sqrt{\operatorname{tr}h^2}\), and the witness already has exactly
   two occupied states per gapped momentum. See `sea_selection.md` for the
   four-level proof; no CAR matrix representation is used to establish this
   spectral inequality.
2. For any positive-semidefinite \(G\), a matrix-unit Clifford witness
   \(A_\mu=\sum_\nu (\sqrt G)_{\mu\nu}\,\mathrm{SELECTED}_\nu\)
   saturates the bound. It is an admissible competitor, not a restriction on
   alternative fields. Thus minimizing (1) gives the true minimum over all
   sixty uniform fields at the stated quadratic-matter saddle.
3. Negative square roots of affine forms in \(G\) are convex, and the quartic
   field potential is strictly \(\Lambda\kappa\)-convex **in Gram coordinates**.
   Hence (1) has a unique Gram minimizer. Cubic signed permutations leave the
   BZ measure and both dispersions invariant. Uniqueness forces
   \(G=\operatorname{diag}(g_m,g_s,g_s,g_s)\). This is a derived cubic metric,
   not an imposed ansatz for the original fields. The nonzero mass functions
   and rich spatial inputs force both entries positive by the directional
   derivatives at each boundary. At the all-zero fields a small common scale
   lowers the sea energy linearly while the bare cost is quartic.
4. Every positive-weight block must saturate. The matrix quadratic equations
   \(h_s(p)^2=(x_s^TGx_s)I\), and a rank-ten design of the products
   \(x_{s\mu}x_{s\nu}\), give
   \(\{A_\mu,A_\nu\}=2G_{\mu\nu}I\). The CLI reports the rank of its actual
   concatenated set. This supplies the selected algebra; a rank-deficient
   grid would not supply it.

The selected light law is therefore, on the very same local path,

\[
 m_L^{\rm sel}=ma_L\sqrt{g_m},\qquad r_L^{\rm sel}=\sqrt{g_m},
 \qquad v_L^{\rm sel}=\sqrt{g_s}. \tag{2}
\]

The heavy one-particle gap is \(\Lambda ma_H\sqrt{g_m}\). This is the uniform
band gap, **not** a dressed gauge-invariant excitation gap. The CLI independently
sums the actual occupied matrix eigenvalues of both reframed local laws, checks
that they attain (1), and checks the selected light matrix mapping.

### Finite heavy scale is not the heavy-only saddle

Write \(F/\Lambda=f_H+\Lambda^{-1}f_L\), where \(f_H\) includes the bare
quartic. Let \(G_H\) minimize \(f_H\). Strong convexity and monotonicity of the
light convex gradient imply the explicit bound

\[
 \|G_\Lambda-G_H\|_F
 \leq\frac{\|\nabla f_L(G_H)\|_F}{\Lambda\kappa}. \tag{3}
\]

In the radial coordinates \((g_m,g_s)\), the spatial gradient is the sum over
three equal entries; its Frobenius norm is
\(\sqrt{(\partial_m f_L)^2+(\partial_s f_L)^2/3}\). For an interior saddle,

\[
 (g_m,g_s)_\Lambda=(g_m,g_s)_H
 -\Lambda^{-1}[D^2f_H]^{-1}\nabla f_L+O(\Lambda^{-2}). \tag{4}
\]

The receipt reports the actual nonzero finite-scale feedback, the bound (3),
the first-order prediction/error, and \(\Lambda\|G_\Lambda-G_H\|\), separately
for each quadrature order. It does not silently replace the selected metric by
its heavy-only limit. Refinement tables report adjacent metric and actual
hard/inclusive rate changes; two orders are numerical refinement observations,
not an all-orders quadrature proof.

## 3. Return to the actual light electromagnetic consumer

Only (2) is supplied to `ContinuousSea`. Heavy electromagnetic response is
identically zero because its microscopic charge is zero; it is not omitted on
the grounds that a large gap makes it small. The charged response uses the same
improved primitive contact, vertex and BZ measure as the selector.

Use the exact analytic transfer-zero response returned by `sea.infrared()`:

\[
 \chi_B=1/.09,\qquad
 \chi_E=\frac{\chi_B+\Delta\chi_{B,\mathrm{IR}}}{(v_L^{\rm sel})^2}
                       -\Delta\chi_{E,\mathrm{IR}}, \tag{5}
\]
\[
 c_{\rm IR}^2=\frac{\chi_B+\Delta\chi_{B,\mathrm{IR}}}
                   {\chi_E+\Delta\chi_{E,\mathrm{IR}}}
              =(v_L^{\rm sel})^2.
\]

This is an **explicitly tuned relativistic critical surface**. It is not a
calculation of dynamical velocity locking. A new `ContinuousSea` is constructed
after matching so that no pole cache mixes models. Positivity of \(\chi_E\)
is required, not repaired by a fallback.

Choose physical mass one only as a unit conversion: \(a=m_L^{\rm sel}\).
The actual consumer call is
`axis_scattering(sea, q=a*.4, a=a)` by default. Its right-angle incoming and
outgoing photons have the actual same-sea poles, residues, group flux and
radial recoil Jacobian. Its `FilledSeaConsumer` evaluates the connected
primitive Pauli paths, pair channels and contact; an independent band-resolvent
comparison is a check, not a separate supplied spinor endpoint. The negative
controls omitting pairs/contact/leg dressing/dispersion remain in that consumer.
No fundamental spin or CAR representation is introduced as the final consumer.

`current_velocity` then evaluates the **same selected** primitive light current
at the actual recoil. `angular_coefficient`, its independent integrated rest
comparison, and `ir_resolution` supply the leading-soft inclusive factor. The
reported rate is the actual primitive hard rate times the per-photon resolution
veto factor, with upper soft energy .01 and each-photon resolution .001 in
physical mass-one units. This is not a total-unresolved-energy cut, a full loop
Compton calculation, or an all-order finite-energy inclusive cross section.
The matter-speed-unit Klein–Nishina comparator uses the actual geometric leg
source and the time/charge conversion described in `elastic_join.py`; the
receipt's deviation is an observed comparator, not imposed recoil kinematics.

## 4. Exact Higgs quadratic form and controlled hierarchy

For generators \(T_a=B_a/(2\sqrt2)\), so
\(\operatorname{tr}T_aT_b=\delta_{ab}/2\), the existing `gauge_mass` helper gives

\[
 (\mathcal M_4^2)_{ab}=\frac{Jg_4^2}{4}
 \sum_\mu\operatorname{tr}\bigl([T_a,A_\mu]^\dagger[T_b,A_\mu]\bigr).
\]

Its fifteen eigenvalues are strictly positive at a positive full-rank selected
frame: a generator commuting with all four selected matrices is scalar, and a
traceless scalar is zero. For Clifford blades the exact eigenvalues are
\(Jg_4^2\sum_{\mu\in I_{\rm anti}}g_\mu/2\), with multiplicities 4, 6, 4, 1
by blade degree; the code verifies the analytic spectrum and adjoint covariance.
It also invokes the existing raw two-word Pauli finite-fiber covariance helper
on the light band and an arbitrary noncentral external SU(4) link. That helper
is explicitly **not** a continuous SU(4) Gauss simulation. The same algebraic
link covariance for the heavy species has its U(1) factor set to one.

There are fifteen tangent conjugation directions and forty-five uniform shape
directions. One can see strict restoring curvature without equating it to this
gauge matrix: at the saddle the second variation is the positive Gram Hessian
on \(\delta G\), plus the nonnegative four-level saturation-excess Hessian.
A null shape requires \(\delta G=0\) and the linearized quadratic saturation
identities on a rank-ten set. These are exactly the infinitesimal Clifford
representation identities, whose four-dimensional solutions are simultaneous
conjugations (the complex Clifford algebra is \(\mathrm{Mat}_4(\mathbb C)\)).
After quotienting those fifteen directions there is no null vector. On a fixed
finite cubic positive-weight set the remaining curvature is strictly positive.
The heavy contribution consequently has a positive finite-dimensional lower
coefficient; (3) and smoothness give
\(H_{\rm shape}=\Lambda H_{\rm shape,H}+O(1)\). The receipt computes the
radial Gram Hessian, **not** the full momentum-dependent 45-mode Hessian.
Spatial stability outside this uniform theorem requires the actual coupled
sea/gradient Hessian and is not asserted from the radial matrix.

Here are the hierarchy dimensions explicitly. Set \(\hbar=1\), lattice length
one, and base energy one. For a dimensionless shape displacement \(\phi\),
\([\phi,p]=i\),

\[
 H_\phi=\frac{p^2}{2M_\phi}+\frac{C_\phi\phi^2}{2},\qquad
 M_\phi=M_0\Lambda^{-1/2},\quad C_\phi=C_0\Lambda+O(1).
\]

\(M_0\) has inverse-energy units, \(C_0\) energy units; these are chosen
microscopic parameters, not derived inertia. Thus

\[
 \omega_\phi=\sqrt{C_\phi/M_\phi}
  =\sqrt{C_0/M_0}\Lambda^{3/4}(1+O(\Lambda^{-1})),\qquad
 \langle\phi^2\rangle=\frac1{2\sqrt{M_\phi C_\phi}}
  =\frac{\Lambda^{-1/4}}{2\sqrt{M_0C_0}}(1+O(\Lambda^{-1})).
\]

For dimensionless local gauge angle \(\theta\),
\(H_\theta=U_{E4}p_\theta^2/2+C_\theta\theta^2/2\), choose
\(J=J_0\Lambda\), \(U_{E4}=U_0\Lambda^{-1/2}\), and hold \(g_4\) fixed.
Both \(U_0,C_{\theta0}\) have energy units. The positive Higgs quadratic form
supplies \(C_\theta=C_{\theta0}\Lambda+O(1)\), so

\[
 \omega_4=\sqrt{U_{E4}C_\theta}=O(\Lambda^{1/4}),\qquad
 \langle\theta^2\rangle=\tfrac12\sqrt{U_{E4}/C_\theta}
                         =O(\Lambda^{-3/4}).
\]

The CLI reports these **harmonic** frequencies/variances from the bare matrix.
They are not measured dressed gaps. At fixed volume and positive restoring
coefficients, amplitudes shrink and heavy virtual denominators grow; heavy gap
\(O(\Lambda)\), shape frequency \(O(\Lambda^{3/4})\), and gauge frequency
\(O(\Lambda^{1/4})\) separate from the fixed light scale. The expansion is local
about one gauge-fixed saddle, with gauge redundancy removed and centre/topological
sectors specified. Uniformity in volume, nonperturbative defects, convergence
of the interacting quantum expansion, residual discrete-centre dynamics, and
light-loop renormalization are additional obligations. These exponents do not
constitute a full rotor run or a quantum 3D Higgs proof.

## 5. Statistics-sector theorem using the SU(4) centre, not heavy U(1)

A neutral heavy transfer cannot have its primitive-code sign cancelled by a
U(1) electric involution: there is no charged heavy U(1) path. The correct shared
supplier is the order-two central element \(-I\in SU(4)\).

On each untruncated link Hilbert space \(L^2(SU(4),dU)\), define the actual
unitary involution

\[
 (R_\ell f)(U)=f(-U).
\]

Haar invariance gives \(R_\ell^\dagger=R_\ell=R_\ell^{-1}\). It sends
fundamental matrix multiplication \(U_{ij}\) to \(-U_{ij}\), commutes with all
left/right electric generators and their Casimir, and leaves adjoint parallel
transport \(U A U^\dagger\) invariant. The charged light U(1) path and its
electric and magnetic energies are unchanged. Therefore it preserves the
continuous SU(4) and U(1) Gauss domains when they are imposed; this operator
identity is not a numerical construction of their vacuum.

For an admissible code-sector link character \(\eta_\ell=\pm1\), define

\[
 Q_\eta=\prod_{\eta_\ell=-1}Z_\ell,\qquad
 R_{SU4,\eta}=\prod_{\eta_\ell=-1}R_\ell,\qquad
 T_\eta=Q_\eta R_{SU4,\eta}.
\]

The actual primitive Pauli path word obeys
\(Q_\eta^\dagger A_\gamma Q_\eta=\eta_\gamma A_\gamma\), while the
fundamental SU(4) path product obeys
\(R_{SU4,\eta}^\dagger U^{(4)}_\gamma R_{SU4,\eta}
=\eta_\gamma U^{(4)}_\gamma\). Signs count each occurrence, so the product
cancels also for improved/repeated-link paths. Internal flavour/tree edges have
character one; the same physical link character is used by both species on the
shared extended primitive-code graph. Occupation/onsite terms, orbital kinetic
terms, the quartic, electric Casimirs, and adjoint Higgs gradients are invariant.
Fundamental hopping with arbitrary noncommuting orbital coefficients is thus
intertwined without requiring those coefficients to commute with their kinetic
Hamiltonian. This is the SU(4)-centre replacement of the coefficient theorem
in `quantum_orbital_sector.md`, not a reuse of its U(1) involution for neutral
matter.

Specify the SU(4) fundamental Wilson term by

\[
 C_{4,p}=\operatorname{Re}\operatorname{Tr}U_p/4,\quad
 H_{\rm mag,4}=-K_4\sum_p C_{4,p},\quad K_4\geq0.
\]

The normalization matters: \(\|C_{4,p}\|\leq1\), since every unitary trace has
real part between −4 and 4. With the primitive stabilizer defect penalty
\(\Delta\sum_p(1-S_p)/2\), and sector
\(s_p=\prod_{\ell\in\partial p}\eta_\ell\), the actual conjugation gives

\[
 T_\eta^\dagger H_sT_\eta
 =H_++2K_4\sum_{s_p=-1}C_{4,p}+\Delta N_{\rm def}I
 \geq H_++(\Delta-2K_4)N_{\rm def}I. \tag{6}
\]

This applies to realizable link-character sectors under the stated topology and
boundary conditions; it is not a claim that every arbitrary plaquette assignment
is realizable. Taking \(\Delta>2K_4\) selects the all-plus bottom in the exact
finite-volume comparison, assuming the Hamiltonian is bounded below. It does
not separate all excited bands. For example \(K_4=\Lambda\),
\(\Delta=3\Lambda\) gives a positive \(\Lambda\) comparison margin, but these
are choices, not computed parameter closure. `heavy_orbital_join.py` does not
pretend to evaluate the spectrum of (6).

### Domains and norm control

For bounded orbital coefficients on a finite graph, take the tensor sum of the
self-adjoint SU(4)/U(1) electric operators and orbital kinetic Hamiltonian, then
add the finite bounded symmetric hopping, Wilson, adjoint-gradient and penalty
perturbations. \(T_\eta\) preserves its domain; (6) is an operator identity on
the corresponding conjugated sector domains. Finite Peter–Weyl electric
compressions must use the same subspace in every sector and commute with central
translation (for example complete chosen representation blocks). Compressed
Wilson multipliers are contractions, so the bound survives. Arbitrary
non-invariant numerical cutoffs do not inherit this assertion.

For the actual unbounded orbital coordinates, use the common closed nonnegative
free form containing electric energy, orbital kinetic energy and the positive
quartic \(\|G\|^2\). The gradient is a nonnegative quadratic form; on a finite
graph it is controlled by the quartic plus a constant. Since fundamental link
multipliers have norm at most one, each hopping form is bounded by a constant
times \(\sum|y_{\mu a}|\,\|\psi\|^2\). Young's inequality against the coercive
quartic makes the total finite-graph hopping infinitesimally form bounded.
Require an invariant smooth finite-matter/finite-Peter–Weyl/Schwartz core, the
same topology/boundaries and coefficient domains in all matched sectors, and
include all path adjoints. KLMN constructs the semibounded closed form;
the equality on that core extends by form continuity and yields (6). No
unspecified unbounded operator product is assumed self-adjoint. The local
Gauss-invariant subspace, when used, is an invariant closed restriction because
central translation commutes with gauge generators and the code involution
fixes occupations.

This exact **statistics-sector** intertwiner is not an arbitrary-SU(4)-fluctuation
\(+/-m\) theorem for physical light matter. The latter remains unproved.
One only has the selected Clifford saddle and the controlled-hierarchy
low-energy expansion described above.

## 6. Reproduction and remaining parameter/limit debt

From `proposals/P256-qed-backward`, using the project's scientific interpreter:

```sh
/home/administrator/substrate-framework/.venv/bin/python heavy_orbital_join.py \
  --orders 24 32 --mass-ratios .2 --lambdas 1 8 64 \
  --output heavy_orbital_join_receipt.json
```

A bounded consumer smoke before the refinement run:

```sh
/home/administrator/substrate-framework/.venv/bin/python heavy_orbital_join.py \
  --orders 8 --mass-ratios .2 --lambdas 8 \
  --output /tmp/heavy_orbital_join_smoke.json
```

To retain both requested light base masses:

```sh
/home/administrator/substrate-framework/.venv/bin/python heavy_orbital_join.py \
  --orders 24 32 --mass-ratios .4 .2 --lambdas 1 8 64 \
  --output heavy_orbital_join_both_masses.json
```

The receipt binds source hashes, selected metrics and finite feedback, actual
poles/residues/recoil, hard and inclusive differential rates, comparator
deviations, exact-IR equality, bare fifteen-mode Higgs positivity/covariance,
and deterministic quadrature refinement. Orders 24/32, midpoint, light .2 and
\(\Lambda=1,8,64\) are defaults; masses, scales, stiffness, momentum, quadrature,
SU(4) coupling, alignment/electric prefactors and output path are exposed. A
CLI exit zero concerns its listed checks only, not a quantum phase assertion.

Parameters still chosen rather than derived are the primitive species/fillings,
base masses and improved dispersion, \(\kappa,\Lambda\), \(J_0,M_0,U_0,g_4\),
SU(4) Wilson/defect coefficients and the U(1) critical-surface tuning (5).
Statistics-sector control is conditional on the stated primitive code, matching
path law and domains; it does not supply a new arbitrary-SU(4) carrier theorem.
The order of limits is fixed-volume saddle/harmonic hierarchy first, quadrature
refinement for the charged uniform response separately; an infinite-volume
interacting Higgs phase, all-momentum stability, nonperturbative Gauss dressing,
parameter closure and physical-spin interpretation are not established. In
particular, adjoint Higgs locking leaves discrete centre questions and (6) is a
code-sector comparison, not their phase diagram. The parent full electron/photon
explanatory goal therefore remains open despite this materially joined supplier.
