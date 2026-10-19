# Actual finite bulk frame, light current, photon pole and Pauli consumer

`bulk_frame_compton.py` reads the **actual bytes** of an explicitly named branch in `periodic_bulk_frame_receipt.json`. The input supplies all 60 uniform Cartesian coordinates, not a selected Gamma frame or a global-ground assertion. It checks every input supplier source/proof hash and the original seed input, and binds its own source/proof/input bytes before and after the calculation. No scientific execution result is asserted by this derivation; the parent integration owns the actual CLI and receipt.

The accepted input is a finite validation-mesh stationary, positive-normal, irreducible rank15 candidate with one-dimensional complex commutant. The environmental and balanced-pair branches satisfy that supplier classification; the rank14 open-cube branch is not silently replaced with either. The conservative supplier quadrature bounds do not rank continuum ground energies. This join does not redo or enlarge the original mean minimization.

## 1. Fields determine the carrier

Let the supplied Hermitian traceless fields be `Phi_mu=sum_a c[mu,a] B_a`, with the executed owner `sea_selection.matrix_unit_basis` giving `Tr(B_a B_b)=4 delta_ab`. Measure

\[
b_\mu^2=\operatorname{Tr}(\Phi_\mu^2)/4,\qquad
\gamma_\mu=\Phi_\mu/b_\mu.
\]

The computation **checks**, rather than imposes, `gamma_mu gamma_nu+gamma_nu gamma_mu=2 delta_mu,nu I`, and `b_1=b_2=b_3=s` to a declared finite numerical tolerance. It rejects a real non-Clifford field mutation. Normalizing a failed frame is not a fallback. All raw input matrices remain available in the output; the reduction's residual is not erased.

The simultaneous equations `gamma_mu U=U Gamma_mu`, where `Gamma=(MIXING,HOPPING)` comes from the existing orbital matrix-unit owner, are solved as a complex linear nullspace problem. The nonsingular polar factor gives a unitary intertwiner; all four equations, uniqueness of its null direction and the actual commutant are checked independently. The field-derived band is

\[
a_{\rm eff}=a,\quad m_{\rm eff}=m b_0,\quad
r_{\rm eff}=r b_0,\quad v_{\rm eff}=v s.
\]

The hopping remains **unimproved**. For the ideal Clifford limit,

\[
h_{\Phi}(p)=U\left[
 \frac{v_{\rm eff}}a\sum_i\sin(ap_i)\Gamma_i
 +\left(m_{\rm eff}+\frac{r_{\rm eff}}a\sum_i[1-\cos(ap_i)]\right)\Gamma_0
\right]U^\dagger.
\]

The executable reports the finite approximation error, verifies the actual raw `h`, finite-transfer Peierls current and two-photon contact at two nontrivial momentum pairs, and checks independent finite differences of the raw path owner. Default frames, unscaled coefficients, default currents/contacts and a non-Clifford frame are real adverse controls. The raw law is built by `local_orbital_stability.scalar_paths(bare)` contracted with the actual fields, not by renormalizing preselected matrices.

The internal generators `S_i=-i gamma_j gamma_k/2` yield the tested SU2/vector-action algebra, `sum_i S_i^2=3I/4`, and `exp(-2pi i S_z)=-I`. This is internal spin algebra of the carrier. Exact continuous coordinate rotations of a finite cubic lattice are **not** derived. Continuum envelope rotations remain conditional on the low-momentum limit and on joining field/link transformations; they do not follow just from the matrix commutators.

## 2. Rest, inertia, Wilson curvature and speed

For the actual unimproved band,

\[
E(p)=m_{\rm eff}
 +\frac12\left(\frac{v_{\rm eff}^2}{m_{\rm eff}}+a r_{\rm eff}\right)|p|^2+O(|p|^4),
\qquad
M_{\rm inert}^{-1}=v_{\rm eff}^2/m_{\rm eff}+a r_{\rm eff}.
\]

Thus rest energy, linear matter speed, inertial mass and scalar Wilson mass curvature `a r_eff/2` are distinct. The relation `M_inert=m_eff/v_eff^2` requires the additional small-Wilson condition `a r_eff m_eff << v_eff^2`; it is not silently enforced. Corner masses are `m_eff+2n r_eff/a`, with multiplicities 3,3,1 for n=1,2,3. Holding the finite frame fixed as a tends to zero makes these corners heavy, but is not a solved bulk cutoff flow or a parameter-closure theorem. The computation reports the bare photon speed and each measured dressed speed separately; it never sets c=1.

## 3. Required Band API and physical units

The clean APIs are:

```python
transverse_spectrum(L, band, mode=1, *, gradient=False, polarization_axis=1)
momentum_response(L, band, mode=1, *, gradient=False)
match_photon(L, band, mode=1, *, electric_coefficient=UE1,
             magnetic_coefficient=K1, polarization_axis=1)
forward_response(L, band, *, electric_coefficient=UE1, magnetic_coefficient=K1)
forward_at_pole(band, measured_pole)
```

`band` is a **required actual RotationBand** and owns a,m,r,v,e and the hopping flag. There is no legacy `ma` overload and no pole-level `improved` override. The old explicit CLI factories construct their former bands at the callers. `vacuum_log.py` also passes its already constructed band. LSP references and literal caller searches are the cutover inventory; `tests/test_photon_matching.py` exercises the actual weak-coupling raw/reduced pole and connected-current path.

Momenta are physical: `q=2pi n/(a L)`, while `qangle=a q` and the Maxwell link difference is `d=2 sin(qangle/2)`, **not** d/a in the compact-link restoring term. The transverse profile is `theta_y=t cos(q x)`, with profile norm `L^3/2` because its mean square is one half, not because fewer cells exist. Both signed transfer channels are retained. The independent z-polarized profile checks the second physical transverse mode. The forward consumer uses one profile's measured pole for both polarizations only after their energy/residue discrepancy passes the declared isotropy tolerance.

The chosen Gaussian Hamiltonian is

\[
H_\gamma=\frac{UE1}{2}\sum_l E_l^2+
\frac{K1}{2}\sum_p(\operatorname{curl}\theta)_p^2,
\qquad \epsilon=1/UE1,\qquad c_{\rm bare}=a\sqrt{UE1 K1}.
\]

CLI `--electric` means **Hamiltonian UE1**, never the Lagrangian epsilon. Neither coefficient is borrowed from the old unit-speed photon defaults, a compact Monte Carlo receipt or the old `.3` current normalization. The link coordinate obeys `theta=a A` for physical A; the integer/light charge is already in the band current.

If the per-cell compact-angle inverse pole residue is `R_theta`, then

\[
Z_\theta=1/R_\theta,\quad
\ell_\theta=R_\theta^{-1/2},\quad
R_A=R_\theta/a,\quad \ell_A=\sqrt{a/R_\theta}.
\]

The physical-field factor uses **continuum-volume normalization**: `sum_cells a^3 -> integral d^3x`. A cell-normalized physical-A oscillator instead has factor `1/(a sqrt(R_theta))`; the output names both, without confusing them. Compton polarizations use `ell_A`, and the physical charge is `band.charge*ell_A`, not another extra charge multiplication.

For the historical physical-spacing factory, `h_phys=h_unit/a`, `UE1_phys=UE1_unit/a`, `K1_phys=K1_unit/a`. Consequently `omega_phys=omega_unit/a`, `R_theta,phys=a R_theta,unit`, and `ell_A,phys=ell_theta,unit`. The previous explicit factories' numerical workflow is preserved by these declared caller coefficients, not by a hidden root or unit change.

## 4. Actual raw sea pole, then actual Pauli response

Raw finite-torus Hamiltonians are assembled from the **unrounded actual fields** and their original scalar paths. Their occupied-to-empty channels supply gaps Delta>0 and nonnegative weights w. Contact C and pairs are both differentiated from the same primitive light paths. With N=L^3 and norm=N/2,

\[
\Pi_0=(C-2\sum w/\Delta)/\mathrm{norm},\quad
\eta_0=2\sum w/\Delta^3/\mathrm{norm},\quad
S=K1 d^2+\Pi_0,
\]
\[
D(s)=S-\epsilon s-\frac{2s}{\mathrm{norm}}\sum
\frac{w}{\Delta(\Delta^2-s)},\qquad
R_\theta=-D'(s_*)=\epsilon+\frac2{\mathrm{norm}}\sum
\frac{w\Delta}{(\Delta^2-s_*)^2}.
\]

For `S>0` and `b=S/(epsilon+eta0)<min Delta^2`, the positive unique root lies in

\[
\frac{S}{\epsilon+\eta_0/(1-b/g^2)}\le s_*\le b,
\qquad g=\min\Delta.
\]

Failure of the actual positive below-gap condition raises a natural error: no retuning, removal of unstable channels, fake pole or prescribed dispersion fallback occurs. Both raw y/z roots and residues are measured. The required-band spectrum/pole path independently checks their transfer to the field-derived reduction, including raw contact, pair, electric, pole and residue errors. The consumer receives the **raw measured** root/leg, not the reduced comparison root. Its band still has the stated finite Clifford-reduction error; no exact raw spinor identity is claimed beyond that tolerance.

A separate raw longitudinal source profile verifies the contact/pair Ward cancellation, and deletion of either term must break it. `FilledSeaConsumer` then acts in its actual primitive Pauli occupation-state sector, keeps electron–pair intermediates and subtracts the forward vacuum-disconnected response. The independent one-particle resolvent uses the same measured energy, current, contact and external spinors; it is a comparator, not the Pauli intermediate supplier. Ward substitutions include both gauge insertions, so deleting the genuine two-photon contact remains detectable even when a single longitudinal/transverse orthogonality would hide it. Pair deletion and the **actual bare** field residue `sqrt(a UE1)` are separate controls. There is no hardcoded canonical `.3` leg.

Deletion effects are measured, not required to exceed a universal absolute
size. Such minimum-effect gates are removed from the raw pole consumer;
full raw Ward cancellation and raw/reduced kernel transfer remain required.
The pole-equation residual is scaled by the actual positive restoring force.
Weak-coupling scattering additionally compares full versus contact-deleted
Ward defects relatively, so a small physical correction is not confused
with permission to omit it. See [the pole regression](photon_matching.md).
The raw pole also uses the algebraically subtracted nonlinear kernel and a
dimensionless fraction of its certified upper bound, retaining the bracket
when a weak sea correction is below a bare subtraction ulp. Positive
`R>=epsilon>0` is the physical residue bound, not a required resolved dressing
size. The raw and reduced contact-dominated recovery is exercised by the
same physical regression.
The row-level longitudinal sea control likewise keeps contact and pair
deletion values as diagnostics, without demanding either exceed `1e-7`.
Weak velocity can make both small while the complete sea Ward identity
remains valid.
The raw/reduced pole-error scalar is normalized to a native Python float
before its maximum becomes a receipt boolean. The actual approximate-frame
`L=16` consumer otherwise produced a NumPy boolean and could not serialize
its completed result; no physics threshold or result was changed to fix it.


## 5. Primitive current-excited neutral Gauss test

The same actual mean is inserted at both endpoints of `FullBandGaussSection(bare, heavy_scale=1)`. This is its inherited **one-active-x-axis two-cell owner**, whose onsite is m+r/a, not the full three-dimensional periodic onsite. The common frame is shared; the geometries are not conflated. A declared genuine nontrivial SU4 link and two real compact source angles are used. These angles are probe inputs, not evidence for a photon phase. Its 4900-state NH=NL=4 section and directed transfer units are unchanged.

For every one of the 15 generators at each endpoint, transform **both** fields and link along the genuine local gauge path. The section derivative is the owner's exact normalized projector/product derivative. Independently differentiate the actual primitive `peierls_action` current at fixed psi by central differences on that path. Then check the actual vector equation

\[
(dJ)\psi+J(d\psi)=iQ(J\psi).
\]

The factor i is the derivative convention for g(t)=exp(i t T). This test does not supply dJ through `i[Q,J]` or differentiate a claimed covariance equation into a tautological check. Both light-only EM endpoint equations are also tested, with their shifted compact angles and neutral background subtraction NL(site)-2. A **real undressed current** uses default orbital fields held fixed while the link and actual section transform; it must fail the combined field/link identity. Norm, all 30 colour residuals, both EM residuals, finite-difference steps and bad-current defects are retained.

The neutral hidden Pauli parity marker in the forward electron consumer is distinct from an external SU4 B4/bar4 charged boundary representation. Neither this neutral Gauss section nor its current excitation proves a physical charged Gauss gap, charged self-energy, quantum normal confinement, chosen photon phase, compact Coulomb phase, nonforward/radiative Compton amplitude or Lorentz parameter closure. `photon_phase` is **null/absent**, not a fabricated numeric zero.

## 6. Parent integration commands

From the repository root, using its scientific interpreter:

```sh
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/bulk_frame_compton.py --input-bulk-receipt proposals/P256-qed-backward/periodic_bulk_frame_receipt.json --branch environmental_Clifford_L3 --lengths 16 32 --electric .4 --magnetic .7 --output proposals/P256-qed-backward/bulk_frame_compton_receipt.json
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/vacuum_limit.py --lengths 8 16 --mass-ratios .05 .2 --output proposals/P256-qed-backward/vacuum_limit_receipt.json
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/vacuum_log.py --output proposals/P256-qed-backward/vacuum_log_receipt.json
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/photon_matching.py --lengths 16 32 --ma .2 --output proposals/P256-qed-backward/photon_matching_receipt.json
/home/administrator/substrate-framework/.venv/bin/python proposals/P256-qed-backward/photon_matched_compton.py --lengths 16 32 --output proposals/P256-qed-backward/photon_matched_compton_receipt.json
```

The new command's electric/magnetic coefficients and finite sizes are an explicitly selected probe, **not** a claimed observed positive pole before execution. The CLI retains natural failure for insufficient below-gap/restoring conditions. Source/proof hashes distinguish any subsequent executed revision from this derivation.
