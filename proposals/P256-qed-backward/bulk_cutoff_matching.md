# Source-selected cutoff family and same-current Gaussian matching

## Scope and declared changes

`bulk_cutoff_matching.py` constructs a **different bare family**, not a repair by relabelling the original point. Primitive `m=0.2`, `r=0.12`, `v=0.3`, charge `e=1` and the unimproved Wilson law remain fixed; the physical quartic stiffness is explicitly `kappa(a)=2/a`. The colour connection is flat. All sixty uniform traceless Hermitian field coordinates and total heavy half filling are retained. The source-selected Clifford representative is reconstructed by the existing primitive frame/current consumer.

The photon coefficients `UE1(a)` and `K1(a)` are explicitly tuned free parameters. The original `a=1, UE1=0.4, K1=0.7` point is not made speed matched by this construction. Results concern the selected **uniform classical field and frozen-band Gaussian sea**. No nonuniform quantum ground state, compact thermodynamic phase, physical charged spin sector or full radiative Lorentz closure is asserted.

## Actual field selection and continuum trajectory

Write dimensionless momentum `theta=a p`. The actual primitive coefficients satisfy

\[
a x_0(p)=a m+r\sum_i(1-\cos\theta_i),\qquad
 a x_i(p)=v\sin\theta_i.
\]

Multiplication of the uniform physical energy by `a` gives exactly the full-Gram functional from `soft_budget.py` with stiffness two and dimensionless mass `mu=a m`:

\[
\widehat F_\mu(G)=\operatorname{Tr}G^2
 -2\langle\sqrt{\widehat x_\mu^T G\widehat x_\mu}\rangle,
 \qquad G\succeq0.
\]

The unrestricted uniform theorem in `bulk_global_gram.md` applies: total occupied heavy energy, not a forced two-per-momentum surrogate, obeys the spectral lower bound; every PSD Gram is attained by a Clifford representative. The positive quartic term makes this functional strictly convex and coercive. Cubic symmetry gives its unique minimizer `diag(g0,gS,gS,gS)`. The producer solves this functional on two actual meshes, verifies the full ten-component gradient and rank-ten quadratic richness, and constructs actual field matrices. Mesh refinement is a numerical observation, not an interval integral certificate.

The continuum-integral minimizers converge to the unique minimizer at `mu=0`: the coefficients converge uniformly on the compact Brillouin torus, the functionals converge uniformly on bounded Gram sets, and coercivity bounds the minimizers. Both limiting scales are positive. On the boundary `g0=0,gS>0`, the one-sided derivative contains the strictly negative integral of `x0^2/sqrt(gS sum xi^2)`. Its isolated three-dimensional singularities are integrable. On `gS=0,g0>0`, the corresponding integral of `sum xi^2/sqrt(g0 x0^2)` is finite and strictly positive; near the origin both numerator and `x0` are quadratic. The origin in Gram space cannot minimize because the negative square-root term dominates the quadratic quartic cost under a small positive scaling. These statements concern the continuum functional; the receipt's endpoint numbers are finite-mesh estimates.

Consequently

\[
 M(a)=m\sqrt{g_0(a)}\to M_0>0,\quad
 c(a)=v\sqrt{g_S(a)}\to c_0>0,\quad
 r_{\rm eff}(a)=r\sqrt{g_0(a)}\to r_0>0.
\]

At bounded physical momentum the actual reconstructed band obeys

\[
 m_a(p)=M(a)+\tfrac12a r_{\rm eff}(a)|p|^2+O(a^3|p|^4),\quad
 s_i(p)=c(a)p_i+O(a^2|p|^3).
\]

Thus its free energy converges to `sqrt(M0^2+c0^2 |p|^2)` uniformly on bounded physical momentum sets. The relative Wilson inertial debt is `a r_eff M/c^2` and tends to zero. A corner with `n` components equal to `pi/a` instead has mass `M+2n r_eff/a`, so all seven other corner species decouple. This is a free frozen-band conclusion, not a claim that its microscopic quantum phase has been established.

## Photon units and explicit matching

The source spectrum is evaluated on the **same** reconstructed band, in dimensionless energy `a H_phys`; it is not an imported improved-band source. A continuum gauge field has link phase `theta_link=a A`. For Hamiltonian rotor coefficients in physical energy,

\[
 \chi_E^{\rm bare}=\frac1{a UE1},\qquad
 \chi_B^{\rm bare}=a K1.
\]

Let `PiE,PiB` be the actual source's infrared coefficients. For a declared target `g2>0`, choose

\[
 \chi_E^{\rm bare}=\frac1{g2\,c}-\Pi_E,\qquad
 \chi_B^{\rm bare}=\frac c{g2}-\Pi_B.
\]

Both must be strictly positive; otherwise this family is rejected. The effective quadratic coefficients then obey `chiB/chiE=c^2` and `g2=1/(chiE c)`. This is **parameter tuning**, not a microscopic prediction of either speed or charge. A deliberately excessive target coupling is checked against the same positivity boundary.

For each finite momentum the producer retains the actual contact tensor and all spectral channels, replacing only the explicitly chosen bare coefficients. It performs no transverse projection, angular symmetrization or coefficient refit on that tensor. Both transverse modes, inverse residues, pair threshold and the **full Ward vector scaled by restoring action** are checked. Dividing the Ward defect only by the much larger contact tensor can conceal errors in a small restoring action; that contact-scaled diagnostic is retained but is not the acceptance boundary. Split-Gauss orders are independently compared. Finite-step photon speeds are compared with `c |2 sin(qhat/2)|`, not with the continuum `c |qhat|` at coarse spacing.

If `Rhat=d Khat/d omegahat^2`, conversion of the physical link inverse kernel gives `Rtheta=a Rhat`. Since `theta_link=a A` and physical cell volume is `a^3`, the physical continuum-field inverse residue is **RA=Rhat**. The existing Compton consumer therefore receives `1/sqrt(Rhat)` as its physical field leg. This is not the oscillator wave leg `1/sqrt(2 omega Rtheta)`.

## Same-current forward response

The actual matched axis pole is passed to `photon_matched_compton.forward_at_pole`, including identical reconstructed band parameters, physical momentum and energy, and the physical rotor coefficient. The consumer compares the native filled-sea Pauli response with its independent one-particle kernel and checks the full same-pole Ward contraction. The neutral hidden Pauli boundary marker remains a **neutral code marker**, not an external `B4/bar4` charged-sector construction.

Time normalization must be explicit. For `tau=c t`, `H_tau=H_t/c`; canonical photon legs divide by `sqrt(c)`. The two-leg amplitude consequently satisfies `amplitude_tau=amplitude_t/c^2`, while `M_tau=M_t/c`. The reported invariant is therefore the native `M_t^2 |amplitude_t|^2 / c^6`. Its continuum forward soft target is `4 e^4 g2^2`. At finite spacing the actual Wilson soft factor is `(1+a r_eff M/c^2)^2`; finite-frequency residue effects are measured rather than assumed absent.

The former universal minimum-size checks for deleting contacts or replacing the dressed residue were incorrect: both corrections can become physically small under cutoff or weak-coupling scaling. They have been removed from the shared API, not re-pinned to a smaller arbitrary threshold. Independent kernel equality, full Ward identity and actual pole checks remain. Raw deletion and bare-leg diagnostics remain in the receipt. The weak-coupling real-path smoke at `a=0.001,g2=0.0001` reached both previously failing size branches while the complete response agreed with its independent kernel and had a Ward residual below `6e-22`.

The same diagnostic-only rule applies to virtual-pair effect sizes: a
contact-dominated but valid source can make that relative correction small.
The shared raw/reduced scalar pole solvers now remove the linear electric
term analytically and solve a dimensionless certified-bound fraction.
Positive residues may equal the bare coefficient after floating rounding;
that is not a failure of physical positivity.

At the finest selected point the producer also runs the existing full 4900-dimensional neutral primitive Gauss/current-excitation construction. This is a joined one-axis core control; it is not a three-dimensional many-body ground-state proof.

## Limits that survive the tuning

The positive electric vacuum response grows logarithmically as the dimensionless fermion mass vanishes in this one-loop Gaussian source. Thus at fixed nonzero target `g2`, eventually `PiE>1/(g2 c)` and the required bare electric coefficient becomes negative. This construction supplies **finite-cutoff shared-speed accuracy**, not an interacting ultraviolet continuum. The free fermion limit remains distinct from the positive-bare-coefficient range of the photon family.

Full primitive quantum uniform/nonuniform selection, a separated compensated-reference charged excitation, a positive charged lower gap and mobile spin sector, compact phase control, and nonlinear photon/vertex/self-energy closure remain open. Neither this receipt nor successful finite tolerances complete the campaign.

## Reproduction

Run with the project's scientific Python environment:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python proposals/P256-qed-backward/bulk_cutoff_matching.py
```

Defaults compare spacings `1,0.2,0.05,0.01,0.005,0.001`, Gram grids `32/48`, split-Gauss orders `16/24`, and physical photon radii near `0.1/0.05` plus a non-axis direction. The output binds every inherited source/proof and this producer/proof, and rejects a mid-run change. `campaign_complete` is deliberately false.
