# A different positive colour-kinetic family with bounded separated trial cost

## Decision and exact change

The native separated `L` insertion with compensating colour and EM reference at `R` exists on the complete Hilbert space, but the original fixed-`ue4` estimate grows as `Lambda^(5/12)`. Its colour derivative contains the full Cartesian field angular momentum. A reference-field unitary does not cancel that term; replacing field momentum with bounded matter charge is invalid.

This construction instead declares a **different microscopic parameter trajectory**:

\[
 u_4(\Lambda)=u_{4,0}\Lambda^{-p},\qquad 5/6<p<1,
 \qquad p=11/12\ \text{by default}.
\]

The original pair law otherwise remains: `a=1,m=.2,r=.12,v=.3`, quartic stiffness two, alignment `.2`, field kinetic coefficient `sqrt(Lambda)`, heavy energy scale `Lambda`, `ue1=.4` and `K=.7`. At every finite `Lambda>=1`, colour kinetic energy is strictly positive. All 120 Cartesian coordinates, complete Haar and torus representations, and neutral/charged native fibres remain. No Hamiltonian is restricted to a field packet, and no exact neutral ground is replaced by a sampled packet. This is **not** a uniform bound at fixed `ue4=.3` or a claim that small colour kinetics are predicted.

## Exact-ground inequality that is reused

`pair_separated_charge_transfer.md` proves, using complete Gauss identities and native CAR, that the two physical path images of the actual neutral ground density give for either charged sign

\[
 E_{\rm charged}-E_N
 \le C_L(\Lambda)+u_4 C_F+u_1/2
       +\sqrt{C_{L,\max}u_4 B_T(\Lambda)},
 \qquad C_F=15/8,\quad C_{L,\max}=5/2.
\]

`C_L` is the invariant-moment upper bound on `-<Hlight>/4`; it tends to `light_anchor_lower_cost/4`. `B_T` bounds the **sum of all positive neutral kinetic energies**. The negative EM variance contribution has merely been dropped. The signed light-charge/Cartesian-angular correlation has not been set to zero: Cauchy controls it through the colour kinetic energy of the actual family.

These are variational upper bounds on actual quantum sector infima on a closed finite graph. The reference compensates the total EM source. Neither sign is an isolated charge, and the upper bound is not a positive charged gap.

## One packet law bounds every member

Compute the full-domain trial packet constants with the fixed largest coefficient `u4,0`. The trial's colour kinetic expectation is nonnegative and linear in this coefficient. Therefore its energy is an upper bound for every member with `0<u4(Lambda)<=u4,0`. The common lower multiplication inequality is independent of the positive colour kinetic coefficient. The same fixed-anchor packet law consequently bounds both the neutral moments and total kinetics of **each actual member's own exact ground**.

This monotonicity is essential: the coefficient of the Hamiltonian changes, and its ground state is not asserted to be the old one. The packet provides only a comparison upper energy, never a coordinate cut or a ground-state approximation substituted into the theorem.

Let `ell,A,Am,B,Bm,CL,K,CY,r` denote respectively `leading_slope`, field Fisher/metric, electric Fisher/metric, light-anchor lower cost, magnetic lower cost, Young coefficient and chart radius from the source's full packet law. Set

\[
 \alpha=(2A/\ell)^{1/3},\qquad
 \rho=\min(r,\alpha\Lambda^{-1/6}).
\]

Retaining the positive kinetic terms in the lower inequality gives the cancellation-free formula

\[
 B_T=\Lambda\ell\rho+
 \sqrt\Lambda(A/\rho^2+A_m)+B/\rho^2+B_m
 +C_L+K+C_Y\Lambda^{-1/3}.
\]

Every summand is nonnegative. Evaluating this formula directly avoids subtracting nearly equal order-one scaled energies at large `Lambda`; such subtraction can numerically erase the kinetic upper excess. This is still a floating evaluation of an analytic inequality, not an interval computation or measured ground kinetic energy.

For all `Lambda>=1`, `rho<=alpha Lambda^-1/6` and
`1/rho^2<=1/r^2+Lambda^(1/3)/alpha^2`. Hence

\[
 B_T\le C_T\Lambda^{5/6},
\]

where the explicitly finite constant used in the producer is

\[
 C_T=\ell\alpha+A/\alpha^2+A/r^2+A_m
 +B/r^2+B/\alpha^2+B_m+C_L+K+C_Y.
\]

In the unclamped asymptotic regime its sharper leading prefactor is just `ell alpha+A/alpha^2`.

## Surviving separated cost and the other scale

The colour cross contribution is now bounded by

\[
 \sqrt{(5/2)u_{4,0}C_T}\;
 \Lambda^{5/12-p/2}.
\]

It tends to zero for `p>5/6`. The colour self term also vanishes. The light moment bound tends to its finite selected value and is bounded for `Lambda>=1`; therefore the separated trial cost is uniformly bounded and

\[
 \limsup_{\Lambda\to\infty}(E_{q,0}-E_N)
 \le C_L(\infty)+u_1/2.
\]

For the default pair this expression is the source's `.627897375...` plus `.2`. The receipt computes it rather than treating that displayed approximation as a proof.

With the field coordinates **frozen**, an isolated positively curved colour-link subblock has potential curvature proportional to `Lambda` and kinetic coefficient proportional to `Lambda^-p`, giving the harmonic frequency power `(1-p)/2`. At `p=11/12` this conditional subblock growth and the colour-cross decay both have exponent magnitude `1/24`. This motivates `p<1` but is **not** the full mixed field/link Gauss-normal spectrum: the Cartesian kinetic coefficient `sqrt(Lambda)` also participates there. Flat/stabilizer directions, the full nonlinear spectrum and a quantum Higgs phase remain unproved.

The fixed-`u4` estimate remains as a control: its cross upper bound grows as `Lambda^(5/12)`. That is failure of this estimate to be uniform, **not** proof that the actual fixed-coupling charge cost diverges or that a better fixed-coupling construction is impossible.

## Why a naive scalar comparison does not suffice

A scalar-coordinate ground times a normalized heavy ground section can remove real scalar-amplitude derivative cross terms. But the heavy section's Cartesian derivative energy is multiplied by `sqrt(Lambda)` in this actual Hamiltonian. Dropping that common section energy from a scalar lower comparison does not give a uniform charged-minus-neutral upper bound. The present route does not use that subtraction; it keeps the exact-ground transfer inequality and changes one declared bare coefficient instead. A sharper common Born–Oppenheimer spectral comparison remains a different possible fixed-coupling route.

## Consumer and evidence boundary

The producer binds a completed native `pair_separated_charge_transfer_receipt.json` and verifies all of its source/proof hashes. Those native full-fibre noncentral-link/all120-point controls prove the same Gauss and kinetic product identities used here; multiplying a coefficient by a positive scalar leaves that algebra unchanged. The new producer evaluates the different-family analytic bounds, checks its uniform envelope, compares fixed coupling, and binds its own source/proof. It does **not** report a numerically computed quantum ground.

```
python proposals/P256-qed-backward/pair_separated_charge_transfer.py
python proposals/P256-qed-backward/pair_separated_charge_scaling.py
```

This pair trajectory is not the separate bulk `kappa(a)=2/a` cutoff family or the nonlinear compact-current ensemble. In particular keeping these pair/photon inputs does not solve their shared-speed mismatch, and sending a photon cutoff to zero would require additional Coulomb/self-energy and volume control. Physical mobile spin, a positive charged lower gap, infinite-volume reference separation, photon phase and the joined electron–photon scattering campaign remain open.
