# Moving-frame coupled Pauli filled-sea scattering

This contribution owns `moving_frame.py`. It extends the **actual**
`pauli_compton.FilledSeaConsumer` to nonzero incoming electron momentum. It does
not replace the primitive edge-Pauli current, real virtual electron-hole channels,
contact term or connected vacuum subtraction with a continuum matrix kernel.
The continuum comparator is an independent scalar invariant formula, not the
supplier's `comparator()` or another evaluation of the candidate kernel.

The numerical run is intentionally delegated to the integration owner. No
scientific execution or numerical outcome is claimed in this contribution.
The default CLI computes and reports the observations described below and
rejects violations of its scoped conservation, Ward and refinement criteria:

```sh
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/moving_frame.py
```

`--output PATH` additionally writes the same JSON. The script has no random
sampling, fit, finite-spacing speed adjustment or measured-charge inference.

## Target selection is not a candidate boost law

Start with a continuum rest Compton event of electron mass `m`, incoming photon
energy `w`, scattering angle `theta`, and a selected incident unit vector `n`.
Choose an orthogonal tangent to specify the scattering plane. With

\[
w'=\frac{w}{1+(w/m)(1-\cos\theta)},\qquad
P=(m,0),\quad K=(w,wn),\quad K'=(w',w'n'),\quad P'=P+K-K',
\]

all four momenta are their own exact continuum event. Apply the active boost

\[
E_b=\gamma(E+\boldsymbol\beta\cdot p),\qquad
p_b=p+\left[\frac{\gamma-1}{\beta^2}
 (\boldsymbol\beta\cdot p)+\gamma E\right]\boldsymbol\beta.
\]

This operation selects the **input** lab electron momentum, incoming photon
momentum and outgoing photon direction. It never transforms a finite-spacing
candidate amplitude or asserts a candidate Lorentz law.

Two default noncollinear boosts and different angles/energies/scales are:

| Case | `m` | positive input `e` | `w/m` | `theta` | boost velocity |
|---|---:|---:|---:|---:|---|
| oblique low energy | 1 | 0.30350 | 0.35 | 1.05 | (0.23, -0.17, 0.11) |
| different frame and scale | 1.7 | 0.21 | 0.85 | 1.4 | (-0.14, 0.27, 0.19) |

Their incident directions are respectively proportional to `(0.12,0.2,1)` and
`(0.4,-0.2,1)`. Both electron lab momenta are nonzero. No selected continuum
outgoing magnitude is substituted into a finite-spacing conservation equation.

## Independent candidate recoil and observable

Use matched `RotationBand` and `WilsonLink` parameters, including the actual
range-two improved hopping/current/contact law. The matter speed is fixed at
`v=1`; matched runs select photon speed `c_gamma=1`. Their dispersions are

\[
E_a(p)=\sqrt{v^2\sum_i\frac{\sin^2(ap_i)}{a^2}+M_a(p)^2},
\quad M_a(p)=m+\frac r a\sum_i(1-\cos ap_i)^2,
\]

\[
\omega_a(k)=c_\gamma\left|\widehat k\right|,
\qquad \widehat k_i=\frac2a\sin(ak_i/2).
\]

Keep only the boosted input spatial momenta `p,k` and target outgoing direction
`n_out`. Solve for a **new** positive magnitude `t`:

\[
E_a(p+k-t n_{out})+\omega_a(t n_{out})
             =E_a(p)+\omega_a(k).
\]

Then set `k_f=t n_out`, `p_f=p+k-k_f`, and pass precisely
`(p,k,p_f,k_f)` to `FilledSeaConsumer`. Incoming energies, outgoing energy,
final electron energy and recoil are all supplied by the candidate. There is no
fake/overridden consumer kinematics. The report includes all four spatial
momenta, candidate energies, energy/momentum residuals and the relative
candidate-vs-target recoil correction.

The scalar solve brackets the low-zone root without using the target recoil as
a root. It scans the bounded outgoing ray for upward crossings and rejects
missing or multiple such crossings. Each external momentum and both virtual
channel momenta `p+k` and `p-k_f` must satisfy
`max_i |a p_i| < 0.6`. This is a scoped low-Brillouin calculation, not a theorem
about uniqueness on the entire periodic dispersion, Umklapp or high-zone roots.

For each of the 16 spin/polarization combinations obtain the primitive connected
amplitude `A_a`. The observable is

\[
\overline{|\mathcal M_a|^2}
 =\frac14\sum_{s,s',\lambda,\lambda'}
       \left|2\sqrt{E_a(p)E_a(p_f)}\,A_a\right|^2.
\]

This uses the same `2 sqrt(Ei Ef)` external normalization as `lattice_compton`.
The denominator averages two initial electron spins and two initial photon
polarizations; all final spins/polarizations are summed. Candidate photon
polarizations are transverse to `khat`, not to an unrelated continuum momentum.
The final electron and normalization use the newly solved candidate recoil.
Calling this an invariant-normalized observable does **not** declare finite-a
Lorentz-invariant phase space or a continuum cross section.

## Independent target and two checks on its meaning

For the **target's own** exact boosted continuum event define, with metric
`(+---)`,

\[
x=\frac{P\cdot K}{m^2},\qquad
 y=\frac{P\cdot K'}{m^2},\qquad d=\frac1x-\frac1y.
\]

The independent spin/polarization-averaged Klein-Nishina target is

\[
\overline{|\mathcal M|^2}_{KN}
 =2e^4\left(\frac xy+\frac yx+2d+d^2\right).
\]

In the original rest event, `x=w/m`, `y=w'/m`, and
`d=-(1-cos(theta))`. Thus `2d+d^2=-sin(theta)^2`, independently giving

\[
2e^4\left(\frac w{w'}+\frac{w'}w-\sin^2\theta\right).
\]

The script evaluates both expressions separately and reports their difference;
it also verifies the target event's mass shells, null photons and four-momentum
balance. It does not evaluate the invariant formula on candidate momenta that
are not continuum mass/null shells. Candidate corrections are measured against
the fixed target, not against a target silently recoiled to agree with it.

For each nonrest case the default refinements have `a*m` equal to
`0.16,0.08,0.04,0.02`. Outputs retain signed observable corrections, absolute
relative errors, successive error ratios and recoil corrections for each frame.
Default acceptance requires the finest relative error below `0.004` and below
one eighth of the coarse error. These are runtime acceptance criteria, **not**
reported measurements before the integration run. The range-two action removes
the standard Wilson `O(a)` mass curvature. At fixed physical scale its remaining
sine-symbol and photon-dispersion corrections start at `O(a^2)` and improved
mass corrections at `O(a^3)`; their coefficients depend on the actual lab
momenta, frame, mass and scattering event. No finite-a exact covariance follows.

## Real Ward and channel-deletion controls

On each candidate event substitute the actual gauge variations

\[
\epsilon_{in}\to(\omega_a(k),\widehat k),\qquad
\epsilon_{out}\to(\omega_a(k_f),\widehat k_f)
\]

one leg at a time. All external spins and the other leg's physical polarizations
are exercised through `FilledSeaConsumer.amplitude`. The report records the
maximum **externally normalized amplitude** residual separately for each leg.
This does not use a formal Ward identity as a replacement for evaluating the
actual consumer.

The same loops perform real `keep_pairs=False` and `keep_contact=False`
evaluations. They report full/no-pair/no-contact squared sums, maximum physical
amplitude changes, and Ward residuals. The strongest actual virtual electron-hole
channel retains its spin, polarizations, complex channel amplitudes and
intermediate-state counts as a witness. The no-pair physical-amplitude change
must remain nonzero at the finest matched spacing. Contact deletion is most
informative at finite spacing: its coefficient vanishes in the continuum, so
its eventual smallness is not evidence the microscopic contact was unnecessary.
No assertion requires a finite continuum contact contribution.

## Speed mismatch surviving refinement despite rest isotropy

Fix **the same** matter law, `v=1`, input mass, charge and boosted target, but set
`c_gamma=1.25`. At `a*m=0.08,0.04,0.02` independently resolve the candidate
recoil and all Pauli observables. No matter speed or charge is re-tuned. Default
acceptance requires a relative KN disagreement above `0.02` at every spacing
and a finest error at least `0.75` times the coarser one. The actual output, not
these thresholds, determines the measured failure.

For the same unequal speeds also run rest electron events with incident axial
and body-diagonal directions at two spacings. The scattering angle and energy
ratio are unchanged. The script requires their orientation spread to shrink
under refinement, while reporting their individual KN errors. An isotropic
continuum matter cone and an isotropic photon cone can have different speeds:
rest-frame isotropy alone is a false positive for joined Lorentz covariance.
Gauge Ward cancellation can likewise survive unequal speeds; it constrains the
consistent current and conservation law, not a common limiting speed.

## Lorentz spinor algebra as a consequence of selected hoppings

The separate algebra probe imports no gamma matrices. Extract the propagation
matrices `C_i` from the anti-Hermitian parts of the **selected actual** nearest
hopping coefficients and the mixing `B` from the onsite coefficient:

\[
C_i=\frac{ia}{v}(T_{i1}-T_{i1}^\dagger),\qquad
B=\frac O{m+9r/(2a)}.
\]

The chosen local orbital law implies
`{C_i,C_j}=2 delta_ij`, `{B,C_i}=0`, `B^2=1`. Therefore set

\[
\gamma^0=B,\qquad\gamma^i=BC_i,
\qquad \{\gamma^\mu,\gamma^\nu\}=2\eta^{\mu\nu}.
\]

The continuum equation from `h_0=v C.p+mB` at the selected common speed
`v=1` becomes `(gamma^mu p_mu-m)u=0`. The rotation and boost generators are

\[
J_i=-\frac i2 C_jC_k,\qquad K_i=\frac i2 C_i
\quad ((i,j,k)\text{ cyclic}),
\]

with `[J_i,J_j]=i epsilon_ijk J_k`, `[J_i,K_j]=i epsilon_ijk K_k` and
`[K_i,K_j]=-i epsilon_ijk J_k`. For the active boost convention above,

\[
S(\eta,n)=\exp(\eta C.n/2)=\exp(-i\eta K.n),
\quad \tanh\eta=|\beta|,
\]

\[
S\slashed P S^{-1}=\slashed{\Lambda P},
\qquad S^\dagger B S=B.
\]

The script reports Clifford, Lie-algebra, slash-covariance and metric-preservation
residuals, and boosts the band's actual rest positive eigenvectors to check the
continuum equation and normalizations `ubar u=2m`, `u^dagger u=2E`.
This is a derivation of the continuum **representation** from selected local
hoppings, not an independently assumed gamma comparator. Boosts are nonunitary
on a fixed four-component orbital norm; the covariant metric identity must not
be confused with microscopic Hilbert-space time evolution. The lattice symbol
is not covariant under these boosts, and no physical all-frame symmetry of the
full compact-rotor/matter Hilbert space is inferred.

## Remaining explanatory inputs

The selected hopping law, continuous time, quantum postulates, compact gauge
law, finite-sector preparation, mass and charge remain inputs. Algebra does not
explain why dynamics selects them. `e=0.30350` is a model normalization, not a
derivation of measured fully dressed fine structure; regulator-sensitive vacuum
feedback cannot be ignored. Matched photon/matter limiting speeds are explicitly
selected, not generated by this exercise. This finite tree-level event sweep
provides neither a universal all-frame theorem nor full microscopic physical
rotations, Coulomb-minimized interacting asymptotic states, thermodynamic or
unquenched gauge control, radiative corrections, or an equivalence theorem to a
continuous-time physical continuum. This scoped contribution does not authorize
campaign promotion, merge or closure.
