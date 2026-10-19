# Near-Dirac cutoff: a positive canonical/harmonic photon family in physical units

This pair owns physical/link units, true-well admission, paid photon/current
seams and the native canonical consumer. It consumes, **does not newly prove**,
`noncompact_wilson_photon_envelope.md` section7's non-relative beta-cone and
anchored two-source theorem. That theorem already requires `xi<mu/2`, not
relative Wilson `kappa<1`. The common governing plan is
`issue-222-synthesis-continuation-20261006`, pinned general
`89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe`,
work campaign, requirements outcome/verification. The campaign is not complete.

## 1. Applicability before selecting a photon family

The actual nearest-axis endpoint at fixed physical cutoff a is

    h_xx=(m+3r/a) Phi0,
    h_x,x+j=-(r Phi0+i v Phij) U_xj exp(i theta_xj)/(2a),
    m=.2, r=.12, v=.3, physical quartic stiffness=2/a.

Let `G` be the actual full ten-component Gram, `s=G00-G0s Gss^-1 Gs0`,
`u=sqrt(G00-s)`, and `beta` the temporal Clifford direction orthogonal to the
spatial plane. The relative spatial norm bound is

    kappa_v²=v² Tr(Gss)/(2 m r a s),
    kappa_total <= kappa_v + u/sqrt(s).

For the selected isotropic continuum Gram, `s=g0`, `TrGss=3gs`, so

    delta_W=m r a g0/(v² gs),
    kappa_v²=3/(2 delta_W), Gordon carrier g=2/(1+delta_W).

Consequently relative admission `kappa_v<1` implies `delta_W>3/2` and this
carrier comparator `g<.8`. It cannot be assumed on the physical near-Dirac
branch `delta_W -> 0`. Tilting replaces `g0` by the true Schur complement
in the bound and adds the separate tilt term; it does not preserve an
isotropic equality for a physical mechanical-spin g. The selected cutoff
comparator uses `bulk_cutoff_matching.selected_gram(a,grid)`, namely the
actual dimensionless mass `am`, not the a1 selected fields. No measured
physical half-spin or joined gyromagnetic observable is inferred from Gordon.

## 2. A source-derived universal true-well admission

This family does not borrow the supplier's declared a1 tighter box. In
`u_time=t/a`, write

    x0=am+r sum_j(1-cos theta_j), xj=v sin theta_j,
    F(G)=||G||F²-2 average sqrt(x^T G x).

The actual full-Gram uniform theorem, including **total** whole-heavy half
filling, is `bulk_global_gram.md`/`soft_budget.py`; its full-ten stationary
condition is `2G=<xx^T/R>`, `R=sqrt(x^TGx)`. For a shifted N-cubed grid,
N>=3, the exact trigonometric second moment is

    Q=<xx^T>=diag((am+3r)²+3r²/2, v²/2, v²/2, v²/2).

It is independent of the shift, hence covers all64 central torons and every
real shift, not only the identity well. Q is strictly positive. A PSD
boundary minimizer is impossible: in a null direction its quartic first
variation is zero whereas the sea first variation is strictly negative
(or negative infinite); the nonzero trigonometric polynomial cannot vanish
on the whole grid because Q>0. The origin is excluded by square-root versus
quadratic scaling. Thus the stationary minimizer is interior.

The existing whole-half/quartic bound gives

    ||G||F <= tstar=(2 sqrt(2) Cx)^(2/3),
    Cx=sqrt((am+6r)²+3v²), R<=sqrt(tstar) Cx.

It follows as a **Loewner inequality**, not just four diagonal estimates,

    G >= Q/(2 sqrt(tstar) Cx) =: D.

The Schur variational formula `s=min_z (1,z)^T G (1,z)` then gives
`s>=D00=:s_floor>0`. This is an explicit analytic all-well admission
`N_metric=3`. The continuum integral obeys the same moment identity and
argument. `G00,Gjj<=tstar`, `u<=sqrt(tstar-s_floor)` are conservative native
inputs. No numerical full-Gram mesh agreement is called interval proof.
This admission concerns the source-selected uniform wells; it does not by
itself prove an arbitrary nonuniform quantum field minimum or BO normal gap.

## 3. Reusing the cutoff-scaled anchored supplier, then paying all seams

All core variables below are dimensionless: `hhat=a hphys`,
`bhat=a a0phys`, `theta=a Aphys`, `u_time=t/a`. The supplier receives
`mhat=am`, the actual r/v, the just-derived s_floor/tstar and tilt-root bound.
Its beta mass is `Mhat=sqrt(s)[am+r L_theta/2]>=muhat=am sqrt(s_floor)`.
It supplies `alpha_site`, `gammahat`, cone radius R<1, point constant C,
positive anchored diameter moment Hhat and four separately paid terms:
triangle, native contact, odd-anchor first derivative and odd-anchor contact.
The current selected core is the source-marked Wilson-energy evaluator in
supplier section9, with explicit spatial size floor64 and its finite-volume
Wilson zero mode. The overall onset is the maximum of64 and all paid seams.
Its applicability is the weighted accretive beta structure for scalar U1
histories; **no relative Wilson series is used**.

The hatted bulk coefficients are

    CE0=16 Hhat/(e² gammahat²),
    CB0=4*64² Hhat sup_(r>=0)(1+r)^4 exp(-alpha_site r).

Here e in the numerical prefactor is Euler's constant; microscopic source
charge remains exactly one. The full source jets, in particular derivatives
of the Wilson mass and the anchor, are retained. Ward completion applies at
each local anchor in both source slots, not only after summing the determinant.

Keep the spatial and circle seam prefactor

    Rstar=2^20 Hhat exp(alpha_site/2)
          P12(alpha_site/4) P12(gammahat/4),
    P12(b)=sup_(x>=0)(1+x)^12 exp(-b x).

The remainder is `Rstar[exp(-alpha_site N/64)+exp(-gammahat That/64)]`.
The actual mass-only reference has source constants
`Lmass²=1+3r² tstar`, `cmass=r sqrt(tstar)`. Its **ordered**, noncommuting
finiteT remainder, with the same scalar interfaces and Haar holonomy, costs
`N^5[That Lmass²+cmass] exp(-muhat That)` in both field-strength coefficients.
It has not been dropped by subtracting a free sea.

Set `tauhat=max(T0hat,(alpha_site+1)/gammahat)`, `That>=tauhat N²`, and
`epsilon=max(CE0,CB0)/100`. The executable evaluates the log of

    N² Rstar/8 [exp(-alpha_site N/64)
                +exp(-gammahat tauhat N²/64)]
      +N^5[tauhat N² Lmass²+cmass] exp(-muhat tauhat N²).

Past the displayed monotonicity floor each term decreases with N and with
That. Doubling/bisection chooses a concrete integer N0 where the total is
<=epsilon, uniformly for **every N>=N0 and every That>=tauhat N²**. The
canonical/global-point error remains the actual Floquet expression
`4N³ log[(1+exp(-muhat That))/(1-exp(-muhat That))]`, tending to zero on
this exhaustion; the finite measure is never replaced by a Haar point.
Take `CE=CE0+epsilon`, `CB=CB0+epsilon` and any prescribed physical speed c>0:

    hE=16 max(CE,CB/c²,1), UEhat=1/(2hE), Khat=c² hE,
    rho=max(CE/hE,CB/Khat)<=1/16.

These are strictly positive, fixed **at this a**, not N-dependent. Conditional
all-background Hessian bounds imply two-sided transverse covariance ratios
`1/(1+rho),1/(1-rho)` relative to the same Maxwell form. Conditioning retains
the entire actual harmonic history/initial point/winding, global alpha and
centre. Nonzero-k conditional means vanish by spatial translation, so the
true mixture retains these covariance bounds. The logarithm of a canonical
integral has an additional score covariance: it is not conditional W''.
There is no Maxwell convexity claim in constant harmonic directions.

## 4. Exact physical volume, field, rotor and charge units

Define exact finite-cutoff physical fields by

    Ephys=(partial_t theta-d a0phys)/a,
    Bphys=C theta/a², dV=a³ per cell.

No continuum Taylor approximation is needed for these definitions. Then
`Ehat=a² Ephys`, `Bhat=a² Bphys`, `du_time=dt/a`, and

    integral du_time sum Ehat² = integral dt a³ sum Ephys²,
    integral du_time sum Bhat² = integral dt a³ sum Bphys².

Thus CE and CB computed **at mhat=am** are exactly the physical-volume
coefficients, not a1 coefficients. In raw site/physical-time notation,
`Hraw=Hhat/a`, `CEraw=a CE`, `CBraw=CB/a`,
`alpha_phys=alpha_site/a`, `gamma_phys=gammahat/a`.
These conversions agree with the supplier's raw hopping/source bounds.

For the law written `UEphys p_theta²+Kphys ||Ctheta||²/2`,

    UEphys=UEhat/a, Kphys=Khat/a,
    chiE=1/(2a UEphys)=hE, chiB=a Kphys=Khat,
    physical bare speed=a sqrt(2UEphys Kphys)=c.

`bulk_cutoff_matching` uses the convention `UE1 p_theta²/2`; hence its
`UE1phys=2UEphys`, `K1phys=Kphys`. This factor two is not a fitted charge.
The dimensionless bare field coupling is `g_bare²=1/(hE c)`. No exact
renormalized tensor, coupling or speed is supplied by covariance comparison,
and no finite `.0001` coefficient is admitted by this theorem. All constants
deteriorate as a->0; this is not a fixed-coupling UV construction.

Actual Peierls chain rules are

    j_theta,phys=j_theta,hat/a,
    c_theta theta,phys=c_theta theta,hat/a,
    j_A,phys=a j_theta,phys=j_theta,hat,
    c_A A,phys=a² c_theta theta,phys=a c_theta theta,hat.

The actual continuous first-order kernel obeys
`Kphys(t,s)=Khat(t/a,s/a)` **without** an extra factor a in its point value;
its inverse integral has dt=a du_time. The native consumer independently
constructs both physical and hatted endpoints, ordered evolution and kernels.

## 5. Current and finite-resolution packet capability

For the static whole-half negative sea only, native endpoint incidence gives

    Tr j_A²=2 sum_e [r²G00+v²Gjj] eta_e²,
    Var(dGamma j_A)<=Tr j_A²/2,
    chi_A<=Tr j_A²/(2 m sqrt(s)).

These are not interacting-ground variance bounds. For the **actual effective
zeroT canonical/harmonic state**, consume the instantaneous score incidence
construction from `noncompact_photon_packet_supplier.md` section3 with the
cutoff-specific point C/J/H, not the old a1 Cbar. At fixed cutoff it gives

    CWick=128 C² Jhat² *3 exp(4alpha_site)
          [1+2/expm1(alpha_site)]³,
    Cscore_bulk=2(8Hhat)² UEhat
      [(2sqrt(3)/c)(128 P2(alpha_site))²+1/(e gammahat)].

This uses the instantaneous slot unchanged and rooted Ward filling only in
its gradient slot, avoiding the illegal delta(t) H1 source. The native contact
belongs in the score derivative, not in equal-time J². For zeroT at each N,
mass-only thermal and circle terms vanish with their previously paid finiteT
bounds. The remaining spatial score seam is paid explicitly by

    Cscore_seam=UEhat [N0/(4c)+N0²/8]
                  Rstar² exp(-2 alpha_site N0/64).

Its polynomial-exponential expression decreases for all N>=N0 by the same
monotonicity floor. Hence `CJhat=CWick+(Cscore_bulk+Cscore_seam)/(1-rho)`.
This capability inherits the supplier's instantaneous incidence derivation;
the finite identity run does not independently certify the interacting
variance or microscopic Gibbs measure. The all-background point/contact and
paid positive-family result are separate from this further current capability.

A normalized physical-volume transverse mode is
`F=sqrt(a) A_theta,f`, `P_F=p_theta,f/sqrt(a)`, `[F,P_F]=i`.
Its kinetic coefficient is UEhat and physical frequency is
`omega_phys=c qhat/a`. Its native current is
`J_F=j_theta,hat/a^(3/2)`, so the physical-volume mode bound is
`CJ,F=CJhat/a³`, **not** CJhat or CJhat/a². The latter is the link-coordinate
physical-current bound. At a fixed nonzero physical frequency band,

    ||(Hphys-E0-omega_phys) a†_f Omega||²
       <=UEhat CJhat/(a³ omega_phys),
    annihilation debt <=UEhat CJhat/(a³ omega_min,phys³),
    finite-time one-photon defect <=|tphys|
           sqrt(UEhat CJhat/(a³ omega_min,phys)).

The creation identity and plus current sign are those of the actual packet
supplier. These conservative bounds need not be small. They unlock an
honestly unit-normalized physical packet/charge join at a near-Dirac cutoff,
not a sharp pole or guaranteed accurate scattering experiment.

## 6. Whole canonical/harmonic and full-field realization

The physical positive measure is the source-owned centered determinant times
positive oscillator/harmonic heat kernels and the true Coulomb Hubbard weight,
with exact full2V Haar coefficient, background2 and scalar site-U1 temporal
interfaces. It includes all harmonic windings and actual64 well offsets. No
generic temporal SU4 determinant positivity is assumed. Full-field Haar is
integrated through the actual electromagnetic-independent isometry W:
all60V fields, whole-heavy Slater line, full light/reference fibre and every
colour Gauss equation remain. `W* O_photon W=O_photon`, including closed
photon/Coulomb forms, is an exact prepared-state identity.

At **fixed a,N**, choose finite Lambda(N,a) using the existing residual/complement
asymptotics so that `d_N<=N^-4/(1+log N)`, normal complement lies above the
actual witness and the symmetric real harmonic Bloch packet's excess is
<=N^-4. The lifted full-field near-infimum preparations have the same photon
expectations; bounded Weyl correlation error is <=2|tphys|d_N. Truncating linear
sources using the new subGaussian coefficient pays their unbounded tails.
The new family can therefore be realized on the legal fixed-a joint
state/dynamics trajectory, not on an invented numerical Lambda, uniform-a BO
bound or asserted exact finite-Lambda microscopic Gibbs marginal. The existing
existence-only normal onset remains distinct from N_metric=3 and N0 above.

## 7. Actual consumer, evidence and constructive successor

The CLI uses a=.001 by default. Its genuine periodic native3D size3 consumer
retains all108 modes and all54 occupations, checks **all64 full-ten** metrics
against the analytic Loewner admission, independently builds hphys/hhat,
keeps arbitrary noncommuting real spatial sources and explicit nonconstant
harmonic histories, and selects the full canonical coefficient with109 exact
roots. Direct determinants independently check positive centered factors.
It evaluates the full canonical source Hessian as conditional Hessian **plus
score covariance**, retains contact/bubble separately and exercises the
pure-gauge Ward boundary. Deleting contact fails that same boundary. Missing
`theta=a A` chain rules, borrowing a1 energy units and a kappa-only admission
are adverse constructions. This finite probe is not claimed to reach the
analytic N0/That exhaustion or simulate a quantum photon ground.

Reproduction uses the scientific interpreter and an explicitly selected
receipt path:

    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      /home/administrator/substrate-framework/.venv/bin/python \
      noncompact_dirac_photon_envelope.py --a .001 --size 3 \
      --output /tmp/issue222-near-dirac-positive-photon-receipt.json

Two concrete successors are (A) select a shared physical a/c/frequency band
and feed this family's exact rotor/source/current metrics to the existing
full-reference charged source and photon receiver/recoil consumer, or
(B) derive anisotropic/spin-resolved current-envelope improvements from the
actual selected continuum Gram to reduce the very loose universal bound.
Select A: the missing relative-convergence hypothesis and units are no longer
obstacles to an honest same-cutoff join. Its conservative packet debt remains
a real decision input; neither this join nor Gordon substitutes for physical
mechanical half-spin, a renormalized scattering match or Lorentz closure.

## 8. Historical wm29 cone-family receipt and language tooling

The first native a=.001/size3/grid24 run exited naturally with all checks
passing in3.36s and all source bindings unchanged. It admitted the analytic
Schur floor.06268305568 and evaluated the concrete fixed-cutoff family
UEphys=4.180482673624184e-99, Kphys=1.1960341401595507e102,
N0=2134396830, Tphys>=61.44916805 N², rho=1/16.
These are floating formula evaluations, not outward-rounded constants.

The selected isotropic comparator gave deltaW=.0007411353191,
kappa_v²=2023.9218957, Gordon carrier g=1.9985188271. The actual tilted
centre123 native kappa was45.46403155 while every one of109 centered
full-canonical Haar factors remained positive (phase error2.41e-14).
The complete canonical Ward Hessian was-1.60e-18; dropping contact left
.007110868281. Physical/hatted endpoint and point-kernel errors were
1.57e-16/6.75e-17; missing the source chain rule failed with error169.47.
The actual static JA variance.01047967413 was below the native HS
bound.01438890710, separately from the interacting current capability.

For the analytic effective zeroT capability, CJhat=5.25531935035e72.
Its correctly volume-normalized physical packet bound at frequencies
.01–.02 and observation time1 gave annihilation debt2.19697715e-14,
energy/time defect1.48222034e-9 and prepared energy upper.0200000014823.
These small bounds come from an extremely weak but strictly positive
cutoff-dependent rotor family. They are not a simulation of that coupled
ground state, a finite weak-band coefficient claim, or a fixed-g2 UV limit.

The first receipt is `/tmp/issue222-near-dirac-positive-photon-receipt.json`.
The final source-bound receipt after scoped type repair and this evidence
addition is `/tmp/issue222-near-dirac-positive-photon-final.json`; the stable
publication packet is `local://issue222-near-dirac-positive-photon-state-packet.json`.
No numerical result is inferred from a missing or failed receipt.

OMP has no mounted LSP device in this session. Actual Pyright stdio
initialize/workspace scientific-interpreter configuration, document symbols,
definition of selected_gram and the imported primitive_cone_envelope, plus
pull diagnostics were exercised at `/home/administrator/substrate-222`
(pyproject root). A positive ast-grep Python function query and the actual
primitive-cone call query both matched. Scoped optional-frame/receipt typing
diagnostics were repaired without altering the physics or other workers'
files. No project-wide build, test suite, linter or formatter was run.

## 9. Selected stronger-coupling Wilson-energy physical successor

The CLI now consumes `primitive_wilson_energy_envelope(size_floor=64,...)`
from the frozen source-marked supplier. It cleanly replaces the old core
selection in this pair; the historical wm29 source and receipts above remain
revision-bound evidence, not current default coefficients. It retains exactly
the same true metric admission, physical units, canonical/harmonic integrations,
current/contact sources and mass-only thermal remainder.

The finite-volume source hypothesis is explicit:

    CM <=3*4^(2/3)*pi/(8r sqrt(s_floor))+4/(64³ muhat).

The zero-mode term is not removed by treating a finite torus as infinite.
Two genuine native source/target transport windows and the two-sided
mass-energy inverse supply the improved point constant. Their cyclic order
retains the AP/holonomy jump. The reusable core updates the four anchored
terms with C powers3,2,2,1; this family recomputes CE/CB, logRstar, the actual
mass/spatial/time seam onset, physical rotor coefficients and current bounds
from those returned moments. It does not substitute the supplier's finiteN5
consumer metric for the universal admitted box.

At the historical SAME fraction.35/metric inputs, the source supplier measured
the new scalar envelope C=3651.610124830094 instead of221998188.7545323,
H=4.6999362055030357e58 instead of1.056056244119568e73, an H gain
224695867761579.12. This is a supplier scalar-interface evaluation, distinct
from the new physical-family fraction optimization and our own canonical
execution. The unchanged old native identity consumer need not be rerun to
prove Sharp's new source-window identity; the new physical CLI does execute
its integrated family and full canonical/unit scenario after stable edits.

The decision payoff is **stronger admitted coupling**, not automatically
better normalized packet accuracy. At unchanged alpha/gamma and metric box,
if H shrinks byF, the matched CE/CB and epsilon shrink byF, UEhat grows byF,
and Khat shrinks byF. The bulk score-current bound shrinks byF, so the matched
product UEhat*CJscore is invariant. The paid spatial-score product has the same
scaling because Rstar and epsilon change together; H cancels from the dominant
spatial onset equation. The Wick contribution has its separate C² scaling.
Optimizing the allowed xi fractions can also change alpha/gamma and must be
reported separately, rather than attributed to a magically improved kernel.

At the SAME **old rotor coefficients**, in contrast, the new instantaneous
score bound shrinks byF². This allows a genuinely smaller packet score debt
without changing the physical Hamiltonian; it is not an additional strong
coupling admission. The new CLI's default is the retuned stronger-coupling
branch, with actual packet bounds reported without an improvement claim.
Static free-sea variance and actual Ward identities are unchanged by this
envelope choice.

Reproduce this successor using the same scientific command, selecting output
`/tmp/issue222-near-dirac-wilson-energy-final.json`. Its publication topic is
`issue222-near-dirac-wilson-energy-physical-family`. The actual finite3D
canonical probe remains below the analytic size/time exhaustion, including
the explicit64 source floor; it does not simulate the new coupled photon
ground or prove a finite weak-band coefficient.

Two concrete successor constructions are (A) choose this stronger-coupling
physical family for a same-cutoff full-reference charged/photon two-receiver
recoil join with its freshly paid packet metric, or (B) keep the historical
rotor Hamiltonian and use the new source-marked envelope to improve the
instantaneous score/finite-resolution detector accuracy at fixed coupling.
Select A for the current coupling match; B remains an explicit alternative,
not a claim that retuning achieves both gains simultaneously.
