# Factorizing one Euler flow into two: a conditional obstruction and exact survivors

## Scope and decision

Fix density `rho>0`, either the flat three-torus `T^3=(R/2pi Z)^3` or `R^3` with smooth finite-energy initial velocity. The *only* microscopic dynamics is incompressible Euler,

    div u=0,       partial_t u+(u·grad)u=-grad p/rho,                 (1)

with periodic mean-zero pressure, or decaying whole-space pressure. In material coordinates a volume-preserving flow `Phi_t` has `Phi_tt=-(grad p/rho)∘Phi`. Its constrained kinetic action is `S[Phi]=(rho/2) integral dt integral |Phi_t(A)|^2 dA`, with incompressibility imposed on `Phi`; no cell stiffness, second density, or second physical pressure is assumed.

**Theorem (bounded, not universal).** Given *arbitrary, independently selected* exact Euler flow maps `eta_t`, `chi_t` initially equal to the identity, the composition `Phi_t=eta_t∘chi_t` need not extremize this one action. On `T^3` there are two global smooth Euler factors whose composition fails (1) at **every real time**; there are also smooth compactly supported finite-energy initial data on `R^3` giving two genuine local Euler factors whose composition fails (1) at the initial time and on a positive-time interval. Conversely a locked Beltrami family gives exact Euler factors with a nonconstant cross pressure and an exact Euler composition for **all time**. More strongly, a different spectral Beltrami family yields two uniquely projected and independently variable Eulerian velocity sectors, with nonzero cross pressure, whose sum solves the single Euler equation for **all time**; their flow-map composition is not the physical flow. Thus neither a universal impossibility of two exact Euler factors nor an automatic independently resolved material-cell/coarse-center Euler reduction follows.

## What the action actually sees

Let `eta` and `chi` be volume-preserving diffeomorphisms and define

    v=eta_t∘eta^{-1},     w=chi_t∘chi^{-1},
    z=eta_*w = Deta(eta^{-1}x) w(eta^{-1}x),
    u=Phi_t∘Phi^{-1}=v+z.                                         (2)

Both `v,z` are solenoidal, but `z` is generally *not* a Euclidean Euler solution even when `w` is: its inherited metric in eta's reference coordinates is `G_eta=(Deta)^T Deta`. In particular the one physical action has kinetic term

    K=(rho/2) integral |v+z|^2 dx
     =(rho/2) integral (|v|^2+2v·z+|z|^2) dx,                   (3)
    integral |z|^2 dx=integral w·G_eta w da.

The last term is generally not `integral |w|^2`: e.g. `eta(a)=(a_x+t sin a_y,a_y,a_z)` and `w(a)=(0,sin a_x,0)` give `|Deta w|^2=(1+t^2 cos^2 a_y)sin^2 a_x`. Splitting (3) into two independent Euclidean kinetic actions drops both this deformation and the mixed term.

At fixed eta the instantaneous kinetic quadratic form on independently varied solenoidal velocities has Hessian

    delta^2 K[(delta v,delta w)]
      =rho integral |delta v+eta_*delta w|^2 dx.                (4)

For every smooth solenoidal `h`, `(-eta_*h,h)` is a null direction. This is an *infinite-dimensional relative-label gauge*, not a proof that (1) lacks kinetic energy: for any smooth volume-preserving path `g_t`,

    (eta,chi) -> (eta∘g^{-1},g∘chi)                             (5)

leaves `Phi`, its action, physical pressure, and velocity exactly unchanged. Each factor's volume constraint introduces no additional microscopic multiplier: `det D(eta∘chi)=(det Deta∘chi)det Dchi=1`; independently assigning pressures to both factors is an extra constrained-model choice, not a derivation from (1). Gauge-fixing cannot manufacture a second observable Euler equation from the null directions.

At a time `t0` when `eta(t0)=chi(t0)=id`, suppose each factor is an Euler trajectory with initial velocities `v,w` and respective auxiliary pressures `p_v,p_w`. Twice differentiating the composition at **fixed material label** gives

    Phi_tt(t0)=-grad(p_v+p_w)/rho+2(Dv)w.                     (6)

Hence a necessary condition that the composition itself be one-action Euler is

    curl[(Dv)w]=0.                                               (7)

On the torus this is also an *instantaneous* sufficient condition: `integral (Dv)w dx=0` by `div w=0`, so a curl-free term is a periodic gradient; volume preservation of `Phi` then fixes its scalar pressure through (1). This is only a jet condition at `t0`, not a positive-time invariant-manifold theorem. On `R^3`, (7) remains necessary and the decaying-gradient/regularity conditions must additionally hold for sufficiency. At other times the factors need not be identities, so (7) is not claimed as a general positive-time criterion.

Independently of factorization, if two solenoidal Eulerian fields `v,z` are added, the *actual* pressure has the exact decomposition, modulo a constant,

    p[u]=p[v]+p[z]+p_cross,
    -Delta p_cross=2rho tr(Dv Dz),                              (8)

since `-Delta p[u]=rho tr[(Du)^2]`. The cross momentum flux is the coefficient-fixed tensor `rho(v⊗z+z⊗v)` in `rho div(u⊗u)`; it is not a new constitutive stress or independently conserved cell momentum. Its flux divergence is `rho[(v·grad)z+(z·grad)v]`, whose gradient part is absorbed in (8). In the map factorization `z=eta_*w`, not generally `w`, and its evolution has deformation terms. Thus (8) by itself is not a pair of autonomous Euler equations.

## Explicit adverse example: global and every positive time

Take periodic coordinates `(X,Y,Z)` and the two time-global flow maps

    eta_t(X,Y,Z)=(X+t sin Y,Y,Z),
    chi_t(A,B,C)=(A,B+t sin A,C).                              (9)

Each separately solves *full* Euler: its Eulerian velocity is respectively `(sin y,0,0)` or `(0,sin x,0)`, both stationary, divergence free, with constant pressure. Both preserve volume for all real `t`. Nevertheless the composition is

    Phi_t(A,B,C)=(A+t sin(B+t sin A), B+t sin A, C).            (10)

Write the Eulerian point as `(x,y,z)` and `A=x-t sin y` (periodic equalities). Its material acceleration, evaluated at that point, is exactly

    a(t,x,y,z)=Phi_tt∘Phi^{-1}
        =(2 sin A cos y-t sin^2 A sin y,0,0).                  (11)

At `(x,y)=(pi/2+t,pi/2)` we have `A=pi/2` and `partial_y A|_x=-t cos y=0`; therefore

    (curl a)_z=-partial_y a_x=2                                  (12)

for **every** `t in R`. Since physical pressure accelerations are gradients, (10) is not a solution of (1) on any positive-time interval, however the two factor trajectories are continued. Direct energy corroboration, not a substitute for (12): volume preservation and trigonometric averages give

    K[Phi_t]=rho |T^3| (1/2+t^2/8),                             (13)

which violates the constant kinetic energy of any smooth periodic Euler solution for `t!=0`. The single action therefore does not produce both shear-factor Euler equations as an unconstrained coupled evolution.

At `t=0`, `u=(sin y,sin x,0)`. Its physical Poisson pressure is `p=rho cos x cos y` (zero mean), whereas each separate shear has `p_v=p_w=0`. Thus `p_cross=rho cos x cos y` and the *true* initial Euler acceleration is `(sin x cos y,cos x sin y,0)`, while (6) gives `(2 sin x cos y,0,0)`. Their difference has nonzero curl. The cross-pressure term is a measurable full-field constraint, not an optional stress adjustment.

The same obstruction occurs for **finite-energy whole-space** data, not only periodic shears. Let `q=(pi/2,pi/2,0)`, choose a smooth compactly supported cutoff `theta` equal to one near `q`, and define

    v0=curl(0,0,-theta cos y),      w0=curl(0,0,theta cos x).  (14)

These are smooth compactly supported divergence-free fields. Near `q` they equal `(sin y,0,0)` and `(0,sin x,0)` respectively. Standard local classical 3-D Euler well-posedness gives actual decaying-pressure Euler flows `eta,chi`, each beginning at identity, on a common interval `(-T,T)`; no global 3-D regularity is invoked. At `q`, formula (6) gives `curl(Phi_tt∘Phi^{-1})(0,q)=2 e_z`, because both factor pressure contributions have zero curl. Smooth dependence of the trajectories and their spatial derivatives implies a `delta in (0,T)` with a nonzero curl at the transported nearby point for all `|t|<delta`. Therefore the factor composition fails the single Euler equation at `t=0` and for `0<t<delta`. The **numerical value** of each whole-space pressure is nonlocal and not claimed to equal the torus pressure; the local curl argument does not make that false assumption.

## Materially different exact survivor: a locked Beltrami leaf

A restriction, rather than arbitrary superposition, can pass (7) for all time. On `T^3` take the nonconstant ABC field

    F=(sin z+cos y, sin x+cos z, sin y+cos x),
    div F=0,       curl F=F,
    (F·grad)F=grad(|F|^2/2),       average |F|^2=3.          (15)

For any *independently specified real constants* `a,b`, put `eta_t=Flow_F(at)` and `chi_t=Flow_F(bt)`. Each is an exact stationary Euler flow, with pressures

    p_a=-rho a^2(|F|^2-3)/2,     p_b=-rho b^2(|F|^2-3)/2.  (16)

They commute, and `Phi_t=Flow_F((a+b)t)` is **also** exact one-action Euler for all real time. Its single physical pressure is

    p_{a+b}=-rho(a+b)^2(|F|^2-3)/2
           =p_a+p_b-rho ab(|F|^2-3).                         (17)

Thus the cross pressure and mixed momentum flux are genuinely nonconstant for `ab!=0`, have no fitted coefficient, and agree with (8). This directly falsifies any proposed universal no-go saying *no* nonzero two-Euler factorization can coexist with the one-action Euler equation. However the result is *one-dimensional in physical flow*: every `(a,b)` with the same `a+b` gives precisely the same `Phi,u,p,K`. Taking `a=-b!=0` produces two individually nonzero Euler factors, separate nonconstant pressures, and the **identically resting** physical fluid. Their pressure cross term cancels the two auxiliary pressures exactly. Consequently the two factors cannot be separately interpreted as independently observable coarse/internal Euler media, nor does (15) define a finite material-cell center law; its nonconstant field is periodic rather than finite-energy isolated on `R^3`.

## Materially different survivor: two distinguishable Eulerian sectors, not two factor maps

The gauge defect in (5) is specific to interpreting factors as observable fields. A genuinely different construction uses a *fixed physical Fourier projection* instead. On `T^3`, let

    E=(sin z,cos z,0),           J=(0,sin x,cos x),
    curl E=E,                    curl J=J,
    average |E|^2=average |J|^2=1,    average E·J=0.         (18)

For any two independent real amplitudes `a,b`, define Eulerian components `v=aE`, `z=bJ`, and actual velocity `u=v+z`. The two spectral supports differ, so the amplitudes are unambiguously *physical observations*, `a=average u·E`, `b=average u·J`, within this specified two-mode family; they are not alterable by the relative-label gauge (5). Each component separately solves stationary incompressible Euler with constant pressure. Since `curl u=u`, the full field is also a stationary, globally smooth *exact* Euler solution, with

    |u|^2=a^2+b^2+2ab sin x cos z,
    (u·grad)u=grad(ab sin x cos z),
    p[u]=-rho ab sin x cos z,      p[v]=p[z]=0.             (19)

The nonconstant pressure is exactly (8), arising from the mixed flux `rho ab(E⊗J+J⊗E)` of the one action. The action energy is `rho |T^3|(a^2+b^2)/2`: the cross energy integrates to zero although the *local* cross pressure and stress do not. Distinct `a,b` are independent initial-state coordinates of a **two-dimensional invariant stationary family**, not two open sets of independently prescribed Euler initial fields. There is no dynamical transfer between these fixed-amplitude modes. Crucially `eta=Flow_E(at)` and `chi=Flow_J(bt)` are **not** a representation of the physical flow for generic `ab!=0`: at the identity `(DE)J=cos x(cos z,-sin z,0)` has nonzero curl, contradicting (7). Thus (18) evades the composition obstruction by changing representation, not by hiding its missing pressure. This is a positive exact two-sector benchmark, but the two delocalized helical modes are neither a material-cell internal continuum nor an autonomous field of material-cell centroids; no isolated finite-energy `R^3` carrier follows.

## Remaining escape and boundary of inference

The theorem excludes the *generic independent composition* ansatz, not every one-action two-sector reduction. Even the two independently observable stationary sectors (18)–(19) do not solve the material-cell problem: a proposed cell construction must specify a physical invariant/controlled family and a **gauge-invariant** second observable (not just the `eta,chi` labels), derive its pressure transport and mixed stress from the one physical action, and show a nonzero autonomous internal-Euler and coarse-Euler decoupled limit with controlled positive-time perturbations. A shared-flow vorticity-packet split can supply a bona fide action-derived cross energy/stress, but its packets are advected by the **same** full velocity and do not thereby become two independent Euler trajectories or determine an autonomous material-cell centroid; the distant-swirl pressure test in `../0001/material-balances.md` still applies. Retaining the unresolved full field/memory as in `../0009/retained-euler-memory.md` is another nonlocal escape, not pressure autonomy for free. Nothing here excludes an e-dependent/high-gradient microstructure, a constrained semidirect-product sector with an observable extra invariant, or a proved family-specific pressure closure. No stable electron, neutrino, spinor, or interaction-particle claim follows.

## Reproducibility

The algebraic checks used the repository `.venv/bin/python` and SymPy 1.14: verified `-Delta(rho cos x cos y)=rho tr[D(v+w)^2]`, the factor-acceleration curl `2 sin x sin y` at `t=0`, (12) identically in `t`, (13) by exact two-torus trigonometric integration, `curl F=F`, `(F·grad)F=grad(|F|^2/2)`, and the separate eigenfield/pressure/curl identities (18)–(19). These are checks of the explicit symbolic identities, not a numerical stability or near-family proof. The R3 finite-energy extension uses only locality of the displayed jets and classical local Euler existence.

The runnable [factorization oracle](verify_factorization.py) and [validation receipt](validation.md) make the symbolic every-time curl check, periodic energy sample, locked ABC survivor, and distinct helical survivor independently replayable. Earlier ad hoc checks also covered the explicit physical Poisson source at `t=0`; those particular checks were not saved as a script, so the paper calculation (8) remains the reproducible source for that pressure formula.
