# Exact Euler helical background and localized material-index pressure response

## 1. One physical state and the changed representation

Work on `R^3` with density `rho>0` and unforced three-dimensional incompressible Euler:

    ∂_t u+(u·∇)u=−∇p/rho,       div u=0,
    −Δp=rho ∂_i∂_j(u_i u_j).                              (1)

The pressure is the whole-space Hodge/Poisson pressure. The microscopic kinetic action with volume-preserving parcel maps is unchanged. A bounded spatially periodic background has infinite *total* energy; the localized perturbations below have finite initial kinetic **excess** over that specified background. No periodic box is asserted to be an isolated particle.

Put

    U_0=(sin z,cos z,0),
    U_side=(0,cos x,−sin x)+(−sin y,0,cos y),
    U_e=U_0+e U_side,             0<|e|<1/2.             (2)

Each summand is divergence free and has `curl U=U`, hence the complete, genuinely three-dimensional field satisfies

    curl U_e=U_e,       (U_e·∇)U_e=∇(|U_e|²/2),
    p_e=−rho |U_e|²/2.                                   (3)

This is an **exact stationary Euler medium**, not a frozen-tangle constitutive ansatz. Each of the two summands in `U_side` has pointwise unit norm, so `|U_side|<=2` and `|U_e|>=1−2|e|>0`: it has no vorticity zeros before the localized modification. Its wavelength and amplitudes remain freely chosen; exactness does not imply all-mode stability or a relativistic wave band.

For a completed calculation use first `e=0`. Let `lambda>=1`, `chi=exp(−lambda |X|²)`, and define divergence-free polynomial velocities

    v_0=(−z/2,0,x/2),          w=(5yz,−4xz,−xy)/3,
    curl v_0=−e_y,             curl w=S X,
    S=diag(1,2,−3).                                     (4)

The polynomial vector potentials

    A_0=−X×v_0/3,
    A_w=−X×w/4,
    V_0=curl(chi A_0),        W=curl(chi A_w)            (5)

are smooth and rapidly decaying after the indicated curls. The factors `1/3,1/4` follow `curl[−X×F_m/(m+2)]=F_m` for a homogeneous solenoidal polynomial of degree `m`; do not replace (5) with `chi v_0` or `chi w`, which would not be divergence free. For `delta>0`, two exact smooth full-Euler initial states are

    u_±(0)=U_0+V_0±delta W.                               (6)

Their initial excesses `rho/2∫(|u_±|²−|U_0|²)dx` are finite. A standard local smooth solution exists in a suitable `H^s` perturbation class around the bounded smooth background; all claims below are on the common smooth lifespan. Propagation of a weighted absolute excess and nonlinear orbital stability are **not** claimed. For each fixed `0<|e|<1/2`, the genuinely 3D variant `u_±^(e)(0)=U_e+V_0±delta W` is also exact smooth divergence-free Euler initial data with finite **initial** excess relative to `U_e`. Sections 2–3 show a zero-index pair for sufficiently large `delta` and the same nonzero pressure octupole for every such `e`; this does not identify or stabilize a particle.

## 2. Exact material topological labels, and what they do not prove

Near the origin `V_0=v_0+O(|X|³)` and `W=w+O(|X|⁴)`, so the full vorticity for (6) obeys

    omega_±(0)=0,
    D omega_±(0)=e_x⊗e_z ± delta S,
    det D omega_±(0)=∓6 delta³.                           (7)

The two nondegenerate isolated zeros have local Brouwer indices `q_+=−1` and `q_−=+1`. For each fixed `|e|<1/2` (including `e=1/10`), the same two signs also give opposite-index zero germs if `delta` is sufficiently large. Indeed set `g_e=U_e+curl V_0`, `f=curl W` and `tau=1/delta`. Near the origin the full vorticity equation is `±f(X)+tau g_e(X)=0`, with `f(0)=0` and `Df(0)=S` invertible. The implicit-function theorem yields distinct roots

    X_±^(e)(tau)=∓tau S^(-1)g_e(0)+O_e,lambda(tau²)
                  =∓tau(0,e/2,−e/3)+O_e,lambda(tau²),     (8)

where `g_e(0)=(0,e,e)`. Their vorticity Jacobian is `±delta S+O_e,lambda(1)` and its determinant has leading sign `∓6delta³`, hence the same `q_±=∓1`. For `e=0` the roots are exactly the origin for every `delta>0`; for fixed nonzero `e` this argument concerns **large**, not vanishing, `delta`. A small `C²` 3D perturbation of either initial velocity preserves a unique nearby nondegenerate zero. This is local identity robustness, **not** a nonlinear invariant particle-shape neighborhood.

For the actual smooth Euler flow map `eta_t`, Cauchy's vorticity formula at a zero `a_*` yields

    omega(t,eta_t(a_*))=D eta_t(a_*) omega(0,a_*)=0,
    D omega(t,eta_t(a_*))
       =D eta_t(a_*) D omega(0,a_*) D eta_t(a_*)^(-1).   (9)

Thus its index is transported exactly on the smooth lifespan, for unrestricted 3D evolution. The distributional **kinematic** current of a selected zero,

    j^0=q delta_(eta_t(a_*)),    j=q eta_dot_t(a_*) delta_(eta_t(a_*)),
    ∂_t j^0+div j=0,                                   (10)

is not an electric or weak current. Because `U_e` is nonzero everywhere and the perturbations decay, the degree on a sufficiently large sphere remains zero: the marked `±1` requires compensating index elsewhere. The amplitude `delta` is continuously selectable, so a conserved integer alone cannot fix a universal coupling or action unit. The raw vorticity-zero index is odd under both Euler parity and time reversal, whereas electric charge is even under each; freezing the background's handedness would conceal, not repair, that mismatch.

## 3. A nonzero exterior *physical* pressure moment on these same states

Set `b=U_e+V_0` for the variant above (`e=0` in (6)). The terms quadratic in `W` cancel across signs, so the exact pressure **difference**, normalized to decay at infinity, is

    −Δ Delta p=2 rho delta ∂_i∂_j(b_i W_j+W_i b_j),
    Delta p=2 rho delta G*∂_i∂_j(b_i W_j+W_i b_j),
    G(X)=1/(4 pi |X|).                                  (11)

The source is rapidly decaying, not an imposed charge. Let `H=x²y−y³/3` (`ΔH=0`) and define the stress harmonic moment

    Q_H=∫(b_iW_j+W_i b_j) ∂_i∂_j H dX
       =2∫ b_i W_j ∂_i∂_j H dX.                         (12)

This is an actual whole-space pressure multipole coefficient: `∫H(−Δ Delta p)=2 rho delta Q_H`. Gaussian moments from (4)–(5), with `∫x^(2m)e^(−lambda x²)dx=sqrt(pi)(2m−1)!!/[2^m lambda^(m+1/2)]`, give **exactly**

    Q_H[U_0,W]
      =−pi^(3/2)(6 lambda−1)e^(−1/(4 lambda))/(24 lambda^(9/2)),
    Q_H[U_side,W]=0,
    Q_H[V_0,W]
      =−13 sqrt(2) pi^(3/2)/(4608 lambda^(7/2)),
    Q_H[U_e+V_0,W]=Q_H[U_0,W]+Q_H[V_0,W] < 0
      for every |e|<1/2, lambda>=1.                     (13)

For the first row one can integrate `W=curl(chi A_w)` by parts, use `∂_z U_{0x}=cos z`, and eliminate the `∂_z U_{0y}=−sin z` term by its odd `x,y` integrand. The remaining Gaussian integrand is `−chi cos z(2x²y²+5y²z²−4x²z²)/3`, giving the displayed row. The compact-cross row is a finite polynomial Gaussian integral; the accompanying oracle exposes its polynomial and exact sum. At `lambda=25`, (13) gives `Q_H=−1.78081100143365e−5`; an independently sampled three-dimensional quadrature at 60³/90³ nodes gave `−1.78079153513e−5` / `−1.78079111956e−5` on a finite `4/sqrt(lambda)` half-box, where the remaining Gaussian tails explain the slight shortfall. This quadrature is a check, **not** the proof or a rigorously bounded tail estimate.

The new side-wave row vanishes **exactly**, not by small-`e` continuity: writing `W=chi P_w`, the polynomial-trigonometric density `2 U_side^T (Hess H) P_w` is

    −(2z/3)(4x² sin y + 9xy cos x − 5y² sin y)
      ·(lambda |X|² − 2),

odd in `z` against the even Gaussian. Thus the physical octupole coefficient and its sign in (13) persist even on the fixed `e=1/10` background supporting the indexed germs of (8).

The stress zeroth moment vanishes: `V_0` is odd under inversion, `W` even; `∫U_{0x} W_j=0` by inversion and `∫U_{0y}W_j=0` by curl integration and odd `x` or `y` (the `j=z` row vanishes directly). Every entry of `∫U_side_i W_j` is likewise odd in at least one of `x,y,z`, so no side-wave term restores a pressure quadrupole. The nonzero harmonic cubic moment (13) forces a nonzero angular `R^(−4)` pressure octupole and an `R^(−5)` material-acceleration signal at some sufficiently distant direction for the **same** fixed nonzero `e`; it is neither a Coulomb pressure monopole nor an inverse-square force. At a fixed exterior point each state's material acceleration is `D_t u_±^(e)=−grad p_±^(e)/rho`, so their difference is exactly `−grad Delta p/rho`. If instead comparing Eulerian time derivatives, `∂_t(u_+^(e)−u_-^(e))=−grad Delta p/rho−[(u_+^(e)·grad)u_+^(e)−(u_-^(e)·grad)u_-^(e)]`; only this latter observable contains the convective difference. These results concern the common smooth lifespan, not permanent localization or finite later-time energy excess.

## 4. Exposing controls and next construction

At `e=0`, hold `lambda` fixed and let `delta→0⁺`: each index in (7) remains exactly `∓1` while `Delta p=O(delta)` at every fixed detector and its far multipoles vanish linearly. At a fixed nonzero `e`, the large-`delta` roots (8) persist through a **local open interval** of `delta` by nondegeneracy, while their same-state pressure moment (13) changes continuously and linearly with `delta` at fixed index. Do not send `delta→0` on this fixed-`e` branch without proving the zeros survive; the small-amplitude countercase uses exactly `e=0`. These adverse transitions refute electric-charge identification from the index/pressure pair. The compensating zeros and P/T mismatch independently prevent that identification. At finite `delta`, changing a remote divergence-free perturbation without changing the local index can change the pressure moments: the full material state, not the index alone, determines coupling.

The **positive earned input** is narrower: exact stationary non-ring 3D Euler background, finite-initial-excess localized topological-index seeds, exact all-3D material transport of their identity labels, and a derived nonzero whole-space exterior mechanical response on that same family. No return, invariant nonlinear carrier tube, isotropic reciprocal `1/d` energy, scale selection, stable photon cone, physical amplitudes/exchange, spin-1/2, electric/magnetic or weak current, neutrino mixing, independent particle prediction, P2–P7 completion, claim promotion or release follows. A new design must test the *full-pressure linearized and nonlinear 3D background/defect stability* and build a dynamical collective mediator whose sign-coupling survives the fixed-index/vanishing-response control. A symmetry-only or frozen-elastic replacement does not pass this test.
