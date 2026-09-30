# A crossing obstruction for the laterally displaced Cao pair

**State:** conditional, exact-Euler necessary condition at a first transverse-impulse return; **not** an actual return, a periodic orbit, a persistence theorem, or a particle mechanism. This uses the smooth, finite-energy, compact-vorticity Cao member and common full-`R^3` local Euler evolution of P253/0163, with one fixed positive kinematic impulse `J e_z`, constant density `rho>0`, and sufficiently large initial lateral displacement `X_2-X_1=d e_x`. P253/0002 imports the Cao existence and its axisymmetric no-swirl orbital stability **only in the source perturbation class**; no such stability is assumed after the pair breaks axial symmetry.

## Exact crossing condition

Let `omega_j` be the two vorticities transported by the *same* three-dimensional Euler flow, `u=B_R3(omega_1+omega_2)`, with its actual nonlocal pressure. On any classical interval set

    J_j(t)=(1/2) integral x cross omega_j(t,x) dx,    I_j=rho J_j,
    X_j(t)=eta_t(X_j(0)),   r=|X_1-X_2|,   n=(X_1-X_2)/r,
    M_j=int |x-X_j|^2 |omega_j(t,x)| dx.

The transported supports are compact. Let `a_j` be any support-radius bound about `X_j`, and assume `r>4 max(a_1,a_2)` at the time in question. Material impulse conservation from P253/0163 gives `J_1+J_2=2J e_z` exactly. The Taylor and differentiated exterior-dipole formulas (its equations (6)–(10)), now applied to the *instantaneous unrestricted shapes*, give

    dot I_(1,x)=-rho/(4 pi r^4) B_x+R_x,                 (1)
    B_x=3 n_x(J_1 dot J_2)+3(J_1 dot n)J_(2,x)
        +3(J_2 dot n)J_(1,x)-15 n_x(J_1 dot n)(J_2 dot n),
    |R_x|<=rho J^2 E/(4 pi r^4),                        (2)
    E=C_0 [ (|J_1| M_2+|J_2| M_1)/(J^2 r)
            +M_1 M_2/(J^2 r^2) ].                     (3)

Here `C_0` is an absolute positive Biot–Savart/Taylor constant (enlarge it once to cover the ball geometry), not a fitted force parameter. In particular `E` is a sufficient **instantaneous** shape/exterior-error bound, not an unproved assertion that it stays small. To check (2), the source multipole remainder obeys `|D(u_2-u_2^dip)(X_1)|<=C M_2/r^5`, and on `B_(a_1)(X_1)`, `|D^2u_2|<=C(|J_2|/r^5+M_2/r^6)`; multiply these respectively by `rho |J_1|` and `rho M_1` in the exact material balance. The pressure has not been set to zero: it determines the actual Euler flow and shape evolution, while the vorticity balance yields this impulse identity. No moment ODE closes the shapes.

The initial calculation in P253/0163 proves `dot I_(1,x)(0)>0` for sufficiently large `d`, so `I_(1,x)>0` immediately afterward. Suppose a first subsequent zero occurs at finite `t_*>0` while the classical labeled solution exists. Its left derivative is nonpositive. At that crossing conservation forces **both** transverse components to vanish; write

    J_1(t_*)=(0,q,J+delta),   J_2(t_*)=(0,-q,J-delta).

A direct contraction, without imposing any later axisymmetry, gives

    B_x=3 n_x F,
    F=J^2(1-5n_z^2)-delta^2-q^2
        +5(n_z delta+n_y q)^2.                            (4)

For retained left-to-right center ordering `n_x<0`, (1)–(2) and `dot I_(1,x)(t_*)<=0` demand `B_x>=-J^2 E`, equivalently

    F <= J^2 E/(3|n_x|).                                 (5)

Consequently **every first return satisfying the stated separation and `n_x<0` must obey**

    delta^2+q^2+5J^2 n_z^2+J^2 E/(3|n_x|) >= J^2.       (6)

This weaker but easily inspected necessity drops the nonnegative last square in (4); the sharper condition is (5). Example of a rigorous excluded crossing: if at a proposed first zero `|n_x|>=1/2`, `|n_z|<=1/4`, `delta^2+q^2<=J^2/2`, and `E<=1/4`, the left side of (6) is at most `(1/2+5/16+1/6)J^2=47J^2/48<J^2`. Thus **there can be no first zero there**, even though all out-of-axis 3D deformations and pressure are admitted in deriving the estimate. For exactly horizontal `n=-e_x`, (6) becomes the sharper explicit requirement `delta^2+q^2 >= J^2(1-E/3)`; if `E<=1/4`, at least `11J^2/12` must enter the squared axial-imbalance/lateral-tilt terms at the crossing. This is a necessity, not a bound that forbids such imbalance in Euler.

## Contrary check and limits

For two initially unmodified parallel Cao rings at separation `X_2-X_1=d e_x+h e_z`, differentiate the *velocity* dipole, not a static cross energy. With fixed `h/d` and `d` sufficiently large,

    dot I_(1,x)(0)
      =3 rho J^2 d(d^2-4h^2)/[4 pi(d^2+h^2)^(7/2)]
       +O(rho J M_2/r^5+rho M_2^2/r^6),                (7)

where `r=sqrt(d^2+h^2)` and each initial `M_j=M_2`. Thus the leading response is positive at `h=0`, zero at `|h|=d/2` (the error decides there), and **negative** at `h=d` for sufficiently large `d`. This sign reversal is a contrary control on the geometric factor `1-5n_z^2`; it is *not* a dynamic return of the original horizontal pair. A standalone numerical differentiation of the dipole `u_z=J(3h^2/R^5-R^(-3))/(4 pi)`, at `d=J=rho=1`, central step `10^-5`, gave force `0.238732414717` versus (7) `0.238732414638` for `h=0`, `-0.063303490988` versus `-0.063303490980` for `h=1`, and approximately zero at `h=1/2`. This checks the leading sign/normalization only; the remainder estimate and return necessity are analytic. No simulation of the nonlinear Euler path has been claimed.

The García–Hassainia–Hmidi Theorem 1.1 imported in P253/0002 is an exact all-time translating-frame periodic **axisymmetric no-swirl, equal-strength patch** pair for selected small core parameters and a Cantor/Borel parameter set. It gives neither an out-of-axis `I_x` crossing nor full-3D Floquet/nonlinear shape control, and its patch family is not the smooth Cao pair. In particular it does not supply the missing shape/exterior estimate (3) along this pair or transfer a periodic orbit to the present unrestricted physical perturbation class. Coaxial axisymmetry keeps transverse impulse identically zero and is not a positive example of this return.

A finite-time return remains possible only after enough axial offset, substantial redistribution/tilt, a sufficiently large shape/exterior error, lost center ordering, or close approach invalidates the separated-core expansion; (6) identifies a **necessary escape from the controlled cone**, not which escape actually occurs. The first-return existence, bounded shape and exterior tail for arbitrarily long times, an invariant tube/restoring estimate for unrestricted three-dimensional perturbations modulo Euclidean symmetries, and inertia/internal observables remain the P2/P3 obligations. P4 still requires quantum amplitudes and exchange/statistics, action normalization, relativistic propagation and physical currents on this same Euler state/action map. Electron charge/spin/magnetic identification, a distinct neutral neutrino sector with weak current and mixing, and independent predictive observables remain entirely unproved. No claim promotion or issue #203 completion follows.
