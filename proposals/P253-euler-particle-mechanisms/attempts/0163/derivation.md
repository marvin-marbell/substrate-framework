# A constructive three-dimensional Euler interaction: lateral material-impulse exchange

## 1. Physical object and what is imported

Fix a sufficiently regular smooth member of the compact-vorticity, finite-energy Cao–Lai–Qin–Zhan–Zou translating ring family, with polynomial exponent chosen sufficiently high for classical local `H^s(R^3)`, `s>5/2`, Euler evolution (for example a smooth `p>=8` member with the corresponding finite Sobolev regularity). Density `rho>0` is constant. In coordinates about its center and axis `e_z`, its vorticity is `omega_C=zeta(r,z) partial_theta=r zeta(r,z)e_theta`, with `zeta>=0`, not identically zero, and `supp omega_C subset B_a(0)` for some finite `a`. Its decaying whole-space velocity is `u_C=B_R3 omega_C`. No assertion of uniformity across the thinness family is needed.

P253/0002 imports existence and finite-energy scope of the smooth Cao ring. P253/0007 equations (1)–(3) already prove the compact-vorticity impulse moment and the full-space exterior dipole with a differentiated moment remainder. P253/0006 and 0012 treat *coaxial axisymmetric* scalar axial impulse exchange; 0007's static cross energy is **not** a derived force. P253/0015 and 0029 work with a distinct time-periodic axisymmetric patch pair. P253/0162 excludes a particular smooth positive-core hyperbolic route on one ring and does not test a laterally interacting pair. These are distinct dependencies, not proof that the result below is new to the mathematics literature.

Put

    J_C=(1/2) integral_R3 y cross omega_C(y) dy
       =pi e_z integral_(r>0,z in R) r^3 zeta(r,z) dr dz
       =J e_z,                                             (1)

so `J>0` and `J` has kinematic impulse units `L^4/T`; `I_C=rho J_C` has physical momentum units `M L/T`. Choose `X_1=0`, `X_2=d e_x`, `d>4a`, and initial data

    omega_j(0,x)=omega_C(x-X_j),
    u(0,x)=u_C(x-X_1)+u_C(x-X_2),
    omega(0)=omega_1(0)+omega_2(0).                         (2)

The two ring axes remain initially parallel to `e_z`. Their *sum* has no common axis of symmetry. It is genuine divergence-free finite-energy whole-space 3D Euler data, not an approximate pair solution or two independently transported rings. Classical local Euler existence gives a common smooth flow map `eta_t` for a nonzero interval; it transports the **separate** initial vorticities by

    omega_j(t,eta_t(y))=D eta_t(y) omega_j(0,y),
    partial_t omega_j=curl(u cross omega_j),
    u=B_R3(omega_1+omega_2).                             (3)

Their supports remain disjoint for as long as the common classical diffeomorphism exists. The same total velocity determines the actual whole-space pressure up to a constant:

    p=rho(-Delta_R3)^(-1) partial_i partial_j(u_i u_j),
    partial_t u+(u dot grad)u=-(1/rho) grad p.              (4)

No independent ring pressure, prescribed center trajectories, static force law, uniform ambient flow, wall or finite-radius field truncation is inserted.

## 2. Exact material exchange identity

Define the *material label* impulse (not the total field's only momentum) by

    I_j(t)=(rho/2) integral_R3 x cross omega_j(t,x) dx.  (5)

For compactly supported smooth `F`, `integral x cross curl F dx=2 integral F dx`. Equation (3), integration by parts and the divergence-free Euler identity give

    dot I_j =rho integral u cross omega_j dx
            =rho integral u_(3-j) cross omega_j dx,        (6)

because `integral u_j cross curl u_j dx=integral [grad(|u_j|^2/2)-(u_j dot grad)u_j] dx=0`, where `u_j=B_R3 omega_j`. Applying the same identity to total `u` proves

    dot I_1+dot I_2=0.                                    (7)

This is an exact identity for the local full Euler solution, not an equation of motion closed on the impulses alone. The velocity, all label shapes and pressure remain coupled.

For a compact divergence-free label with kinematic impulse `J_j=I_j/rho`, divergence identities imply

    integral omega_j dx=0,
    integral (x-X_j)_a (omega_j)_k dx=epsilon_(a k l)(J_j)_l.  (8)

At `t=0` expand `u_2` across `B_a(X_1)` in (6). The constant term vanishes by (8). Contracting the first moment with `epsilon_(i b k)` and using `div u_2=0` gives the **velocity-gradient**, not static-energy-gradient, law

    dot I_(1,i)(0)=-rho (J_1)_b partial_i(u_2)_b(X_1)+R_i,
    |R_i|<=C rho M_(2,1) sup_(B_a(X_1)) |D^2 u_2|,       (9)

where `M_(2,j)=integral |x-X_j|^2 |omega_j(0,x)| dx`. There is no claim that (9) is an autonomous collective equation at later times; it is an exact initial Taylor identity with an explicit bounded remainder.

## 3. Signed non-axisymmetric positive consequence

The P253/0007 dipole and its differentiated remainder yield, for `|x-X_2|>2a`,

    u_2(0,x)=[3 n(n dot J_2)-J_2]/(4pi |x-X_2|^3)
              +O(M_(2,2)|x-X_2|^(-4)),
    D u_2=D[3 n(n dot J_2)-J_2]/(4pi |x-X_2|^3)
              +O(M_(2,2)|x-X_2|^(-5)),                   (10)

with `n=(x-X_2)/|x-X_2|`. Twice differentiating the source expansion gives `sup_(B_a(X_1))|D^2u_2|<=C(J d^(-5)+M_(2,2)d^(-6))`; constants depend on the one fixed compact ring, not `d`. At the equatorial point `(x,0,0)` with `x<d`, `J_1=J_2=J e_z`, so

    (u_2)_z(x,0,0)=-J/[4pi(d-x)^3]+O(M_(2,2)d^(-4)),
    partial_x (u_2)_z(0)=-3J/(4pi d^4)+O(M_(2,2)d^(-5)). (11)

Substitution into (9) proves the **new P253/0163 positive initial-response theorem**:

    dot I_(1,x)(0)= +3 rho J^2/(4pi d^4)
        +O(rho J M_2 d^(-5)+rho M_2^2 d^(-6)) > 0,       (12)

for sufficiently large fixed separation `d`, with `M_2=M_(2,1)=M_(2,2)`. By (7), `dot I_(2,x)(0)=-dot I_(1,x)(0)`. Both initial transverse components vanish. Continuity of the classical local flow and its impulse moments then gives a `tau_d>0` such that

    I_(1,x)(t)>0,   I_(2,x)(t)<0,   0<t<tau_d.            (13)

This constructs an *actual non-axisymmetric, material two-carrier Euler interaction* with signed, oppositely exchanged transverse physical impulse; it does **not** prove the core planes rotate by the same angle as the impulse, recurrent motion, localization in an unrestricted neighborhood, or a charged/spinning particle. Dimensions: `rho J^2 d^(-4)` is momentum/time, as required by (12). Pressure (4) and the complete Biot–Savart tail are part of the local solution, not optional boundary corrections.

**Independent check of sign and normalization.** Differentiate the analytic dipole `u_(2,z)=J/(4pi rho)[3z^2/R^5-R^(-3)]` in the *physical* convention `I_(j,z)=J`: `partial_x u_(2,z)(0)=-3J/(4pi rho d^4)`, hence `-I_(1,z)partial_x u_(2,z)=+3I_(1,z)I_(2,z)/(4pi rho d^4)`. A separate throwaway SymPy evaluation with symbolic positive `d,J,rho` returned exactly this derivative and expression. This checks the leading dipole contraction/sign, not the nonlinear Euler evolution or the remainder estimate; (6)–(13) are proved analytically above.

## 4. Mechanism, contrary case and next construction

The mechanism is **steering through exchange of vector impulse on material vorticity labels**, not solving a raw transport resonance on one axisymmetric ring. The dynamically meaningful state remains `(omega_1,omega_2)` on full `R^3`, with `I_1+I_2` fixed; the vectors `I_j` are measured coordinates, not a closed particle ansatz. The guaranteed positive step is (12)–(13). Unlike a signed interaction inferred from a static cross-energy, it differentiates the actual Euler label balance (6), and its sign survives a controlled far-field error for real smooth Cao data.

The discriminator against a gratuitous universal-force story is **coaxial initial placement**: simultaneous axisymmetry forces both transverse impulses to stay zero on its classical symmetric branch, even though axial impulse exchange can occur. Opposite orientation/negative relative vorticity changes the leading sign without changing its `d^(-4)` order; it is outside the identical-positive-Cao hypothesis. These are controls, not proof of a restoring cycle.

A constructive continuation is to search for an **actual labeled full-3D return** using a Poincaré section defined by a transverse material-impulse crossing, while retaining both full vorticity shapes, pressure, exterior flow and the symmetry quotient. One must construct a subsequent transverse return, control the shape/ambient remainder there, and then prove an invariant return tube in an unrestricted physical 3D perturbation class; a 3D fluid simulation or a finite impulse ODE alone cannot supply that theorem. The all-time GHM axisymmetric patch pair provides a separate *positive* material exchange example, but no stability transfer between the different carrier families is assumed. The source-specific adjoint of P253/0062 remains another open route, not a gate imposed on this construction.

Issue #203 still requires persistent one-family localization, inertia/internal dynamics, and a physical quantum/relativistic/charge bridge for electron and neutrino sectors. Equations (12)–(13) earn a positive finite-time P3 interaction input **only**; they leave P2 full-3D persistence and all particle identification open. The parent remains open. No claim registry, release, generated state, merge or global originality assertion follows from this research attempt.
