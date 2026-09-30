# A second, adverse ray at the fixed `e=1/10` Euler background

Status: an interval-certified **material** monodromy combined with an exact **full-pressure** geometric-optics polarization and the fixed-horizon whole-`R³` packet argument of `packet.md`. It rejects uniform all-mode linear-`L²`-norm stability of this particular periodic background. It is not a nonlinear defect-instability theorem, a persistent particle, or an electron/neutrino mechanism.

## Exact relative orbit and material growth

Use the exact Beltrami Euler state of `derivation.md`,

    U_e=(sin z−e sin y, cos z+e cos x, e cos y−e sin x),  e=1/10.

On `y=π/2−x, z=−π/4`, `U_z=U_x+U_y=0` and `xdot=−h(x)`, with `h(x)=1/√2+e cos x>0`. Starting at `(0,π/2,−π/4)`, the **whole-space** endpoint is `(−2π,π/2+2π,−π/4)` after `T=2π/√(1/2−e²)=20π/7`; this is translation-periodic, not closed in `R³`. The velocity-gradient matrix on the orbit is

    A(x)=[[0,−e sin x,1/√2],
          [−e sin x,0,1/√2],
          [−e cos x,−e cos x,0]].

The tangent `(1,−1,0)` is an invariant line. In transverse coordinates `n=(δx+δy)/2`, `w=δz`, material displacement obeys `d(n,w)/dt=B(x)(n,w)` with `B=[[-e sin x,1/√2],[-2e cos x,0]]`. Thus its full-period matrix `M` solves `M_x=−B(x)M/h(x)`, `M(0)=I`, from `0` to `−2π`. Its determinant is exactly one: `tr(−B/h)=−(log h)'` and `h(0)=h(−2π)`. The tangent multiplier is exactly one because the relative endpoint returns to the same periodic phase.

`certify_helical_material.py` uses 600,000 outward-interval Cayley-midpoint steps. Rational outward bounds `h>.607`, `a<.708`, `||C||∞<1.332`, `||C'||∞<.39`, `||C''||∞<.52` for `C=−B/h` give local matrix error `<3.5d³` and global row-norm error `<3.5·6.284·(6.284/600000)²·5000<1.21·10⁻⁵`. The enclosed Cayley trace is approximately `2.00003883474930394414`; the **true** trace is in `[2.00001470937,2.00006296013]`, strictly above two. Hence `M` has positive reciprocal real eigenvalues `μ_+>1.00384264`, `μ_-<1`. This certified inequality depends on `mpmath.iv` enclosing arithmetic and trigonometric values and the explicit Peano/Cayley error bound; the independent RK4 and DOP853 values `tr M≈2.0000388347259`, `μ_+≈1.00625119905` are numerical controls, not the certificate.

## Pressure-projected polarization on the same orbit

Choose a left eigenvector `l` of `M` with eigenvalue `μ_+` and lift it to a physical covector `ξ_*=(l_n/2,l_n/2,l_w)`; choose `b_*` in the same transverse physical plane with `ξ_*·b_*=0`. Their **Euler pressure-aware** bicharacteristic equations are

    ξdot=−Aᵀξ,   bdot=−A b+2ξ(ξ·A b)/|ξ|².

Then `ξ(T)=μ_+⁻¹ξ_*`. **This is a special invariant-plane calculation, not the false general identity `(ξ×b)dot=A(ξ×b)`.** In the orthonormal transverse plane spanned by `(1,1,0)/√2` and `e_z`, the restriction of `A` is `D=[[-e sin x,1],[-√2 e cos x,0]]`; the tangent line decouples. Both `ξ` and `b` remain in this plane and the full-pressure equations preserve `ξ·b=0`. With hats denoting unit directions, `d log|ξ|/dt=−ξhat·D ξhat` and `d log|b|/dt=−bhat·D bhat`, because the pressure correction is parallel to `ξ`. Orthogonal unit vectors span this two-dimensional plane, so `d log(|ξ||b|)/dt=−tr D=e sin x`; its integral over a full period is zero by the periodic `log h` identity. Since `ξ(T)=μ_+⁻¹ξ_*` and the transverse perpendicular amplitude line returns with positive multiplier, **`b(T)=μ_+ b_*`** and `b(mT)=μ_+^m b_*`. The tangent covector `ξ∥(1,−1,0)` in `floquet.md` has an elliptic amplitude monodromy at the same `e`; one benign polarization did not test all transverse covectors. Material stretching alone would not have supplied a pressure-aware result without this confined polarization argument.

The whole-space Leray packet construction in `packet.md:15-62` applies at each **fixed** horizon `mT`: the nonzero central covector returns to its eigenline (its shrinking magnitude is allowed), the background and gradient coefficients repeat after the lattice translation, and the full-pressure symbol and `O_{m,σ}(N⁻¹)` remainder are unchanged. First take packet frequency `N→∞` at fixed small radius `σ` and fixed `m`, then `σ→0` in the `L²` norm ratio. It follows for the exact `R³` linearized Euler semigroup that `||S(mT)||_{L²_sol→L²_sol}≥μ_+^m` and `limsup_{t→∞}t⁻¹log||S(t)||≥log μ_+/T>0.00042728`. Packets can differ with `m`; this is neither a growing fixed finite-frequency eigenfunction nor a nonlinear instability of the indexed Gaussian seeds. It does remove `e=1/10`, like the already excluded `e=3/10`, as an **all-mode linear-L²-norm-stable** candidate medium; #203's full carrier, charge, shared quantum bridge and neutral chiral sector remain unbuilt.
