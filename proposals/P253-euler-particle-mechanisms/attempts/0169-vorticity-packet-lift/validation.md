# P253/0169 validation boundary

The periodic algebra control exited normally in the repository Python environment:

```text
.venv/bin/python proposals/P253-euler-particle-mechanisms/attempts/0169-vorticity-packet-lift/verify_label_stress.py
two label velocities reconstruct an exact global Euler trajectory with shared pressure zero
mixed-energy virtual derivative/(rho volume): stress=0.25000000, vorticity=0.25000000
initial mixed-energy value is zero although its displacement derivative is nonzero
```

Central differences of the actual global triangular 3-D Euler solution at three nonsymmetric points distinguish its nonzero cross advection from the wrong uncoupled evolution. Independently sampled stress-divergence and vorticity-displacement expressions agree at `1/4` per `ρ|T³|`; a zero mixed-energy *value* is not a zero variational derivative. These periodic fields are **not** compact packets. Packet persistence on `R³`, the `H^{s−1}` algebra bound and the disjoint-swirl pressure witness are separate analytic arguments in [derivation.md](derivation.md) and P253/0001. The oracle does not establish a coarse center Euler field, internal fiber pressure autonomy, or issue #203's particle requirements. Pyright diagnostics on the script returned `OK`.
