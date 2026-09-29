# P253/0167 validation boundary

The exact formula oracle exited normally in the repository Python environment:

```text
.venv/bin/python proposals/P253-euler-particle-mechanisms/attempts/0167-two-euler-obstruction/verify_factorization.py
factor-composition time=-2: curl witness=2, energy/(rho volume)=1.00000000
factor-composition time=0: curl witness=2, energy/(rho volume)=0.50000000
factor-composition time=1: curl witness=2, energy/(rho volume)=0.62500000
factor-composition time=3: curl witness=2, energy/(rho volume)=1.62500000
locked ABC composition and distinct helical-mode cross pressure satisfy exact Euler
```

SymPy verifies that the acceleration of the composed periodic shear maps has nonzero curl at an Eulerian point **for symbolic time `t`**, not merely four numerical time samples. Finite-grid averages independently check the positive-time energy formula; the positive control verifies both the locked ABC Euler composition and the physically distinguishable helical-mode stationary Euler sum with its nonconstant **single** pressure. A proposed universal factorization no-go would fail those controls. The smooth compactly supported `R³` construction and its short-time continuation rely on the local-jet and local-well-posedness argument in [derivation.md](derivation.md), not on the periodic script. This does not establish a material-cell second Euler law, a uniform near-family bound, or a particle. Pyright diagnostics of the runnable verifier returned `OK`.
