# P253/0165 validation boundary

The coefficient oracle exited naturally from this repository's Python environment:

```text
.venv/bin/python proposals/P253-euler-particle-mechanisms/attempts/0165-integrable-helical-sector/verify_helical_sector.py
compact bump integral: 0.443993816168081
nonzero initial phase-column coefficient: 0.409859132390344
prepared zero phase column: -1.796e-15
direct Euler column derivative: 0.120176057732823
integrated-by-parts derivative: 0.120176057732824
exact linear shear mean cosine: 0.943765900834140
exact linear shear width coefficient: 0.109305924422724
prepared packet initial L2 norm squared: 0.014509142870328
stationary receiver support cutoff: 2.828427124746190
centroid receiver support cutoff: 50.297367019321683
physical stationary receiver t=0,0.5,1,3: [0.014509142867502801, 0.005795343380851974, -0.0006179867494019456, 0.0]
physical centroid receiver t=51: 0.000000000000000
fixed-Q leading energies, sigma 1,2,4,8,16: [0.1973920880217872, 0.0986960440108936, 0.0493480220054468, 0.0246740110027234, 0.0123370055013617]
quadratic Euler pressure transverse virial, analytic/direct: 0.115080553157302, 0.115080553157078
```

The independent coefficient control in the derivation is integration by parts: `∫cos z (d''+d)=0`, `∫(d''+d)=∫d`, and `∫₀^(π/2)sin⁴t dt=3π/16`. The quadrature script checks those against direct finite sums, checks packet width growth, and evaluates the **actual compact velocity overlap receiver** on the exact linear Euler solution. The initial overlap independently matches its positive physical `L²` norm; a sign reversal at `t=1` is followed by identically zero receiver output at `t=3`, so it cannot be read as sustained flavor oscillation. The exact support-distance argument, not a quadrature near zero, gives receiver extinction. The script does **not** numerically solve nonlinear Euler, prove a weighted pressure estimate, select a carrier, or establish a particle. The full-pressure linear equation is exact because the prepared perturbations have zero vertical velocity, making their linear pressure source exactly zero. No independent revision-bound scientific review has yet accepted this attempt. No registry claim, release or issue completion is authorized.

For the separate `carrier-probe.md`, the source virial `∫x²∂a∂b(w_aw_b)=2∫w_x²` is exact by two transverse integrations by parts. The oracle now checks the compact radial-swirl witness two ways—one-dimensional analytic angular reduction versus independent physical polar quadrature—with residual `2.24e-13`; it does **not** evolve nonlinear Euler or certify general carrier instability. Reversing the swirl changes neither the quadratic stress nor the nonzero vertical pressure acceleration.

The second oracle exited successfully in the project `.venv` using SciPy's `K₁`, spherical-Bessel and adaptive radial quadrature. Its physical regularization is **the same Plummer `θ_Q,σ` tail**, not the full compensated dress; the analytic estimate of the dress remainder is separate in `derivation.md`.

```text
.venv/bin/python proposals/P253-euler-particle-mechanisms/attempts/0165-integrable-helical-sector/verify_qtail_interaction.py
sigma=1 R=8: transverse=0.730558326033 axial=0.083878831440 transverse/(2pi/R)=0.930175750440
sigma=1 R=16: transverse=0.383642751404 axial=0.014978412883 transverse/(2pi/R)=0.976938244277
sigma=1 R=32: transverse=0.194948100441 axial=0.002416506742 transverse/(2pi/R)=0.992862522611
sigma=0.5 R=8: transverse=0.767285502808 axial=0.029956825766 transverse/(2pi/R)=0.976938244277
sigma=0.5 R=16: transverse=0.389896200883 axial=0.004833013483 transverse/(2pi/R)=0.992862522611
sigma=0.5 R=32: transverse=0.195932427177 axial=0.000738123835 transverse/(2pi/R)=0.997875657509
```

At fixed `σ`, the transverse `R`-weighted coefficient approaches `2π`; the axial `R`-weighted coefficient decreases instead. The complementary analytic check is the distributional inverse Fourier transform of `16π²/k⁴` and two derivatives giving `2π(1-n_z²)/R`; this does not evolve the separated cores or equate an energy with a force. SciPy's quadrature error estimate is numerical evidence, not an independently certified enclosure or scientific review.

After adding the `0165` attempt registry entry, `env PYTHONPATH=src .venv/bin/python scripts/validate_repository.py` exited `WORKFLOW VALID: 271 claims, 271 accepted, 14 proposals, 5 skills; MIGRATION QUEUE: 218 units, 0 pending, 0 partial`. These are repository schema/workflow counts, not scientific acceptance.
