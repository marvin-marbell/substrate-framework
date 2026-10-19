# Primitive local colour feedback

`colour_gauss_feedback.py` distinguishes an unprojected filled-sea marginal from a continuous SU(4) Gauss state. It uses all sixteen actual primitive-code occupations per species. This is a diagnostic of the selected saddle, not a solution of its quantum Gauss constraint.

## Exact local calculation

Use canonical generators `Tr(T_a T_b)=δ_ab/2` and the selected involution `β²=I`, `Tr β=0`. The one-body marginal is `C_s=(I-s b_s β)/2`, with heavy polarity `s=±1` and light polarity `+1`. Its sixteen-dimensional Gaussian density is obtained from the actual primitive lift of `c†βc`, not an assumed local number eigenstate. The two species' colour operators are even, so their tensor-product calculation does not require a separate inter-species ladder convention.

For the combined colour charge,

\[
\langle Q_a\rangle=-\tfrac12(s b_H+b_L)\operatorname{Tr}(T_a\beta),\qquad
\langle Q^2\rangle=\tfrac12(s b_H+b_L)^2+4-\tfrac18(2-b_H^2-b_L^2).
\]

The code independently evaluates these expressions using the full density matrices. Balanced opposite polarities with identical bands cancel the **mean** charge, not the Casimir. In the two-particles-per-species conditional sector, the primitive decomposition is `6⊗6=1⊕15⊕20`, with total Casimir eigenvalues `0,4,6`. The singlet probability is computed by the actual Casimir spectral projector; centre neutrality alone does not select that singlet.

## Classical mean-cancellation cost and units

Write the unrestricted fields as `Φ_μ=Σ_b u_{μb} B_b`, with `Tr(B_b B_c)=4δ_bc`; `u` and `Π_u` are canonical. If `D` maps colour rotation generators into these sixty coordinates, the minimum classical momentum satisfying `Dᵀp=-⟨Q⟩` has

\[
p=-D(D^TD)^{-1}\langle Q\rangle,\qquad
E_{\mathrm{mean}}=\tfrac12\sqrt\Lambda\,\langle Q\rangle^T(D^TD)^{-1}\langle Q\rangle.
\]

This is **not a quantum lower bound**: field/momentum correlations and the boundary/link dressing are absent. The canonical tangent projection divides raw matrix coefficients by four. An actual basis-Gram probe corrected an earlier factor-eight assumption; both the canonical metric and the constraint are independently checked.

## Executed evidence

The actual project-Python CLI completed all eighteen cases (`orders=24,32`, `Λ=1,8,64`, three polarity/band combinations), naturally exiting zero. `colour_gauss_feedback_receipt.json` retains the complete result. At order 32 and `Λ=64`:

| Sea | squared mean charge | total matter Casimir | conditional singlet probability | classical mean-cancellation cost |
|---|---:|---:|---:|---:|
| Same sign, unequal bands | 1.98399815 | 5.981999 | 2.02e-11 | 68.41673865 |
| Opposite sign, unequal bands | 8.305e-6 | 3.998009 | 0.1666599 | 0.00028640344 |
| Balanced opposite, identical bands | 1.344e-31 | 3.995553 | 0.16663983 | 6.47e-31 |

All asserted primitive identities and canonical residuals are below `1e-9`. The separate `quantum_su4_gauss.py` consumer imposes genuine continuous Gauss equivariance and returns positive finite charged Ritz differences for **both** polarities. This diagnostic does not overturn those results, prove a charged gap, establish a three-dimensional phase, or establish pure mechanical half-spin.
