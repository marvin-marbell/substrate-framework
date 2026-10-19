# Exact continuous-Haar four-field alignment

## Scope and interface

`exact_haar_alignment.py` supplies

```python
alignment = ExactHaarAlignment(link, frame)
W = alignment.matrix                     # scipy CSR, link.dimension square
block = alignment.blocks[target, source]  # CSR, d_target**2 by d_source**2
receipt = alignment.evidence()            # executes checks; JSON-native values
```

`link` is an existing canonical `quantum_su4_gauss.HaarLink`, with its
Peter–Weyl order and `offset` unchanged. All three named cutoffs are supported:
minimal `1,4,bar4`, medium `1,4,bar4,6,15`, and full
`1,4,bar4,6,10,bar10,15`. The field input has shape `(4,4,4)` and consists of
four arbitrary traceless Hermitian matrices. It need not be Clifford,
independent, normalized, or nonzero. Nonfinite, wrong-shaped, genuinely
nontraceless and non-Hermitian inputs are rejected. Hermiticity and trace
validation permit only float64-scale numerical residuals, with per-field
threshold `2e-11 * max(1, maximum absolute entry)`. The quadratic norm must be
representable as a finite float64 value.

`matrix` and `blocks` are the alignment potential itself, with no coupling
`J` included. Blocks include every retained nonzero representation channel
and every diagonal constant block; absent off-diagonal keys mean zero.
Construction performs the CG and weighted block calculations, but neither
link spectra nor independent fundamental-product controls.
`evidence()` explicitly executes those additional checks. No matter or charged
Hamiltonian is constructed here.

The exactness claim concerns Haar integrals and representation projection,
not symbolic or interval-certified floating-point linear algebra. Numerical
CG kernels are classified with declared tolerances and checked against
intertwining, isometry, known channels, and independent controls. Scientific
receipt values are emitted only when the caller executes `evidence()` or the
CLI; this note does not assert an unexecuted receipt.

## Continuous multiplication first, projection second

Use the normalization of the existing source:

\[
T_a=B_a/\sqrt8,\qquad \operatorname{Tr}(B_aB_b)=4\delta_{ab},
\qquad \operatorname{Tr}(T_aT_b)=\tfrac12\delta_{ab}.
\]

For four Hermitian traceless fields define real coefficients and their
positive semidefinite Gram matrix

\[
\Phi_\mu=p_{\mu a}T_a,\quad
p_{\mu a}=2\operatorname{Tr}(T_a\Phi_\mu),\quad
K_{ab}=\sum_\mu p_{\mu a}p_{\mu b},\quad
s=\operatorname{Tr}K=\sum_{\mu a}p_{\mu a}^2.
\]

The real orthogonal adjoint representation is

\[
D^{15}_{ab}(U)=2\operatorname{Tr}(T_aUT_bU^\dagger).
\]

The **continuous** multiplication function is exactly

\[
W(U)=\sum_\mu\|\Phi_\mu-U\Phi_\mu U^\dagger\|_{\rm HS}^2
=s-\sum_{ab}K_{ab}D^{15}_{ab}(U).
\]

It obeys `0 <= W(U) <= 2s`. For any orthogonal Peter–Weyl projector `P`, the
wanted matrix is `P M_W P`. Therefore it is Hermitian positive semidefinite
and satisfies the same operator upper bound. These properties follow before
truncation; they are not obtained by symmetrizing a wrong finite product or
clipping a negative eigenvalue. The implementation leaves the computed
alignment matrix unmodified and checks its residuals and spectrum.

In particular, `P M_U P M_Udagger P` generally differs from
`P M_(U Udagger) P`: the former inserts a projector at the intermediate
representation boundary. Continuous matrix-valued `U` is unitary, while its
minimal compression is not. Neither products of minimal `link.U` nor a
quadrature approximation define this provider.

## Complete adjoint Clebsch–Gordan construction

For a source representation `s`, construct the canonical Hermitian product
generators

\[
T_a^{\rm prod}=T_a^{15}\otimes I_s+I_{15}\otimes T_a^s.
\]

An embedding `C_t : r -> 15 tensor s` satisfies

\[
T_a^{\rm prod}C_t=C_tT_a^r.
\]

First diagonalize the tensor-product Casimir. Every such embedding lies in
the eigenspace of eigenvalue `C_r`; let `Q` be an orthonormal basis for that
whole eigenspace. Equal Casimirs do **not** identify a representation: for
example `10` and `bar10` must still be distinguished. Restrict each generator
to `Q` and solve the complete positive Hom Laplacian

\[
L=\sum_a A_a^\dagger A_a,\qquad
A_a=I_r\otimes(Q^\dagger T_a^{\rm prod}Q)
      -(T_a^r)^T\otimes I_Q.
\]

This is the stated sum of squares because these `A_a` are Hermitian. Use
column-major vectorization: `vec_F(Tprod C - C Tr) = A vec_F(C)`. **All** zero
eigenvectors are retained, not an arbitrary first one. If `v_t` are
orthonormal kernel vectors, lift them as

\[
C_t=\sqrt{d_r}\,Q\,\operatorname{unvec}_F(v_t).
\]

Schur's lemma and the vector normalization imply

\[
C_t^\dagger C_u=\delta_{tu}I_r.
\]

The code checks these isometries, mutual orthogonality, all fifteen generator
intertwining identities, and finite group exponentials. Its Casimir selection
tolerance is `2e-9 * max(1, abs(C_r))`; its Hom zero threshold is
`2e-9 * max(1, largest Hom eigenvalue)`. The receipt records the selected
eigenspace dimensions, kernel dimensions, low Hom eigenvalues, first nonzero
Hom eigenvalues, residuals, and tensor-product Casimir spectra. There is no
claimed diagnostic success from an unavailable editor diagnostics device.

Within the seven canonical representations, the complete retained channels
are

| Source | Retained targets and multiplicities in `15 tensor source` |
| --- | --- |
| `1` | `15` once |
| `4` | `4` once |
| `bar4` | `bar4` once |
| `6` | `6`, `10`, `bar10` once each |
| `10` | `6`, `10` once each |
| `bar10` | `6`, `bar10` once each |
| `15` | `1` once and `15` twice |

Other irreps in the tensor products are not states of these named link
cutoffs. Their omission does not justify dropping multiplicities of any
**retained** target. The implementation compares the complete solved Hom
multiplicities, including zeros, against this table. Centre-forbidden
channels vanish because the adjoint has zero four-ality.

Independent known-positive embeddings are `I_15` for `15 tensor 1 -> 15`
and `delta_ic / sqrt(15)` for `15 tensor 15 -> 1`. For the two adjoint copies,
the invariant tensors provide phase-independent span controls:

\[
(C_f)_{ica}=f_{ica}/2,\quad
(C_d)_{ica}=d_{ica}/\sqrt3,\quad
d_{ica}=2\operatorname{Tr}(\{T_i,T_c\}T_a).
\]

For this SU(4) normalization their flattened embeddings are mutually
orthogonal isometries. The receipt checks that the **two-dimensional solved
Hom space** contains both tensors. Those controls are executed even when
`15` is not a retained state of the requested cutoff.

## Peter–Weyl matrix and weighted streaming

The link basis is `sqrt(d_s) D^s_cd(U)`, ordered with row index `c`, then
column index `d`. With `C_t` reshaped as `C_t[i,c,a]`, exact representation
multiplication and Haar Schur orthogonality give

\[
(M^{15}_{ij})_{(r,a,b),(s,c,d)}
=\sqrt{d_s/d_r}\sum_t(C_t)_{ica}(C_t)^*_{jdb}.
\]

The sum includes **every** intertwiner in the Hom kernel. This identity is
`einsum('ica,jdb->abcd', C_t, C_t.conj())` followed by the square-PW reshape;
it is not multiplication within the kept link space.

Only its weighted contraction is needed. Because `K` has rank at most four,
form

\[
X_{t,\mu,c,a}=\sum_i p_{\mu i}(C_t)_{ica},\quad
B_{(r,a,b),(s,c,d)}=\sqrt{d_s/d_r}
\sum_{t,\mu}X_{t,\mu,c,a}X^*_{t,\mu,d,b}.
\]

The block of `W` is `s I` on equal representations minus `B`. This streams
one weighted representation block at a time, never allocating 225 full link
matrices. The CSR assembly uses the supplied `link.offset`, preserving the
original consumer order. All blocks preserve centre four-ality, including
the nontrivial mixing between `6,10,bar10` in the full cutoff.

## Minimal independent complete-intermediate control

For minimal only, embed its complete PW basis in `HaarLink('full')`. Every
intermediate reached by multiplication with `U_ad^dagger` is present:

- `1 -> bar4`;
- `4 -> 1 + 15`;
- `bar4 -> 6 + bar10`.

Thus if `E` is the minimal-to-full isometric embedding and
`X_ad = U_ad(full)^dagger E`, then

\[
E^\dagger U_{bc}({\rm full})U_{ad}({\rm full})^\dagger E
=X_{bc}^\dagger X_{ad}
=P_{\min}M_{U_{bc}\overline{U_{ad}}}P_{\min}.
\]

The full matrix is used only as a **complete intermediate** for this
particular operation; its general compressed `U` is not asserted unitary.
The exact independent alignment is

\[
2\sum_\mu\operatorname{Tr}(\Phi_\mu^2)I
-2\sum_{\mu abcd}(\Phi_\mu)_{ab}(\Phi_\mu)_{cd}
X_{bc}^\dagger X_{ad}.
\]

The control compares this fundamental-product expression to the adjoint
provider. It also restores

\[
P_{\min}\sum_k U_{ki}^\dagger U_{kj}P_{\min}=\delta_{ij}P_{\min},
\]

using the complete `U` intermediates `4`, `6+10`, and `1+15`. The analogous
minimal-internal product is required to have a nonzero compressed-unitarity
defect. Its alignment defect is also reported, but need not be nonzero for
zero or exceptional fields. The independent product comparison executes for
both the caller's frame and a deterministic arbitrary non-Clifford frame.
It is deliberately **not** advertised as complete for medium/full, where
more fundamental intermediate representations are missing.

## Scalar, canonical and numerical controls

`evidence()` checks all of the following on the requested cutoff:

- positive, Hermitian link spectrum and the `2s` upper bound;
- exact centre-block support and four-field reconstruction;
- the three known-positive Hom controls, including the two adjoint copies;
- all solved channel multiplicities and full-representation finite group CG
  covariance, including representations outside minimal used by the controls;
- the continuous HS norm against `s - K:D15` at a fixed generic SU(4) element;
- canonical trace normalization, adjoint exponential, and actual centre
  exponentials for all seven canonical representations;
- simultaneous scalar conjugation covariance across the complete kept PW
  space, not merely fundamental coefficients;
- invariance under a generic orthogonal change of the adjoint basis, with
  coefficients and CG first indices transformed together;
- arbitrary non-Clifford/nonorthogonal fields, identically zero fields, and
  rejection of independent nontraceless/non-Hermitian inputs;
- the minimal independent fundamental-product and adverse controls above.

The conjugation convention is explicit: `V_g f(U)=f(g U gdagger)`, whose PW
block is `D_r(g).T tensor D_r(g).conj().T`. Therefore
`W[g Phi gdagger] = V_gdagger W[Phi] V_g`. This is covariance of the same-frame
potential, not invariance of a frozen frame under unrelated endpoint gauge
transformations.

The CLI additionally compares nested requested cutoffs by restricting the
larger exact matrix back to the smaller order. It emits the actual selected
frame, selected Gram and stationarity residual, source hashes (including this
derivation), full link spectra, every check and its scope. It uses the same
`uniform_data(band, size=3, stiffness=2, preferred=0)` selection as the
existing Gauss/photon default consumer; `--m`, `--r`, `--v`, and `--stiffness`
permit explicit choices without importing a different field normalization.

Run with the project's scientific Python, for example:

```sh
python exact_haar_alignment.py --cutoffs minimal medium full --output exact_haar_alignment_receipt.json
```

This is a construction/control command, not a full charged-eigenrun command.
The integration owner executes and reviews its receipt separately.

### Executed provider and same-H join

The parent-owned all-cutoff command completed naturally in 5.71 s.
[exact_haar_alignment_receipt.json](exact_haar_alignment_receipt.json) binds
provider SHA256
`42883b7bd4eb7c0e725dd6284f376f665c8e6fe0604b492d0f84c5297524d370`
and `m=.2,r=.12,v=.3,stiffness=2`. Minimal/medium/full dimensions are
`33/294/494`, with finite Ritz minima
`4.414793085586/3.808560281602/3.808560281602`; every nested restriction
residual is zero. The minimal complete-intermediate product residual is
`1.546e-14`, while the naive compressed-product alignment defect is
`.923048897976`.

The [aligned photon receipt](quantum_su4_photon_aligned_receipt.json) executes
this provider in the complete three-sector compact EFT at `Jpref=.2,R=2`,
with coefficient `Lambda*Jpref/8=.025` for `Lambda=1`. Its parent-owned
2211.53 s run and the independently matched
[dressed-carrier receipt](gauss_photon_charge_verified_receipt.json) are
finite-H evidence, not a gap of standalone `W` or a full-field phase.
Independent source/proof review accepts the exact projection, single-bond
coefficient and tensor embedding; it does not license those larger claims.
Receipt-time derivation hashes remain historical snapshots after this evidence
update; no new scientific execution is implied.

## What this does not prove

This exact multiplication identity accepts unrestricted classical field
values, but **does not** quantize their sixty real coordinates. Feeding the
selected frame to a quantum Haar rotor adds an exact potential to a
**fixed-orbit EFT**, not a solution of the full quantum orbital law.

For an irreducible Clifford frame the simultaneous commutant is scalar, so
its stabilizer in SU(4) is `Z4`. This identifies the classical zeros and
unbroken centre of the alignment function. It neither selects a quantum
three-dimensional phase nor proves confinement/deconfinement, radial
stability, a photon sector, a physical spinor, or a charged mass gap.

Even strictly positive finite alignment Ritz minima do not certify an
infinite-cutoff Higgs gap. When `W` vanishes at a centre element, arbitrarily
small positive-Haar-measure neighborhoods have arbitrarily small `W`;
continuous **multiplication alone** consequently has spectral infimum zero.
Any spectral-gap claim for a rotor plus electric energy, matter and spatial
couplings needs that actual combined Hamiltonian, its parameters and its
limits. No canonical QED, full phase, or pure mechanical half-spin claim is
made here.
