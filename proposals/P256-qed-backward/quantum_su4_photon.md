# Continuous SU(4) Gauss and a genuinely coupled compact photon

`quantum_su4_photon.py` extends the **actual** `quantum_su4_gauss.PhysicalHamiltonian`, without changing that module or discarding any of its physical states. It is a two-cell selected-orbit EFT, not a three-dimensional photon, a quantization of all 60 field coordinates, a Lorentz-invariant theory, an on-shell scattering amplitude, or canonical QED. The underlying Pauli qubits are bosonic; the matter transfers are the existing primitive full-CAR/code words, not fermionic signs reconstructed from minors. The neutral parity marker at mode 16 and the charged SU(4) boundary representations are unchanged.

## Exact domain and laws

There are two parallel EM paths: edge 0 is L→R and edge 1 is R→L. The independent integer is `ell=e1`. With `n=nlight_left-2`,

\[
 e_0=\ell+n,\quad e_1=\ell,\quad q=N_L-4,
 \qquad e_0-e_1=n,\quad -e_0+e_1+q=n_{\rm light,right}-2.
\]

Only the independent loop variable is truncated: `ell=-R,...,R`. This is **not** the intersection of two edge-electric cutoffs. For every retained `ell`, the complete old number/centre domain is retained, including every boundary state. Each tensor has five separate axes

\[
 (\text{matter fibre},d_r,d_r,B,\ell).
\]

`B` is 4 for `NL=3`, 1 for `NL=4`, and 4 for `NL=5`. It is never identified with the loop axis when applying a boundary generator. The constructor shallow-copies block metadata into new dictionaries and does not mutate the base blocks, arrays, or Hamiltonian.

Both colour paths use the **same genuine Haar connection**, `U` for edge 0 and `U†` for edge 1. Tying these is an explicit *flat-colour-loop EFT constraint*, not a claim that two independent SU(4) links have been quantized. There is one colour canonical momentum. The original left/right centre selection, Haar Peter–Weyl projection, full Lie-Gauss angular Hamiltonian, inertia, onsite operators and colour electric energy are untouched.

The family now includes one positive colour-bond potential `Lambda*Jpref*W/8`, where `W=sum_mu||Phi_mu-U Phi_mu Udagger||HS^2`. `ExactHaarAlignment` projects the continuous multiplication function before truncation, including complete adjoint CG multiplicities; it is not a product of separately compressed `PUP` coordinates. Its centre-neutral blocks act on the two colour indices with matter, boundary and loop axes as spectators, and contribute their actual diagonal to the preconditioner. Two EM paths do **not** double this single underlying colour bond. `--Jpref 0` retains the original zero-alignment family; alignment is independent of the external EM sources.

The colour angular term is evaluated using separate five-axis actions. In particular, the boundary generator acts only on `B`. Its independent diagnostic applies the actual operators twice to verify

\[
 H_{\rm angular}=\frac1{2I}\sum_a
 [(Q_L^a-E_L^a)^2+(Q_R^a+Q_B^a-E_R^a)^2]
\]

on a random vector of the **complete physical domain**, in addition to comparison with the original four-axis angular action in every loop slot. This is an eliminated-Gauss/unitary-gauge EFT; no extra unimplemented gauge-generator Hilbert space is asserted.

## Electromagnetic Hamiltonian and directed hopping

Let `S|ell>=|ell+1>` unless that target lies outside the independent loop cutoff. There is no wrapping. The electric and magnetic terms are

\[
 H_{\rm EM}=\frac{U_{E1}}2[(\ell+n)^2+\ell^2+q^2]
             -\frac K2(S+S^\dagger).
\]

The base already includes `UE1*(n²+q²)/2`, so the added electric diagonal is exactly

\[
 \Delta H_E=U_{E1}(\ell^2+\ell n).
\]

This avoids counting the old tree energy twice.

For each actual primitive directed right→left light transfer `T` in `base.edges`, the two half-weight paths give

\[
 T_{\rm tot}(A_0,A_1)=\frac12 T
 [e^{iA_0}I+e^{-iA_1}S^\dagger]+\text{Hermitian adjoint}.
\]

Path 0 raises `nleft` and `e0` while preserving `ell`. Path 1 raises `nleft` and lowers `ell`; hence `e0` is unchanged and `e1` decreases. Both preserve the displayed Gauss equations. Heavy matter is EM-neutral: the two unshifted half-weight heavy paths sum exactly to the old heavy hopping, with no light Peierls derivative. When `S` is *formally* replaced by `I` and phases vanish, the two light half paths likewise recover exactly the old coefficients. This formal coefficient identity is not a finite normalizable photon vacuum.

Actual sparse `mm⊗lm` products are built once per primitive edge, combined by `(target,source,flavour)`, and their adjoints cached. During application, `B` and `ell` are spectator columns of these directed products. This safe grouping is unrelated to the orientation operator, where they remain separate generator axes. No directed hopping is recovered by repeated `H±iJ` decompositions.

## Source ownership and Ward identities

The primary sources are **external light-matter sources only**. Changing `(A0,A1)` does not change the dynamical quantum magnetic potential. `apply(..., magnetic_background=Bext)` is an explicit alternative choice of an additional external magnetic background, replacing `-K cosB` by

\[
 -\frac K2(e^{iB_{\rm ext}}S+e^{-iB_{\rm ext}}S^\dagger).
\]

Only if such a background is explicitly chosen may one set `Bext=A0+A1`. This is not silently done in the primary current/resolvent response. Under a pure gradient

\[
 A_0\mapsto A_0+\theta,\quad A_1\mapsto A_1-\theta,
 \quad H_\theta=W H W^\dagger,\quad W=e^{i\theta n_{\rm left}}.
\]

The curl and quantum magnetic potential are unchanged. For a real two-path source vector `a`, the derivative multipliers of the two directed light paths are `i a0` and `-i a1`. Mixed contact differentiation adds `i b0` and `-i b1`, respectively. Adjoints use conjugate multipliers. Thus for gradient `g=(1,-1)` and `rho=nleft`,

\[
 J_g=i[\rho,H],\qquad C_{gg}=-[\rho,[\rho,H]].
\]

Operator evidence includes full-H Hermiticity, finite-difference current/contact derivatives at nonzero source, exact gradient covariance, both density commutator identities, angular Gauss squares, heavy-source neutrality, and actual wrong-sign Gauss, wrong-heavy-electric-count and wrapping controls. A control is not declared successful without a nonzero measured defect.

## Same-H ground frame, photon observables and spectral scope

The ground is never borrowed from the old tree model. A lowest-eigenvalue solve is followed by QR, a Hermitian Ritz diagonalization, **actual full-H actions**, and enforced residual/orthogonality limits. Ground degeneracy is resolved against an actual separated next eigenvalue, not a presumed charged T-doublet: the boundary representation can compensate matter degeneracy. Independent random deflated solves search the measured complement for additional ground vectors; newly found vectors are included in the QR/Ritz frame until an actual separated complement is measured. If all computed vectors are degenerate, rank exceeds the explicit ceiling, or the measured separation is not safely larger than residual uncertainty, execution fails instead of claiming a complete ground frame. Increase `--levels`/`--max-ground-rank` when the exception requires it.

Reported observables use the normalized full ground projector, so they do not depend on an arbitrary choice inside a multiplet. They include:

- actual `ell` probabilities and boundary-shell probability;
- actual finite-cutoff `sinB=(S-S†)/(2i)` spectral probabilities;
- loop reduced purity;
- matter/loop and joint matter/colour/loop distributions and their independence defects;
- connected `ell`/`sinB` correlations with light density and colour Casimir.

Correlations are measured; a symmetry-forced zero is not replaced by an invented nonzero value. Reduced purity for a mixed ground-projector ensemble is not by itself an entanglement certificate.

For either Hermitian photon source `O=ell` or `sinB`, with ground frame `G`, energy `E0` and `Q=I-GG†`, the complete-action moments are

\[
 m_0=\frac1{\operatorname{rank}G}\|QOG\|_F^2,\quad
 m_1=\frac1{\operatorname{rank}G}\operatorname{Tr}[(QOG)^\dagger(H-E_0)(QOG)].
\]

Equal-time closure includes elastic weight. The first moment is checked independently against the actual half-double-commutator action. The low computed **same-H eigenstates** give positive Lehmann poles with full-H residuals. Their weight is subtracted from `m0`, explicitly reporting the unseen inelastic weight. There is no assertion that a low eigenstate list captures the whole photon response.

An additional source-active, reorthogonalized finite Krylov projection supplies positive Ritz quadrature poles, their full-H residuals, and projected zeroth/first moments even when the lowest eigenstates are photon-dark. These are explicitly **finite projection poles, not exact eigenstate poles**. The quadrature retains total source norm but does not resolve the unseen exact spectral measure. Increasing `--photon-steps` refines this projection; the complete-action moments remain its reference.

## Complete-domain matter resolvent and two-source kernel

The response acts on the complete projected Gauss-plus-loop domain, not a list of low intermediates. For `|omega|` strictly below the measured complement gap, CG solves

\[
 Q(H-E_0-\omega)Q X=QJG.
\]

A ground-block term regularizes only the redundant ground direction. The preconditioner uses the actual full-H diagonal (including exact colour angular diagonals), with positive absolute shifted entries, projected on both sides; it changes no Hamiltonian or denominator. Every solve enforces its true action residual and reports iterations, leakage, frequency and status.

The ground-frame kernel includes contact, both complement orderings, and finite-frequency ground Born commutators:

\[
 \mathcal K_{ab}(\omega)=G^\dagger C_{ab}G
 -G^\dagger J_a Q(H-E_0-\omega)^{-1}QJ_bG
 -G^\dagger J_b Q(H-E_0+\omega)^{-1}QJ_aG
 +[G^\dagger J_aG,G^\dagger J_bG]/\omega.
\]

At static gradient source, the exact complement solution is `-i Q rho G`. It is compared with the full iterative solution and with the actual `(H-E0)` action. Contact/pair cancellation must hold with nonzero contact and pair terms. Omitting contact gives `-pair`; freezing all intermediates to the ground gives `contact`. Both genuinely violate Ward when measured nonzero. Complete-action matter current zeroth/first moment sum rules are also checked.

The CLI chooses independent source vectors `(1,-1)` and `(1,1)` (gradient and curl), reports every entry of their nontrivial two-source ground-frame kernel at a frequency below the measured gap, and identifies all couplings as finite external matter sources. They span the two external path phases. This is not an on-shell photon process, a 3D polarization construction, Lorentz covariance, or recoil scattering.

Both `NL=3` and `NL=5` coupled-H finite ground Ritz energies are compared with `NL=4` at identical parameters. A positive numerical difference is not promoted to an infinite-cutoff positive-gap theorem.

## Portable commands

From the repository root, choose a Python environment containing NumPy and SciPy:

```sh
python proposals/P256-qed-backward/quantum_su4_photon.py --output /tmp/su4-photon.json
python proposals/P256-qed-backward/quantum_su4_photon.py --R 1 2 --levels 12 --output /tmp/su4-photon-R.json
python proposals/P256-qed-backward/quantum_su4_photon.py --cutoffs minimal medium --lambdas 1 8 --inertias 2 --heavy-sign -1 --output /tmp/su4-photon-colour.json
python proposals/P256-qed-backward/quantum_su4_photon.py --operators-only --R 1 2 --output /tmp/su4-photon-operators.json
python proposals/P256-qed-backward/quantum_su4_photon.py --Jpref .2 --R 2 --output /tmp/su4-photon-aligned.json
```

Defaults are the minimal 33-function Haar link, `R=1`, `Lambda=1`, `I=2`, `Jpref=0`, balanced whole-heavy-H sign `-1`, `UE4=.3`, `UE1=.4`, `K=.7`, light `m=.2,r=.12,v=.3`, and `NL=3,4,5`. The CLI exposes independent heavy bare coefficients, exact alignment strength, all named cutoffs, loop cutoffs, inertia, Lambda, polarity and source-solver/eigen tolerances. Every receipt includes local loaded-source SHA-256 hashes, actual tensor dimensions, explicit finite-projection interpretation and native JSON booleans. The fixed field orbit remains a declared EFT input, not an unrestricted-field phase solution.

## Executed coupled-H consumer

The parent-owned default CLI naturally exited zero in 1058.51 seconds. `quantum_su4_photon_receipt.json` binds the calculation to source SHA-256 `5b3acd65f24c365ba1a77d2a350591033eba23480208208bc254b50485272eb3`, the displayed parameters, and actual full-H residuals.

| Light number | full physical dimension | lowest Ritz energy | measured complement spacing | two-source kernel norm |
|---|---:|---:|---:|---:|
| 3 | 392160 | -0.524048698428 | 0.215069383148 | 0.00214572418001 |
| 4 | 122172 | -0.808814868612 | 0.300789935938 | 0.00180984191720 |
| 5 | 392160 | -0.524048698428 | 0.215069383148 | 0.00214572418001 |

All three measured ground frames have rank one. Both charged finite-energy differences are `0.284766170184`. Static pure-gradient kernel norms are at most `4.52e-17`, while omitted-contact and frozen-intermediate controls leave defects of order `0.0021–0.0024`. The neutral electric-loop observable has a same-H eigenstate pole at `0.734367678683` with weight `0.312708691755`; its full-action zeroth moment is `0.312885761817`, leaving `0.000168461004` unseen inelastic weight after the computed low poles. These are finite-model observations, not an on-shell photon S matrix.

The `R=1` boundary-shell probabilities are about `0.313`, not convergence. The completed separate `R=2`, `Jpref=0` three-sector CLI naturally exited zero in 2263.30 seconds and retains the same historical executable hash in `quantum_su4_photon_R2_receipt.json`. Dimensions are `653600/203620/653600`, charged/neutral energies `-0.544586407794/-0.829330570948`, and both charged differences `0.284744163154`. Shell probabilities fall to `0.0108056862` charged and `0.0107939383` neutral. The neutral dominant electric-loop pole moves to `0.659966878107`, weight `0.375928006706`, total zeroth moment `0.378540228660`. Stable charged energy does not establish photon cutoff convergence: the pole and total weight change substantially from `R=1`.

## Executed nonzero exact alignment

`quantum_su4_photon_aligned_receipt.json` records the actual `Jpref=.2,R=2` three-sector CLI, naturally exited zero in 2211.53 seconds, executable SHA-256 `ffadad919f74fe510484530037d4ae11b7860046fa384aa721839d2771d5c0bf` and provider SHA-256 `42883b7bd4eb7c0e725dd6284f376f665c8e6fe0604b492d0f84c5297524d370`. The potential coefficient is `0.025`; all other bare parameters match the preceding `Jpref=0,R=2` calculation.

| Light number | lowest energy | measured complement spacing | two-source kernel norm | loop shell probability |
|---|---:|---:|---:|---:|
| 3 | -0.426339574565 | 0.190246727369 | 0.00239387946083 | 0.0108058908 |
| 4 | -0.711082837965 | 0.275933498164 | 0.00201543248359 | 0.0107940859 |
| 5 | -0.426339574565 | 0.190246727369 | 0.00239387946083 | 0.0108058908 |

Ground rank is one in each sector; maximum recorded full-H eigenresidual is `7.46e-13`. Both charged finite differences are `0.284743263400`. Static Ward norms are at most `6.77e-17`, while damaged contact/intermediate controls leave `0.00228–0.00270`. The neutral dominant electric-loop eigenstate pole is `0.659967581904` with weight `0.375928564476`, total zeroth moment `0.378541634791`; computed low eigenstates leave `0.00258768206740` unseen inelastic weight.

The separate `quantum_su4_photon_aligned_domains_receipt.json` naturally exited zero in 153.14 seconds on complete neutral medium/full Haar domains, exercising cross-irrep alignment and source/Gauss operators, **not** their spectra. Independent source/proof review accepts the exact projection, single-bond coefficient and five-axis embedding. These executed finite consumers do not supply an infinite Haar cutoff, nonlinear sixty-field phase, mechanical spin, three-dimensional photon or photon S matrix.

