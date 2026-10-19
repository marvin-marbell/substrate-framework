# Gauged orbital locking and conditional rotation fractionalization

This changes the interpretation of `spin_orbital_locking`: flavour is now a **local SU(4) gauge redundancy**, as in `orbital_gauge_locking`, rather than a global flavour symmetry. The conclusion is a conditional symmetry-fractionalization theorem for a deconfined Higgs/Z4 charge sector. It is not a mechanical spin measurement, a proof of the quantum Higgs phase, or an SU(4) Peter–Weyl simulation.

A crucial negative result is retained: a complete local gauge-frame restoration, including its link endpoint action, has **phase +1 on physical states**. The compensating matter matrix alone cannot be called a physical rotation. A localized rotation sign additionally requires a physical source/order rotation and a specified, independently held reference seam. The finite consumer checks the resulting algebra and the competing trivial complete-gauge loop; it does not build that apparatus.

## 1. Microscopic law and the derived lift

The bosonic adjoint fields have the transformation assignments

- $A_0$: spatial scalar, SU(4) adjoint;
- $A_i$, $i=1,2,3$: spatial vector, SU(4) adjoint;
- microscopic complex matter: spatial scalar, SU(4) fundamental;
- the flavour connection: a genuine gauge connection, with electric momentum and Gauss constraint.

Use the occupied-sea-selected positive Gram Clifford frame and scalar gradient/hopping law of the existing helpers. Spatial isotropy gives equal spatial Gram entries. The Clifford representation generates the full complex $4\times4$ matrix algebra. Its common matrix commutant is therefore the scalar matrices; unitarity and determinant one reduce the SU(4) common commutant to

$$Z_4=\{I,iI,-I,-iI\}.$$

The CLI independently computes the exact complex common-commutant kernel in all 16 matrix coordinates. It does not infer this discrete conclusion merely from the absence of infinitesimal internal stabilizers.

A physical rotation $R$ changes the vector order. In a selected gauge, restore the order with $U_R$:

$$U_R A_0 U_R^\dagger=A_0,\qquad
  \sum_j R_{ij}U_R A_jU_R^\dagger=A_i.$$

For infinitesimal rotations, solve the existing **18-real-unknown** stabilizer problem: all 15 traceless Hermitian flavour coordinates and all three spatial coordinates. No fundamental spin matrices are inserted to solve it. The exact Clifford-frame solve has three connected stabilizers, projects bijectively onto the spatial rotations, and has no pure internal Lie stabilizer. The actual scaled, generically conjugated selected frame is solved again numerically by `solve_selected_generators`; its result is checked against conjugation of the exact solution.

Write the resulting generators $T_i$. The solve derives

$$[T_i,T_j]=i\epsilon_{ijk}T_k,\quad
  \sum_iT_i^2=\tfrac34 I,\quad
  T_iT_j+T_jT_i=\tfrac12\delta_{ij}I.$$

Thus the connected compensator is $U(\theta)=\exp(-i\theta n_iT_i)$ and its continuous $2\pi$ lift ends at $-I$. These are properties of the **derived gauge-fixed frame lift**, not of the bare microscopic spatial-scalar action.

## 2. Rotation section, Z4 cocycle, and its meaning

The full frame-preserving extension is

$$1\longrightarrow Z_4\longrightarrow\widehat{SO(3)}
 \longrightarrow SO(3)\longrightarrow1.$$

Its connected component is the derived Spin(3), whose kernel over SO(3) is $\{I,-I\}$. Equivalently, the full extension is $(\mathrm{Spin}(3)\times Z_4)/\langle(-1,-I)\rangle$. On the exact cubic group, the connected preimage has 48 elements and the full preimage has 96. No claim identifies the connected 48 with the full stabilizer.

Choose one actual derived lift for each of the 24 proper cubic rotations, with $U_I=I$. For that section,

$$U_RU_S=z(R,S)U_{RS},\qquad z(R,S)\in Z_4.$$

The section inherited from the connected lift has $z\in\{I,-I\}$. Rephasing $U_R\mapsto i^{b(R)}U_R$ changes its exponent by the Z4 coboundary $b(R)+b(S)-b(RS)$ and can produce all four centre values. Associativity gives

$$z(R,S)z(RS,T)=z(S,T)z(R,ST).$$

The consumer checks all 576 section products using the actual derived $4\times4$ lifts, all 13,824 triple identities modulo four, and a genuinely rephased Z4 section. It also checks the selected order covariance. Commuting spatial $\pi$ rotations about $x$ and $y$ have derived lift commutator $-I$: this witnesses a nontrivial class, not just a removable minus sign in one chosen section.

A centre charge $q\in\mathbb Z/4$ has character $\chi_q(i^kI)=i^{qk}$. Accordingly, **if the physical symmetry admits the stated charge-sector localization**, its localized multiplication factor is $\chi_q(z(R,S))$ and its continuous $2\pi$ action is

$$\chi_q(-I)=(-1)^q.$$

Odd charges are projective; even charges are not spinorial under this connected lift. A lattice cubic result by itself is not continuous SO(3) symmetry: the latter requires the rotational longwave limit of the same microscopic scalar/vector law.

## 3. Noether charge is Gauss plus boundary flux

Set $\hbar=1$ and absorb the gauge coupling into the connection/electric momentum. Choose gauge conventions $\delta_\alpha\psi=-i\alpha\psi$, $\delta_\alpha a_i=D_i\alpha$, and let $\rho$ include fundamental matter and adjoint-Higgs canonical momenta. The canonical generator is

$$Q_{\rm gauge}[\alpha]=\int_\Omega
 \mathrm{tr}(\alpha\rho+E_iD_i\alpha).$$

Covariant integration by parts, with
$\mathcal G=\rho-D_iE_i$, gives the exact boundary relation

$$Q_{\rm gauge}[\alpha]=\int_\Omega\mathrm{tr}(\alpha\mathcal G)
 +\oint_{\partial\Omega}\mathrm{tr}(\alpha E_i)n_i.$$

The combined frame-restoring rotation generator is

$$K_{\rm diag}[T_i]=J_{\rm space}[i]+Q_{\rm gauge}[T_i].$$

$J_{\rm space}$ includes coordinates, the integer-vector orbital/Higgs contributions, and the gauge field; it does not insert a matter spin matrix. On closed physical states, bulk Gauss vanishes and there is no boundary term. Consequently $K_{\rm diag}=J_{\rm space}$ there and the complete physical $2\pi$ rotation is $+1$.

On an open charged region, bulk Gauss still vanishes, but the boundary electric flux does not. Dropping it would incorrectly conclude $Q_{\rm gauge}=0$ on an individual charged state. A separated pair can have projective endpoint factors whose product is an ordinary physical representation. This is compatible with, and does not overturn, the closed-state Noether identity.

## 4. Actual primitive Pauli matter and a genuine finite electric link

`gauged_rotation_fractionalization.py` uses `pauli_compton.EdgeWords.transfer` and `boundary_ladder`, not CAR/wedge matrices as the matter builder. The tree has eight physical orbital vertices (four per cell) and one **neutral parity-marker** vertex. Its eight bosonic edge qubits contain all $2^8=256$ physical matter states. The marker occupation supplies odd physical parity without itself becoming an SU(4), Z4, or electromagnetic charge.

The link is an independent four-dimensional quantum Hilbert space with electric states $|\ell\rangle$, $\ell=0,1,2,3$, clock $Z|\ell\rangle=i^\ell|\ell\rangle$, and cyclic shift $X|\ell\rangle=|\ell+1\pmod4\rangle$. The raw matter/link space has dimension 1024. No electric truncation of an infinite rotor is being pretended to satisfy this exact finite group law.

For the light-only residual model, each cell has the necessary conjugate-six sea reference of centre charge $-2=2$ modulo four. The selected reference is a connected-rotation singlet and has centre-$i$ character $-1$. Only this exact **residual Z4 character** is frozen into the finite model. A frozen colour vector would not be a covariant full-SU(4) reference, and no such full-SU(4) projection is claimed here.

The actual matter number is $N_x$ and the charge entering residual Gauss is $q_x=N_x-2$. Orient the link from left to right. Both endpoint constraints are projected:

$$G_L=i^{q_L-\ell},\qquad G_R=i^{q_R+\ell},\qquad
 P=\prod_{x=L,R}\frac14\sum_{k=0}^{3}G_x^k.$$

This is the **complete Z4 Gauss projection**, not just its Z2 subgroup. There are 72 allowed states: the 70 states at total physical $N=4$, plus the $N=0$ and $N=8$ states. The latter two are Z4-neutral but have normal-ordered integer charge $\mp4$; Z4 Gauss alone is not electromagnetic Gauss. The selected electron/hole pair lies at $N=4$ and has actual integer charges $q_L=+1$, $q_R=-1$, hence zero total charge.

The sea, electron and hole are built from the existing selected-band rest spinors and charge-conjugation helper using primitive boundary ladders. Filling the two negative rest orbitals in each cell gives the sea. Creating a left positive-rest electron and removing a right negative-rest electron gives a four-state electron/hole pair space, dressed with $\ell=1$. The CLI also creates an actual selected pair using the corresponding Wilson-dressed primitive Pauli bilinear and compares it to the ladder-built state, up to its measured overall ordering phase.

An actual transfer $c_L^\dagger c_R$ raises $q_L$ and lowers $q_R$, so it must include $X$, raising $\ell$. Its reverse includes $X^\dagger$. Every one of the 64 orbital transfers is evaluated from the primitive words, combined with the correct cyclic electric shift, and checked against both Gauss generators on the **whole raw Hilbert space**. Onsite transfers do not shift the link. Deleting the shift fails Gauss. Omitting the centre-two references fails Gauss on the actual filled sea. Counting the neutral parity marker as matter fails the physical parity control.

## 5. Pure gauge cancellation versus conditional localized rotation

The complete local centre-$-I$ operations are

$$G_L^2=(-1)^{q_L}Z^{-2},\qquad
  G_R^2=(-1)^{q_R}Z^{2}.$$

They include the link endpoint action. They are exactly the identity on physical states and commute with the Wilson-dressed transfer. A matter-only centre operation reverses the transfer and is not this gauge operation. This is the decisive no-pure-gauge control.

To state a projective **physical** localization, hold a separate reference seam while rotating the physical source/order region. Let $B_x(-I)$ denote its boundary matching. At the endpoint of a $2\pi$ loop the assumed localized symmetry factor is

$$L_L(2\pi)=G_L^2B_L(-I)^{-1},\qquad
 L_R(2\pi)=G_R^2B_R(-I)^{-1}.$$

For the electric-link realization, $B_L(-I)^{-1}=B_R(-I)^{-1}=Z^2$. Thus on the physical fibre

$$L_L(2\pi)=L_R(2\pi)=Z^2.$$

This is a **boundary electric operator**, not a missing link factor disguised as matter spin. The gauge-restoring link endpoint and the inverse reference-seam matching are both retained. Their two minus factors on the $\ell=1$ pair cancel each other; the resulting localized operator is $-1$, while $L_LL_R=Z^4=+1$. The actual primitive local derived-generator exponential is checked against this boundary parity on the entire 72-state Gauss fibre, including its even-charge states. On the selected pair, all 24 localized matrices and all 576 local products have the derived cocycle; their two endpoint factors have ordinary global multiplication. The full physical-space global derived rotation gives $2\pi=+1$.

**What this does and does not establish.** It establishes consistency of a specified symmetry-localization class with the actual Pauli/Z4 charge and flux Hilbert space. The construction of $U_R$ and gauge-equivalence of the Higgs order alone do not establish that the physical apparatus implements $B^{-1}$. Without the independent reference-seam prescription, the complete gauge-restored endpoint loop is simply $G_L^2=1$. This is an explicit counterexample to the stronger unconditional mechanical-spin conclusion. Choosing a physical source/order rotation, rather than applying a local flavour gauge transformation, is an essential additional premise, not an experimentally checked event in this finite model.

## 6. Bare scalar and Berry controls

At rest a bare microscopic spatial-scalar internal generator is zero, so its $2\pi$ action is identity. The selected positive-rest projector also commutes with the derived generators. It therefore admits a constant rest basis and identity Berry transport. The consumer evaluates 64 polar-overlap transport steps in that constant basis, checks the constant projector under a nontrivial derived lift, and separately checks the compensated rest lift ending at $-I$.

Neither a spatial-scalar identity nor an identity constant-projector Berry holonomy proves the charged localization theorem. Conversely, a minus sign obtained by exponentiating the compensator does not establish an interferometer unless its physical source, gauge restoration, boundary/reference treatment, and coherent comparison arm are specified and implemented. No such mechanical interferometer is claimed.

## 7. CLI and claim boundary

Run with the project Python:

```sh
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/gauged_rotation_fractionalization.py \
  --output proposals/P256-qed-backward/gauged_rotation_fractionalization_receipt.json
```

The default `--preferred 0` uses the occupied energy to induce the nonzero selected Gram; `--preferred 1` exercises the alternative existing soft-law regime. `--a`, `--m`, `--r`, `--v`, and `--stiffness` parameterize the same helpers. The JSON includes derived lifts, both section cocycles, multiplication tables, complete Gauss and primitive transfer controls, actual pair actions, the Noether boundary relation, counterexample, and explicit hypotheses/operational limits. Exit zero means the finite algebraic controls pass; it is not a certificate of a deconfined 3D phase or a physical arm protocol.

Remaining conditional assumptions are deconfinement, a quantum Higgs/Z4 low-energy sector, physical spatial symmetry in the appropriate lattice/continuum domain, and the charged-sector/reference-seam symmetry localization. No full SU(4) quantum-link Hilbert space, full SU(4) Gauss projection, Higgs-phase simulation, fundamental spin insertion, canonical full-QED result, or observed mechanical spin is asserted.
