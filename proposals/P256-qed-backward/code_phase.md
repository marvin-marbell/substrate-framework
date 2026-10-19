# Coupled quantum sector selection, not matter-only projection

This supplies a conditional **ground-sector selection theorem for the full
bosonic matter + compact-photon Hamiltonian**, and an actual finite quantum
square prototype. The selected primitive Pauli law, quantum dynamics, and
coupling constants remain inputs. Nothing here establishes a generic substrate
origin of that law, a photon Coulomb phase, or continuum/dressed-charge matching.

## 1. Domain and signs

Take a finite connected open decorated cubic graph. Every spatial cell has the
three internal tree edges `0-alpha`; only orbital 0 carries cubic spatial
edges. Put one **code qubit** on each graph edge and one distinct **physical
compact U(1) rotor** on each spatial edge. Internal tree edges have no physical
rotor. The `primitive_pauli_words` law specifies diagonal vertex words B_v and
oriented edge words A_uv, with A_vu=-A_uv. In increasing-edge orientation,
A_e contains X_e and only Z on earlier incident edges. Occupation is
n_v=(1-B_v)/2. For a simple open path gamma, define

\[
 A_\gamma=i^{|\gamma|-1}\prod_{e\in\gamma}A_e,
 \qquad T_\gamma=-i A_\gamma n_{\rm source}(1-n_{\rm target}).
\]

Here |gamma| counts edges, and the product is ordered along the oriented path.
For an oriented plaquette boundary p,

\[
 S_p=i^{|p|}\prod_{e\in\partial p}A_e.
\]

These conventions give S_p=+1 in the desired even-algebra sector. The primitive
words, not imported CAR matrices, establish that S_p is Hermitian, S_p^2=I,
and every S_p commutes with every B_v and A_e. Consequently all gauged path
bilinears commute with S_p. The finite program uses those actual words.

Orient every physical spatial link from its lower endpoint to its upper
endpoint. Use U_lower|E>=|E-1>, hence [E,U_lower]=-U_lower. A directed
source-to-target matter transfer uses U_lower when traversing that orientation
and U_lower^dagger in reverse. Define divergence as outgoing minus incoming.
Then

\[
 G_v=\operatorname{div}E_v-(n_v-b_v)
\]

commutes with each gauged transfer: a transfer reduces both n and div E at its
source and increases both at its target. Products along multi-link paths do the
same, with zero change at intermediate vertices. Internal paths affect orbital
occupations but not spatial-link divergence. **On the decorated graph Gauss is
therefore imposed per spatial cell with n_cell=sum_alpha n_(cell,alpha), not
per orbital with nonexistent internal electric links.** The square prototype
has one orbital per spatial vertex, so its displayed G_v is literal.

Use a fixed integer neutralizing background b_cell. On an open graph with no
external electric links, Gauss requires total N=sum b_cell. Also
prod_v B_v=pi, where pi is the explicitly selected boundary parity sign, so
(-1)^N=pi is necessary. The code construction preserves that fixed parity; it
does not manufacture a local odd-charge operator. Neither Q nor R below changes
N or a Gauss eigenvalue.

The [neutral boundary parity extension](pauli_compton.md) now supplies a full
odd/even **physical** Fock space and a charged creator with electric boundary
flux. It adds no graph cycle; its neutral marker and boundary conditions must
be kept distinct from electromagnetic charge. Q_eta preserves its occupation,
so the same sector-intertwiner and ground-energy proof extend to each matched
physical domain. The square prototype below retains its stated N=2 domain.

The full microscopic Hamiltonian is

\[
 H=H_{\rm matter}(\{U\})+\frac{U_E}{2}\sum_l E_l^2
   -K\sum_p C_p+\Delta\sum_p\frac{1-S_p}{2},
 \qquad C_p=\frac{W_p+W_p^\dagger}{2},
 \quad K\geq0.
\]

W_p is the oriented physical rotor product around p. It is independent of the
code words. Matter can have arbitrary fixed complex hopping coefficients,
on-site terms, and nearest/improved straight-two-link terms: every such term
must use the **same graph path** in its Pauli word and physical rotor product.
A path may include internal star edges, which contribute no rotor factor.
Sector-independent interactions made solely of occupations are also allowed.
These are explicit restrictions, not claims about all bosonic Hamiltonians.

## 2. Admissible signs and an explicit microscopic intertwiner

All plaquette signs cannot be chosen independently in three dimensions. A
closed product of boundaries is zero in the binary cycle space, and the
corresponding product of these primitive S_p is +I, not -I. Thus an admissible
assignment obeys prod_(p in any boundary relation) s_p=+1. On a simply connected
open cubic box the elementary plaquettes span the entire graph cycle space;
adding the internal stars introduces no new cycles. Each admissible assignment
is therefore a character of the cycle space.

For completeness, choose a spanning tree with r=E_graph-V+1 chords. Each
fundamental-cycle S has X support on its own chord and no other chord. Every
nonempty product therefore has nonzero X support, so cannot be +I or -I.
The r commuting independent generators have all joint sign eigenspaces of
dimension 2^(E_graph-r)=2^(V-1). Cycle decomposition using the oriented A
anticommutation identities generates other S words with the displayed i
convention; deleting a repeated edge and recombining boundaries introduces
no extra minus sign. Thus all+ exists, and every boundary relation has product
+I. Its joint occupation spaces are rank one for each parity-compatible
occupation pattern. Q then supplies the same statement for any admissible
sector. These statements concern code qubits before imposing Gauss; Q's
occupation preservation gives identical physical rotor domains afterwards.

A binary link assignment eta_l=+/-1 realizes this character:

\[
 s_p=\prod_{l\in\partial p}\eta_l.
\]

This follows directly from linear algebra over F_2: the annihilator of the
boundary relations is the image of the transpose boundary matrix. One can
choose eta=1 on a spanning tree and solve the independent chord values; all
internal star edges can be included in that tree. Different choices related by
vertex sign changes give the same s_p. There is no presumption that an
inconsistent sign assignment is an existing Hilbert sector.

A stronger, explicit realization avoids assuming equal phases of arbitrary
occupation bases. Define the **code-qubit** unitary

\[
 Q_\eta=\prod_{l:\eta_l=-1} Z_l.
\]

Because only A_l contains X/Y on code qubit l (other A words use only Z there),

\[
 Q_\eta^\dagger A_e Q_\eta=\eta_e A_e,\quad
 Q_\eta^\dagger B_v Q_\eta=B_v,\quad
 Q_\eta^\dagger S_p Q_\eta=s_pS_p.
\]

Thus Q_eta maps the actual all+ eigenspace bijectively onto sector s, including
its fixed occupation parity and physical Gauss subspace. This proves both the
existence and equal dimension of admissible sectors without assuming a
canonical CAR representation. It also proves the eta factor for **every open
path**, not only for nearest-neighbor terms. In canonical projector-built
occupation states the actual map has phase

\[
 f_n=\langle n,s|Q_\eta|n,+\rangle,\qquad |f_n|=1.
\]

The program derives these phases and verifies their amplitudes; it does not set
f_n=1 by convention. It checks every oriented simple square path of one, two,
and three links, including both two-link routes between opposite vertices.
The two-link square terms are corner paths, not geometric straight-link
improvement terms: a single square cannot contain a non-backtracking straight
two-link path. The general Q identity covers straight paths on the cubic graph
without this finite geometry limitation.

## 3. Quantum photon rotation and the local bound

Define the independent **physical rotor** unitary

\[
 R_\eta=\prod_{l:\eta_l=-1}\exp(i\pi E_l).
\]

Integer rotor electric spectra imply

\[
 R_\eta^\dagger U_l R_\eta=\eta_l U_l,\quad
 R_\eta^\dagger E_l R_\eta=E_l,\quad
 R_\eta^\dagger W_pR_\eta=s_pW_p.
\]

This is a quantum operator transformation, not a shift of an external
classical background. R_eta leaves Gauss invariant. On a truncated electric
basis E=-c,...,c, use **noncyclic** lowering: U|-c>=0. The commutator and pi
rotation identities remain exact. U is then not unitary, but it and W_p are
contractions, so -I<=C_p<=I still holds. No wraparound matrix element is used.

Let T_eta=Q_eta R_eta map the physical all+ domain to the physical s domain.
Each matter path acquires the same eta product twice, once from its Pauli word
and once from its rotor product; these cancel. Electric energy, occupations,
background, and Gauss are unchanged. Conjugating the **full sector Hamiltonian**
therefore gives, as operators on the same all+ physical domain,

\[
 T_\eta^\dagger H_s T_\eta
 =H_+ +2K\sum_{p:s_p=-1}C_p+\Delta N_{\rm def} I.
\]

The proposed identity is correct with this explicit domain identification.
Without T_eta the symbols H_s and H_+ act on different subspaces, so a direct
operator inequality between them would be ill-typed. Since each C_p>=-I,

\[
 \boxed{T_\eta^\dagger H_s T_\eta
 \ \geq\ H_++(\Delta-2K)N_{\rm def}I.}
\]

No matter norm enters this bound and no independent-photon or empty-band
approximation is made. For Delta>2K, every defective sector's infimum exceeds
the all+ infimum by at least (Delta-2K)N_def. In finite dimension, the min-max
principle gives the same bound for each ordered sector eigenvalue. A uniform
strict positive lower margin is sufficient for **ground-sector selection**;
it does not imply that every all+ excited state lies below every defective
state, and it does not make unitary evolution remove initially present defects.
At nonzero temperature defect entropy and temperature matter. There is no
claim of universal equilibrium preparation. For unequal plaquette couplings
and penalties the bound is the sum of Delta_p-2|K_p| over defective p.

For untruncated compact rotors, this is a quadratic-form inequality on the
common electric-energy form domain. On a finite graph, matter hopping and
magnetic terms are bounded, and R commutes with E; the form domain is preserved.
Usual self-adjoint electric Hamiltonians plus these bounded terms suffice.
Zero U_E still allows the bounded-operator version of the inequality.

## 4. Boundary and global-loop qualifications

The uniqueness assertion concerns the loop **sign sector**, not uniqueness of
its physical ground state. On a periodic box or a graph with holes,
plaquettes do not span all graph cycles. Noncontractible code Wilson signs
remain and may need eta assignments with nontrivial holonomy. T_eta still maps
between them when the physical rotor domain is invariant, but local Hcode
assigns no penalty to flat global defects. No strict preference among those
sectors follows from this theorem; indeed an allowed flat eta gives exact
isospectrality. Open simply connected topology is therefore essential for
selection of the entire desired cycle character by local plaquette penalties.

Fixed electric boundary fluxes and fixed global **electric** flux constraints
are preserved, since E and n are unchanged. External nondynamical link phases,
fixed magnetic/Wilson boundary data, or a restriction not invariant under
R_eta may prevent the map. They must be absent, invariant, or transformed to
matched domains before applying the bound. Matter paths crossing an external
boundary require the same accounting. A periodic Wilson-loop restriction must
not silently be held fixed while applying an eta that flips it.

## 5. Actual finite prototype and falsifying adversary

`code_phase.py` starts with four primitive edge qubits and four quantum rotors,
then **enumerates physical raw basis states before allocating matrices**.
The unconstrained cutoff-one tensor product has dimension 16*3^4=1296, but
that matrix is never allocated. It requires N=2, background (1,0,1,0), and
G_v=0. It applies Pauli-word bit actions and noncyclic integer electric shifts
directly to the physical raw basis. Every inserted matrix element must remain
in that basis; an escape raises an error, rather than being discarded.

S-sector matrices are extracted only after building the full physical
Hamiltonian. Bases are actual normalized (I+sS)/2 projector columns, jointly
resolved by occupations and electric fields. The program derives Q occupation
phases, combines them with the field-dependent R=(-1)^E factor, and reports:

- physical constraint preservation, rotor commutator and pi-rotation identities;
- actual projector reconstruction and raw commuting-sector residuals;
- exact full coupled transformed identity and matter cancellation;
- actual + and - spectra, their union versus the full physical spectrum;
- minimum operator and ordered-spectrum margins for Delta-2K;
- all oriented simple-path phase identities with complex, phase-shifted
  hopping coefficients and competing multi-link paths.

The adverse model replaces the dressed A on edge (0,3) by its oriented bare X
**before** constructing the Hamiltonian, using the same physical rotor. It
remains number/Gauss preserving and Hermitian, but does not commute with S.
The full raw commutator and actual off-diagonal +/- block must be nonzero.
This mutation invalidates the proof's commuting-sector premise; no claim is
salvaged by silently discarding that block. On the square, S has Y on this edge,
so the bare X anticommutes with S. A mere overall sign change of a correctly
dressed A would not be this adverse mutation: it preserves commutation and
can represent a different hopping coefficient.

### Parent verification entrypoint

Smallest representative electric cutoff and the default nonzero phase/multi-link
parameters:

```sh
/home/administrator/substrate-framework/.venv/bin/python \
  proposals/P256-qed-backward/code_phase.py \
  --cutoff 1 --magnetic 0.7 --electric 0.8 --delta 2.0 \
  --hopping 0.6 --improved 0.19 --phase 0.23
```

The CLI raises with its JSON report if its exact identities, bound, or bare-X
negative control fails. `acceptance: true` refers only to this finite quantum
prototype. This assignment intentionally does **not** run the CLI mid-flight;
actual numeric results must be read from the parent-run output, not inferred
from the theorem or the existence of the code. Cutoff 2 is an optional second
finite domain, not a continuum extrapolation.

## 6. Input debt and scope of the achievement

The selected local Pauli law and quantum Hilbert-space/dynamics postulates are
not derived here. K, U_E, Delta, complex hopping coefficients, on-site energies,
integer background, topology, and boundary domain are declared inputs. The
Delta>2K hierarchy proves selection **conditional on** that hierarchy and exact
sector-preserving microscopic terms; a bare-X perturbation exhibits why that
qualification is physical. The theorem establishes no generic energetic
mechanism that produces the hierarchy itself, no robustness against arbitrary
sector-mixing terms, no emergent photons in a Coulomb phase, no matter vacuum
polarization correction, and no dressed-charge matching. It strengthens the
prior matter-only penalty estimate to an exact full coupled local bound, rather
than claiming completion of those independent obligations.
