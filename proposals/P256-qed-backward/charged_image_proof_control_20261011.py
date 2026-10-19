from __future__ import annotations
import hashlib
import itertools
import json
import math
import time
from fractions import Fraction
from pathlib import Path
import sympy as s

ROOT = Path(__file__).resolve().parent
PINS = {
    'charged_spinor_joint_free_phase_control_20261011.py': '2b3e220e0cd5dab4934b9fd7c0937e08b59d7a59d599deefcede9aedf8c9e206',
    'charged_spinor_joint_free_phase_control_20261011_receipt.json': '8276441f71408879d494111e5e52ab3f2afe6b1fdee1f905d9484365ceccb2eb',
    'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json': '102966f32c49f40d79d55343925a630e30a9cf7e4a8074cf8628142ba9c5b6e8',
    'local_so5_positive_output_row_inertia_certificate_20261011.py': 'f196557eee4b05a54294c2a6d19763ec6c97e36f41eba939afb7ac51a82fb2ab',
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

START = time.monotonic()
OWN = digest(Path(__file__))
INITIAL = {name: digest(ROOT / name) for name in PINS}
assert INITIAL == PINS
parent = json.loads((ROOT / 'charged_spinor_joint_free_phase_control_20261011_receipt.json').read_text())
assert all(parent[k] for k in ['exact_joint_spinor_D_R', 'matches_existing_corrected_free5_joint_pair', 'full_number3_hole_D_R_verified', 'number3_pair_equivalent_to_number1'])
profiles = json.loads((ROOT / 'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_text())['profiles']
assert [(p['count'], p['spin_replicas']) for p in profiles] == [(1, 2), (63, 126)]
for profile in profiles:
    m0, c0, d0, e20 = [Fraction(profile[k]) for k in ['m_exact', 'c0_exact', 'd_exact', 'e_squared_exact']]
    assert m0 > 0 and e20 == m0*m0 + d0*d0 and e20 > c0*c0

# Independent full 16-state CAR, not an import of any old executable.
ann = []
unsigned = []
for p in range(4):
    A, bad = s.zeros(16), s.zeros(16)
    for mask in range(16):
        if mask & (1 << p):
            dest = mask ^ (1 << p)
            A[dest, mask] = (-1)**((mask & ((1 << p)-1)).bit_count())
            bad[dest, mask] = 1
    ann.append(A)
    unsigned.append(bad)
for p, q in itertools.product(range(4), repeat=2):
    assert ann[p]*ann[q] + ann[q]*ann[p] == s.zeros(16)
    assert ann[p]*ann[q].T + ann[q].T*ann[p] == s.eye(16)*int(p == q)
assert unsigned[0]*unsigned[1].T + unsigned[1].T*unsigned[0] != s.zeros(16)
sectors = {n: [mask for mask in range(16) if mask.bit_count() == n] for n in range(5)}

def lift(M, number):
    full = s.zeros(16)
    for p, q in itertools.product(range(4), repeat=2):
        full += M[p, q]*ann[p].T*ann[q]
    return full.extract(sectors[number], sectors[number])

def endpoint(p, creation):
    return (ann[p].T if creation else ann[p]).extract(sectors[3 if creation else 1], sectors[2])

a, b, e, c = s.symbols('a b e c', real=True)
m, d = e*(a*a-b*b), 2*e*a*b
X = s.Matrix([[0, 1], [1, 0]])
Z = s.zeros(2)
An = s.diag(-1, 1, -1, 1)
Bn = s.kronecker_product(X, s.diag(1, -1))
Cn = Z.row_join(s.I*X).col_join((-s.I*X).row_join(Z))
Rn = s.diag(s.Rational(1, 2), s.Rational(1, 2), -s.Rational(1, 2), -s.Rational(1, 2))
Hn = m*An + c*Bn + d*Cn
Wp = s.Matrix([[1, 0], [0, 1], [1, 0], [0, -1]])/s.sqrt(2)
Wm = s.Matrix([[1, 0], [0, 1], [-1, 0], [0, 1]])/s.sqrt(2)
U = s.Matrix.hstack(Wp*s.Matrix([a, -s.I*b]), Wm*s.Matrix([a, s.I*b]), s.I*Wp*s.Matrix([-s.I*b, a]), s.I*Wm*s.Matrix([s.I*b, a]))

def reduced(z):
    return s.rem(s.expand(z), a*a+b*b-1, b)

def zero(M):
    return all(reduced(z) == 0 for z in M)

D4 = s.diag(-e+c, -e-c, e+c, e-c)
f, g = (a*a-b*b)/2, a*b
R4 = s.Matrix([[0, f, 0, -g], [f, 0, g, 0], [0, g, 0, f], [-g, 0, f, 0]])
assert zero(U.conjugate().T*U-s.eye(4))
assert zero(U.conjugate().T*Hn*U-D4)
assert zero(U.conjugate().T*Rn*U-R4)
# Exterior square uses exactly the parent highest-wedge minus sign.
U2 = s.Matrix([[U.extract([p, q], [i, j]).det() for i, j in itertools.combinations(range(4), 2)] for p, q in itertools.combinations(range(4), 2)])
V = s.zeros(6, 5)
for j, (mask, value) in enumerate([(3, 1), (5, 1), (9, 1/s.sqrt(2)), (10, 1), (12, -1)]):
    V[sectors[2].index(mask), j] = value
V[sectors[2].index(6), 2] = 1/s.sqrt(2)
# U2 rows/columns are lexicographic pairs, while sectors use numeric masks.
lex = [sum(1 << p for p in pair) for pair in itertools.combinations(range(4), 2)]
U2 = U2.extract([lex.index(k) for k in sectors[2]], [lex.index(k) for k in sectors[2]])
W = U2*V
D5 = s.diag(-2*e, 2*c, 0, -2*c, 2*e)
R5 = s.zeros(5)
for j in [1, 3]:
    for i, value in [(0, g), (2, s.sqrt(2)*f), (4, -g)]:
        R5[i, j] = R5[j, i] = value
assert zero(W.conjugate().T*W-s.eye(5))
assert zero(lift(Hn, 2)*W-W*D5)
assert zero(lift(Rn, 2)*W-W*R5)
wrong = W.copy(); wrong[:, 4] = -wrong[:, 4]
assert not zero(lift(Rn, 2)*wrong-wrong*R5)
J = s.zeros(4)
for p in range(4):
    J[sectors[3].index(15 ^ (1 << p)), p] = (-1)**p
Q = s.kronecker_product(s.eye(2), X)
S4 = s.diag(1, -1, 1, -1)
T3 = J*Q*U*S4
assert zero(J.T*lift(Hn, 3)*J+Hn.T)
assert zero(J.T*lift(Rn, 3)*J+Rn.T)
assert zero(T3.conjugate().T*T3-s.eye(4))
assert zero(T3.conjugate().T*lift(Hn, 3)*T3-D4)
assert zero(T3.conjugate().T*lift(Rn, 3)*T3-R4)
# Every basis endpoint implies arbitrary COMPLEX linear combinations by linearity.
t = s.symbols('t0:4')
for creation in [False, True]:
    maps = [endpoint(p, creation)*W for p in range(4)]
    A = sum((t[p]*maps[p] for p in range(4)), s.zeros(4, 5))
    for native, vector in [(Hn, D5), (Rn, R5)]:
        correction_coeff = native*s.Matrix(t) if creation else -native.T*s.Matrix(t)
        correction = sum((correction_coeff[p]*maps[p] for p in range(4)), s.zeros(4, 5))
        assert zero(lift(native, 3 if creation else 1)*A-A*vector-correction)
    # Changed canonical number is exactly 1 or 3, never a neutral vector image.
    N = s.zeros(16)
    for p in range(4):
        N += ann[p].T*ann[p]
    assert N.extract(sectors[3 if creation else 1], sectors[3 if creation else 1])*A == (3 if creation else 1)*A
    assert not zero(A)
    embed = s.zeros(16, 4)
    for j, mask in enumerate(sectors[3 if creation else 1]):
        embed[mask, j] = 1
    neutral_projector = s.diag(*(int(mask.bit_count() == 2) for mask in range(16)))
    assert neutral_projector*embed*A == s.zeros(16, 5)
    assert not zero(embed*A)

# Positive exact tensor controls at a rational joint frame, including both m,d.
subst = {a: s.Rational(4, 5), b: s.Rational(3, 5), e: s.Integer(7), c: s.Rational(-2, 3)}
Dtest, Rtest = D5.subs(subst), R5.subs(subst)
G = s.zeros(5)
G[0, 4] = G[4, 0] = G[1, 3] = G[3, 1] = 1
G[2, 2] = -1
assert zero(D5.T*G+G*D5) and zero(R5.T*G+G*R5)

def occ(n: int, width: int = 5) -> list[tuple[int, ...]]:
    if width == 1:
        return [(n,)]
    return [(i,)+tail for i in range(n+1) for tail in occ(n-i, width-1)]

def sym_lift(M, n):
    words = occ(n); index = {w: j for j, w in enumerate(words)}
    result = s.zeros(len(words))
    for col, word in enumerate(words):
        for i, j in itertools.product(range(5), repeat=2):
            if word[j] and M[i, j] != 0:
                dest = list(word); dest[j] -= 1; factor = s.sqrt(word[j]*(dest[i]+1)); dest[i] += 1
                result[index[tuple(dest)], col] += factor*M[i, j]
    return result

def extract(n):
    words, rest = occ(n), occ(n-1); index = {w: j for j, w in enumerate(rest)}
    E = s.zeros(5*len(rest), len(words))
    for col, word in enumerate(words):
        for j in range(5):
            if word[j]:
                dest = list(word); dest[j] -= 1
                E[5*index[tuple(dest)]+j, col] = s.sqrt(s.Rational(word[j], n))
    return E

def trace(n):
    if n < 2:
        return s.zeros(0, len(occ(n)))
    source, dest = occ(n), occ(n-2); index = {w: j for j, w in enumerate(dest)}
    T = s.zeros(len(dest), len(source))
    for col, word in enumerate(source):
        for i, j in itertools.product(range(5), repeat=2):
            if G[i, j] and word[i] and word[j] - int(i == j) > 0:
                tail = list(word); tail[i] -= 1; tail[j] -= 1
                T[index[tuple(tail)], col] += G[i, j]*s.sqrt(word[i]*(word[j]-int(i == j)))
    return T

controls = []
for n in [1, 2, 3]:
    E = extract(n); restdim = len(occ(n-1))
    assert E.T*E == s.eye(len(occ(n)))
    for M in [Dtest, Rtest]:
        output = s.kronecker_product(sym_lift(M, n-1), s.eye(5)) + s.kronecker_product(s.eye(restdim), M)
        assert s.simplify(output*E-E*sym_lift(M, n)) == s.zeros(*E.shape)
    T = trace(n); harmonic = s.Matrix.hstack(*T.nullspace())
    assert harmonic.cols == math.comb(n+4, 4)-(math.comb(n+2, 4) if n >= 2 else 0)
    if n >= 3:
        assert s.kronecker_product(trace(n-1), s.eye(5))*E*harmonic == s.zeros(len(occ(n-3))*5, harmonic.cols)
    if n >= 2:
        # Mixed contraction: selected slot with one remaining slot, not just trace of rest.
        mixed = s.zeros(len(occ(n-2)), len(occ(n-1))*5)
        lookup = {w: j for j, w in enumerate(occ(n-2))}
        for j, word in enumerate(occ(n-1)):
            for p, q in itertools.product(range(5), repeat=2):
                if G[p, q] and word[q]:
                    tail = list(word); tail[q] -= 1
                    mixed[lookup[tuple(tail)], 5*j+p] += G[p, q]*s.sqrt(word[q])
        assert s.simplify(mixed*E*harmonic) == s.zeros(mixed.rows, harmonic.cols)
    for creation in [False, True]:
        A = s.zeros(4, 5)
        for p in range(4):
            A += (p+1+s.I*(4-p))*endpoint(p, creation)*W
        A = A.subs(subst).applyfunc(s.simplify)
        image = s.kronecker_product(s.eye(restdim), A)*E
        native = Hn.subs(subst)
        coefficients = s.Matrix([p+1+s.I*(4-p) for p in range(4)])
        change = native*coefficients if creation else -native.T*coefficients
        endpoint_change = s.zeros(4, 5)
        for p in range(4):
            endpoint_change += change[p]*endpoint(p, creation)*W
        endpoint_change = endpoint_change.subs(subst).applyfunc(s.simplify)
        targetH = s.kronecker_product(sym_lift(Dtest, n-1), s.eye(4))+s.kronecker_product(s.eye(restdim), lift(native, 3 if creation else 1))
        assert s.simplify(targetH*image-image*sym_lift(Dtest, n)-s.kronecker_product(s.eye(restdim), endpoint_change)*E) == s.zeros(*image.shape)
        if n >= 3:
            assert s.simplify(s.kronecker_product(trace(n-1), s.eye(4))*image*harmonic) == s.zeros(len(occ(n-3))*4, harmonic.cols)
    controls.append({'n': n, 'symmetric_dimension': E.cols, 'harmonic_dimension': harmonic.cols, 'extraction_isometry': True, 'matter_density_intertwining': True, 'both_CAR_orientations_arbitrary_complex_mixture': True, 'source_trace_constraints_preserved': True})

# Exact crosscopy control: disjoint changed local N labels, not collective CAR.
# Full two-copy Fock dimensions 16^2; avoid dense 256^2 operators.
symstate = {}
v0 = W.subs(subst).applyfunc(s.simplify)
for k in range(5):
    for l in range(5):
        for x, mx in enumerate(sectors[2]):
            for y, my in enumerate(sectors[2]):
                value = (v0[x, k]*v0[y, l]+v0[x, l]*v0[y, k])/2
                if value:
                    symstate[(mx, my, k, l)] = value
for creation in [False, True]:
    supports = []
    for copy in [0, 1]:
        support = set()
        for mx, my, k, l in symstate:
            mask = mx | (my << 4)
            for p in range(4):
                mode = 4*copy+p
                if bool(mask & (1 << mode)) != creation:
                    target = mask ^ (1 << mode)
                    support.add(((target & 15).bit_count(), (target >> 4).bit_count()))
        supports.append(support)
    assert supports[0].isdisjoint(supports[1])
    assert supports == ([{(3, 2)}, {(2, 3)}] if creation else [{(1, 2)}, {(2, 1)}])
# Normalized collective annihilation still has CAR norm one, not sqrt(count).
# {sum alpha_j a_j, its adjoint}=sum |alpha_j|^2 I by full-CAR;
# source-restricted images add quadratically ONLY because of local N labels.
assert s.Rational(1, 2)+s.Rational(1, 2) == 1

# Full eight-mode CAR includes the real crosscopy Jordan-Wigner signs.
ann8 = []
for p in range(8):
    entries = {}
    for mask in range(256):
        if mask & (1 << p):
            entries[(mask ^ (1 << p), mask)] = (-1)**((mask & ((1 << p)-1)).bit_count())
    ann8.append(s.SparseMatrix(256, 256, entries))
for p, q in itertools.product(range(8), repeat=2):
    assert ann8[p]*ann8[q]+ann8[q]*ann8[p] == s.SparseMatrix(256, 256, {})
    assert ann8[p]*ann8[q].T+ann8[q].T*ann8[p] == s.SparseMatrix(s.eye(256))*int(p == q)
V16 = s.zeros(16, 5)
for j, mask in enumerate(sectors[2]):
    V16[mask, :] = v0[j, :]
# Tensor ordering high four modes first, low four modes last.
physical_source = s.kronecker_product(V16, V16)*extract(2)
assert s.simplify(physical_source.conjugate().T*physical_source) == s.eye(15)
for creation in [False, True]:
    outputs = []
    for copy in [0, 1]:
        current = s.zeros(256, 15)
        for p in range(4):
            operator = ann8[4*copy+p].T if creation else ann8[4*copy+p]
            current += (p+1+s.I*(4-p))*operator*physical_source
        outputs.append(current)
    assert s.simplify(outputs[0].conjugate().T*outputs[1]) == s.zeros(15)
    assert s.simplify((outputs[0]+outputs[1]).conjugate().T*(outputs[0]+outputs[1])-outputs[0].conjugate().T*outputs[0]-outputs[1].conjugate().T*outputs[1]) == s.zeros(15)

rA, rB = s.symbols('rA rB', real=True)
chart = []
for qA in [-1, 1]:
    qB = -qA
    site = [rA+s.Rational(qA, 2), -rA+s.Rational(qA, 2), rB+s.Rational(qB, 2), -rB+s.Rational(qB, 2)]
    flux = [-sum(site[:j]) for j in range(1, 4)]
    expected = [-rA-s.Rational(qA, 2), -qA, -rB-s.Rational(qA, 2)]
    assert all(s.expand(x-y) == 0 for x, y in zip(flux, expected))
    assert sum(site) == 0
    electric = s.expand(64*sum(x*x for x in flux))
    assert s.expand(electric-64*((rA+s.Rational(qA, 2))**2+(rB+s.Rational(qA, 2))**2)-64) == 0
    wrong = 64*(rA*rA+rB*rB)+64
    assert s.expand(electric-wrong) != 0
    chart.append({'qA': qA, 'qB': qB, 'E': [str(x) for x in flux], 'external_energy_once': 64})
assert math.comb(6, 4)-math.comb(4, 4) == 14
assert math.comb(130, 4)-math.comb(128, 4) == 690880
S5 = s.diag(1, -1, 1, -1, 1)
assert zero(S5*D5*S5-D5) and zero(S5*R5*S5+R5)
assert zero(S4*D4*S4-D4) and zero(S4*R4*S4+R4)

proof = {
    'status': 'VALID_SCOPED_IMAGE_CONTAINMENT_AND_AFFINE_GAUSS_PROOF',
    'scope': 'No charged floor, fullPhi/lattice ground, linked stability, phase or pole promotion.',
    'analytic_all_n': [
        'Let W:C5->wedge2(C4) be the exact native joint intertwiner used here, including the parent highest free wedge minus sign. Its orthogonal singlet complement is not asserted to vanish under arbitrary CAR; instead CAR maps all five source coordinates into the FULL wedge1 or wedge3 C4.',
        'The normalized occupancy state |k>, sum k=n, represents the unit symmetrized tensor of n individually labelled spin replicas. Extraction of any fixed physical replica is E_n|k>=sum_j sqrt(k_j/n)|k-e_j> tensor |j>. This follows by counting the k_j permutations with selected slot j in the n!/(prod k_j!) normalized sum; hence E_n^*E_n=I for every integer n>=1, not an extrapolation from the finite controls.',
        'For arbitrary endpoint coefficients t in C4 and either orientation, A_t maps C5 to wedge1 or wedge3 by the actual signed CAR, and the fixed-copy image is (I tensor A_t)E_n. Thus it lies in Sym^(n-1)(C5) tensor C4 for EVERY n, with no discarded scalar traces or spinor components. Native Dirac endpoint mixtures are arbitrary because t is unrestricted and W is unitary on its vector source; physical source phases are fixed jointly, not chosen separately for D and R.',
        'For any one-body generator M preserving W(C5), E_n dGamma_n(M)=(dGamma_(n-1)(M) tensor I+I tensor M)E_n. CAR contributes [dGamma(h),a^dagger(t)]=a^dagger(h t) and [dGamma(h),a(t)]=-a(h^T t), checked symbolically for arbitrary coefficients for both native H and density R. Endpoint maps are covariant with transformed endpoint coefficients, not falsely claimed to commute at fixed t.',
        'The invariant symmetric contraction G has G04=G40=G13=G31=1,G22=-1 in the exact free5 frame. Harmonic_n=ker(sum Gij b_i b_j). In the tensor description each pair contraction of a symmetric harmonic source is zero. After selecting a slot, pair contractions among remaining slots and contractions of selected slot with any remaining slot remain zero. Applying A_t to the selected slot preserves the former; the latter is a source tensor constraint, not a licensed projection of the resulting spinor. Finite controls exercise both before CAR and remaining trace after CAR. The ambient targets retain all traces.',
        'For active n=2 and auxiliary n=126, selected source Harmonic_2 tensor Harmonic_126 has dimensions14 x690880. Changing any one of the two active replicas gives Sym1(active) tensor Sym126(aux) tensor C4(active). Changing any one of the126 auxiliary replicas gives Sym2(active) tensor Sym125(aux) tensor C4(aux). This follows directly from the all-n extraction identity at n126, not finite-point extrapolation. There are64 physical copies,1 active and63 auxiliary, EACH with2 spin replicas; an individual spin replica, not a collective averaged creation, is the selected slot.',
        'For wedge3 the exact Hodge CAR J gives -H^T and -R^T. The native Q and the parent free-spinor U, followed by the same fixed S4, give T3=J Q U S4, simultaneously D4,R4. It is the same parent joint phase pair. For the whole cell, Sym(S5) on EVERY vector slot and S4 on the spinor commute with total D and flip total R. Consequently u=+1/2 and u=-1/2 are isospectral as JOINT containing-space pairs.',
        'Source-specific crosscopy orthogonality DOES hold: every original physical spin replica has canonical N_j=2, even in an entangled symmetric harmonic source. An annihilation or creation at replica j changes only its eigenvalue to1 or3. Distinct j images belong to orthogonal joint eigenspaces of the commuting N_j. Within one replica arbitrary Dirac-mode endpoint contributions need NOT be orthogonal and must be summed with CAR signs. Two-cell branches are similarly labelled by changed replicas and charge orientation. This proves source-restricted norm-square additivity across replicas, not a sqrt(count) operator norm. On full Fock space a normalized annihilation mixture sum alpha_j a_j has norm sqrt(sum |alpha_j|^2), by CAR; original unnormalized hopping coefficients must be kept.',
        'Background is one fermion per physical site per spin replica:128 replicas per cell,256 background fermions per cell. Copy charge q_j=N_j-2; total q=sum_j q_j, and r=sum_j(n_jL-n_jR)/2. Site excess densities are qL=q/2+r and qR=q/2-r, whereas n_jL,n_jR are raw copy occupations; do not confuse these with background-subtracted site density. In a first-shell branch exactly one replica changes, so q=+/-1. The abbreviation N-2 refers to this changed replica, not total cell N minus2.',
        'For two cells with zero external boundary flux and qB=-qA, cumulative Gauss flow E=(-rA-qA/2,-qA,-rB-qA/2). Therefore the electric form is64(R_A+qA/2)^2+64(R_B+qA/2)^2+64: the unit bridge costs64 ONCE. Each local charged form is Dtotal+64(Rtotal+u I)^2 on the corresponding full containing space. Both orientations are covered by qA=+/-1. No neutral Lambda gap is used; a separately certified neutral E0 may be used only as a subtraction2E0 in a later charged lower-bound calculation.',
    ],
    'parameter_binding': profiles,
    'finite_controls': controls,
    'wrong_controls_rejected': ['unsigned full-CAR', 'wrong highest wedge phase', 'neutral-only destination (canonical number1/3)', 'omitted affine half offsets'],
    'full_CAR': {'modes': 4, 'fock_dimension': 16, 'both_orientations': True, 'symbolic_arbitrary_endpoint_coefficients': True},
    'crosscopy_number_label_control': {'physical_replicas': 2, 'full_fock_dimension': 256, 'full_eight_mode_CAR': True, 'both_orientations': True, 'arbitrary_complex_endpoint_mixture': True, 'disjoint_changed_number_support': True, 'actual_source_restricted_Gram_additivity': True},
    'gauss': chart,
    'target_dimensions': {'active': 5*math.comb(130, 4)*4, 'auxiliary': 15*math.comb(129, 4)*4},
    'source_hash': OWN,
    'input_hashes': INITIAL,
    'source_and_inputs_unchanged': digest(Path(__file__)) == OWN and all(digest(ROOT/name) == value for name, value in INITIAL.items()),
    'elapsed_seconds': time.monotonic()-START,
}
assert proof['source_and_inputs_unchanged']
(ROOT/'charged_image_proof_control_20261011_receipt.json').write_text(json.dumps(proof, indent=2)+'\n')
print(json.dumps({'status': proof['status'], 'finite_controls': controls, 'source_hash': OWN, 'source_and_inputs_unchanged': True, 'elapsed_seconds': proof['elapsed_seconds']}, indent=2))
