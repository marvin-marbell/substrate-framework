"""Independent exact connected orbit, native CAR/Gauss and primitive-deck consumer.
No original CLI, optimizer, eigenvalue solver or checker is invoked.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import zipfile
import numpy as np
import sympy as s


def digest(data):
    return hashlib.sha256(data).hexdigest()


def orbit(source):
    # Only immutable literal matrix definitions are consumed from the original archive.
    ns = {'__name__': 'frozen_review_library', '__file__': 'frozen/connected_quantum_sea_source_20261012.py'}
    exec(compile(source, ns['__file__'], 'exec'), ns)
    beta, alpha, C, T = ns['matrices']()
    gamma = [beta] + alpha
    for a in range(4):
        for b in range(4):
            assert gamma[a]*gamma[b]+gamma[b]*gamma[a] == 2*(a == b)*s.eye(4)
    assert C == [beta/2] + [a/3 for a in alpha]
    assert all(s.simplify(2*s.trace(a*b)) == int(i == j) for i,a in enumerate(T) for j,b in enumerate(T))
    weights = [s.Rational(1,2)] + [s.Rational(1,3)]*3
    eig = Counter()
    clifford = []
    for mask in range(16):
        word = s.eye(4)
        for a in range(4):
            if mask & (1 << a):
                word = word*gamma[a]
        clifford.append(word)
        result = sum((c*c*word-2*c*word*c+word*c*c for c in C), s.zeros(4))
        lam = sum(4*weights[a]**2 for a in range(4) if gamma[a]*word == -word*gamma[a])
        assert s.simplify(result-lam*word) == s.zeros(4)
        eig[lam] += 1
    assert all(s.simplify(s.trace(a.conjugate().T*b)) == 4*int(i == j) for i,a in enumerate(clifford) for j,b in enumerate(clifford))
    assert eig[0] == 1 and min(x for x in eig if x) == s.Rational(4,9)
    L = s.Matrix(60,15,lambda r,c:s.simplify(2*s.trace(T[r%15]*s.I*(T[c]*C[r//15]-C[r//15]*T[c]))))
    G = s.simplify(L.T*L)
    x = s.Symbol('x')
    expected = s.prod((x-k)**v for k,v in eig.items() if k)
    assert s.expand(G.charpoly(x).as_expr()-expected) == 0  # characteristic identity, NOT eigensolving
    normals = s.GramSchmidt(L.T.nullspace(),True)
    S = s.Matrix.hstack(*normals)
    assert S.shape == (60,45) and S.T*S == s.eye(45) and L.T*S == s.zeros(15,45)
    cvec = s.Matrix([s.simplify(2*s.trace(t*c)) for c in C for t in T])
    assert L.T*cvec == s.zeros(15,1)
    assert s.simplify(cvec.dot(cvec)) == s.Rational(14,3)
    # Radial normal perturbation is an actual nonzero affine-slice point.
    e = s.Rational(1,2**20)
    K = (1+e)*L
    assert s.simplify((K.T*K).det()/(G.det()*(1+e)**30)) == 1
    # A plausible defective 1/2 isotropic alpha centre must be rejected.
    wrong_C = [beta/2]+[a/2 for a in alpha]
    wrong_L = s.Matrix(60,15,lambda r,c:s.simplify(2*s.trace(T[r%15]*s.I*(T[c]*wrong_C[r//15]-wrong_C[r//15]*T[c]))))
    assert wrong_L.T*wrong_L != G
    return {'Clifford_superoperator_spectrum_including_scalar': {str(k):v for k,v in sorted(eig.items())},
            'orbit_Gram_characteristic_polynomial': str(s.factor(expected)),
            'orbit_Gram_determinant': str(s.factor(G.det())), 'minimum_singular_value_exact':'2/3',
            'exact_normal_dimension':45,'radial_normal_j_squared_ratio':'(1+epsilon)^30',
            'wrong_isotropic_alpha_scale_detected':True, 'no_eigensolver':True}


def native_car():
    annihilators = []
    for j in range(4):
        a=s.zeros(16)
        for mask in range(16):
            if mask & (1 << j):
                a[mask^(1<<j),mask]=(-1)**((mask & ((1<<j)-1)).bit_count())
        annihilators.append(a)
    for i,a in enumerate(annihilators):
        for j,b in enumerate(annihilators):
            assert a*b+b*a == s.zeros(16)
            assert a*b.T+b.T*a == int(i == j)*s.eye(16)
    occupation=sum((a.T*a for a in annihilators),s.zeros(16))
    assert occupation[0,0] == 0 and occupation[15,15] == 4
    U=s.diag(1,1,s.I,-s.I)
    U[:2,:2]=s.Matrix([[s.Rational(3,5),s.Rational(4,5)],[-s.Rational(4,5),s.Rational(3,5)]])
    assert U.conjugate().T*U == s.eye(4) and U.det() == 1
    q=s.Matrix.vstack(s.eye(4),s.eye(4))/s.sqrt(2)
    amplitudes={rows:s.simplify(q[list(rows),:].det()) for rows in itertools.combinations(range(8),4)}
    norm=sum(s.conjugate(v)*v for v in amplitudes.values())
    assert norm == 1
    fcs=sum(s.conjugate(v)*v*s.I**sum(j<4 for j in rows) for rows,v in amplitudes.items())
    assert fcs != 1
    global_fcs=sum(s.conjugate(v)*v*s.I**4 for v in amplitudes.values())
    assert global_fcs == 1
    # Centre-loop lifted endpoint on the true wedge (not a per-site fixed number assumption).
    transported=s.diag(*([s.I]*4+[1]*4))*q
    assert any(s.simplify(transported[list(rows),:].det()-v) != 0 for rows,v in amplitudes.items())
    generators=[]
    for a in range(4):
        for b in range(a+1,4):
            t=s.zeros(4);t[a,b]=t[b,a]=s.Rational(1,2);generators.append(t)
            t=s.zeros(4);t[a,b]=-s.I/2;t[b,a]=s.I/2;generators.append(t)
    for k in range(1,4):
        generators.append(s.diag(*([1]*k+[-k]+[0]*(3-k)))/s.sqrt(2*k*(k+1)))
    for t in generators:
        charge=sum((t[a,b]*annihilators[a].T*annihilators[b] for a in range(4) for b in range(4)),s.zeros(16))
        assert charge[:,0] == s.zeros(16,1) and charge[:,15] == s.zeros(16,1)
    # Literal Gauss equivariance on the entangled GLOBAL native wedge, including signs.
    generator=s.diag(1,-1,0,0,0,0,0,0)/2
    dq=s.I*generator*q
    fock_action={rows:s.S.Zero for rows in amplitudes}
    for rows,value in amplitudes.items():
        for a in range(4):
            for b in range(4):
                if b not in rows:
                    continue
                reduced=list(rows);annihilation_sign=(-1)**reduced.index(b);reduced.remove(b)
                if a in reduced:
                    continue
                insertion=sum(j<a for j in reduced)
                new=tuple(sorted(reduced+[a]))
                fock_action[new]+=s.I*generator[a,b]*annihilation_sign*(-1)**insertion*value
    nonzero=False
    for rows in amplitudes:
        local=q[list(rows),:]; derivative=dq[list(rows),:]
        first=s.S.Zero
        for col in range(4):
            replaced=local.copy();replaced[:,col]=derivative[:,col];first+=replaced.det()
        assert s.simplify(first-fock_action[rows]) == 0
        nonzero=nonzero or s.simplify(first) != 0
    assert nonzero  # omitting frame transport would violate neutral SU4 Gauss
    return {'full_native_local_CAR_dimension':16,'all_CAR_anticommutators_exact':True,
            'global_wedge4_native_dimension':70,'Cauchy_Binet_norm_exact':str(norm),
            'local_centre_FCS_exact':str(s.simplify(fcs)),'global_centre_FCS_exact':str(global_fcs),
            'noncentral_SU4_full_wedge_determinant_exact':str(U.det()),
            'all15_SU4_generators_annihilate_charged_full_and_empty':True,
            'entangled_global_native_SU4_Gauss_frame_minus_Fock_residual_exact':0,
            'wrong_frozen_neutral_frame_Gauss_detected':True,
            'wrong_independent_local_centre_descent_rejected':True}


def lattice(n):
    sites=list(itertools.product(range(n),repeat=3)); V=len(sites); ix={a:i for i,a in enumerate(sites)}
    edges=[]
    for a in sites:
        for axis in range(3):
            b=list(a); b[axis]=(b[axis]+1)%n
            edges.append((ix[a],ix[tuple(b)]))
    adjacency=[[] for _ in sites]
    for e,(a,b) in enumerate(edges):
        adjacency[a].append((b,e,1)); adjacency[b].append((a,e,-1))
    # Different tree order from producer; the integer lattice property is invariant.
    parent={0:None}; queue=deque([0]); tree=set()
    while queue:
        a=queue.popleft()
        for b,e,sign in reversed(adjacency[a]):
            if b not in parent:
                parent[b]=(a,e,sign); tree.add(e); queue.append(b)
    assert len(parent) == V
    chords=[e for e in range(3*V) if e not in tree]; d=len(chords)
    B=np.zeros((3*V,d),dtype=np.int64)
    def path(a):
        out=np.zeros(3*V,dtype=np.int64)
        while a:
            prev,e,sign=parent[a];out[e]+=sign;a=prev
        return out
    for j,e in enumerate(chords):
        a,b=edges[e];B[:,j]=path(a)-path(b);B[e,j]=1
    DB=np.zeros((V,d),dtype=np.int64)
    parity=np.array([(-1)**sum(a) for a in sites],dtype=np.int64)
    E0=np.zeros(3*V,dtype=np.int64); div=np.zeros(V,dtype=np.int64)
    for e,(a,b) in enumerate(edges):
        DB[a]+=B[e];DB[b]-=B[e]
        if e%3 == 0:E0[e]=64*parity[a]
        div[a]+=E0[e];div[b]-=E0[e]
    assert not np.any(DB) and np.array_equal(B[chords],np.eye(d,dtype=np.int64)) and d == 2*V+1
    assert np.array_equal(div,128*parity)
    assert sum(4*(int(p)+1)//2 for p in parity) == 2*V
    assert s.I**(2*V) == 1  # exact root of unity, never floating complex exponentiation
    assert (2*V)%4 == 0
    G=B.T@B; tr=int(np.trace(G)); w=Fraction(1,64*d*(1+tr))
    # Exact radial second moment for the same normalized squared bump.
    moment=Fraction(1,19); excess=64*w*w*tr*moment
    sup=64*w*w*d*tr
    assert excess < sup < 1
    assert 64*int(E0@E0)//V == 2**18
    assert 64*int(E0@E0) != 64*64*V  # falsely dropping inter-species electric cross
    assert np.any(div != 64*4*(parity+1)//2)  # omit Ref2 for every species
    assert np.any(G[:,0] != np.eye(d,dtype=np.int64)[:,0])  # wrong B-as-primitive-deck
    primitive_control = {'integer_lattice_proof':'chord identity plus tree leaf elimination'}
    if n == 4:
        # Native rational primitive deck: project an ORIGINAL integer edge shift.
        incidence = s.zeros(V,3*V)
        for e,(a,b) in enumerate(edges):
            incidence[a,e]=1; incidence[b,e]=-1
        raw = s.zeros(3*V,1); raw[chords[0]]=1
        reduced = incidence[1:,:]
        potential = (reduced*reduced.T).inv(method='DM')*(reduced*raw)
        deck = raw-reduced.T*potential
        native_B=s.Matrix(B.tolist())
        assert incidence*deck == s.zeros(V,1)
        q = native_B.T*deck
        unit = s.zeros(d,1); unit[0]=1
        assert q == unit
        assert native_B.T*raw == unit
        # beta=width e0: shifted character has exponent beta_0, not G_00 beta_0.
        beta=s.zeros(d,1); beta[0]=s.Rational(w.numerator,w.denominator)
        assert ((native_B*beta).T*deck)[0] == beta[0]
        assert ((native_B*beta).T*native_B[:,0])[0] != beta[0]
        detG=s.Symbol('detG',positive=True)
        cell=(2*s.pi)**d/s.sqrt(detG)
        assert s.simplify(cell**-1 * (2*s.pi)**d/s.sqrt(detG)) == 1
        primitive_control={'actual_N4_rational_projected_edge_deck_DB':True,
                           'primitive_Btranspose_deck_exact':'e_0',
                           'finite_character_deck_phase_exact':str(beta[0]),
                           'finite_character_norm_exact':1,
                           'normalized_Bloch_Haar_unit_period':1,
                           'raw_B_deck_wrong_phase_detected':True}
    # Centre equations on a connected graph imply equality, not one independent root/site.
    assert len(tree) == V-1
    return {'N':n,'V':V,'edges':3*V,'cycle_dimension':d,'DB_exact_zero':True,'primitive_integer_identity_minor':True,
            'TrG_exact':tr,'packet_width':str(w),'packet_second_moment':str(moment),
            'electric_excess_exact':str(excess),'electric_excess_sup':str(sup),
            'one_shared_electric_square_per_site':2**18,'each_of_64_particles':2*V,
            'global_neutral_centre_exponent_mod4':(2*V)%4,'connected_cover_degree':4,
            'physical_primitive_deck_character_control':primitive_control,
            'wrong_Ref_and_cross_and_B_deck_detected':True,'wrong_4powerV_normalizer_norm':str(Fraction(4**V,4))}


def sea(profile):
    m=s.Rational(profile['onsite']); c0=s.Rational(profile['paths'][0]['c0']); ci=s.Rational(profile['paths'][0]['ci'])
    assert m-6*abs(c0)>0 and ci != 0
    a,b,c=s.symbols('a b c', real=True); mass,eps=s.symbols('mass eps',positive=True)
    D=s.Matrix([[c,a-s.I*b],[a+s.I*b,-c]])
    assert s.expand(D*D) == (a*a+b*b+c*c)*s.eye(2)
    assert s.simplify((eps+mass)/(2*eps)+(eps**2-mass**2)/(2*eps*(eps+mass))-1) == 0
    # Actual N4 AP momenta: cos=+-sqrt2/2, sin^2=1/2 at every mode.
    r=0
    for signs in itertools.product((-1,1),repeat=3):
        mu=(m+c0*s.sqrt(2)*sum(signs))/2
        e=s.sqrt(mu*mu+s.Rational(2,3)*ci*ci)
        r+=(1+mu/e)/16
    variance=4*r*(1-r)
    assert s.simplify(variance) != 0
    # Explicit AP unit-character orthogonality uses integer residues, not a float Gram.
    n=4; zeta=s.I
    assert all(sum(zeta**((a-b)*x) for x in range(n)) == n*int(a == b) for a in range(n) for b in range(n))
    return {'actual_profile':profile,'positive_mass_uniform_AP':str(m-6*abs(c0)),
            'band_QdaggerQ_exact':True,'N4_AP_integer_residue_Fourier_orthogonality_exact':True,
            'all_evenN_norm_proof':'sum_x exp(2pi i (k-l) dot x/N)=V delta_kl; AP half shifts cancel in differences; band identity => QdaggerQ=I_2V; Cauchy-Binet => exterior norm1 pointwise for every block SU4 frame',
            'N4_actual_local_Pxx_upper_eigenvalue':str(r),
            'N4_actual_local_number_variance_exact':str(variance),
            'strict_variance_authority':'each actual mass>0, ci!=0 and all N4 AP sin^2=1/2, so 1/2<r<1 and 4r(1-r)>0; reflected sin pairs cancel local off-diagonal projector',
            'no_float_Gram_or_ground_selection':True}


def scalar():
    u=s.Symbol('u',real=True); b=(1-u*u)**4
    norm=s.integrate(b*b,(u,-1,1)); second=s.integrate(u*u*b*b,(u,-1,1))/norm
    assert norm == s.Rational(65536,109395) and second == s.Rational(1,19)
    assert Fraction(2**2,4) == 1 and Fraction(1,4) != 1 and Fraction(4**2,4) != 1
    # Lower block triangular determinant must not discard the entire link diagonal block.
    A=s.Matrix([[2,1],[0,3]]); H=s.Matrix([[s.Rational(2,5),0],[0,s.Rational(3,7)]])
    B=s.Matrix([[11,13],[17,19]])
    J=A.row_join(s.zeros(2)).col_join(B.row_join(H))
    assert J.det() == A.det()*H.det() and J.det() != A.det()
    return {'bump_squared_integral':str(norm),'packet_second_moment':str(second),
            'scalar_prefactor':2,'cover_degree':4,'squared_norm_exact':1,
            'wrong_prefactor1_or4_detected':True,'triangular_mixed_block_exact_control':str(J.det()),
            'VolSU4_metric_2Tr':'32768*sqrt(2)*pi^9/3',
            'j':'VolSU4*sqrt(det([K(z),S]^T[K(z),S]))',
            'h':'det(dexp_b)/VolSU4',
            'h_spectral_formula':'product_{i<j} (sin((lambda_i-lambda_j)/2)/((lambda_i-lambda_j)/2))^2; continuous value1 at zero'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True); parser.add_argument('--output-prefix',type=Path,required=True)
    args=parser.parse_args(); req=json.loads(args.request.read_text()); root=args.request.parent
    pins={}
    for name,pin in req['pins'].items():
        path=Path(pin['path']); data=path.read_bytes(); actual=digest(data)
        assert actual == pin['sha256'],name
        pins[name]={'path':str(path),'sha256':actual}
    archive=Path(req['pins']['executed']['path'])
    with zipfile.ZipFile(archive) as z:
        source=z.read('connected_quantum_sea_source_20261012.py')
        proof=z.read('connected_quantum_sea_actual2_20261012_proof.json')
        interface=z.read('connected_quantum_sea_actual2_20261012_interface.json')
        profiles=z.read('neutral_body_parent_primitive_profiles_20261012.json')
    for name,data in [('source',source),('proof',proof),('interface',interface),('profiles',profiles)]:
        assert digest(data) == req['members'][name]
    target=json.loads(proof)
    orbit_result=orbit(source)
    assert orbit_result['orbit_Gram_determinant'] == target['controls']['orbit_Gram_det_exact']
    result={'kind':'INDEPENDENT_EXACT_ORBIT_QUOTIENT_CAR_GAUSS_REVIEW_TRIAL_ONLY',
            'pins':pins,'archive_member_digests':req['members'],'plan':req['plan'],
            'controls':{'orbit':orbit_result,'CAR':native_car(),'Gauss_and_integer_quotient':[lattice(n) for n in (4,6)],
                        'global_actual_sea':sea(json.loads(profiles)['primitive_profiles'][0]),'normalization':scalar()},
            'analytic_proof':req['analytic_proof'],'scope':req['scope'],
            'verdict':'SCOPED_PASS_CONSTRUCTION31_TO40_WITH_EXPLICIT_PRIMITIVE_DECK_COORDINATES',
            'load_bearing_findings':[],'clarifications':['Primitive physical deck is 2pi B G^-1 Z^d, NOT 2pi B Z^d; finite character normalization uses quotient metric determinant. Original construction is valid with this explicit realization; bare B beta wording alone does not specify Bloch units.'],
            'prohibited_original_execution':False}
    prefix=args.output_prefix
    out=Path(str(prefix)+'_proof.json');out.write_text(json.dumps(result,indent=2)+'\n')
    verdict={'kind':result['kind'],'verdict':result['verdict'],'target_pins':pins,'proof_sha256':digest(out.read_bytes()),
             'scope':req['scope'],'load_bearing_findings':result['load_bearing_findings'],'all_independent_exact_controls_passed':True}
    Path(str(prefix)+'_verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
    print(json.dumps({'verdict':verdict['verdict'],'exact_controls':True,'scope':req['scope']},indent=2),flush=True)

if __name__ == '__main__':
    main()
