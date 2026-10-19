"""Independent exact modular native controls; no historical producer is invoked."""
from __future__ import annotations
import argparse
import hashlib
import io
import itertools
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy import sparse
import sympy as sp
P = 65537
J = 256
DENOMINATORS = set()


def residue(v):
    re, im = sp.expand(v).as_real_imag()
    out = 0
    for factor, part in ((1, re), (J, im)):
        r = sp.Rational(part)
        DENOMINATORS.add(int(r.q))
        assert int(r.q) % P
        out += factor * int(r.p) * pow(int(r.q), -1, P)
    return out % P


def field_matrix(m):
    return np.array([[residue(x) for x in row] for row in m.tolist()], dtype=np.int64)


def witness(m):
    a = np.asarray(m, dtype=np.int64).copy() % P
    rows = list(range(a.shape[0])); pivrows = []; columns = []; product = 1; swaps = 0
    r = 0
    for c in range(a.shape[1]):
        candidates = np.flatnonzero(a[r:, c])
        if not len(candidates):
            continue
        k = r + int(candidates[0])
        if k != r:
            a[[r, k]] = a[[k, r]]; rows[r], rows[k] = rows[k], rows[r]; swaps += 1
        pivrows.append(rows[r]); columns.append(c)
        value = int(a[r, c]); product = product * value % P
        a[r] = a[r] * pow(value, -1, P) % P
        for rr in range(r + 1, len(a)):
            if a[rr, c]:
                a[rr] = (a[rr] - a[rr, c] * a[r]) % P
        r += 1
        if r == len(a):
            break
    # Pivot-product nonzero certifies the indexed minor; independent determinant
    # recomputation below binds the recorded original row/column selection.
    minor = np.asarray(m)[np.ix_(pivrows, columns)].copy() % P
    determinant = 1
    for k in range(r):
        choices = np.flatnonzero(minor[k:, k]); assert len(choices)
        j = k + int(choices[0])
        if j != k:
            minor[[k, j]] = minor[[j, k]]; determinant = -determinant
        v = int(minor[k, k]); determinant = determinant * v % P
        inv = pow(v, -1, P)
        for rr in range(k + 1, r):
            if minor[rr, k]:
                minor[rr] = (minor[rr] - minor[rr, k] * inv % P * minor[k]) % P
    assert determinant % P
    return {'rank': r, 'pivot_rows': pivrows, 'pivot_columns': columns,
            'selected_minor_determinant_mod_prime': int(determinant % P),
            'matrix_sha256_mod_prime': hashlib.sha256(np.asarray(m, dtype=np.int64).tobytes()).hexdigest()}


def assemble(profiles, phase_mode):
    sites = list(itertools.product(range(4), repeat=3)); positions = {x:j for j,x in enumerate(sites)}
    points = [(3,0,0),(0,0,0),(0,1,0),(3,1,0)]
    phi = {}
    for x in points:
        position = positions[x]; fields = []
        for mu in range(4):
            b = sp.diag(3+mu+position%5, -1-mu, -1-position%5, -1)
            for a in range(4):
                for c in range(a+1,4):
                    v = sp.Rational((1+position+mu+a+2*c)%7+1, 11+mu+a+c)
                    if (position+mu+a+c)%2: v *= sp.I
                    b[a,c] = v; b[c,a] = sp.conjugate(v)
            assert b.H == b and sp.trace(b) == 0
            fields.append(b)
        phi[x] = fields
    edges = [(points[0],points[1],0,False),(points[1],points[2],1,False),
             (points[3],points[2],0,True),(points[0],points[3],1,True)]
    legs = []; links = []; phases = []
    for x,y,axis,reverse in edges:
        u = sp.eye(4)
        for step,(a,b) in enumerate(((0,1),(1,2),(2,3),(0,2))):
            rot = sp.eye(4); alpha = sp.Rational(3,5)
            beta = sp.Rational(4,5)*(sp.I if (sum(x)+axis+step)%2 else 1)
            rot[a,a] = rot[b,b] = alpha; rot[a,b] = beta; rot[b,a] = -sp.conjugate(beta)
            u = (u*rot).applyfunc(sp.expand)
        assert (u.H*u).applyfunc(sp.expand) == sp.eye(4) and sp.simplify(u.det()) == 1
        links.append(u)
        phase = ((3+4*sp.I)/5 if (sum(x)+axis)%2 else (5+12*sp.I)/13) if phase_mode == 'original' else sp.Integer(1)
        phases.append(phase)
        path = next(p for p in profiles[1]['paths'] if p['axis']==axis and p['length']==1)
        c0,ci = sp.Rational(path['c0']), sp.I*sp.Rational(path['ci'])
        fx,fy = c0*phi[x][0]+ci*phi[x][axis+1], c0*phi[y][0]+ci*phi[y][axis+1]
        body = ((-1 if x[axis]==3 else 1)*phase*(fx*u+u*fy)/2).applyfunc(sp.expand)
        t = sp.zeros(16)
        r,c = (points.index(y),points.index(x)) if reverse else (points.index(x),points.index(y))
        t[4*r:4*r+4,4*c:4*c+4] = body.H if reverse else body
        legs.append(t)
    assert any(links[0]*u != u*links[0] for u in links[1:])
    flux = sp.simplify(phases[0]*phases[1]*sp.conjugate(phases[2]*phases[3]))
    return legs, {'closed_face_Peierls_holonomy':str(flux), 'raw_phases':[str(v) for v in phases],
                  'all4Phi_traceless_Hermitian':True, 'noncommuting_SU4':True, 'AP_edge_signs':[-1,1,-1,1]}


def ordered_sum(legs, adjoint=False):
    n = legs[0].rows; eye = sparse.eye(n,dtype=np.int64,format='csr')
    ads=[]
    for leg in legs:
        t=field_matrix(leg.H if adjoint else leg)
        ads.append(sparse.kron(sparse.csr_matrix(t),eye,format='csr')-sparse.kron(eye,sparse.csr_matrix(t.T),format='csr'))
    out=np.zeros((n*n,n*n),dtype=np.int64)
    for order in itertools.permutations(range(4)):
        product=sparse.eye(n*n,dtype=np.int64,format='csr')
        for slot in order:
            product=ads[slot]@product; product.data%=P; product.eliminate_zeros()
        out=(out+product.toarray())%P
    return out


def cross_sum(legs):
    out=np.zeros((16,16),dtype=np.int64)
    for dagger in (False,True):
        mats=[field_matrix(t.H if dagger else t) for t in legs]
        for order in itertools.permutations(range(4)):
            product=np.eye(16,dtype=np.int64)
            for j in order: product=mats[j]@product%P
            out=(out+product)%P
    return out*pow(2,-1,P)%P


def controls(legs, global_block=False):
    b=ordered_sum(legs); bh=ordered_sum(legs,True)
    a=(b+bh)*pow(2,-1,P)%P; cross=cross_sum(legs)
    eye=np.eye(16,dtype=np.int64).reshape(-1)
    assert not np.any(a@eye%P) and not np.any(b@eye%P) and not np.any(bh@eye%P)
    result={'End16_A':witness(a),'End16_joint_B_Bdag':witness(np.vstack((b,bh))), 'cross16':witness(cross), 'upper_rank255_by_I16':True}
    assert result['End16_A']['rank']==255 and result['cross16']['rank']==16
    if global_block:
        extended=[]
        for t in legs:
            v=sp.zeros(18);v[:16,:16]=t;extended.append(v)
        aa=(ordered_sum(extended)+ordered_sum(extended,True))*pow(2,-1,P)%P
        ff=[18*r+c for r in range(16) for c in range(16)]
        fx=[18*r+c for r in range(16) for c in range(16,18)]
        xf=[18*r+c for r in range(16,18) for c in range(16)]
        xx=[18*r+c for r in range(16,18) for c in range(16,18)]
        assert np.array_equal(aa[np.ix_(ff,ff)],a)
        assert np.array_equal(aa[np.ix_(fx,fx)],np.kron(cross,np.eye(2,dtype=np.int64)))
        assert np.array_equal(aa[np.ix_(xf,xf)],np.kron(np.eye(2,dtype=np.int64),cross.T))
        assert not np.any(aa[xx]) and not np.any(aa[:,xx])
        assert witness(aa)['rank']==319
        result['independent_End18_block_control']={'rank':319,'nullity':5,'EndF':255,'both_cross_blocks':32,'EndX_annihilated':True,'not_fullM_rank255':True}
    return result


def faces(n):
    sites=list(itertools.product(range(n),repeat=3)); allfaces=[]
    for x in sites:
        for a,b in ((0,1),(0,2),(1,2)):
            def shift(axes):
                y=list(x)
                for axis in axes:y[axis]=(y[axis]+1)%n
                return tuple(y)
            allfaces.append({x,shift([a]),shift([b]),shift([a,b])})
    assert all(len(f)==4 for f in allfaces)
    for x in sites:
        containing=[f for f in allfaces if x in f]
        assert set.intersection(*containing)=={x}
    return {'N':n,'translated_oriented_face_count':len(allfaces),'all_face_vertices_distinct':True,'intersection_all_faces_through_site_is_site':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    request=json.loads(args.request.read_text())
    with zipfile.ZipFile(request['preexecution']) as archive:
        for name,pin in request['pins'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==pin
        snap=zipfile.ZipFile(io.BytesIO(archive.read('ground_loop_support_parent_review_postexecution_snapshot_20261012.zip')))
        first=zipfile.ZipFile(io.BytesIO(snap.read('ground_loop_support_actual1_20261012_preexecution.zip')))
        original=zipfile.ZipFile(io.BytesIO(first.read('joined_phi_u1_uniform_gauss_limit_epoch4_20261010_actual_executed_inputs.zip')))
        profiles=json.loads(original.read('joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json'))['primitive_profiles']
    assert sp.isprime(P) and J*J%P==P-1
    assert profiles[1]['paths']==profiles[2]['paths']
    observations={}
    for mode in ('original','physical_zero_transverse'):
        legs,metadata=assemble(profiles,mode);observations[mode]={'native_input':metadata,'ranks':controls(legs,mode!='original')}
    result={'kind':'INDEPENDENT_NATIVE_MODULAR_AND_ACTUAL_ZERO_TRANSVERSE_QUOTIENT_WITNESSES_NOT_GROUND_RESPONSE',
            'prime':P,'i_image':J,'denominators':sorted(DENOMINATORS),'all_denominators_invertible':True,
            'observations':observations,'face_overlap_controls':[faces(n) for n in (4,6,8)],
            'all24_explicit_permutations':True,'all64_identity':'E(t)X(g)=X([t,g]) via CAR and E(t)P_S=0; symbolic all multiplicities, not enumerated Fock vectors',
            'fullM_local_kernel':'span(I_F) directsum End(X); sl(M) dimension (M-16)^2',
            'not_higher_irrep_classification':True,'not_originating_weight':True}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'End16_rank':255,'cross_rank':16,'physical_zero_transverse_witness':True}))


if __name__=='__main__':main()
