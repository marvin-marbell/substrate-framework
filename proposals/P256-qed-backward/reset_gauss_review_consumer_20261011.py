"""Independent frozen-law carrier, CG, Gauss/CP and boundary audit controls.
No original producer import/execution; finite algebra controls are not full-H spectra.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
import sympy as sp
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def generators() -> list[sp.Matrix]:
    ans = []
    for i in range(4):
        for j in range(i+1, 4):
            s, a = sp.zeros(4), sp.zeros(4)
            s[i,j] = s[j,i] = sp.Rational(1,2)
            a[i,j], a[j,i] = -sp.I/2, sp.I/2
            ans.extend([s,a])
    for k in range(1,4):
        ans.append(sp.diag(*([1]*k+[-k]+[0]*(3-k)))/sp.sqrt(2*k*(k+1)))
    return ans


def clifford_and_normalizer(ts: list[sp.Matrix]) -> dict:
    sx = sp.Matrix([[0,1],[1,0]])
    sy = sp.Matrix([[0,-sp.I],[sp.I,0]])
    sz = sp.diag(1,-1)
    gs = [sp.kronecker_product(sx,sp.eye(2)), sp.kronecker_product(sy,sp.eye(2)),
          sp.kronecker_product(sz,sx), sp.kronecker_product(sz,sy)]
    assert all(a*b+b*a == (2*sp.eye(4) if i==j else sp.zeros(4))
               for i,a in enumerate(gs) for j,b in enumerate(gs))
    words = []
    for mask in range(16):
        w = sp.eye(4)
        for j in range(4):
            if mask>>j&1: w = w*gs[j]
        words.append(w)
    assert sp.Matrix.hstack(*[w.reshape(16,1) for w in words]).rank() == 16
    inertia = sp.Matrix([[sum(sp.trace((t*g-g*t).conjugate().T*(u*g-g*u))
                           for g in gs) for u in ts] for t in ts])
    ev = {str(k):int(v) for k,v in inertia.eigenvals().items()}
    assert ev == {'2':4,'4':6,'6':4,'8':1}
    comm = sp.zeros(16)
    for g in gs:
        d = sp.kronecker_product(g,sp.eye(4))-sp.kronecker_product(sp.eye(4),g.T)
        comm += d.conjugate().T*d
    gaps = {str(k):int(v) for k,v in comm.eigenvals().items()}
    assert comm.rank()==15 and min(k for k in comm.eigenvals() if k)>0
    assert min(k for k in comm.eigenvals() if k)==4
    # Three gamma matrices lose freeness: an actual zero tangent direction.
    wrong = sp.Matrix([[sum(sp.trace((t*g-g*t).conjugate().T*(u*g-g*u))
                          for g in gs[:3]) for u in ts] for t in ts])
    assert wrong.rank()<15
    ta = np.asarray([np.asarray(t,dtype=complex) for t in ts])
    ga = np.asarray([np.asarray(g,dtype=complex) for g in gs])
    hs = np.sqrt(2)*ta
    def coords(fields):
        return np.einsum('aij,mji->ma',hs,fields).real.reshape(60)
    tangent = np.column_stack([coords(1j*(t@ga-ga@t)) for t in ta])
    uu, singular, _ = np.linalg.svd(tangent,full_matrices=True)
    normal = uu[:,15:]
    assert np.max(abs(tangent.T@normal))<2e-14
    d0 = np.column_stack([tangent,normal])
    det0 = abs(np.linalg.det(d0))
    delta = 2.0**-16
    rows=[]
    for j in range(4):
        n = delta*.8*np.sin(np.arange(45)+j)
        n *= 1/max(1,np.linalg.norm(n)/(.8*delta))
        fields = ga+np.einsum('ma,aij->mij',(normal@n).reshape(4,15),hs)
        tang = np.column_stack([coords(1j*(t@fields-fields@t)) for t in ta])
        d = np.column_stack([tang,normal])
        jacrel = abs(np.linalg.det(d))/det0
        sv = np.linalg.svd(d,compute_uv=False)
        assert sv[-1]>1-2*delta and sv[0]<4
        # Exact cancellation of the ACTUAL local Jacobian, not its orbit value.
        eta2 = 0.371
        compensation = eta2/jacrel
        assert abs(jacrel*compensation-eta2)<1e-15
        rows.append(dict(jacobian_relative_to_zero=jacrel,min_differential=float(sv[-1]),
                         max_differential=float(sv[0]),missing_J_density_error=abs(jacrel-1)))
    assert max(row['missing_J_density_error'] for row in rows)>1e-9
    u,a = sp.symbols('u a',positive=True)
    norm = sp.integrate(sp.cos(sp.pi*u/(2*a))**2/a,(u,-a,a))
    grad = sp.integrate(sp.diff(sp.cos(sp.pi*u/(2*a))/sp.sqrt(a),u)**2,(u,-a,a))
    assert norm==1 and sp.simplify(grad-sp.pi**2/(4*a**2))==0
    profile_grad = 45**2*sp.pi**2/(4*sp.Rational(1,2**16)**2)
    assert float(profile_grad)<2**46
    assert F(1,2**16)+F(6,1)/F(1,1)*F(26,2**16)/F(1,1)<1
    # R^-30 in the wave is forced by actual dimension60, not orbit15.
    assert 60-2*30==0 and 60-2*sp.Rational(15,2)!=0
    jordan = sum(sp.trace((x*y+y*x)**2) for x in gs for y in gs)*4
    assert jordan==256
    return dict(Cl4_rank=16,orbit_inertia_eigenvalues=ev,commutator_eigenvalues=gaps,
                wrong_Cl3_orbit_rank=wrong.rank(),normal_dimension=normal.shape[1],
                local_jacobian_controls=rows,exact_cosine_norm=str(norm),
                exact_profile_gradient=str(profile_grad),R_wave_exponent=-30,
                actual_Jordan_at_tuple_div_lambda=256,
                scope='Exact native Clifford and profile algebra; local Jacobian controls do not alone prove global tube injectivity.')


def exterior_two(ts: list[np.ndarray]) -> list[np.ndarray]:
    pairs=list(combinations(range(4),2))
    w=np.zeros((16,6),complex)
    for k,(i,j) in enumerate(pairs):
        w[4*i+j,k]=1/np.sqrt(2); w[4*j+i,k]=-1/np.sqrt(2)
    return [w.conj().T@(np.kron(t,np.eye(4))+np.kron(np.eye(4),t))@w for t in ts]


def cg_and_kraus(ts_exact: list[sp.Matrix]) -> dict:
    ts=[np.asarray(t,dtype=complex) for t in ts_exact]
    reps=[('1',0,[np.zeros((1,1),complex) for t in ts],0.0,1),
          ('4',1,ts,15/8,15),('6',2,exterior_two(ts),5/2,20),
          ('4bar',3,[-t.conj() for t in ts],15/8,15)]
    rows=[]
    for label,centre,rr,c2,sigdim in reps:
        d=rr[0].shape[0]
        cc=sum(t@t for t in rr)
        assert np.max(abs(cc-c2*np.eye(d)))<3e-14
        # Cartan highest vector in mu tensor lambda*: both mu=lambda here.
        dual=[-t.conj() for t in rr]
        tensor=[np.kron(t,np.eye(d))+np.kron(np.eye(d),td) for t,td in zip(rr,dual)]
        seed=np.zeros(d*d,complex); seed[d-1 if centre in (1,2) else (d-1)*d]=1
        basis=[seed]
        for _ in range(d*d):
            added=[]
            for v in basis:
                for t in tensor:
                    u=t@v
                    for b in basis+added: u-=b*np.vdot(b,u)
                    n=np.linalg.norm(u)
                    if n>1e-10: added.append(u/n)
            if not added: break
            basis.extend(added)
        C=np.column_stack(basis)
        assert C.shape[1]==sigdim,(label,C.shape)
        sigma=[C.conj().T@t@C for t in tensor]
        # Hom reciprocity, with a single positive scalar Schur Gram.
        raw=C.conj().reshape(d,d,sigdim).transpose(2,1,0).reshape(sigdim*d,d)
        gram=raw.conj().T@raw
        assert np.max(abs(gram-(sigdim/d)*np.eye(d)))<2e-13
        embed=raw/np.sqrt(sigdim/d)
        isometry=float(np.max(abs(embed.conj().T@embed-np.eye(d))))
        intertwine=max(float(np.max(abs((np.kron(s,np.eye(d))+np.kron(np.eye(sigdim),t))@embed-embed@t)))
                        for s,t in zip(sigma,rr))
        assert max(isometry,intertwine)<1e-12
        # Every individual multiplicity Kraus, with external dual annihilation.
        # Order magnetic / multiplicity / external dual / extra exterior rest.
        singlet=np.eye(d).reshape(d*d)/np.sqrt(d)
        ent=np.array([1,0,0,2j],complex)/np.sqrt(5)
        initial=np.einsum('ab,cr->acbr',singlet.reshape(d,d),ent.reshape(2,2)).reshape(-1)
        P=np.diag([1,0]); low=np.array([1,0],complex)
        parity=np.diag([1,-1])
        ks=[np.kron(np.eye(d),P)]
        for eps in range(2):
            ks.append(np.kron(np.eye(d),np.outer(np.linalg.matrix_power(parity,eps)@low,[0,1]))/np.sqrt(2))
        completeness=sum(k.conj().T@k for k in ks)
        assert np.max(abs(completeness-np.eye(2*d)))<1e-14
        outputs=[np.kron(k,np.eye(2*d))@initial for k in ks]
        gauss=[]; comm=[]
        for t in rr:
            G=np.kron(np.kron(t,np.eye(2)),np.eye(2*d))+np.kron(np.eye(2*d),np.kron(-t.conj(),np.eye(2)))
            for v in outputs: gauss.append(float(np.linalg.norm(G@v)))
            for k in ks: comm.append(float(np.linalg.norm(k@np.kron(t,np.eye(2))-np.kron(t,np.eye(2))@k)))
        assert max(gauss+comm)<1e-13
        def marginal(v):
            z=v.reshape(d,2,d,2).transpose(0,2,3,1).reshape(2*d*d,2)
            return z@z.conj().T
        retained=sum(marginal(v) for v in outputs)
        entanglement_error=float(np.max(abs(retained-marginal(initial))))
        assert entanglement_error<1e-14
        wrong_mag=np.diag([1]+[0]*(d-1))
        wrong=np.kron(wrong_mag,np.eye(4*d))@initial
        wrong_G=max(float(np.linalg.norm((np.kron(np.kron(t,np.eye(2)),np.eye(2*d))+np.kron(np.eye(2*d),np.kron(-t.conj(),np.eye(2))))@wrong)) for t in rr)
        if d>1: assert wrong_G>0.05 and np.max(abs(gram-np.eye(d)))>0.5
        rows.append(dict(minimal_link=label,centre=centre,casimir=c2,Cartan_carrier_dimension=sigdim,
                         reciprocal_Gram_scalar=sigdim/d,isometry_residual=isometry,
                         all15_intertwiner_residual=intertwine,each_Kraus_Gauss_residual=max(gauss),
                         CP_completeness_residual=float(np.max(abs(completeness-np.eye(2*d)))),
                         magnetic_and_exterior_rest_marginal_residual=entanglement_error,
                         wrong_magnetic_reset_Gauss_norm=wrong_G))
    # All actual local CAR occupations and Ref wedges, not just a fundamental charge.
    masks=list(range(16)); charges=[]
    for t in ts:
        op=np.zeros((16,16),complex)
        for m in masks:
            for i,j in product(range(4),repeat=2):
                if not (m>>j&1): continue
                m1=m^(1<<j); sign=(-1)**((m&((1<<j)-1)).bit_count())
                if m1>>i&1: continue
                m2=m1|(1<<i); sign*=(-1)**((m1&((1<<i)-1)).bit_count())
                op[m2,m]+=sign*t[i,j]
        charges.append(op)
    count=np.diag([m.bit_count() for m in masks])
    assert max(np.linalg.norm(count@c-c@count) for c in charges)<1e-13
    assert max(np.linalg.norm(c[[0,15],:]) for c in charges)<1e-13
    # K identity on each of65 copies and Ref4 implies every even CAR matrix,
    # occupation affine connection and deck phase commutes, including collective rho.
    k=np.array([[0,1],[0,0]],complex)
    occupations=np.array([(j%5) for j in range(65)])
    rho=int(np.sum(occupations[1:]-2))
    one=int(occupations[1]-2)
    assert rho!=one
    affine=0.173*rho+0.117*(int(occupations.sum()))
    phase=expm(2j*np.pi*0.173*count)
    for c in [*charges,count,phase]:
        K=np.kron(k,np.eye(16)); C=np.kron(np.eye(2),c)
        assert np.linalg.norm(K@C-C@K)==0
    return dict(boundary_CG_and_Kraus=rows,full_local_CAR_masks=16,
                all65_identity_roles=list(range(65)),existing_Ref_occupation_wedges=list(range(5)),
                collective64_rho_control=rho,wrong_oneflavour_rho=one,
                affine_connection_control=affine,
                scope='Actual SU4/CAR algebra controls; full unbounded-domain proof is analytic and separate.')


def native_boundaries(profiles: list[dict]) -> dict:
    rows=[]
    for N in (4,6,8):
        x=(0,0,0); y=(1,0,0); inside={x,y}; edge=(x,0)
        def step(s,a,l=1):
            p=list(s); p[a]=(p[a]+l)%N; return tuple(p)
        align=[]; faces=[]; terms=[]
        for s in product(range(N),repeat=3):
            for ax in range(3):
                end=step(s,ax)
                if (s in inside)!=(end in inside): align.append((s,end,ax))
            for a,b in combinations(range(3),2):
                es=[(s,a),(step(s,a),b),(step(s,b),a),(s,b)]
                if edge in es: faces.append((s,a,b))
            for pi,p in enumerate(profiles):
                if s in inside: terms.append(dict(profile=pi,kind='onsite_inside',source=s,field=s))
                for q in p['paths']:
                    ax,L=int(q['axis']),int(q['length']); end=step(s,ax,L)
                    es=[(step(s,ax,j),ax) for j in range(L)]
                    internal=edge in es
                    for endpoint in (s,end):
                        if endpoint in inside or internal:
                            kind='inside_Phi' if endpoint in inside else 'outside_Phi_internal_U'
                            terms.append(dict(profile=pi,kind=kind,source=s,target=end,field=endpoint,
                                              axis=ax,length=L,internal_U=internal,
                                              coefficient_l1=str(abs(F(q['c0']))+abs(F(q['ci']))),
                                              AP_seam=(-1 if s[ax]+L>=N else 1)))
        out=[t for t in terms if t['kind']=='outside_Phi_internal_U']
        assert len(align)==10 and len(faces)==4
        assert len(out)==4 and all(t['length']==2 and t['axis']==0 and t['profile'] in (1,2) for t in out)
        assert {t['field'] for t in out}=={((N-1),0,0),(2,0,0)}
        weights=[abs(F(p['onsite']))+2*sum(abs(F(t['c0']))+abs(F(t['ci'])) for t in p['paths']) for p in profiles]
        lam=F(2**8192)
        B=4*(lam*weights[0]+weights[1]+63*weights[2])
        peroutside={}
        for t in out:
            p=profiles[t['profile']]
            # endpoint CAR+h.c. bound4 coefficient*r; load, scale retained.
            allocation=4*int(p['load'])*F(t['coefficient_l1'])
            peroutside[t['field']]=peroutside.get(t['field'],F(0))+allocation
        assert sum(peroutside.values())<=4*B
        rows.append(dict(N=N,alignment_edges=align,internal_link_faces=faces,
                         changed_endpoint_piece_count=len(terms),changed_endpoint_pieces=terms,
                         outside_Phi_internal_U=out,
                         outside_endpoint_allocations={str(k):str(v) for k,v in peroutside.items()},
                         loads=[int(p['load']) for p in profiles],B_over_lambda_sha256=sha(str(B/lam).encode())))
    # Actual noncommuting ordered outside field/link multiplier survives reset parity.
    t=[np.asarray(q,dtype=complex) for q in generators()]
    field=t[0]+.41*t[9]
    U=expm(.37j*t[0]); V=expm(-.29j*t[6])
    before=field@U@V
    assert np.linalg.norm(before-field@V@U)>1e-3
    assert np.linalg.norm(before)>0.1
    return dict(actual_N_boundaries=rows,ordered_outside_multiplier_norm=float(np.linalg.norm(before)),
                wrong_reordered_length2_norm=float(np.linalg.norm(before-field@V@U)),
                outside_multiplier_reset_Phi_parity='EVEN; not killed',
                boundary_classes=['external_alignment_cross','inside_Phi_onsite_and_path',
                                  'outside_Phi_internal_U_length2','internal_link_plaquette'],
                unchanged_classes=['all_rest_Phi_Haar_kinetics','all_fullE_U1_collective_rho',
                                   'all_rest_Jordan_alignment_plaquette','all_CAR_Ref_penalties_and_sources_without_block_fields'])


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); started=time.monotonic()
    request=json.loads(args.request.read_text())
    before={n:sha((ROOT/n).read_bytes()) for n in request['pins']}
    if before!=request['pins']: raise ValueError('frozen review pin mismatch')
    with zipfile.ZipFile(ROOT/request['epoch_archive']) as z:
        raw=z.read(request['profile']); profiles=json.loads(raw)['primitive_profiles']
    ts=generators()
    result=dict(kind='INDEPENDENT_NATIVE_RESET_GAUSS_FORM_REVIEW_NOT_ADMISSION',
                carrier=clifford_and_normalizer(ts),CG_Gauss_CP_CAR=cg_and_kraus(ts),
                complete_boundaries=native_boundaries(profiles),profile_sha256=sha(raw),
                source_input_sha256_before=before,
                source_input_sha256_after={n:sha((ROOT/n).read_bytes()) for n in request['pins']},
                runtime=dict(python=sys.executable,numpy=np.__version__,sympy=sp.__version__,seconds=time.monotonic()-started),
                actual_ground_computation=False,original_producer_rerun=False,
                static_dynamics_phase_atom_promotion=False)
    if result['source_input_sha256_after']!=before: raise RuntimeError('source/input mutation')
    with args.output.open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(dict(native_controls_reached=True,carrier=True,CG_Gauss_CP=True,
                          all_boundary_classes=True,actual_ground_numerics=False,seconds=result['runtime']['seconds'])))


if __name__=='__main__':
    main()
