"""New individual-CAR consumer of immutable native harmonic source; no solver."""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, time
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast
import numpy as np
import sympy as s
from scipy import sparse
ROOT = Path(__file__).resolve().parent
PINS = {
 'local_so5_resum_source_20261011.py':'7f73170ca3a33782b5b391a6f6d3b6e0e3b7bf385c2f03cc2b5f7696fde8e009',
 'local_so5_resum_actual04_20261011_state.npz':'dd06400e1b7f70df935cf7ca02cf5ebf598481a8d5e209d1241aba5ed3fea2a2',
 'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json':'102966f32c49f40d79d55343925a630e30a9cf7e4a8074cf8628142ba9c5b6e8',
 'charged_image_proof_control_20261011.py':'de8173ae6d2a59eb7bab6f7457b04a98c6735941beafacc2e4ef50792430c500',
 'charged_image_proof_control_20261011_receipt.json':'c45fd28aee5c1cd50c2e7350fc3ba5a276029fd118073da80655d88007975590',
 'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json':'2f019dd49873a5e13f130c65ef6739b9af78a16f85138dc6c6ab6e32e65c7a12',
 'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz':'f8a0793adcbf4909402cd089471aa8c33fc74a31b66e99618f8baaab8d6dfb65'}
EXTRA = ['charged_first_shell_actual04_20261011_receipt.json','charged_output_lower_attempt3_20261011_receipt.json','joined_phi_u1_harmonic_transverse_affine_boundary_control_20261011_receipt.json']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def labels(n): return [(l,p,q) for l in range(n+1) for p in range(l+1) for q in range(l+1)]
def dim(n): return (n+1)*(n+2)*(2*n+3)//6
def idx(l,p,q): return l*(l+1)*(2*l+1)//6+p*(l+1)+q

def generators(n, exact=False) -> list[Any]:
    ls=labels(n); d=len(ls); out=[s.zeros(d) for _ in range(4)] if exact else [sparse.lil_matrix((d,d),dtype=complex) for _ in range(4)]
    sqrt=s.sqrt if exact else np.sqrt
    for j,(l,p,q) in enumerate(ls):
        out[0][j,j]=2*(p+q-l);out[3][j,j]=p-q
        for axis,k in [(0,p),(1,q)]:
            if k<l:
                i=idx(l,p+1,q) if axis==0 else idx(l,p,q+1)
                v=(s.I if exact else 1j)*sqrt((l-k)*(k+1));out[2][i,j]=v;out[2][j,i]=-v
        if l<n:
            for dp,dq,sign in [(1,0,1),(0,1,-1)]:
                rad=F((n-l)*(n+l+3)*(p+1 if dp else l-p+1)*(q+1 if dq else l-q+1),(l+1)*(l+2))
                v=sign*(s.sqrt(s.Rational(rad.numerator,rad.denominator)) if exact else np.sqrt(float(rad)));i=idx(l+1,p+dp,q+dq)
                out[1][i,j]=v;out[1][j,i]=v
    return out if exact else [x.tocsr() for x in out]

def extraction(n, exact=False) -> list[Any]:
    ds,dt=dim(n),dim(n-1);rows=[[] for _ in range(5)];cols=[[] for _ in range(5)];vals=[[] for _ in range(5)]
    def add(k,i,j,r,sign=1):
        if r:
            rows[k].append(i);cols[k].append(j)
            vals[k].append(sign*(s.sqrt(s.Rational(r.numerator,r.denominator)) if exact else math.sqrt(float(r))))
    for j,(l,p,q) in enumerate(labels(n)):
        if l<n: add(0,idx(l,p,q),j,F((n-l)*(n+l+2),n*(2*n+1)))
        for a,b in itertools.product(range(2),repeat=2):
            k=1+2*a+b
            sp=(l-p if a==0 else p);sq=(l-q if b==0 else q)
            if l and sp and sq:
                add(k,idx(l-1,p-a,q-b),j,F((n+l+1)*(n+l+2)*sp*sq,2*n*l*(l+1)*(2*n+1)))
            if l<n-1:
                tp=(p+1 if a==0 else l-p+1);tq=(q+1 if b==0 else l-q+1)
                add(k,idx(l+1,p+1-a,q+1-b),j,F((n-l)*(n-l-1)*tp*tq,2*n*(l+1)*(l+2)*(2*n+1)),(-1)**(a+b))
    if exact:
        out=[]
        for r,c,v in zip(rows,cols,vals):
            z=s.zeros(dt,ds)
            for i,j,w in zip(r,c,v):z[i,j]=w
            out.append(z)
        return out
    return [sparse.csr_matrix((v,(r,c)),shape=(dt,ds)) for r,c,v in zip(rows,cols,vals)]

def occupancy(n):
    if n==0:return [(0,)*5]
    return [w for w in itertools.product(range(n+1),repeat=5) if sum(w)==n]
def polynomial_basis(n):
    x=s.symbols('z x00 x01 x10 x11'); Q=x[1]*x[4]-x[2]*x[3];words=occupancy(n);cols=[]
    for l,p,q in labels(n):
        psi=0
        for t in range(l+1):
            k=(l-p-q+t,q-t,p-t,t)
            if min(k)<0:continue
            fac=s.sqrt(s.Rational(math.factorial(p)*math.factorial(q)*math.factorial(l-p)*math.factorial(l-q),math.factorial(l)))
            psi+=fac*s.prod(x[j+1]**k[j]/math.factorial(k[j]) for j in range(4))
        poly=0
        for r in range((n-l)//2+1):
            c=s.Rational(math.factorial(n-l),2**r*math.factorial(r)*math.factorial(n-l-2*r)*s.rf(l+2,r))
            poly+=c*x[0]**(n-l-2*r)*Q**r*psi
        pd=s.Poly(s.expand(poly),*x);v=s.Matrix([pd.coeff_monomial(w)*s.sqrt(math.prod(math.factorial(k) for k in w)) for w in words]);norm=s.simplify((v.T*v)[0]);cols.append(v/s.sqrt(norm))
    return x,words,s.Matrix.hstack(*cols)
def fock_lift(M, words):
    look={w:i for i,w in enumerate(words)};out=s.zeros(len(words))
    for j,w in enumerate(words):
        for a,b in itertools.product(range(5),repeat=2):
            if M[a,b] and w[b]:
                t=list(w);t[b]-=1;v=s.sqrt(w[b]*(t[a]+1));t[a]+=1;out[look[tuple(t)],j]+=M[a,b]*v
    return out

def car_exact():
    ann=[]
    for p in range(4):
        A=s.zeros(16)
        for m in range(16):
            if m>>p&1:A[m^(1<<p),m]=(-1)**((m&((1<<p)-1)).bit_count())
        ann.append(A)
    sectors={n:[m for m in range(16) if m.bit_count()==n] for n in [1,2,3]}
    V=s.zeros(16,5)
    for j,m in enumerate([5,3,12,10]):V[m,j]=1
    V[9,4]=V[6,4]=1/s.sqrt(2)
    A=s.diag(-1,1,-1,1);X=s.Matrix([[0,1],[1,0]]);B=s.kronecker_product(X,s.diag(1,-1));C=s.zeros(2).row_join(s.I*X).col_join((-s.I*X).row_join(s.zeros(2)));R=s.diag(s.Rational(1,2),s.Rational(1,2),-s.Rational(1,2),-s.Rational(1,2))
    def lift(h):return sum((ann[p].T*ann[q]*h[p,q] for p,q in itertools.product(range(4),repeat=2)),s.zeros(16))
    ops=[lift(h) for h in [A,B,C,R]];native=[s.simplify(V.T*o*V) for o in ops];g1=generators(1,True)
    variables=s.symbols('u0:25');J=s.Matrix(5,5,variables);eq=[]
    for a,b in zip(native,g1):eq+=list(a*J-J*b)
    eq+=list(J[:,idx(1,0,0)]-s.eye(5)[:,0]);sol=s.linsolve(eq,variables)
    assert isinstance(sol,s.FiniteSet) and len(sol.args)==1
    solution=sol.args[0]
    assert isinstance(solution,s.Tuple)
    J=J.subs(dict(zip(variables,solution.args)))
    assert s.simplify(J.conjugate().T*J)==s.eye(5)
    for a,b in zip(native,g1):assert s.simplify(a*J-J*b)==s.zeros(5)
    wrong=J*s.diag(1,1,-1,1,1)
    assert s.simplify(native[1]*wrong-wrong*g1[1])!=s.zeros(5)
    for p,q in itertools.product(range(4),repeat=2):
        assert ann[p]*ann[q].T+ann[q].T*ann[p]==s.eye(16)*int(p==q)
    unsigned=[z.applyfunc(abs) for z in ann]
    assert unsigned[0]*unsigned[1].T+unsigned[1].T*unsigned[0]!=s.zeros(16)
    maps=[]
    slotphase=s.diag(1,-s.I,1,1,s.I)
    for creation in [False,True]:
        maps.append([(a.T if creation else a).extract(sectors[3 if creation else 1],range(16))*V*J*slotphase for a in ann])
        assert s.simplify(cast(Any,sum((z.conjugate().T*z for z in maps[-1]),s.zeros(5))))==2*s.eye(5)
    return J,maps,[A,B,C,R],ops,ann,V

def exact_controls():
    J,maps,h4,ops,ann,V=car_exact();rows=[]
    for n in [1,2,3]:
        _,words,U=polynomial_basis(n);_,rest,Un=polynomial_basis(n-1);Es=extraction(n,True);g=generators(n,True);gn=generators(n-1,True);one=generators(1,True)
        assert s.simplify(U.T*U)==s.eye(dim(n))
        for a,b in zip(g,one):assert s.simplify(fock_lift(b,words)*U-U*a)==s.zeros(len(words),dim(n))
        assert s.simplify(cast(Any,sum((e.T*e for e in Es),s.zeros(dim(n)))))==s.eye(dim(n))
        for k,e in enumerate(Es):
            raw=s.zeros(len(rest),len(words));look={w:i for i,w in enumerate(rest)}
            for j,w in enumerate(words):
                if w[k]:
                    t=list(w);t[k]-=1;raw[look[tuple(t)],j]=s.sqrt(s.Rational(w[k],n))
            assert s.simplify(raw*U-Un*e)==s.zeros(len(rest),dim(n))
        for gn0,g0,h in zip(gn,g,one):
            for k,e in enumerate(Es):
                assert s.simplify(gn0*e+sum((h[k,j]*Es[j] for j in range(5)),s.zeros(*e.shape))-e*g0)==s.zeros(*e.shape)
        if n>=2:
            _,rr,Ur=polynomial_basis(n-2);look={w:i for i,w in enumerate(rr)};T=s.zeros(len(rr),len(words))
            metric=[(0,0,1),(1,4,-1),(4,1,-1),(2,3,1),(3,2,1)]
            for j,w in enumerate(words):
                for a,b,v in metric:
                    if w[b] and w[a]-int(a==b)>0:
                        t=list(w);t[b]-=1;c=s.sqrt(w[b]*t[a]);t[a]-=1;T[look[tuple(t)],j]+=v*c
            assert s.simplify(T*U)==s.zeros(len(rr),dim(n));assert T.rank()==len(rr)
            assert T*T.T!=s.zeros(len(rr))
        wrong=Es.copy();wrong[0]=-wrong[0]
        if n>=2:assert s.simplify(gn[1]*wrong[2]+sum((one[1][2,j]*wrong[j] for j in range(5)),s.zeros(*wrong[2].shape))-wrong[2]*g[1])!=s.zeros(*wrong[2].shape)
        rows.append({'n':n,'polynomial_orthonormal':True,'trace_kernel_complete':True,'normalized_annihilator_matches_polynomial_derivative':True,'joint_native_A_B_C_R_intertwining':True,'wrong_phase_rejected':n>=2})
    # Exact full-CAR affine commutator action, both charges and arbitrary endpoint mixtures.
    H=3*ops[0]-ops[1]+4*ops[2];rho=ops[3];h=3*h4[0]-h4[1]+4*h4[2]
    for creation in [False,True]:
        for u in [s.Rational(-1,2),s.Rational(1,2)]:
            for p in range(4):
                a=ann[p].T if creation else ann[p];delta=(1 if creation else -1)*h4[3][p,p]+u
                change=sum(((h[q,p] if creation else -h[p,q])*(ann[q].T if creation else ann[q]) for q in range(4)),s.zeros(16))
                left=(H+64*(rho+u*s.eye(16))**2)*a
                right=a*(H+64*rho**2)+change+64*a*(2*delta*rho+delta**2*s.eye(16))
                assert s.simplify(left-right)==s.zeros(16)
    return {'small_n':rows,'joint_n1_native_embedding':[[str(z) for z in row] for row in J.tolist()], 'signed_CAR_both_orientations_arbitrary_complex_mixtures':True,'unsigned_CAR_rejected':True,'complete_affine_H_commutator_identity':True,'traceful_complement_rejected':True},np.array(J.evalf(),complex),[[np.array(z.evalf(),complex) for z in row] for row in maps]

def native_matrix(n,m,c,d):
    size=dim(n);rows=np.empty(9*size,np.int32);cols=np.empty(9*size,np.int32);values=np.empty(9*size);rho=np.empty(size);used=0
    def add(r,c0,v):
        nonlocal used
        count=len(r);rows[used:used+count]=r;cols[used:used+count]=c0;values[used:used+count]=v;used+=count
    for l in range(n+1):
        p,q=np.meshgrid(np.arange(l+1),np.arange(l+1),indexing='ij');p=p.reshape(-1);q=q.reshape(-1);j=idx(l,p,q);rho[j]=p-q
        add(j,j,2*m*(p+q-l))
        for axis,k in [(0,p),(1,q)]:
            ok=k<l;i=idx(l,p[ok]+int(axis==0),q[ok]+int(axis==1));v=d*np.sqrt((l-k[ok])*(k[ok]+1));add(i,j[ok],v);add(j[ok],i,v)
        if l<n:
            for dp,dq,sign in [(1,0,1),(0,1,-1)]:
                i=idx(l+1,p+dp,q+dq);v=sign*c*np.sqrt((n-l)*(n+l+3)*(p+1 if dp else l-p+1)*(q+1 if dq else l-q+1)/((l+1)*(l+2)));add(i,j,v);add(j,i,v)
    return sparse.csr_matrix((values[:used],(rows[:used],cols[:used])),shape=(size,size)),rho
def serialize(M):return [[[float(z.real),float(z.imag)] for z in row] for row in np.asarray(M)]
def quadratic(C,K):
    # C: endpoints x charged component x (field,slot), K complete streamed Gram.
    return np.einsum('pki,ij,qkj->pq',C.conj(),K,C,optimize=True)
def mixed(C,D,K):return np.einsum('pki,ij,qkj->pq',C.conj(),K,D,optimize=True)

def role_consume(role,n,X,fields,allmaps,h,epoch):
    Es=extraction(n);shape=(dim(n-1),X.shape[1]) if role==0 else (X.shape[0],dim(n-1));paths=[]
    for f,W in enumerate(fields):
        for k,E in enumerate(Es):
            path=ROOT/f'harmonic_car_bridge_{epoch}_scratch_{role}_{f}_{k}.npy';Z=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=shape)
            if role==0:Z[:]=E@W
            else:Z[:]=(E@W.T).T
            Z.flush();del Z;paths.append(path)
    K=np.zeros((15,15));views=[np.load(p,mmap_mode='r').reshape(-1) for p in paths]
    for start in range(0,views[0].size,65536):
        a=np.stack([v[start:start+65536] for v in views]);K+=a@a.T
    del views
    for p in paths:p.unlink()
    orientations=[]
    for creation,maps in zip([False,True],allmaps):
        S=np.zeros((4,4,15),complex);S[:,:,:5]=np.asarray(maps)
        G=quadratic(S,K);moments={}
        for u in [-.5,.5]:
            H=np.zeros_like(S);H[:,:,10:]=np.asarray(maps)
            for p in range(4):
                change=sum(((h[q,p] if creation else -h[p,q])*np.asarray(maps[q]) for q in range(4)),np.zeros((4,5),complex))
                delta=(1 if creation else -1)*(.5 if p<2 else -.5)+u
                H[p,:,:5]=change+64*delta*delta*np.asarray(maps[p]);H[p,:,5:10]=128*delta*np.asarray(maps[p])
            moments[str(u)]={'G':G,'T':mixed(S,H,K),'U':quadratic(H,K)}
        orientations.append(moments)
    return K,orientations

def run(args):
    start=time.monotonic();pins={name:sha(ROOT/name) for name in [*PINS,*EXTRA]};assert all(pins[n]==h for n,h in PINS.items())
    control,J,maps=exact_controls();(ROOT/f'harmonic_car_bridge_{args.epoch}_controls.json').write_text(json.dumps(control,indent=2)+'\n')
    profile=json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_bytes())['profiles']
    with np.load(ROOT/'local_so5_resum_actual04_20261011_state.npz',allow_pickle=False) as data:X=np.array(data['state'],copy=True).reshape(14,690880)
    normal=float(np.sum(X*X));assert abs(normal-1)<1e-10
    matrices=[];rhos=[];hs=[]
    for n,p in zip([2,126],profile):
        m,c,d=[float(F(p[k])) for k in ['m_exact','c0_exact','d_exact']];M,r=native_matrix(n,m,c,d);matrices.append(M);rhos.append(r)
        hs.append(np.array([[0,0,0,0]]*4,complex));hs[-1]=m*np.diag([-1,1,-1,1])+c*np.kron([[0,1],[1,0]],np.diag([1,-1]))+d*np.block([[np.zeros((2,2)),1j*np.array([[0,1],[1,0]])],[-1j*np.array([[0,1],[1,0]]),np.zeros((2,2))]])
    rho=rhos[0][:,None]+rhos[1][None,:];HX=matrices[0]@X+(matrices[1]@X.T).T+64*rho*rho*X;energy=float(np.sum(X*HX)/normal);residual=float(np.linalg.norm(HX-energy*X))
    fields=[X,rho*X,HX];allmom=[];role_rows=[]
    for role,n in enumerate([2,126]):
        K,orientations=role_consume(role,n,X,fields,maps,hs[role],args.epoch)
        for creation,mom in zip([False,True],orientations):
            assert abs(np.trace(mom['0.5']['G']).real-2*normal)<2e-10
            role_rows.append({'role':profile[role]['role'],'physical_copy_count':profile[role]['count'],'spin_replicas':n,'orientation':'creation' if creation else 'annihilation','single_individually_labelled_replica':True,'trace_source_norm':float(np.trace(mom['0.5']['G']).real),'endpoint_reduced_matrices':{u:{k:serialize(v) for k,v in z.items()} for u,z in mom.items()},'complete_streamed_field_slot_Gram':serialize(K),'harmonic_changed_dimension':dim(n-1)*dim(126 if role==0 else 2)*4})
        allmom.append(orientations)
    with np.load(ROOT/'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz',allow_pickle=False) as data:phi=data['Phi_binary_section']
    had=np.array([[1,0,1,0],[0,1,0,1],[-1,0,1,0],[0,-1,0,1]],float)/np.sqrt(2);frame=np.einsum('ia,mij,jb->mab',had,phi,had);gamma=frame[1]/np.sqrt(np.trace(frame[1]@frame[1]).real/4);spin=np.column_stack([np.eye(4)[:,0],gamma@np.eye(4)[:,0],np.eye(4)[:,1],gamma@np.eye(4)[:,1]])
    assert np.linalg.norm(spin.conj().T@spin-np.eye(4))<1e-12
    upper=json.loads((ROOT/'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json').read_bytes());E0=F(upper['ground_energy_interval_exact'][1]);old=json.loads((ROOT/EXTRA[0]).read_bytes())['all64_boundary'];lower=json.loads((ROOT/EXTRA[1]).read_bytes());floors={b['branch']:F(b['floor_exact']) for b in lower['branches']};out=[];qguard=F(0);qfloat=0.;qresum=0.
    flow_geometry=json.loads((ROOT/EXTRA[2]).read_bytes());geometry={}
    assert flow_geometry['axis0_line_positive_control'] and flow_geometry['transverse_common_u_adoption_rejected']
    for row in flow_geometry['rows']:
        key=(row['axis'],row['length'],tuple(row['sites']),row['qa'])
        offsets=tuple(F(z) for z in row['affine_u'])
        assert geometry.setdefault(key,offsets)==offsets
    def endpoint(mom,site):
        ix=[2*site,2*site+1];return {k:np.kron(np.eye(2),v[np.ix_(ix,ix)]) for k,v in mom.items()}
    def contract(M,A,B):return np.einsum('ij,kl,ik,jl->',M.conj(),M,A,B).real
    for alloc in old['all64_once_shared_pointwise_norms']:
        role_paths=[];weight=F(alloc['root_allocation_exact']);total=F(0);tf=0.;tr=0.
        for r,p in enumerate(old['roles']):
            path=next(z for z in p['paths'] if z['axis']==alloc['axis'] and z['length']==alloc['length']);physical=np.array([[complex(float(F(z[0])),float(F(z[1]))) for z in row] for row in path['matrix_exact']]);M=spin.conj().T@physical@spin;ext=64*path['external_edges_min'];g=F(ext)+2*floors['active' if r==0 else 'aux']-2*E0;assert g>0
            choices=[]
            for sa,sb in sorted({key[2] for key in geometry if key[:2]==(alloc['axis'],alloc['length'])}):
                vals=[]
                for qa in [-1,1]:
                    ua,ub=geometry[(alloc['axis'],alloc['length'],(sa,sb),qa)]
                    A=endpoint(allmom[r][int(qa==1)][str(float(ua))],sa);B=endpoint(allmom[r][int(qa==-1)][str(float(ub))],sb)
                    Pa=np.diag([1,-1,1,-1])*(-1)**sa;Pb=np.diag([1,-1,1,-1])*(-1)**sb
                    jointM=Pa@M@Pb;N=jointM if qa==1 else jointM.conj();cshift=ext-2*energy
                    w=contract(N,A['G'],B['G']);first=contract(N,A['T'],B['G'])+contract(N,A['G'],B['T'])+cshift*w
                    second=contract(N,A['U'],B['G'])+contract(N,A['G'],B['U'])+2*contract(N,A['T'],B['T'])+2*cshift*(first-cshift*w)+cshift*cshift*w
                    mu=first/w;var=max(0.,second/w-mu*mu);gf=float(g)
                    res=min(w/gf**2,w/mu**2+w*var*(mu+2*gf)/(gf*gf*mu**3)) if mu>gf else w/gf**2
                    # A materially different source-weighted heat/resummed Galerkin observation, not an upper certificate.
                    gal=w/mu**2
                    vals.append({'qA':qa,'affine_u_exact':[str(ua),str(ub)],'source_weight':w,'complete_charged_relative_mean':mu,'complete_charged_relative_variance':var,'moment_heat_inverse_square_upper_float':res,'mean_resummed_inverse_square_lower_float':gal})
                choices.append({'sites':[sa,sb],'orientations':vals,'source_weight':sum(v['source_weight'] for v in vals),'inverse_upper':sum(v['moment_heat_inverse_square_upper_float'] for v in vals),'inverse_resummed_lower':sum(v['mean_resummed_inverse_square_lower_float'] for v in vals)})
            chosen=max(choices,key=lambda z:z['source_weight']);wf=chosen['source_weight'];absum=sum(abs(F(z[0]))+abs(F(z[1])) for row in path['matrix_exact'] for z in row)
            # Two orientations; absolute entry enclosure includes ground norm-distance 1/60000 and streaming roundoff.
            epsilon=F(1,8192);error=2*(2*epsilon+epsilon**2)*absum**2;wu=F(wf)+error
            total+=p['count']*wu/g**2;tf+=p['count']*max(z['inverse_upper'] for z in choices);tr+=p['count']*max(z['inverse_resummed_lower'] for z in choices)
            role_paths.append({'role':p['role'],'count':p['count'],'physical_spin_pair_labels':[[a,b] for a,b in itertools.product(range(2),repeat=2)],'charged_gap_lower_exact':str(g),'source_weight_float_max_all_sites':wf,'source_weight_upper_exact':str(wu),'source_weight_entry_error_exact':str(epsilon),'all_sites_both_charge_choices':choices})
        qguard+=weight*total;qfloat+=float(weight)*tf;qresum+=float(weight)*tr;out.append({'axis':alloc['axis'],'length':alloc['length'],'allocation_exact':str(weight),'roles':role_paths,'weighted_inverse_square_interval_upper_exact':str(weight*total)})
    receipt={'kind':'NEW_EXACT_NATIVE_HARMONIC_INDIVIDUAL_CAR_COMPLETE_SOURCE_CHARGED_MOMENTS_AND_HEAT_CONSUMER','controls':control,'input_sha256':pins,'own_source_sha256':sha(Path(__file__)),'immutable_source_dimension':X.size,'source_norm_float':normal,'source_energy_float':energy,'source_residual_float_observation_not_recertification':residual,'source_roles':role_rows,'paths':out,'sourceweighted_gap_guard_upper_exact':str(qguard),'sourceweighted_gap_guard_upper_float':float(qguard),'sourceweighted_gap_guard_less_than_one':qguard<1,'moment_heat_guard_float_not_interval':qfloat,'mean_resummed_guard_lower_float_not_acceptance':qresum,'roundoff_scope':{'density_entry_enclosure_exact':'1/8192','derivation':'normalized binary source E_i contractions: radical coefficient error <2^-40 per entry, <=2 terms/column, n<=126; gamma_(4D+32)<2^-26 for D=9672320, action norm<=1, conservative density formation/reduction margin2^-18. Ground norm distance1/60000 changes each CAR density by<=2/60000. Physical spin rotation roundoff included in remaining1/8192 margin. Exact rational interval uses source weights only; charged moments and heat factors remain float observations.','standard_binary64_model_assumed':True},'analytic_derivation':{'polynomial':'psi_lpq = sqrt(p!q!(l-p)!(l-q)!/l!) sum_{k10+k11=p,k01+k11=q} x^k/prod(k!); F=sum_r (n-l)! z^(n-l-2r) Q^r psi/[2^r r! (n-l-2r)! (l+2)_r], Q=x00*x11-x01*x10. Norm by Fock factorial sum; exact derivative maps recorded in extraction().','all_n':'Fischer harmonic polynomial construction solves T=dz^2-2(d00*d11-d01*d10). Differentiation preserves T kernel. Coefficient matching against the normalized SO4 polynomial gives extraction() for every n; sum b_i^dag b_i=n yields isometry. The n1 native joint embedding fixes ALL A/B/C/R phases; real gauge i^(p+q-l) gives slot phases (1,-i,1,1,i).','individual_copy':'E=b/sqrt(n) is selected-slot extraction, not physical collective CAR. Native signed W5->wedge1/wedge3 is applied to selected slot; arbitrary endpoint mixtures by linearity. All128 labels have N_j=2, so differing changed labels and opposite charges are orthogonal; within-copy spin modes remain coherent. All1+63 physical copies, each2 spin replicas, are summed exactly once.','complete_H':'H_ch(u) A_p psi = A_p H_neutral psi + [dGamma(h),A_p]psi +64 A_p[2delta_p rho+delta_p^2]psi; delta_p=u +/- r_p. Exact full16 control proves this identity. Full original H action and rho-weighted source are extracted, no neutral gap substitution and no PH crop. Complete charged first two moments are reduced without target tensor allocation.','heat_resummed':'For x>=g and mu>g, x^-2 <= mu^-2-2(x-mu)/mu^3+[(mu+2g)/(g^2 mu^3)](x-mu)^2. This is integration of the quadratic heat-kernel remainder; source mean/variance give a second representation. Jensen gives w/mu^2 only a LOWER inverse-square bound, explicitly not acceptance.'},'new_material_next_criterion':'If the gap guard fails, enclose the observed complete charged moment-heat guard or compute source-Lanczos/Stieltjes spectral measures on these individual charged harmonic/spinor images with certified tail and affine fullE64. Require weighted upper <1 before addressing overlapping normalized allorder. The tested mean resummation and moment heat representation are observations, not an unexecuted fork declared blocked.','debts':['FullPhi/intercell stationary ground and physicalPhi lift','normalized allorder N-uniform overlapping shared-flow linked control','sourceweighted heat/moment interval certification','bulk phase/stiffness/IRatom/pole/commoncone/originalSOURCE'], 'fullPhi_ground':False,'phase':False,'campaign_complete':False,'source_and_inputs_unchanged':all(sha(ROOT/n)==v for n,v in pins.items()),'elapsed_seconds':time.monotonic()-start}
    receipt['literal_native_path_affine_geometry']=EXTRA[2]
    receipt['axis0_common_offsets_valid_transverse_common_offsets_withdrawn']=True
    receipt['site_pairs_are_actual_primitive_paths_not_arbitrary_directsum']=True
    assert receipt['source_and_inputs_unchanged'];(ROOT/f'harmonic_car_bridge_{args.epoch}_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ['paths','source_roles','controls','analytic_derivation']},indent=2),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',required=True);run(parser.parse_args())
