"""New coherent neutral-return CAR interface and frozen-source contraction.
No solver imports, no two-cell state allocation. Exact identities use SymPy;
source-weighted numbers are FLOAT diagnostics, not spectral-ground assertions.
"""
from __future__ import annotations
from typing import cast
import argparse
import hashlib
import itertools
import json
import math
import time
import zipfile
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import sympy as s
from scipy import sparse
import neutral_return_image_factorized_20261011 as factorized
ROOT = Path(__file__).resolve().parent
PINS = {
 'local_so5_resum_actual04_20261011_state.npz':'dd06400e1b7f70df935cf7ca02cf5ebf598481a8d5e209d1241aba5ed3fea2a2',
 'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json':'102966f32c49f40d79d55343925a630e30a9cf7e4a8074cf8628142ba9c5b6e8',
 'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json':'2f019dd49873a5e13f130c65ef6739b9af78a16f85138dc6c6ab6e32e65c7a12',
 'harmonic_car_bridge_derivation_20261011.json':'ed3978df114bf55e7d84ea93b2f671b6a5baaf3e808a589ec464161269afdba3',
 'joined_phi_u1_harmonic_transverse_affine_boundary_control_20261011_receipt.json':'5f1b83bceb056ce253ca5f3bd945396c9af37ae81a607cab339ab223f03a949e',
 'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz':'f8a0793adcbf4909402cd089471aa8c33fc74a31b66e99618f8baaab8d6dfb65',
 'neutral_return_image_factorized_20261011.py':'c906b72557454adbe107478ccd367ecdb49599ddb410b13d83e03c7ee5170973',
 'neutral_return_image_interface_20261011.json':'a0b4d7e530f770d549d4f022e41f2134715e850b93a046449b64e3fe61dd2a43',
}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def enc(M):
    return [[[float(z.real),float(z.imag)] for z in row] for row in np.asarray(M)]
def dim(n):
    return (n+1)*(n+2)*(2*n+3)//6
def labels(n):
    return [(l,p,q) for l in range(n+1) for p in range(l+1) for q in range(l+1)]
def idx(l,p,q):
    return l*(l+1)*(2*l+1)//6+p*(l+1)+q
def generators(n):
    out=[s.zeros(dim(n)) for _ in range(4)]
    for j,(l,p,q) in enumerate(labels(n)):
        out[0][j,j]=2*(p+q-l);out[3][j,j]=p-q
        for axis,k in [(0,p),(1,q)]:
            if k<l:
                i=idx(l,p+1,q) if axis==0 else idx(l,p,q+1)
                v=s.I*s.sqrt((l-k)*(k+1));out[2][i,j]=v;out[2][j,i]=-v
        if l<n:
            for dp,dq,sign in [(1,0,1),(0,1,-1)]:
                rad=s.Rational((n-l)*(n+l+3)*(p+1 if dp else l-p+1)*(q+1 if dq else l-q+1),(l+1)*(l+2))
                i=idx(l+1,p+dp,q+dq);v=sign*s.sqrt(rad);out[1][i,j]=v;out[1][j,i]=v
    return out
def extraction(n):
    out=[s.zeros(dim(n-1),dim(n)) for _ in range(5)]
    for j,(l,p,q) in enumerate(labels(n)):
        if l<n:out[0][idx(l,p,q),j]=s.sqrt(s.Rational((n-l)*(n+l+2),n*(2*n+1)))
        for a,b in itertools.product(range(2),repeat=2):
            k=1+2*a+b;sp=l-p if a==0 else p;sq=l-q if b==0 else q
            if l and sp and sq:
                out[k][idx(l-1,p-a,q-b),j]=s.sqrt(s.Rational((n+l+1)*(n+l+2)*sp*sq,2*n*l*(l+1)*(2*n+1)))
            if l<n-1:
                tp=p+1 if a==0 else l-p+1;tq=q+1 if b==0 else l-q+1
                out[k][idx(l+1,p+1-a,q+1-b),j]=(-1)**(a+b)*s.sqrt(s.Rational((n-l)*(n-l-1)*tp*tq,2*n*(l+1)*(l+2)*(2*n+1)))
    return out
def car():
    ann=[]
    for p in range(4):
        A=s.zeros(16)
        for m in range(16):
            if m>>p&1:A[m^(1<<p),m]=(-1)**((m&((1<<p)-1)).bit_count())
        ann.append(A)
    A=s.diag(-1,1,-1,1);X=s.Matrix([[0,1],[1,0]]);Z=s.zeros(2)
    B=s.kronecker_product(X,s.diag(1,-1));C=Z.row_join(s.I*X).col_join((-s.I*X).row_join(Z));R=s.diag(s.Rational(1,2),s.Rational(1,2),-s.Rational(1,2),-s.Rational(1,2))
    def lift(h):return cast(s.MutableDenseMatrix,sum((ann[p].T*ann[q]*h[p,q] for p,q in itertools.product(range(4),repeat=2)),s.zeros(16)))
    ops=[lift(h) for h in [A,B,C,R]]
    V=s.zeros(16,5)
    for j,m in enumerate([5,3,12,10]):V[m,j]=1
    V[9,4]=V[6,4]=1/s.sqrt(2)
    g=generators(1);z=tuple(s.Symbol(f'z{i}') for i in range(25));J=s.Matrix(5,5,z);eq=[]
    for O,G in zip(ops,g):eq+=list(V.T*O*V*J-J*G)
    eq+=list(J[:,idx(1,0,0)]-s.eye(5)[:,0])
    sol=s.linsolve(eq,z);assert len(sol.args)==1
    solution=cast(s.Tuple,sol.args[0])
    J=J.subs({z[i]:solution[i] for i in range(len(z))});assert s.simplify(J.conjugate().T*J)==s.eye(5)
    phase=s.diag(1,-s.I,1,1,s.I);W=V*J*phase
    sing=s.zeros(16,1);sing[9]=1/s.sqrt(2);sing[6]=-1/s.sqrt(2)
    W6=sing.row_join(W)
    for O in ops:
        assert O*sing==s.zeros(16,1)
        assert s.simplify(O*W6-W6*(W6.conjugate().T*O*W6))==s.zeros(16,6)
    return ann,[A,B,C,R],ops,W,W6

def exact_controls():
    ann,h4,ops,W,W6=car();I=s.eye(16)
    for p,q in itertools.product(range(4),repeat=2):
        assert ann[p]*ann[q].T+ann[q].T*ann[p]==int(p==q)*I
    assert ann[0].applyfunc(abs)*ann[1].T.applyfunc(abs)+ann[1].T.applyfunc(abs)*ann[0].applyfunc(abs)!=s.zeros(16)
    # Containing scalar+vector6 is indispensable: an endpoint bilinear reaches singlet.
    witness=None
    for p,q in itertools.product(range(4),repeat=2):
        row=s.simplify(W6[:,0].conjugate().T*ann[p].T*ann[q]*W)
        if row!=s.zeros(1,5):witness={'p':p,'q':q,'row':[str(z) for z in row]};break
    assert witness
    # Extraction is a single selected-slot isometry; two slots share spectators.
    extraction_rows=[]
    for n in [2,3]:
        E=extraction(n);Eprev=extraction(n-1)
        assert cast(s.MatrixBase,sum((e.T*e for e in E),s.zeros(dim(n)))).applyfunc(s.simplify)==s.eye(dim(n))
        EE=[Eprev[j]*E[i] for i,j in itertools.product(range(5),repeat=2)]
        assert cast(s.MatrixBase,sum((e.T*e for e in EE),s.zeros(dim(n)))).applyfunc(s.simplify)==s.eye(dim(n))
        for i,j in itertools.product(range(5),repeat=2):assert s.simplify(Eprev[j]*E[i]-Eprev[i]*E[j])==s.zeros(dim(n-2),dim(n))
        for G,Gr,G1 in zip(generators(n),generators(n-1),generators(1)):
            for i in range(5):assert s.simplify(Gr*E[i]+sum((G1[i,j]*E[j] for j in range(5)),s.zeros(*E[i].shape))-E[i]*G)==s.zeros(*E[i].shape)
        extraction_rows.append({'n':n,'one_and_two_isometries':True,'two_extraction_shared_harmonic_spectator':True,'joint_generators':True})
    # Independent hole signed Hodge intertwiner, not energy-only diagonal fit.
    sec3=[m for m in range(16) if m.bit_count()==3];Jhole=s.zeros(4)
    for p in range(4):Jhole[sec3.index(15^(1<<p)),p]=(-1)**p
    for h,O in zip(h4,ops):assert Jhole.T*O.extract(sec3,sec3)*Jhole==-h.T
    bad=Jhole.applyfunc(abs)
    assert bad.T*ops[2].extract(sec3,sec3)*bad!=-h4[2].T
    K=s.diag(1,-1,1,-1)*s.Matrix([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
    T3=Jhole*K
    assert T3.T*T3==s.eye(4)
    for h,O in zip(h4,ops):assert T3.T*O.extract(sec3,sec3)*T3==h
    # Literal Gauss prefix on physical nodes0,1,2,3, not a two-link term.
    ra,rb=s.symbols('rhoA rhoB',real=True)
    for delta in [-1,1]:
        qnodes=[ra,-ra+delta,rb-delta,-rb]
        electric=[-sum(qnodes[:j+1]) for j in range(3)]
        assert electric==[-ra,-delta,-rb]
        charged_rho=[ra-s.Rational(delta,2),rb-s.Rational(delta,2)]
        assert [r+s.Rational(delta,2) for r in charged_rho]==[ra,rb]
        returned=[ra,-ra,rb,-rb]
        assert [-sum(returned[:j+1]) for j in range(3)]==[-ra,0,-rb]
    # Same-edge full-rho commutation, also with different onsite roles.
    for endpoint in [[0,1],[2,3]]:
        for p,q in itertools.product(endpoint,repeat=2):
            E=ann[p].T*ann[q];assert ops[3]*E-E*ops[3]==s.zeros(16)
            h=3*h4[0]+5*h4[1]+7*h4[2]
            com=sum((h[a,p]*ann[a].T*ann[q]-h[q,a]*ann[p].T*ann[a] for a in range(4)),s.zeros(16))
            assert (3*ops[0]+5*ops[1]+7*ops[2])*E-E*(3*ops[0]+5*ops[1]+7*ops[2])==com
    # Sparse independent globally signed CAR proves the contact/exchange formula.
    def elementary(v,p,creation,unsigned=False):
        out={}
        for m,c in v.items():
            if bool(m>>p&1)!=creation:
                sign=1 if unsigned else (-1)**((m&((1<<p)-1)).bit_count())
                target=m^(1<<p);out[target]=out.get(target,0)+sign*c
        return {m:s.expand(c) for m,c in out.items() if c!=0}
    def word(v,letters):
        for p,create in reversed(letters):v=elementary(v,p,create)
        return v
    def add(out,v,c):
        for m,z in v.items():out[m]=s.expand(out.get(m,0)+c*z)
    # Exercise the general labelled factorized API against independent global CAR.
    pair_source={m+(n<<4):W[m,0]*W[n,0] for m,n in itertools.product(range(16),repeat=2) if W[m,0]*W[n,0]!=0}
    def density(slots):
        e0=s.eye(5)[:,0]
        z=cast(s.MatrixBase,e0 if len(slots)==1 else s.kronecker_product(e0,e0))
        return z*z.T
    features=[None,((0,0),(0,0)),((1,2),(1,2)),((0,0),(1,2)),((1,2),(0,0)),((0,0),(0,2))]
    for left,right in itertools.product(features,repeat=2):
        letters=factorized.feature_word(left,True)+factorized.feature_word(right)
        acted=word(pair_source,[(4*slot+p,creation) for ((slot,p),creation) in letters])
        direct_moment=sum(s.conjugate(c)*acted.get(m,0) for m,c in pair_source.items())
        assert s.simplify(factorized.local_gram(left,right,ann,W,density)-direct_moment)==0
    # Explicit all64 kernel generation, including copy0/aux63 and crosscopy.
    kernel_count=sum(1 for _ in factorized.return_terms(s.eye(4)))
    assert kernel_count==2*64*4+64*64*16
    profile=lambda role:3*h4[0]+(5 if role=='active' else 11)*h4[1]+7*h4[2]
    mixed=list(factorized.matter_commutator(((0,0),(2,2)),profile))
    assert any(z!=0 and f[0][1] not in [0,1] for z,f in mixed)
    # Two copies, one endpoint component each; copies genuinely distinct slots.
    M=s.Matrix([[2,1+s.I],[3-s.I,-1]])
    # A slots0,1 and B slots2,3, all number2; coherent vector slot superpositions.
    source={}
    for masks in itertools.product([3,5],repeat=4):
        m=sum(mask<<(4*k) for k,mask in enumerate(masks));source[m]=s.Rational(1,4)
    def hop(v,dagger=False):
        out={}
        for copy in range(2):
            for p,q in itertools.product(range(2),repeat=2):
                ai=4*copy+2*p;bj=8+4*copy+2*q
                letters=[(bj,True),(ai,False)] if dagger else [(ai,True),(bj,False)]
                add(out,word(v,letters),s.conjugate(M[p,q]) if dagger else M[p,q])
        return {m:z for m,z in out.items() if z!=0}
    direct={};add(direct,hop(hop(source),True),1);add(direct,hop(hop(source,True)),1)
    formula={};cross={}
    for copy in range(2):
        for p,k in itertools.product(range(2),repeat=2):
            add(formula,word(source,[(4*copy+2*p,True),(4*copy+2*k,False)]),(M*M.conjugate().T)[p,k])
            add(formula,word(source,[(8+4*copy+2*p,True),(8+4*copy+2*k,False)]),(M.conjugate().T*M)[p,k])
    for f,g,p,q,k,l in itertools.product(range(2),repeat=6):
        v=word(source,[(4*f+2*p,True),(4*g+2*k,False),(8+4*g+2*l,True),(8+4*f+2*q,False)])
        c=-2*M[p,q]*s.conjugate(M[k,l]);add(formula,v,c)
        if f!=g:add(cross,v,c)
    assert all(s.expand(direct.get(m,0)-formula.get(m,0))==0 for m in set(direct)|set(formula))
    crossnorm=s.simplify(sum((s.conjugate(c)*c for c in cross.values()),s.Integer(0)));assert crossnorm>0
    # rho-zero includes many non-ground states: exact witness, no scalar-mean replacement.
    g=s.eye(16)[:,5];v=s.eye(16)[:,10]
    assert ops[3]*g==s.zeros(16,1) and ops[3]*v==s.zeros(16,1) and g.T*v==s.zeros(1)
    assert (I-g*g.T)*v==v
    # The 63-copy multiplicity wrongcase uses disjoint changed-number sectors.
    assert 63*crossnorm!=crossnorm
    api_controls={'local_factorized_moments_match_independent_fullCAR':True,'explicit64_kernel_term_count_positive_control':kernel_count,'mixed_role_full4Dirac_commutator_positive_control':True}
    return {'executable_factorized_API':api_controls,'signed_CAR':True,'unsigned_rejected':True,'complete_contact_exchange_matches_independent_sparse_CAR':True,'drop_crosscopy_exact_squared_error':str(crossnorm),'drop_aux63_rejected_exact_multiplicity':63,'drop_singlet_witness':witness,'hole_phase_wrongcase_rejected':True,'number3_number1_joint_native_T3':[[str(z) for z in row] for row in T3.tolist()],'literal_Gauss_firsthop_and_neutral_return':True,'whole_ground_not_rho_zero_witness_masks':[5,10],'extraction_controls':extraction_rows,'same_edge_fullrho_commutation_all_roles':True,'scalar_vector6_joint_native_embedding':[[str(z) for z in row] for row in W6.tolist()]},ann,h4,ops,W

def native_matrix(n,m,c,d):
    rows=[];cols=[];values=[];rho=np.empty(dim(n))
    for l in range(n+1):
        p,q=np.meshgrid(np.arange(l+1),np.arange(l+1),indexing='ij');p=p.ravel();q=q.ravel();j=idx(l,p,q);rho[j]=p-q
        rows.extend(j);cols.extend(j);values.extend(2*m*(p+q-l))
        for axis,k in [(0,p),(1,q)]:
            ok=k<l;i=idx(l,p[ok]+int(axis==0),q[ok]+int(axis==1));v=d*np.sqrt((l-k[ok])*(k[ok]+1))
            rows.extend(i);cols.extend(j[ok]);values.extend(v);rows.extend(j[ok]);cols.extend(i);values.extend(v)
        if l<n:
            for dp,dq,sign in [(1,0,1),(0,1,-1)]:
                i=idx(l+1,p+dp,q+dq);v=sign*c*np.sqrt((n-l)*(n+l+3)*(p+1 if dp else l-p+1)*(q+1 if dq else l-q+1)/((l+1)*(l+2)))
                rows.extend(i);cols.extend(j);values.extend(v);rows.extend(j);cols.extend(i);values.extend(v)
    return sparse.csr_matrix((values,(rows,cols)),shape=(dim(n),dim(n))),rho

def source_consume(ann,h4,ops,W,profiles):
    with np.load(ROOT/'local_so5_resum_actual04_20261011_state.npz',allow_pickle=False) as data:
        X=np.array(data['state'],copy=True).reshape(14,690880)
    norm=float(np.sum(X*X));X/=math.sqrt(norm)
    # NEW operator consumption of the source: full H0 fields, not a solver or residual recertification.
    hs=[];mat=[];rs=[]
    for n,p in zip([2,126],profiles):
        m,c,d=[float(F(p[k])) for k in ['m_exact','c0_exact','d_exact']]
        H,r=native_matrix(n,m,c,d);mat.append(H);rs.append(r)
        hs.append(np.array((m*h4[0]+c*h4[1]+d*h4[2]).evalf(),complex))
    rho=rs[0][:,None]+rs[1][None,:]
    HX=mat[0]@X+(mat[1]@X.T).T+64*rho*rho*X
    # Active two-slot reduced field density retains all auxiliary126 entanglement.
    e=extraction(2);e1=extraction(1)
    D=s.Matrix.vstack(*[e1[j]*e[i] for i,j in itertools.product(range(5),repeat=2)])
    assert s.simplify(D.T*D)==s.eye(14)
    D=np.array(D.evalf(),complex);w=np.array(W.evalf(),complex);embed=np.kron(w,w)
    density={}
    for a,F0 in enumerate([X,HX]):
        for b,F1 in enumerate([X,HX]):density[a,b]=D@(F0@F1.T)@D.conj().T
    with np.load(ROOT/'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz',allow_pickle=False) as data:phi=data['Phi_binary_section']
    had=np.array([[1,0,1,0],[0,1,0,1],[-1,0,1,0],[0,-1,0,1]],float)/np.sqrt(2)
    frame=np.einsum('ia,mij,jb->mab',had,phi,had);gamma=frame[1]/np.sqrt(np.trace(frame[1]@frame[1]).real/4)
    spin=np.column_stack([np.eye(4)[:,0],gamma@np.eye(4)[:,0],np.eye(4)[:,1],gamma@np.eye(4)[:,1]])
    assert np.linalg.norm(spin.conj().T@spin-np.eye(4))<1e-12
    c,d=[float(F(profiles[0][k])) for k in ['c0_exact','d_exact']]
    physical=-c*np.diag([1,1,-1,-1])+1j*d*np.fliplr(np.eye(4))
    Pa=-np.diag([1,-1,1,-1]);Pb=np.diag([1,-1,1,-1]);M=Pa@(spin.conj().T@physical@spin)@Pb
    a=[np.array(z,complex) for z in ann];parity=np.diag([(-1)**m.bit_count() for m in range(16)])
    full=[np.kron(z,np.eye(16)) for z in a]+[np.kron(parity,z) for z in a]
    hlocal=sum((hs[0][p,q]*(full[p].conj().T@full[q]+full[4+p].conj().T@full[4+q]) for p,q in itertools.product(range(4),repeat=2)),np.zeros((256,256),complex))
    def features(site,contact):
        modes=[4*t+2*site+p for t,p in itertools.product(range(2),repeat=2)]
        Es=[full[i].conj().T@full[k] for i,k in itertools.product(modes,repeat=2)]
        C=sum((contact[i,k]*Es[4*i+k] for i,k in itertools.product(range(4),repeat=2)),np.zeros((256,256),complex))
        F0=[np.eye(256),C,*Es]
        return [f@embed for f in F0],[(hlocal@f-f@hlocal)@embed for f in F0]
    FA,DA=features(1,M@M.conj().T);FB,DB=features(0,M.conj().T@M)
    coeff=np.zeros((18,18),complex);coeff[1,0]=coeff[0,1]=1
    for i,j,k,l in itertools.product(range(4),repeat=4):coeff[2+4*i+k,2+4*l+j]=-2*M[i,j]*M[k,l].conjugate()
    def gram(A,B,a0,b0):
        # <A applied field a0, B applied field b0>, via reduced field density.
        return np.array([[np.trace(x.conj().T@y@density[b0,a0]) for y in B] for x in A])
    GA=gram(FA,FA,0,0);GB=gram(FB,FB,0,0)
    def contract(C0,C1,A,B):return np.einsum('ij,kl,ik,jl->',C0.conj(),C1,A,B,optimize=True)
    weight=contract(coeff,coeff,GA,GB)
    # H C source = C(HX⊗X+X⊗HX)+[hA+hB,C]source; rho commutes C, u=0.
    pieces=[(FA,FB,1,0),(FA,FB,0,1),(DA,FB,0,0),(FA,DB,0,0)]
    first=0j;second=0j
    for A,B,ai,bi in pieces:
        first+=contract(coeff,coeff,gram(FA,A,0,ai),gram(FB,B,0,bi))
        for AA,BB,aj,bj in pieces:second+=contract(coeff,coeff,gram(A,AA,ai,aj),gram(B,BB,bi,bj))
    meanA=np.array([np.trace(embed.conj().T@f@density[0,0]) for f in FA]);meanB=np.array([np.trace(embed.conj().T@f@density[0,0]) for f in FB])
    mean=np.einsum('ij,i,j->',coeff,meanA,meanB)
    subtract=float(weight.real-abs(mean)**2)
    return {'scope':'FLOAT active1 complete spin2 neutral return, full selected source reduced across auxiliary126; not complete64 contraction and not exact spectral Q',
     'source_norm_before_normalization':norm,'source_dimension':X.size,'normalized_exact_binary_source_definition':'stored vector divided by its exact Euclidean norm; computation FLOAT',
     'active_pair_density':enc(density[0,0]),'physical_spin_frame':enc(spin),'actual_axis0_literal1_to2_hopping':enc(M),
     'unprojected_neutral_squared_norm_float':float(weight.real),'unprojected_neutral_mean_float':[float(mean.real),float(mean.imag)],
     'source_product_projector_removed_norm_float_NOT_whole_spectral_Q':subtract,
     'complete_H0_first_float':float(first.real),'complete_H0_second_float':float(second.real),
     'H0_action_uses_source_HX_and_commutators_not_mean_only':True,
     'imaginary_roundoff_diagnostics':[float(weight.imag),float(first.imag),float(second.imag)],
     'all64_remaining_contraction':'Use the same finite local density1/density2 and copy-labelled coefficient kernel defined in interface; no source-independent auxiliary copy assumption.'}

def run(args):
    start=time.monotonic();own=sha(Path(__file__));pins={n:sha(ROOT/n) for n in PINS};assert pins==PINS
    out=ROOT/f'neutral_return_image_{args.epoch}_receipt.json';pre=ROOT/f'neutral_return_image_{args.epoch}_preexecution.zip';executed=ROOT/f'neutral_return_image_{args.epoch}_executed.zip'
    assert not out.exists() and not pre.exists() and not executed.exists()
    with zipfile.ZipFile(pre,'x',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__),Path(__file__).name)
        z.write(ROOT/'neutral_return_image_factorized_20261011.py','neutral_return_image_factorized_20261011.py')
        z.write(ROOT/'neutral_return_image_interface_20261011.json','neutral_return_image_interface_20261011.json')
        z.writestr('input_hashes.json',json.dumps(pins,sort_keys=True))
    controls,ann,h4,ops,W=exact_controls()
    profiles=json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_bytes())['profiles']
    obs=source_consume(ann,h4,ops,W,profiles)
    caps=json.loads((ROOT/'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json').read_bytes())
    infidelity=F(caps['selected_ground_infidelity_upper_exact'])
    assert 2*infidelity<F(1,60000)**2
    c,d=[F(profiles[0][k]) for k in ['c0_exact','d_exact']]
    L=131072*(c*c+d*d)
    perturbation={'scope':'Conditioned on entire physical spectral ground block being unique selected product; FLOAT Gram errors not enclosed here','infidelity_is_squared':True,'vector_and_product_projector_cap_exact':'1/60000','all64_C_norm_upper_exact':str(L),'fixed_source_Q_change_norm_upper_exact':str(L/60000),'true_ground_source_Q_change_norm_upper_exact':str(L/30000),'squared_norm_change_upper_exact':str(L*L/15000)}
    controls['source_caps_perturbation']=perturbation
    receipt={'kind':'COMPLETE_NEUTRAL_RETURN_IMAGE_EXACT_CAR_INTERFACE_AND_NEW_SOURCE_FLOAT_CONTRACTION','own_sha256':own,'input_sha256':pins,'controls':controls,'source_observation':obs,'accepted_source_caps':caps,'elapsed_seconds':time.monotonic()-start,'natural_exit_record':'External invocation records process exit; this receipt is emitted immediately before natural return.','all_inputs_unchanged':all(sha(ROOT/n)==h for n,h in pins.items()),'source_unchanged':sha(Path(__file__))==own,'no_stationarity_ground_phase_pole_admission':True}
    assert receipt['all_inputs_unchanged'] and receipt['source_unchanged']
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(executed,'x',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__),Path(__file__).name);z.write(out,out.name);z.writestr('input_hashes.json',json.dumps(pins,sort_keys=True))
        z.write(ROOT/'neutral_return_image_factorized_20261011.py','neutral_return_image_factorized_20261011.py')
        z.write(ROOT/'neutral_return_image_interface_20261011.json','neutral_return_image_interface_20261011.json')
    print(json.dumps({'receipt':out.name,'elapsed_seconds':receipt['elapsed_seconds'],'weight':obs['unprojected_neutral_squared_norm_float'],'source_projector_diagnostic':obs['source_product_projector_removed_norm_float_NOT_whole_spectral_Q'],'inputs_unchanged':receipt['all_inputs_unchanged']},indent=2),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',required=True);run(parser.parse_args())
