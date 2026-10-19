"""Exact full-Fock low-order contractions, not a single-particle closure."""
from itertools import combinations
import numpy as np
from scipy.linalg import expm


def moments(left,right,rho,amplitude=1.):
    """Augmented [IDENTITY,E(A)] products; rho_ij=<c_j^dag c_i>."""
    lr=left@rho;rr=right@rho;la=np.einsum('aij,ji->a',left,rho);rb=np.einsum('aij,ji->a',right,rho)
    result=np.empty((len(left)+1,len(right)+1),complex);result[0,0]=1.;result[1:,0]=la;result[0,1:]=rb
    result[1:,1:]=np.outer(la,rb)+np.einsum('aij,bji->ab',left,rr)-np.einsum('aij,bji->ab',lr,rr)
    return amplitude*result


def transition(orbitals,T):
    overlap=orbitals.conj().T@T@orbitals;amplitude=np.linalg.det(overlap)
    rho=T@orbitals@np.linalg.solve(overlap,orbitals.conj().T)
    return rho,amplitude,float(np.linalg.svd(overlap,compute_uv=False).min())


def prepare(A,orbitals,T):
    rho,amplitude,minimum=transition(orbitals,T);AT=T@A@T.conj().T
    return dict(aa=moments(A,A,rho,amplitude),at=moments(A,AT,rho,amplitude),tt=moments(AT,AT,rho,amplitude),amplitude=amplitude,minimum=minimum)


def prepared_expansion(data,B,field,U):
    """c0+t*c1+t²*c2; cache only identical OWN Slater moments, not responses."""
    aa,at,tt=data['aa'],data['at'],data['tt'];amplitude=data['amplitude'];minimum=data['minimum']
    bf=B@field;bd=B.conj().transpose(0,2,1)@field
    uf=U@field;ud=U.conj().T@field
    hho=bd.conj()@(B@uf).T;hoh=bd.conj()@(U@bf.T);ohh=(B.conj().transpose(0,2,1)@ud).conj()@bf.T
    first=1j*(np.dot(aa[:,0],bd.conj()@uf)-np.dot(tt[:,0],np.conj(ud)@bf.T))
    second=-.5*np.sum(aa*hho-2*at*hoh+tt*ohh)
    return np.array([amplitude*np.vdot(field,uf),first,second]),minimum


def expansion(A,B,orbitals,field,T,U):
    return prepared_expansion(prepare(A,orbitals,T),B,field,U)


def current_response(A,B,orbitals,field,D,M=None):
    M=moments(A,A,orbitals@orbitals.conj().T) if M is None else M
    bf=B@field;bd=B.conj().transpose(0,2,1)@field;df=D@field;dd=D.conj().transpose(0,2,1)@field
    mean=np.dot(M[1:,0],np.conj(field)@df.T)
    derivative=1j*(np.sum(M[:,1:]*(bd.conj()@df.T))-np.sum(M[1:,:]*(dd.conj()@bf.T)))
    return np.array([mean,derivative])


def gram(A,orbitals,field,D,M=None):
    M=moments(A,A,orbitals@orbitals.conj().T) if M is None else M
    M=M[1:,1:];vectors=D@field
    return np.einsum('cd,acu,bdu->ab',M,vectors.conj(),vectors,optimize=True)


def inserted_character(A,orbitals,field,D,T,U,prepared=None):
    data=prepare(A,orbitals,T) if prepared is None else prepared
    M=data['at'][1:,1:];minimum=data['minimum'];vectors=D@field
    value=np.einsum('cd,acu,uv,bdv->ab',M,vectors.conj(),U,vectors,optimize=True)
    return value,minimum


def covariance_residual(A,B,T,U,character=1.,bare=None):
    """Positive full-Fock HS norm, chunked by actual12-orbital kxy blocks."""
    transformed=U@B@U.conj().T;trace=np.zeros(B.shape[1:],complex);squared=0.
    for start in range(0,A.shape[1],12):
        local=A[:,start:start+12,start:start+12];t=T[start:start+12,start:start+12]
        value=np.einsum('cij,cuv->ijuv',t@local@t.conj().T,transformed,optimize=True)-character*np.einsum('cij,cuv->ijuv',local,B,optimize=True)
        squared+=.25*float(np.vdot(value,value).real);trace+=np.einsum('iiuv->uv',value)
    constant=np.zeros_like(trace) if bare is None else U@bare@U.conj().T-character*bare
    squared+=float(np.linalg.norm(constant+.5*trace)**2)
    return float(np.sqrt(max(0.,squared)))


def operator_norm(A,B,bare=None):
    trace=np.zeros(B.shape[1:],complex);squared=0.
    for start in range(0,A.shape[1],12):
        value=np.einsum('cij,cuv->ijuv',A[:,start:start+12,start:start+12],B,optimize=True)
        squared+=.25*float(np.vdot(value,value).real);trace+=np.einsum('iiuv->uv',value)
    constant=np.zeros_like(trace) if bare is None else bare
    return float(np.sqrt(squared+np.linalg.norm(constant+.5*trace)**2))


def exact_CAR_formula_fixture():
    """SMALL algebra fixture ONLY; primary physical consumer retains108CAR/75."""
    rng=np.random.default_rng(20261017049);n=5;occupied=2;labels=list(combinations(range(n),occupied));masks=[sum(1<<i for i in row) for row in labels];lookup={m:i for i,m in enumerate(masks)}
    def hermitian(size):
        value=rng.normal(size=(size,size))+1j*rng.normal(size=(size,size));return (value+value.conj().T)/2
    V=np.linalg.qr(rng.normal(size=(n,occupied))+1j*rng.normal(size=(n,occupied)))[0];A=np.array([hermitian(n),hermitian(n)]);B=np.array([hermitian(3) for _ in range(3)]);T=expm(.17j*hermitian(n));U=hermitian(3);f=rng.normal(size=3)+1j*rng.normal(size=3);f/=np.linalg.norm(f)
    def body(matrix):
        out=np.zeros((len(masks),len(masks)),complex)
        for col,mask in enumerate(masks):
            for j in range(n):
                if not(mask>>j&1):continue
                first=mask^(1<<j);s1=(-1)**((mask&((1<<j)-1)).bit_count())
                for i in range(n):
                    if first>>i&1:continue
                    result=first^(1<<i);s2=(-1)**((first&((1<<i)-1)).bit_count());out[lookup[result],col]+=matrix[i,j]*s1*s2
        return out
    lifted=np.array([[np.linalg.det(T[np.ix_(row,col)]) for col in labels] for row in labels]);state=np.kron(np.array([np.linalg.det(V[list(row)]) for row in labels]),f)
    H=np.kron(np.eye(len(masks)),B[0])+sum(np.kron(body(a),b) for a,b in zip(A,B[1:]));O=np.kron(lifted,U);comm=H@O-O@H
    direct=np.array([np.vdot(state,O@state),1j*np.vdot(state,comm@state),-.5*np.vdot(state,(H@comm-comm@H)@state)])
    calculated,_=expansion(A,B,V,f,T,U);error=float(np.max(abs(direct-calculated)))
    if error>2e-11:raise ValueError(('exact-CAR fixture rejects twisted Wick/response engine',error))
    return dict(max_error=error,exact_CAR_modes=n,full_fixed_N_dimension=len(masks),meaning='Only an independent sign/order/formula fixture. NOT a smaller substitute for the108-mode physical law or its full-Fock virtual moments.')
