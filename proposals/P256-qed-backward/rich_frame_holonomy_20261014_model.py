"""Scalar colour charge, exact right Gauss, physical LEFT-frame path sources.
Pure helper angular/cg reuse; no old producer execution or inserted spin matrices.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import block_diag, eigh, expm
from rich_frame_gauss_20261014_model import angular, cg, vector_field


def raw_frame():
    js = (0., .5, 1.)
    basis = [(j,m,n) for j in js for m in np.arange(-j,j+1) for n in np.arange(-j,j+1)]
    left = [block_diag(*[np.kron(-angular(j)[a].T, np.eye(int(2*j+1))) for j in js]) for a in range(3)]
    right = [block_diag(*[np.kron(np.eye(int(2*j+1)), angular(j)[a]) for j in js]) for a in range(3)]
    create = []
    for mode in range(4):
        c = np.zeros((16,16),complex)
        for state in range(16):
            if not state & (1<<mode):
                c[state|(1<<mode),state] = (-1)**((state&((1<<mode)-1)).bit_count())
        create.append(c)
    number = sum(c@c.conj().T for c in create)
    colour = [sum(angular(.5)[a][x,y]*create[2*t+x]@create[2*t+y].conj().T for t in range(2) for x in range(2) for y in range(2)) for a in range(3)]
    gauss = [np.kron(r,np.eye(16))+np.kron(np.eye(14),c) for r,c in zip(right,colour)]
    gv,p = eigh(sum(g@g for g in gauss))
    p = p[:,gv<1e-9]
    nv,u = eigh(p.conj().T@np.kron(np.eye(14),number)@p)
    p=p@u
    eps=np.array([[0,1],[-1,0]])
    raw=[]
    for t in range(2):
        for m in (-.5,.5):
            d=np.zeros((224,224),complex)
            for alpha in (-.5,.5):
                mult=np.zeros((14,14),complex)
                for col,(j,mm,nn) in enumerate(basis):
                    for row,(jp,mp,np_) in enumerate(basis):
                        if abs(mp-mm-m)<1e-9 and abs(np_-nn-alpha)<1e-9:
                            mult[row,col]=np.sqrt((2*j+1)/(2*jp+1))*cg(j,mm,.5,m,jp)*cg(j,nn,.5,alpha,jp)
                for beta in range(2):
                    d+=eps[int(alpha+.5),beta]*np.kron(mult,create[2*t+beta])
            raw.append(d)
    source=[p.conj().T@d@p for d in raw]
    s=[p.conj().T@np.kron(l,np.eye(16))@p for l in left]
    n=p.conj().T@np.kron(np.eye(14),number)@p
    # Build the actual CAR occupation basis from the physical dressed vacuum.
    vacuum=np.eye(16,dtype=complex)[:,np.argmin(nv)]
    w=np.zeros((16,16),complex)
    for state in range(16):
        v=vacuum.copy()
        for mode in reversed(range(4)):
            if state&(1<<mode):
                v=source[mode]@v
        w[:,state]=v
    source=[w.conj().T@d@w for d in source]
    s=[w.conj().T@a@w for a in s]
    p=p@w
    n=w.conj().T@n@w
    return dict(P=p,N=n,S=s,source=source,raw_source=raw,left=left,G=gauss,bare=[np.kron(np.eye(14),c) for c in create],raw_number=np.kron(np.eye(14),number),basis=basis)


def paths(frame, theta, imbalance=0.):
    """Conjugate the SAME source before charge transfer, not vacuum rotations."""
    outputs=[]
    for s in frame['S']:
        r=expm(-1j*theta*s)
        outputs.append([(1+imbalance)*r@d@r.conj().T-(1-imbalance)*r.conj().T@d@r for d in frame['source']])
    return outputs


def canonical_c(source):
    # Channel antisymmetric swap x spin antisymmetric intertwiner. No extra copy.
    e=np.array([[0,1],[-1,0]],complex)
    u=np.kron(e,e)
    target=[sum(u[a,b]*source[b] for b in range(4)) for a in range(4)]
    full=np.zeros(16,complex); full[-1]=1
    cf=np.zeros((16,16),complex)
    for state in range(16):
        v=full.copy()
        for a in reversed(range(4)):
            if state&(1<<a):
                v=target[a].conj().T@v
        cf[:,state]=v
    return cf,u


def local(frame,p,k=(0.,0.,0.),imbalance=0.):
    d=frame['source']; s=frame['S']; fld=vector_field(p['field_cutoff'])
    nf=len(fld['basis']); ident=np.eye(16); fi=np.eye(nf)
    bridge=paths(frame,p['theta'],imbalance)
    mass=p['mass']*sum(d[a]@d[a+2].conj().T+d[a+2]@d[a].conj().T for a in range(2))
    forward=[]
    for axis in range(3):
        # Keep the raw scalar part for unequal path weights; no trace removal.
        forward.append(p['hop']*sum((1 if a<2 else -1)*d[a]@bridge[axis][a].conj().T for a in range(4))/2)
    links=[np.exp(1j*k[a])*forward[a] for a in range(3)]
    s2=sum(a@a for a in s)
    hk=mass+sum(b+b.conj().T for b in links)+p['frame_cost']*s2
    h=np.kron(hk,fi)+np.kron(ident,p['omega']*(fld['H']-1.5*fi))
    bare_current=[1j*(b-b.conj().T) for b in links]
    bare_contact=[-(b+b.conj().T) for b in links]
    # Material successor: the actual path current, not Noether S, sources Q.
    # This linear neutral-vector coupling is the supplied law, not full Maxwell.
    h+=p['feedback']*sum(np.kron(bare_current[a],fld['Q'][a]) for a in range(3))
    current=[np.kron(bare_current[a],fi)+p['feedback']*np.kron(bare_contact[a],fld['Q'][a]) for a in range(3)]
    total=[np.kron(s[a],fi)+np.kron(ident,fld['F'][a]) for a in range(3)]
    return dict(H=h,current=current,source=[np.kron(a,fi) for a in d],S=[np.kron(a,fi) for a in s],J=total,N=np.kron(frame['N'],fi),field=fld,forward=forward,nf=nf)


def two_site(frame,p,angle=0.):
    d=frame['source']; n=frame['N']; nfock=16
    parity=np.diag([(-1)**i.bit_count() for i in range(nfock)])
    ds=[np.kron(a,np.eye(nfock)) for a in d]+[np.kron(parity,a) for a in d]
    ss=[[np.kron(a,np.eye(nfock)) for a in frame['S']],[np.kron(np.eye(nfock),a) for a in frame['S']]]
    nums=[np.kron(n,np.eye(nfock)),np.kron(np.eye(nfock),n)]
    fld=vector_field(p['field_cutoff']); nf=len(fld['basis']); fi=np.eye(nf)
    h=p['mass']*sum(ds[o+a]@ds[o+a+2].conj().T+ds[o+a+2]@ds[o+a].conj().T for o in (0,4) for a in range(2))
    h+=p['frame_cost']*sum(a@a for side in ss for a in side)
    # An oriented x-edge uses actual endpoint frame conjugation of the source.
    r=expm(-1j*p['theta']*ss[1][0])
    br=[r@ds[4+a].conj().T@r.conj().T-r.conj().T@ds[4+a].conj().T@r for a in range(4)]
    forward=p['hop']*sum((1 if a<2 else -1)*ds[a]@br[a] for a in range(4))/2
    link=np.exp(1j*angle)*forward
    h+=link+link.conj().T
    current=1j*(link-link.conj().T)
    contact=-(link+link.conj().T)
    h=np.kron(h,fi)+np.kron(np.eye(256),p['omega']*(fld['H']-1.5*fi))
    st=[ss[0][a]+ss[1][a] for a in range(3)]
    h+=p['feedback']*np.kron(current,fld['Q'][0])
    actual_current=np.kron(current,fi)+p['feedback']*np.kron(contact,fld['Q'][0])
    actual_contact=np.kron(contact,fi)-p['feedback']*np.kron(current,fld['Q'][0])
    return dict(H=h,current=actual_current,contact=actual_contact,N=np.kron(nums[0]+nums[1],fi),source=[np.kron(a,fi) for a in ds],S=[np.kron(a,fi) for a in st],field=fld,nf=nf)
