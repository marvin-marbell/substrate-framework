"""NEW complete angular-shell molecular spinor with before-Galerkin EM sources.
Physical charge q=-1, hbar=charged sphere radius=1. Both sphere inertias positive.
Optional isotropic three-vector external cavity: input oscillator restoration,
NOT a derived two-transverse massless photon phase. All old epochs immutable.
"""
from dataclasses import dataclass,asdict
from itertools import product
import numpy as np
from scipy.linalg import eigh,expm
from scipy.special import sph_harm_y

LAW='rotational-sphere-neutral-orientor-covariant-cavity-v1'

@dataclass(frozen=True)
class Parameters:
    charged_lmax:int=2
    neutral_lmax:int=1
    inverse_inertia:float=1.4
    neutral_inverse_inertia:float=2.0
    orientation:float=.85
    rotor_spin:float=1.1
    omega:float=1.15
    coupling:float=0.0
    field_cutoff:int=1
    charge:float=-1.0
    particle_gap:float=2.0


def angular_geometry(lmax,ntheta=18,nphi=36):
    basis=[(l,m) for l in range(lmax+1) for m in range(-l,l+1)]
    x,w=np.polynomial.legendre.leggauss(ntheta)
    phi=2*np.pi*np.arange(nphi)/nphi
    theta=np.arccos(x)
    tt,pp=np.meshgrid(theta,phi,indexing='ij')
    r=np.column_stack([np.sin(tt).ravel()*np.cos(pp).ravel(),np.sin(tt).ravel()*np.sin(pp).ravel(),np.cos(tt).ravel()])
    weights=np.repeat(w,nphi)*2*np.pi/nphi
    y=np.column_stack([sph_harm_y(l,m,tt,pp).ravel() for l,m in basis])
    dim=len(basis); lookup={b:i for i,b in enumerate(basis)}
    lp=np.zeros((dim,dim),complex); lz=np.diag([m for l,m in basis]).astype(complex)
    for a,(l,m) in enumerate(basis):
        if m<l:lp[lookup[(l,m+1)],a]=np.sqrt(l*(l+1)-m*(m+1))
    angular=np.array([(lp+lp.conj().T)/2,(lp-lp.conj().T)/(2j),lz])
    geometry={'basis':basis,'r':r,'weights':weights,'Y':y,'L':angular,
              'L2':np.diag([l*(l+1) for l,m in basis]),'parity':np.diag([(-1)**l for l,m in basis])}
    geometry['R']=np.array([multiplication(geometry,r[:,i]) for i in range(3)])
    return geometry


def multiplication(geometry,values):
    """P_L f(r) P_L; product functions MUST be multiplied BEFORE this operation."""
    y=geometry['Y'];return y.conj().T@((geometry['weights']*values)[:,None]*y)


def one_current(geometry,params,q,epsilon):
    """P[-d (L-qcharge r cross A)^2/(2I)/dA]P from continuum kinetic.
    Since L preserves whole shells, anticommutators with P f P are exact Galerkin
    of the first derivative. No commutation of finite projected R is assumed.
    """
    r=geometry['r']; phase=np.exp(1j*r@np.asarray(q)); f=np.cross(r,np.asarray(epsilon))*phase[:,None]
    result=np.zeros_like(geometry['L2'],dtype=complex)
    for i in range(3):
        fm=multiplication(geometry,f[:,i]); l=geometry['L'][i]
        result+=params.charge*params.inverse_inertia*(l@fm+fm@l)/2
    return result


def one_contact(geometry,params,qout,epsout,qin,epsin):
    """P [qcharge^2/I (r cross Aout*) dot (r cross Ain)]P.
    Deliberately NOT product of separately Galerkin-projected R or vector fields.
    """
    r=geometry['r']; out=np.cross(r,np.asarray(epsout).conj()); inc=np.cross(r,np.asarray(epsin))
    values=np.sum(out*inc,axis=1)*np.exp(1j*r@(np.asarray(qin)-np.asarray(qout)))
    return params.charge**2*params.inverse_inertia*multiplication(geometry,values)


def field_geometry(cutoff):
    """Complete isotropic oscillator excitation shells, QiQj before field projection."""
    large=[n for n in product(range(cutoff+3),repeat=3) if sum(n)<=cutoff+2]
    selected=[i for i,n in enumerate(large) if sum(n)<=cutoff]
    lookup={n:i for i,n in enumerate(large)}; dim=len(large)
    annih=[]
    for axis in range(3):
        a=np.zeros((dim,dim),complex)
        for column,n in enumerate(large):
            if n[axis]:
                m=list(n);m[axis]-=1;a[lookup[tuple(m)],column]=np.sqrt(n[axis])
        annih.append(a)
    qlarge=np.array([(a+a.conj().T)/np.sqrt(2) for a in annih]); sel=np.ix_(selected,selected)
    q=np.array([a[sel] for a in qlarge]); qq=np.array([[(qlarge[i]@qlarge[j])[sel] for j in range(3)] for i in range(3)])
    lf=np.zeros((3,dim,dim),complex)
    for i,j,k,sign in [(0,1,2,1),(1,2,0,1),(2,0,1,1),(0,2,1,-1),(2,1,0,-1),(1,0,2,-1)]:
        lf[i]+=-1j*sign*annih[j].conj().T@annih[k]
    ns=np.array([sum(large[i]) for i in selected])
    return {'basis':[large[i] for i in selected],'Q':q,'QQ':qq,'L':np.array([a[sel] for a in lf]),
            'number':np.diag(ns),'parity':np.diag((-1)**ns)}


def build(params=Parameters()):
    c=angular_geometry(params.charged_lmax); n=angular_geometry(params.neutral_lmax)
    nc=len(c['basis']); nn=len(n['basis']); ispin=np.eye(2); ic=np.eye(nc); inn=np.eye(nn)
    spin=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)/2
    hbody=np.kron(np.kron(params.inverse_inertia*c['L2']/2,ispin),inn).astype(complex)
    hbody+=np.kron(np.eye(nc*2),params.neutral_inverse_inertia*n['L2']/2)
    hbody+=params.particle_gap*np.eye(len(hbody))
    angular=[]; primitive_spin=[]
    for i in range(3):
        hbody+=params.orientation*np.kron(np.kron(c['R'][i],ispin),n['R'][i])
        hbody+=params.rotor_spin*np.kron(np.kron(ic,spin[i]),n['L'][i])
        angular.append(np.kron(np.kron(c['L'][i],ispin),inn)+np.kron(np.kron(ic,spin[i]),inn)+np.kron(np.eye(nc*2),n['L'][i]))
        primitive_spin.append(np.kron(np.kron(ic,spin[i]),inn))
    parity=np.kron(np.kron(c['parity'],ispin),n['parity'])
    # Keep zero-coupling receiver as actual bare body, not a borrowed frozen tensor.
    field=field_geometry(params.field_cutoff) if params.coupling else None
    nf=len(field['basis']) if field is not None else 1
    h=np.kron(hbody,np.eye(nf)); generators=np.array([np.kron(a,np.eye(nf)) for a in angular])
    if field is not None:
        h+=np.kron(np.eye(len(hbody)),params.omega*field['number'])
        for i in range(3):
            ji=np.kron(np.kron(one_current(c,params,np.zeros(3),np.eye(3)[i]),ispin),inn)
            h-=params.coupling*np.kron(ji,field['Q'][i])
            for j in range(3):
                cij=np.kron(np.kron(one_contact(c,params,np.zeros(3),np.eye(3)[i],np.zeros(3),np.eye(3)[j]),ispin),inn)
                h+=params.coupling**2*np.kron(cij,field['QQ'][i,j])/2
            generators[i]+=np.kron(np.eye(len(hbody)),field['L'][i])
        parity=np.kron(parity,field['parity'])
    return {'H':h,'body_H':hbody,'charged':c,'neutral':n,'field':field,'params':params,'parameters':asdict(params),
            'angular':generators,'body_angular':np.array(angular),'spin':np.array([np.kron(a,np.eye(nf)) for a in primitive_spin]),
            'parity':parity,'field_dimension':nf,'law':LAW}


def current(model,q,epsilon):
    p=model['params']; c=model['charged']; nn=len(model['neutral']['basis']); nf=model['field_dimension']
    j=np.kron(np.kron(one_current(c,p,q,epsilon),np.eye(2)),np.eye(nn))
    result=np.kron(j,np.eye(nf))
    if model['field'] is not None:
        for i in range(3):
            # Complex incoming probe: contact signature requires real outgoing uniform mode.
            ci=np.kron(np.kron(one_contact(c,p,np.zeros(3),np.eye(3)[i],q,epsilon),np.eye(2)),np.eye(nn))
            result-=p.coupling*np.kron(ci,model['field']['Q'][i])
    return result


def contact(model,qout,epsout,qin,epsin):
    nn=len(model['neutral']['basis']); nf=model['field_dimension']
    return np.kron(np.kron(np.kron(one_contact(model['charged'],model['params'],qout,epsout,qin,epsin),np.eye(2)),np.eye(nn)),np.eye(nf))


def solve(model):
    e,v=eigh(model['H'],check_finite=False); start=0
    while start<len(e):
        end=start+1
        while end<len(e) and abs(e[end]-e[start])<1e-9:end+=1
        w=v[:,start:end]; _,u=eigh(w.conj().T@(10*model['parity']+model['angular'][2])@w)
        v[:,start:end]=w@u;start=end
    model['energies']=e;model['vectors']=v;return model


def amplitude(model,initial,final,jin,jout,co,omega,eta=1e-5,virtual=None):
    e=model['energies'];v=model['vectors']; ids=np.arange(len(e)) if virtual is None else np.asarray(virtual)
    a=v[:,initial];b=v[:,final];w=v[:,ids];ei=e[initial];wo=omega+ei-e[final]
    return b.conj()@co@a+np.sum((b.conj()@jout.conj().T@w)*(w.conj().T@jin@a)/(ei+omega-e[ids]+1j*eta))+np.sum((b.conj()@jin@w)*(w.conj().T@jout.conj().T@a)/(ei-wo-e[ids]+1j*eta))


def vacuum_sources(model):
    """N0=empty charge, neutral l0 and field vacuum. All charged angular/spin sources.
    Full N1 addition G. No occupied-reference removal poles or positron interpretation.
    """
    nc=len(model['charged']['basis']); nn=len(model['neutral']['basis']); nf=model['field_dimension']
    source=np.zeros((len(model['H']),2*nc),complex)
    # Angular ordering starts l0,m0; field basis starts (0,0,0).
    for a in range(2*nc):source[a*nn*nf,a]=1
    return source


def charged_green(model,omega,eta=1e-5):
    amplitudes=model['vectors'].conj().T@vacuum_sources(model)
    return np.einsum('na,nb,n->ab',amplitudes.conj(),amplitudes,1/(omega+1j*eta-model['energies']))


def rotate(model,axis,angle):
    return expm(-1j*angle*sum(np.asarray(axis)[i]*model['angular'][i] for i in range(3)))
