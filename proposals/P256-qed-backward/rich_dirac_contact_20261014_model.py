"""Full angular charged contact array. Sources resolve macrocells, not shell position.
Pure geometric helpers are reused; every joint state/receiver is recomputed here.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.linalg import eigh, expm
from rich_spinor_rot_20261014_model import angular_geometry, field_geometry
from rich_mobile_spinor_20261014_model import contact_values

LAW = 'charged-LS-dynamic-orientor-geometric-interference-Peierls-cavity-v1'

@dataclass(frozen=True)
class Parameters:
    geometry: str = 'facing'
    charged_lmax: int = 2
    neutral_lmax: int = 1
    inverse_inertia: float = 1.4
    neutral_inverse_inertia: float = 2.0
    charged_spin_orbit: float = 1.1
    neutral_spin_orbit: float = .25
    orientation: float = .45
    hopping: float = .55
    particle_gap: float = 2.0
    charge: float = -1.0
    spacing: float = 1.0
    field_cutoff: int = 1
    cavity_frequency: float = 1.15
    cavity_coupling: float = .22


def build(params=Parameters()):
    c=angular_geometry(params.charged_lmax); n=angular_geometry(params.neutral_lmax)
    f=field_geometry(params.field_cutoff)
    nc=len(c['basis']); nn=len(n['basis']); nf=len(f['basis'])
    spin=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)/2
    eye=np.eye(nc*2*nn)
    body=params.particle_gap*eye.astype(complex)
    body+=np.kron(np.kron(params.inverse_inertia*c['L2']/2,np.eye(2)),np.eye(nn))
    body+=np.kron(np.eye(nc*2),params.neutral_inverse_inertia*n['L2']/2)
    spins=[]; angular=[]
    for i in range(3):
        body+=params.charged_spin_orbit*np.kron(np.kron(c['L'][i],spin[i]),np.eye(nn))
        body+=params.neutral_spin_orbit*np.kron(np.kron(np.eye(nc),spin[i]),n['L'][i])
        body+=params.orientation*np.kron(np.kron(c['R'][i],np.eye(2)),n['R'][i])
        spins.append(np.kron(np.kron(np.kron(np.eye(nc),spin[i]),np.eye(nn)),np.eye(nf)))
        j=np.kron(np.kron(c['L'][i],np.eye(2)),np.eye(nn))+np.kron(np.kron(np.eye(nc),spin[i]),np.eye(nn))+np.kron(np.eye(nc*2),n['L'][i])
        angular.append(np.kron(j,np.eye(nf))+np.kron(eye,f['L'][i]))
    h0=np.kron(body,np.eye(nf))+np.kron(eye,params.cavity_frequency*f['number'])
    tunnels=[]; directions=params.spacing*np.eye(3)
    for i,d in enumerate(directions):
        a=contact_values(c['basis'],d); b=contact_values(c['basis'],-d)
        facing=np.outer(a.conj(),b)/np.vdot(a,a).real
        if params.geometry=='facing': orbital=-params.hopping*facing
        elif params.geometry=='interference': orbital=-params.hopping*(facing-facing.conj().T)/2
        else: raise ValueError(params.geometry)
        # The relative minus sign is a supplied coherent path sign, not a projected spin matrix.
        u=expm(-1j*params.charge*params.cavity_coupling*params.spacing*f['Q'][i])
        tunnels.append(np.kron(np.kron(np.kron(orbital,np.eye(2)),np.eye(nn)),u))
    source=np.zeros((len(h0),nc*2),complex)
    vacuum=f['basis'].index((0,0,0))
    for a in range(nc*2): source[a*nn*nf+vacuum,a]=1
    parity=np.kron(np.kron(np.kron(c['parity'],np.eye(2)),n['parity']),f['parity'])
    lc2=np.kron(np.kron(np.kron(c['L2'],np.eye(2)),np.eye(nn)),np.eye(nf))
    ln2=np.kron(np.kron(np.eye(nc*2),n['L2']),np.eye(nf))
    return {'H0':h0,'body_H':body,'T':tunnels,'directions':directions,'spin':np.array(spins),'J':np.array(angular),'parity':parity,'Lc2':lc2,'Ln2':ln2,'field_number':np.kron(eye,f['number']),'field_Q':np.array([np.kron(eye,q) for q in f['Q']]),'source':source,'params':params,'parameters':asdict(params),'charged':c,'neutral':n,'field':f,'law':LAW}


def hamiltonian(model,k):
    h=model['H0'].copy()
    for d,t in zip(model['directions'],model['T']):
        phase=np.exp(1j*np.dot(k,d)); h+=phase*t+phase.conjugate()*t.conj().T
    return h


def spectrum(model,k):
    e,v=eigh(hamiltonian(model,k),check_finite=False)
    # Canonical primitive Sz resolution within actual energy-degenerate subspaces.
    a=0
    while a<len(e):
        b=a+1
        while b<len(e) and abs(e[b]-e[a])<1e-9: b+=1
        w=v[:,a:b]; _,u=eigh(w.conj().T@model['spin'][2]@w)
        v[:,a:b]=w@u; a=b
    return e,v


def vertex(model,kout,kin,epsilon):
    q=np.asarray(kout)-np.asarray(kin); midpoint=(np.asarray(kout)+np.asarray(kin))/2
    j=np.zeros_like(model['H0'])
    for d,t in zip(model['directions'],model['T']):
        phase=np.exp(1j*np.dot(midpoint,d))
        j+=1j*model['params'].charge*np.dot(epsilon,d)*np.sinc(np.dot(q,d)/(2*np.pi))*(phase*t-phase.conjugate()*t.conj().T)
    return j


def contact(model,kout,kin,qin,epsin,qout,epsout):
    midpoint=(np.asarray(kout)+np.asarray(kin))/2; c=np.zeros_like(model['H0'])
    for d,t in zip(model['directions'],model['T']):
        phase=np.exp(1j*np.dot(midpoint,d))
        factor=np.dot(np.conjugate(epsout),d)*np.dot(epsin,d)*np.sinc(np.dot(qin,d)/(2*np.pi))*np.sinc(np.dot(qout,d)/(2*np.pi))
        c-=model['params'].charge**2*factor*(phase*t+phase.conjugate()*t.conj().T)
    return c


def spin_current(model,kout,kin,epsilon,axis=2):
    j=vertex(model,kout,kin,epsilon); s=model['spin'][axis]
    return (s@j+j@s)/2


def charged_green(model,k,z):
    e,v=spectrum(model,k); a=v.conj().T@model['source']
    return a.conj().T@((1/(z-e))[:,None]*a)


def scattering(model,kin,qin,epsin,qout,epsout,omega,initial=3,final=3,eta=.01,virtual=None):
    kin=np.asarray(kin); qin=np.asarray(qin); qout=np.asarray(qout); kout=kin+qin-qout
    ei,vi=spectrum(model,kin); ef,vf=spectrum(model,kout)
    ea,va=spectrum(model,kin+qin); eb,vb=spectrum(model,kin-qout)
    ids=np.arange(len(ea)) if virtual is None else np.asarray(virtual)
    a=vi[:,initial]; b=vf[:,final]; wo=omega+ei[initial]-ef[final]
    jin=vertex(model,kin+qin,kin,epsin); jout=vertex(model,kout,kin+qin,np.conjugate(epsout))
    je=vertex(model,kin-qout,kin,np.conjugate(epsout)); ji=vertex(model,kout,kin-qout,epsin)
    co=contact(model,kout,kin,qin,epsin,qout,epsout)
    aa=va[:,ids]; bb=vb[:,ids]
    direct=b.conj()@co@a
    absorption=np.sum((b.conj()@jout@aa)*(aa.conj().T@jin@a)/(ei[initial]+omega-ea[ids]+1j*eta))
    emission=np.sum((b.conj()@ji@bb)*(bb.conj().T@je@a)/(ei[initial]-wo-eb[ids]+1j*eta))
    return {'amplitude':direct+absorption+emission,'contact':direct,'absorption':absorption,'emission':emission,'incoming_energy':omega,'outgoing_energy':wo,'recoil_energy':ef[final]-ei[initial],'kout':kout}
