"""Mobile charged molecule: geometric angular contacts and a co-moving quantum orientor.
No low-band Hamiltonian, Dirac matrices, cavity, or borrowed response tensors.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.special import sph_harm_y
from scipy.linalg import eigh

LAW = 'mobile-angular-contact-comoving-quantum-orientor-v1'

@dataclass(frozen=True)
class Parameters:
    charged_lmax: int = 2
    neutral_lmax: int = 2
    inverse_inertia: float = 1.4
    neutral_inverse_inertia: float = 2.0
    orientation: float = .85
    rotor_spin: float = 1.1
    hopping: float = .55
    polar_bias: float = .65
    particle_gap: float = 2.0
    charge: float = -1.0
    spacing: float = 1.0


def angular(lmax):
    basis = [(l,m) for l in range(lmax+1) for m in range(-l,l+1)]
    x,w = np.polynomial.legendre.leggauss(18)
    phi = 2*np.pi*np.arange(36)/36
    theta,azimuth = np.meshgrid(np.arccos(x),phi,indexing='ij')
    r = np.column_stack([np.sin(theta).ravel()*np.cos(azimuth).ravel(),np.sin(theta).ravel()*np.sin(azimuth).ravel(),np.cos(theta).ravel()])
    weights = np.repeat(w,36)*2*np.pi/36
    y = np.column_stack([sph_harm_y(l,m,theta,azimuth).ravel() for l,m in basis])
    lookup = {b:i for i,b in enumerate(basis)}
    raising = np.zeros((len(basis),len(basis)),complex)
    for j,(l,m) in enumerate(basis):
        if m<l: raising[lookup[(l,m+1)],j] = np.sqrt(l*(l+1)-m*(m+1))
    angular_momentum = np.array([(raising+raising.conj().T)/2,(raising-raising.conj().T)/(2j),np.diag([m for l,m in basis])])
    position = np.array([y.conj().T@((weights*r[:,i])[:,None]*y) for i in range(3)])
    return {'basis':basis,'L':angular_momentum,'R':position,'L2':np.diag([l*(l+1) for l,m in basis])}


def contact_values(basis,direction):
    d = np.asarray(direction,float)/np.linalg.norm(direction)
    theta = np.arccos(d[2]); phi = np.arctan2(d[1],d[0])
    return np.array([sph_harm_y(l,m,theta,phi) for l,m in basis])


def build(params=Parameters()):
    c=angular(params.charged_lmax); n=angular(params.neutral_lmax)
    nc=len(c['basis']); nn=len(n['basis']); eye=np.eye(nc*2*nn)
    spin=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)/2
    h=params.particle_gap*eye.astype(complex)
    h+=np.kron(np.kron(params.inverse_inertia*c['L2']/2,np.eye(2)),np.eye(nn))
    h+=np.kron(np.eye(nc*2),params.neutral_inverse_inertia*n['L2']/2)
    h+=params.polar_bias*np.kron(np.kron(c['R'][2],np.eye(2)),np.eye(nn))
    spins=[]; total=[]; neutral_l=[]
    for i in range(3):
        h+=params.orientation*np.kron(np.kron(c['R'][i],np.eye(2)),n['R'][i])
        h+=params.rotor_spin*np.kron(np.kron(np.eye(nc),spin[i]),n['L'][i])
        spins.append(np.kron(np.kron(np.eye(nc),spin[i]),np.eye(nn)))
        neutral_l.append(np.kron(np.eye(nc*2),n['L'][i]))
        total.append(np.kron(np.kron(c['L'][i],np.eye(2)),np.eye(nn))+spins[-1]+neutral_l[-1])
    directions=params.spacing*np.eye(3)
    tunnels=[]
    for d in directions:
        facing=contact_values(c['basis'],d); receiving=contact_values(c['basis'],-d)
        # Normalize by the complete angular-shell point evaluation norm, not fit velocity.
        orbital=-params.hopping*np.outer(facing.conj(),receiving)/np.vdot(facing,facing).real
        tunnels.append(np.kron(np.kron(orbital,np.eye(2)),np.eye(nn)))
    source=np.zeros((len(h),nc*2),complex)
    for a in range(nc*2): source[a*nn,a]=1
    return {'H0':h,'T':tunnels,'directions':directions,'spin':np.array(spins),'J':np.array(total),'Ln':np.array(neutral_l),'N2':np.kron(np.eye(nc*2),n['L2']),'source':source,'params':params,'parameters':asdict(params),'charged':c,'neutral':n,'law':LAW}

def hamiltonian(model,k):
    h=model['H0'].copy()
    for d,t in zip(model['directions'],model['T']):
        phase=np.exp(1j*np.dot(k,d)); h+=phase*t+phase.conjugate()*t.conj().T
    return h


def spectrum(model,k):
    return eigh(hamiltonian(model,k),check_finite=False)


def vertex(model,kout,kin,epsilon):
    """J=-dH/dA for H(k-qcharge A); exact straight-cell-link Fourier source."""
    q=np.asarray(kout)-np.asarray(kin); midpoint=(np.asarray(kout)+np.asarray(kin))/2
    j=np.zeros_like(model['H0'])
    for d,t in zip(model['directions'],model['T']):
        phase=np.exp(1j*np.dot(midpoint,d))
        j+=1j*model['params'].charge*np.dot(epsilon,d)*np.sinc(np.dot(q,d)/(2*np.pi))*(phase*t-phase.conjugate()*t.conj().T)
    return j


def contact(model,kout,kin,qin,epsin,qout,epsout):
    """Mixed d2H/dAout* dAin, net momentum qin-qout and both line factors."""
    midpoint=(np.asarray(kout)+np.asarray(kin))/2
    c=np.zeros_like(model['H0'])
    for d,t in zip(model['directions'],model['T']):
        phase=np.exp(1j*np.dot(midpoint,d))
        factor=np.dot(np.conjugate(epsout),d)*np.dot(epsin,d)*np.sinc(np.dot(qin,d)/(2*np.pi))*np.sinc(np.dot(qout,d)/(2*np.pi))
        c-=model['params'].charge**2*factor*(phase*t+phase.conjugate()*t.conj().T)
    return c


def spin_current(model,kout,kin,epsilon,axis):
    j=vertex(model,kout,kin,epsilon); s=model['spin'][axis]
    return (s@j+j@s)/2


def charged_green(model,k,omega,eta=.01):
    e,v=spectrum(model,k); a=v.conj().T@model['source']
    return a.conj().T@((1/(omega+1j*eta-e))[:,None]*a)


def scattering(model,kin,qin,epsin,qout,epsout,omega,initial=0,final=0,eta=.01,virtual=None):
    """Finite-energy nonforward external-vector-source response; no photon normalization."""
    kin=np.asarray(kin); qin=np.asarray(qin); qout=np.asarray(qout); kout=kin+qin-qout
    ei,vi=spectrum(model,kin); ef,vf=spectrum(model,kout)
    ea,va=spectrum(model,kin+qin); eb,vb=spectrum(model,kin-qout)
    ids=np.arange(len(ea)) if virtual is None else np.asarray(virtual)
    a=vi[:,initial]; b=vf[:,final]; outgoing=omega+ei[initial]-ef[final]
    jin=vertex(model,kin+qin,kin,epsin)
    jout=vertex(model,kout,kin+qin,np.conjugate(epsout))
    je=vertex(model,kin-qout,kin,np.conjugate(epsout))
    ji=vertex(model,kout,kin-qout,epsin)
    c=contact(model,kout,kin,qin,epsin,qout,epsout)
    aa=va[:,ids]; bb=vb[:,ids]
    direct=b.conj()@c@a
    absorption=np.sum((b.conj()@jout@aa)*(aa.conj().T@jin@a)/(ei[initial]+omega-ea[ids]+1j*eta))
    emission=np.sum((b.conj()@ji@bb)*(bb.conj().T@je@a)/(ei[initial]-outgoing-eb[ids]+1j*eta))
    return {'amplitude':direct+absorption+emission,'contact':direct,'absorption':absorption,'emission':emission,'incoming_energy':omega,'outgoing_energy':outgoing,'recoil_energy':ef[final]-ei[initial],'kout':kout}
