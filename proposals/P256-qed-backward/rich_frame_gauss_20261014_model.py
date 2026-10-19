"""New colour-Gauss field-frame carrier; no primitive physical spin input.

All matrices use hbar=1. A finite SO(3) material rotor and three vector
oscillators are regulators, not a Maxwell continuum or a self-bound electron.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy.linalg import block_diag, eigh
from scipy.special import erf
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan


def angular(j):
    m = np.arange(-j, j+1)
    plus = np.diag(np.sqrt((j-m[:-1])*(j+m[:-1]+1)), -1)
    return [(plus+plus.T)/2, (plus-plus.T)/(2j), np.diag(m)]


def charge_geometry(p):
    """Finite Gaussian clouds; exact longitudinal electric-Gauss energy.

    rho(x)=qN/6 sum Gaussian_sigma(x-W r_a); div E=rho/epsilon.
    Minimizing epsilon/2 integral E^2 gives the pair kernel below, including
    finite Gaussian self-energy. This supplied electric quadratic law is not
    claimed to emerge from the frame metric.
    """
    points = p['radius']*np.concatenate((np.eye(3),-np.eye(3)))
    distance = np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2)
    kernel = np.full((6,6),1/(np.sqrt(np.pi)*p['cloud_width']))
    off = distance > 0
    kernel[off] = erf(distance[off]/(2*p['cloud_width']))/distance[off]
    energy = p['charge']**2*np.sum(kernel)/(8*np.pi*p['permittivity']*36)
    magnetic_tensor = p['charge']*(p['radius']**2/3+p['cloud_width']**2)
    return float(energy), float(magnetic_tensor)


@lru_cache(None)
def cg(j, m, k, n, jp):
    if abs(m+n) > jp:
        return 0.
    r = lambda x: Rational(int(round(2*x)), 2)
    return float(clebsch_gordan(r(j), r(k), r(jp), r(m), r(n), r(m+n)))


def frame_gauss():
    """Full j=0,1/2,1 frame regulator x full two-colour four-state Fock.

    The physical left pullback f(U)->f(g^-1 U) is -J^T on m;
    right pullback f(U)->f(Ug) is J on n. Gauss is right+colour.
    Dressed source is epsilon contraction D(U)c^dag, not bare c^dag.
    """
    basis = [(j, m, n) for j in (0., .5, 1.) for m in np.arange(-j,j+1) for n in np.arange(-j,j+1)]
    left = [block_diag(*[np.kron(-a.T, np.eye(int(2*j+1))) for j in (0., .5, 1.) for a in [angular(j)[axis]]]) for axis in range(3)]
    right = [block_diag(*[np.kron(np.eye(int(2*j+1)), angular(j)[axis]) for j in (0., .5, 1.)]) for axis in range(3)]
    create = []
    for colour in range(2):
        c = np.zeros((4,4),complex)
        for state in range(4):
            if not state & (1 << colour):
                c[state | (1 << colour),state] = (-1)**((state & ((1 << colour)-1)).bit_count())
        create.append(c)
    number = sum(c @ c.conj().T for c in create)
    colour = [sum(a[x,y]*create[x]@create[y].conj().T for x in range(2) for y in range(2)) for a in angular(.5)]
    gauss = [np.kron(r,np.eye(4))+np.kron(np.eye(14),t) for r,t in zip(right,colour)]
    g2 = sum(g@g for g in gauss)
    values, vectors = eigh(g2)
    p = vectors[:,values < 1e-9]
    n = p.conj().T@np.kron(np.eye(14),number)@p
    nv, un = eigh(n)
    p = p@un
    # Keep number sectors grouped. Eigensolver freedom within the doublet is harmless.
    eps = np.array([[0,1],[-1,0]])
    dressed = []
    for m in (-.5,.5):
        raw = np.zeros((56,56),complex)
        for alpha in (-.5,.5):
            mult = np.zeros((14,14),complex)
            for col,(j,mm,nn) in enumerate(basis):
                for row,(jp,mp,np_) in enumerate(basis):
                    if abs(mp-mm-m)<1e-9 and abs(np_-nn-alpha)<1e-9:
                        mult[row,col] = np.sqrt((2*j+1)/(2*jp+1))*cg(j,mm,.5,m,jp)*cg(j,nn,.5,alpha,jp)
            for beta in range(2):
                raw += eps[int(alpha+.5),beta]*np.kron(mult,create[beta])
        dressed.append((raw,p.conj().T@raw@p))
    bare = [np.kron(np.eye(14),c) for c in create]
    physleft = [p.conj().T@np.kron(a,np.eye(4))@p for a in left]
    return dict(basis=basis,P=p,N=nv,S=physleft,G=gauss,source=[d[1] for d in dressed],raw_source=[d[0] for d in dressed],bare=bare,left=left,number=number)


def vector_field(cutoff):
    basis = [(x,y,z) for x in range(cutoff+1) for y in range(cutoff+1-x) for z in range(cutoff+1-x-y)]
    index = {n:i for i,n in enumerate(basis)}
    annih = []
    for a in range(3):
        b = np.zeros((len(basis),len(basis)),complex)
        for col,n in enumerate(basis):
            if n[a]:
                nn = list(n); nn[a] -= 1
                b[index[tuple(nn)],col] = np.sqrt(n[a])
        annih.append(b)
    q = [(b+b.conj().T)/np.sqrt(2) for b in annih]
    f = [-1j*(annih[(a+1)%3].conj().T@annih[(a+2)%3]-annih[(a+2)%3].conj().T@annih[(a+1)%3]) for a in range(3)]
    return dict(basis=basis,Q=q,F=f,H=np.diag([sum(n)+1.5 for n in basis]),vacuum=index[(0,0,0)])


def carrier(p, l, occupation=1):
    """Every M and K of this l is present as identical K copies.

    Calculations return one K block; the receiver must restore 2l+1 copies.
    No l or physical J is selected as an assumed ground sector.
    """
    frame = frame_gauss()
    ix = np.flatnonzero(abs(frame['N']-occupation)<1e-8)
    s = [a[np.ix_(ix,ix)] for a in frame['S']]
    nf = len(ix); nm = 2*l+1
    field = vector_field(p['field_cutoff']); nb = len(field['basis'])
    li = [np.kron(a,np.eye(nf)) for a in angular(l)]
    si = [np.kron(np.eye(nm),a) for a in s]
    ji = [a+b for a,b in zip(li,si)]
    ident = np.eye(nm*nf)
    electric_energy, magnetic_tensor = charge_geometry(p)
    c = magnetic_tensor*occupation
    da = [np.kron(a,np.eye(nb))-c*p['magnetic_amplitude']*np.kron(ident,b) for a,b in zip(ji,field['Q'])]
    db = [np.kron(a,np.eye(nb))-c*p['magnetic_amplitude']*np.kron(ident,b) for a,b in zip(li,field['Q'])]
    h = (p['field_frequency']*np.kron(ident,field['H'])).astype(complex)
    h += sum(a@a for a in da)/(2*p['common_inertia'])
    h += sum(a@a for a in db)/(2*p['relative_inertia'])
    h += np.kron(sum(a@a for a in si),np.eye(nb))/(2*p['frame_inertia'])
    h += electric_energy*occupation**2*np.eye(len(h))
    mu = [c*(a/p['common_inertia']+b/p['relative_inertia']) for a,b in zip(da,db)]
    total = [np.kron(a,np.eye(nb))+np.kron(ident,b) for a,b in zip(ji,field['F'])]
    return dict(H=h,L=li,S=si,J=ji,field=field,mu=mu,total=total,C=c,frame=frame,sector=ix,nb=nb,da=da,db=db)


def rotation(l, alpha, beta, gamma):
    m = np.arange(-l,l+1)
    vals, vec = eigh(angular(l)[1])
    d = (vec*np.exp(-1j*beta*vals))@vec.conj().T
    return np.exp(-1j*alpha*m)[:,None]*d*np.exp(-1j*gamma*m)[None,:]


def spatial_multipliers(lmax, wavevector, radius, quadrature, cloud_width):
    """Haar projection of six finite Gaussian charge clouds on a material rotor.

    SO(3) basis sqrt(2l+1) D^l*_{MK}, all M,K and l=0..lmax.
    Returns density and velocity multipliers, not an assigned spin current.
    """
    basis = [(l,m,k) for l in range(lmax+1) for m in range(-l,l+1) for k in range(-l,l+1)]
    size = len(basis); rho = np.zeros((size,size),complex)
    vel = np.zeros((3,3,size,size),complex)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    angles = np.arange(2*quadrature)*np.pi/quadrature
    points = radius*np.concatenate((np.eye(3),-np.eye(3)))
    for t,w in zip(nodes,weights):
        beta = np.arccos(t)
        for alpha in angles:
            for gamma in angles:
                rz = lambda a: np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
                ry = np.array([[t,0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,t]])
                xyz = points@(rz(alpha)@ry@rz(gamma)).T
                phase = np.exp(-1j*xyz@np.array(wavevector)-.5*cloud_width**2*np.dot(wavevector,wavevector))
                functions = np.concatenate([np.sqrt(2*l+1)*rotation(l,alpha,beta,gamma).conj().ravel() for l in range(lmax+1)])
                outer = np.outer(functions.conj(),functions)*w/(2*len(angles)**2)
                rho += np.mean(phase)*outer
                for b in range(3):
                    velocity = np.cross(np.eye(3)[b],xyz-1j*cloud_width**2*np.array(wavevector))
                    for a in range(3):
                        vel[a,b] += np.mean(phase*velocity[:,a])*outer
    # D* basis has conventional positive physical left generators on M.
    ll = [block_diag(*[np.kron(angular(l)[a],np.eye(2*l+1)) for l in range(lmax+1)]) for a in range(3)]
    return dict(basis=basis,rho=rho,velocity=vel,L=ll)


def spatial_consumer(p, wavevector):
    spatial = spatial_multipliers(p['lmax'],wavevector,p['radius'],p['quadrature'],p['cloud_width'])
    frame = frame_gauss(); ix = np.flatnonzero(abs(frame['N']-1)<1e-8)
    s = [a[np.ix_(ix,ix)] for a in frame['S']]
    nw = len(spatial['basis']); field = vector_field(p['field_cutoff']); nb = len(field['basis'])
    electric_energy, c = charge_geometry(p)
    eye = np.eye(nw*2)
    l = [np.kron(a,np.eye(2)) for a in spatial['L']]
    j = [a+np.kron(np.eye(nw),b) for a,b in zip(l,s)]
    d = [np.kron(a,np.eye(nb))-c*p['magnetic_amplitude']*np.kron(eye,b) for a,b in zip(j,field['Q'])]
    e = [np.kron(a,np.eye(nb))-c*p['magnetic_amplitude']*np.kron(eye,b) for a,b in zip(l,field['Q'])]
    h = p['field_frequency']*np.kron(eye,field['H'])+sum(a@a for a in d)/(2*p['common_inertia'])+sum(a@a for a in e)/(2*p['relative_inertia'])
    h += (.75/(2*p['frame_inertia'])+electric_energy)*np.eye(len(h))
    rho = p['charge']*np.kron(np.kron(spatial['rho'],np.eye(2)),np.eye(nb))
    current = []
    for a in range(3):
        current_a = np.zeros_like(h)
        for b in range(3):
            v = p['charge']*np.kron(np.kron(spatial['velocity'][a,b],np.eye(2)),np.eye(nb))
            omega = d[b]/p['common_inertia']+e[b]/p['relative_inertia']
            current_a += (omega@v+v@omega)/2
        current.append(current_a)
    return dict(H=h,rho=rho,current=current,spatial=spatial,field=field,frame=frame)
