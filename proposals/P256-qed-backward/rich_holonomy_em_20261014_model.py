"""Finite spatial EM collective law on the complete positive-frame Fock space.
Only the raw scalar-colour/right-Gauss encoder is reused. No old oscillator law.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh, expm
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import expm_multiply
from scipy.special import erf
from rich_frame_holonomy_20261014_model import raw_frame, canonical_c


def profile(points, axis, width):
    """curl(exp(-r²/(2w²)) e_axis), an actual divergence-free 3D vector."""
    g=np.exp(-np.sum(points*points,axis=-1)/(2*width**2))
    grad=-points*g[...,None]/width**2
    direction=np.eye(3)[axis]
    return np.cross(grad,direction)


def geometry(p):
    width=p['profile_width']
    axis=np.linspace(-p['spatial_wall'],p['spatial_wall'],p['spatial_grid'])
    mesh=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
    dv=(axis[1]-axis[0])**3
    raw=np.stack([profile(mesh,a,width) for a in (1,2)],axis=-2)
    gram=np.einsum('nia,nja->ij',raw.reshape(-1,2,3),raw.reshape(-1,2,3))*dv
    val,vec=eigh(gram)
    transform=vec@np.diag(val**-.5)@vec.T
    fields=np.einsum('ij,...ja->...ia',transform,raw)
    routes=[]
    # Distinct bowed paths in all three dimensions; no single fictitious k.
    bends=[(0.,0.,0.),(0.,.8,.25),(0.,-.45,.9),(0.,-.6,-.65)]
    endpoints=np.array([[-p['separation']/2,0,0],[p['separation']/2,0,0]])
    line_coeff=[]
    for bend in bends:
        t=np.linspace(0,1,p['path_points'])
        points=(1-t[:,None])*endpoints[0]+t[:,None]*endpoints[1]+np.sin(np.pi*t)[:,None]*np.array(bend)
        steps=np.diff(points,axis=0)
        middle=(points[1:]+points[:-1])/2
        # Density/current use a common Gaussian smear. Source coefficient is
        # therefore the field profile convolved with that SAME charge shape.
        effective_width=np.sqrt(width**2+p['charge_width']**2)
        nodes,weights=np.polynomial.legendre.leggauss(8)
        samples=middle[:,None,:]+nodes[None,:,None]*steps[:,None,:]/2
        fs=np.stack([profile(samples,a,effective_width)*(width/effective_width)**3 for a in (1,2)],axis=2)
        fs=np.einsum('ij,nkja->nkia',transform,fs)
        fs=np.einsum('k,nkia->nia',weights/2,fs)
        line_coeff.append(np.einsum('nia,na->i',fs,steps))
        routes.append({'points':points,'steps':steps,'middle':middle,'mode_values':fs})
    sigma=p['charge_width']
    dist=p['separation']
    coulomb=np.array([[1/(np.sqrt(np.pi)*sigma),erf(dist/(2*sigma))/dist],[erf(dist/(2*sigma))/dist,1/(np.sqrt(np.pi)*sigma)]])/(4*np.pi*p['epsilon'])
    density=[]; electric=[]
    for centre in endpoints:
        r=mesh-centre
        radii=np.linalg.norm(r,axis=-1)
        safe=np.maximum(radii,1e-12)
        rho=np.exp(-radii**2/(2*sigma**2))/(2*np.pi*sigma**2)**1.5
        enclosed=erf(safe/(np.sqrt(2)*sigma))-np.sqrt(2/np.pi)*safe/sigma*np.exp(-safe**2/(2*sigma**2))
        e=r*enclosed[...,None]/(4*np.pi*p['epsilon']*safe[...,None]**3)
        density.append(rho);electric.append(e)
    # Momentum distribution is the DFT of the actual profiles, not a fitted q.
    ff=np.fft.fftn(fields,axes=(0,1,2),norm='ortho')
    freq=2*np.pi*np.fft.fftfreq(len(axis),axis[1]-axis[0])
    qmesh=np.stack(np.meshgrid(freq,freq,freq,indexing='ij'),axis=-1)
    weights=np.sum(abs(ff)**2,axis=-1)
    qrms=[np.sqrt(np.sum(weights[...,a]*np.sum(qmesh*qmesh,axis=-1))/np.sum(weights[...,a])) for a in range(2)]
    return dict(axis=axis,mesh=mesh,dv=dv,fields=fields,gram=gram,transform=transform,routes=routes,line=np.array(line_coeff),centres=endpoints,coulomb=coulomb,density=np.array(density),electric=np.array(electric),momentum_rms=qrms)


def field_grid(p):
    n=p['field_grid'];wall=p['field_wall'];step=2*wall/(n+1)
    x=np.linspace(-wall+step,wall-step,n)
    q0,q1=np.meshgrid(x,x,indexing='ij')
    q=[q0.ravel(),q1.ravel()]
    lap=diags([-np.ones(n-1),2*np.ones(n),-np.ones(n-1)],[-1,0,1],format='csr')/step**2
    kinetic=(kron(lap,eye(n))+kron(eye(n),lap))/(2*p['epsilon'])
    mom=diags([1j*np.ones(n-1),-1j*np.ones(n-1)],[-1,1],format='csr')/(2*step)
    momentum=[kron(mom,eye(n),format='csr'),kron(eye(n),mom,format='csr')]
    return dict(q=q,P=momentum,kinetic=kinetic,step=step,x=x,parity=csr_matrix(np.eye(n*n)[::-1]))


def build(p, frame=None, external=(0.,0.), induction=1., theta=None, spatial_external=(0.,0.,0.)):
    frame=raw_frame() if frame is None else frame
    g=geometry(p);f=field_grid(p);local_d=frame['source'];local_s=frame['S']
    parity=np.diag([(-1)**state.bit_count() for state in range(16)])
    d=[np.kron(a,np.eye(16)) for a in local_d]+[np.kron(parity,a) for a in local_d]
    site_s=[[np.kron(a,np.eye(16)) for a in local_s],[np.kron(np.eye(16),a) for a in local_s]]
    s=[site_s[0][a]+site_s[1][a] for a in range(3)]
    theta=p['theta'] if theta is None else theta
    nf=p['field_grid']**2;fi=eye(nf,format='csr')
    nums=[np.kron(frame['N'],np.eye(16)),np.kron(np.eye(16),frame['N'])]
    rho=[2*np.eye(256)-number for number in nums]
    long=p['charge']**2*sum(g['coulomb'][a,b]*rho[a]@rho[b] for a in range(2) for b in range(2))/2
    s2=sum(a@a for a in s)
    local_metric=sum(a@a for ss in site_s for a in ss)
    internal_mass=p['mass']*sum(d[o+a]@d[o+a+2].conj().T+d[o+a+2]@d[o+a].conj().T for o in (0,4) for a in range(2))
    material_base=internal_mass+p['frame_cost']*local_metric+long
    h=kron(csr_matrix(material_base),fi,format='csr')+kron(eye(256),f['kinetic'],format='csr')
    q=np.array(f['q'])
    currents=[csr_matrix(h.shape,dtype=complex) for _ in range(2)]
    contacts=[[csr_matrix(h.shape,dtype=complex) for _ in range(2)] for _ in range(2)]
    edge=[];forwards=[]
    for route_index,(route,coeff) in enumerate(zip(g['routes'],g['line'])):
        if route_index==0:
            # Spatial scalar bridge: it too owns a complete EM link source.
            forward=p['scalar_hop']*sum((1 if a<2 else -1)*d[a]@d[4+a].conj().T for a in range(4))
        else:
            r=np.eye(16,dtype=complex)
            # Genuine ordered frame-coordinate rotations along a curved path.
            for step in route['steps']:
                r=expm(-1j*theta*sum(step[a]*local_s[a] for a in range(3)))@r
            bridge=[np.kron(parity,r@local_d[a].conj().T@r.conj().T-r.conj().T@local_d[a].conj().T@r) for a in range(4)]
            forward=p['hop']*sum((1 if a<2 else -1)*d[a]@bridge[a] for a in range(4))/2
        forwards.append(forward)
        # F creates LEFT and removes RIGHT: its charge path is opposite steps.
        phase=-p['charge']*(coeff@(induction*q+np.asarray(external)[:,None])+np.asarray(spatial_external)@route['steps'].sum(axis=0))
        link=kron(csr_matrix(forward),diags(np.exp(1j*phase)),format='csr')
        h+=link+link.conj().T
        j=1j*(link-link.conj().T)
        k=-(link+link.conj().T)
        edge.append(j)
        for a in range(2):
            currents[a]+=p['charge']*coeff[a]*j
            for b in range(2):
                contacts[a][b]+=p['charge']**2*coeff[a]*coeff[b]*k
    cf,u=canonical_c(local_d)
    cf=np.kron(cf,cf)
    total_number=nums[0]+nums[1]
    return dict(H=h.tocsr(),J=currents,K=contacts,edgeJ=edge,source=[kron(csr_matrix(a),fi,format='csr') for a in d],S=[kron(csr_matrix(a),fi,format='csr') for a in s],S2=kron(csr_matrix(s2),fi,format='csr'),N=kron(csr_matrix(total_number),fi,format='csr'),nums=nums,rho=rho,long=long,frame=frame,matter_spin=s,matter_sources=d,matter_base=material_base,geometry=g,field=f,Q=[kron(eye(256),diags(a),format='csr') for a in f['q']],P=[kron(eye(256),a,format='csr') for a in f['P']],CF=kron(csr_matrix(cf),f['parity'],format='csr'),source_C=np.kron(np.eye(2),u),nf=nf,nmatter=256,forwards=forwards,induction=induction,parameters=p)


def sector(model, number):
    return np.flatnonzero(np.repeat([left.bit_count()+right.bit_count()==number for left in range(16) for right in range(16)],model['nf']))


def evolve(h,psi,time):
    return expm_multiply(-1j*time*h,psi,traceA=-1j*time*h.diagonal().sum())


def expectation(operator,psi):
    return float(np.vdot(psi,operator@psi).real)


def matter_at(model, coordinates):
    p=model['parameters']
    h=model['matter_base'].copy()
    for forward,coeff in zip(model['forwards'],model['geometry']['line']):
        link=np.exp(-1j*p['charge']*coeff@coordinates)*forward
        h+=link+link.conj().T
    return h


def spatial_current(model, momentum=(0.,0.,0.)):
    """Physical -dH/dA(r) Fourier current, all scalar and frame paths included.

    Route density is convolved with the SAME Gaussian charge shape, so spatial
    continuity holds for the actual smooth endpoint densities, not point charges.
    Exact straight-segment Fourier integration enforces smooth charge continuity;
    actual field-profile line coefficients use eight-point segment quadrature.
    """
    k=np.asarray(momentum)
    form=np.exp(-model['parameters']['charge_width']**2*(k@k)/2)
    coeff=[form*np.einsum('n,na->a',np.exp(-1j*route['middle']@k)*np.sinc(route['steps']@k/(2*np.pi)),route['steps']) for route in model['geometry']['routes']]
    return [model['parameters']['charge']*sum(c[a]*j for c,j in zip(coeff,model['edgeJ'])) for a in range(3)]
