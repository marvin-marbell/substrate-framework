"""Internal mirror grading retains ordered-frame vectors on one occupied Fock.

Only raw Gauss/frame, fermionic permutation and spatial-source pure helpers are
reused. No imported gamma, independently fitted gap or oscillator stiffness.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.special import erf
from rich_frame_holonomy_20261014_model import raw_frame, canonical_c
from rich_spatial_holonomy_20261014_model import embed, momentum_basis, fourier_source
from rich_parity_sea_field_20261014_model import permutation


def field_values(points, p):
    """Two exactly orthogonal transverse directions at common q on finite T3."""
    x = points[:,0]; k = 2*np.pi/p['spacing']; profile = np.cos(k*x)
    zeros = np.zeros_like(profile)
    return np.stack((np.column_stack((zeros,zeros,-profile)), np.column_stack((zeros,profile,zeros))))


def geometry(p):
    t = np.linspace(0., 1., p['path_points']); a = p['spacing']
    routes = []
    for angle in (0., 1.07):
        phase = 2*np.pi*t+angle
        points = np.column_stack((a*t, p['route_bow']*np.sin(np.pi*t)*np.cos(phase), p['route_bow']*np.sin(np.pi*t)*np.sin(phase)))
        for channel, curve in enumerate((points, np.array([a,0.,0.])-points[::-1])):
            ds = np.diff(curve, axis=0); mid = (curve[1:]+curve[:-1])/2
            relative = mid-np.array([a/2,0.,0.])
            routes.append(dict(points=curve, axial=np.cross(relative,ds)/a**2,
                               line=np.einsum('asi,si->a',field_values(mid,p),ds), channel=channel))
    length = p['sites']*a; k=2*np.pi/a
    gram = np.eye(2)*length*p['transverse_period']**2/2
    distance=np.array([[min(abs(i-j),p['sites']-abs(i-j))*a for j in range(p['sites'])] for i in range(p['sites'])])
    kernel=np.full_like(distance,1/(4*np.pi**1.5*p['epsilon']*p['charge_width']))
    mask=distance>0
    kernel[mask]=erf(distance[mask]/(2*p['charge_width']))/(4*np.pi*p['epsilon']*distance[mask])
    return dict(routes=routes,gram=gram,coulomb=kernel,common_q=[k,0.,0.],central_polarizations=[[0.,0.,-1.],[0.,1.,0.]],transverse_period=p['transverse_period'])


def ordered_transfers(f, p, g):
    """Opposite mirror channels retain D_r and D_m=-D_r, not their sum."""
    records=[]
    for route in g['routes']:
        rotation=np.eye(16,dtype=complex)
        for omega in route['axial']:
            rotation=expm(-1j*p['theta']*sum(omega[a]*f['S'][a] for a in range(3)))@rotation
        bridge=[rotation@d.conj().T@rotation.conj().T-rotation.conj().T@d.conj().T@rotation for d in f['source']]
        matrix=np.zeros((4,4),complex)
        for a in range(4):
            for b in range(4):
                matrix[a,b]=np.trace(f['source'][b]@bridge[a])/8
        records.append(dict(rotation=rotation,bridge=bridge,matrix=matrix,channel=route['channel']))
    return records


def build(p, external=(0.,0.), twist=0., vector_scale=1., frame=None):
    f=raw_frame() if frame is None else frame; sites=p['sites']; matter=16**sites
    grid=p['field_grid']; nf=grid**2; mi=eye(matter,format='csr'); fi=eye(nf,format='csr')
    local_parity=diags([(-1)**state.bit_count() for state in range(16)])
    creators=[]
    for site in range(sites):
        string=mi
        for previous in range(site):string=string@embed(local_parity,previous,sites)
        creators.extend(string@embed(d,site,sites) for d in f['source'])
    nums=[embed(f['N'],site,sites) for site in range(sites)]
    spins=[[embed(s,site,sites) for s in f['S']] for site in range(sites)]
    spin=[sum(ss[a] for ss in spins) for a in range(3)]
    rho=[p['charge']*(n-2*mi) for n in nums]; g=geometry(p)
    longitudinal=sum(g['coulomb'][a,b]*(rho[a]@rho[b]) for a in range(sites) for b in range(sites))/2
    frame_energy=p['frame_cost']*sum(s@s for ss in spins for s in ss)
    step=2*p['field_wall']/(grid+1); coordinates=np.linspace(-p['field_wall']+step,p['field_wall']-step,grid)
    dq=diags(coordinates); pi=-1j*diags([-np.ones(grid-1),np.ones(grid-1)],[-1,1],format='csr')/(2*step)
    lap=diags([-np.ones(grid-1),2*np.ones(grid),-np.ones(grid-1)],[-1,0,1],format='csr')/step**2
    qf=[kron(dq,eye(grid)),kron(eye(grid),dq)]; pf=[kron(pi,eye(grid)),kron(eye(grid),pi)]
    metric=p['epsilon']*g['gram']; inverse=np.linalg.inv(metric)
    electric=(inverse[0,0]*kron(lap,eye(grid))+inverse[1,1]*kron(eye(grid),lap)+2*inverse[0,1]*(pf[0]@pf[1]))/2
    h=kron(frame_energy+longitudinal,fi,format='csr')+kron(mi,electric,format='csr')
    current=[csr_matrix(h.shape,dtype=complex) for _ in range(2)]
    contact=[[csr_matrix(h.shape,dtype=complex) for _ in range(2)] for _ in range(2)]
    velocity=csr_matrix(h.shape,dtype=complex); edge=[]; links=[]; forwards=[]
    transfers=ordered_transfers(f,p,g); qcols=np.column_stack([q.diagonal() for q in qf])
    local_sewing=permutation([2,3,0,1],16).toarray()
    for site in range(sites):
        target=(site+1)%sites; flux=[]
        sewing=embed(local_sewing,target,sites)
        scalar=p['scalar_hop']*sum(creators[4*site+a]@(sewing@creators[4*target+a].getH()@sewing.getH()) for a in range(4))
        channels=[(scalar,np.zeros(2))]
        for route,record in zip(g['routes'],transfers):
            r=embed(record['rotation'],target,sites)
            selected=range(2*route['channel'],2*route['channel']+2)
            vector=vector_scale*p['hop']/2*sum(creators[4*site+a]@(r@creators[4*target+a].getH()@r.getH()-r.getH()@creators[4*target+a].getH()@r) for a in selected)
            channels.append((vector,route['line']))
        for forward,line in channels:
            phase=-p['charge']*((qcols+np.asarray(external))@line+twist/sites)
            link=kron(forward,diags(np.exp(1j*phase)),format='csr')
            j=1j*(link-link.getH()); kk=-(link+link.getH())
            h+=link+link.getH(); velocity+=p['spacing']*j; flux.append(j)
            for a in range(2):
                current[a]+=p['charge']*line[a]*j
                for b in range(2):contact[a][b]+=p['charge']**2*line[a]*line[b]*kk
            links.append(link);forwards.append(forward)
        edge.append(sum(flux))
    tm=permutation([4*((a//4+1)%sites)+a%4 for a in range(4*sites)],matter)
    pm=permutation([4*((-(a//4))%sites)+(a%4+2)%4 for a in range(4*sites)],matter)
    reflect=kron(csr_matrix(np.eye(grid)[::-1]),csr_matrix(np.eye(grid)[::-1]))
    cf,source_c=canonical_c(f['source']); cm=embed(cf,0,sites)
    for site in range(1,sites):cm=cm@embed(cf,site,sites)
    raw_tr=expm(-1j*np.pi*f['S'][1]);mt=embed(raw_tr,0,sites)
    for site in range(1,sites):mt=mt@embed(raw_tr,site,sites)
    return dict(H=h.tocsr(),J=current,K=contact,V=velocity,edgeJ=edge,links=links,forwards=forwards,
                source=[kron(d,fi,format='csr') for d in creators],nums=[kron(n,fi,format='csr') for n in nums],
                S=[kron(s,fi,format='csr') for s in spin],S2=kron(sum(s@s for s in spin),fi,format='csr'),
                Q=[kron(mi,q,format='csr') for q in qf],E=[kron(mi,a,format='csr') for a in pf],
                electric=kron(mi,electric,format='csr'),P=kron(pm,reflect,format='csr'),CF=kron(cm,reflect,format='csr'),
                T=kron(tm,fi,format='csr'),Tm=tm,AT=kron(mt,reflect,format='csr'),source_C=source_c,nf=nf,matter=matter,M=metric,
                frame=f,geometry=g,transfers=transfers,parameters=p)
