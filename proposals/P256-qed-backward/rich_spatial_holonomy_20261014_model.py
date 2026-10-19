"""Literal periodic three-site full physical-frame Fock law; no Bloch-cell vacuum."""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.special import erf
from rich_frame_holonomy_20261014_model import raw_frame, canonical_c


def embed(a, site, sites):
    factors = [csr_matrix(a) if j == site else eye(16, format='csr') for j in reversed(range(sites))]
    out = factors[0]
    for factor in factors[1:]:
        out = kron(out, factor, format='csr')
    return out


def geometry(p):
    """Actual ordered curves in universal cover of T1 x R2 and div-free A_x(y,z)."""
    t = np.linspace(0, 1, p['path_points'])
    curves = [np.column_stack((p['spacing']*t, np.zeros_like(t), np.zeros_like(t)))]
    for angle in (0., 2*np.pi/3, 4*np.pi/3):
        curves.append(np.column_stack((p['spacing']*t, p['route_bow']*np.sin(np.pi*t)*np.cos(angle), p['route_bow']*np.sin(np.pi*t)*np.sin(angle))))
    routes = []
    for points in curves:
        steps = np.diff(points, axis=0); mid = (points[1:]+points[:-1])/2
        profile = np.exp(-np.sum(mid[:,1:]**2, axis=1)/(2*p['profile_width']**2))
        line = float(profile @ steps[:,0])
        routes.append({'points':points,'steps':steps,'line':line})
    # Gaussian self and finite minimum-image cross energies. This is a specified
    # finite periodic interaction kernel, not continuum periodic Maxwell proof.
    distance = np.array([[min(abs(a-b), p['sites']-abs(a-b))*p['spacing'] for b in range(p['sites'])] for a in range(p['sites'])])
    kernel = np.zeros_like(distance)
    mask = distance > 0
    kernel[mask] = erf(distance[mask]/(2*p['charge_width']))/(4*np.pi*p['epsilon']*distance[mask])
    kernel[~mask] = 1/(4*np.pi**1.5*p['epsilon']*p['charge_width'])
    return {'routes':routes,'coulomb':kernel,'circumference':p['sites']*p['spacing'],'field':'A=(Q exp(-(y²+z²)/(2w²)),0,0); B=(0,dz A_x,-dy A_x); periodic in x'}


def build(p, external=0., twist=0., hopping_scale=1., frame=None):
    f = raw_frame() if frame is None else frame
    sites = p['sites']; matter = 16**sites; nf = p['field_grid']
    ident = eye(matter, format='csr'); fi = eye(nf, format='csr')
    parity = diags([(-1)**state.bit_count() for state in range(16)])
    creators = []
    for site in range(sites):
        string = ident
        for previous in range(site):
            string = string @ embed(parity, previous, sites)
        creators.extend([string @ embed(d, site, sites) for d in f['source']])
    nums = [embed(f['N'], site, sites) for site in range(sites)]
    spins = [[embed(a, site, sites) for a in f['S']] for site in range(sites)]
    spin = [sum(ss[a] for ss in spins) for a in range(3)]
    local_metric = sum(a@a for ss in spins for a in ss)
    rho = [p['charge']*(number-2*ident) for number in nums]
    g = geometry(p)
    longitudinal = sum(g['coulomb'][a,b]*(rho[a]@rho[b]) for a in range(sites) for b in range(sites))/2
    base = p['frame_cost']*local_metric + longitudinal
    step = 2*p['field_wall']/(nf+1); q = np.linspace(-p['field_wall']+step, p['field_wall']-step, nf)
    kinetic = p['electric_metric']*diags([-np.ones(nf-1),2*np.ones(nf),-np.ones(nf-1)],[-1,0,1],format='csr')/(2*step**2)
    momentum = -1j*diags([-np.ones(nf-1),np.ones(nf-1)],[-1,1],format='csr')/(2*step)
    h = kron(base,fi,format='csr') + kron(ident,kinetic,format='csr')
    current = csr_matrix(h.shape,dtype=complex); contact = current.copy()
    edge = []; links = []; fluxes = []; forwards = []
    for site in range(sites):
        target = (site+1)%sites
        route_flux = []
        for ir, route in enumerate(g['routes']):
            if ir == 0:
                # The target annihilator must have its own actual CAR string.
                bridge = [creators[4*target+a].conj().T for a in range(4)]
                forward = p['scalar_hop']*sum((1 if a<2 else -1)*creators[4*site+a]@bridge[a] for a in range(4))
            else:
                rotation = np.eye(16,dtype=complex)
                for ds in route['steps']:
                    rotation = expm(-1j*p['theta']*sum(ds[a]*f['S'][a] for a in range(3))) @ rotation
                r = embed(rotation,target,sites)
                bridge = [r@creators[4*target+a].conj().T@r.conj().T-r.conj().T@creators[4*target+a].conj().T@r for a in range(4)]
                forward = p['hop']*sum((1 if a<2 else -1)*creators[4*site+a]@bridge[a] for a in range(4))/2
            forward = hopping_scale*forward
            phase = -p['charge']*(route['line']*(q+external)+twist/sites)
            link = kron(forward,diags(np.exp(1j*phase)),format='csr')
            j = 1j*(link-link.conj().T); k = -(link+link.conj().T)
            h += link + link.conj().T
            current += p['charge']*route['line']*j
            contact += p['charge']**2*route['line']**2*k
            route_flux.append(j); links.append(link); forwards.append(forward)
        edge.append(sum(route_flux)); fluxes.extend(route_flux)
    cf, u = canonical_c(f['source']); cm = embed(cf,0,sites)
    for site in range(1,sites):
        cm = cm @ embed(cf,site,sites)
    translation_rows = []; translation_cols = []; signs = []
    for state in range(matter):
        occupied = [m for m in range(4*sites) if state & (1<<m)]
        mapped = [4*((m//4+1)%sites)+m%4 for m in occupied]
        inversions = sum(mapped[a]>mapped[b] for a in range(len(mapped)) for b in range(a+1,len(mapped)))
        translated = sum(1<<m for m in mapped)
        translation_rows.append(translated); translation_cols.append(state); signs.append((-1)**inversions)
    tm = csr_matrix((signs,(translation_rows,translation_cols)),shape=(matter,matter))
    return {'H':h.tocsr(),'J':current.tocsr(),'K':contact.tocsr(),'edgeJ':edge,'routeJ':fluxes,'links':links,'forwards':forwards,'source':[kron(d,fi,format='csr') for d in creators],'nums':[kron(a,fi,format='csr') for a in nums],'S':[kron(a,fi,format='csr') for a in spin],'S2':kron(sum(a@a for a in spin),fi,format='csr'),'Q':kron(ident,diags(q),format='csr'),'P':kron(ident,momentum,format='csr'),'electric':kron(ident,kinetic,format='csr'),'T':kron(tm,fi,format='csr'),'Tm':tm,'CF':kron(cm,csr_matrix(np.eye(nf)[::-1]),format='csr'),'source_C':u,'frame':f,'geometry':g,'q':q,'nf':nf,'matter':matter,'parameters':p,'base':base}


def momentum_basis(model, number, momentum_index):
    """Complete orthonormal orbit projector; T eigenvalue exp(+i 2πm/L)."""
    sites = model['parameters']['sites']; tm = model['Tm']; size = model['matter']
    wave = 2*np.pi*momentum_index/sites; seen = set(); columns = []
    for state in range(size):
        if state.bit_count() != number or state in seen:
            continue
        orbit = []; signs = []; current = state; sign = 1
        while current not in orbit:
            orbit.append(current); signs.append(sign); seen.add(current)
            col = tm.getcol(current); current = int(col.indices[0]) if col.format == 'csc' else int(col.nonzero()[0][0])
            sign *= int(tm[current,orbit[-1]])
        values = np.exp(-1j*wave*np.arange(len(orbit)))*np.array(signs)
        if abs(np.exp(1j*wave*len(orbit))-sign)>1e-9:
            continue
        columns.append((orbit,values/np.sqrt(len(orbit))))
    rows = []; cols = []; data = []
    for j,(orbit,values) in enumerate(columns):
        rows.extend(orbit); cols.extend([j]*len(orbit)); data.extend(values)
    um = csr_matrix((data,(rows,cols)),shape=(size,len(columns)))
    return kron(um,eye(model['nf'],format='csr'),format='csr')


def fourier_source(model, momentum_index):
    sites = model['parameters']['sites']; phase = np.exp(-2j*np.pi*momentum_index*np.arange(sites)/sites)/np.sqrt(sites)
    return [sum(phase[j]*model['source'][4*j+a] for j in range(sites)) for a in range(4)]
