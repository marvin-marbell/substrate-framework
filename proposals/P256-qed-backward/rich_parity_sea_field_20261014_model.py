"""Parity-sewn occupied frame lattice and two primitive-electric field profiles.

Only pure raw-frame and full-CAR/momentum helpers are imported. The route
connection is axial r cross dr, not polar ds contracted with axial spin.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.special import erf
from rich_frame_holonomy_20261014_model import raw_frame, canonical_c
from rich_spatial_holonomy_20261014_model import embed, momentum_basis, fourier_source


def permutation(modes, size):
    rows, signs = [], []
    for state in range(size):
        mapped = [modes[a] for a in range(len(modes)) if state & (1 << a)]
        rows.append(sum(1 << a for a in mapped))
        signs.append((-1) ** sum(mapped[a] > mapped[b] for a in range(len(mapped)) for b in range(a + 1, len(mapped))))
    return csr_matrix((signs, (rows, np.arange(size))), shape=(size, size))


def geometry(p, mirror=True):
    t = np.linspace(0, 1, p['path_points'])
    curves = [np.column_stack((p['spacing'] * t, 0 * t, 0 * t))]
    for angle in (0., .83):
        phase = 2 * np.pi * t + angle
        points = np.column_stack((p['spacing'] * t, p['route_bow'] * np.sin(np.pi * t) * np.cos(phase), p['route_bow'] * np.sin(np.pi * t) * np.sin(phase)))
        curves.append(points)
        if mirror:
            curves.append(np.array([p['spacing'], 0., 0.]) - points[::-1])
    widths = np.array(p['profile_widths'])
    routes = []
    for points in curves:
        ds = np.diff(points, axis=0)
        mid = (points[1:] + points[:-1]) / 2
        relative = mid - np.array([p['spacing'] / 2, 0., 0.])
        axial = np.cross(relative, ds) / p['spacing'] ** 2
        profile = np.exp(-.5 * ((mid[:, 1, None] / widths[:, 0]) ** 2 + (mid[:, 2, None] / widths[:, 1]) ** 2))
        routes.append(dict(points=points, axial=axial, line=profile.T @ ds[:, 0]))
    length = p['sites'] * p['spacing']
    gram = np.empty((2, 2))
    for a in range(2):
        for b in range(2):
            gram[a, b] = length * 2 * np.pi / np.sqrt(np.prod(1 / widths[a] ** 2 + 1 / widths[b] ** 2))
    distance = np.array([[min(abs(a-b), p['sites']-abs(a-b))*p['spacing'] for b in range(p['sites'])] for a in range(p['sites'])])
    kernel = np.full_like(distance, 1 / (4 * np.pi ** 1.5 * p['epsilon'] * p['charge_width']))
    mask = distance > 0
    kernel[mask] = erf(distance[mask] / (2 * p['charge_width'])) / (4 * np.pi * p['epsilon'] * distance[mask])
    return dict(routes=routes, gram=gram, coulomb=kernel, circumference=length, widths=widths)


def build(p, external=(0., 0.), twist=0., mirror=True, frame=None):
    f = raw_frame() if frame is None else frame
    sites = p['sites']; matter = 16 ** sites; grid = p['field_grid']; nf = grid ** 2
    ident = eye(matter, format='csr'); fi = eye(nf, format='csr')
    local_parity = diags([(-1) ** a.bit_count() for a in range(16)])
    creators = []
    for site in range(sites):
        string = ident
        for previous in range(site):
            string = string @ embed(local_parity, previous, sites)
        creators.extend(string @ embed(d, site, sites) for d in f['source'])
    nums = [embed(f['N'], site, sites) for site in range(sites)]
    spins = [[embed(a, site, sites) for a in f['S']] for site in range(sites)]
    spin = [sum(ss[a] for ss in spins) for a in range(3)]
    rho = [p['charge'] * (n - 2 * ident) for n in nums]
    g = geometry(p, mirror)
    longitudinal = sum(g['coulomb'][a,b] * (rho[a] @ rho[b]) for a in range(sites) for b in range(sites)) / 2
    positive_frame = p['frame_cost'] * sum(a @ a for ss in spins for a in ss)
    base = positive_frame + longitudinal
    step = 2 * p['field_wall'] / (grid + 1)
    q = np.linspace(-p['field_wall'] + step, p['field_wall'] - step, grid)
    dq = diags(q); pi = -1j * diags([-np.ones(grid-1), np.ones(grid-1)], [-1, 1], format='csr') / (2 * step)
    lap = diags([-np.ones(grid-1), 2*np.ones(grid), -np.ones(grid-1)], [-1, 0, 1], format='csr') / step ** 2
    qf = [kron(dq, eye(grid)), kron(eye(grid), dq)]
    pf = [kron(pi, eye(grid)), kron(eye(grid), pi)]
    mass_metric = p['epsilon'] * g['gram']; inverse = np.linalg.inv(mass_metric)
    electric = (inverse[0,0]*kron(lap,eye(grid)) + inverse[1,1]*kron(eye(grid),lap) + 2*inverse[0,1]*(pf[0] @ pf[1])) / 2
    h = kron(base, fi, format='csr') + kron(ident, electric, format='csr')
    current = [csr_matrix(h.shape, dtype=complex) for _ in range(2)]
    contact = [[csr_matrix(h.shape, dtype=complex) for _ in range(2)] for _ in range(2)]
    velocity = csr_matrix(h.shape, dtype=complex); edge = []; route_flux = []; forwards = []; links = []
    # Channel exchange is the actual internal mirror sewing of the two scalar
    # source families. Its spin dependence is obtained solely from raw S.
    sewing = np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]], complex)
    local_sewing = permutation([2,3,0,1], 16).toarray()
    coordinates = np.column_stack([a.diagonal() for a in qf])
    for site in range(sites):
        target = (site + 1) % sites; fluxes = []
        for index, route in enumerate(g['routes']):
            rotation = np.eye(16, dtype=complex)
            for omega in route['axial']:
                rotation = expm(-1j * p['theta'] * sum(omega[a]*f['S'][a] for a in range(3))) @ rotation
            # This is a route amplitude, not a local fitted carrier mass.
            weight = p['scalar_hop'] if index == 0 else p['hop'] / (len(g['routes']) - 1)
            r = embed(local_sewing @ rotation, target, sites)
            bridge = [r @ creators[4*target+a].getH() @ r.getH() for a in range(4)]
            forward = weight * sum(creators[4*site+a] @ bridge[a] for a in range(4))
            phase = -p['charge'] * ((coordinates + np.array(external)) @ route['line'] + twist / sites)
            link = kron(forward, diags(np.exp(1j*phase)), format='csr')
            j = 1j*(link-link.getH()); k = -(link+link.getH())
            h += link + link.getH(); velocity += p['spacing'] * j
            for a in range(2):
                current[a] += p['charge'] * route['line'][a] * j
                for b in range(2):
                    contact[a][b] += p['charge']**2 * route['line'][a] * route['line'][b] * k
            fluxes.append(j); route_flux.append(j); forwards.append(forward); links.append(link)
        edge.append(sum(fluxes))
    tm = permutation([4*((a//4+1)%sites)+a%4 for a in range(4*sites)], matter)
    pm = permutation([4*((-(a//4))%sites)+(a%4+2)%4 for a in range(4*sites)], matter)
    field_reflection = kron(csr_matrix(np.eye(grid)[::-1]), csr_matrix(np.eye(grid)[::-1]), format='csr')
    cf, source_c = canonical_c(f['source']); cm = embed(cf, 0, sites)
    for site in range(1, sites):
        cm = cm @ embed(cf, site, sites)
    # Antiunitary T is the raw spin intertwiner followed by complex conjugation;
    # it is separately measured, never silently identified with P or C.
    tr = expm(-1j*np.pi*f['S'][1]); mt = embed(tr, 0, sites)
    for site in range(1, sites):
        mt = mt @ embed(tr, site, sites)
    return dict(H=h.tocsr(), J=current, K=contact, V=velocity, edgeJ=edge, routeJ=route_flux, links=links, forwards=forwards,
                source=[kron(d,fi,format='csr') for d in creators], nums=[kron(n,fi,format='csr') for n in nums],
                S=[kron(s,fi,format='csr') for s in spin], S2=kron(sum(s@s for s in spin),fi,format='csr'),
                Q=[kron(ident,a,format='csr') for a in qf], E=[kron(ident,a,format='csr') for a in pf],
                electric=kron(ident,electric,format='csr'), longitudinal=kron(longitudinal,fi,format='csr'),
                positive_frame=kron(positive_frame,fi,format='csr'), P=kron(pm,field_reflection,format='csr'),
                CF=kron(cm,field_reflection,format='csr'), AT=kron(mt,field_reflection,format='csr'),
                source_C=source_c, sewing=sewing, frame=f, T=kron(tm,fi,format='csr'), Tm=tm,
                geometry=g, nf=nf, matter=matter, parameters=p, M=mass_metric)
