"""Conditional ordered-frame 3D filled-sea CAR x five quantum coordinates.
Fixed orthonormal Slater configurations, not instantaneous ground surfaces.
"""
from __future__ import annotations
from itertools import product
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from rich_common_colour_lepton_20261015_model import common_root


def apply(mask, index, create):
    if bool(mask & (1 << index)) == create:
        return None
    return mask ^ (1 << index), (-1) ** ((mask & ((1 << index) - 1)).bit_count())


def transfer(mask, a, b):
    first = apply(mask, b, False)
    if first is None:
        return None
    second = apply(first[0], a, True)
    return None if second is None else (second[0], first[1] * second[1])


def geometry(config):
    n = 3
    cells = np.array(list(product(range(n), repeat=3)))
    lookup = {tuple(x): i for i, x in enumerate(cells)}
    cliff = common_root()['cliff']
    mass = sparse.kron(sparse.eye(27), cliff[0]).toarray()
    components = [mass]
    labels = ['phi_mass']
    edges = []
    velocity, wilson = config['velocity'], config['wilson']
    for axis in range(3):
        for level in (range(3) if axis < 2 else [0]):
            forward = np.zeros((108, 108), complex)
            for site, x in enumerate(cells):
                if axis < 2 and x[2] != level:
                    continue
                y = x.copy(); y[axis] = (y[axis] + 1) % 3
                target = lookup[tuple(y)]
                block = -0.5j * velocity * cliff[axis + 1] - 0.5 * wilson * cliff[0]
                forward[4*site:4*site+4, 4*target:4*target+4] += block
                edges.append((site, target, axis, level, block))
            components += [forward + forward.conj().T, 1j * (forward - forward.conj().T)]
            labels += [f'hop_{axis}_{level}_cos', f'hop_{axis}_{level}_sin']
    # Wilson on-site term is geometric scalar regulator, not a bare lepton mass.
    onsite = 3 * wilson * mass
    reference = config['coupling'] * config['phi_center'] * mass + onsite
    for k in range(1, len(components), 2):
        reference += components[k]
    values, orbitals = eigh(reference)
    occupied = int(np.count_nonzero(values < 0))
    if occupied != 54:
        raise ValueError(f'Expected 54 occupied modes, found {occupied}')
    transformed = np.array([orbitals.conj().T @ h @ orbitals for h in components])
    onsite_orb = orbitals.conj().T @ onsite @ orbitals
    return dict(cells=cells, cliff=cliff, components=np.array(components), transformed=transformed,
                onsite=onsite_orb, orbitals=orbitals, reference=reference,
                reference_values=values, edges=edges, labels=labels, sea=(1 << 54)-1)


def fields(config):
    sizes = [config['phi_points']] + [config['field_points']] * 4
    widths = [config['phi_width']] + [config['field_width']] * 4
    centers = [config['phi_center']] + [0.] * 4
    axes, momenta, kinetic = [], [], sparse.csr_matrix((int(np.prod(sizes)),)*2, dtype=complex)
    for axis, (size, width, center) in enumerate(zip(sizes, widths, centers)):
        x = center + 2 * width * (np.arange(size) - (size-1)/2) / size
        p = 2*np.pi*np.fft.fftfreq(size, d=2*width/size)
        fourier = np.exp(2j*np.pi*np.outer(np.arange(size), np.arange(size))/size)/np.sqrt(size)
        pmatrix = fourier @ np.diag(p) @ fourier.conj().T
        inertia = config['volume'] * (config['scalar_inertia'] if axis == 0 else 0.5)
        term = sparse.csr_matrix(pmatrix @ pmatrix / (2*inertia))
        full = sparse.csr_matrix([[1.]])
        for j, m in enumerate(sizes):
            full = sparse.kron(full, term if j == axis else sparse.eye(m), format='csr')
        kinetic += full
        axes.append(x); momenta.append(pmatrix)
    coordinates = np.array(list(product(*axes)))
    phi = coordinates[:, 0]
    potential = config['volume'] * config['scalar_stiffness'] * (phi-config['phi_center'])**2 / 2
    boson = kinetic + sparse.diags(potential)
    coefficients = [config['coupling']*phi]
    derivative, contact = [], []
    q = 2*np.pi/3
    for axis in range(3):
        for level in (range(3) if axis < 2 else [0]):
            phase = config['charge'] * (coordinates[:,1+2*axis]*np.cos(q*level) + coordinates[:,2+2*axis]*np.sin(q*level)) if axis < 2 else np.zeros(len(phi))
            coefficients += [np.cos(phase), np.sin(phase)]
    coefficients = np.array(coefficients)
    for a in range(4):
        d = np.zeros_like(coefficients); dd = np.zeros_like(coefficients)
        component = 1
        for axis in range(3):
            for level in (range(3) if axis < 2 else [0]):
                profile = config['charge'] * (np.cos(q*level) if a % 2 == 0 else np.sin(q*level)) if axis == a//2 else 0.
                d[component] = -profile*coefficients[component+1]
                d[component+1] = profile*coefficients[component]
                dd[component] = -profile**2*coefficients[component]
                dd[component+1] = -profile**2*coefficients[component+1]
                component += 2
        derivative.append(d); contact.append(dd)
    return dict(coordinates=coordinates, sizes=sizes, kinetic=kinetic, boson=boson,
                coefficients=coefficients, derivative=np.array(derivative), contact=np.array(contact), momenta=momenta)


def configurations(geo, field, config, rank):
    sea = geo['sea']
    # Diabatic nonadiabatic/current-connected determinant selection from the actual H.
    pairweight = np.zeros((54,54))
    for coefficient, h in zip(field['coefficients'], geo['transformed']):
        pairweight += np.mean(abs(coefficient)**2)*abs(h[54:,:54])**2
    for derivative in field['derivative']:
        for coefficient, h in zip(derivative, geo['transformed']):
            pairweight += np.mean(abs(coefficient)**2)*abs(h[54:,:54])**2
    order = np.argsort(pairweight.ravel())[::-1][:rank]
    pairs = [(54+int(i//54), int(i%54)) for i in order]
    neutral = [sea] + [sea ^ (1 << p) ^ (1 << h) for p,h in pairs]
    # Include doubly excited determinants: a genuine multi-pair direction, not CIS alone.
    for (p,h),(r,s) in zip(pairs[::2], pairs[1::2]):
        if p != r and h != s:
            neutral.append(sea ^ (1<<p) ^ (1<<h) ^ (1<<r) ^ (1<<s))
    positive = [sea | (1<<p) for p in range(54,108)]
    negative = [sea ^ (1<<h) for h in range(54)]
    packet = source_packet(geo, config)
    plus_orbitals = np.argsort(abs(packet[54:]))[::-1][:config['source_extensions']] + 54
    minus_orbitals = np.argsort(abs(packet[:54]))[::-1][:config['source_extensions']]
    for mask in neutral[1:]:
        for p in plus_orbitals:
            step = apply(mask, int(p), True)
            if step is not None: positive.append(step[0])
        for h in minus_orbitals:
            step = apply(mask, int(h), False)
            if step is not None: negative.append(step[0])
    return [sorted(set(x)) for x in [neutral, positive, negative]], pairs, pairweight


def source_packet(geo, config):
    x = geo['cells'].astype(float)
    dx = (x-np.array(config['packet_center'])+1.5)%3-1.5
    f = np.zeros((27,4), complex)
    f[:,0] = np.exp(-np.sum(dx*dx,axis=1)/(2*config['packet_width']**2) + 1j*x@np.array(config['packet_momentum']))
    f[:,2] = 0.3j*f[:,0]
    f = f.ravel(); f /= np.linalg.norm(f)
    return geo['orbitals'].conj().T @ f


def one_body(masks, matrices):
    matrices = np.asarray(matrices)
    if matrices.ndim == 2: matrices = matrices[None]
    rows, cols, values = [], [], []
    for j, ket in enumerate(masks):
        occ = [i for i in range(108) if ket & (1<<i)]
        for i, bra in enumerate(masks):
            difference = bra ^ ket
            if difference == 0:
                value = np.sum(matrices[:,occ,occ], axis=1)
            elif difference.bit_count() == 2:
                a = (bra & difference).bit_length()-1
                b = (ket & difference).bit_length()-1
                step = transfer(ket,a,b)
                value = matrices[:,a,b]*step[1]
            else:
                continue
            if np.max(abs(value)) > 1e-13:
                rows.append(i); cols.append(j); values.append(value)
    data = np.array(values)
    return [sparse.csr_matrix((data[:,k], (rows,cols)), shape=(len(masks),)*2) for k in range(len(matrices))]


def sector(masks, geo, field, config):
    operators = one_body(masks, geo['transformed'])
    onsite = one_body(masks, geo['onsite'])[0]
    nf, nb = len(masks), len(field['coordinates'])
    h = sparse.kron(sparse.eye(nf), field['boson'], format='csr') + sparse.kron(onsite, sparse.eye(nb), format='csr')
    for op, coefficient in zip(operators, field['coefficients']):
        h += sparse.kron(op, sparse.diags(coefficient), format='csr')
    # Static ionic compensation fixes only the uniform charge mode. Nonuniform
    # longitudinal Coulomb coupling is deliberately zero in this selected law.
    neutral_offset = np.sum(geo['reference_values'][:54])
    h -= neutral_offset*sparse.eye(nf*nb,format='csr')
    energy, vector = eigsh(h,k=1,which='SA',tol=config['eigensolver_tolerance'],maxiter=5000,
                           v0=np.ones(nf*nb,dtype=complex)/np.sqrt(nf*nb))
    return dict(masks=masks, operators=operators, h=h, energy=float(energy[0]), ground=vector[:,0],
                gram=np.eye(nf), gram_eigenvalues=np.ones(nf), whitener=np.eye(nf))


def field_operator(sector, coefficients):
    nb = coefficients.shape[-1]
    result = sparse.csr_matrix(sector['h'].shape, dtype=complex)
    for op, coefficient in zip(sector['operators'], coefficients):
        result += sparse.kron(op, sparse.diags(coefficient),format='csr')
    return result


def source_operator(source, target, packet, create):
    lookup = {m:i for i,m in enumerate(target['masks'])}
    rows,cols,data = [],[],[]
    for j,mask in enumerate(source['masks']):
        for orbital,value in enumerate(packet if create else packet.conj()):
            step = apply(mask,orbital,create)
            if step is not None and step[0] in lookup:
                rows.append(lookup[step[0]]);cols.append(j);data.append(value*step[1])
    return sparse.csr_matrix((data,(rows,cols)),shape=(len(target['masks']),len(source['masks'])))


def leakage(sector, geo, field, state):
    """Exact omitted-determinant H action for this state; field grid stays fixed."""
    masks=sector['masks']; retained=set(masks); outside={}; rows=[];cols=[];values=[]
    matrices=np.concatenate([geo['onsite'][None],geo['transformed']])
    nonzero=np.max(abs(matrices),axis=0)>1e-11
    for j,mask in enumerate(masks):
        for b in range(108):
            if not mask & (1<<b):continue
            for a in np.flatnonzero(nonzero[:,b]):
                a=int(a)
                if mask & (1<<a):continue
                step=transfer(mask,a,b)
                if step[0] in retained:continue
                row=outside.setdefault(step[0],len(outside))
                rows.append(row);cols.append(j);values.append(step[1]*matrices[:,a,b])
    coefficients=np.vstack([np.ones(len(field['coordinates'])),field['coefficients']])
    output=np.zeros((len(outside),len(field['coordinates'])),complex)
    shaped=state.reshape(len(masks),-1)
    data=np.array(values)
    if len(data):
        for k in range(len(matrices)):
            r=sparse.csr_matrix((data[:,k],(rows,cols)),shape=(len(outside),len(masks)))
            output += (r@shaped)*coefficients[k]
    return dict(omitted_determinants=len(outside),absolute_H_projection_residual=float(np.linalg.norm(output)),
                residual_units='energy, norm((1-P)H psi); NOT a gap bound',relative_to_full_H_action=float(np.linalg.norm(output)/np.sqrt(np.linalg.norm(sector['h']@state)**2+np.linalg.norm(output)**2)))
