"""NEW particle-only SU(3) triangle candidate; no old producer imports.
The repeated quaternion module and the gauge Hamiltonian are explicit new laws.
Tree electric fields are solved from Gauss, not replaced by a pair attraction.
"""
from __future__ import annotations

from itertools import product
import numpy as np
from scipy.linalg import eigh, null_space


def epsilon(i, j, k):
    if len({i, j, k}) != 3:
        return 0
    return 1 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1


def generators():
    out = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        x = np.zeros((3, 3), complex)
        x[a, b] = x[b, a] = .5
        out.append(x)
        y = np.zeros((3, 3), complex)
        y[a, b], y[b, a] = -.5j, .5j
        out.append(y)
    out += [np.diag([.5, -.5, 0]), np.diag([1, 1, -2]) / np.sqrt(12)]
    return out


def material_root():
    # Primitive complex quaternion left module, NOT a free Dirac kinetic law.
    field = [np.array([[0, 1], [1, 0]], complex),
             np.array([[0, -1j], [1j, 0]], complex),
             np.diag([1., -1.])]
    repeated = [np.kron(f, np.eye(3)) for f in field]
    equation = np.vstack([np.kron(np.eye(6), f) - np.kron(f.T, np.eye(6)) for f in repeated])
    commutant = null_space(equation)
    intertwiners = []
    for a, b in product(range(3), repeat=2):
        e = np.zeros((3, 3)); e[a, b] = 1
        intertwiners.append(np.kron(np.eye(2), e))
    colour = [np.kron(np.eye(2), t) for t in generators()]
    return dict(field=field, repeated=repeated, spin=[f / 2 for f in repeated],
                commutant=commutant, intertwiners=intertwiners, colour=colour)


def loop_field():
    """Exact Haar Galerkin multiplication on 1 + (3 x 3*) + (3* x 3)."""
    ts = generators()
    left, right = [], []
    for t in ts:
        l = np.zeros((19, 19), complex); r = l.copy()
        l[1:10, 1:10] = np.kron(-t.T, np.eye(3))
        r[1:10, 1:10] = np.kron(np.eye(3), t)
        l[10:, 10:] = np.kron(t, np.eye(3))
        r[10:, 10:] = np.kron(np.eye(3), -t.T)
        left.append(l); right.append(r)
    u = np.zeros((3, 3, 19, 19), complex)
    for a, b in product(range(3), repeat=2):
        u[a, b, 1 + 3*a + b, 0] = 1 / np.sqrt(3)
        u[a, b, 0, 10 + 3*a + b] = 1 / np.sqrt(3)
        for k, l, i, j in product(range(3), repeat=4):
            u[a, b, 10 + 3*k + l, 1 + 3*i + j] = .5 * epsilon(k, a, i) * epsilon(l, b, j)
    casimir = sum(r @ r for r in right)
    wilson = sum(u[a, a] for a in range(3))
    boundary = np.diag([0.] + [1.] * 18)
    return dict(L=left, R=right, U=u, C=casimir, W=wilson,
                B=(wilson + wilson.conj().T)/2, boundary=boundary)


def tensor_on(matrix, particle, count=3):
    result = np.ones((1, 1), complex)
    for index in range(count):
        result = np.kron(result, matrix if index == particle else np.eye(matrix.shape[0]))
    return result


def colour_space(loop):
    ts = generators()
    charges = [[tensor_on(t, particle) for t in ts] for particle in range(3)]
    total = [sum(charges[i][a] for i in range(3)) for a in range(8)]
    gauss = [np.kron(total[a], np.eye(19)) + np.kron(np.eye(27), loop['L'][a] + loop['R'][a]) for a in range(8)]
    values, vectors = eigh(sum(g @ g for g in gauss))
    p = vectors[:, values < 1e-8]
    swaps = np.zeros((27, 27))
    for c in product(range(3), repeat=3):
        swaps[9*c[1]+3*c[0]+c[2], 9*c[0]+3*c[1]+c[2]] = 1
    permutation = p.conj().T @ np.kron(swaps, np.eye(19)) @ p
    transfers = []
    for particle in range(3):
        raw = np.zeros((513, 513), complex)
        for a, b in product(range(3), repeat=2):
            e = np.zeros((3, 3)); e[a, b] = 1
            raw += np.kron(tensor_on(e, particle), loop['U'][a, b])
        transfers.append(p.conj().T @ raw @ p)
    return dict(P=p, q=charges, G=gauss, permutation=permutation,
                transfer=transfers, gauss_values=values)


def build(p):
    loop = loop_field(); colour = colour_space(loop); cp = colour['P']; nc = cp.shape[1]
    positions = list(product(range(3), repeat=3)); lookup = {v:i for i,v in enumerate(positions)}
    spinbasis = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # 1 means down: total M=+1/2, not fixed S.
    nbase = len(positions)*nc; nraw = nbase*3
    electric = np.zeros((nbase, nbase), complex)
    magnetic = np.kron(np.eye(27), -p['magnetic'] * (cp.conj().T @ np.kron(np.eye(27), loop['B']) @ cp))
    edgepairs = [(0, 1), (1, 2), (2, 0)]  # source -> destination; chord uses U.
    transfers = [[np.zeros((nbase, nbase), complex) for _ in range(3)] for _ in range(3)]
    gauss_error = 0.
    for ix, pos in enumerate(positions):
        qsite = [[sum((colour['q'][i][a] for i in range(3) if pos[i] == site), np.zeros((27, 27), complex)) for a in range(8)] for site in range(3)]
        e01 = [np.kron(qsite[1][a]+qsite[2][a], np.eye(19)) + np.kron(np.eye(27), loop['R'][a]) for a in range(8)]
        e12 = [np.kron(qsite[2][a], np.eye(19)) + np.kron(np.eye(27), loop['R'][a]) for a in range(8)]
        block = p['g2']/2 * sum(p['lengths'][0]*((a@cp).conj().T@(a@cp)) + p['lengths'][1]*((b@cp).conj().T@(b@cp)) + p['lengths'][2]*(cp.conj().T@np.kron(np.eye(27), r@r)@cp) for a,b,r in zip(e01,e12,loop['R']))
        electric[ix*nc:(ix+1)*nc, ix*nc:(ix+1)*nc] = block
        # After exact tree elimination G1=G2=0 as operator identities;
        # G0 is the remaining full non-Abelian total Gauss constraint.
        for a in range(8):
            g0 = np.kron(qsite[0][a], np.eye(19))+e01[a]+np.kron(np.eye(27),loop['L'][a])
            gauss_error = max(gauss_error, float(np.linalg.norm(g0@cp)))
        for particle in range(3):
            for edge, (source, destination) in enumerate(edgepairs):
                if pos[particle] == source:
                    new = list(pos); new[particle] = destination; iy = lookup[tuple(new)]
                    block = colour['transfer'][particle] if edge == 2 else np.eye(nc)
                    transfers[particle][edge][iy*nc:(iy+1)*nc, ix*nc:(ix+1)*nc] += block
    swap = np.zeros((nraw, nraw), complex)
    for ix,pos in enumerate(positions):
        iy = lookup[(pos[1],pos[0],pos[2])]
        for s,spin in enumerate(spinbasis):
            ss = spinbasis.index((spin[1],spin[0],spin[2]))
            swap[(iy*nc)*3+ss:((iy+1)*nc)*3:3, (ix*nc)*3+s:((ix+1)*nc)*3:3] = colour['permutation']
    sv, su = eigh(swap)
    pauli = su[:,sv < -.5]
    def project(matrix):
        return pauli.conj().T @ np.kron(matrix,np.eye(3)) @ pauli
    t = [[project(x) for x in particle] for particle in transfers]
    he = project(electric); hb = project(magnetic)
    s2small = np.full((3,3), 1., complex) + .75*np.eye(3)
    spin2 = pauli.conj().T @ np.kron(np.eye(nbase),s2small) @ pauli
    occupancy = [project(np.diag(np.repeat([sum(x==site for x in pos) for pos in positions],nc))) for site in range(3)]
    charge = [project(np.diag(np.repeat([sum(p['charges'][i] for i in range(3) if pos[i]==site) for pos in positions],nc))) for site in range(3)]
    distributions = [project(np.diag(np.repeat([float(pos==target) for pos in positions],nc))) for target in positions]
    r2values = [sum(float(pos[i]!=pos[j])*p['side_squared'][min(pos[i],pos[j])][max(pos[i],pos[j])] for i in range(3) for j in range(i))/3 for pos in positions]
    r2 = project(np.diag(np.repeat(r2values,nc)))
    co = project(np.diag(np.repeat([float(len(set(pos))==1) for pos in positions],nc)))
    loopobs = {key:project(np.kron(np.eye(27), cp.conj().T@np.kron(np.eye(27),matrix)@cp)) for key,matrix in [('C',loop['C']),('B',loop['B']),('boundary',loop['boundary'])]}
    return dict(parameters=p, loop=loop, colour=colour, nc=nc, positions=positions,
                spinbasis=spinbasis, Pauli=pauli, HE=he, HB=hb, T=t, S2=spin2,
                n=occupancy, rho=charge, distributions=distributions, r2=r2, co=co,
                loopobs=loopobs, local_gauss_error=gauss_error, raw_dim=nraw)


def hamiltonian(model, angles=(0.,0.,0.)):
    p=model['parameters']; h=model['HE']+model['HB']; j=[]; contact=[]
    for edge in range(3):
        forward=[np.exp(1j*p['charges'][i]*angles[edge])*model['T'][i][edge] for i in range(3)]
        h=h-p['hop']*sum(x+x.conj().T for x in forward)
        j.append(1j*p['hop']*sum(p['charges'][i]*(x-x.conj().T) for i,x in enumerate(forward)))
        contact.append(p['hop']*sum(p['charges'][i]**2*(x+x.conj().T) for i,x in enumerate(forward)))
    return h,j,contact


def baryon_source(model, site, vacuum, spin=0.5):
    epsilon_state=np.array([epsilon(*c) for c in product(range(3),repeat=3)],complex)/np.sqrt(6)
    c=model['colour']['P'].conj().T@np.kron(epsilon_state,vacuum)
    spinstate=np.array([-1/np.sqrt(6),-1/np.sqrt(6),np.sqrt(2/3)]) if spin==.5 else np.ones(3)/np.sqrt(3)
    raw=np.zeros(model['raw_dim'],complex)
    ix=model['positions'].index((site,site,site));nc=model['nc']
    raw[ix*nc*3:(ix+1)*nc*3]=np.kron(c,spinstate)
    return model['Pauli'].conj().T@raw
