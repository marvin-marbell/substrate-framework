"""NEW quantum colour-vector transport in H(0), CAR32, and physical path rotor.
Explicit variational rotated-pair cut; no inherited scalar invariant closure.
Old modules supply pure CAR/Gauss/Haar/source algebra only, never producers.
"""
from __future__ import annotations
from itertools import combinations, product
import numpy as np
from scipy import sparse
from sympy.physics.wigner import gaunt
from rich_common_colour_lepton_20261015_model import (
    mode, string, one_body, gauge_sector, conjugate_sector, conjugation,
    scalar_field, common_root, routes)
from rich_colour_composite_20261014_model import loop_field
from rich_quantum_common_field_20261015_sources import physical_source_matrix


def clean(matrix, tolerance=1e-13):
    result = sparse.csr_matrix(matrix)
    result.data[abs(result.data) < tolerance] = 0
    result.eliminate_zeros()
    return result


def project(sector, matrix):
    return clean(sector['P'].conj().T @ matrix @ sector['P'])


def masks_for(delta, pair_cut):
    # Complete spin multiplets; no fixed spin-axis population cut.
    flavours = [0, 1, 0, 1]; members = [range(3), range(3), [3], [3]]
    neg = [[mode(f, 1, s, c) for s, c in product(range(2), cs)] for f, cs in zip(flavours, members)]
    pos = [[mode(f, 0, s, c) for s, c in product(range(2), cs)] for f, cs in zip(flavours, members)]
    sea = sum(1 << i for channel in neg for i in channel); masks = []
    for holes in product(range(pair_cut + 1), repeat=4):
        if sum(holes) > pair_cut:
            continue
        counts = [h + d for h, d in zip(holes, delta)]
        if any(not (0 <= n <= len(ps)) for n, ps in zip(counts, pos)):
            continue
        choices = [list(combinations(ns, h)) for ns, h in zip(neg, holes)] + [list(combinations(ps, n)) for ps, n in zip(pos, counts)]
        for choice in product(*choices):
            mask = sea
            for indices in choice:
                for i in indices:
                    mask ^= 1 << i
            masks.append(mask)
    return sorted(masks)


def frame():
    sigma = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex), np.diag([1., -1.])]
    # Actual Hadamard of original Xmirror and Ymirror tensor sigma.
    mass = np.kron(sigma[2], np.eye(2)); gamma = [np.kron(-sigma[1], s) for s in sigma]
    spin = [np.kron(np.eye(2), s / 2) for s in sigma]
    return dict(mass=mass, gamma=gamma, spin=spin, sigma=sigma)


def orientation(lmax):
    basis = [(l, m) for l in range(lmax + 1) for m in range(-l, l + 1)]
    size = len(basis); lookup = {x:i for i,x in enumerate(basis)}
    plus = np.zeros((size, size), complex)
    for j, (l, m) in enumerate(basis):
        if m < l:
            plus[lookup[l, m + 1], j] = np.sqrt(l * (l + 1) - m * (m + 1))
    angular = [(plus + plus.conj().T) / 2, (plus - plus.conj().T) / (2j), np.diag([m for l,m in basis])]
    coordinate = []
    harmonics = [{-1:np.sqrt(2*np.pi/3), 1:-np.sqrt(2*np.pi/3)},
                 {-1:1j*np.sqrt(2*np.pi/3), 1:1j*np.sqrt(2*np.pi/3)}, {0:np.sqrt(4*np.pi/3)}]
    for coeff in harmonics:
        matrix = np.zeros((size, size), complex)
        for i,(l,m) in enumerate(basis):
            for j,(k,n) in enumerate(basis):
                matrix[i,j] = sum(value * (-1)**m * complex(gaunt(l, 1, k, -m, q, n)) for q,value in coeff.items())
        coordinate.append(matrix)
    parity = np.diag([(-1)**l for l,m in basis]); time_unitary = np.zeros((size,size))
    for j,(l,m) in enumerate(basis):
        time_unitary[lookup[l,-m],j] = (-1)**m
    return dict(basis=basis, L=[sparse.csr_matrix(x) for x in angular], n=[sparse.csr_matrix(x) for x in coordinate],
        L2=sparse.diags([l*(l+1) for l,m in basis]), parity=sparse.csr_matrix(parity), T=sparse.csr_matrix(time_unitary),
        boundary=sparse.diags([float(l == lmax) for l,m in basis]), dimension=size)


def raw_vector(masks, loop, axis, angle=0., target_masks=None, derivative=0, only_q=None):
    """Gamma_i times Hermitian oriented (e^iqA U-e^-iqA Udagger)/(2i)."""
    target_masks = masks if target_masks is None else target_masks
    result = sparse.csr_matrix((19*len(target_masks),19*len(masks)), dtype=complex)
    gamma = frame()['gamma'][axis]
    for f in range(2):
        for colourless in [False, True]:
            q = ([0., -1.] if colourless else [2/3, -1/3])[f]
            if only_q is not None and q != only_q:
                continue
            phase = np.exp(1j*q*angle)
            left = phase * (1j*q)**derivative / (2j)
            right = -phase.conjugate() * (-1j*q)**derivative / (2j)
            members = [3] if colourless else range(3)
            for a,b in product(members, repeat=2):
                terms = [(mode(f,m,s,a), mode(f,n,t,b), gamma[2*m+s,2*n+t])
                         for m,s,n,t in product(range(2),repeat=4) if gamma[2*m+s,2*n+t] != 0]
                body = one_body(masks, terms, target_masks)
                field = (left*loop['W'] + right*loop['W'].conj().T)/3 if colourless else left*loop['U'][a,b] + right*loop['U'][b,a].conj().T
                result += sparse.kron(body, field, format='csr')
    return result


def matter_operators(sector, loop, pair_baseline):
    masks = sector['masks']; n = len(masks); pauli = frame()['sigma']
    scalar_routes = [(q,project(sector,d),project(sector,f),project(sector,b)) for q,d,f,b in routes(masks,loop)]
    spin = []
    for sigma in pauli:
        terms = [(mode(f,m,s,c),mode(f,m,t,c),sigma[s,t]/2)
            for f,m,s,t,c in product(range(2),range(2),range(2),range(2),range(4)) if sigma[s,t] != 0]
        spin.append(project(sector,sparse.kron(one_body(masks,terms),sparse.eye(19))))
    positive = np.array([sum(bool(mask & (1<<mode(f,0,s,c))) for f,s,c in product(range(2),range(2),range(4))) for mask in masks],float)
    number = np.array([mask.bit_count() for mask in masks],float)
    charge = np.array([sum(([2/3,2/3,2/3,0.] if i//16==0 else [-1/3,-1/3,-1/3,-1.])[i%4]
        for i in range(32) if mask & (1<<i)) for mask in masks])
    parity = np.array([(-1)**sum(bool(mask & (1<<mode(f,1,s,c))) for f,s,c in product(range(2),range(2),range(4))) for mask in masks])
    diagonal = lambda x:project(sector,sparse.kron(sparse.diags(x),sparse.eye(19)))
    loopops = {key:project(sector,sparse.kron(sparse.eye(n),loop[key])) for key in ['C','B','boundary']}
    vectors = [[], [], []]; channels = []; gauss_vector = []
    for axis in range(3):
        channel = []; raw_sum = sparse.csr_matrix((19*n,19*n),dtype=complex)
        for q in [2/3,-1/3,0.,-1.]:
            minus_raw = raw_vector(masks,loop,axis,only_q=q); raw_sum += minus_raw
            minus = project(sector,minus_raw)
            plus = project(sector,raw_vector(masks,loop,axis,derivative=1,only_q=q))/q if q else sparse.csr_matrix(minus.shape,dtype=complex)
            channel.append((q,minus,plus))
        channels.append(channel)
        vectors[0].append(clean(sum(x for q,x,y in channel)))
        vectors[1].append(clean(sum(q*y for q,x,y in channel)))
        vectors[2].append(clean(sum(-q*q*x for q,x,y in channel)))
        image = raw_sum @ sector['P']
        gauss_vector.append(max(float(sparse.linalg.norm(g @ image)) for g in sector['G']))
    return dict(scalar_routes=scalar_routes,spin=spin,S2=clean(sum(s@s for s in spin)),
        positive=diagonal(positive),pairs=diagonal(positive-pair_baseline),N=diagonal(number),Q=diagonal(charge),
        parity=diagonal(parity),vectors=vectors,vector_channels=channels,gauss_vector=gauss_vector,
        matter_octet=project(sector,sparse.kron(sum(q@q for q in sector['Q'])/3,sparse.eye(19))),**loopops)


def scalar_matter(op,p,angle=0.,derivative=0):
    size=op['N'].shape[0]; result=sparse.csr_matrix((size,size),dtype=complex)
    for q,direct,forward,backward in op['scalar_routes']:
        phase=np.exp(1j*q*angle); alpha=p['scalar_coupling']*p['loop_route']/2
        t=alpha*((1j*q)**derivative*phase*forward+(-1j*q)**derivative*phase.conjugate()*backward)
        if derivative==0:t+=p['scalar_coupling']*direct
        result+=t+t.conj().T
    return clean(result)


def embed_matter(op,rotor,indices):
    return clean(sparse.kron(op,sparse.eye(rotor['dimension']),format='csr')[indices][:,indices])


def sector_geometry(sector,op,rotor,twice_m):
    spin_z=op['spin'][2]; labels=np.rint(2*spin_z.diagonal().real).astype(int)
    spin_z_off=spin_z-sparse.diags(spin_z.diagonal())
    indices=np.array([i*rotor['dimension']+j for i,s in enumerate(labels) for j,(l,m) in enumerate(rotor['basis']) if s+2*m==twice_m],int)
    angular=[clean(sparse.kron(s,sparse.eye(rotor['dimension']))+sparse.kron(sparse.eye(sector['dimension']),l)) for s,l in zip(op['spin'],rotor['L'])]
    j2=clean(sum(j@j for j in angular))
    return dict(indices=indices,dimension=len(indices),J2=j2[indices][:,indices],Jfull=angular,
        Sz_offdiagonal=float(sparse.linalg.norm(spin_z_off)),
        L2=clean(sparse.kron(sparse.eye(sector['dimension']),rotor['L2'],format='csr')[indices][:,indices]),
        orientation_boundary=clean(sparse.kron(sparse.eye(sector['dimension']),rotor['boundary'],format='csr')[indices][:,indices]))


def law(sector,op,geo,rotor,scalar,p,angle=0.,vector_scale=1.):
    nf=len(scalar['grid']); indices=geo['indices']; nmo=geo['dimension']; ir=sparse.eye(rotor['dimension']); iff=sparse.eye(nf)
    h= sparse.kron(sparse.eye(nmo),sparse.csr_matrix(scalar['H']),format='csr')
    loopmatter=p['electric']*op['C']/2-p['plaquette']*op['B']
    h+=sparse.kron(embed_matter(loopmatter,rotor,indices)+geo['L2']/(2*p['orientation_inertia']),iff,format='csr')
    derivatives=[]
    for derivative in range(3):
        sm=scalar_matter(op,p,angle,derivative)
        term=sparse.kron(embed_matter(sm,rotor,indices),sparse.csr_matrix(scalar['phi']),format='csr')
        vector=sparse.csr_matrix((sector['dimension']*rotor['dimension'],)*2,dtype=complex)
        if vector_scale:
            for axis in range(3):
                vm=sum(q**derivative*(np.cos(q*angle+derivative*np.pi/2)*minus+np.sin(q*angle+derivative*np.pi/2)*plus)
                       for q,minus,plus in op['vector_channels'][axis])
                vector+=sparse.kron(vm,rotor['n'][axis],format='csr')
            term+=p['vector_velocity']*vector_scale*sparse.kron(vector[indices][:,indices],iff,format='csr')
        derivatives.append(clean(term))
    h+=derivatives[0]
    return clean(h),clean(-derivatives[1]),clean(derivatives[2])


def physical_source(neutral,target,ng,tg,rotor,scalar,kind,spin=.5,removal=False):
    raw=physical_source_matrix(neutral,target,kind,spin,removal)
    matter_orientation=sparse.kron(sparse.csr_matrix(raw),sparse.eye(rotor['dimension']),format='csr')
    reduced=matter_orientation[tg['indices']][:,ng['indices']]
    return clean(sparse.kron(reduced,sparse.eye(len(scalar['grid'])),format='csr'))


def whole_C(source,target,sg,tg,rotor,scalar):
    matter=clean(target['P'].conj().T@conjugation(source['masks'],target['masks'])@source['P'])
    mo=sparse.kron(matter,sparse.eye(rotor['dimension']),format='csr')[tg['indices']][:,sg['indices']]
    return clean(sparse.kron(mo,sparse.eye(len(scalar['grid'])),format='csr'))


def whole_observable(matter,geo,rotor,scalar):
    return clean(sparse.kron(embed_matter(matter,rotor,geo['indices']),sparse.eye(len(scalar['grid'])),format='csr'))


def expanded_state(state,sector,geo,rotor,scalar):
    nf=len(scalar['grid']); full=np.zeros((sector['dimension']*rotor['dimension'],nf),complex)
    full[geo['indices']]=state.reshape(geo['dimension'],nf)
    return full.reshape(sector['dimension'],rotor['dimension'],nf)


def cut_residuals(state,sector,geo,rotor,scalar,p,derivative=0):
    tensor=expanded_state(state,sector,geo,rotor,scalar); nf=len(scalar['grid']); nr=rotor['dimension']
    # All next pair-sector masks generated by the actual vector's mirror and frame terms.
    present=set(sector['masks']); missing=set()
    for mask in sector['masks']:
        for f,m,s,t,a,b in product(range(2),range(2),range(2),range(2),range(4),range(4)):
            if (a==3)!=(b==3):continue
            step=string(mask,[(mode(f,m,s,a),True),(mode(f,1-m,t,b),False)])
            if step is not None and step[0] not in present:missing.add(step[0])
    missing=sorted(missing)
    larger=orientation(p['orientation_lmax']+1); old=rotor['dimension']; missing_n=[n[old:,:old] for n in larger['n']]
    raw=sector['P']@tensor.reshape(sector['dimension'],-1)
    leaked=np.zeros((19*len(missing),nr,nf),complex)
    intersection=np.zeros((19*len(missing),larger['dimension']-old,nf),complex)
    for axis in range(3):
        rv=raw_vector(sector['masks'],p['_loop'],axis,target_masks=missing,derivative=derivative)
        image=(rv@raw).reshape(19*len(missing),nr,nf)
        leaked+=p['vector_velocity']*np.einsum('rt,atf->arf',rotor['n'][axis].toarray(),image)
        intersection+=p['vector_velocity']*np.einsum('rt,atf->arf',missing_n[axis].toarray(),image)
    pair_norm=float(np.linalg.norm(leaked)); joint_norm=float(np.linalg.norm(intersection))
    del leaked,intersection,raw,image,rv
    orient_leak=np.zeros((sector['dimension'],larger['dimension']-old,nf),complex)
    for axis in range(3):
        image=(project(sector,raw_vector(sector['masks'],p['_loop'],axis,derivative=derivative))@tensor.reshape(sector['dimension'],-1)).reshape(sector['dimension'],nr,nf)
        orient_leak+=p['vector_velocity']*np.einsum('rt,atf->arf',missing_n[axis].toarray(),image)
    orient_norm=float(np.linalg.norm(orient_leak))
    result=dict(pair_omitted_raw_masks=len(missing),pair_retained_orientation_H_leak_norm=pair_norm,
        orientation_retained_pair_H_leak_norm=orient_norm,pair_orientation_intersection_H_leak_norm=joint_norm,
        full_pair_H_leak_norm=float(np.hypot(pair_norm,joint_norm)),full_orientation_H_leak_norm=float(np.hypot(orient_norm,joint_norm)),
        combined_pair_orientation_H_leak_norm=float(np.sqrt(pair_norm**2+orient_norm**2+joint_norm**2)),
        operator='H0' if derivative==0 else 'physical current -dH/dA at A0',
        scope='Exact full one-operator-step norm into union of all omitted rotated-pair masks and next orientation irrep, including their intersection, within PW19 and DVR. Orthogonal missing sectors combined coherently inside each sector. Higher PW irreps and outside-DVR field states unmeasured. Not an eigenenergy or resolvent error bound.')
    return result if derivative==0 else {key.replace('H_leak','current_leak'):value for key,value in result.items()}


def rotation_audit(sector,op,rotor,p):
    ir=sparse.eye(rotor['dimension']); im=sparse.eye(sector['dimension'])
    angular=[clean(sparse.kron(s,ir)+sparse.kron(im,l)) for s,l in zip(op['spin'],rotor['L'])]
    vector=sum(sparse.kron(op['vectors'][0][i],rotor['n'][i],format='csr') for i in range(3))
    scalar=sparse.kron(scalar_matter(op,p),ir,format='csr')
    return dict(vector_J_commutators=[float(sparse.linalg.norm(j@vector-vector@j)) for j in angular],
        scalar_J_commutators=[float(sparse.linalg.norm(j@scalar-scalar@j)) for j in angular],
        vector_rotated_population_commutator=float(sparse.linalg.norm(op['positive']@sum(op['vectors'][0])-sum(op['vectors'][0])@op['positive'])),
        vector_operator_norm=float(sparse.linalg.norm(vector)),
        meaning='J=L_n+S_frame is derived from actual Y_lm path-orientation wavefunctions and quaternion frame rotation, not labels. Internal SU3 loop has no spatial orbital coordinate here. No three-quark relative positions or COM Noether claim.')


def build(p):
    loop=loop_field(); scalar=scalar_field(p); rotor=orientation(p['orientation_lmax']); parameters=dict(p,_loop=loop)
    sectors={'neutral':gauge_sector(masks_for((0,0,0,0),p['pair_cut']),loop),
        'baryon':gauge_sector(masks_for((2,1,0,0),p['pair_cut']),loop),
        'lepton':gauge_sector(masks_for((0,0,0,1),p['pair_cut']),loop)}
    for name,anti in [('baryon','antibaryon'),('lepton','antilepton')]:sectors[anti]=conjugate_sector(sectors[name],loop)
    ops={name:matter_operators(s,loop,3 if name=='baryon' else 1 if name=='lepton' else 0) for name,s in sectors.items()}
    geos={name:sector_geometry(s,ops[name],rotor,0 if name=='neutral' else 1) for name,s in sectors.items()}
    sources={}
    for name,kind,removal in [('baryon','baryon',False),('lepton','lepton',False),('antibaryon','baryon',True),('antilepton','lepton',True)]:
        sources[name]={str(spin):physical_source(sectors['neutral'],sectors[name],geos['neutral'],geos[name],rotor,scalar,kind,spin,removal)
            for spin in ([.5,1.5] if kind=='baryon' else [.5])}
    cs={'neutral':whole_C(sectors['neutral'],sectors['neutral'],geos['neutral'],geos['neutral'],rotor,scalar)}
    for name,anti in [('baryon','antibaryon'),('lepton','antilepton')]:cs[name]=whole_C(sectors[name],sectors[anti],geos[name],geos[anti],rotor,scalar)
    root=common_root()
    root['inputs']='Primitive ordered C0=X tensor I2,Ci=Y tensor sigma_i repeated4; fixed Z rank3+1, CAR, flavour/Q and gauging. Actual scalar Phi0=phi C0 and physical path rotor n couple alongside NEW Ci-oriented colour transport. No full fluctuating matrix-Phi or dynamical rank selection.'
    root['field_ansatz']='Quantum radial scalar supplies beta mass; quantum orientation n and quantum PW19 U supply Gamma dot n times (U-Udag)/(2i) transport already at A0. Vector is NOT restricted to the external EM vertex, and no old rotated-population closure is asserted.'
    return dict(parameters=parameters,root=root,scalar=scalar,rotor=rotor,loop=loop,sectors=sectors,operators=ops,geometry=geos,sources=sources,C=cs)
