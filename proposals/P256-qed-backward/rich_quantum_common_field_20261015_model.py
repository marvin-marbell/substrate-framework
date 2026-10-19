"""Two compact Wilson coordinates coupled nonlinearly to one occupied CAR32 law.
Depth-two population / low-energy Galerkin approximation; no inherited linear closure.
"""
from __future__ import annotations
from itertools import product
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from rich_common_vector_20261015_model import (
    signatures, complete_masks, gamma_matrices, vector_terms, mode, string,
    one_body, gauge_sector, operators, law, build, conjugate_sector, conjugation)
from rich_common_colour_lepton_20261015_model import source_matrix, charge_conjugate


def neighbours(signature):
    reached = set()
    for f, c, m, s, n, t in product(range(2), range(2), range(2), range(2), range(2), range(2)):
        # c=0 is the complete colour multiplet, c=1 the colourless member.
        if f == 0 and c == 1:
            continue  # chosen neutrino has Q=0
        for axis in range(3):
            if gamma_matrices()[axis][2*m+s, 2*n+t] == 0:
                continue
            a = ((f*2+m)*2+s)*2+c; b = ((f*2+n)*2+t)*2+c
            capacity = 3 if c == 0 else 1
            if signature[b] and signature[a] < capacity:
                target = list(signature); target[a] += 1; target[b] -= 1
                reached.add(tuple(target))
    return reached


def signature_layers(seeds, depth):
    seen = set(seeds); frontier = seen.copy(); layers = [len(seen)]
    for _ in range(depth):
        frontier = set().union(*(neighbours(s) for s in frontier)) - seen
        seen |= frontier; layers.append(len(frontier))
    omitted = set().union(*(neighbours(s) for s in seen)) - seen
    return sorted(seen), sorted(omitted), layers


def rotor(p):
    cutoff = p['electric_cut']; electric = np.arange(-cutoff, cutoff+1)
    size = len(electric); eye = sparse.eye(size, format='csr'); ident = sparse.eye(size*size, format='csr')
    # Three disjoint cycle tubes with line angles (3theta_x,3theta_y,3(theta_x+theta_y)).
    metric = 9*p['permittivity']*p['tube_area']/p['cycle_circumference']*np.array([[2.,1.],[1.,2.]])
    n = sparse.diags(electric, format='csr')
    numbers = [sparse.kron(n,eye,format='csr'), sparse.kron(eye,n,format='csr')]
    inverse = np.linalg.inv(metric)
    he = sum(inverse[a,b]*(numbers[a]@numbers[b])/2 for a,b in product(range(2),repeat=2))
    shifts = {}
    for q in [2,-1,-3,0]:
        shift = sparse.diags(np.ones(size-abs(q)), -q, shape=(size,size),format='csr') if q else eye
        shifts[q] = [sparse.kron(shift,eye,format='csr'),sparse.kron(eye,shift,format='csr'),sparse.kron(shift,shift,format='csr')]
    edge = sparse.diags((abs(electric) > cutoff-3).astype(float))
    boundary = ident-sparse.kron(eye-edge,eye-edge,format='csr')
    return dict(size=size*size, electric=electric, shifts=shifts, E=numbers, H=he, metric=metric,
                boundary=boundary, identity=ident)


def expand_sector(base, loop, scalar, p, anti=False):
    seed = sorted({signatures(mask) for mask in base['masks']})
    sigs, omitted, layers = signature_layers(seed,p['population_depth'])
    blocks=[]; offset=0
    for signature in sigs:
        sector=gauge_sector(complete_masks(signature),loop)
        if not sector['dimension']:
            continue
        op=operators(sector,loop); h,_,_=law(op,scalar,p)
        e,v=eigh(h,subset_by_index=[0,min(p['states_per_signature']+4,len(h))-1]); threshold=e[min(p['states_per_signature'],len(e))-1]+1e-9
        keep=int(np.count_nonzero(e<=threshold)); v=v[:,:keep]
        if anti:
            sector=conjugate_sector(sector,loop)
            # Its projected CAR/loop coordinates are already C aligned with the positive sector.
            op=operators(sector,loop); h,_,_=law(op,scalar,p)
        blocks.append(dict(signature=signature,sector=sector,op=op,H0=h,V=v,e=e[:keep],offset=offset,dimension=keep))
        offset+=keep
    return dict(blocks=blocks,dimension=offset,omitted=omitted,layers=layers,seeds=seed,anti=anti)


def compressed(sector, name, scalar):
    nf=len(scalar['grid']); output=[]
    for block in sector['blocks']:
        n=block['sector']['dimension']; v=block['V']
        if name in ['phi','phi2','scalar_boundary']:
            field={'phi':scalar['phi'],'phi2':scalar['phi']@scalar['phi'],'scalar_boundary':scalar['boundary']}[name]
            op=np.kron(np.eye(n),field)
        else:
            op=np.kron(block['op'][name],np.eye(nf))
        output.append(sparse.csr_matrix(v.conj().T@op@v))
    return sparse.block_diag(output,format='csr')


def vector_flavour_terms(axis, harmonic):
    # Remove the q factor from the derivative helper: H=v sin(qunit*3theta) Gamma.
    return [(a,b,value/(harmonic/3)) for a,b,value in vector_terms(axis)
            if round(3*([2/3,2/3,2/3,0.] if a//16==0 else [-1/3,-1/3,-1/3,-1.])[a%4])==harmonic]


def vector_maps(sector, scalar):
    nf=len(scalar['grid']); count=len(sector['blocks']); maps={}
    for axis,harmonic in product(range(3),[2,-1,-3]):
        matrix=[[None]*count for _ in range(count)]
        terms=vector_flavour_terms(axis,harmonic)
        for j,source in enumerate(sector['blocks']):
            reachable=neighbours(source['signature'])|{source['signature']}
            for i,target in enumerate(sector['blocks']):
                # Anti signatures are stored in positive coordinates; reachability is C invariant.
                if target['signature'] not in reachable:
                    continue
                raw=sparse.kron(one_body(source['sector']['masks'],terms,target['sector']['masks']),sparse.eye(19),format='csr')
                matter=target['sector']['P'].conj().T@raw@source['sector']['P']
                value=target['V'].conj().T@(sparse.kron(matter,sparse.eye(nf))@source['V'])
                if np.linalg.norm(value)>1e-13: matrix[i][j]=sparse.csr_matrix(value)
        # Explicit zero diagonal fixes otherwise empty rows/columns in bmat.
        for i,block in enumerate(sector['blocks']):
            if matrix[i][i] is None: matrix[i][i]=sparse.csr_matrix((block['dimension'],block['dimension']),dtype=complex)
        maps[axis,harmonic]=sparse.bmat(matrix,format='csr')
    return maps


def whole_law(sector, scalar, field, p):
    n=sector['dimension']; nf=len(scalar['grid']); fi=field['identity']
    h=sparse.kron(sparse.block_diag([sparse.csr_matrix(b['V'].conj().T@b['H0']@b['V']) for b in sector['blocks']],format='csr'),fi,format='csr')
    h+=sparse.kron(sparse.eye(n),field['H'],format='csr')
    charge=compressed(sector,'Q',scalar)
    # Uniform spherical matter skin and grounded concentric outer conductor: includes all Q^2 cross terms.
    longitudinal=(1/p['charge_radius']-1/p['boundary_radius'])/(8*np.pi*p['permittivity'])
    h+=sparse.kron(longitudinal*(charge@charge),fi,format='csr')
    j=[sparse.csr_matrix(h.shape,dtype=complex) for _ in range(2)]
    k=[x.copy() for x in j]
    for axis,harmonic in product(range(3),[2,-1,-3]):
        shift=field['shifts'][harmonic][axis]; cosine=(shift+shift.conj().T)/2; sine=(shift-shift.conj().T)/(2j)
        g=sector['vector'][axis,harmonic]
        h+=p['vector_velocity']*sparse.kron(g,sine,format='csr')
        for physical in ([axis] if axis < 2 else [0,1]):
            j[physical]-=p['vector_velocity']*(harmonic/3)*sparse.kron(g,cosine,format='csr')
            k[physical]-=p['vector_velocity']*(harmonic/3)**2*sparse.kron(g,sine,format='csr')
    # Physical mass Peierls paths use the x cycle; ordered Clifford paths use all three.
    for block in sector['blocks']:
        sl=slice(block['offset'],block['offset']+block['dimension']); v=block['V']
        for q,direct,forward,backward in block['op']['transfers']:
            harmonic=round(3*q); shift=field['shifts'][harmonic][0]
            alpha=p['scalar_coupling']*p['loop_route']/2
            route=alpha*np.kron(forward+backward.conj().T,scalar['phi'])
            reduced=v.conj().T@route@v
            embedded=sparse.coo_matrix((reduced.ravel(),(np.repeat(np.arange(sl.start,sl.stop),len(reduced)),np.tile(np.arange(sl.start,sl.stop),len(reduced)))),shape=(n,n)).tocsr()
            delta=shift-fi
            h+=sparse.kron(embedded,delta,format='csr')+sparse.kron(embedded.conj().T,delta.conj().T,format='csr')
            derivative=1j*q*sparse.kron(embedded,shift,format='csr')
            j[0]-=derivative+derivative.conj().T
            contact=-q*q*sparse.kron(embedded,shift,format='csr')
            k[0]+=contact+contact.conj().T
    return h.tocsr(),j,k


def source_between(neutral,target,kind,scalar,spin=.5):
    nf=len(scalar['grid']); rows=[]
    for tb in target['blocks']:
        row=[]
        for nb in neutral['blocks']:
            difference=np.array(tb['signature'])-np.array(nb['signature'])
            if np.any(difference<0) or difference.sum()!=(1 if kind=='lepton' else 3):
                row.append(sparse.csr_matrix((tb['dimension'],nb['dimension']),dtype=complex)); continue
            raw=source_matrix(nb['sector'],tb['sector'],kind,spin)
            row.append(sparse.csr_matrix(tb['V'].conj().T@np.kron(raw,np.eye(nf))@nb['V']))
        rows.append(row)
    return sparse.bmat(rows,format='csr')


def omitted_vector_norm(sector, state, scalar, field, loop, p):
    """Actual H-to-unretained-population residual, retaining coherent electric/source amplitudes."""
    nf=len(scalar['grid']); total=0.; largest=0.; count=0
    for signature in sector['omitted']:
        target=gauge_sector(complete_masks(signature),loop)
        if not target['dimension']: continue
        if sector['anti']: target=conjugate_sector(target,loop)
        image=np.zeros((target['dimension']*nf,field['size']),complex)
        for block in sector['blocks']:
            if signature not in neighbours(block['signature']): continue
            coefficient=state[block['offset']*field['size']:(block['offset']+block['dimension'])*field['size']].reshape(block['dimension'],field['size'])
            full=block['V']@coefficient
            for axis,harmonic in product(range(3),[2,-1,-3]):
                raw=sparse.kron(one_body(block['sector']['masks'],vector_flavour_terms(axis,harmonic),target['masks']),sparse.eye(19),format='csr')
                matter=target['P'].conj().T@raw@block['sector']['P']
                shift=field['shifts'][harmonic][axis]; sine=(shift-shift.conj().T)/(2j)
                image+=p['vector_velocity']*(sparse.kron(matter,sparse.eye(nf))@full)@sine.T
        value=float(np.linalg.norm(image)**2); total+=value; largest=max(largest,value); count+=1
    return dict(norm=float(np.sqrt(total)),largest_block_norm=float(np.sqrt(largest)),omitted_gauge_blocks=count)


def energy_projection_residual(sector,state,scalar,field,p):
    """Residual outside the retained energy vectors INSIDE retained population blocks."""
    nf=len(scalar['grid']); total=0.
    coefficients=[state[b['offset']*field['size']:(b['offset']+b['dimension'])*field['size']].reshape(b['dimension'],field['size']) for b in sector['blocks']]
    for target,coefficient in zip(sector['blocks'],coefficients):
        v=target['V']; full=v@coefficient
        image=target['H0']@full
        for q,_,forward,backward in target['op']['transfers']:
            shift=field['shifts'][round(3*q)][0]; delta=shift-field['identity']
            route=p['scalar_coupling']*p['loop_route']/2*np.kron(forward+backward.conj().T,scalar['phi'])
            image+=route@full@delta.T+route.conj().T@full@delta.conj()
        for source,source_coeff in zip(sector['blocks'],coefficients):
            if target['signature'] not in neighbours(source['signature'])|{source['signature']}: continue
            for axis,harmonic in product(range(3),[2,-1,-3]):
                raw=sparse.kron(one_body(source['sector']['masks'],vector_flavour_terms(axis,harmonic),target['sector']['masks']),sparse.eye(19),format='csr')
                matter=target['sector']['P'].conj().T@raw@source['sector']['P']
                shift=field['shifts'][harmonic][axis]; sine=(shift-shift.conj().T)/(2j)
                image+=p['vector_velocity']*(sparse.kron(matter,sparse.eye(nf))@(source['V']@source_coeff))@sine.T
        residual=image-v@(v.conj().T@image); total+=np.linalg.norm(residual)**2
    return float(np.sqrt(total))
