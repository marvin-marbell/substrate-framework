"""Exact omitted-population H image, without diagonalizing a redundant target Gauss kernel.
A colour-singlet current maps physical inputs into ker(all eight G). Its RAW image
norm is therefore the complete physical-target norm; no target virtuals are removed.
"""
from itertools import product
import numpy as np
from scipy import sparse
from rich_quantum_common_field_20261015_model import complete_masks, neighbours, vector_flavour_terms, one_body, charge_conjugate


def omitted_population(sector,state,scalar,field,p):
    nf=len(scalar['grid']);total=0.;largest=0.;blocks=0;largest_raw=0
    for signature in sector['omitted']:
        masks=complete_masks(signature)
        if sector['anti']:masks=sorted(charge_conjugate(mask)[0] for mask in masks)
        rows=len(masks)*19*nf;largest_raw=max(largest_raw,rows);maps=[]
        for source in sector['blocks']:
            if signature not in neighbours(source['signature']):continue
            coefficient=state[source['offset']*field['size']:(source['offset']+source['dimension'])*field['size']].reshape(source['dimension'],field['size'])
            for axis,harmonic in product(range(3),[2,-1,-3]):
                body=one_body(source['sector']['masks'],vector_flavour_terms(axis,harmonic),masks)
                if not body.nnz:continue
                raw=sparse.kron(body,sparse.eye(19),format='csr')@source['sector']['P']
                if not raw.nnz:continue
                mapped=sparse.kron(raw,sparse.eye(nf),format='csr')
                shift=field['shifts'][harmonic][axis];sine=(shift-shift.conj().T)/(2j)
                maps.append((mapped,source['V'],coefficient,sine))
        value=0.
        for start in range(0,field['size'],7):
            stop=min(start+7,field['size']);image=np.zeros((rows,stop-start),complex)
            for mapped,inclusion,coefficient,sine in maps:
                field_shifted=coefficient@sine.T[:,start:stop]
                image+=p['vector_velocity']*(mapped@(inclusion@field_shifted))
            value+=float(np.linalg.norm(image)**2)
        total+=value;largest=max(largest,value);blocks+=1
    return dict(norm=float(np.sqrt(total)),largest_block_norm=float(np.sqrt(largest)),omitted_signature_blocks=blocks,largest_raw_target_scalar_dimension=largest_raw,electric_column_batch=7,meaning='Exact coherent raw CAR/PW19/scalar image outside retained populations. Current is colour-singlet and maps Gauss-null input into Gauss-null target, so raw norm equals COMPLETE target Gauss-kernel norm without target diagonalization. Not a sector cut, current norm probability, or inherited linear closure.')


def full_source_strength(neutral,kind,reference,scalar,field,loop):
    """Full original-positive CAR norm, streamed in raw targets (source is a colour singlet)."""
    from rich_quantum_common_field_20261015_sources import source_terms, physical_source_matrix
    nf=len(scalar['grid']);changes=set()
    for operations,_ in source_terms(kind):
        delta=[0]*16
        for index,_ in operations:
            f=index//16;m=(index%16)//8;s=(index%8)//4;c=0 if index%4<3 else 1
            delta[((f*2+m)*2+s)*2+c]+=1
        changes.add(tuple(delta))
    targets=set()
    for block in neutral['blocks']:
        for delta in changes:
            target=tuple(np.array(block['signature'])+np.array(delta))
            if all(value <= (3 if i%2==0 else 1) for i,value in enumerate(target)):targets.add(target)
    total=0.;missing=0.;retained={b['signature'] for b in neutral['target_blocks']}
    for signature in sorted(targets):
        masks=complete_masks(signature);rawdim=len(masks)*19
        target=dict(masks=masks,P=sparse.eye(rawdim,format='csr'));maps=[]
        for source in neutral['blocks']:
            difference=np.array(signature)-np.array(source['signature'])
            if np.any(difference<0) or difference.sum()!=(1 if kind=='lepton' else 3):continue
            matrix=physical_source_matrix(source['sector'],target,kind)
            if not np.any(matrix):continue
            coefficient=reference[source['offset']*field['size']:(source['offset']+source['dimension'])*field['size']].reshape(source['dimension'],field['size'])
            maps.append((matrix,source['V'],coefficient,source['sector']['P'].shape[1]))
        value=0.
        for start in range(0,field['size'],7):
            stop=min(start+7,field['size']);image=np.zeros((rawdim,nf*(stop-start)),complex)
            for matrix,inclusion,coefficient,dimension in maps:
                full=(inclusion@coefficient[:,start:stop]).reshape(dimension,nf*(stop-start))
                image+=matrix@full
            value+=float(np.linalg.norm(image)**2)
        total+=value
        if signature not in retained:missing+=value
    return dict(full_unprojected_CAR_strength=total,omitted_population_source_strength=missing,target_signature_count=len(targets),primitive_amplitude_factor=(2**-.5 if kind=='lepton' else 2**-1.5),original_positive_mirror_assignments=(2 if kind=='lepton' else 8),norm_method='Exact coherent raw source image over every valid target signature, streaming7 electric columns. Epsilon uud/lepton sources commute colour Gauss, so no target Gauss diagonalization or target-state omission.')
