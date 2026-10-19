"""Actual original-positive CAR sources after Hadamard; never assume a filled negative mirror."""
from itertools import product
import numpy as np
from scipy import sparse
from rich_common_colour_lepton_20261015_model import mode, string, gauge_sector
from rich_common_vector_20261015_model import complete_masks, signatures
from rich_colour_composite_20261014_model import epsilon


def source_terms(kind,spin=.5):
    if kind=='lepton':
        return [([(mode(1,m,0,3),True)],2**-.5) for m in range(2)]
    amplitudes={(0,0,1):np.sqrt(2/3),(0,1,0):-np.sqrt(1/6),(1,0,0):-np.sqrt(1/6)} if spin==.5 else {(0,0,1):1/np.sqrt(3),(0,1,0):1/np.sqrt(3),(1,0,0):1/np.sqrt(3)}
    output=[]
    for colours in product(range(3),repeat=3):
        if not epsilon(*colours):continue
        for spins,amplitude in amplitudes.items():
            for mirrors in product(range(2),repeat=3):
                operations=[(mode(f,m,s,c),True) for f,m,s,c in zip((0,0,1),mirrors,spins,colours)]
                output.append((operations,epsilon(*colours)*amplitude/np.sqrt(6)*2**-1.5))
    return output


def physical_source_matrix(neutral,target,kind,spin=.5,removal=False):
    lookup={mask:i for i,mask in enumerate(target['masks'])};rows=[];cols=[];values=[]
    terms=source_terms(kind,spin)
    if removal:
        transformed=[]
        for operations,amplitude in terms:
            mapped=[]
            for index,_ in operations:
                m=(index%16)//8;s=(index%8)//4
                amplitude*=1 if m==s else -1
                mapped.append((index^12,False))
            transformed.append((mapped,amplitude))
        terms=transformed
    for j,mask in enumerate(neutral['masks']):
        for operations,amplitude in terms:
            step=string(mask,operations)
            if step is not None and step[0] in lookup:
                rows.append(lookup[step[0]]);cols.append(j);values.append(amplitude*step[1])
    raw=sparse.csr_matrix((values,(rows,cols)),shape=(len(target['masks']),len(neutral['masks'])))
    return (target['P'].conj().T@sparse.kron(raw,sparse.eye(19))@neutral['P']).toarray()


def compressed_source(neutral,target,kind,scalar,removal=False):
    nf=len(scalar['grid']);rows=[]
    for tb in target['blocks']:
        row=[]
        for nb in neutral['blocks']:
            difference=(np.array(signatures(nb['sector']['masks'][0]))-np.array(signatures(tb['sector']['masks'][0]))) if removal else np.array(tb['signature'])-np.array(nb['signature'])
            if np.any(difference<0) or difference.sum()!=(1 if kind=='lepton' else 3):
                row.append(sparse.csr_matrix((tb['dimension'],nb['dimension']),dtype=complex));continue
            raw=physical_source_matrix(nb['sector'],tb['sector'],kind,removal=removal)
            row.append(sparse.csr_matrix(tb['V'].conj().T@(sparse.kron(raw,sparse.eye(nf))@nb['V'])))
        rows.append(row)
    return sparse.bmat(rows,format='csr')

def full_source_strength(neutral,kind,reference,scalar,field,loop):
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
    total=0.;missing=0.;retained_signatures={b['signature'] for b in neutral.get('target_blocks',[])}
    for signature in sorted(targets):
        target=gauge_sector(complete_masks(signature),loop)
        if not target['dimension']:continue
        image=np.zeros((target['dimension']*nf,field['size']),complex)
        for block in neutral['blocks']:
            difference=np.array(signature)-np.array(block['signature'])
            if np.any(difference<0) or difference.sum()!=(1 if kind=='lepton' else 3):continue
            coefficient=reference[block['offset']*field['size']:(block['offset']+block['dimension'])*field['size']].reshape(block['dimension'],field['size'])
            source=physical_source_matrix(block['sector'],target,kind)
            image+=sparse.kron(source,sparse.eye(nf))@(block['V']@coefficient)
        value=float(np.linalg.norm(image)**2);total+=value
        if signature not in retained_signatures:missing+=value
    return dict(full_unprojected_CAR_strength=total,omitted_population_source_strength=missing,target_signature_count=len(targets),primitive_amplitude_factor=(2**-.5 if kind=='lepton' else 2**-1.5),original_positive_mirror_assignments=(2 if kind=='lepton' else 8))
