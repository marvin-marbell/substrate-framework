"""Small physical-Gauss algorithm control, NOT replacement for the full author regulator."""
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from rich_quantum_common_field_20261015_model import build,rotor,expand_sector,omitted_vector_norm,neighbours,complete_masks,one_body,vector_flavour_terms
from rich_quantum_common_field_20261015_boundary import omitted_population,full_source_strength
from rich_quantum_common_field_20261015_sources import full_source_strength as projected_source_strength


def incoherent_wrong(sector,state,scalar,field,p):
    nf=len(scalar['grid']);total=0.
    for signature in sector['omitted']:
        masks=complete_masks(signature)
        for block in sector['blocks']:
            if signature not in neighbours(block['signature']):continue
            coefficient=state[block['offset']*field['size']:(block['offset']+block['dimension'])*field['size']].reshape(block['dimension'],field['size'])
            for axis,harmonic in product(range(3),[2,-1,-3]):
                raw=sparse.kron(one_body(block['sector']['masks'],vector_flavour_terms(axis,harmonic),masks),sparse.eye(19),format='csr')@block['sector']['P']
                shift=field['shifts'][harmonic][axis];sine=(shift-shift.conj().T)/(2j)
                image=p['vector_velocity']*(sparse.kron(raw,sparse.eye(nf))@(block['V']@coefficient))@sine.T
                total+=np.linalg.norm(image)**2
    return float(np.sqrt(total))


def main():
    inp,out=map(Path,sys.argv[1:3]);p=json.loads(inp.read_text());base=build(p);scalar=base['scalar'];field=rotor(p)
    neutral=expand_sector(base['sectors']['neutral'],base['loop'],scalar,p)
    rng=np.random.default_rng(912837);state=rng.normal(size=neutral['dimension']*field['size'])+1j*rng.normal(size=neutral['dimension']*field['size']);state/=np.linalg.norm(state)
    exact=omitted_vector_norm(neutral,state,scalar,field,base['loop'],p);raw=omitted_population(neutral,state,scalar,field,p);wrong=incoherent_wrong(neutral,state,scalar,field,p)
    sources={}
    for name in ['lepton','baryon']:
        target=expand_sector(base['sectors'][name],base['loop'],scalar,p);neutral['target_blocks']=target['blocks']
        projected=projected_source_strength(neutral,name,state,scalar,field,base['loop']);streamed=full_source_strength(neutral,name,state,scalar,field,base['loop'])
        sources[name]=dict(projected=projected,raw_streamed=streamed,absolute_strength_difference=abs(projected['full_unprojected_CAR_strength']-streamed['full_unprojected_CAR_strength']))
    result=dict(kind='POSITIVE_AND_ADVERSE_COHERENCE_ALGORITHM_CONTROL_NOT_FULL_AUTHOR_PHYSICS',parameters=p,dimension=len(state),projected_complete_Gauss_boundary=exact,raw_streamed_boundary=raw,absolute_norm_difference=abs(exact['norm']-raw['norm']),incoherent_wrong_norm=wrong,incoherent_difference=abs(wrong-raw['norm']),sources=sources)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
