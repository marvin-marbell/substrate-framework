"""NEW checkpointed same-law construction, without obsolete filled-sea source diagnostics.
Saves the actual full projected H/eigenbasis before any costly physical receiver work.
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from rich_quantum_common_field_20261015_model import build, rotor, expand_sector, vector_maps, whole_law, conjugate_sector, operators, law
from rich_quantum_common_field_20261015_sources import compressed_source
from rich_quantum_common_field_20261015_calculate import save_sparse


def main():
    inp,out=map(Path,sys.argv[1:3]);p=json.loads(inp.read_text());base=build(p);scalar=base['scalar'];field=rotor(p)
    sectors={};spectra={};results={};stored={}
    for name in ['neutral','lepton','baryon']:
        sector=expand_sector(base['sectors'][name],base['loop'],scalar,p);sectors[name]=sector
        sector['vector']=vector_maps(sector,scalar);h,j,k=whole_law(sector,scalar,field,p)
        e,v=eigsh(h,k=p['low_poles'],which='SA',tol=2e-10,maxiter=20000)
        q,_=np.linalg.qr(v);e,r=np.linalg.eigh(q.conj().T@(h@q));v=q@r;spectra[name]=(e,v)
        if name=='neutral':state=v[:,0];reference=state
        else:
            source=sparse.kron(compressed_source(sectors['neutral'],sector,name,scalar),field['identity'],format='csr')
            raw=source@reference;amplitudes=v.conj().T@raw;weights=abs(amplitudes)**2
            bright=np.flatnonzero(weights>p['bright_threshold'])
            if not len(bright):raise RuntimeError('No low source-bright pole; no source gap is invented')
            pole=abs(e-e[bright[0]])<1e-7;state=v[:,pole]@amplitudes[pole];state/=np.linalg.norm(state)
            save_sparse(stored,name+'_source',source);stored[name+'_raw_source']=raw
        stored[name+'_stationary']=state;stored[name+'_e']=e;stored[name+'_v']=v;save_sparse(stored,name+'_H',h)
        for axis in range(2):save_sparse(stored,name+'_J'+str(axis),j[axis]);save_sparse(stored,name+'_K'+str(axis),k[axis])
        for index,block in enumerate(sector['blocks']):
            prefix=name+'_block'+str(index);stored[prefix+'_V']=block['V'];stored[prefix+'_signature']=block['signature'];stored[prefix+'_masks']=np.array(block['sector']['masks'],np.uint64);save_sparse(stored,prefix+'_P',block['sector']['P'])
        results[name]=dict(ground_energy=float(e[0]),retained_gauge_blocks=len(sector['blocks']),population_layers=sector['layers'],dimension=h.shape[0],max_colour_Gauss=max(b['sector']['gauge_error'] for b in sector['blocks']),orthonormal_Gram=float(np.linalg.norm(v.conj().T@v-np.eye(v.shape[1]))),eigen_residual=float(np.linalg.norm(h@state-np.vdot(state,h@state)*state)))
        if name!='neutral':
            anti=dict(sector);anti['blocks']=[];anti['anti']=True
            for block in sector['blocks']:
                ab=dict(block);ab['sector']=conjugate_sector(block['sector'],base['loop']);ab['op']=operators(ab['sector'],base['loop']);ab['H0']=law(ab['op'],scalar,p)[0];anti['blocks'].append(ab)
            anti['vector']=vector_maps(anti,scalar);ah,aj,ak=whole_law(anti,scalar,field,p);ca=sparse.kron(sparse.eye(sector['dimension']),sparse.csr_matrix(np.fliplr(np.eye(field['size']))),format='csr')
            save_sparse(stored,'anti'+name+'_H',ah)
            results['anti'+name+'_C']=dict(H=float(sparse.linalg.norm(ah@ca-ca@h)),J=[float(sparse.linalg.norm(aj[a]@ca+ca@j[a])) for a in range(2)],K=[float(sparse.linalg.norm(ak[a]@ca-ca@k[a])) for a in range(2)])
            del anti,ah,aj,ak
        output=dict(kind='CHECKPOINTED_SAME_NONLINEAR_COMMON_H_WITH_FULL_HADAMARD_CAR_SOURCE',parameters=p,results=results,electric_metric=field['metric'].tolist())
        out.write_text(json.dumps(output,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**stored)
        print(json.dumps(dict(stage='checkpoint',sector=name,result=results[name])),flush=True)
    print(json.dumps(dict(stage='complete',meaning='Exact own whole-H/source construction checkpoint; physical field-only reaction is separate same-H consumer.')),flush=True)


if __name__=='__main__':main()
