"""NEW exact two-polarization source/current Gram consumer of owned states."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from rich_extensive_photon_20261014_model import matter, whole_h, signed_lehmann


def plain(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic): return obj.item()
    raise TypeError(type(obj).__name__)


def tensor(amplitudes):
    value=amplitudes.conj() @ amplitudes.T
    return dict(real=value.real, imag=value.imag, eigenvalues=np.linalg.eigvalsh(value), determinant=float(np.linalg.det(value).real))


def main():
    inp,out=map(Path,sys.argv[1:3]);request=json.loads(inp.read_text());root=inp.parent
    for name,expected in request['owned_inputs'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('Owned input changed: '+name)
    cfg=json.loads((root/request['physical_input']).read_text());mat=matter(cfg)
    raw=json.loads((root/request['result']).read_text());states=np.load(root/request['states'])
    result=dict(schema=1,meaning='Exact physical polarization spectral tensors, not two arbitrary labels or diagonal weights alone',points=[])
    saved={}
    for multiplier,point in zip(cfg['exact_q_harmonics'],raw['exact']):
        law=whole_h(mat,cfg,np.array(point['q']));energy=states[f'q{multiplier}_energies'];vectors=states[f'q{multiplier}_vectors'];ground=vectors[:,0];gaps=energy[1:]-energy[0]
        coordinate=np.array([vectors[:,1:].conj().T @ a @ ground for a in law['Q']]);current=np.array([vectors[:,1:].conj().T @ a @ ground for a in law['currents']])
        first=np.flatnonzero(np.sum(abs(coordinate)**2,axis=0)>1.e-6)[:2]
        low=point['bright_groups'][0]['energy'];window=(gaps>0.8*low)&(gaps<1.2*low)
        current_gram=tensor(current[:,window]);coordinate_gram=tensor(coordinate[:,window])
        poles=[]
        for index in first:
            poles.append(dict(energy=float(gaps[index]),coordinate=tensor(coordinate[:,index:index+1]),current=tensor(current[:,index:index+1])))
        contact=np.array([[np.vdot(ground,law['contacts'][a][b] @ ground).real for b in range(2)] for a in range(2)])
        static=contact+signed_lehmann(gaps,current,0.).real
        result['points'].append(dict(q=point['q'],polarizations=point['polarizations'],all_retained_virtuals=len(gaps),
            low_field_window=[0.8*low,1.2*low],window_state_count=int(window.sum()),poles=poles,
            low_field_current_Gram=current_gram,low_field_coordinate_Gram=coordinate_gram,
            full_current_Gram=tensor(current),physical_contact=contact,static_contact_plus_signed_current=static,
            independent_low_polarization_rank=int(np.sum(np.array(current_gram['eigenvalues'])>1.e-12))))
        saved[f'q{multiplier}_Q_amplitude']=coordinate;saved[f'q{multiplier}_J_amplitude']=current
    out.write_text(json.dumps(result,default=plain,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**saved)
    print(json.dumps(dict(stage='complete',independent_ranks=[x['independent_low_polarization_rank'] for x in result['points']])) )


if __name__=='__main__':main()
