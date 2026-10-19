"""Owning before-cut current/contact source matrices, not extra four-point seagulls."""
from typing import Any
from itertools import product
from pathlib import Path
import json,resource,sys,time
import numpy as np
from rich_spatial_translation_20261015_model import shell_fields
ROOT=Path(__file__).resolve().parent


def main()->None:
    started=time.time();config:Any=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);law:Any=json.loads((ROOT/config['law_input']).read_text());primary:Any=json.loads((ROOT/config['primary_result']).read_text());states=np.load(ROOT/config['primary_states']);cached=np.load(ROOT/config['reconstruction']);field=shell_fields(law);site=primary['exact_transverse_fiber']['selected_source_site'];arrays={};report={}
    for name,state_key,index,number in [('neutral','coupled_reference',0,27),('positive','coupled_positive_sources',site,162),('negative','coupled_negative_sources',site,162)]:
        vectors=states[state_key]
        if vectors.ndim==1:vectors=vectors[:,None]
        f=cached[name+'_projected_components'][index];a=vectors.reshape(number,field['dimension'],-1);columns=a.shape[-1]
        def matrix(terms: Any)->Any:
            acted=np.zeros_like(a,complex)
            for fc,kc in zip(f,terms):
                if kc.nnz:
                    left=(fc@a.reshape(number,-1)).reshape(number,field['dimension'],columns)
                    acted+=(kc@left.transpose(1,0,2).reshape(field['dimension'],-1)).reshape(field['dimension'],number,columns).transpose(1,0,2)
            return vectors.conj().T@acted.reshape(vectors.shape)
        currents=np.array([-matrix(d) for d in field['derivatives']]);contacts=np.array([[matrix(d) for d in row] for row in field['contacts']]);arrays[name+'_current_source_matrix']=currents;arrays[name+'_contact_source_matrix']=contacts
        data=dict(raw_source_Gram_real=(vectors.conj().T@vectors).real.tolist(),raw_source_Gram_imag=(vectors.conj().T@vectors).imag.tolist(),
            current_real=currents.real.tolist(),current_imag=currents.imag.tolist(),contact_real=contacts.real.tolist(),contact_imag=contacts.imag.tolist())
        if name!='neutral':
            fourier=np.empty((2,2,12,12),complex);allowed=[];forbidden=[]
            for pol in range(2):
                for sign_index,eta in enumerate([-1,1]):
                    fourier[pol,sign_index]=(currents[2*pol]+1j*eta*currents[2*pol+1])/np.sqrt(2)
                    for outmode,inmode in product(range(12),repeat=2):
                        kout,kin=outmode//4,inmode//4;expected=(kin+(eta if name=='positive' else -eta))%3
                        (allowed if kout==expected else forbidden).append(fourier[pol,sign_index,outmode,inmode])
            arrays[name+'_Fourier_current_source_matrix']=fourier
            data.update(allowed_current_Frobenius=float(np.linalg.norm(allowed)),forbidden_current_Frobenius=float(np.linalg.norm(forbidden)),
                current_selection='N55: kout=kin+eta; N53 removal source labels: kout=kin-eta because hole total momentum is opposite canonical removed mode. No channels filtered.')
        report[name]=data
    result=dict(kind='OWNING_RECOIL_RESOLVED_CURRENT_CONTACT_DATA_NOT_EXTRA_SCATTERING_SEAGULL',input=config,reports=report,
        semantics='Raw canonical-source 12x12 matrices Vdag J_a V and Vdag K_ab V on SAME fresh coupled owning reference. Both CAR branches, all4spinors/3kz, both pols/+/-q retained. NOT amputated on-shell vertices, COM momentum definition, normalized probabilities or isolated electron spinflip.',
        no_double_count='The full-H B-Q-Q-Bdag connected tensor already contains every retained Peierls/contact effect. These owning derivative data are interpretation/diagnostics ONLY; never add them as a second seagull to that full correlator.',
        EM_off_control='Complete EM-off law has charge coupling0, hence its owning EM current/contact derivatives are ZERO. Coupled derivative matrices are not falsely assigned to that control.',
        elapsed_seconds=time.time()-started,peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    np.savez_compressed(output.with_suffix('.npz'),**arrays);output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(result=str(output),elapsed=result['elapsed_seconds'],peak_resident_KiB=result['peak_resident_KiB'])),flush=True)


if __name__=='__main__':main()
