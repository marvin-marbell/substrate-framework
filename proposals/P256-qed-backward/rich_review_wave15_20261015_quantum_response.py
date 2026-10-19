"""Combine already exercised full-H retarded orders with independent full contact tensor."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse


def load(data,key):return sparse.csr_matrix((data[key+'_data'],data[key+'_indices'],data[key+'_indptr']),shape=tuple(data[key+'_shape']))


def main():
    inp,out=map(Path,sys.argv[1:3]);req=json.loads(inp.read_text());observed=json.loads(Path(req['owner_json']).read_text());data=np.load(req['owner_npz']);h=load(data,'neutral_H');ground=data['neutral_stationary'];cut=req['electric_cut'];ns=2*cut+1;field=ns*ns;matter=h.shape[0]//field;values=np.arange(-cut,cut+1);nx=np.tile(np.repeat(values,ns),matter)/3;ny=np.tile(np.tile(values,ns),matter)/3;number=[sparse.diags(nx),sparse.diags(ny)];contact=np.zeros((2,2));errors=[]
    for a in range(2):
        ja=1j*(h@number[a]-number[a]@h);errors.append(float(sparse.linalg.norm(ja-load(data,'neutral_J'+str(a)))))
        for b in range(2):
            k=-(number[a]@number[b]@h-number[a]@h@number[b]-number[b]@h@number[a]+h@number[a]@number[b]);contact[a,b]=float(np.vdot(ground,k@ground).real)
    response=[]
    for item in observed['signed_neutral_retarded_response']:
        response.append(dict(frequency=item['frequency'],full_physical_real=(np.array(item['paramagnetic_real'])+contact).tolist(),full_physical_imag=item['paramagnetic_imag'],solver=item['solver'],all_retained_virtuals=item['all_retained_virtuals']))
    result=dict(kind='FULL_TWO_CURRENT_PHYSICAL_CONTACT_AND_SIGNED_RETAINED_RESPONSE',contact_tensor=contact.tolist(),current_n_over3_errors=errors,retarded=response,source='Both signed resolvent orders reused ONLY from exact own NEW full-cut natural consumer. Kxy independently computed as -[nx/3,[ny/3,H]], including actual shared third cycle, not set to zero.',meaning='a=3theta, so physical source differentiates a and uses n/3. Uniform external a offsets form an exactly unitary compact-coordinate family; cancellation or electric-box poles do not establish absent magnetism or positive massless-photon stiffness. Finite eta response and original nonlinear projection limits explicit.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
