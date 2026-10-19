"""Actual Haar/material/finite-q kernel contraction equivalence, not a state run."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import expm
from rich_gauge_band_frame_20261016_model import angular,jmat
from rich_local_frame_spatial_carrier_20261016_model import field_basis,displacement,CENTRES
from rich_spatial_frame_photon_20261016_kernel import default_modes,segment
ROOT=Path(__file__).resolve().parent
P='rich_local_frame_spatial_carrier_20261016_'


def main():
    R=angular((0,1,2));J=jmat(.5);pauli=[2*x for x in J];rot=[]
    for a,b,c in R['points']:
        u=expm(-1j*a*J[2])@expm(-1j*b*J[1])@expm(-1j*c*J[2])
        rot.append([[np.trace(pauli[i]@u@pauli[j]@u.conj().T).real/2 for j in range(3)] for i in range(3)])
    rot=np.asarray(rot);body=np.array([CENTRES[0]+.24*(np.array([0,0,0])-.5),CENTRES[2]+.24*(np.array([1,1,0])-.5)])
    lab=np.einsum('qab,sb->qsa',rot,body);modes=default_modes();modes[:,3:6]*=np.sinc(.08*np.linalg.norm(modes[:,:3],axis=1)/np.pi)[:,None]
    phi=segment(lab[:,0],lab[:,1],modes)[0];values=displacement(field_basis(1),-phi)*rot[:,1,2,None,None]
    f=R['F'];w=R['weights'];nR=f.shape[1];nf=values.shape[1]
    t=time.monotonic();old=np.einsum('qi,q,qab,qj->iajb',f.conj(),w,values,f,optimize=True).reshape(nR*nf,nR*nf);oldtime=time.monotonic()-t
    t=time.monotonic();G=(f.conj()[:,:,None]*f[:,None,:]*w[:,None,None]).reshape(len(w),nR*nR).T.copy();new=(G@values.reshape(len(w),nf*nf)).reshape(nR,nR,nf,nf).transpose(0,2,1,3).reshape(nR*nf,nR*nf);newtime=time.monotonic()-t
    result=dict(actual_material_js=[0,1,2],actual_profile='shell-formfactor finite-q Peierls between distinct cells, times actual R12 multiplier',shape=list(old.shape),absolute_L2=float(np.linalg.norm(new-old)),max_absolute=float(np.max(abs(new-old))),old_seconds=oldtime,Gram_BLAS_seconds=newtime,model_sha256=hashlib.sha256((ROOT/(P+'model.py')).read_bytes()).hexdigest(),scope='Algebraic computational representation equivalence only; not scientific transport/evolution evidence.')
    (ROOT/(P+'projection_equivalence.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
