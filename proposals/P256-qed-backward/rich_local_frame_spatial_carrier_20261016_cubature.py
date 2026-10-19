"""Same retained R2 quantum operator: Gaunt identity and independent cubature."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from rich_gauge_band_frame_20261016_model import angular,geometry
from rich_local_frame_spatial_carrier_20261016_angular import material,project,rotations
from rich_local_frame_spatial_carrier_20261016_model import CENTRES,field_basis,field_moments,displacement
import rich_spatial_frame_photon_20261016_kernel as spatial
ROOT=Path(__file__).resolve().parent
P='rich_local_frame_spatial_carrier_20261016_'


def operators(R):
    body=(CENTRES[:,None,:]+.24*(geometry()['sites'][None,:,:]-.5)).reshape(32,3)
    lab=np.einsum('qab,sb->qsa',rotations(R['points']),body)
    modes=spatial.default_modes();modes[:,3:6]*=np.sinc(.08*np.linalg.norm(modes[:,:3],axis=1)/np.pi)[:,None]
    phi=spatial.segment(lab[:,0],lab[:,31],modes)[0]
    phase=displacement(field_basis(1),-phi)*rotations(R['points'])[:,1,0,None,None]
    charges=np.zeros(32);charges[[0,7,13,22,31]]=[-1,1,-1,-1,1]
    connection=np.cross(lab[:,:,None,:],spatial.profiles(lab,modes))
    Lambda=np.einsum('s,qsla->qla',charges,connection)
    Q,QQ,PP=field_moments(field_basis(1))
    square=np.einsum('qla,qma,lmij->qij',Lambda,Lambda,QQ,optimize=True)/12
    current=-np.einsum('qa,qla,lij->qij',Lambda[:,0],Lambda,Q)/6
    return dict(Peierls_times_R=phase,full_mechanical_Lambda2=square,mechanical_current0=current)


def gram(R,values):
    f=R['F'];w=R['weights'];d=f.shape[1];nf=values.shape[1]
    G=(f.conj()[:,:,None]*f[:,None,:]*w[:,None,None]).reshape(len(w),d*d).T
    return (G@values.reshape(len(w),nf*nf)).reshape(d,d,nf,nf).transpose(0,2,1,3).reshape(d*nf,d*nf)


def main():
    start=time.time();R24=material(2,24,16);R32=material(2,32,20);old=angular((0,1,2))
    values24=operators(R24);values32=operators(R32);valuesold=operators(old)
    outputs={};raw={}
    for name in values24:
        direct=gram(R24,values24[name]);fast=project(R24,values24[name]);fine=project(R32,values32[name]);oldmat=gram(old,valuesold[name])
        outputs[name]=dict(Gaunt_vs_direct_norm=float(np.linalg.norm(fast-direct)),Gaunt_vs_direct_max=float(np.max(abs(fast-direct))),grid24_vs32_norm=float(np.linalg.norm(fast-fine)),grid24_vs32_max=float(np.max(abs(fast-fine))),old7roots_vs32_norm=float(np.linalg.norm(oldmat-fine)),old7roots_vs32_max=float(np.max(abs(oldmat-fine))))
        raw[name+'_24']=fast;raw[name+'_32']=fine;raw[name+'_old']=oldmat
    result=dict(same_retained_material_js=[0,1,2],field_cap=1,coarse=R24['quadrature'],fine=R32['quadrature'],same_law_no_state_cap_change=True,results=outputs,gram24=R24['gram_error'],gram32=R32['gram_error'],elapsed_seconds=time.time()-start,scope='Selected full finite-q raw Peierls/Lambda²/current blocks, not all-H angular or field convergence theorem.')
    tag=sys.argv[1]+'_' if len(sys.argv)>1 else ''
    np.savez_compressed(ROOT/(P+tag+'cubature_raw.npz'),**raw)
    (ROOT/(P+tag+'cubature_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
