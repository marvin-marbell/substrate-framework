"""Exact retained-product Gaunt check against independent direct physical projection."""
from pathlib import Path
import json
import numpy as np
import rich_local_frame_spatial_carrier_20261016_angular as angular
import rich_local_frame_spatial_carrier_20261016_model as carrier
import rich_gauge_band_frame_20261016_model as band
import rich_spatial_frame_photon_20261016_kernel as kernel
import rich_spatial_frame_photon_20261016_fields as fields

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def run():
    a=angular.material(2);F=a['F'];w=a['weights'];nR=len(a['labels']);basis=fields.basis(1);nf=len(basis)
    body=(carrier.CENTRES[:,None,:]+.24*(band.geometry()['sites'][None,:,:]-.5)).reshape(32,3)
    modes=kernel.default_modes();modes[:,3:6]*=np.sinc(.08*np.linalg.norm(modes[:,:3],axis=1)/np.pi)[:,None]
    x=np.einsum('qij,sj->qsi',a['rotations'],body)
    phi=kernel.segment(x[:,0],x[:,10],modes)[0];disp=fields.displacement(basis,basis,-phi)
    gram=(F.conj()[:,:,None]*F[:,None,:]*w[:,None,None]).reshape(len(w),nR*nR).T
    def direct(v):return (gram@v.reshape(len(w),-1)).reshape(nR,nR,nf,nf).transpose(0,2,1,3).reshape(nR*nf,nR*nf)
    profile=kernel.profiles(x,modes);connection=np.cross(x[:,:,None,:],profile)
    Q=np.zeros(32);Q[0]=-1;Q[10]=1;Q[11]=-1
    lam=np.einsum('s,qsla->qla',Q,connection)
    ext=fields.basis(2);qe=np.array([fields.primitive(ext,basis,l,'q').toarray() for l in range(8)])
    image=np.einsum('qla,lek->qaek',lam,qe,optimize=True)
    square=np.einsum('qaei,qaej->qij',image.conj(),image,optimize=True)
    radialF=kernel.segment(np.zeros(3),x[:,0],modes)[0]
    samples=dict(Peierls_R=disp*a['rotations'][:,1,2,None,None],Lambda_squared=square,mixed_source_contact=disp*(-phi[:,0]*phi[:,6])[:,None,None],radial_Wilson_source=fields.displacement(basis,basis,-radialF))
    errors={};arrays={}
    for name,v in samples.items():
        actual=angular.project(a,v);expected=direct(v);errors[name]=dict(norm=float(np.linalg.norm(actual-expected)),max=float(np.max(abs(actual-expected))))
        assert errors[name]['norm']<1e-11,(name,errors[name])
        arrays[name+'_Gaunt']=actual;arrays[name+'_direct']=expected
    result=dict(kind='OWN_ACTUAL47_GAUNT_DIRECT_SOURCE_RELEVANT_REVIEW',retained_R=35,field=9,integration_nodes=len(w),errors=errors,limits='Auxiliary l<=4 is exact expansion of products of retained l<=2 PW bra/ket, NOT extra high-l evolved state or a truncation of finite-q profiles. Direct comparison uses same oversampled SO3 integration; separate cubature refinement evidence is required. Includes actual pointwise Peierls*R, Lambda², source mixed contact and radial Wilson source; SU2 g unchanged.')
    np.savez_compressed(ROOT/(P+'gaunt_raw.npz'),**arrays)
    (ROOT/(P+'gaunt_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
