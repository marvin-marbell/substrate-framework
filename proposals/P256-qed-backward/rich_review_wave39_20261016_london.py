"""Only invalidated evidence: unprojected London vs finite complement, endpoint and Gauss signs."""
import json
from pathlib import Path
import numpy as np
from rich_local_frame_carrier_20261015_model import Model,VERTICES
from rich_spatial_gauge_source_20261016_family import SpatialFamily,Terms
import rich_bulk_photon_20261016_state as b
import rich_flux_phase_photon_20261016_state as s
ROOT=Path(__file__).resolve().parent
P='rich_review_wave39_20261016_'

def main():
    interface=json.loads((ROOT/'rich_dressed_local_carrier_20261015_interface.json').read_text())
    nm=Model(6,interface['parameters']);m=Model(7,interface['parameters'])
    with np.load(ROOT/interface['state_file']) as saved:neutral=saved['neutral'].copy()
    a=np.array([.2,-.3,.7]);a/=np.linalg.norm(a)
    f=SpatialFamily(m,a,'constant',8);nf=SpatialFamily(nm,a,'constant',8)
    B=m.source_from(nm,1,'addition',0,0);v=B@neutral;norm=np.linalg.norm(v)
    # Q=ions-electrons, so electron creation at endpoint1 changes dipole by -a.x1.
    ward=f.dipole()@v-B@(nf.dipole()@neutral)
    endpoint=Terms(m,[(np.ones(m.sector.dimension),f.project(f.x[:,1]@a),1)])@v
    endpoint_error=np.linalg.norm(ward+endpoint)/norm
    endpoint_bad=np.linalg.norm(ward-endpoint)/norm
    # Full-space derivative of exp(i a sum Q x) under LEFT rotor action.
    oneq=np.array([1.,-1.,-1.]);delta=1e-5;axis_errors=[]
    for axis in range(3):
        basis=np.eye(3)[axis];K=np.array([[0,-basis[2],basis[1]],[basis[2],0,-basis[0]],[-basis[1],basis[0],0.]])
        rotplus=np.eye(3)+np.sin(delta)*K+(1-np.cos(delta))*(K@K)
        rotminus=np.eye(3)-np.sin(delta)*K+(1-np.cos(delta))*(K@K)
        xp=np.einsum('ab,qvb->qva',rotplus,f.x);xm=np.einsum('ab,qvb->qva',rotminus,f.x)
        derivative=(np.einsum('qva,a,v->q',xp,a,oneq)-np.einsum('qva,a,v->q',xm,a,oneq))/(2*delta)
        lam=np.einsum('v,qv->q',oneq,f.connection[:,:,axis])
        axis_errors.append(float(np.max(abs(derivative-lam))))
    # Independently sampled Gauss: this L4 producer uses divE=1-n, not n-1.
    F,G,codes,_=s.kernel(True,'pair',.8)
    u,d,E,acc,prop=b.sample(391199,48,2000,80,F,G,.45,.15,.08,.2,1e-12)
    gauss=[];badgauss=[]
    for uu,dd,ee in zip(u,d,E):
        density=np.bincount(uu,minlength=64)+np.bincount(dd,minlength=64)
        gauss.append(np.max(abs(b.INCIDENCE@ee-(1-density))))
        badgauss.append(np.max(abs(b.INCIDENCE@ee-(density-1))))
    result=dict(endpoint_site=1,source_covariance_derivative_error=float(endpoint_error),wrong_endpoint_phase_error=float(endpoint_bad),full_London_connection_derivative_max=axis_errors,correct_Gauss_max=float(max(gauss)),wrong_Gauss_sign_max=float(max(badgauss)),samples=48,seed=391199,corrections=['Prior pi_result Gauss_max used wrong reviewer sign n-1; it is an adverse-sign statistic, not a physical violation. Fresh selected correct sign measured here.','Prior spatial exit1 was overstrict reviewer assertion constant_Ward_relative=0 at finite P. Measured .27267 finite complement is NOT failed full-space London covariance. Derivative/contact/spin checks passed before it.','Prior spatial scalar_z_spinflip_complex field label was wrong: actual direction was normalized(.2,-.3,.7), a VECTOR magnetic perturbation. Nonzero offdiagonal is allowed, not forced scalar spinflip or intrinsic g2.'])
    np.savez_compressed(ROOT/(P+'london_raw.npz'),gauss=np.array(gauss),wrong_gauss=np.array(badgauss))
    (ROOT/(P+'london_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert endpoint_error<1e-10 and endpoint_bad>1e-3
    assert max(axis_errors)<1e-8 and max(gauss)==0 and max(badgauss)>0
if __name__=='__main__':main()
