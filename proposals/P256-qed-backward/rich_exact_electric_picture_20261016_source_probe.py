"""Actual rotating-microsite Wilson source: absolute time/BCH/R derivatives."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
import rich_exact_electric_picture_20261016_fields as f
import rich_spatial_frame_photon_20261016_fields as primitive
import rich_spatial_frame_photon_20261016_kernel as geometry
import rich_spatial_frame_photon_20261016_model as frozen
ROOT=Path(__file__).resolve().parent
P='rich_exact_electric_picture_20261016_'


def main():
    R=frozen.rotations(np.array([[.37,.81,-.29]]))[0];body=frozen.band.geometry()['sites']-.5;x=body@R.T;modes=geometry.default_modes();g=geometry.geometry(x,modes)
    c=-g['F'][3];eps=np.zeros(8);eps[0]=.23;t=.3;I=2.;h=2.e-6;result={};raw={}
    for cut in (1,2):
        b=primitive.basis(cut);ext=primitive.basis(cut+2);mid=primitive.basis(cut+1)
        left=f.primitive_images(mid,b,0.,I,eps);right=f.primitive_images(mid,ext,0.,I,eps)
        pl=left['p']+eps[:,None,None]*left['embedding'];pr=right['p']+eps[:,None,None]*right['embedding']
        cross=np.einsum('lei,lej->ij',pl.conj(),pr,optimize=True)/(2*I)
        D=f.displacement(b,b,c,t,I,eps)
        timefd=(f.displacement(b,b,c,t+h,I,eps)-f.displacement(b,b,c,t-h,I,eps))/(2*h)
        exactdt=1j*(cross@f.displacement(ext,b,c,t,I,eps)-f.displacement(b,ext,c,t,I,eps)@cross.conj().T)
        cutH=np.einsum('lei,lej->ij',pl.conj(),pl,optimize=True)/(2*I)
        wrongdt=1j*(cutH@D-D@cutH)
        R_errors=[]
        for axis in range(3):
            j=np.eye(3)[axis];generator=np.array([[0.,-j[2],j[1]],[j[2],0.,-j[0]],[-j[1],j[0],0.]])
            xp=x@expm(h*generator).T;xm=x@expm(-h*generator).T
            cp=-geometry.geometry(xp,modes)['F'][3];cm=-geometry.geometry(xm,modes)['F'][3]
            fd=(f.displacement(b,b,cp,t,I,eps)-f.displacement(b,b,cm,t,I,eps))/(2*h)
            exact=sum(-g['drotation'][3,l,axis]*f.coefficient_derivative(b,b,c,[l],t,I,eps) for l in range(8))
            R_errors.append(float(np.linalg.norm(fd-exact)))
        record=dict(full_initial_field_dimension=len(b),absolute_time=t,physical_site=3,source_coefficient=c.tolist(),time_derivative_full_H0_commutator_error=float(np.linalg.norm(timefd-exactdt)),WRONG_cut_H0_commutator_error=float(np.linalg.norm(timefd-wrongdt)),R_derivative_errors=R_errors,WRONG_time_zero_source_matrix_error=float(np.linalg.norm(D-f.displacement(b,b,c,0.,I,eps))))
        result[str(cut)]=record;raw.update({f'cut{cut}_source':D,f'cut{cut}_time_FD':timefd,f'cut{cut}_full_H0_derivative':exactdt,f'cut{cut}_wrong_cut_H0_derivative':wrongdt});print(cut,json.dumps(record),flush=True)
    np.savez_compressed(ROOT/(P+'source_probe_raw.npz'),**raw)
    (ROOT/(P+'source_probe_result.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
