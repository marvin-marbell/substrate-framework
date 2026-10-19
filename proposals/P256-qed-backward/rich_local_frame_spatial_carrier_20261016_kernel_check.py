"""Actual cube-spin and finite-q Hermite exact-expression equivalence only."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from rich_gauge_band_frame_20261016_model import geometry
from rich_local_frame_spatial_carrier_20261016_model import field_basis,displacement,CENTRES
from rich_local_frame_spatial_carrier_20261016_angular import rotations
from rich_spatial_frame_photon_20261016_fields import displacement as reference_displacement
from rich_spatial_frame_photon_20261016_kernel import default_modes,segment
ROOT=Path(__file__).resolve().parent
P='rich_local_frame_spatial_carrier_20261016_'


def main():
    points=np.array([[.23,.71,1.2],[2.3,.4,5.7],[5.8,2.7,.7]])
    spin=geometry()['spin'];D=np.broadcast_to(np.eye(8),(len(points),8,8)).astype(complex).copy()
    for a,theta in ((2,points[:,0]),(1,points[:,1]),(2,points[:,2])):
        U=np.cos(theta/2)[:,None,None]*np.eye(8)-2j*np.sin(theta/2)[:,None,None]*spin[a]
        D=np.einsum('qab,qbc->qac',D,U)
    exact=np.array([expm(-1j*a*spin[2])@expm(-1j*b*spin[1])@expm(-1j*c*spin[2]) for a,b,c in points])
    R=rotations(points);a=np.einsum('qab,b->qa',R,CENTRES[1]+.12);b=np.einsum('qab,b->qa',R,CENTRES[3]-.12)
    modes=default_modes();modes[:,3:6]*=np.sinc(.08*np.linalg.norm(modes[:,:3],axis=1)/np.pi)[:,None]
    phi=segment(a,b,modes)[0];basis=field_basis(1)
    current=displacement(basis,-phi);reference=reference_displacement(list(map(tuple,basis)),list(map(tuple,basis)),-phi)
    result=dict(geometry_spin_square_error=max(float(np.linalg.norm(s@s-np.eye(8)/4)) for s in spin),band_D_cos_vs_expm_norm=float(np.linalg.norm(D-exact)),cap1_whole_Hermite_vs_general_Laguerre_norm=float(np.linalg.norm(current-reference)),cap1_max_absolute=float(np.max(abs(current-reference))),scope='Exact kernel computational-expression comparison on actual cube/physical path families; NOT a full-H/state transport claim.')
    np.savez_compressed(ROOT/(P+'kernel_check_raw.npz'),D_cos=D,D_expm=exact,displacement=current,reference=reference)
    (ROOT/(P+'kernel_check_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
