"""Independent retained-Wigner API, LEFT and bounded-polynomial positive controls."""
from pathlib import Path
import json
import numpy as np
import rich_gauge_band_frame_20261016_model as band
import rich_local_frame_spatial_carrier_20261016_model as carrier
import rich_spatial_frame_photon_20261016_quadrature as shared
import rich_spatial_frame_photon_20261016_kernel as kernel
from rich_spatial_frame_photon_20261016_model import rotations
from rich_review_wave49_20261016_cubature import material_grid,block

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def run():
    supplied=shared.material(2);independent=material_grid(2,24,16,24)
    F=supplied['F'];w=supplied['weights'];rot=rotations(supplied['points']);old=band.angular((0,1,2))
    values_error=float(np.max(abs(F-independent['F'])));points_error=float(np.max(abs(supplied['points']-independent['points'])))
    rotations_error=0.;projected=[]
    for i in range(3):
        row=[]
        for j in range(3):
            actual=F.conj().T@(w[:,None]*rot[:,i,j,None]*F);row.append(actual)
            rotations_error=max(rotations_error,float(np.max(abs(actual-old['O'][i][j].toarray()))))
        projected.append(row)
    eps=np.zeros((3,3,3));eps[0,1,2]=eps[1,2,0]=eps[2,0,1]=1;eps[0,2,1]=eps[2,1,0]=eps[1,0,2]=-1
    left_error=0.
    for axis in range(3):
        L=supplied['L'][axis].toarray()
        for i in range(3):
            for j in range(3):
                expected=1j*sum(eps[axis,i,k]*projected[k][j] for k in range(3))
                left_error=max(left_error,float(np.linalg.norm(L@projected[i][j]-projected[i][j]@L-expected)))
    body=(carrier.CENTRES[:,None,:]+.24*(band.geometry()['sites'][None,:,:]-.5)).reshape(32,3)
    Q=np.zeros(32);Q[0]=-1;Q[10]=1;Q[11]=-1
    zero=kernel.default_modes();zero[:,:3]=0
    uniform_old,_=block(old,body,Q,zero);uniform_new,_=block(supplied,body,Q,zero)
    uniform_error=float(np.linalg.norm(uniform_old-uniform_new))
    assert values_error<1e-12 and points_error<1e-12 and rotations_error<1e-12 and left_error<1e-11 and uniform_error<1e-11
    assert supplied['labels']==old['labels']
    g=band.angular((0,.5,1,1.5,2));own=np.load(ROOT/(P+'local_primitives_raw.npz'))
    assert np.array_equal(np.array(g['labels']),own['g_labels'])
    result=dict(kind='OWN_SHARED_INTEGER_SO3_MATERIAL_API_REVIEW',retained_labels_unchanged=True,integer_R_F_value_error=values_error,points_error=points_error,linear_R_operator_error=rotations_error,physical_LEFT_vector_commutator_error=left_error,old_uniform_q0_fullproduct_positive_control_error=uniform_error,unchanged_SU2_sewn_g_domain=dict(alpha_period='4pi',labels_equal_own55g_audit=True,Gram_error=g['gram_error']),limits='Only integer material SO3 uses independently oversampled2pi grid. Half-integer/sewn SU2 g is NOT globally switched. Old bounded-degree uniform kernels pass this positive control; demonstrated finite-q alias does not invalidate all old polynomial laws. Sample refinement does not establish angular exterior/continuum/photon closure.')
    np.savez_compressed(ROOT/(P+'quadrature_api_raw.npz'),shared_F=F,independent_F=independent['F'],points=supplied['points'],weights=w,uniform_old=uniform_old,uniform_new=uniform_new)
    (ROOT/(P+'quadrature_api_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
