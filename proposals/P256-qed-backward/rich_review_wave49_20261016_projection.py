"""Independent mathematical equivalence check for construction-only Gram-BLAS regrouping."""
from pathlib import Path
import json
import time
import numpy as np
import rich_gauge_band_frame_20261016_model as band
import rich_spatial_frame_photon_20261016_kernel as kernel
import rich_spatial_frame_photon_20261016_fields as fields

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def run():
    a=band.angular((0,1,2));F=a['F'];w=a['weights'];R=len(F[0]);basis=fields.basis(1);d=len(basis)
    # Genuine finite-q path multiplication, not random or mock solved output.
    from rich_spatial_frame_photon_20261016_model import rotations
    rot=rotations(a['points']);start=np.einsum('qij,j->qi',rot,[.12,-.12,.12]);end=np.einsum('qij,j->qi',rot,[1.18,.12,-.12])
    modes=kernel.default_modes();phi=kernel.segment(start,end,modes)[0]
    values=fields.displacement(basis,basis,-phi)
    before=time.perf_counter();old=np.einsum('qi,q,qab,qj->iajb',F.conj(),w,values,F,optimize=True).reshape(R*d,R*d);oldtime=time.perf_counter()-before
    before=time.perf_counter();G=np.einsum('qi,q,qj->qij',F.conj(),w,F).reshape(len(w),R*R).T
    new=(G@values.reshape(len(w),d*d)).reshape(R,R,d,d).transpose(0,2,1,3).reshape(R*d,R*d);newtime=time.perf_counter()-before
    error=float(np.linalg.norm(new-old));relative=error/max(float(np.linalg.norm(old)),1e-14)
    assert relative<1e-12,relative
    result=dict(kind='OWN_FULL_R2_FIELD1_EXACT_CONSTRUCTION_CONTRACTION_CHECK',angular_dimension=R,field_dimension=d,quadrature_count=len(w),absolute_error=error,relative_error=relative,old_seconds=oldtime,Gram_BLAS_seconds=newtime,limits='Exactly same Haar basis, pointwise finite-q path displacement and RF bra/ket ordering; only contraction grouping changes. This is not a state evolution, reduced-cap proposal, quadrature-convergence proof or substitute for own integrated47 consumer.')
    np.savez_compressed(ROOT/(P+'projection_raw.npz'),original=old,Gram_BLAS=new,path_flux=phi,field_basis=np.array(basis))
    (ROOT/(P+'projection_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
