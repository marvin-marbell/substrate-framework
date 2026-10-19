"""NEW generic-q state-level charge-source boundary, independent of author q."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
import rich_spinor_rot_20261014_model as law

def main():
    q=np.array([.43,-.37,.61]);rows=[]
    for L,coupling in [(2,0.),(3,0.),(2,.45),(3,.45)]:
        m=law.solve(law.build(law.Parameters(charged_lmax=L,coupling=coupling,field_cutoff=1)))
        c=m['charged'];density=law.multiplication(c,np.exp(1j*c['r']@q))*m['params'].charge
        rho=np.kron(np.kron(np.kron(density,np.eye(2)),np.eye(len(m['neutral']['basis']))),np.eye(m['field_dimension']))
        j=law.current(m,q,q/np.linalg.norm(q));g=m['vectors'][:,0];e=m['energies'][0]
        residual=m['H']@(rho@g)-e*(rho@g)-np.linalg.norm(q)*(j@g)
        rows.append(dict(L=L,coupling=coupling,fieldcut=1,dimension=len(g),q=q.tolist(),Ward_on_ground=float(np.linalg.norm(residual)),density_source_norm=float(np.linalg.norm(rho@g)),current_source_norm=float(np.linalg.norm(j@g))))
    result=dict(rows=rows,scope='Continuum-before-Galerkin sources are rotationally covariant but finite projection is not exactly closed under local gauge transformations. This state-level finite-q boundary does not turn primitive oscillator cavity into Gauss photon phase. Angular enrichment is actual calculation, not source residual suppression.')
    Path(__file__).with_name('rich_review_wave2_20261014_rot_ward_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
