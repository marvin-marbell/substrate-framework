"""New owning stationary solve at I16, starting from a fresh singlet seed."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse.linalg import eigsh
from rich_open_gauss_carrier_20261015_model import NeutralCube, LAW_ID
ROOT = Path(__file__).resolve().parent
P = 'rich_open_gauss_carrier_20261015_'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    start = time.monotonic()
    config = json.loads((ROOT/(P+'input.json')).read_text())
    model = NeutralCube()
    energies, vectors = eigsh(model.H,k=1,which='SA',v0=model.seed(),tol=2e-12,maxiter=3000,ncv=30)
    psi = vectors[:,0]
    residual = float(np.linalg.norm(model.H@psi-energies[0]*psi))
    hashes = dict(model_SHA256=sha(ROOT/(P+'model.py')),input_SHA256=sha(ROOT/(P+'input.json')),basis_SHA256=model.basis_hash)
    np.savez_compressed(ROOT/(P+'neutral_reference.npz'),psi=psi,energy=energies[0],residual=residual,fock=model.fock,old_indices=model.old_indices,electric=model.electric,**hashes)
    result = dict(law_id=LAW_ID,dimension=model.dimension,H_nnz=model.H.nnz,energy=float(energies[0]),residual=residual,reference_scope='Conditional seed-start stationary neutral reference; not global ground or vacuum theorem',preparation='Fresh four-singlet seed, no historical eigendata',observables=model.measure(psi),hashes=hashes,elapsed_seconds=time.monotonic()-start,config=config)
    (ROOT/(P+'reference_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    main()
