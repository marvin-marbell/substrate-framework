"""Actual one-particle CAR hopping/current matrices, contrasted with quantum-field rings."""
import json
import numpy as np
from rich_review_wave35_20261016_geometry import ROOT,P


def main():
    result=[];arrays={}
    for L in (2,4):
        H=np.zeros((L,L));J=np.zeros((L,L),complex)
        for a in range(L):
            b=(a+1)%L;H[b,a]-=1;H[a,b]-=1;J[b,a]-=1j;J[a,b]+=1j
        k=2*np.pi*np.arange(L)/L;phi=np.exp(1j*np.outer(np.arange(L),k))/np.sqrt(L)
        band=phi.conj().T@H@phi;current=phi.conj().T@J@phi
        result.append(dict(L=L,dispersion=band.diagonal().real.tolist(),current_eigenvalues=current.diagonal().real.tolist(),current_frobenius=float(np.linalg.norm(J)),band_offdiagonal=float(np.linalg.norm(band-np.diag(band.diagonal()))),current_offdiagonal=float(np.linalg.norm(current-np.diag(current.diagonal())))))
        arrays[f'H{L}']=H;arrays[f'J{L}']=J;arrays[f'phi{L}']=phi
    assert result[0]['current_frobenius']==0 and result[1]['current_frobenius']>2
    np.savez_compressed(ROOT/(P+'clamped_raw.npz'),**arrays)
    (ROOT/(P+'clamped_findings.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
