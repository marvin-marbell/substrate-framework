"""Fresh-state nonunitary qutrit Wilson-loop witness, not a cnumber substitution."""
import json
from pathlib import Path
import numpy as np
from numba import njit
import rich_bulk_photon_20261016_state as base
import rich_flux_phase_photon_20261016_state as s
ROOT=Path(__file__).resolve().parent
P='rich_flux_phase_photon_20261016_'


@njit(cache=True)
def observe(us,ds,Es,F,G,codes,g,a,b,epsilon,floor):
    out=np.zeros((len(us),3))
    for row in range(len(us)):
        u=us[row];d=ds[row];E=Es[row];env=base.envelope(u,d,E,g,a,b)
        det=np.linalg.det(base.submatrix(F,u,d));gd=np.linalg.det(base.submatrix(G,u,d))
        weight=env*env*(det*det+epsilon*epsilon*gd*gd+floor*floor)
        out[row,0]=det*det*env*env/weight
        for loop in range(len(s.LOOPS)):
            value=0.
            for direction in (-1,1):
                ne=E+direction*s.LOOPS[loop]
                if np.any(np.abs(ne)>1):continue
                sign=1.
                for e in range(192):
                    if s.LOOPS[loop,e]!=0 and codes[e]%2:sign=-sign
                value+=sign*det*det*env*base.envelope(u,d,ne,g,a,b)/weight/2.
            out[row,1 if loop<192 else 2]+=value/(192. if loop<192 else 48.)
    return out


def main():
    cfg=json.loads((ROOT/(P+'input.json')).read_text());results={}
    for pi in (False,True):
        label='pi_pair' if pi else 'zero_pair';F,G,codes,metadata=s.kernel(pi,'pair',cfg['delta']);chains=[];raws=[]
        for seed in (9101691,9101699):
            u,d,E,acc,prop=base.sample(seed,384,18000,160,F,G,*cfg['correlation'],*cfg['guide'])
            raw=observe(u,d,E,F,G,codes,*cfg['correlation'],*cfg['guide']);raws.append(raw)
            chains.append(dict(seed=seed,plaquette=float(raw[:,1].sum()/raw[:,0].sum()),winding=float(raw[:,2].sum()/raw[:,0].sum()),accepted=acc.tolist(),proposed=prop.tolist()))
        raw=np.concatenate(raws);norm=raw[:,0].sum();means=raw.sum(axis=0)/norm
        blocks=np.array([chunk.sum(axis=0)[1:]/chunk[:,0].sum() for chunk in np.array_split(raw,24)])
        np.savez_compressed(ROOT/(P+label+'_quantum_loops_raw.npz'),raw=raw)
        results[label]=dict(plaquette=float(means[1]),winding=float(means[2]),blockSE=(blocks.std(axis=0,ddof=1)/np.sqrt(24)).tolist(),chains=chains,samples=len(raw))
    result=dict(results=results,source='fresh384x2 independent chains each family, same fixed input law/state; new seeds9101691/9101699',operator='Mean (W_square+W_square_dagger)/2 over192 plaquettes and (W_winding+W_winding_dagger)/2 over48 windings; W is PRODUCT OF ACTUAL UNIT QUTRIT SHIFTS, with nilpotent boundary zeros, no phase cnumber eigenstate',meaning='Negative measured pi-family loop expectation is an intrinsic quantum-state witness. Does not establish magnetic stiffness, stationarity, photon or an external B.',natural_scientific_completion=True)
    (ROOT/(P+'quantum_loops.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
