"""Fresh site2 CAR/tree-dressed charge packet and actual middle-cut comparator."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
import rich_spatial_20261014_model as law

ROOT=Path(__file__).resolve().parent

def main():
    rows=[]; times=np.linspace(0.,5.2,53)
    for name,hop,rail,cut in [('connected',1.,1.,False),('outer_rails_removed',1.,0.,False),('middle_cut',1.,1.,True),('no_motion',0.,1.,False)]:
        s=law.Sector(law.Parameters(cells=3,hopping=hop,rail_hopping=rail),1,0)
        if cut:
            for edge in (3,4,5):
                s.H-=s.transfers[edge]+s.transfers[edge].T
                s.currents[edge]=sparse.csr_matrix(s.H.shape,dtype=complex)
        g,metadata=law.physical_charge_source(s,2)
        waves=expm_multiply(-1j*s.H,g,start=0.,stop=5.2,num=len(times),traceA=-1j*float(s.H.diagonal().sum()))
        prob=abs(waves)**2; cells=(prob@s.occupation).reshape(len(times),4,3).sum(axis=2)
        currents=np.array([[np.vdot(w,j@w).real for j in s.currents] for w in waves])
        energy=np.array([np.vdot(w,s.H@w).real for w in waves]); electric=prob@s.electric_energy
        rows.append(dict(name=name,parameters=vars(s.p),dimension=s.dimension,source=metadata,
                         receiver_last_cell_max=float(np.max(cells[:,3])),receiver_cell2_max=float(np.max(cells[:,2])),
                         initial_last_cell=float(cells[0,3]),electric_energy_range=float(np.ptp(electric)),
                         norm_error=float(np.max(abs(prob.sum(axis=1)-1))),energy_drift=float(np.ptp(energy)),
                         middle_total_current_max=float(np.max(abs(currents[:,3:6].sum(axis=1)))),
                         density=cells.tolist()))
    result=dict(source_site=2,times=times.tolist(),rows=rows,scope='Static anchor -charge and actual tree electric source; no bare charged creation allowed. Entire admitted one-charge/field state evolved, no frozen link comparator or massless speed claim.')
    (ROOT/'rich_review_wave3_20261014_charge_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({r['name']:{k:v for k,v in r.items() if k!='density'} for r in rows}),flush=True)

if __name__=='__main__':main()
