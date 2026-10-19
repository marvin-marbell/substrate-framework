"""Quantify full-source versus Galerkin dynamical continuity; never hide cut terms."""
import json
from pathlib import Path
import numpy as np
import rich_coupled_frame_photon_20261016_model as law

ROOT=Path(__file__).resolve().parent;P='rich_review_wave45_20261016_'


def main():
    result=dict(kind='PHYSICAL_SOURCE_GALERKIN_CONTINUITY_COMPLEMENT',branches=[])
    for number in (5,3):
        m=law.Model(number);raw=np.load(ROOT/(P+f'coupled_N{number}_raw.npz'));v,_=m.source_vertices();J=[-m.lift(a) for a in v];records=[]
        for key in ('initial','final0','final1'):
            x=raw[key];res=[];source=[];compressed=[]
            for axis in range(3):
                E=m.electric(axis);dotE=1j*(m.H@(E@x)-E@(m.H@x));jx=J[axis]@x
                res.append(float(np.linalg.norm(dotE-jx)));source.append(float(np.linalg.norm(jx)));compressed.append(float(np.linalg.norm(dotE)))
            records.append(dict(state=key,full_source_current_norm=source,projected_dotE_norm=compressed,full_source_minus_projected_dotE_norm=res))
        result['branches'].append(dict(number=number,records=records))
    result['interpretation']='Full unbounded law has dotE=J. Nonzero retained-operator residual is the explicit angular/field regulator complement, not a license to relabel graph flow as physical lab current. Main own dynamics are finite Galerkin, not full unbounded convergence.'
    (ROOT/(P+'coupled_continuity_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
