"""Additional actual cold-state Trotter/seed fork using the same sampler/law."""
import json
from pathlib import Path
import numpy as np
from rich_collective_photon_response_20261016_run import sample, summarize

ROOT=Path(__file__).resolve().parent
P='rich_collective_photon_response_20261016_'


def main():
    entries=[]
    for beta,M,seed in ((4.,32,46204),(8.,32,46205)):
        arrays,meta=sample(beta,M,seed,steps=1800,burn=600,stride=60,stabilized=True)
        name=f'b{beta:g}_M{M}_s{seed}';np.savez_compressed(ROOT/(P+name+'_raw.npz'),**arrays)
        keys=('source_hessian','logdet_hessian','source_one','interval_contact','interval_current_matsubara','EE_matsubara','EE_tau',
              'qminus_EE_matsubara','qminus_EE_tau','direct_EE_matsubara','direct_EE_tau','derivative_step_difference')
        entry=dict(name=name,metadata=meta,observables={k:summarize(arrays[k]) for k in keys},
                   max_source_matrix_condition=float(arrays['max_source_matrix_condition'].max()),
                   max_transverse_divergence=float(arrays['transverse_divergence'].max()),
                   maximum_temporal_winding=float(abs(arrays['compact_temporal_winding']).max()),
                   selected_step_halving_exercised=int(arrays['step_comparison_exercised'].sum()))
        (ROOT/(P+name+'_result.json')).write_text(json.dumps(entry,indent=2)+'\n');entries.append(entry)
        (ROOT/(P+'coldrefine_result.json')).write_text(json.dumps(dict(cases=entries),indent=2)+'\n')
        print(json.dumps(dict(name=name,samples=meta['samples'],elapsed=meta['elapsed_seconds'],accepts=meta['accepts'],
                             D_E_n1=(meta['dt']*arrays['EE_matsubara'][:,:,1]).mean(axis=0).tolist())),flush=True)


if __name__=='__main__':main()
