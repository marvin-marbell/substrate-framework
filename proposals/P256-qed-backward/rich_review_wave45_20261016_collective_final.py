"""Re-evaluate only revision-invalidated vertices on own physical V64 paths."""
import json
from pathlib import Path
import numpy as np
import rich_unitary_rotor_photon_20261016_thermal as r
import rich_collective_photon_response_20261016_model as model

ROOT=Path(__file__).resolve().parent;P='rich_review_wave45_20261016_'


def main():
    pos,arcs,D=r.geometry(4);result=dict(kind='OWN_PHYSICAL_PATH_FINALIZED_INTERVAL_ESTIMATOR',cases=[])
    for beta,M,seed in [(1.,4,45101),(2.,8,45102),(1.,8,45103)]:
        name=f'collective_b{beta:g}_M{M}_s{seed}';raw=np.load(ROOT/(P+name+'_raw.npz'));rows=[];weight_error=0.
        for i,(theta,alpha,s) in enumerate(zip(raw['theta'],raw['alpha'],raw['HS'])):
            data=r.path_data(theta,alpha,s,arcs,D,beta)
            weight_error=max(weight_error,abs(data['logweight']-float(raw['logweight'][i])))
            rows.append(model.interval_response(theta,alpha,s,data,pos,arcs,D,beta))
        arrays={k:np.array([z[k] for z in rows]) for k in rows[0]}
        np.savez_compressed(ROOT/(P+name+'_final_response.npz'),**arrays)
        def summary(key):
            x=arrays[key];means=np.array([z.mean(axis=0) for z in np.array_split(x,5)]);mu=x.mean(axis=0)
            return dict(mean_real=mu.real.tolist(),mean_imag=mu.imag.tolist(),five_block_SE=(means.std(axis=0,ddof=1)/np.sqrt(5)).tolist())
        entry=dict(name=name,beta=beta,M=M,dt=beta/M,seed=seed,samples=len(rows),own_physical_input_weight_error=float(weight_error),
            EE_matsubara=summary('EE_matsubara'),EE_tau=summary('EE_tau'),direct_EE_matsubara=summary('direct_EE_matsubara'),
            source_hessian=summary('source_hessian'),logdet_hessian=summary('logdet_hessian'),interval_contact=summary('interval_contact'),interval_current_matsubara=summary('interval_current_matsubara'),
            invalidated_positive_n1_difference=float(np.max(abs(arrays['EE_matsubara'][:,:,1]-raw['EE_matsubara'][:,:,1]))),
            invalidated_all_n_difference=float(np.max(abs(arrays['EE_matsubara']-raw['EE_matsubara']))),
            qminus_swap_error=float(np.max(abs(arrays['qminus_EE_matsubara']-arrays['EE_matsubara'][:,:,(-np.arange(M))%M]))),
            per_path_parity_difference=float(np.max(abs(arrays['EE_matsubara']-arrays['EE_matsubara'][:,:,(-np.arange(M))%M]))),
            compact_temporal_winding_max=float(np.max(abs(arrays['compact_temporal_winding']))),max_source_matrix_condition=float(arrays['max_source_matrix_condition'].max()),
            limits='Own freshly generated physical inputs preserved in original raw arrays. Only finalized interval estimator recomputed; all n independently, complex tau retained. Short correlated paths; block error not equilibration. No photon-pole or irreducible-stiffness claim.')
        result['cases'].append(entry)
        (ROOT/(P+'collective_final_result.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:entry[k] for k in ('name','own_physical_input_weight_error','invalidated_positive_n1_difference','invalidated_all_n_difference','per_path_parity_difference')}),flush=True)


if __name__=='__main__':main()
