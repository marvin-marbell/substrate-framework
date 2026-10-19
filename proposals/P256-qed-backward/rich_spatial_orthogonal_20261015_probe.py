"""Small-stage cause discrimination, NOT a rerun of reported whole-H skew."""
from typing import Any
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.linalg import svdvals
from rich_spatial_translation_20261015_model import build
from rich_spatial_collective_20261015_model import multiply_except
ROOT=Path(__file__).resolve().parent


def main()->None:
    started=time.time();config:Any=json.loads(Path(sys.argv[1]).read_text());law:Any=json.loads((ROOT/config['law_input']).read_text());model:Any=build(law)
    grams=model['grams'];raw=np.zeros((16,33,33),complex);local=[]
    for b,block in enumerate(model['blocks']):
        v=block['v0'];f=np.array([v.conj().T@(op@v) for op in block['operators'][6]])
        local.append(dict(block=b,single_particle_owning_skew=[float(np.linalg.norm(h-h.conj().T)) for h in model['local_components'][b]],
            full_local_Fock_operator_skew=[float(np.linalg.norm((op-op.conj().T).data)) for op in block['operators'][6]],
            local_raw_projection_skew=[float(np.linalg.norm(h-h.conj().T)) for h in f]))
        raw+=f*multiply_except(grams,[b])[None]
    x=model['neutral']['whiteners'][0];xl=x.astype(np.clongdouble);stages=[]
    for c,f in enumerate(raw):
        old=model['neutral']['matrices'][0][c];fl=f.astype(np.clongdouble)
        high=xl.conj().T@fl@xl;amplified=xl.conj().T@(fl-fl.conj().T)@xl
        stages.append(dict(component=c,raw_norm=float(np.linalg.norm(f)),raw_skew=float(np.linalg.norm(f-f.conj().T)),
            original_sandwich_skew=float(np.linalg.norm(old-old.conj().T)),
            same_raw_extended_accumulation_skew=float(np.linalg.norm(np.asarray(high-high.conj().T,complex))),
            independently_accumulated_amplified_raw_skew=float(np.linalg.norm(np.asarray(amplified,complex))),
            double_vs_extended_sandwich_difference=float(np.linalg.norm(old-np.asarray(high,complex)))))
    ranks=[]
    for b,block in enumerate(model['blocks']):
        entry={}
        for key in ['v0','vp','vm']:
            s=svdvals(block[key]);entry[key]=dict(rank_at_2e_minus14=int(np.count_nonzero(s>2e-14*s[0])),singular_values=s.tolist())
        ranks.append(dict(block=b,frames=entry))
    result=dict(kind='CAUSE_STAGE_PROBE_NOT_WHOLE_H_RECONFIRMATION',input=config,reported_ground_truth='Reviewer whole2025 neutral Frobenius skew5.617985676e-4 accepted, not rebuilt/retested here.',
        local_stages=local,global_Gram_skew=float(np.linalg.norm(model['neutral']['grams'][0]-model['neutral']['grams'][0].conj().T)),
        whiten_map_operator_norm=float(np.linalg.norm(x,2)),components=stages,local_physical_column_ranks=ranks,
        hypothesis_test='A real conjugation/assembly defect predicts large owning/local raw skew; amplification predicts roundoff-size raw skew with large Cdag(delta raw)C and sandwich rounding. Extended accumulation is diagnostic only, NOT repaired H or posthoc Hermitian averaging.',elapsed_seconds=time.time()-started)
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(elapsed=result['elapsed_seconds'],whiten_map_operator_norm=result['whiten_map_operator_norm'],components=stages,
        local_ranks=[{k:v['rank_at_2e_minus14'] for k,v in r['frames'].items()} for r in ranks])),flush=True)


if __name__=='__main__':main()
