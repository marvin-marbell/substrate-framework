"""Fresh independent physics from frozen declared F, not any producer trajectory."""
import json
import time
import numpy as np
from typing import Any
from rich_review_wave35_20261016_consume import ROOT,P,run_chain


def main():
    config=json.loads((ROOT/(P+'dimer_input.json')).read_text());frozen=np.load(ROOT/config['source']);F=frozen['F']
    G=F+.02*np.random.default_rng(config['guide_seed']).normal(size=F.shape)
    np.savez_compressed(ROOT/(P+'dimer_declared_input_arrays.npz'),F=F,G=G)
    result: dict[str,Any]
    result=dict(kind='INDEPENDENT_FROZEN_AF_DIMER_INPUT_SAME_PRIMITIVE_H',input=config,F_singular_values=np.linalg.svd(F,compute_uv=False).tolist(),chains=[]);started=time.time()
    for seed in config['seeds']:
        r=run_chain(seed,config,F,G,config['params']);result['chains'].append(r)
        (ROOT/(P+'dimer_findings.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(r),flush=True)
    result['seconds']=time.time()-started;(ROOT/(P+'dimer_findings.json')).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
