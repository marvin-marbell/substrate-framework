"""Same independently contracted primitive H on a materially different orbital state."""
import json
import time
import numpy as np
from typing import Any
from rich_review_wave35_20261016_consume import ROOT,P,declared,run_chain


def main():
    config=json.loads((ROOT/(P+'AF_input.json')).read_text());F,_,meta=declared.kernel(**config['kernel'])
    G=F+.02*np.random.default_rng(config['guide_seed']).normal(size=F.shape)
    np.savez_compressed(ROOT/(P+'AF_declared_input_arrays.npz'),F=F,G=G)
    result: dict[str,Any]
    result=dict(kind='INDEPENDENT_MATERIAL_AF_STATE_SAME_PRIMITIVE_H_NOT_A_SINGLET_CLAIM',input=config,kernel_metadata=meta,chains=[]);started=time.time()
    for seed in config['seeds']:
        r=run_chain(seed,config,F,G,config['params']);result['chains'].append(r)
        (ROOT/(P+'AF_findings.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(r),flush=True)
    result['seconds']=time.time()-started;(ROOT/(P+'AF_findings.json')).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
