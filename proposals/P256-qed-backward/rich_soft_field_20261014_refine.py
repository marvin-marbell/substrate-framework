"""Retained-matter sensitivity and complete-shell density fork, exact capsule."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def calculate():
    import numpy as np
    from dataclasses import replace
    from rich_soft_field_20261014_model import Parameters, Matter
    from rich_soft_field_20261014_density import DensityJoint
    from rich_soft_field_20261014_calculate import summarize
    request=json.loads(Path('rich_soft_field_20261014_refine_input.json').read_text())
    p=Parameters(**{**request['parameters'],'shape':tuple(request['parameters']['shape'])})
    energies=np.linalg.eigvalsh(Matter(p).operators(0.)[0].toarray())
    cutoff=energies[request['requested_retention']-1]
    retain=int(np.sum(energies<cutoff+1e-9))
    cases=[]
    cases.append(summarize(DensityJoint(p,1,suppliers=1,retain=retain),f'one_supplier_complete_shell{retain}'))
    pp=replace(p,shape=tuple(request['density_preserving_shape']))
    for n in request['q_indices']:
        cases.append(summarize(DensityJoint(pp,n,suppliers=2,retain=retain),f'two_supplier_complete_shell{retain}'))
    result={'input':request,'retained_actual':retain,'energies':energies.tolist(),'cases':cases,
            'scope':'Exact full products of retained complete-shell microscopic matter states; source model uses all64 single-cluster states, so this truncation is assessed against original full64. No fitted oscillator frequency or velocity.'}
    Path('rich_soft_field_20261014_refine_actual1_result.json').write_text(json.dumps(result,indent=2))


def capture():
    root=Path(__file__).resolve().parent; prefix='rich_soft_field_20261014_'
    names=['model.py','calculate.py','density.py','refine.py','refine_input.json']
    paths=[root/(prefix+n) for n in names]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={p.name:sha(p) for p in paths}
    command=[sys.executable,str(Path(__file__).resolve()),'calculate']
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.time()
    with (root/(prefix+'refine_actual1_stdout.txt')).open('wb') as out,(root/(prefix+'refine_actual1_stderr.txt')).open('wb') as err:
        process=subprocess.run(command,cwd=root,env=environment,stdout=out,stderr=err)
    outputs=[root/(prefix+'refine_actual1_'+n) for n in ['stdout.txt','stderr.txt','result.json']]
    record={'command':command,'exit_code':process.returncode,'natural_exit':True,'elapsed_seconds':time.time()-start,
            'sources_before':before,'sources_after':{p.name:sha(p) for p in paths},'outputs':{p.name:sha(p) for p in outputs if p.exists()},'python':sys.version}
    exitpath=root/(prefix+'refine_actual1_exit.json');exitpath.write_text(json.dumps(record,indent=2))
    with zipfile.ZipFile(root/(prefix+'refine_actual1_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths+outputs+[exitpath]:
            if p.exists():z.write(p,p.name)
    print(json.dumps(record,indent=2));sys.exit(process.returncode)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='calculate':calculate()
    else:capture()
