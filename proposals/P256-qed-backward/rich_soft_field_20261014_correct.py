"""Natural correction capsule for decision-changing degenerate eigenspaces."""
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
    from rich_soft_field_20261014_model import Parameters, Joint
    from rich_soft_field_20261014_density import DensityJoint
    from rich_soft_field_20261014_calculate import summarize
    request=json.loads(Path('rich_soft_field_20261014_input.json').read_text())
    p=Parameters(**{**request['parameters'],'shape':tuple(request['parameters']['shape'])})
    cases=[]
    for joint,label in [(Joint(replace(p,halfwidth=3*np.pi),1),'orthogonal_actual_box3pi'),
                        (DensityJoint(replace(p,shape=(64,4,8),count=8),1,suppliers=2,retain=33),'orthogonal_two_supplier_complete_shell33')]:
        result=summarize(joint,label)
        gaps=np.array(result['gaps'])
        groups=[];start=0
        while start<len(gaps):
            stop=start+1
            while stop<len(gaps) and gaps[stop]-gaps[start]<1e-8:stop+=1
            groups.append({'gap':float(np.mean(gaps[start:stop])),'columns':list(range(start,stop)),
                           'weights':{name:float(sum(weights[start:stop])) for name,weights in result['weights'].items()}})
            start=stop
        result['orthonormal_eigenspace_spectral_sums']=groups
        cases.append(result)
    Path('rich_soft_field_20261014_correct_actual1_result.json').write_text(json.dumps({'cases':cases,
        'correction':'Selected ARPACK columns orthonormalized by QR and Rayleigh-Ritz in microscopic H. Degenerate groups summed, not independently counted nonorthogonal columns. Historical actual1/rawrefine energies remain useful; their degenerate column weights withdrawn.'},indent=2))


def capture():
    root=Path(__file__).resolve().parent;prefix='rich_soft_field_20261014_'
    paths=[root/(prefix+n) for n in ['model.py','density.py','calculate.py','correct.py','input.json']]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={p.name:sha(p) for p in paths};command=[sys.executable,str(Path(__file__).resolve()),'calculate']
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.time()
    with (root/(prefix+'correct_actual1_stdout.txt')).open('wb') as out,(root/(prefix+'correct_actual1_stderr.txt')).open('wb') as err:
        process=subprocess.run(command,cwd=root,env=environment,stdout=out,stderr=err)
    outputs=[root/(prefix+'correct_actual1_'+n) for n in ['stdout.txt','stderr.txt','result.json']]
    record={'command':command,'natural_exit':True,'exit_code':process.returncode,'elapsed_seconds':time.time()-start,
            'sources_before':before,'sources_after':{p.name:sha(p) for p in paths},'outputs':{p.name:sha(p) for p in outputs if p.exists()},'python':sys.version}
    exitpath=root/(prefix+'correct_actual1_exit.json');exitpath.write_text(json.dumps(record,indent=2))
    with zipfile.ZipFile(root/(prefix+'correct_actual1_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths+outputs+[exitpath]:
            if p.exists():z.write(p,p.name)
    print(json.dumps(record,indent=2));sys.exit(process.returncode)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='calculate':calculate()
    else:capture()
