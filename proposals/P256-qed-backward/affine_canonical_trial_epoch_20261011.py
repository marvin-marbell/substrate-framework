from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

from affine_canonical_epoch_20261011 import ROOT, PYTHON, SOURCES, PINS, digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('--receipt',required=True)
    args=parser.parse_args();assert args.tag.startswith('affine_canonical_')
    input_receipt=json.loads((ROOT/args.receipt).read_text())
    inputs=[args.receipt,*PINS]
    for family in input_receipt['families']:
        if family['proven_lower_form_failure']:
            for prefix in ('matrix','witness'):
                name=family[prefix+'_path'];assert digest(ROOT/name)==family[prefix+'_sha256'];inputs.append(name)
    sources=[*SOURCES,Path(__file__).name,'affine_canonical_trial_20261011.py']
    sp={n:digest(ROOT/n) for n in sources};ip={n:digest(ROOT/n) for n in inputs}
    command=[PYTHON,str(ROOT/'affine_canonical_trial_20261011.py'),'--tag',args.tag,'--receipt',args.receipt]
    pre=ROOT/(args.tag+'_preexecution.zip')
    with zipfile.ZipFile(pre,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [*sources,*inputs]:archive.write(ROOT/name,name)
        archive.writestr('manifest.json',json.dumps({'command':command,'source_sha256':sp,'input_sha256':ip},indent=2))
    pre.chmod(0o444);start=time.time()
    stdout=ROOT/(args.tag+'_stdout.log');stderr=ROOT/(args.tag+'_stderr.log')
    with stdout.open('xb') as out,stderr.open('xb') as err:
        process=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,
                               env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),check=False)
    outputs=[stdout,stderr,*sorted(ROOT.glob(args.tag+'_*trial.npz'))]
    receipt=ROOT/(args.tag+'_receipt.json')
    if receipt.exists():outputs.append(receipt)
    natural={'command':command,'natural_exit_code':process.returncode,'signal_exit':process.returncode<0,
             'elapsed_seconds':time.time()-start,'source_sha256':sp,'input_sha256':ip,
             'sources_unchanged':all(digest(ROOT/n)==v for n,v in sp.items()),
             'inputs_unchanged':all(digest(ROOT/n)==v for n,v in ip.items()),
             'preexecution_zip_sha256':digest(pre),'output_sha256':{p.name:digest(p) for p in outputs}}
    exit_path=ROOT/(args.tag+'_natural_exit.json')
    with exit_path.open('x') as f:json.dump(natural,f,indent=2)
    outputs.append(exit_path)
    for path in outputs:path.chmod(0o444)
    executed=ROOT/(args.tag+'_executed.zip')
    with zipfile.ZipFile(executed,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(pre,pre.name)
        for path in outputs:archive.write(path,path.name)
    executed.chmod(0o444)
    print(json.dumps({'tag':args.tag,'natural_exit_code':process.returncode,'elapsed_seconds':natural['elapsed_seconds'],
                      'executed_zip_sha256':digest(executed)}),flush=True)
    raise SystemExit(process.returncode)


if __name__=='__main__':
    main()
