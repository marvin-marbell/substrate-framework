from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
SOURCES=['affine_canonical_lower_20261011.py','affine_canonical_epoch_20261011.py',
         'neutral_return_lower_control_20261011.py','canonical_four_odd_lower_20261011.py',
         'canonical_odd_vector_control_20261011.py','neutral_return_lower_boundary_20261011.py']
PINS={'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json':'102966f32c49f40d79d55343925a630e30a9cf7e4a8074cf8628142ba9c5b6e8',
      'canonical_ground_review_proof_20261011.json':'ea69bfb67bfffc8eb5a9b5a4efbaf0b54edbb9a8ba06839b49c03faa3e60fd65'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    args,extra=parser.parse_known_args();assert args.tag.startswith('affine_canonical_')
    source_pins={name:digest(ROOT/name) for name in SOURCES}
    assert source_pins['neutral_return_lower_control_20261011.py']=='f2a3ff7efce4960ee38a32055e9daf6ae97e465ab23514c7bee206b60a32338f'
    input_pins={name:digest(ROOT/name) for name in PINS};assert input_pins==PINS
    command=[PYTHON,str(ROOT/SOURCES[0]),'--tag',args.tag,*extra]
    pre=ROOT/(args.tag+'_preexecution.zip')
    with zipfile.ZipFile(pre,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [*SOURCES,*PINS]:archive.write(ROOT/name,name)
        archive.writestr('manifest.json',json.dumps({'command':command,'source_sha256':source_pins,
                                                  'input_sha256':input_pins,'old_solver_or_checker_execution':False},indent=2))
    pre.chmod(0o444);start=time.time()
    stdout=ROOT/(args.tag+'_stdout.log');stderr=ROOT/(args.tag+'_stderr.log')
    with stdout.open('xb') as out,stderr.open('xb') as err:
        process=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,
                               env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),check=False)
    outputs=sorted(ROOT.glob(args.tag+'_*matrix.json'))+sorted(ROOT.glob(args.tag+'_*witness.npz'))
    receipt=ROOT/(args.tag+'_receipt.json')
    if receipt.exists():outputs.append(receipt)
    outputs.extend([stdout,stderr])
    natural={'command':command,'natural_exit_code':process.returncode,'signal_exit':process.returncode<0,
             'elapsed_seconds':time.time()-start,'source_sha256':source_pins,'input_sha256':input_pins,
             'sources_unchanged':all(digest(ROOT/n)==v for n,v in source_pins.items()),
             'inputs_unchanged':all(digest(ROOT/n)==v for n,v in input_pins.items()),
             'preexecution_zip_sha256':digest(pre),'output_sha256':{p.name:digest(p) for p in outputs}}
    exit_path=ROOT/(args.tag+'_natural_exit.json')
    with exit_path.open('x') as f:json.dump(natural,f,indent=2);f.write('\n')
    outputs.append(exit_path)
    for path in outputs:path.chmod(0o444)
    executed=ROOT/(args.tag+'_executed.zip')
    with zipfile.ZipFile(executed,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(pre,pre.name)
        for path in outputs:archive.write(path,path.name)
    executed.chmod(0o444)
    print(json.dumps({'tag':args.tag,'natural_exit_code':process.returncode,
                      'elapsed_seconds':natural['elapsed_seconds'],'executed_zip_sha256':digest(executed)}),flush=True)
    sys.exit(process.returncode)


if __name__=='__main__':
    main()
