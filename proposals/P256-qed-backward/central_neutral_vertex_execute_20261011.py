"""Archive immutable inputs and naturally execute only the NEW central consumer."""
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
import shared_star_dynamic_scientist_20261011 as frozen

ROOT=Path(__file__).resolve().parent
PREFIX='central_neutral_vertex_'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,data):path.write_text(json.dumps(data,indent=2)+'\n')

def run(epoch):
    files=set(frozen.INPUTS)
    files.update(['shared_star_dynamic_scientist_20261011.py','shared_star_dynamic_proof_20261011.json',
                  'shared_star_dynamic_actual2_20261011_receipt.json','reference_h0_mathreview_verdict_20261011.json',
                  'charged_cell_heat_proof_20261011.json','charged_outer_vector_interface_20261011.json',
                  'charged_outer_vector_proof_20261011.json','charged_outer_vector_actual1_20261011_receipt.json',
                  'charged_outer_vector_scientist_20261011.py','charged_outer_vector_actual1_20261011_exit.json',
                  'central_neutral_vertex_actual1_20261011_invalidation.json',
                  'star_outer_projection_parent_typed2_20261011_receipt.json','central_neutral_vertex_proof_20261011.json',
                  'central_neutral_vertex_scientist_20261011.py','central_neutral_vertex_execute_20261011.py',
                  'central_neutral_vertex_language_20261011.json'])
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            path=Path(file).resolve()
            if path.parent==ROOT and path.suffix=='.py':files.add(path.name)
    expected={**frozen.native.PINS,
              'shared_star_dynamic_scientist_20261011.py':'2f02fee85495c4ea6ad6b687add5e0ebf1a9fb975c01205bc82f3c98d3c29fa3',
              'shared_star_dynamic_actual2_20261011_receipt.json':'3f690a820fbc84095804d0470386911dd8a9298c396fb9f2c915da870fdfe789'}
    pins={name:sha(ROOT/name) for name in sorted(files)}
    for name,pin in expected.items():
        if pins[name]!=pin:raise ArithmeticError(('frozen input changed',name))
    request={'kind':'NEW_ACTUAL_CENTRAL_CONSUMER_REQUEST','epoch':epoch,'input_sha256':pins,
             'plan':{'id':'issue-222-induced-photon-delta13-continuation-20261008','revision':'c6e1ec4424d6c94ad36f5ceca2d5bd72fbbff3ec25d964a0c15a2fe9e565f060',
                     'accepted_science':'2ec1fe2f4ebfad5032c59dd9ed0d6ef3a57d15e58247f175c670c5034c7d6208','template':'general',
                     'template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe','work_item':'induced-photon'},
             'command':[PYTHON,'central_neutral_vertex_scientist_20261011.py','--epoch',epoch],
             'no_old_cli_solver_root_checker':True,'caps':{'external_deadline_seconds':900,'OMP_BLAS_THREADS':1}}
    req=ROOT/f'{PREFIX}{epoch}_request.json';dump(req,request)
    pre=ROOT/f'{PREFIX}{epoch}_preexecution.zip'
    with zipfile.ZipFile(pre,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):archive.write(ROOT/name,name)
        archive.write(req,req.name)
    env=os.environ.copy();env.update({name:'1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    start=time.monotonic();natural=True
    out=ROOT/f'{PREFIX}{epoch}_stdout.txt';err=ROOT/f'{PREFIX}{epoch}_stderr.txt'
    with out.open('wb') as stdout,err.open('wb') as stderr:
        try:
            completed=subprocess.run(request['command'],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=900,check=False)
            code=completed.returncode
        except subprocess.TimeoutExpired:
            natural=False;code=None
    result={'kind':'NATURAL_EXIT_AND_IMMUTABLE_CAPSULE','epoch':epoch,'natural_exit':natural,'exit_code':code,
            'elapsed_seconds':time.monotonic()-start,'preexecution_zip_sha256':sha(pre),
            'request_sha256':sha(req),'input_sha256':pins,
            'inputs_unchanged':all(sha(ROOT/name)==pin for name,pin in pins.items()),'stdout_sha256':sha(out),'stderr_sha256':sha(err)}
    exitfile=ROOT/f'{PREFIX}{epoch}_exit.json';dump(exitfile,result)
    executed=ROOT/f'{PREFIX}{epoch}_executed.zip'
    with zipfile.ZipFile(executed,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):archive.write(ROOT/name,name)
        for path in (req,out,err,exitfile,ROOT/f'{PREFIX}{epoch}_receipt.json',ROOT/f'{PREFIX}interface_20261011.json'):
            if path.exists():archive.write(path,path.name)
    delivery={**result,'executed_zip_sha256':sha(executed)}
    dump(ROOT/f'{PREFIX}{epoch}_capsule.json',delivery)
    print(json.dumps(delivery),flush=True)
    if not natural or code!=0:raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--epoch',required=True)
    run(parser.parse_args().epoch)
