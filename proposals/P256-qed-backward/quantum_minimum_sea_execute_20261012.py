"""Immutable capture and natural-exit launch of the new parameterized scientific source."""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CONTRACT=Path('/home/administrator/.omp/agent/sessions/-/2026-10-09T16-26-59-293Z_01a1217d-415d-7376-a17d-16be721cbe28/local/issue222-open-box-brillouin-quantum-minimum-contract-20261012.json')
INPUTS={'contract':(CONTRACT,'e672db31e4b0dc8db4c4536b46ef9c883d8ca8077f55a541abc2f8869d2cf5c6'),
'profiles':(ROOT/'neutral_body_parent_primitive_profiles_20261012.json','36f653729d91d0d8c021e708faed651098eb8e1e3e0a062d6f362e114fbf3c72'),
'connected_proof':(ROOT/'connected_quantum_sea_actual2_20261012_proof.json','d151aa935a6dd3d19f18bbd24393bf988c29d51db02ec6843127fd3b7ab1c117'),
'connected_interface':(ROOT/'connected_quantum_sea_actual2_20261012_interface.json','aa91517f56034f78b2258ea0efa3ffa9b99092bd17dbdcdc2f669cec02d2273c'),
'orbit_review':(ROOT/'sea_orbit_review_actual3_20261012_proof.json','59aa71bf0b31e3ae144557f75d6beeb84e357a3634ab20c12bdca1ff467c1dee'),
'energy_review':(ROOT/'sea_energy_review_actual5_20261012_proof.json','a890fd6f0bde4aa291918a04f42046b9f81493656fc2aaa0f190ee9789bf85a3'),
'uniform_class':(ROOT/'neutral_body_parent_uniform_class_proof_20261012.json','7daca5503e6d44c9c718c1b55191053449f45c08ad46f08b838fe732ea0d417f')}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    attempt=sys.argv[1]
    prefix=ROOT/f'quantum_minimum_sea_{attempt}_20261012'
    source=ROOT/'quantum_minimum_sea_source_20261012.py'
    pins={}
    for name,(path,expected) in INPUTS.items():
        actual=digest(path)
        if actual!=expected:raise ValueError(f'input drift {name}: {actual}')
        pins[name]={'path':str(path),'sha256':actual}
    pins['source']={'path':str(source),'sha256':digest(source)}
    pins['executor']={'path':str(Path(__file__).resolve()),'sha256':digest(Path(__file__).resolve())}
    governor={'plan_id':'issue-222-induced-photon-delta13-continuation-20261008','captured_revision':'11c9171134619d7fd92c3acb040ba42d3b4d36ec42819ec809ba2bbdcf8229a4','accepted_science_capture':'2f0bfe8e2ca9c558be5ad3eab0e06e6773692b7b573886a0d4dcead3d2118ea6','template_id':'general','template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe','work_item_id':'induced-photon','requirement_ids':['outcome','verification'],'parent_plan_owner':True}
    request=prefix.with_name(prefix.name+'_request.json')
    request.write_text(json.dumps({'pins':pins,'governor':governor},indent=2)+'\n')
    pre=prefix.with_name(prefix.name+'_preexecution.zip')
    with zipfile.ZipFile(pre,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for pin in pins.values():archive.write(pin['path'],Path(pin['path']).name)
        archive.write(request,request.name)
    command=['/home/administrator/substrate-framework/.venv/bin/python',str(source),'--request',str(request),'--output-prefix',str(prefix)]
    start=time.time()
    result=subprocess.run(command,cwd=ROOT,capture_output=True)
    stdout=prefix.with_name(prefix.name+'_stdout.txt');stderr=prefix.with_name(prefix.name+'_stderr.txt')
    stdout.write_bytes(result.stdout);stderr.write_bytes(result.stderr)
    exitpath=prefix.with_name(prefix.name+'_natural_exit.json')
    exitpath.write_text(json.dumps({'command':command,'returncode':result.returncode,'natural_exit':True,'elapsed_seconds':time.time()-start,'pins':pins,'preexecution_sha256':digest(pre)},indent=2)+'\n')
    executed=prefix.with_name(prefix.name+'_executed.zip')
    outputs=[stdout,stderr,exitpath]
    if result.returncode==0:outputs += [prefix.with_name(prefix.name+'_proof.json'),prefix.with_name(prefix.name+'_interface.json')]
    with zipfile.ZipFile(executed,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for pin in pins.values():archive.write(pin['path'],Path(pin['path']).name)
        for path in [request]+outputs:archive.write(path,path.name)
    receipt={'returncode':result.returncode,'natural_exit':True,'source_sha256':pins['source']['sha256'],'preexecution':{'path':str(pre),'sha256':digest(pre)},'executed':{'path':str(executed),'sha256':digest(executed)},'outputs':{p.name:digest(p) for p in outputs}}
    prefix.with_name(prefix.name+'_delivery.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)
    if result.returncode:print(result.stderr.decode(),file=sys.stderr)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
