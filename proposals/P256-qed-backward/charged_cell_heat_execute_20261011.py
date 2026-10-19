"""Exclusive immutable preexecution/executed capsule for NEW charged heat CLI."""
import argparse,hashlib,json,subprocess,sys,time,zipfile
from pathlib import Path
import charged_cell_heat_scientist_20261011 as science
ROOT=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def exclusive_json(path,value):
    with path.open('x') as stream: json.dump(value,stream,indent=2);stream.write('\n')

def run(epoch):
    stem=f'charged_cell_heat_{epoch}'
    inputs=list(dict.fromkeys([*science.INPUTS,'charged_cell_heat_scientist_20261011.py','charged_cell_heat_helper_20261011.py','charged_cell_heat_execute_20261011.py','charged_cell_heat_proof_20261011.json','charged_cell_heat_language_20261011.json']))
    pins={name:sha(ROOT/name) for name in inputs}
    command=[sys.executable,str(ROOT/'charged_cell_heat_scientist_20261011.py'),'--epoch',epoch]
    request={'kind':'NEW_CHARGED_HEAT_IMMUTABLE_EXECUTION_REQUEST','command':command,'input_and_source_sha256':pins,'historical_main_rerun':False}
    request_path=ROOT/f'{stem}_request.json';exclusive_json(request_path,request)
    before=ROOT/f'{stem}_preexecution.zip'
    with zipfile.ZipFile(before,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in inputs: archive.write(ROOT/name,name)
        archive.write(request_path,request_path.name)
    started=time.time()
    with (ROOT/f'{stem}_stdout.txt').open('xb') as out,(ROOT/f'{stem}_stderr.txt').open('xb') as err:
        proc=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,check=False)
    outputs=[ROOT/f'{stem}_stdout.txt',ROOT/f'{stem}_stderr.txt']
    for path in (ROOT/f'{stem}_receipt.json',ROOT/f'{stem}_interface.json'):
        if path.exists(): outputs.append(path)
    exitdata={'command':command,'exit_code':proc.returncode,'natural_exit':proc.returncode>=0,'elapsed_seconds':time.time()-started,'preexecution_sha256':sha(before),'source_and_input_unchanged':pins=={name:sha(ROOT/name) for name in inputs},'output_sha256':{p.name:sha(p) for p in outputs}}
    exitpath=ROOT/f'{stem}_exit.json';exclusive_json(exitpath,exitdata)
    after=ROOT/f'{stem}_executed.zip'
    with zipfile.ZipFile(after,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in inputs: archive.write(ROOT/name,name)
        archive.write(request_path,request_path.name)
        for p in outputs+[exitpath]: archive.write(p,p.name)
    capsule={'request_sha256':sha(request_path),'preexecution_sha256':sha(before),'executed_sha256':sha(after),'exit_sha256':sha(exitpath),'exit_code':proc.returncode,'natural_exit':proc.returncode>=0}
    exclusive_json(ROOT/f'{stem}_capsule.json',capsule)
    print(json.dumps(capsule),flush=True)
    return proc.returncode

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',required=True)
    raise SystemExit(run(parser.parse_args().epoch))
