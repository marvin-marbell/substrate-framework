"""Exclusive byte capture and genuine natural-exit evidence for each new epoch."""
from __future__ import annotations
import argparse, ast, hashlib, json, os, subprocess, sys, time, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE='harmonic_car_bridge_scientist_20261011.py'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def capture(path,members):
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(members.items()):z.writestr(name,raw)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',required=True);args=parser.parse_args();base='harmonic_car_bridge_'+args.epoch
    source=(ROOT/SOURCE).read_bytes();tree=ast.parse(source);definitions={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ['PINS','EXTRA']:definitions[target.id]=ast.literal_eval(node.value)
    members={SOURCE:source,Path(__file__).name:Path(__file__).read_bytes(),'harmonic_car_bridge_lsp_ast_20261011.json':(ROOT/'harmonic_car_bridge_lsp_ast_20261011.json').read_bytes()}
    for name in [*definitions['PINS'],*definitions['EXTRA']]:members[name]=(ROOT/name).read_bytes()
    for name,expected in definitions['PINS'].items():assert digest(members[name])==expected,name
    capture(ROOT/(base+'_preexecution.zip'),members)
    command=[sys.executable,SOURCE,'--epoch',args.epoch];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');start=time.monotonic()
    with (ROOT/(base+'_stdout.txt')).open('xb') as out,(ROOT/(base+'_stderr.txt')).open('xb') as err:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err);returncode=process.wait()
    record={'command':command,'returncode':returncode,'natural_exit':True,'elapsed_seconds':time.monotonic()-start,'source_and_input_sha256':{n:digest(v) for n,v in members.items()},'source_inputs_unchanged':all((ROOT/n).read_bytes()==v for n,v in members.items()),'epoch':args.epoch}
    exitpath=ROOT/(base+'_exit.json');exitpath.write_text(json.dumps(record,indent=2)+'\n')
    for suffix in ['_exit.json','_stdout.txt','_stderr.txt','_receipt.json','_controls.json']:
        p=ROOT/(base+suffix)
        if p.exists():members[p.name]=p.read_bytes()
    capture(ROOT/(base+'_executed.zip'),members);print(json.dumps(record,indent=2));sys.exit(returncode)
if __name__=='__main__':main()
