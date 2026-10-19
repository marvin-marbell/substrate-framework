"""Independent Wave19 exact source/input/body/state/output natural capsules."""
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def closure(path,seen):
    if path in seen:return
    seen.add(path)
    for node in ast.walk(ast.parse(path.read_text())):
        names=[node.module] if isinstance(node,ast.ImportFrom) else [x.name for x in node.names] if isinstance(node,ast.Import) else []
        for name in names:
            candidate=ROOT/(str(name).split('.')[0]+'.py')
            if candidate.is_file():closure(candidate,seen)


def limit():
    maximum=3800*1024**2;resource.setrlimit(resource.RLIMIT_AS,(maximum,maximum))


def main():
    prefix=sys.argv[1];inp=ROOT/(prefix+'_input.json');p=json.loads(inp.read_text());consumer=ROOT/p['consumer'];output=ROOT/(prefix+'_result.json');cap=ROOT/(prefix+'_capsule.zip')
    if cap.exists():raise RuntimeError('Immutable attempt already exists; select a new prefix')
    sources=set();closure(consumer,sources);sources.update([Path(__file__).resolve(),inp,ROOT/p['owner_input']]);sources.update(ROOT/name for name in p.get('assets',[]))
    before={path.name:sha(path) for path in sorted(sources)};audit={}
    for item in p.get('owner_audits',[]):
        receipt=json.loads((ROOT/item['exit']).read_text());expected=receipt.get('source_before',receipt.get('source_input_before',{})).copy();expected.update(receipt.get('outputs',{}))
        expected.update({name:entry['sha256'] for name,entry in receipt.get('files',{}).items()})
        with zipfile.ZipFile(ROOT/item['capsule']) as z:
            names={Path(n).name:n for n in z.namelist()};rows={}
            for name,value in expected.items():
                payload=hashlib.sha256(z.read(names[name])).hexdigest();rows[name]=dict(expected=value,payload=payload,live=sha(ROOT/name),exact=value==payload==sha(ROOT/name))
        audit[item['exit']]=dict(capsule_sha256=sha(ROOT/item['capsule']),members=rows,all_exact=all(v['exact'] for v in rows.values()))
        if not audit[item['exit']]['all_exact']:raise RuntimeError('Frozen owning capsule mismatch')
    command=[PYTHON,str(consumer),str(inp),str(output)];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0');start=time.time();stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt')
    with stdout.open('wb') as out,stderr.open('wb') as err:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,preexec_fn=limit);pid=process.pid;returncode=process.wait()
    after={path.name:sha(path) for path in sorted(sources)};files=sorted(sources)+[stdout,stderr]
    for path in [output,output.with_suffix('.npz')]:
        if path.is_file():files.append(path)
    record=dict(kind='WAVE19_NEW_DISTINCT_CANDIDATE_PHYSICAL_CONSUMER',command=command,pid=pid,returncode=returncode,natural_exit=returncode>=0,termination_signal=-returncode if returncode<0 else None,elapsed_seconds=time.time()-start,peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,address_space_limit_MiB=3800,source_before=before,source_after=after,source_unchanged=before==after,owner_audits=audit,files={path.name:dict(bytes=path.stat().st_size,sha256=sha(path)) for path in files})
    ep=ROOT/(prefix+'_exit.json');ep.write_text(json.dumps(record,indent=2)+'\n')
    with zipfile.ZipFile(cap,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files+[ep]:z.write(path,path.name)
    print(json.dumps(dict(returncode=returncode,natural_exit=record['natural_exit'],seconds=record['elapsed_seconds'],peak_RSS_KiB=record['peak_child_RSS_KiB'],source_unchanged=before==after,capsule_sha256=sha(cap))))
    sys.exit(returncode)


if __name__=='__main__':main()
