"""Separate source epoch; no edits to old guarded consumer/capture files."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,zipfile
from rich_review_wave49_20261016_execute import dependencies

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_';PY='/home/administrator/substrate-framework/.venv/bin/python'


def main():
    mode=sys.argv[1];attempt=sys.argv[2];own=ROOT/(P+'priority_consumer.py');tag=P+attempt
    deps=set();dependencies(own,deps);deps.add(Path(__file__).resolve());hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(deps)}
    with zipfile.ZipFile(ROOT/(tag+'_source_capsule.zip'),'x',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(deps):archive.write(path,path.name)
    command=[PY,str(own),*sys.argv[3:],mode]
    inputs=dict(plan_id='issue222-rich-backward-candidates-20261014',captured_contract_revision='6facdc39fc29c4b59769e6335b1c31bc95f322e3bf55184de7505d83d9008b08',template='general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',work_item='independent-review',command=command,source_sha256=hashes,capture_epoch='transport_priority2',priority_handoff_receipt=P+'priority_transport_handoff.json',affected_checkpoint_repair_receipt=P+'priority_transport1_assessment.json',meaning='Same frozen operator/H and own committed physical preparation; phase-exact/direct native checkpoint restoration; physical carrier/field paths before optional mixed-contact inventory. Sequential one-model streaming loads, no live heavy clone.')
    (ROOT/(tag+'_input.json')).write_text(json.dumps(inputs,indent=2)+'\n')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',REVIEW49_INPUT_RECEIPT=tag+'_input.json')
    start=time.time()
    with (ROOT/(tag+'_stdout.txt')).open('x') as stdout,(ROOT/(tag+'_stderr.txt')).open('x') as stderr:
        child=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,env=env)
    result=dict(natural_exit=child.returncode,elapsed_seconds=time.time()-start,sources_unchanged={p.name:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name] for p in sorted(deps)},meaning='A negative signal return is an interruption, not a natural physical acceptance; original stdout/stderr retained.')
    (ROOT/(tag+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);sys.exit(child.returncode)

if __name__=='__main__':main()
