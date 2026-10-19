"""Preserve timeout evidence and capture a naturally completed new continuation."""
from __future__ import annotations
import ast,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def sha(p: Path) -> str:return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    prefix=sys.argv[1];supplier='rich_mobile_packet_20261015_actual1';source=ROOT/'rich_mobile_packet_20261015_continue.py'
    inputs=sorted(p for p in ROOT.glob(supplier+'_*') if p.is_file() and p.suffix!='.zip')
    timeout=ROOT/(supplier+'_timeout_exit.json')
    timeout.write_text(json.dumps(dict(kind='OBSERVED_TOOL3600S_TIMEOUT_NOT_NATURAL_CHILD_EXIT',tool_job='bg_225',observed_tool_error='Command timed out after3600seconds; wall3600.10s',
        natural_returncode=None,peak_resident_KiB=None,completed_baseline='charged_no_incoming0..8',unfinished_incoming='observations0..2, last full incoming state t0',stderr_bytes=(ROOT/(supplier+'_stderr.txt')).stat().st_size,
        owner_resolution='One exact process-identity recovery probe found no remaining rich_mobile_packet writer; no concurrent replay.'),indent=2)+'\n')
    inputs.append(timeout);timeoutmanifest=ROOT/(supplier+'_manifest.json');timeoutmanifest.write_text(json.dumps({p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in inputs},indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(supplier+'_timeout_capsule.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in inputs+[timeoutmanifest]:z.write(p,p.name)
    sources={Path(__file__).resolve()};pending=[source]
    while pending:
        p=pending.pop()
        if p in sources:continue
        sources.add(p)
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node,ast.ImportFrom) and node.module:
                q=ROOT/(node.module+'.py')
                if q.is_file() and q not in sources:pending.append(q)
    bound=sorted(set(inputs+[timeoutmanifest,ROOT/(prefix+'_input.json')])|sources);before={p.name:sha(p) for p in bound};started=time.time()
    stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt');env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with stdout.open('w') as o,stderr.open('w') as e:
        child=subprocess.run(['/home/administrator/substrate-framework/.venv/bin/python',str(source),prefix],cwd=ROOT,env=env,stdout=o,stderr=e)
    after={p.name:sha(p) for p in bound};result=ROOT/(prefix+'_result.json');data=json.loads(result.read_text()) if result.is_file() else {}
    receipt=dict(returncode=child.returncode,wall_seconds=time.time()-started,sources_inputs_before=before,sources_inputs_after=after,source_inputs_unchanged=before==after,
        scientific_elapsed_seconds=data.get('elapsed_seconds'),scientific_peak_resident_KiB=data.get('peak_resident_KiB'),stdout_sha256=sha(stdout),stderr_sha256=sha(stderr),supplier_timeout_manifest=timeoutmanifest.name)
    exitpath=ROOT/(prefix+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
    outputs=sorted(p for p in ROOT.glob(prefix+'_*') if p.is_file() and p.suffix!='.zip');allfiles=sorted(set(bound+outputs));manifest=ROOT/(prefix+'_manifest.json')
    manifest.write_text(json.dumps({p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in allfiles},indent=2)+'\n')
    capsule=ROOT/(prefix+'_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in allfiles+[manifest]:z.write(p,p.name)
    print(json.dumps(dict(**receipt,capsule=capsule.name,capsule_sha256=sha(capsule))),flush=True);sys.exit(child.returncode)


if __name__=='__main__':main()
