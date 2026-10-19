"""Natural child exit and immutable exact scientific source/input/output capsule."""
from __future__ import annotations
import ast,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def digest(p: Path) -> str:return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    prefix=Path(sys.argv[1]);inputpath=ROOT/(prefix.name+'_input.json');source=ROOT/'rich_mobile_packet_20261015_calculate.py'
    sources={Path(__file__).resolve()};pending=[source]
    while pending:
        p=pending.pop()
        if p in sources:continue
        sources.add(p)
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node,ast.ImportFrom) and node.module:
                q=ROOT/(node.module+'.py')
                if q.is_file() and q not in sources:pending.append(q)
    bound=sorted(sources|{inputpath});before={p.name:digest(p) for p in bound};started=time.time()
    out=ROOT/(prefix.name+'_stdout.txt');err=ROOT/(prefix.name+'_stderr.txt');env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with out.open('w') as stdout,err.open('w') as stderr:
        child=subprocess.run([PYTHON,str(source),str(inputpath),str(ROOT/prefix.name)],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    after={p.name:digest(p) for p in bound};exitpath=ROOT/(prefix.name+'_exit.json')
    result=ROOT/(prefix.name+'_result.json');data=json.loads(result.read_text()) if result.is_file() else None
    receipt=dict(kind='NEW_NATURAL_SCIENTIFIC_CHILD_EXIT',argv=[PYTHON,str(source),str(inputpath),str(ROOT/prefix.name)],returncode=child.returncode,
        wall_seconds=time.time()-started,sources_input_before=before,sources_input_after=after,source_input_unchanged=before==after,
        scientific_elapsed_seconds=data.get('elapsed_seconds') if data else None,scientific_peak_resident_KiB=data.get('peak_resident_KiB') if data else None,
        stdout_sha256=digest(out),stderr_sha256=digest(err),environment_thread_counts={k:env[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']})
    exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
    artifacts=sorted(p for p in ROOT.glob(prefix.name+'_*') if p.is_file() and p.suffix!='.zip')
    manifest={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in artifacts+sorted(sources)}
    manifestpath=ROOT/(prefix.name+'_manifest.json');manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
    capsule=ROOT/(prefix.name+'_capsule.zip')
    with zipfile.ZipFile(capsule,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in artifacts+sorted(sources)+[manifestpath]:z.write(p,p.name)
    print(json.dumps(dict(**receipt,capsule=capsule.name,capsule_sha256=digest(capsule),manifest=manifestpath.name)),flush=True)
    sys.exit(child.returncode)


if __name__=='__main__':main()
