"""Natural exit evidence for the actual state/current/tail analysis consumer."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def sha(p: Path) -> str:return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    prefix=sys.argv[1];source=ROOT/'rich_mobile_packet_20261015_analyze.py';model=ROOT/'rich_mobile_packet_20261015_model.py'
    primarymanifest=ROOT/(prefix+'_manifest.json');manifest=json.loads(primarymanifest.read_text())
    sourcechain=[ROOT/name for name in manifest if name.endswith('.py')]
    inputs={p.name:sha(p) for p in [source,model,Path(__file__).resolve(),primarymanifest,ROOT/(prefix+'_exit.json')]+sourcechain};started=time.time()
    stdout=ROOT/'rich_mobile_packet_20261015_analysis_stdout.txt';stderr=ROOT/'rich_mobile_packet_20261015_analysis_stderr.txt'
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with stdout.open('w') as o,stderr.open('w') as e:
        child=subprocess.run(['/home/administrator/substrate-framework/.venv/bin/python',str(source),prefix],cwd=ROOT,env=env,stdout=o,stderr=e)
    output=ROOT/'rich_mobile_packet_20261015_observed.json';data=json.loads(output.read_text()) if output.is_file() else {}
    after={name:sha(ROOT/name) for name in inputs};consumed=data.get('inputs_sha256',{})
    mismatches={name:dict(observed=value,manifest=manifest.get(name)) for name,value in consumed.items() if name in manifest and manifest[name]['sha256']!=value}
    receipt=dict(returncode=child.returncode,wall_seconds=time.time()-started,source_inputs_before=inputs,source_inputs_after=after,
        source_inputs_unchanged=inputs==after,primary_manifest_consumer_mismatches=mismatches,
        scientific_elapsed_seconds=data.get('elapsed_seconds'),scientific_peak_resident_KiB=data.get('peak_resident_KiB'),stdout_sha256=sha(stdout),stderr_sha256=sha(stderr))
    exitpath=ROOT/'rich_mobile_packet_20261015_analysis_exit.json';exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
    names=set(consumed)|set(inputs)|{stdout.name,stderr.name,exitpath.name}
    if output.is_file():names.add(output.name)
    bare=ROOT/'rich_mobile_packet_20261015_bare_electric_states.npz'
    if bare.is_file():names.add(bare.name)
    finalmanifest=ROOT/'rich_mobile_packet_20261015_analysis_manifest.json';finalmanifest.write_text(json.dumps({name:dict(sha256=sha(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in sorted(names)},indent=2)+'\n')
    capsule=ROOT/'rich_mobile_packet_20261015_analysis_capsule.zip'
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name in sorted(names):z.write(ROOT/name,name)
        z.write(finalmanifest,finalmanifest.name)
    print(json.dumps(dict(**receipt,capsule=capsule.name,capsule_sha256=sha(capsule))),flush=True)
    sys.exit(child.returncode)


if __name__=='__main__':main()
