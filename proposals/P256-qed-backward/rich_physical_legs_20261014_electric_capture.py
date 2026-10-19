"""Exact source/input/OWN state/output capsule for independent electric consumer."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parent;prefix='rich_physical_legs_20261014_electric'
    inp=root/(prefix+'_input.json');out=root/(prefix+'_result.json');data=json.loads(inp.read_text())
    names=['rich_physical_legs_20261014_electric.py','rich_physical_legs_20261014_electric_capture.py','rich_physical_legs_20261014_model.py','rich_physical_legs_20261014_calculate.py',
           'rich_field_packet_20261014_connected.py','rich_field_packet_20261014_model.py','rich_field_packet_20261014_calculate.py','rich_field_packet_20261014_localwell.py',
           'rich_common_mobile_20261014_model.py','rich_frame_photon_20261014_model.py','rich_frame_gauss_20261014_model.py',inp.name]
    names.extend(data[key] for key in ['state_npz','state_receipt','original_input','original_result'])
    paths=[root/name for name in names];raw={p.name:p.read_bytes() for p in paths};before={p.name:sha(p) for p in paths}
    environment=dict(os.environ)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:environment[key]='1'
    command=[sys.executable,str(root/(prefix+'.py')),str(inp),str(out)]
    started=time.time();p=subprocess.run(command,cwd=root,env=environment,capture_output=True,text=True)
    stdout=root/(prefix+'_stdout.txt');stderr=root/(prefix+'_stderr.txt');stdout.write_text(p.stdout);stderr.write_text(p.stderr)
    after={p.name:sha(p) for p in paths};outputs=[stdout,stderr]+[p for p in [out,out.with_suffix('.npz')] if p.exists()]
    receipt=dict(command=command,cwd=str(root),returncode=p.returncode,natural_exit=True,elapsed_seconds=time.time()-started,
                 source_before=before,source_after=after,source_unchanged=before==after,outputs={p.name:sha(p) for p in outputs},executable=sys.executable)
    exitpath=root/(prefix+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n');capsule=root/(prefix+'_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in raw.items():archive.writestr(name,content)
        for path in outputs+[exitpath]:archive.write(path,path.name)
    print(p.stdout)
    if p.stderr:print(p.stderr,file=sys.stderr)
    print(json.dumps(dict(returncode=p.returncode,seconds=receipt['elapsed_seconds'],capsule=str(capsule),capsule_sha256=sha(capsule))))
    raise SystemExit(p.returncode)


if __name__=='__main__':main()
