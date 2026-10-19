"""Own-prefix natural-exit receipt and immutable predecessor byte audit."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='rich_review_wave3_20261014'
SOURCES=[f'{PREFIX}_consumer.py',f'{PREFIX}_capture.py','rich_spatial_20261014_model.py','rich_photon_fullq_20261014_model.py','rich_photon_20261014_loop.py','rich_spinor_rot_20261014_model.py','rich_spinor_gauss_20261014_model.py','rich_spinor_20261014_model.py','rich_composite_rot_20261014_consumer.py']
INPUTS=['rich_composite_rot_20261014_magnetic_request.json','rich_composite_rot_20261014_magnetic1_receipt.npz']

def sha(data):
    return hashlib.sha256(data).hexdigest()

def audit():
    cases=[('rich_spatial_20261014_cells3_execution.zip','rich_spatial_20261014_cells3_natural_exit.json'),('rich_spinor_rot_20261014_actual3_capsule.zip','rich_spinor_rot_20261014_actual3_exit.json'),('rich_spinor_gauss_20261014_flux_actual2_capsule.zip','rich_spinor_gauss_20261014_flux_actual2_exit.json'),('rich_composite_rot_20261014_magnetic1_capsule.zip','rich_composite_rot_20261014_magnetic1_exit.json')]
    output=[]
    for capsule, receipt in cases:
        with zipfile.ZipFile(ROOT/capsule) as z:
            comparisons=[]
            for member in z.namelist():
                name=Path(member).name
                path=ROOT/name
                if path.is_file():
                    captured=sha(z.read(member)); current=sha(path.read_bytes())
                    comparisons.append(dict(member=member,current_file=name,capsule_sha256=captured,current_sha256=current,equal=captured==current))
            output.append(dict(capsule=capsule,capsule_sha256=sha((ROOT/capsule).read_bytes()),receipt=json.loads((ROOT/receipt).read_text()),comparisons=comparisons))
    fullq=ROOT/'rich_charged_gauss_20261014_fullq.py'
    return dict(cases=output,fullq_current_sha256=sha(fullq.read_bytes()),fullq_typing_revision=json.loads((ROOT/'rich_charged_gauss_20261014_fullq_typing_revision.json').read_text()),scope='Byte audit only; predecessor scientific CLIs never reexecuted. Changed capture or typed current source does not inherit earlier science stamp.')

def main():
    source_bytes={n:(ROOT/n).read_bytes() for n in SOURCES}
    input_bytes={n:(ROOT/n).read_bytes() for n in INPUTS}
    before={n:sha(b) for n,b in source_bytes.items()}
    env=os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):env[key]='1'
    command=[sys.executable,str(ROOT/f'{PREFIX}_consumer.py')]
    started=time.monotonic()
    proc=subprocess.run(command,cwd=ROOT,env=env,capture_output=True)
    elapsed=time.monotonic()-started
    (ROOT/f'{PREFIX}_stdout.txt').write_bytes(proc.stdout)
    (ROOT/f'{PREFIX}_stderr.txt').write_bytes(proc.stderr)
    after={n:sha((ROOT/n).read_bytes()) for n in SOURCES}
    (ROOT/f'{PREFIX}_audit.json').write_text(json.dumps(audit(),indent=2)+'\n')
    outnames=[f'{PREFIX}_stdout.txt',f'{PREFIX}_stderr.txt',f'{PREFIX}_audit.json']
    if (ROOT/f'{PREFIX}_result.json').exists():outnames.append(f'{PREFIX}_result.json')
    receipt=dict(command=command,cwd=str(ROOT),runtime=sys.version,platform=platform.platform(),returncode=proc.returncode,natural_exit=proc.returncode>=0,elapsed_seconds=elapsed,source_before=before,source_after=after,source_unchanged=before==after,input_sha256={n:sha(b) for n,b in input_bytes.items()},output_sha256={n:sha((ROOT/n).read_bytes()) for n in outnames},plan='issue222-rich-backward-candidates-20261014')
    (ROOT/f'{PREFIX}_exit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/f'{PREFIX}_capsule.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n,b in source_bytes.items():z.writestr('source/'+n,b)
        for n,b in input_bytes.items():z.writestr('input/'+n,b)
        for n in outnames+[f'{PREFIX}_exit.json']:z.write(ROOT/n,'output/'+n)
    print(json.dumps(receipt,indent=2))
    return proc.returncode

if __name__=='__main__':
    raise SystemExit(main())
