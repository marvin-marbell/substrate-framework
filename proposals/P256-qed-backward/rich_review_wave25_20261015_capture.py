"""Capture precisely one selected independent finite case and its actual exit."""
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

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave25_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
FROZEN = {
    'rich_photon_fullq_20261014_model.py': 'f3af36d79727738822ed2d19b04bf6905c90de0110265717570e7868828dbf02',
    'rich_photon_20261014_loop.py': 'a6674ba9e4eab0cd66bef1bd0143d4c6d36a823297fe7bab3f960c60df0b0948',
}


def identity(name: str) -> dict[str, object]:
    blob = (ROOT/name).read_bytes()
    return dict(sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))


def main() -> None:
    cutoff = int(sys.argv[1])
    assert cutoff in (1,2)
    for name,sha in FROZEN.items():
        assert identity(name)['sha256'] == sha
    capsule = 'rich_photon_gauss_transport_20261015_delivery_capsule.zip'
    assert identity(capsule)['sha256'] == 'f98cf61d31e01dfdba8b49929ec659d9090dc9eb0da2564522f3117e6c05bdfb'
    sources = list(FROZEN)+[f'{PREFIX}consume.py',f'{PREFIX}capture.py',f'{PREFIX}input.json']
    before = {name:identity(name) for name in sources}
    source_capsule = ROOT/f'{PREFIX}L{cutoff}_source_capsule.zip'
    with zipfile.ZipFile(source_capsule,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sources:
            z.write(ROOT/name,name)
    command = [PYTHON,f'{PREFIX}consume.py',f'{PREFIX}input.json',str(cutoff)]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    start = dict(command=command,cwd=str(ROOT),python=PYTHON,platform=platform.platform(),
                 sources=before,source_capsule=identity(source_capsule.name),
                 target_capsule=identity(capsule),threads=1,start_unix=time.time())
    (ROOT/f'{PREFIX}L{cutoff}_start.json').write_text(json.dumps(start,indent=2)+'\n')
    begin = time.monotonic()
    process = subprocess.run(command,cwd=ROOT,env=env,capture_output=True)
    (ROOT/f'{PREFIX}L{cutoff}_stdout.txt').write_bytes(process.stdout)
    (ROOT/f'{PREFIX}L{cutoff}_stderr.txt').write_bytes(process.stderr)
    after = {name:identity(name) for name in sources}
    exit_record = dict(returncode=process.returncode,natural_exit=process.returncode>=0,
         signal=-process.returncode if process.returncode<0 else None,
         elapsed_seconds=time.monotonic()-begin,sources_before=before,sources_after=after,
         unchanged=before==after,stdout=identity(f'{PREFIX}L{cutoff}_stdout.txt'),
         stderr=identity(f'{PREFIX}L{cutoff}_stderr.txt'))
    (ROOT/f'{PREFIX}L{cutoff}_exit.json').write_text(json.dumps(exit_record,indent=2)+'\n')
    print(process.stdout.decode(),end='')
    print(json.dumps(exit_record,indent=2))
    sys.exit(process.returncode)


if __name__ == '__main__':
    main()
