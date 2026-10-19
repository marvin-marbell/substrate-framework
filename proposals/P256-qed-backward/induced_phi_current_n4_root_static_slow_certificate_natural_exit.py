"""Capture the actual independent consumer CLI, natural exit and runtime bytes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

PREFIX = 'induced_phi_current_n4_root_static_slow_certificate'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--label',default='actual')
    args = parser.parse_args()
    root: Path = args.directory.resolve()
    source = root/(PREFIX+'.py')
    supervisor = Path(__file__).resolve()
    helper = root/(PREFIX+'_frame_bound.py')
    spectral = root/(PREFIX+'_spectral_bound.py')
    pinned = {'consumer':sha(source),'supervisor':sha(supervisor),'frame_bound':sha(helper),'spectral_bound':sha(spectral),'python':sha(args.python.resolve())}
    receipt = root/(PREFIX+'_'+args.label+'_receipt.json')
    archive = root/(PREFIX+'_'+args.label+'_immutable_inputs.zip')
    command = [str(args.python),str(source),'--gaussian',str(root/'induced_phi_gaussian_action_certificate_receipt.json'),'--output',str(receipt),'--archive',str(archive)]
    start = time.monotonic()
    process = subprocess.run(command,capture_output=True,text=True,cwd=root,check=False)
    capsule: dict[str,Any] = {'command':command,'cwd':str(root),'returncode':process.returncode,
                             'natural_exit':True,'wall_seconds':time.monotonic()-start,
                             'stdout':process.stdout,'stderr':process.stderr,'pinned_before':pinned,
                             'source_unchanged':sha(source)==pinned['consumer'] and sha(supervisor)==pinned['supervisor'] and sha(helper)==pinned['frame_bound'] and sha(spectral)==pinned['spectral_bound'],
                             'python_unchanged':sha(args.python.resolve())==pinned['python']}
    if receipt.is_file():
        capsule['receipt'] = {'path':str(receipt),'sha256':sha(receipt),'bytes':receipt.stat().st_size}
    if archive.is_file():
        capsule['immutable_archive'] = {'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size}
    path = root/(PREFIX+'_'+args.label+'_natural_exit_capsule.json')
    path.write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps(capsule,indent=2))
    sys.exit(process.returncode)


if __name__ == '__main__':
    main()
