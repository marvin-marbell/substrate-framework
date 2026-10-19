"""Capture executed inputs before the joint22 CLI and its stable natural exit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import joined_phi_u1_joint_pauli_22_certificate as consumer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epoch',required=True)
    parser.add_argument('--cells',type=int,default=16384)
    args = parser.parse_args()
    stem = consumer.ROOT/(consumer.PREFIX+args.epoch)
    if args.output.exists() or stem.with_suffix('.capsule.json').exists():
        raise FileExistsError('Preserve existing scientific epochs; select a fresh epoch')
    captured = consumer.snapshot()
    captured[Path(__file__).name] = Path(__file__).read_bytes()
    before = {name:hashlib.sha256(raw).hexdigest() for name,raw in captured.items()}
    archive = stem.with_suffix('.execution_source.zip')
    command = [sys.executable,str(Path(consumer.__file__)),'--output',str(args.output),'--cells',str(args.cells)]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in captured.items(): z.writestr(name,raw)
        z.writestr(consumer.PREFIX+'command.json',json.dumps(command))
    started = time.monotonic()
    process = subprocess.run(command,cwd=consumer.ROOT,capture_output=True,text=True)
    stdout = stem.with_suffix('.stdout.txt'); stderr = stem.with_suffix('.stderr.txt')
    stdout.write_text(process.stdout); stderr.write_text(process.stderr)
    after = {name:hashlib.sha256((consumer.ROOT/name).read_bytes()).hexdigest() for name in captured}
    capsule = dict(command=command,python=sys.executable,exit_code=process.returncode,natural_completed_exit=True,elapsed_seconds=time.monotonic()-started,source_input_sha256_before=before,source_input_sha256_after=after,source_inputs_unchanged=before==after,immutable_execution_archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),stdout=str(stdout),stderr=str(stderr),receipt=str(args.output),receipt_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest() if args.output.exists() else None)
    stem.with_suffix('.capsule.json').write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps(dict(exit_code=process.returncode,elapsed_seconds=capsule['elapsed_seconds'],source_inputs_unchanged=capsule['source_inputs_unchanged'],stdout=process.stdout,stderr=process.stderr),indent=2))
    if before != after: raise RuntimeError('Execution bytes changed')
    raise SystemExit(process.returncode)


if __name__ == '__main__': main()
