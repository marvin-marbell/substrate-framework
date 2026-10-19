"""Exclusive byte capsules and natural-exit receipts for the new graph CLI."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
import normalized_root_graph_scientist_20261011 as scientist

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(path: Path, obj: object) -> None:
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')


def archive(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, raw in sorted(files.items()):
            z.writestr(name, raw)
    path.chmod(0o444)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--epoch', required=True)
    p.add_argument('--peer-interface', required=True)
    args = p.parse_args()
    prefix = f'normalized_root_graph_{args.epoch}_'
    inputs = [Path(__file__).name, Path(scientist.__file__).name, *scientist.PINS, args.peer_interface,
              'normalized_root_graph_derivation_20261011.json', 'normalized_root_graph_language_attempt3_20261011.json']
    peer = json.loads((ROOT/args.peer_interface).read_bytes())
    inputs.extend([peer['receipt'],peer['natural_exit'],peer['graph_interface']['executed_archive'],
                   'normalized_root_graph_language_postedit_20261011.json'])
    assert sha((ROOT/peer['receipt']).read_bytes())==peer['receipt_sha256']
    assert sha((ROOT/peer['graph_interface']['executed_archive']).read_bytes())==peer['graph_interface']['executed_archive_sha256']
    peer_exit = json.loads((ROOT/peer['natural_exit']).read_bytes())
    assert peer_exit['natural_returncode']==0 and peer_exit['natural_exit']
    before = {name:(ROOT/name).read_bytes() for name in inputs}
    pins = {name:sha(raw) for name, raw in before.items()}
    request = {'epoch':args.epoch, 'input_sha256':pins,
               'command':['/home/administrator/substrate-framework/.venv/bin/python', str(ROOT/Path(scientist.__file__).name),
                          '--epoch', args.epoch, '--peer-interface', args.peer_interface],
               'natural_exit_required':True, 'old_solver_rerun':False, 'physical_hopping':'1',
               'thread_environment':{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}}
    save(ROOT/(prefix+'request.json'), request)
    before[prefix+'request.json'] = (ROOT/(prefix+'request.json')).read_bytes()
    pre = ROOT/(prefix+'preexecution.zip')
    archive(pre, before)
    t0 = time.monotonic()
    env = dict(os.environ, **request['thread_environment'])
    proc = subprocess.run(request['command'], cwd=ROOT, env=env, capture_output=True, check=False)
    elapsed = time.monotonic()-t0
    for suffix, raw in [('stdout.txt',proc.stdout),('stderr.txt',proc.stderr)]:
        with (ROOT/(prefix+suffix)).open('xb') as f:
            f.write(raw)
    after = {name:(ROOT/name).read_bytes() for name in inputs}
    unchanged = {name:sha(raw)==pins[name] for name, raw in after.items()}
    outputs = {prefix+s:(ROOT/(prefix+s)).read_bytes() for s in ['request.json','stdout.txt','stderr.txt']}
    receipt = ROOT/(prefix+'receipt.json')
    if receipt.exists():
        outputs[receipt.name] = receipt.read_bytes()
    witness = ROOT/(prefix+'fullcar.npz')
    if witness.exists():
        outputs[witness.name] = witness.read_bytes()
    exitinfo = {'returncode':proc.returncode,'natural_exit':proc.returncode>=0,'signal':-proc.returncode if proc.returncode<0 else None,
                'elapsed_seconds':elapsed,'preexecution_sha256':sha(pre.read_bytes()),'input_sha256':pins,
                'post_input_sha256':{name:sha(raw) for name, raw in after.items()},'unchanged':unchanged,
                'output_sha256':{name:sha(raw) for name, raw in outputs.items()},
                'complete_cli_receipt_observed':receipt.exists(),'all64_physical1':True}
    save(ROOT/(prefix+'exit.json'), exitinfo)
    outputs[prefix+'exit.json'] = (ROOT/(prefix+'exit.json')).read_bytes()
    executed = ROOT/(prefix+'executed.zip')
    archive(executed, {**after, **outputs})
    save(ROOT/(prefix+'delivery.json'), {'returncode':proc.returncode,'natural_exit':exitinfo['natural_exit'],
                                       'executed_sha256':sha(executed.read_bytes()),'exit_sha256':sha(outputs[prefix+'exit.json']),
                                       'preexecution_sha256':exitinfo['preexecution_sha256']})
    print(json.dumps(exitinfo, indent=2), flush=True)
    if proc.returncode != 0 or not all(unchanged.values()) or not receipt.exists():
        raise SystemExit(1)

if __name__ == '__main__':
    main()
