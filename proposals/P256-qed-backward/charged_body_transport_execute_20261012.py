"""Capture and execute only the NEW charged transport scientific CLI."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.request.read_text())
    root=args.request.parent
    prefix=config['epoch']
    target=lambda suffix:root/f'{prefix}_{suffix}'
    request=Path(config['consumer_request'])
    spec=json.loads(request.read_text())
    inputs=[Path(p) for p in spec['pins']]+[request,args.request,Path(__file__)]
    inputs=list(dict.fromkeys(p.resolve() for p in inputs))
    for p,sha in spec['pins'].items():
        assert digest(Path(p))==sha,p
    pre=target('preexecution.zip')
    assert not pre.exists(),pre
    hashes={str(p):digest(p) for p in inputs}
    with zipfile.ZipFile(pre,'x',zipfile.ZIP_DEFLATED) as z:
        for p in inputs:
            z.write(p,arcname='inputs/'+p.name)
    command=[config['python'],config['consumer'],'--request',str(request),'--output',str(target('receipt.json'))]
    begin=time.time_ns()
    result=subprocess.run(command,cwd=root,capture_output=True,check=False)
    end=time.time_ns()
    target('stdout.txt').write_bytes(result.stdout)
    target('stderr.txt').write_bytes(result.stderr)
    exitpath=target('exit.json')
    exitpath.write_text(json.dumps({'command':command,'start_ns':begin,'end_ns':end,
                                   'elapsed_seconds':(end-begin)/1e9,'natural_returncode':result.returncode,
                                   'timeout_or_signal_intervention':False},indent=2)+'\n')
    unchanged={str(p):digest(p)==hashes[str(p)] for p in inputs}
    outputs=[target('stdout.txt'),target('stderr.txt'),exitpath]
    if target('receipt.json').exists(): outputs.append(target('receipt.json'))
    executed=target('executed.zip')
    with zipfile.ZipFile(executed,'x',zipfile.ZIP_DEFLATED) as z:
        with zipfile.ZipFile(pre) as source:
            for member in source.namelist():z.writestr(member,source.read(member))
        for p in outputs:z.write(p,arcname='outputs/'+p.name)
    capsule={'kind':'IMMUTABLE_NEW_CHARGED_TRANSPORT_EXECUTED_CAPSULE',
             'epoch':prefix,'command':command,'natural_returncode':result.returncode,
             'elapsed_seconds':(end-begin)/1e9,'input_hashes':hashes,'post_input_unchanged':unchanged,
             'preexecution':{'path':str(pre),'sha256':digest(pre)},
             'executed':{'path':str(executed),'sha256':digest(executed)},
             'outputs':{str(p):digest(p) for p in outputs},
             'failed_scientific_gate_preserved_in_receipt':True,'old_entrypoint_replayed':False,
             'diagnostics_build_lint_tests_formatters_run':False}
    target('capsule.json').write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps({'epoch':prefix,'returncode':result.returncode,'elapsed_seconds':(end-begin)/1e9,
                      'capsule':str(target('capsule.json'))}))
    assert all(unchanged.values()),'captured input changed during execution'
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
