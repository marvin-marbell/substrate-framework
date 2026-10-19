"""Immutable preexecution and executed capsules for one independent review CLI."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capsule(path, files):
    with path.open('xb') as stream:
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for member,source in files.items():
                z.writestr(member,source.read_bytes())
    os.chmod(path,0o444)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True);p.add_argument('--output-prefix',type=Path,required=True)
    args=p.parse_args(); request=json.loads(args.request.read_text()); prefix=args.output_prefix
    source=Path(request['consumer']); runner=Path(__file__).resolve()
    files={source.name:source,runner.name:runner,args.request.name:args.request}
    for label,pin in request['pins'].items():
        target=Path(pin['path']);assert sha(target)==pin['sha256'],label;files[target.name]=target
    pre=Path(str(prefix)+'_preexecution.zip');capsule(pre,files)
    cmd=[request['python'],str(source),'--request',str(args.request),'--output-prefix',str(prefix)]
    started=time.time_ns();proc=subprocess.run(cmd,cwd=args.request.parent,capture_output=True,check=False);ended=time.time_ns()
    stdout=Path(str(prefix)+'_stdout.txt');stdout.write_bytes(proc.stdout)
    stderr=Path(str(prefix)+'_stderr.txt');stderr.write_bytes(proc.stderr)
    receipt={'kind':'NATURAL_EXIT_NOT_PROCESS_HEALTH','command':cmd,'cwd':str(args.request.parent),'started_unix_ns':started,
             'ended_unix_ns':ended,'elapsed_seconds':(ended-started)/1e9,'returncode':proc.returncode,'natural_exit_observed':True,
             'runtime':{'python':request['python'],'version':subprocess.check_output([request['python'],'--version'],text=True).strip(),
                        'platform':platform.platform()},'preexecution_zip_sha256':sha(pre),'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr)}
    exitpath=Path(str(prefix)+'_natural_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n')
    files.update({stdout.name:stdout,stderr.name:stderr,exitpath.name:exitpath})
    for suffix in ('_proof.json','_verdict.json'):
        path=Path(str(prefix)+suffix)
        if path.exists():files[path.name]=path
    executed=Path(str(prefix)+'_executed.zip');capsule(executed,files)
    delivery={'kind':'INDEPENDENT_REVIEW_EXECUTION_DELIVERY_NOT_ADMISSION','exit':receipt,
              'artifacts':{str(path):sha(path) for path in [pre,executed,exitpath,stdout,stderr]+[p for p in files.values() if p.name.startswith(prefix.name) and p.suffix=='.json']},
              'target_pins':request['pins'],'scope':request['scope'],'plan':request['plan'],'success':proc.returncode==0}
    deliverypath=Path(str(prefix)+'_delivery.json');deliverypath.write_text(json.dumps(delivery,indent=2)+'\n')
    print(json.dumps({'natural_returncode':proc.returncode,'elapsed_seconds':receipt['elapsed_seconds'],
                      'preexecution_sha256':sha(pre),'executed_sha256':sha(executed),'delivery':str(deliverypath),'delivery_sha256':sha(deliverypath)},indent=2),flush=True)
    raise SystemExit(proc.returncode)

if __name__ == '__main__':
    main()
