"""Freeze source/input bytes; run only NEW flux CLI, then selected language diagnostics."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import zipfile
from typing import Any

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
SERVER='/home/administrator/.local/bin/pyright-langserver'
SCIENCE='smoothed_flux_principal_filter_20261011.py'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def archive(path: Path, payload: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in payload.items():
            z.writestr(name,raw)


def language() -> dict[str, Any]:
    p=subprocess.Popen([SERVER,'--stdio'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=PROJECT)
    q: queue.Queue[Any]=queue.Queue(); notifications=[]
    def receive() -> None:
        assert p.stdout is not None
        while True:
            headers={}
            while True:
                line=p.stdout.readline()
                if not line:return
                if line==b'\r\n':break
                key,value=line.decode().split(':',1);headers[key.lower()]=value.strip()
            q.put(json.loads(p.stdout.read(int(headers['content-length']))))
    threading.Thread(target=receive,daemon=True).start()
    def send(obj: Any) -> None:
        assert p.stdin is not None
        raw=json.dumps(obj).encode()
        p.stdin.write(f'Content-Length: {len(raw)}\r\n\r\n'.encode()+raw);p.stdin.flush()
    def req(i: int, method: str, params: Any) -> Any:
        send(dict(jsonrpc='2.0',id=i,method=method,params=params))
        while True:
            obj=q.get(timeout=60)
            if 'method' in obj and 'id' in obj:
                send(dict(jsonrpc='2.0',id=obj['id'],result=None))
            elif 'method' in obj:notifications.append(obj)
            if obj.get('id')==i:return obj
    init=req(1,'initialize',dict(processId=None,rootUri=PROJECT.as_uri(),capabilities={}))
    send(dict(jsonrpc='2.0',method='initialized',params={}))
    send(dict(jsonrpc='2.0',method='workspace/didChangeConfiguration',params=dict(settings=dict(python=dict(pythonPath=PYTHON)))))
    reports=[]
    for i,name in enumerate((SCIENCE,Path(__file__).name)):
        src=ROOT/name; text=src.read_text()
        send(dict(jsonrpc='2.0',method='textDocument/didOpen',params=dict(textDocument=dict(uri=src.as_uri(),languageId='python',version=1,text=text))))
        symbols=req(2+i,'textDocument/documentSymbol',dict(textDocument=dict(uri=src.as_uri())))
        tree=ast.parse(text)
        functions=[node.name for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)]
        assert symbols.get('result') and functions
        reports.append(dict(source=name,symbols=symbols,AST_functions=functions))
    src=ROOT/SCIENCE
    tree=ast.parse(src.read_text())
    call=next(node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='incidence')
    definition=req(5,'textDocument/definition',dict(textDocument=dict(uri=src.as_uri()),position=dict(line=call.lineno-1,character=call.col_offset)))
    assert definition.get('result')
    end=time.monotonic()+4
    while time.monotonic()<end:
        try:
            obj=q.get(timeout=end-time.monotonic())
            if 'method' in obj:notifications.append(obj)
        except queue.Empty:break
    req(6,'shutdown',None);send(dict(jsonrpc='2.0',method='exit',params=None));p.wait(timeout=10)
    diagnostics=[n for n in notifications if n.get('method')=='textDocument/publishDiagnostics']
    return dict(project_root=str(PROJECT),server=init,binary=SERVER,reports=reports,definition=definition,
                diagnostics=diagnostics,exit=p.returncode,diagnostics_not_claimed_green=True,
                mounted_lsp_device_unavailable=True)


def run(epoch: str, request_path: Path) -> int:
    assert epoch.startswith('smoothed_flux_') and '/' not in epoch
    paths={key:ROOT/(epoch+'_'+key) for key in ('preexecution.zip','executed.zip','stdout.txt','stderr.txt','receipt.json','language.json','exit.json','capsule.json')}
    assert not any(path.exists() for path in paths.values()),'Immutable epoch exists'
    request=json.loads(request_path.read_text())
    names=list(request['input_sha256'])+[request_path.name,Path(__file__).name,SCIENCE]
    captured={name:(ROOT/name).read_bytes() for name in dict.fromkeys(names)}
    for name,pin in request['input_sha256'].items():
        assert sha(captured[name])==pin,name
    for filename,pin in request['external_input_sha256'].items():
        raw=Path(filename).read_bytes();assert sha(raw)==pin,filename
        captured['external/'+Path(filename).name]=raw
    argv=[PYTHON,str(ROOT/SCIENCE),'--request',str(request_path),'--output',str(paths['receipt.json'])]
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    manifest=dict(kind='PREEXECUTION_NEW_NATIVE_FLUX_OPERATOR_EPOCH',argv=argv,cwd=str(ROOT),
        source_input_sha256={name:sha(raw) for name,raw in captured.items()},external_input_sha256=request['external_input_sha256'],
        old_producers_or_solvers_run=False)
    captured['smoothed_flux_preexecution_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    archive(paths['preexecution.zip'],captured)
    started=time.monotonic();proc=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,check=False)
    elapsed=time.monotonic()-started
    paths['stdout.txt'].write_bytes(proc.stdout);paths['stderr.txt'].write_bytes(proc.stderr)
    final_language=language()
    paths['language.json'].write_text(json.dumps(final_language,indent=2)+'\n')
    unchanged=all((ROOT/name).read_bytes()==captured[name] for name in dict.fromkeys(names))
    external_unchanged=all(sha(Path(name).read_bytes())==pin for name,pin in request['external_input_sha256'].items())
    info=dict(kind='NEW_SCIENTIFIC_NATURAL_EXIT',returncode=proc.returncode,natural_exit=proc.returncode>=0,
        elapsed_seconds=elapsed,sources_inputs_unchanged=unchanged,external_inputs_unchanged=external_unchanged,argv=argv,
        preexecution_sha256=sha(paths['preexecution.zip'].read_bytes()),stdout_sha256=sha(proc.stdout),stderr_sha256=sha(proc.stderr),
        receipt_sha256=sha(paths['receipt.json'].read_bytes()) if paths['receipt.json'].exists() else None,
        language_sha256=sha(paths['language.json'].read_bytes()),language_exit=final_language['exit'])
    paths['exit.json'].write_text(json.dumps(info,indent=2)+'\n')
    executed=dict(captured)
    for key in ('stdout.txt','stderr.txt','receipt.json','language.json','exit.json'):
        if paths[key].exists():executed[paths[key].name]=paths[key].read_bytes()
    archive(paths['executed.zip'],executed)
    capsule=dict(epoch=epoch,**info,executed_sha256=sha(paths['executed.zip'].read_bytes()),
        source_input_sha256=manifest['source_input_sha256'],files={key:path.name for key,path in paths.items()},
        failed_epochs_preserved=True,physical_ground_or_atom=False,build_lint_tests_formatters_run=False)
    paths['capsule.json'].write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps(capsule,indent=2))
    return proc.returncode if unchanged and external_unchanged else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch',required=True)
    parser.add_argument('--request',required=True,type=Path)
    args=parser.parse_args()
    sys.exit(run(args.epoch,args.request))
