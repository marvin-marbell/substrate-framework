"""Preserve native IR scientific and final selected-checker natural exits."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_native_ir_mechanism_'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
SOURCE=PREFIX+'consumer.py'
PROOF=PREFIX+'proof_corrected_20261010.json'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run(epoch: str, checker: bool) -> dict[str,Any]:
    source_bytes=(ROOT/SOURCE).read_bytes()
    tree=ast.parse(source_bytes)
    pins: dict[str,str]={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in node.targets):
            pins=ast.literal_eval(node.value)
    assert pins
    captured={name:(ROOT/name).read_bytes() for name in pins}
    assert all(sha(captured[name])==pin for name,pin in pins.items())
    for name in (SOURCE,PROOF,Path(__file__).name):captured[name]=(ROOT/name).read_bytes()
    label=PREFIX+epoch+('_checker' if checker else '')
    pre=ROOT/(label+'_preexecution.zip')
    assert not pre.exists(), 'Immutable epoch already exists; choose a new epoch'
    with zipfile.ZipFile(pre,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in captured.items():archive.writestr('preexecution/'+name,raw)
    config_bytes: bytes | None=None
    start=time.monotonic()
    if checker:
        with tempfile.TemporaryDirectory(prefix=PREFIX+'selected_') as tmp:
            config=Path(tmp)/'pyrightconfig.json'
            config_bytes=json.dumps({'include':[os.path.relpath(ROOT/SOURCE,tmp),os.path.relpath(Path(__file__),tmp)],
                                    'extraPaths':[str(ROOT)],'typeCheckingMode':'basic',
                                    'pythonVersion':'3.12','pythonPlatform':'Linux'},indent=2).encode()
            config.write_bytes(config_bytes)
            command=['/home/administrator/.local/bin/pyright','--project',str(config),'--pythonpath',PYTHON,
                     str(ROOT/SOURCE),str(Path(__file__).resolve()),'--outputjson']
            child=subprocess.run(command,cwd=ROOT,capture_output=True,check=False)
        selected=json.loads(child.stdout)
        assert selected['summary']['filesAnalyzed']==2
        assert not any('ignored' in str(row).lower() for row in selected.get('generalDiagnostics',[]))
    else:
        command=[PYTHON,str(ROOT/SOURCE),'--output',str(ROOT/(label+'_actual.json'))]
        child=subprocess.run(command,cwd=ROOT,capture_output=True,check=False)
    elapsed=time.monotonic()-start
    (ROOT/(label+'_stdout.txt')).write_bytes(child.stdout)
    (ROOT/(label+'_stderr.txt')).write_bytes(child.stderr)
    unchanged=all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    receipt={'kind':'SELECTED_ACTUAL_SOURCE_CHECKER' if checker else 'ACTUAL_PROJECTPYTHON_SCIENTIFIC_NATURAL_EXIT',
             'command':command,'cwd':str(ROOT),'projectPython':PYTHON,
             'natural_exit':child.returncode,'elapsed_seconds':elapsed,'stderr_empty':not child.stderr,
             'source_inputs_unchanged':unchanged,'preexecution_archive':pre.name,
             'preexecution_archive_sha256':sha(pre.read_bytes()),
             'captured_sha256':{name:sha(raw) for name,raw in captured.items()},
             'stdout_sha256':sha(child.stdout),'stderr_sha256':sha(child.stderr)}
    if checker:
        receipt['selected_files']=[SOURCE,Path(__file__).name]
        receipt['checker_summary']=selected['summary']
        receipt['owned_external_configuration_removed']=True
    elif child.returncode==0:
        scientific=ROOT/(label+'_actual.json');archive=scientific.with_suffix('.zip')
        receipt['actual_receipt_sha256']=sha(scientific.read_bytes())
        receipt['actual_executed_archive_sha256']=sha(archive.read_bytes())
    executed=ROOT/(label+'_execution.zip')
    with zipfile.ZipFile(executed,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in captured.items():archive.writestr('executed_bytes/'+name,raw)
        archive.writestr('stdout.txt',child.stdout);archive.writestr('stderr.txt',child.stderr)
        if config_bytes is not None:archive.writestr('external_config_bytes.json',config_bytes)
        archive.writestr('natural_exit.json',json.dumps(receipt,indent=2))
    receipt['execution_archive']=executed.name
    receipt['execution_archive_sha256']=sha(executed.read_bytes())
    (ROOT/(label+'_natural_exit.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    assert unchanged
    print(json.dumps({'natural_exit':child.returncode,'elapsed_seconds':elapsed,'stderr_empty':not child.stderr,
                      'source_inputs_unchanged':unchanged,'capsule':label+'_natural_exit.json'},indent=2))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch',required=True)
    parser.add_argument('--checker',action='store_true')
    args=parser.parse_args()
    result=run(args.epoch,args.checker)
    raise SystemExit(result['natural_exit'])
