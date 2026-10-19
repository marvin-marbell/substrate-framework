"""Record actual scientific exit, exact input/source bytes and selected checker."""
from __future__ import annotations
from io import BytesIO
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'induced_phi_fixed_gaussian_hermite_action_closure'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'
CHECKER = '/home/administrator/.local/bin/pyright'
sys.set_int_max_str_digits(0)

def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def run() -> None:
    before = {n:(ROOT/n).read_bytes() for n in (PREFIX+'.py', PREFIX+'_inputs.json', Path(__file__).name)}
    commands = [
        [PYTHON, PREFIX+'.py', '--inputs', PREFIX+'_inputs.json', '--output', PREFIX+'_receipt.json'],
        [CHECKER, '--pythonpath', PYTHON, '--stats', PREFIX+'.py', PREFIX+'_natural_exit.py'],
    ]
    records = []
    for command in commands:
        start = time.monotonic()
        child = subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        records.append(dict(command=command,cwd=str(ROOT),natural_exit=child.returncode,
                            stdout=child.stdout,stderr=child.stderr,wall_seconds=time.monotonic()-start))
        if child.returncode:
            capsule = dict(status='FAILED_NATURAL_EXIT',commands=records,
                           inputs_sha256={n:digest(b) for n,b in before.items()})
            (ROOT/(PREFIX+'_natural_exit_capsule.json')).write_text(json.dumps(capsule,indent=2)+'\n')
            print(child.stdout,child.stderr)
            raise SystemExit(child.returncode)
    receipt_raw = (ROOT/(PREFIX+'_receipt.json')).read_bytes()
    receipt = json.loads(receipt_raw)
    stable = all((ROOT/n).read_bytes()==b for n,b in before.items())
    if not stable or not receipt['input_unchanged'] or not all(receipt['source_unchanged'].values()):
        raise ArithmeticError('Natural execution source/input epoch changed')
    sources = {n:v['bytes_utf8'].encode() for n,v in receipt['source_at_execution'].items()}
    sources[Path(__file__).name] = before[Path(__file__).name]
    archive = BytesIO()
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for category,collection in (('source',sources),('selected-input',{PREFIX+'_inputs.json':before[PREFIX+'_inputs.json']}),('result',{PREFIX+'_receipt.json':receipt_raw})):
            for name,raw in sorted(collection.items()):
                info = zipfile.ZipInfo(category+'/'+name,date_time=(2026,10,10,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                z.writestr(info,raw)
    archive_raw = archive.getvalue()
    archive_name = PREFIX+'_immutable_'+digest(archive_raw)+'.zip'
    (ROOT/archive_name).write_bytes(archive_raw)
    capsule = dict(status='ACTUAL_SCIENCE_AND_FINAL_SELECTED_PYRIGHT_NATURAL_ZERO',commands=records,
        receipt_sha256=digest(receipt_raw),source_sha256={n:digest(b) for n,b in sources.items()},
        selected_input_sha256=digest(before[PREFIX+'_inputs.json']),immutable_archive=dict(path=archive_name,sha256=digest(archive_raw),bytes=len(archive_raw)),
        source_input_stable=stable,governing_assignment_revision='213ae802ebba544ced47e9cb553dde901d46dc57ca94beca9ce17256fa8e41a7',
        parent_current_revision='027e45d965834c9861190f672dd346fc3387dd8d4a6a73e0b5f9346e3de817f9',
        no_old_producer_or_root_rerun=True,selected_checker_files=[PREFIX+'.py',PREFIX+'_natural_exit.py'])
    (ROOT/(PREFIX+'_natural_exit_capsule.json')).write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps(dict(status=capsule['status'],receipt_sha256=capsule['receipt_sha256'],archive=capsule['immutable_archive'],prepared=receipt['outward_action']['action_PREPARED']),indent=2))

if __name__=='__main__':
    run()
