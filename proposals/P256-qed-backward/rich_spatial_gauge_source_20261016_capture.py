"""Freeze declared inputs, capture actual child exit, preserve complete artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parent
P='rich_spatial_gauge_source_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'


def digest(name):
    return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


def dump(name,value):
    (ROOT/(P+name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


def main():
    prior=json.loads((ROOT/'rich_review_wave34_20261016_input.json').read_text())
    inputs=dict(prior['expected_input_SHA256'])
    # Preserve the known identities; never bless changed ancestor inputs.
    for name,expected in inputs.items():
        if digest(name)!=expected:
            raise ValueError('Frozen input changed '+name)
    selected=['family.py','consume.py','capture.py','equations.json']
    config=dict(plan_id='issue222-rich-backward-candidates-20261014',
        captured_revision='68b1a18dac8e728c1f1db869b97ab73de831d3f1e55c9c2c2297bf35ccbdcf7f',
        template='general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        work_item='interaction-candidate',interface_file=prior['interface_file'],
        expected_input_SHA256=inputs,source_SHA256={P+n:digest(P+n) for n in selected},
        units='hbar=1; original finite-law energy/time; geometry VERTICES in declared model length units',
        preparation='NEW matched neutral .35 then NEW full HN filtered source .35, root0 both physical LEFT components',
        measurement='B=.08 z, t=.18; all axis moment/current/contact matrices; two-sided own-H derivatives eps=.001',
        inputs_are_not_outputs='Only saved neutral is loaded. No saved charged state or evolved result is replayed.')
    dump('input',config)
    with zipfile.ZipFile(ROOT/(P+'source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for name in list(inputs)+[P+n for n in selected]+[P+'input.json']:
            z.write(ROOT/name,name)
    command=[PY,str(ROOT/(P+'consume.py'))]
    start=time.time();dump('start',dict(command=command,start_unix=start,cwd=str(ROOT)))
    with (ROOT/(P+'stdout.txt')).open('w') as out,(ROOT/(P+'stderr.txt')).open('w') as err:
        run=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err)
    record=dict(command=command,natural_exit=True,returncode=run.returncode,start_unix=start,
        end_unix=time.time(),elapsed_seconds=time.time()-start,
        frozen_inputs_unchanged=all(digest(name)==expected for name,expected in inputs.items()),
        source_unchanged=all(digest(name)==expected for name,expected in config['source_SHA256'].items()))
    dump('exit',record)
    print(json.dumps(record),flush=True)
    raise SystemExit(run.returncode)


if __name__=='__main__':
    main()
