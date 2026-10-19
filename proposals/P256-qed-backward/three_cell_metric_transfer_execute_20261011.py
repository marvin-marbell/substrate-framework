"""Immutable pre/executed capsules and natural exits for the new consumer only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
CONSUMER='three_cell_metric_transfer_consumer_20261011.py'
DENSITY='three_cell_density_formation_interface_20261011.json'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main(epoch):
 stem=f'three_cell_metric_transfer_{epoch}_20261011'
 # Loading source for its static pin dictionary is unnecessary: the manifest
 # takes the exact scientific prerequisites, independently named here.
 names=[CONSUMER,Path(__file__).name,'three_cell_metric_transfer_proof_20261011.json',
  DENSITY,'three_cell_density_formation_proof_20261011.json',
  'three_cell_density_formation_consumer_20261011.py','three_cell_density_formation_actual2_20261011_exit.json',
  'three_cell_native_actual3_20261011_receipt.json','three_cell_native_scientist_20261011.py',
  'three_cell_native_proof_20261011.json','charged_first_shell_actual04_20261011_receipt.json',
  'connected_root_exterior_actual1_20261011_moments.npz','harmonic_car_bridge_typed_attempt3_20261011_receipt.json',
  'connected_root_exterior_scientist_20261011.py','joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz']
 assert not any((ROOT/f'{stem}_{suffix}').exists() for suffix in ['manifest.json','preexecution.zip','executed.zip','receipt.json','exit.json'])
 before={name:sha(ROOT/name) for name in names}
 command=[sys.executable,str(ROOT/CONSUMER),'--density',str(ROOT/DENSITY),'--output',str(ROOT/f'{stem}_receipt.json')]
 manifest={'kind':'IMMUTABLE_NEW_INTERVAL_CONSUMER_MANIFEST','command':command,'source_and_input_SHA':before,
  'density_capsule_external':'three_cell_density_formation_actual2_20261011_executed.zip',
  'density_capsule_SHA':sha(ROOT/'three_cell_density_formation_actual2_20261011_executed.zip')}
 manifest_path=ROOT/f'{stem}_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
 pre=ROOT/f'{stem}_preexecution.zip'
 with zipfile.ZipFile(pre,'x',zipfile.ZIP_DEFLATED) as z:
  for name in names:z.write(ROOT/name,name)
  z.write(manifest_path,manifest_path.name)
 start=time.monotonic()
 with (ROOT/f'{stem}_stdout.txt').open('w') as out,(ROOT/f'{stem}_stderr.txt').open('w') as err:
  proc=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,check=False)
 elapsed=time.monotonic()-start
 after={name:sha(ROOT/name) for name in names}
 core={'returncode':proc.returncode,'elapsed_seconds':elapsed,'source_and_input_SHA':before,
  'sources_inputs_unchanged':after==before,'preexecution_archive':pre.name,'preexecution_SHA':sha(pre),
  'natural_exit':True,'receipt_present':(ROOT/f'{stem}_receipt.json').exists()}
 exitpath=ROOT/f'{stem}_exit.json';exitpath.write_text(json.dumps(core,indent=2)+'\n')
 executed=ROOT/f'{stem}_executed.zip'
 with zipfile.ZipFile(executed,'x',zipfile.ZIP_DEFLATED) as z:
  for name in names:z.write(ROOT/name,name)
  for suffix in ['manifest.json','stdout.txt','stderr.txt','exit.json','receipt.json']:
   path=ROOT/f'{stem}_{suffix}'
   if path.exists():z.write(path,path.name)
 core.update({'executed_archive':executed.name,'executed_SHA':sha(executed),
  'archive_exit_convention':'Archive stores immutable natural-exit core; disk exit adds executed hash to avoid circular self-hash.'})
 exitpath.write_text(json.dumps(core,indent=2)+'\n')
 print(json.dumps(core,indent=2),flush=True)
 return proc.returncode

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--epoch',required=True);a=p.parse_args();sys.exit(main(a.epoch))
