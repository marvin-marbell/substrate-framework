"""Archive new scientific sources/inputs before execution; retain failed epochs."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='vacuum_connected_native_'
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def source_closure(names):
    queue=[name for name in names if name.endswith('.py')]
    seen=set(queue)
    while queue:
        name=queue.pop()
        tree=ast.parse((ROOT/name).read_text(),filename=name)
        modules=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module:
                modules.append(node.module)
        for module in modules:
            candidate=module.split('.')[0]+'.py'
            if (ROOT/candidate).is_file() and candidate not in seen:
                seen.add(candidate);queue.append(candidate)
    return seen


def run(epoch):
    request_path=ROOT/f'{PREFIX}{epoch}_request.json'
    outputs=[ROOT/f'{PREFIX}{epoch}_{suffix}' for suffix in
             ('request.json','preexecution.zip','executed.zip','exit.json','stdout.txt','stderr.txt','capsule.json','receipt.json')]
    if any(path.exists() for path in outputs):
        raise FileExistsError('Immutable epoch exists; use a new epoch, never overwrite failures')
    names={Path(__file__).name,f'{PREFIX}scientist_20261011.py',f'{PREFIX}proof_20261011.json',f'{PREFIX}language_20261011.json',
           f'{PREFIX}actual1_20261011_invalidation.json',
           'central_neutral_vertex_interface_20261011.json','central_neutral_vertex_proof_20261011.json',
           'charged_outer_vector_interface_20261011.json','charged_outer_vector_proof_20261011.json',
           'vacuum_connected_schur_method_actual2_20261011_receipt.json',
           'joined_phi_u1_normalized_defect_star_operator_parent_packet_20261011.json',
           'joined_phi_u1_normalized_defect_star_operator_postacceptance_snapshot_20261011.zip',
           'joined_phi_u1_native_ir_mechanism_proof_corrected_20261010.json',
           'joined_phi_u1_native_ir_mechanism_mathreview_NativeSchurProofReview_20261010.json',
           'joined_phi_u1_full_physical_source_local_ground_parent_packet_20261011.json',
           'joined_phi_u1_ir_uniform_charged_parent_packet_20261010.json'}
    # Historical capsule members are inputs, not new runtime preexecution proof.
    pins={}
    for iface in ('central_neutral_vertex_interface_20261011.json','charged_outer_vector_interface_20261011.json'):
        value=json.loads((ROOT/iface).read_bytes())
        for name,pin in value['input_sha256'].items():
            if name in pins and pins[name]!=pin:
                raise ArithmeticError(('Historical input disagreement',name))
            pins[name]=pin
    names.update(pins)
    names.update(source_closure(names))
    current={name:sha(ROOT/name) for name in sorted(names)}
    for name,pin in pins.items():
        if current[name]!=pin:
            raise ArithmeticError(('Historical pin mismatch',name))
    exact_pins={
        'central_neutral_vertex_interface_20261011.json':'2f9faa2fcc4201b98a0e1b98edb49b46066c3bffd0f5176b1caaa31dc88402e3',
        'central_neutral_vertex_proof_20261011.json':'fa7226a9a9e7bf7152245a9601f37653eab256025c6a1a5aa5f1e702163f8e14',
        'charged_outer_vector_interface_20261011.json':'eb99d207b4af5536bb6ed137ce1ef364149b796e94cd5373675752acc0e29a50',
        'charged_outer_vector_proof_20261011.json':'bcf4cbc0ea521c1b67388dc72691e48eff7309eb5081aeb37f8cb8bedbab593a',
        'vacuum_connected_schur_method_actual2_20261011_receipt.json':'af9f06612bef762321fbb9848f6030faf1705e50f02883f3213f2c867f8aaa9c',
        'joined_phi_u1_normalized_defect_star_operator_parent_packet_20261011.json':'98bbd97d979e34673d6e5b8025657cdb51bc963c3dda8629dd622c8a95dd5c27',
        'joined_phi_u1_normalized_defect_star_operator_postacceptance_snapshot_20261011.zip':'a7c811a233c9e8e3a3eacfadc374bdca7de7b73491e7f66872fa6ceaaa4193ad'}
    for name,pin in exact_pins.items():
        if current[name]!=pin:
            raise ArithmeticError(('Governing contract pin mismatch',name))
    request={'kind':'NEW_NATIVE_SCIENTIFIC_REQUEST_NOT_HISTORICAL_REEXECUTION','epoch':epoch,
             'input_sha256':current,'governing_plan':{'id':'issue-222-induced-photon-delta13-continuation-20261008',
             'captured_revision':'452eb157f253695e4f03bb98d41c71ac660e559a02a019c3502a6f59b9a8babc',
             'accepted_scoped_science':'452eb157f253695e4f03bb98d41c71ac660e559a02a019c3502a6f59b9a8babc',
             'template_id':'general','template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
             'work_item':'induced-photon'},'python':PYTHON,
             'control_and_baseline_are_inputs_not_native_preexecution_proof':True}
    dump(request_path,request)
    pre=ROOT/f'{PREFIX}{epoch}_preexecution.zip'
    with zipfile.ZipFile(pre,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in current:
            archive.write(ROOT/name,'inputs/'+name)
        archive.write(request_path,'request.json')
    began=time.monotonic()
    completed=subprocess.run([PYTHON,str(ROOT/f'{PREFIX}scientist_20261011.py'),'--epoch',epoch],cwd=ROOT,
                             capture_output=True,check=False)
    elapsed=time.monotonic()-began
    stdout=ROOT/f'{PREFIX}{epoch}_stdout.txt';stderr=ROOT/f'{PREFIX}{epoch}_stderr.txt'
    stdout.write_bytes(completed.stdout);stderr.write_bytes(completed.stderr)
    unchanged=all(sha(ROOT/name)==pin for name,pin in current.items())
    exit_path=ROOT/f'{PREFIX}{epoch}_exit.json'
    dump(exit_path,{'kind':'NEW_NATIVE_SCIENTIFIC_NATURAL_EXIT','returncode':completed.returncode,
                    'natural_exit':completed.returncode>=0,'elapsed_seconds':elapsed,
                    'sources_inputs_unchanged':unchanged,'all_source_count':sum(n.endswith('.py') for n in current),
                    'all_input_count':len(current),'input_sha256':current,
                    'preexecution_sha256':sha(pre),'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr),
                    'old_cli_solver_root_checker_rerun':False})
    executed=ROOT/f'{PREFIX}{epoch}_executed.zip'
    receipt=ROOT/f'{PREFIX}{epoch}_receipt.json'
    with zipfile.ZipFile(executed,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in current:
            archive.write(ROOT/name,'executed_inputs/'+name)
        for path in (request_path,exit_path,stdout,stderr):
            archive.write(path,path.name)
        if receipt.exists():
            archive.write(receipt,receipt.name)
    capsule={'kind':'IMMUTABLE_NEW_NATIVE_ALL_SOURCE_NATURAL_EXIT_CAPSULE',
             'epoch':epoch,'returncode':completed.returncode,'natural_exit':completed.returncode>=0,
             'sources_inputs_unchanged':unchanged,'elapsed_seconds':elapsed,
             'preexecution':{'path':pre.name,'sha256':sha(pre)},
             'executed':{'path':executed.name,'sha256':sha(executed)},
             'exit':{'path':exit_path.name,'sha256':sha(exit_path)},
             'receipt':{'path':receipt.name,'sha256':sha(receipt)} if receipt.exists() else None,
             'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr),
             'limitations':'Only new scientific calculation; no build/lint/tests/formatter/selected-Pyright. Finite frozen reference, not interacting phase or photon.'}
    dump(ROOT/f'{PREFIX}{epoch}_capsule.json',capsule)
    print(json.dumps(capsule),flush=True)
    return completed.returncode if unchanged else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch',required=True)
    sys.exit(run(parser.parse_args().epoch))
