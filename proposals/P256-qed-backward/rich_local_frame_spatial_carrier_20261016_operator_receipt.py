"""Bind compiled operator-only artifact, numeric generation and runtime layers."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_local_frame_spatial_carrier_20261016_'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    attempt=sys.argv[1]
    terminal=json.loads((ROOT/(P+attempt+'_exit.json')).read_text())
    if terminal.get('natural_exit')!=0:raise ValueError('Operator generation has no successful natural exit')
    inputs=json.loads((ROOT/(P+attempt+'_input.json')).read_text());build=json.loads((ROOT/(P+'build.json')).read_text())
    artifact=ROOT/(P+'model.pkl')
    runtime={f.name:digest(f) for f in sorted(ROOT.glob(P+'*.py'))}
    receipt=dict(kind='FROZEN_COMPILED_OPERATOR_ONLY_NO_SOLVED_STATE',artifact=str(artifact),artifact_bytes=artifact.stat().st_size,artifact_sha256=digest(artifact),generation_attempt=attempt,generation_exit=terminal,generation_source_sha256=inputs['source_sha256'],generation_source_capsule_sha256=digest(ROOT/(P+attempt+'_source_capsule.zip')),build_metadata_sha256=digest(ROOT/(P+'build.json')),law_signature=build['signature'],parameters=build['parameters'],graph_storage=build['full_graph_storage_audit'],runtime_source_sha256=runtime,runtime_layer_note='After the constructor starts, source preparation/image norm semantics, dynamical phase storage and lean raw-state I/O may be improved without changing the frozen numeric operator. Runtime exact source bytes are separately captured per state/evolution attempt. No operator data is reconstructed from evolved states.',versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','physics-tenpy','sympy')},loader='pickle.load(open(artifact,rb)); do not call Model() or full-MPO assembly. H is exact lazy bounded-MPO sum; own fresh preparation/evolution only.')
    path=ROOT/(P+'operator_receipt.json');path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('artifact','artifact_bytes','artifact_sha256','law_signature','parameters')}),flush=True)


if __name__=='__main__':main()
