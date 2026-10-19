"""Read exact scientific capsules without executing any producer entrypoint."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parent


def main():
    inputs=json.loads((ROOT/sys.argv[1]).read_text())
    audits=[]
    for name in inputs['capsules']:
        path=ROOT/name
        with zipfile.ZipFile(path) as capsule:
            receipts=[member for member in capsule.namelist() if member.endswith('_exit.json')]
            if len(receipts)!=1:
                raise ValueError((name,'Expected exactly one exit receipt',receipts))
            receipt=json.loads(capsule.read(receipts[0]))
            maps={}
            for key in ('source_before','source_input_before','sources_and_inputs_sha256','outputs','output_hashes','outputs_sha256'):
                maps.update(receipt.get(key,{}))
            if not maps:
                raise ValueError((name,'No recognized source/input/output binding'))
            members={};current={}
            for member,expected in maps.items():
                actual=hashlib.sha256(capsule.read(member)).hexdigest()
                members[member]={'receipt':expected,'capsule':actual,'matches':actual==expected}
                local=ROOT/member
                if local.is_file():
                    current[member]=hashlib.sha256(local.read_bytes()).hexdigest()==expected
            audits.append({'capsule':name,'capsule_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                           'exit_receipt':receipts[0],'returncode':receipt.get('returncode',receipt.get('natural_exit')),
                           'natural_exit':receipt.get('natural_exit'),'source_input_output_binding':members,
                           'every_declared_byte_matches':all(item['matches']for item in members.values()),
                           'current_bytes_match_captured_revision':current})
    result={'kind':'EXACT_CAPSULE_SOURCE_INPUT_OUTPUT_BYTE_REVIEW_NOT_PRODUCER_REPLAY','inputs':inputs,'audits':audits,
            'all_declared_bytes_match':all(item['every_declared_byte_matches']for item in audits),
            'meaning':'Natural-exit receipts are captured evidence; this audit only reads hashes and exact archived/current bytes. Earlier failed scientific attempts remain immutable and are not promoted as stable final law.'}
    (ROOT/'rich_review_wave7_20261014_byte_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'all_declared_bytes_match':result['all_declared_bytes_match'],'capsules':[{'name':item['capsule'],'returncode':item['returncode'],'binding':item['every_declared_byte_matches'],'current_mismatches':[name for name,match in item['current_bytes_match_captured_revision'].items()if not match]}for item in audits]},indent=2))


if __name__=='__main__':
    main()
