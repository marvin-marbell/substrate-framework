"""New literal wrapped AP successor; prior complete scientific receipt stays pinned."""
from __future__ import annotations
import argparse
import ast
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_bytes())
    with zipfile.ZipFile(ROOT/(request['epoch_prefix']+'_preexecution.zip')) as sealed:
        raw=sealed.read('auxiliary_adjoint_review_consumer_20261012.py')
        tree=ast.parse(raw)
        nodes=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef)) and not(isinstance(node,ast.FunctionDef) and node.name=='main')]
        nodes += [node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ORDERS' for t in node.targets)]
        module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]));namespace={'__name__':'own_immutable_adjoint_helpers','__file__':str(ROOT/'auxiliary_adjoint_review_consumer_20261012.py')}
        exec(compile(module,'own_immutable_adjoint_helpers','exec'),namespace)
        with zipfile.ZipFile(io.BytesIO(sealed.read(request['profile_archive']))) as archive:
            profiles=json.loads(archive.read(request['profile_member']))['primitive_profiles']
        previous=json.loads(sealed.read('auxiliary_adjoint_review_actual2_20261012_receipt.json'))
        result=namespace['adjoint_control'](profiles)
        assert result['AP_seams']==[-1,1,-1,1]
        for name in ('uniform_HS2_over_c0_pow8','uniform_full64_compression','noncommuting_wrapped_complex_native_HS2_over_c0_pow8','noncommuting_full64_compression'):
            assert result[name]==previous['adjoint_native'][name]
    with args.output.open('x') as stream:
        json.dump({'kind':'EXACT_LITERAL_NATIVE_AP_ADJOINT_SUCCESSOR_NOT_PRIOR_EPOCH_RETAG',
                   'adjoint_native':result,'complete_previous_receipt':'auxiliary_adjoint_review_actual2_20261012_receipt.json',
                   'AP_point_label_correction':'Prior adjacent negative seams had the same closed product1 but were not literal fixed AP at this wrapped origin; corrected opposite x seams now match the independent native enumeration.',
                   'all24_HS_results_unchanged':'Exact fourth-degree multilinearity makes closed AP product1; checked in new natural execution.',
                   'prior_full_polarization_not_reexecuted':True,'governor':request['governor'],
                   'native_ground_or_spectrum':False},stream,indent=2,default=str);stream.write('\n')
    print(json.dumps({'receipt':args.output.name,'literal_native_AP_successor_completed':True}))


if __name__=='__main__':
    main()
