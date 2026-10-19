"""New two-polarization cross consumer using own immutable exact CAR helpers.
Loads definitions, never invokes the previous consumer main or any old producer.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any
import zipfile
import sympy as sp

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_bytes())
    archive_path=ROOT/(request['epoch']+'_preexecution.zip')
    with zipfile.ZipFile(archive_path) as archive:raw=archive.read('inputs/native_loop_dark_consumer_20261012.py')
    assert hashlib.sha256(raw).hexdigest()==request['pins']['native_loop_dark_consumer_20261012.py']
    namespace: dict[str, Any]={'__name__':'native_loop_dark_immutable_helpers','__file__':str(ROOT/'native_loop_dark_consumer_20261012.py')}
    exec(compile(raw,namespace['__file__'],'exec'),namespace)
    rows=[]
    prior=json.loads((ROOT/request['diagonal_receipt']).read_bytes())
    assert hashlib.sha256((ROOT/request['diagonal_receipt']).read_bytes()).hexdigest()==request['pins'][request['diagonal_receipt']]
    for index,n in enumerate((4,6,8)):
        modes=4*n**3;root=sp.exp(2*sp.pi*sp.I/n)
        first=namespace['face']((0,0,0),(1,2),n)
        total=sp.S.Zero
        for d in itertools.product(range(n),repeat=3):
            second=namespace['face'](d,(2,0),n)
            pattern=namespace['covariance_pattern'](first,second)
            value=namespace['face_covariance'](*pattern,modes)
            total+=sp.Rational(value.numerator,value.denominator)*root**d[2]
        cross=sp.simplify(sp.expand_complex(total))
        diagonal: Any=sp.sympify(prior['normalized_physical1_spatial_attempt']['rows'][index]['two_transverse_polarizations'][0]['normalized_fourier_covariance_over_c0_pow8'])
        gram: Any=sp.Matrix([[diagonal,cross],[sp.conjugate(cross),diagonal]])
        determinant=sp.simplify(gram.det())
        assert determinant>=0
        rows.append({'N':n,'q_index':[0,0,1],'physical_hopping':1,'normalization':'V^-1/2 per source, separate canonical extraction in all65 factors',
                     'conditional_full_complex_polarization_Gram_over_native_c0_pow8':str(gram),
                     'cross':str(cross),'determinant':str(determinant),'rank':2 if determinant>0 else 1,
                     'wholeP0_removed':False,'originating_actual_ground':False})
    result={'kind':'FULL64_NORMALIZED_CONDITIONAL_NATIVE_COMPLEX_POLARIZATION_MATRIX_NOT_SPECTRAL_TRANSFER',
            'rows':rows,'definitions_selected_from_preexecution_zip':True,'old_main_executed':False,
            'actual_IR_atom':'UNPAID: these are exact canonical conditional equal-time forms, not SAME-L wholeP0 spectral forms',
            'governor':request['governor'],'campaign_complete':False}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'receipt':args.output.name,'conditional_full_polarization_matrix_completed':True,'actual_ground_transfer':'UNPAID'}))


if __name__=='__main__':main()
