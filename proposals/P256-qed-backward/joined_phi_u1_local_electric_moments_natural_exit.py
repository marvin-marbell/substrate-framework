"""One selected numeric natural-exit capture plus exact/outward consumer checks."""
from __future__ import annotations
from fractions import Fraction as F
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import sympy as sy

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_local_electric_moments'

def run() -> int:
    started=time.monotonic();source=ROOT/(PREFIX+'.py');own=Path(__file__)
    raw=source.read_bytes();ownraw=own.read_bytes()
    names=[n.name for n in ast.walk(ast.parse(raw)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    if not {'request_wind','inserted','directional_insertion','evolve'}.issubset(names): raise ArithmeticError('Positive actual AST missing')
    language=ROOT/(PREFIX+'_final_language_receipt.json')
    language_command=[sys.executable,'/tmp/issue222-em-scoped-lsp.py',str(source),str(language),'root_upper']
    language_run=subprocess.run(language_command,cwd=ROOT,capture_output=True)
    (ROOT/(PREFIX+'_language_stdout.txt')).write_bytes(language_run.stdout)
    (ROOT/(PREFIX+'_language_stderr.txt')).write_bytes(language_run.stderr)
    if language_run.returncode: raise ArithmeticError('Actual final LSP diagnostics failed: '+language_run.stdout.decode(errors='replace')+language_run.stderr.decode(errors='replace'))
    output=ROOT/(PREFIX+'_receipt.json')
    command=[sys.executable,str(source),'--output',str(output),'--order','4']
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    process=subprocess.run(command,cwd=ROOT,env=env,capture_output=True)
    stdout=ROOT/(PREFIX+'_stdout.txt');stderr=ROOT/(PREFIX+'_stderr.txt')
    stdout.write_bytes(process.stdout);stderr.write_bytes(process.stderr)
    capsule={'command':command,'natural_numeric_exit':process.returncode,'source_sha256':hashlib.sha256(raw).hexdigest(),'runner_sha256':hashlib.sha256(ownraw).hexdigest(),'elapsed_seconds':time.monotonic()-started,'source_stable':source.read_bytes()==raw and own.read_bytes()==ownraw,
             'interpreter_sha256':hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),'stdout':{'path':stdout.name,'sha256':hashlib.sha256(process.stdout).hexdigest()},'stderr':{'path':stderr.name,'sha256':hashlib.sha256(process.stderr).hexdigest()},'positive_AST':names,'final_language_receipt':{'path':language.name,'sha256':hashlib.sha256(language.read_bytes()).hexdigest()}}
    capsule_path=ROOT/(PREFIX+'_natural_exit_capsule.json')
    capsule_path.write_text(json.dumps(capsule,indent=2)+'\n')
    if process.returncode:
        print(json.dumps(capsule),flush=True)
        return process.returncode
    result=json.loads(output.read_bytes());ctrl=result['operator_controls']
    if result['clock_exact']!='1/65536' or len(result['native_signed_path_price_ids'])!=768:
        raise ArithmeticError('Full native clock/topology missing')
    if ctrl['native_counts']['one']!=384 or ctrl['native_counts']['two']!=384 or not ctrl['native_counts']['AP_negative_forward']:
        raise ArithmeticError('Literal one/two/AP scope narrowed')
    if ctrl['full_cross_exact_defect'] or ctrl['faithful_Weyl_operator_words']['full']['phase_angle_exact']!='0':
        raise ArithmeticError('Full electric Gauss cancellation failed')
    if any(ctrl['faithful_Weyl_operator_words'][key]['phase_angle_exact']=='0' for key in ('omittedHC','HE32_HC64')):
        raise ArithmeticError('MissingHC/unequal electric controls blind')
    if ctrl['omitted_torons_disjoint_cross']=='0' or ctrl['E0_omittedRef_commutator_energy_shift']=='0':
        raise ArithmeticError('Toron/Ref controls blind')
    if F(ctrl['full_sigma2_exact'])<=F(ctrl['missing_aux_sigma2_exact']) or ctrl['Gaussian_Pi_moments_exact']['2']==ctrl['wrong_unit_Pi_fourth_exact']:
        raise ArithmeticError('Auxiliary/unitvariance controls blind')
    # Independent exact noncommuting two-state Slater Hessian; wrong /4 fails.
    p=sy.Matrix([[sy.Rational(1,2)]*2]*2);v=sy.diag(-1,1)
    actual=sy.trace(p*v*v)-sy.trace(p*v*p*v)
    if actual!=1 or actual<=sy.trace(v*v)/4 or actual!=sy.trace(v*v)/2:
        raise ArithmeticError('Noncommuting variance price is wrong')
    for toron in result['toron_winds']:
        if F(toron['q_squared_exact'])!=64 or F(toron['density_v_squared_exact'])!=0:
            raise ArithmeticError('Full3 torons missing')
    prices=result['price_catalogue']
    if not result['linked_words'] or not any(row[2]==-1 for row in result['linked_words']):
        raise ArithmeticError('Linked composite differences missing')
    for price in prices:
        q2=F(price['q_squared_exact']);v2=F(price['density_v_squared_exact'])
        if F(price['centered_sigma_squared_upper_exact'])!=q2/8+128*v2:
            raise ArithmeticError('Full64species variance price altered')
        force=list(map(F,price['force_Lp_upper']))
        for label,mass in [('background',F(1)),('raw_source',F(1,2))]:
            state=price['states'][label];initial=list(map(F,state['initial_a_upper']));future=list(map(F,state['horizon_a_upper']))
            if F(state['mass_exact'])!=mass or initial[0]*initial[0]<mass or future[0]!=initial[0]:
                raise ArithmeticError('Raw source norm changed')
            if any(b<a for a,b in zip(initial,future)):
                raise ArithmeticError('Nonpositive triangular comparison')
            # Independently solve polynomial triangular integral, not matrix exponential copy.
            polys=[[initial[k]] for k in range(5)]
            for k in range(1,5):
                for j in range(k):
                    coefficient=F(__import__('math').comb(k,j))*force[k-j]
                    for degree,value in enumerate(polys[j]):
                        while len(polys[k])<degree+2:polys[k].append(F(0))
                        polys[k][degree+1]+=coefficient*value/(degree+1)
            endpoint=[sum((c*F(1,65536)**n for n,c in enumerate(poly)),F(0)) for poly in polys]
            if endpoint!=future: raise ArithmeticError('Analytic hierarchy horizon differs')
    example=result['paid_insertion_examples'][2]
    if example['electric_shift_exact']!=0: raise ArithmeticError('Opposite-word overlap cancellation failed')
    for example in result['paid_insertion_examples']:
        if F(example['word_norm_upper'])<=0: raise ArithmeticError('Unpaid insertion coefficient')
    archive=ROOT/result['input_bytes']['path']
    with zipfile.ZipFile(archive) as z:
        if any(hashlib.sha256(z.read(name)).hexdigest()!=sha for name,sha in result['source_input_sha256'].items()):
            raise ArithmeticError('Source capsule SHA mismatch')
    capsule.update({'receipt_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'input_bytes':result['input_bytes'],'runtime':result['runtime'],'checker_exact_variance':str(actual),'checked_price_classes':len(prices),'checked_composite_words':len(result['linked_words']),
                    'scientific_scope':'Derived exact/outward full-H local moment comparison and paid insertion vectors. Not sign/pole/cone/charged redistribution/source admission.'})
    capsule_path.write_text(json.dumps(capsule,indent=2)+'\n')
    print(json.dumps(capsule),flush=True)
    return 0

if __name__=='__main__':sys.exit(run())
