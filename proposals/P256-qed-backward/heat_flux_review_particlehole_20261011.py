"""NEW exact finite-CAR particle-hole symmetry proof consumer; no old CLI."""
import argparse
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from typing import cast
ROOT = Path(__file__).resolve().parent
INPUTS = ['flux_laplace_actual1_20261011_preexecution.zip', 'charged_cell_heat_interface_20261011.json']
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path, value): path.write_text(json.dumps(value, indent=2, default=int)+'\n')
def control():
    import sympy as s
    pins = {p: sha(ROOT/p) for p in INPUTS}
    sys.path.insert(0, str(ROOT/INPUTS[0]))
    import harmonic_car_bridge_scientist_20261011 as native
    Jnative, maps, bases, ops, ann, V = native.car_exact()  # Pure exact map declarations only.
    J4 = s.Matrix([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])
    swap = (1,0,3,2)
    U = s.zeros(16)
    for mask in range(16):
        image = s.eye(16)[:,15]
        for p in reversed([p for p in range(4) if (mask>>p)&1]):
            image = ann[swap[p]]*image
        U[:,mask] = image
    assert U.T*U == s.eye(16)
    for p in range(4):
        assert U*ann[p]*U.T == ann[swap[p]].T
    assert U*ops[3]*U.T == -ops[3]
    for h in bases[:3]:
        assert J4*h*J4 == -h.T
    with zipfile.ZipFile(ROOT/INPUTS[0]) as archive:
        profiles = json.loads(archive.read('local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json'))['profiles']
    for profile in profiles:
        h = cast(s.MutableDenseMatrix, sum((s.Rational(profile[key])*base for key,base in zip(('m_exact','c0_exact','d_exact'),bases[:3])),s.zeros(4)))
        D = sum((ann[p].T*ann[q]*h[p,q] for p in range(4) for q in range(4)),s.zeros(16))
        assert s.trace(h) == 0 and U*D*U.T == D
        for r in (0,1,-1):
            R = ops[3]+r*s.eye(16)
            # Whole-cell spectator density must flip along with selected replica.
            assert U*(D+64*(R-s.eye(16)/2)**2)*U.T == D+64*(-ops[3]+r*s.eye(16)-s.eye(16)/2)**2
    slotphase = s.diag(1,-s.I,1,1,s.I)
    W = V*Jnative*slotphase
    assert U*W == W*(W.conjugate().T*U*W)
    U5 = s.simplify(W.conjugate().T*U*W)
    # General all-n harmonic trace metric, not a small-n extrapolation.
    metric = s.zeros(5)
    metric[0,0] = 1
    metric[1,4] = metric[4,1] = -1
    metric[2,3] = metric[3,2] = 1
    assert s.simplify(U5.T*metric*U5) == metric
    assert s.simplify(U5.conjugate().T*U5) == s.eye(5)
    for h in ops[:3]:
        assert s.simplify(U5*(W.conjugate().T*h*W)*U5.conjugate().T-W.conjugate().T*h*W) == s.zeros(5)
    assert s.simplify(U5*(W.conjugate().T*ops[3]*W)*U5.conjugate().T+W.conjugate().T*ops[3]*W) == s.zeros(5)
    result = {'kind':'EXACT_NATIVE_PARTICLEHOLE_SYMMETRY_CONSUMER', 'own_source_sha256':sha(Path(__file__)), 'input_sha256':pins, 'J4':J4.tolist(), 'U16':U.tolist(), 'U5_native_harmonic':[[str(z) for z in row] for row in U5.tolist()], 'CAR_c_to_J_cdagger':True, 'actual_profile_D_invariant':True, 'rho_flips':True, 'native_W5_invariant':True, 'harmonic_trace_metric_preserved_all_n':True, 'whole_cell_condition':'Apply U to ALL128 labelled replicas, not just charged spinor; every spectator R flips. Induced Sym(n) action preserves harmonic trace kernel because U5.T metric U5=metric. Each neutral labelled Nj=2 is preserved; charged selected Nj1 and3 exchange, and total cell255 and257 exchange.', 'ground_condition':'Only if accepted unique g belongs to this invariant frozen canonical module does Cg=phase*g follow. No exact symmetry of approximate source s is inferred.', 'heat_identity':'With frozen unique g and native indices, Gminus[q,p]=Gplus[Jp,Jq], equivalently Gminus(t)=J2 Gplus(t).T J2. No complex conjugation beyond the indicated transpose; C is linear unitary, not antiunitary.', 'example':'Gplus=[[a,ib],[-ib,d]] implies Gminus=[[d,ib],[-ib,a]]. Actual g identity cannot be imposed directly on uncertain s-centered M moments without paid ground transfer.', 'no_physical_ground_or_allorder_claim':True, 'old_CLI_executed':False}
    assert pins == {p:sha(ROOT/p) for p in INPUTS}
    dump(ROOT/f'{PREFIX}_receipt.json',result)
    print(json.dumps({k:result[k] for k in ('kind','own_source_sha256','U5_native_harmonic','heat_identity')}),flush=True)
def execute():
    request={'source_sha256':sha(Path(__file__)), 'input_sha256':{p:sha(ROOT/p) for p in INPUTS}, 'command':[sys.executable,str(Path(__file__)),'--control','--epoch',args.epoch]}
    dump(ROOT/f'{PREFIX}_request.json',request)
    pre=ROOT/f'{PREFIX}_preexecution.zip'
    with zipfile.ZipFile(pre,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in [Path(__file__).name,f'{PREFIX}_request.json',*INPUTS]: archive.write(ROOT/p,p)
    with (ROOT/f'{PREFIX}_stdout.txt').open('wb') as out,(ROOT/f'{PREFIX}_stderr.txt').open('wb') as err:
        process=subprocess.Popen(request['command'],cwd=ROOT,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        code=process.wait()
    exitdata={'returncode':code,'natural_exit':True,'source_unchanged':sha(Path(__file__))==request['source_sha256'],'input_unchanged':request['input_sha256']=={p:sha(ROOT/p) for p in INPUTS},'preexecution_sha256':sha(pre)}
    dump(ROOT/f'{PREFIX}_exit.json',exitdata)
    executed=ROOT/f'{PREFIX}_executed.zip'
    with zipfile.ZipFile(executed,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(Path(__file__),Path(__file__).name)
        for suffix in ('request.json','stdout.txt','stderr.txt','exit.json','receipt.json'):
            p=ROOT/f'{PREFIX}_{suffix}'
            if p.exists():archive.write(p,p.name)
    dump(ROOT/f'{PREFIX}_capsule.json',{'preexecution_sha256':sha(pre),'executed_sha256':sha(executed),'exit_sha256':sha(ROOT/f'{PREFIX}_exit.json'),'receipt_sha256':sha(ROOT/f'{PREFIX}_receipt.json') if (ROOT/f'{PREFIX}_receipt.json').exists() else None,'natural_exit':True,'returncode':code})
    print(json.dumps(exitdata),flush=True)
    return code
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    PREFIX = 'heat_flux_review_particlehole_' + args.epoch + '_20261011'
    if args.control: control()
    else: raise SystemExit(execute())
