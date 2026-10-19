"""Native heat-domain/complex-polarization consumer, not phase acceptance."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import mpmath as mp
import numpy as np
import charged_cell_heat_helper_20261011 as heat

ROOT=Path(__file__).resolve().parent

def exact_box(v:complex)->dict[str,list[str]]:
    return {k:[str(F(float(x)))]*2 for k,x in (('re',v.real),('im',v.imag))}

def contains(pair:list[str],value:Any)->bool:
    def real(x:str)->Any:
        a=F(x)
        return mp.mpf(a.numerator)/a.denominator
    return real(pair[0])-mp.mpf('1e-70')<=value<=real(pair[1])+mp.mpf('1e-70')

def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--interface',required=True)
    args=parser.parse_args();start=time.monotonic()
    names=[args.interface,'charged_cell_heat_helper_20261011.py','flux_laplace_actual1_20261011_receipt.json']
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    interface=json.loads((ROOT/names[0]).read_bytes())
    assert interface['helper_sha256']==hashes[names[1]]
    assert F(interface['rigorous_residual_exact'])==F(1,65536)
    assert F(interface['formation_included_in_residual_exact'])==F(1,262144)
    mp.mp.dps=90
    ABI=[]
    for role in ('active','aux'):
        plus=heat.evaluate(interface,role,'creation',F(0))
        minus=heat.evaluate(interface,role,'annihilation',F(0))
        for i,j in ((0,0),(0,1),(1,0),(1,1)):
            for key in ('re','im'):
                lo=F(plus[i][j][key][0])+F(minus[i][j][key][0])
                hi=F(plus[i][j][key][1])+F(minus[i][j][key][1])
                wanted=int(i==j and key=='re')
                assert lo<=wanted<=hi
        ABI.append({'role':role,'Gplus0_plus_Gminus0_encloses_identity':True})
    # Exact dyadic full-CAR orientation witness on canonical number2,
    # (mask5+i mask6)/sqrt2; use unnormalized numerator and divide by2.
    ann=np.zeros((4,16,16),dtype=complex)
    for p in range(4):
        for mask in range(16):
            if mask&(1<<p):ann[p,mask^(1<<p),mask]=(-1)**((mask&((1<<p)-1)).bit_count())
    v=np.zeros(16,dtype=complex);v[5]=1;v[6]=1j
    creation=np.array([[np.vdot(v,ann[q]@ann[p].conj().T@v)/2 for p in range(2)] for q in range(2)])
    annihilator_gram=np.array([[np.vdot(v,ann[p].conj().T@ann[q]@v)/2 for q in range(2)] for p in range(2)])
    assert np.array_equal(creation+annihilator_gram.T,np.eye(2))
    wrong_transpose=float(np.linalg.norm(creation+annihilator_gram-np.eye(2)))
    assert wrong_transpose>1
    # Independent positive atomic matrix measure, not a native ground model.
    # Real moment matrices do not commute. Complex offdiagonal remainder
    # has no entrywise-positive meaning despite its positive Loewner order.
    energies=[4,17,51]
    Y=np.array([[.5,.25j],[.25j,.5],[.125,-.125]],dtype=complex)
    moments=[Y.conj().T@np.diag(np.array(energies)**k)@Y for k in range(3)]
    assert np.linalg.norm(moments[0]@moments[1]-moments[1]@moments[0])>.01
    toy={'charged_floor_exact':'4','ground_distance_exact':['0','0'],'roles':[{'role':'toy','orientation':'creation','moments':{name:[[exact_box(complex(v)) for v in row] for row in matrix] for name,matrix in zip(('M0','M1','M2'),moments)},'ground_second_moment_trace_upper_exact':str(F(float(np.trace(moments[2]).real)))}]}
    controls=[]
    for t in (F(1,512),F(1,32),F(1,4),F(1)):
        result=heat.evaluate(toy,'toy','creation',t)
        tt=mp.mpf(t.numerator)/t.denominator
        actual=[[sum(mp.conj(mp.mpc(complex(Y[k,i])))*mp.exp(-tt*energies[k])*mp.mpc(complex(Y[k,j])) for k in range(3)) for j in range(2)] for i in range(2)]
        for i in range(2):
            for j in range(2):
                assert contains(result[i][j]['re'],actual[i][j].real)
                assert contains(result[i][j]['im'],actual[i][j].imag)
        controls.append({'t_exact':str(t),'independent_positive_matrix_measure_enclosed':True})
    t=mp.mpf(1)/32;a=20
    remainder_real=-mp.mpf(1)/64*(mp.exp(-51*t)-mp.exp(-a*t)*(1+t*(a-51)))
    assert remainder_real<mp.mpf('-.001')
    assert hashes=={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    output={'kind':'NATIVE_HEAT_ABI_AND_INDEPENDENT_COMPLEX_LOEWNER_PARENT_CONSUMER','own_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':hashes,'native_ABI':ABI,'complex_canonical2_CAR_wrong_transpose_norm':wrong_transpose,'independent_atomic_matrix_controls':controls,'wrong_entrywise_Loewner_positivity_rejected':{'offdiagonal_real_remainder':str(remainder_real),'meaning':'Positive Loewner remainder has negative real offdiagonal entry; source matrix enclosure must use polarization, not scalar entrywise order.'},'scope':'Native identity tested with priced g intervals; atomic90digit control is an independent finite method observation, not native spectrum/interval ground/phase.','historical_producer_or_solver_rerun':False,'phase_or_pole_admission':False,'elapsed_seconds':time.monotonic()-start}
    Path(args.output).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'native_ABI':ABI,'wrong_transpose_norm':wrong_transpose,'wrong_Loewner_entry_real':str(remainder_real),'elapsed_seconds':output['elapsed_seconds']}))

if __name__=='__main__':main()
