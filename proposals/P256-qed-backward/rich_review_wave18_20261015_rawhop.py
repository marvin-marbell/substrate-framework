"""NEW actual raw hopping/EM-charge/colour ownership BEFORE native h selection."""
from itertools import product
from pathlib import Path
from typing import Any
import json,resource,sys,time
import numpy as np
from scipy import sparse
from rich_spatial_colour_charge_20261015_model import forward,loop_field,generators,frame,CHARGES3
ROOT=Path(__file__).resolve().parent


def norm(a: Any)->float:return float(np.linalg.norm(a.data))


def main()->None:
    start=time.monotonic();request=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);loop=loop_field();sigma=frame()['sigma'];q=sparse.lil_matrix((32,32),dtype=complex)
    for f,s,c in product(range(2),range(2),range(4)):
        charge=float(CHARGES3[(f*2+s)*4+c]/3)
        for m,n in product(range(2),repeat=2):q[16*f+8*m+4*s+c,16*f+8*n+4*s+c]=charge/2
    q=q.tocsr();Q=sparse.kron(q,sparse.eye(19),format='csr');bad=sparse.kron(sparse.diags(q.diagonal()),sparse.eye(19),format='csr');gauss=[]
    for a,t in enumerate(generators()):
        matter=sparse.lil_matrix((32,32),dtype=complex)
        for f,m,s,c,d in product(range(2),range(2),range(2),range(3),range(3)):matter[16*f+8*m+4*s+c,16*f+8*m+4*s+d]=t[c,d]
        gauss.append(sparse.kron(matter.tocsr(),sparse.eye(19),format='csr')+sparse.kron(sparse.eye(32),loop['L'][a]+loop['R'][a],format='csr'))
    cases=[]
    for edge,f,white,spin_index in product(range(2),range(2),range(2),range(4)):
        spin=[np.eye(2)]+sigma;rows=[];cols=[];values=[]
        for mode in range(32):
            for field in range(19):
                vector=np.zeros(19,complex);vector[field]=1;acted=forward({1<<mode:vector},loop,edge,f,bool(white),spin[spin_index])
                for mask,value in acted.items():
                    if mask.bit_count()!=1:raise ValueError('Raw bilinear left N1 sector')
                    target=mask.bit_length()-1
                    for r in np.flatnonzero(value!=0):rows.append(19*target+int(r));cols.append(19*mode+field);values.append(value[r])
        F=sparse.csr_matrix((values,(rows,cols)),shape=(608,608));charge=([2/3,0.] if f==0 else [-1/3,-1.])[white]
        cases.append(dict(edge=edge,flavour=f,colourless=bool(white),spin_component=spin_index,physical_charge=charge,raw_forward_norm=norm(F),original_raw_charge_eigenoperator_error=norm(Q@F-F@Q+charge*F),wrong_band_diagonal_charge_error=norm(bad@F-F@bad+charge*F),global_colour_commutator_max=max(norm(g@F-F@g) for g in gauss)))
    result=dict(kind='NEW_INDEPENDENT_UNPROJECTED_ORIGINAL_FORWARD_CHARGE_AND_GAUSS_OPERATOR_CONSUMER',dimension=608,cases=cases,maximum_original_raw_charge_error=max(x['original_raw_charge_eigenoperator_error'] for x in cases),maximum_raw_colour_error=max(x['global_colour_commutator_max'] for x in cases),meaningful_wrongcase_max=max(x['wrong_band_diagonal_charge_error'] for x in cases),meaning='All32 ORIGINAL model.forward operators measured in complete N1 CAR32xPW19, BEFORE band/native h-block selection. Physical Qsite contains offdiagonal band terms and fractional q. CAR bilinear algebra lifts the exact charge relation to all N; this is not current/H-range closure, higher PW or a full-law no-go.',elapsed_seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['maximum_original_raw_charge_error','maximum_raw_colour_error','meaningful_wrongcase_max','elapsed_seconds','peak_RSS_KiB']}),flush=True)


if __name__=='__main__':main()
