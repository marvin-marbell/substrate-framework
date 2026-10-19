"""NEW four-cell full-CAR method control, NOT the 9.6M native ground.
Compare complete sector resolvents with charged-cell spectral Laplace factors.
Every local number2 exterior is six-dimensional; no five-slot projection.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
INPUT='local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    started=time.monotonic()
    profile=json.loads((ROOT/INPUT).read_text())['profiles'][0]
    from fractions import Fraction
    m,c,d=(float(Fraction(profile[k])) for k in ('m_exact','c0_exact','d_exact'))
    h=m*np.diag([-1,1,-1,1])+c*np.array([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]])+d*np.array([[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]])
    ann=np.zeros((4,16,16),dtype=complex)
    for p,mask in itertools.product(range(4),range(16)):
        if mask&(1<<p):ann[p,mask^(1<<p),mask]=(-1)**((mask&((1<<p)-1)).bit_count())
    assert max(np.linalg.norm(ann[p]@ann[q].conj().T+ann[q].conj().T@ann[p]-np.eye(16)*(p==q)) for p,q in itertools.product(range(4),repeat=2))<1e-14
    D=sum(h[p,q]*ann[p].conj().T@ann[q] for p,q in itertools.product(range(4),repeat=2))
    R=sum((.5 if p<2 else -.5)*ann[p].conj().T@ann[p] for p in range(4))
    masks={n:np.array([x for x in range(16) if x.bit_count()==n]) for n in (1,2,3)}
    spectral={}
    for n,u in ((1,.5),(2,0.),(3,-.5)):
        H=D+64*(R+u*np.eye(16))@(R+u*np.eye(16))
        values,vectors=np.linalg.eigh(H[np.ix_(masks[n],masks[n])])
        spectral[n]=(values,vectors)
    Eg=float(spectral[2][0][0])
    assert spectral[2][0][1]-Eg>1e-6
    operators={}
    for n,creation in ((1,True),(2,False),(2,True),(3,False)):
        after=n+(1 if creation else -1)
        for p in (0,1):
            full=ann[p].conj().T if creation else ann[p]
            operators[n,creation,p]=spectral[after][1].conj().T@full[np.ix_(masks[after],masks[n])]@spectral[n][1]
    # Independent rational complex cell-to-cell coefficients: method inputs,
    # not a replacement for the campaign's native endpoint matrices.
    M0=np.array([[1,.3j],[-.2,.7]],dtype=complex)
    M1=np.array([[.8,-.4],[.2j,1.1]],dtype=complex)
    matrices=(M0,M1,M0.conj().T,M1.conj().T)
    def apply_axis(v,A,axis):
        return np.moveaxis(np.tensordot(A,v,axes=(1,axis)),0,axis)
    def apply_edge(v,n,e):
        row,col=e,(e+1)%4
        sign=(-1)**(sum(n[:col])+sum(n[:row])-(1 if col<row else 0))
        out=None
        for p,q in itertools.product(range(2),repeat=2):
            w=apply_axis(v,operators[n[col],False,q],col)
            w=apply_axis(w,operators[n[row],True,p],row)
            term=sign*matrices[e][p,q]*w
            out=term if out is None else out+term
        assert out is not None
        new=n.copy();new[row]+=1;new[col]-=1
        return out,new,sign
    def diagonal(n,k,z):
        shape=tuple(len(spectral[x][0]) for x in n)
        out=np.full(shape,64*k+z-4*Eg)
        for axis,x in enumerate(n):
            view=[1]*4;view[axis]=shape[axis]
            out+=spectral[x][0].reshape(view)
        assert np.min(out)>0
        return out
    rows=[]
    for order in itertools.permutations(range(4)):
        v=np.zeros((6,)*4,dtype=complex);v[(0,)*4]=1
        n=[2]*4;graded=1
        for stage,e in enumerate(order):
            v,n,sign=apply_edge(v,n,e);graded*=sign
            if stage<3:v/=diagonal(n,stage+1,0)
        direct=complex(v[(0,)*4])
        weights=[]
        creation_first=[]
        for vertex in range(4):
            first=order.index(vertex)<order.index((vertex-1)%4)
            creation_first.append(first)
            W=np.empty((4,2,2),dtype=complex)
            for a,p,q in itertools.product(range(4),range(2),range(2)):
                if first:
                    W[a,q,p]=operators[3,False,q][0,a]*operators[2,True,p][a,0]
                else:
                    W[a,q,p]=operators[2,False,q][a,0]*operators[1,True,p][0,a]
            weights.append(W)
        K=[weights[v]@matrices[v] for v in range(4)]
        coefficients=graded*np.einsum('aij,bjk,ckl,dli->abcd',*K)
        denom=np.ones((4,)*4)
        changed=[0]*4
        for stage,e in enumerate(order[:3]):
            changed[e]+=1;changed[(e+1)%4]-=1
            energy=np.full((4,)*4,64*(stage+1),dtype=float)
            for vertex,charge in enumerate(changed):
                if charge:
                    view=[1]*4;view[vertex]=4
                    energy+=(spectral[2+charge][0]-Eg).reshape(view)
            assert np.min(energy)>0
            denom*=energy
        factor=complex(np.sum(coefficients/denom))
        wrong_equal_time=complex(np.sum(coefficients)/(64*128*192))
        rows.append({'order':order,'complete_sector_resolvent':{'re':direct.real,'im':direct.imag},'charged_cell_Laplace_spectral_factor':{'re':factor.real,'im':factor.imag},'absolute_defect':abs(direct-factor),'wrong_equal_time_absolute_defect':abs(direct-wrong_equal_time),'graded_closed_sign':graded,'creation_first_at_vertices':creation_first})
    scale=max(abs(complex(r['complete_sector_resolvent']['re'],r['complete_sector_resolvent']['im'])) for r in rows)
    maxdef=max(r['absolute_defect'] for r in rows)
    wrong=max(r['wrong_equal_time_absolute_defect'] for r in rows)
    assert scale>1e-10 and maxdef<1e-11*scale
    assert wrong>.01*scale
    out={'kind':'FULL_CAR_FOUR_CELL_LAPLACE_FACTORIZATION_METHOD_CONTROL_NOT_NATIVE_GROUND','input_profile_sha256':hashlib.sha256((ROOT/INPUT).read_bytes()).hexdigest(),'own_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'method':'Complete three-resolvent sector calculation versus exact finite spectral Laplace factorization; the latter keeps entire charged number1/3 sectors. Local neutral excited components cancel only at terminal ground bra, not by projecting the intermediate operator.','local_neutral_dimension':6,'charged_dimensions':[4,4],'orders':rows,'maximum_relative_defect':maxdef/scale,'wrong_equal_time_relative_defect':wrong/scale,'ground_energy':Eg,'all24_orders_exercised':True,'no_native_ground_solver_or_old_producer_run':True,'native_9_6M_coefficient_certified':False,'phase_or_pole_claim':False,'elapsed_seconds':time.monotonic()-started}
    Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('kind','maximum_relative_defect','wrong_equal_time_relative_defect','elapsed_seconds')},indent=2))

if __name__=='__main__':main()
