#!/usr/bin/env python3
"""Complete neutral cut2 factor reduced by commuting copy and charge involutions."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, block_diag, eye
from threadpoolctl import threadpool_limits
import covariant_native_collective_primitive_electric as base
from covariant_native_paired_charge_conjugation import conjugate_occupation
ROOT = Path(__file__).resolve().parent
SOURCE_AT_IMPORT: dict[str, bytes] = {}
for module in tuple(sys.modules.values()):
    filename=getattr(module,'__file__',None)
    if isinstance(filename,str):
        source=Path(filename).resolve()
        if source.parent==ROOT and source.suffix=='.py':
            SOURCE_AT_IMPORT[source.name]=source.read_bytes()
SOURCE_AT_IMPORT[Path(__file__).name] = Path(__file__).read_bytes()
INPUT_AT_IMPORT = {'plan': base.PLAN_BYTES_AT_IMPORT}
for module in tuple(sys.modules.values()):
    filename=getattr(module,'__file__',None)
    if isinstance(filename,str) and Path(filename).resolve().parent==ROOT:
        for key,value in vars(module).copy().items():
            if key.endswith('_BYTES_AT_IMPORT') and isinstance(value,bytes):
                INPUT_AT_IMPORT[module.__name__+':'+key]=value
if hashlib.sha256(SOURCE_AT_IMPORT['covariant_native_collective_primitive_electric.py']).hexdigest() != '085c316c7ae4cf64d4cb9e09ee8429294db191602f26addacaabd0626c5681ea':
    raise RuntimeError('Frozen supplier changed')
def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8388608), b''):
            h.update(chunk)
    return h.hexdigest()
def charge_basis(physical: dict[str, Any], sector: dict[str, Any], chi: int):
    reps = sector['representative_rows']; lookup = {int(r): i for i, r in enumerate(reps)}
    aux = {a: i for i, a in enumerate(physical['auxiliary'])}
    permutation = []; signs = []
    for r in reps:
        state, flux, electric = physical['physical'][int(r)]
        l, a, b = physical['states'][state]
        lc, ls = conjugate_occupation(l, 8); ac, sa = conjugate_occupation(a, 4); bc, sb = conjugate_occupation(b, 4)
        s = ls * sa * sb
        if aux[ac] > aux[bc]:
            ac, bc = bc, ac; s *= sector['sign']
        target = physical['index'][(physical['state_index'][(lc, ac, bc)], -flux)]
        if physical['physical'][target][2] != tuple(-x for x in electric):
            raise ArithmeticError('C electric reversal')
        permutation.append(lookup[target]); signs.append(s)
    perm = np.array(permutation); signs = np.array(signs)
    if not np.array_equal(perm[perm], np.arange(len(perm))) or not np.all(signs * signs[perm] == 1):
        raise ArithmeticError('Integer C involution')
    rr=[]; cc=[]; vv=[]; contributors=[]
    for i, j in enumerate(perm):
        if j < i or (j == i and signs[i] != chi): continue
        k=len(contributors)
        terms=[(i, 1.)] if j == i else [(i, 1/math.sqrt(2)), (int(j), chi*int(signs[i])/math.sqrt(2))]
        contributors.append(terms)
        for r,v in terms: rr.append(r); cc.append(k); vv.append(v)
    P=coo_matrix((vv,(rr,cc)),shape=(len(perm),len(contributors))).tocsr()
    return P, contributors, perm, signs
def physical_intertwining(physical, sector, perm, signs):
    rr=[]; vv=[]
    for state, flux, electric in physical['physical']:
        l,a,b=physical['states'][state]
        lc,ls=conjugate_occupation(l,8);ac,sa=conjugate_occupation(a,4);bc,sb=conjugate_occupation(b,4)
        rr.append(physical['index'][(physical['state_index'][(lc,ac,bc)],-flux)]);vv.append(ls*sa*sb)
    n=len(rr)
    C=coo_matrix((vv,(rr,np.arange(n))),shape=(n,n)).tocsr()
    Cs=coo_matrix((signs,(perm,np.arange(len(perm)))),shape=(len(perm),len(perm))).tocsr()
    copy=sector['embedding'][:n,:len(perm)]
    defect=float(abs(C@copy-copy@Cs).max())
    electric_defect=int(np.max(abs(sector['electric'][perm]+sector['electric'])))
    if defect>1e-14 or electric_defect: raise ArithmeticError('Physical C intertwining failed')
    return dict(Cphysical_Pcopy_equals_Pcopy_Ccopy_defect=defect,ALL4edge_C_reversal_defect=electric_defect,Phi_heat_commutes='C acts identically on each of three Phi nodes',integer_involution_columns=len(perm))


def stream(gamma, nodes, heat, physical, sector, contributors, path):
    m=len(contributors); F=np.memmap(path, mode='w+',dtype=np.complex128,shape=(3*m,3*m),order='F')
    ids=np.zeros((m,2),dtype=int); weights=np.zeros((m,2))
    for i,terms in enumerate(contributors):
        for j,(r,w) in enumerate(terms): ids[i,j]=r;weights[i,j]=w
    lr,ar,n3,flux=(sector[k] for k in ('light_ids','pair_ids','light_n3','flux'))
    E=np.exp(-np.sum(sector['electric']**2,axis=1)/2); lower=[]
    for p,normal in enumerate(nodes):
        actual=gamma.copy(); actual[1]*=1+float(normal)
        half,pref=base.collective.matched_tree_half(actual,0.,0.)
        mean=base.canonical(half,pref,np.array(physical['light']))
        samples=[]; minimum=math.inf
        for theta in 2*math.pi*np.arange(32)/32:
            half,pref=base.collective.auxiliary_path_half(actual,.04,complex((1+np.exp(1j*theta))/2))
            A=base.canonical(half,pref,np.array(physical['auxiliary']))
            minimum=min(minimum,float(eigh(A,eigvals_only=True)[0]))
            samples.append(sector['auxiliary_map'].T@np.kron(A,A)@sector['auxiliary_map'])
        lower.append(float(eigh(mean,eigvals_only=True)[0])*minimum**2)
        coeff=np.fft.fft(np.array(samples),axis=0)/32
        for start in range(0,m,64):
            stop=min(m,start+64); block=np.zeros((m,stop-start),complex)
            for a in range(2):
                r=ids[:,a]
                for b in range(2):
                    c=ids[start:stop,b]
                    delta=flux[r,None]-flux[c][None,:]-n3[r,None]+n3[c][None,:]
                    block+=weights[:,a,None]*weights[start:stop,b][None,:]*mean[lr[r,None],lr[c][None,:]]*coeff[delta%32,ar[r,None],ar[c][None,:]]*E[c][None,:]
            for out in range(3): F[out*m:(out+1)*m,p*m+start:p*m+stop]=heat[out,p]*block
        F.flush()
    if min(lower)<=0: raise ArithmeticError('Nonpositive canonical bound')
    return F,lower

def controls(gamma,nodes,heat,scratch):
    physical=base.space(0,0); rows=[]
    for sign in (1,-1):
        sector=base.copy_sector(physical,sign)
        direct,_,_=base.streamed_factor(gamma,nodes,heat,physical,sector,32,1.,scratch/f'control_S{sign}.bin')
        scale=float(np.max(abs(direct)))
        Ps=[]; factors=[]
        for chi in (1,-1):
            P,terms,perm,signs=charge_basis(physical,sector,chi)
            total=block_diag((P,P,P),format='csr'); Ps.append(total)
            F,_=stream(gamma,nodes,heat,physical,sector,terms,scratch/f'control_S{sign}_C{chi}.bin'); factors.append(F)
            expected=total.T@(direct@total.toarray())
            defect=float(np.max(np.abs(F-expected)))/scale
            C=coo_matrix((signs,(perm,np.arange(len(perm)))),shape=(len(perm),len(perm))).tocsr()
            iso=P.T@P-eye(len(terms),format='csr'); eig=C@P-chi*P
            rows.append(dict(sign=sign,charge_C_sign=chi,columns=F.shape[1],ALL_column_factor_defect=defect,isometry_defect=float(abs(iso).max()),C_eigen_defect=float(abs(eig).max())))
        cross=float(np.max(abs(Ps[0].T@(direct@Ps[1].toarray()))))/scale
        resolution=Ps[0]@Ps[0].T+Ps[1]@Ps[1].T-eye(direct.shape[0],format='csr')
        if cross>1e-11 or abs(resolution).max()>1e-11 or max(r.get('ALL_column_factor_defect',0.) for r in rows)>1e-11: raise ArithmeticError('Complete cut0 control failed')
        rows.append(dict(sign=sign,cross_C_ALLcolumns_relative=cross,actual_factor_max_scale=scale,completeness_defect=float(abs(resolution).max())))
    return rows

def load_spectrum(capsule_path: Path):
    cap=json.loads(capsule_path.read_bytes())
    for name,sha in cap['source_hashes_at_import'].items():
        if digest(ROOT/name)!=sha: raise RuntimeError('Source changed: '+name)
    for name,sha in cap['input_hashes_at_import'].items():
        if name not in INPUT_AT_IMPORT or hashlib.sha256(INPUT_AT_IMPORT[name]).hexdigest()!=sha:
            raise RuntimeError('Imported input changed: '+name)
    physical=base.space(0,2); blocks=[]
    for row in cap['blocks']:
        sector=base.copy_sector(physical,row['sign']); P,_,_,_=charge_basis(physical,sector,row['charge_C_sign'])
        total=sector['embedding']@block_diag((P,P,P),format='csr')
        path=Path(row['U_path'])
        if digest(path)!=row['generated_U_sha256']: raise RuntimeError('Generated U changed')
        U=np.memmap(path,mode='r',dtype=np.complex128,shape=tuple(row['shape']),order='F')
        blocks.append(dict(V=U,q=np.array(row['original_sigma_per_column']),P=total,sign=row['sign'],charge_C_sign=row['charge_C_sign']))
    return base.Spectrum(blocks,3*len(physical['physical'])),cap['controls']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scratch',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.scratch.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    with threadpool_limits(limits=4):
        gamma,embedding,nodes,heat,Hphi,metric=base.left.frame()
        proof=controls(gamma,nodes,heat,args.scratch)
        phase=base.validate_copy_identity(gamma,nodes)
        physical=base.space(0,2);blocks=[]
        for sign in (1,-1):
            sector=base.copy_sector(physical,sign)
            for chi in (1,-1):
                P,terms,perm,signs=charge_basis(physical,sector,chi)
                physical_proof=physical_intertwining(physical,sector,perm,signs)
                path=(args.scratch/f'neutral_S{sign}_C{chi}_leftU.complex128').resolve()
                F,lower=stream(gamma,nodes,heat,physical,sector,terms,path)
                q,abi=base.left.lapack_left(F);F.flush()
                if np.any(q<=0): raise ArithmeticError('Nonpositive actual singular values')
                blocks.append(dict(sign=sign,charge_C_sign=chi,U_path=str(path),shape=list(F.shape),generated_U_sha256=digest(path),original_sigma_per_column=q.tolist(),pointcanonical_lower_bounds=lower,ABI=abi,physical_proof=physical_proof,basis_description={'copy_representative_rows':sector['representative_rows'].tolist(),'C_permutation':perm.tolist(),'C_integer_signs':signs.tolist(),'C_contributors':terms}))
                print(json.dumps(dict(sign=sign,charge_C_sign=chi,rank=len(q),sigma_min=float(q[-1]))),flush=True)
                del F
        ranks=sum(len(b['original_sigma_per_column']) for b in blocks)
        if ranks!=29124: raise ArithmeticError('Incomplete neutral rank')
        cap=dict(schema='neutral-cut2-copy-C-v1',literal_inputs={'charge':0,'Nlight':4,'UE':1,'ALL4edge_cut':2,'grid':32,'auxiliary_copies':2,'auxiliary_occupations_each':2,'Phi_nodes':nodes.tolist(),'law':'old conditional kappa400 Gamma/Phi heat, not selfconsistent'},source_hashes_at_import={k:hashlib.sha256(v).hexdigest() for k,v in SOURCE_AT_IMPORT.items()},input_hashes_at_import={k:hashlib.sha256(v).hexdigest() for k,v in INPUT_AT_IMPORT.items()},blocks=blocks,total_ranks=ranks,controls={'cut0_complete_blocks':proof,'ALLcolumn_nonzero_phase_identity':phase},runtime_seconds=time.perf_counter()-start,versions={'numpy':np.__version__,'scipy':scipy.__version__},descriptor={'gamma':gamma.tolist(),'heat':heat.tolist(),'metric':metric})
        cap['input_bytes_at_import_utf8']={k:v.decode('utf-8') for k,v in INPUT_AT_IMPORT.items()}
        cap['descriptor_sha256']=hashlib.sha256(json.dumps(cap['descriptor'],sort_keys=True,default=str).encode()).hexdigest()
        args.output.write_text(json.dumps(cap,indent=2,default=lambda x: x.tolist() if isinstance(x,np.ndarray) else str(x)))
        spectrum,_=load_spectrum(args.output)
        print(json.dumps(dict(actual_loader_ranks=len(spectrum.q),natural_completion=True,runtime_seconds=cap['runtime_seconds'])),flush=True)
if __name__=='__main__':main()
