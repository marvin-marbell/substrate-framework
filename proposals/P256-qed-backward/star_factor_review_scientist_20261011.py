"""Independent signed CAR/action/domain review. Imports frozen helpers, never mains."""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import shared_star_dynamic_scientist_20261011 as frozen

ROOT=Path(__file__).resolve().parent
PREFIX='star_factor_review_'
PIN='2f02fee85495c4ea6ad6b687add5e0ebf1a9fb975c01205bc82f3c98d3c29fa3'

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,data):
    with (ROOT/name).open('x') as out: json.dump(data,out,indent=2,default=lambda value:value.item());out.write('\n')

# Independent bit-string representation: all spin slots plus a central spectator.
SLOTS=[(0,0),(0,1),(1,0),(1,1),(1,2),(1,3),(1,4),(2,2),(2,3)]
INDEX={slot:i for i,slot in enumerate(SLOTS)}

def car_word(state,word):
    out=defaultdict(complex)
    for mask,z in state.items():
        current=mask;sign=1
        for cell,slot,p,creation in reversed(word):
            mode=4*INDEX[cell,slot]+p;occupied=(current>>mode)&1
            if occupied==creation: break
            sign*=(-1)**((current&((1<<mode)-1)).bit_count());current^=1<<mode
        else: out[current]+=sign*z
    return {m:z for m,z in out.items() if z}

def rho(mask,cell):
    # On fixed N2 slots rho is an integer; no projected onebody Hamiltonian.
    return sum(((mask>>(4*i))&3).bit_count()-((mask>>(4*i+2))&3).bit_count() for i,(c,s) in enumerate(SLOTS) if c==cell)/2

def polynomial(state,terms):
    out=defaultdict(complex)
    for z,w,mem in terms:
        weighted={m:q*np.prod([rho(m,c)**mem[c] for c in range(3)]) for m,q in state.items()}
        for m,q in car_word(weighted,w).items():out[m]+=z*q
    return {m:q for m,q in out.items() if q}

def subtract(a,b):return {m:a.get(m,0)-b.get(m,0) for m in a.keys()|b.keys() if a.get(m,0)!=b.get(m,0)}
def inner(a,b):return sum(z.conjugate()*b.get(m,0) for m,z in a.items())

def exact_controls():
    # Gaussian integer coefficients and integer electric energies: every operation
    # here is exact binary64 while magnitudes stay below 2^53 (recorded below).
    source={}
    for outera,outerc,branch in itertools.product((3,12),(3,12),range(2)):
        masks=[outera,5,3 if branch==0 else 12,5,12 if branch==0 else 3,10,3 if branch==0 else 12,outerc,10]
        mask=sum(m<<(4*i) for i,m in enumerate(masks));source[mask]=1+branch*1j
    norm=inner(source,source).real
    h=np.array([[1,2,3j,1],[2,-1,1,2j],[-3j,1,2,3],[1,-2j,3,-2]],complex)
    k=object.__new__(frozen.DynamicKernel);k.h=[h,h]
    Ma=np.array([[1,1j,2,1],[2,-1,1j,2],[1,2j,1,-1],[1j,2,-1,1]],complex)
    Mb=np.array([[2,1,1j,-1],[1j,1,2,1],[1,-1,2j,2],[2,1j,1,1]],complex)
    edges=[{'central_is_tail':True,'central_site':0,'outer_site':0,'axis':1}, {'central_is_tail':True,'central_site':0,'outer_site':0,'axis':2}]
    roots=[frozen.native.root_terms(edges[0],1,0,Ma,0),frozen.native.root_terms(edges[1],1,2,Mb,1)]
    raw=[(z*q,w+v,(0,0,0)) for z,w in roots[0] for q,v in roots[1]]
    O=frozen.collect(raw);flow=(1,1);u=(F(1,2),F(-1),F(1,2))
    A=k.action(O,flow,u);B=k.action(A,flow,u)
    assert not any(any(mem) for z,w,mem in A)
    def H(state,offset,external):
        out=defaultdict(complex)
        for mask,z in state.items():out[mask]+=z*(64*sum((rho(mask,c)+float(offset[c]))**2 for c in range(3))+external)
        for cell,slot in SLOTS:
            for p,q in itertools.product(range(4),repeat=2):
                for mask,z in car_word(state,((cell,slot,p,True),(cell,slot,q,False))).items():out[mask]+=h[p,q]*z
        return {m:z for m,z in out.items() if z}
    Os=polynomial(source,O);As=polynomial(source,A);Bs=polynomial(source,B)
    directA=subtract(H(Os,u,128),polynomial(H(source,(0,0,0),0),O))
    directB=subtract(H(As,u,128),polynomial(H(source,(0,0,0),0),A))
    assert not subtract(Os,polynomial(source,raw))
    assert not subtract(As,directA)
    assert not subtract(Bs,directB)
    without_memory=polynomial(source,[t for t in B if not any(t[2])])
    wrongrho=subtract(Bs,without_memory)
    assert inner(wrongrho,wrongrho).real>0
    # A genuine first source residual does not identify K O s with [H,O]s.
    illegal=subtract(H(Os,u,128),As)
    assert inner(illegal,illegal).real>0
    # Scalar normal-order fold and cross-cell sign are discriminated independently.
    mode=(1,0,0)
    fold=((mode+(False,)),(mode+(True,)))
    folded=frozen.collect([(1,fold,(0,0,0))])
    assert not subtract(car_word(source,fold),polynomial(source,folded))
    assert any(not w for z,w,m in folded)
    maximum=max(abs(v.real)+abs(v.imag) for state in (source,Os,As,Bs,directA,directB) for v in state.values())
    assert maximum<2**53
    return {'method':'Independent global 36-mode signed bit CAR, full4x4 spin roots, three cells, central two-copy entanglement plus spectator; exact Gaussian integer arithmetic','source_norm_square':norm,'term_counts':[len(O),len(A),len(B)],'state_support_counts':[len(source),len(Os),len(As),len(Bs)],'maximum_integer_component_L1':maximum,'O_normal_order_exact':True,'A_full_fock_intertwining_exact':True,'B_full_fock_intertwining_exact':True,'first_electric_cancellation_exact':True,'second_memory_term_count':sum(bool(any(m)) for z,w,m in B),'wrong_delete_totalrho_squared_distance':inner(wrongrho,wrongrho).real/norm,'illegal_identify_KOs_with_commutator_squared_distance':inner(illegal,illegal).real/norm,'scalar_fold_retained_exact':True,'toy_not_native_ground':True}

def native_controls():
    receipt=json.loads((ROOT/'shared_star_dynamic_actual2_20261011_receipt.json').read_bytes())
    boundary=json.loads((ROOT/'charged_first_shell_actual04_20261011_receipt.json').read_bytes())['all64_boundary']
    _,_,L,exact=frozen.formation.derive_physical(boundary)
    matrices={(axis,s,t):(-1)**(s+t)*np.array([[complex(frozen.formation.sp.sympify(z)) for z in row] for row in exact[axis]]) for axis,s,t in itertools.product(range(3),range(2),range(2))}
    k=frozen.DynamicKernel();star=frozen.Star(k,receipt['edges'],matrices)
    component=(1.,0,1,1,-1)
    O,A,B=star.vectors(component,(0,1))
    controls=[]
    for j,terms in enumerate((O,A,B)):
        # Entire O, stratified A/B incl all available cells' rho-memory branches.
        selected=terms if j==0 else terms[::max(1,len(terms)//24)]+[t for t in terms if any(t[2])][::max(1,sum(any(t[2]) for t in terms)//24)]
        direct=0j
        for z,w,lm in selected:
            dagger=tuple((c,s,p,not cr) for c,s,p,cr in reversed(w))
            for q,v,rm in selected:direct+=z.conjugate()*q*k.moment_memory(dagger+v,lm,rm)
        fast=k.gram_terms(selected,selected)
        assert abs(direct-fast)<=1e-10*max(1,abs(direct))
        controls.append({'order':j,'selected_terms':len(selected),'direct_global_signed_real':direct.real,'vectorized_cell_real':fast.real,'difference':abs(direct-fast)})
    base=k.gram_terms(O,O).real
    k.independent=True;k.local_memory.cache_clear()
    reset=k.gram_terms(O,O).real
    assert abs(reset-base)>1e-8
    k.independent=False;k.local_memory.cache_clear()
    bbase=k.gram_terms(B,B).real;k.delete_memory=True
    bdrop=k.gram_terms(B,B).real;k.delete_memory=False
    assert abs(bdrop-bbase)>1e-8
    unsupported=False
    try:k.local_memory(((0,0,True),(0,0,False),(1,0,True),(1,0,False),(2,0,True),(2,0,False)),0,0)
    except ArithmeticError:unsupported=True
    assert unsupported
    rejected=False
    try:k.action(B,(1,-1),(F(1,2),F(0),F(-1,2)))
    except ArithmeticError:rejected=True
    assert rejected
    alpha=[float(F(r['alpha_exact'])) for r in receipt['single_roots']]
    aa=[(1/alpha[0],0,o,0,-o) for o in (-1,1)]
    bb=[(1/alpha[1],1,o,1,-o) for o in (-1,1)]
    mixed=[]
    for oa,ob in itertools.product((-1,1),repeat=2):mixed +=[(1/alpha[1],0,oa,1,ob),(1/alpha[0],1,ob,0,oa)]
    raw_reviews={};maxpairs=0;maxslots=0;maxmemory=0
    for name,comps in [('neutral_plus',aa+bb),('neutral_minus',aa+[(-w,a,o,b,p) for w,a,o,b,p in bb]),('mixed',mixed)]:
        rawbudget=collapsedbudget=0.
        for left,right in itertools.product(comps,repeat=2):
            if star.flow_u(left)[0]!=star.flow_u(right)[0]:continue
            for copies,mult,_ in frozen.assignments():
                vec=[];rawnorm=[]
                for comp,labels in [(left,copies[:2]),(right,copies[2:])]:
                    w,e,o,f,p=comp
                    rawterms=[(z*q,x+y) for z,x in star.root(e,o,labels[0]) for q,y in star.root(f,p,labels[1])]
                    rawnorm.append(sum(abs(z) for z,w in rawterms));vec.append(star.vectors(comp,labels))
                    for terms in vec[-1]:
                        for z,word,mem in terms:
                            maxmemory=max(maxmemory,*mem)
                            for cell in range(3):maxslots=max(maxslots,len({s for c,s,p,cr in word if c==cell}))
                weight=abs(left[0]*right[0])*mult
                rawbudget+=weight*np.prod(rawnorm)
                collapsedbudget+=weight*np.prod([sum(abs(z) for z,w,m in v[0]) for v in vec])
                maxpairs=max(maxpairs,max(len(a)*len(b) for a in vec[0] for b in vec[1]))
        raw_reviews[name]={'raw_product_L1_gram_budget':rawbudget,'collapsed_O_L1_gram_budget':collapsedbudget,'sixteen_collapsed_dominates_raw':16*collapsedbudget>=rawbudget,'ratio':rawbudget/collapsedbudget}
    assert maxmemory==1 and maxslots<=2
    return {'global_vs_vectorized':controls,'native_central_independent_reset':{'actual':base,'wrong':reset,'difference':reset-base},'native_B_totalrho_delete':{'actual':bbase,'wrong':bdrop,'difference':bdrop-bbase},'third_slot_rejected':unsupported,'third_action_with_rho_rejected':rejected,'max_source_memory_power_per_cell_per_operand':maxmemory,'max_touched_slots_per_cell_per_operand':maxslots,'largest_vectorized_pair_reduction':maxpairs,'bound_2pow28':2**28,'raw_uncollapsed_reviews':raw_reviews,'physical_copy_partition_count':len(frozen.assignments()),'four_label_count':sum(m for c,m,n in frozen.assignments()),'rho_cache_shapes':{str(key):list(value.shape) for key,value in k.fields.items()}}

def run(epoch):
    assert digest(ROOT/'shared_star_dynamic_scientist_20261011.py')==PIN
    expected=json.loads((ROOT/f'{PREFIX}{epoch}_request.json').read_bytes())['input_and_source_sha256']
    for name,pin in expected.items():assert digest(ROOT/name)==pin,(name,'changed')
    result={'kind':'INDEPENDENT_GLOBAL_CAR_THREECELL_FULLRHO_DOMAIN_ADVERSE_CONSUMER','frozen_shared_star_sha256':PIN,'exact_toy':exact_controls(),'native':native_controls(),'input_and_source_sha256':expected,'natural_main_finished':True,'scope':'Algebra and formation review only; no rerun of historical target main, no physical1/sign promotion'}
    for name,pin in expected.items():assert digest(ROOT/name)==pin
    save(PREFIX+epoch+'_20261011_receipt.json',result)
    print(json.dumps({'natural_main_finished':True,'receipt':PREFIX+epoch+'_20261011_receipt.json','exact_toy':result['exact_toy'],'native_adverse_differences':[result['native']['native_central_independent_reset']['difference'],result['native']['native_B_totalrho_delete']['difference']]},default=lambda value:value.item()),flush=True)

if __name__=='__main__':run(__import__('sys').argv[1])
