"""Actual high-index neutral physical basis action, not arbitrary mode counting."""
import json
import numpy as np
from rich_review_wave35_20261016_consume import ROOT,P,POS,ARCS,B,occupations,car


def main():
    up=np.nonzero(POS.sum(1)%2==0)[0];down=np.setdiff1d(np.arange(64),up);occ=occupations(up,down)
    bits=sum(1<<(2*int(v)+s) for s in (0,1) for v in np.nonzero(occ[s])[0])
    edge=3*63;tail,head,mu=map(int,ARCS[edge]);spin=1;p=2*tail+spin;q=2*head+spin
    target,h=car(bits,p,q)
    occupied=[i for i in range(128) if (bits>>i)&1]
    # Independent ordered Fock annihilation then insertion, no fixed-width shift.
    annihilation=(-1)**occupied.index(p);occupied.remove(p)
    creation=(-1)**sum(i<q for i in occupied);independent_h=-annihilation*creation
    expected=(bits^(1<<p))|(1<<q)
    assert p==127 and q==31 and bits.bit_length()==128 and h==independent_h and h!=0 and target==expected
    badtarget,badh=car(bits&((1<<64)-1),p,q)
    assert badh==0 and target!=badtarget
    newocc=occ.copy();newocc[spin,tail]=0;newocc[spin,head]=1;E=np.zeros(192,int);E[edge]=1
    gauss=B@E+newocc.sum(0)-1
    assert np.max(abs(gauss))==0 and np.sum(newocc.sum(0)-1)==0
    result=dict(kind='ACTUAL_NEUTRAL_V64_HIGH_INDEX_INTERLEAVED_CAR_ACTION',orbital_from=p,orbital_to=q,edge=edge,spin=spin,python_integer_bits=bits.bit_length(),actual_H_matrix_element=h,independent_ordered_Fock_matrix_element=independent_h,masked_uint64_action=badh,gauss_max=int(np.max(abs(gauss))),net_charge=int(np.sum(newocc.sum(0)-1)),ordering='up0,down0,...,up63,down63; pair parity and CAR use same canonical order')
    np.savez_compressed(ROOT/(P+'highindex_raw.npz'),before_occ=occ,after_occ=newocc,E=E)
    (ROOT/(P+'highindex_findings.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
