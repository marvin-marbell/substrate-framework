"""Independent combinatorial/CAR controls for the real symmetric63 scientific graph."""
from __future__ import annotations
from fractions import Fraction
import math
from typing import Any
import numpy as np


def ordered_car(mask: int, create: int, annihilate: int) -> tuple[int,int] | None:
    occupied=[i for i in range(256) if mask & (1<<i)]
    if annihilate not in occupied:return None
    parity=occupied.index(annihilate);occupied.remove(annihilate)
    if create in occupied:return None
    parity+=sum(i<create for i in occupied)
    occupied.append(create)
    return sum(1<<i for i in occupied),(-1 if parity%2 else 1)


def run(graph: Any, first: Any, car: Any, native_paths: list[Any]) -> dict[str,Any]:
    # Enumerate the labelled occupations for a small symmetric bath independently
    # of graph.moves, then prove the general63 matrix element by exact ratios.
    toy_n=5;toy_r=2
    sources=[mask for mask in range(1<<toy_n) if mask.bit_count()==toy_r]
    targets=[mask for mask in range(1<<toy_n) if mask.bit_count()==toy_r+1]
    matrix_elements=sum(1 for a in sources for b in targets if (a^b).bit_count()==1 and a&b==a)
    normalized=Fraction(matrix_elements**2,len(sources)*len(targets))
    assert normalized==(toy_n-toy_r)*(toy_r+1)
    counts=[]
    for r in (0,1,2,30,31,61,62):
        old=math.comb(63,r);new=math.comb(63,r+1)
        edges=old*(63-r)
        coefficient2=Fraction(edges**2,old*new)
        assert coefficient2==(63-r)*(r+1)
        counts.append(dict(r=r,orbit_size=old,next_orbit_size=new,exact_transition_coefficient_squared=str(coefficient2)))
    # Exercise the actual production graph transition on a real native hop.
    vertex=next(v for v in first if len(v[1])==1)
    mask,count=vertex[1][0];assert count==1
    shift=vertex[2]
    local_value=sum(a for dest,k,a in graph.local(1,graph.base) if dest==mask and k==shift)
    assert abs(local_value)>0
    seed_value=sum(a for dest,a in graph.moves(graph.seed,True) if dest==vertex)
    assert abs(seed_value/local_value-math.sqrt(63))<1e-10
    repeated=(graph.base,((mask,2),),tuple((e,2*n) for e,n in shift))
    next_value=sum(a for dest,a in graph.moves(vertex,True) if dest==repeated)
    assert abs(next_value/local_value-math.sqrt(62*2))<1e-10
    wrong_coefficient=63*local_value
    assert abs(wrong_coefficient-seed_value)>1
    # Independent wedge-order reference; covers every mode as create and destroy,
    # including occupied/occupied, empty/empty and diagonal Pauli cases.
    checked=0;negative_sign=0;blocked=0
    for mask0 in (graph.base,mask):
        for a in range(256):
            for b in (a,(a+1)%256,(37*a+19)%256):
                correct=ordered_car(mask0,a,b)
                assert car(mask0,a,b)==correct
                checked+=1
                blocked+=int(correct is None)
                negative_sign+=int(correct is not None and correct[1]<0)
    assert negative_sign>0 and blocked>0
    two=sum(len(p['edges'])==2 for p in native_paths)
    ap=sum(p['seam']<0 for p in native_paths)
    assert two>0 and ap>0
    return dict(independent_labelled_toy_normalized_coefficient_squared=str(normalized),all63_exact_multiplicity_controls=counts,actual_seed_aux_coefficient_squared=float(abs(seed_value/local_value)**2),actual_repeated_aux_coefficient_squared=float(abs(next_value/local_value)**2),wrong_linear63_coefficient_distance=float(abs(wrong_coefficient-seed_value)),ordered_wedge_CAR_checks=checked,negative_CAR_sign_witnesses=negative_sign,Pauli_blocked_witnesses=blocked,native_path_count=len(native_paths),two_link_path_count=two,AP_seam_path_count=ap)


def density_correlations(graph: Any, state: Any) -> dict[str,Any]:
    mean=np.zeros(64);second=np.zeros(64);pair=np.zeros(64)
    cache: dict[int,np.ndarray]={}
    for (_,bath,_),amplitude in state.items():
        total=np.zeros(64);squares=np.zeros(64)
        for mask,count in bath:
            if mask not in cache:
                cache[mask]=np.array([((mask>>(4*x))&15).bit_count()-2 for x in range(64)])
            deviation=cache[mask]
            total+=count*deviation;squares+=count*deviation**2
        weight=abs(amplitude)**2
        mean+=weight*total/63
        second+=weight*squares/63
        pair+=weight*(total**2-squares)/(63*62)
    covariance=pair-mean**2
    assert np.max(np.abs(covariance))>1e-12
    return dict(auxiliary_copy_count=63,every_auxiliary_has_same_marginal_by_exact_permutation_symmetry=True,onsite_density_single_copy_variance=(second-mean**2).tolist(),onsite_density_distinct_copy_connected_covariance=covariance.tolist(),max_abs_distinct_aux_density_covariance=float(np.max(np.abs(covariance))),independent_copy_prediction=0.,scope='Actual executed-state pair covariance for ANY two distinct auxiliaries, not two labelled chosen copies.')
