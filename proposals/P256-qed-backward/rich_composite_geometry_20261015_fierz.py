"""Exact same FULL electric-square law via SU3 Fierz identity, no binder.
Sum_a t^a_ij t^a_kl=(delta_il delta_jk-delta_ij delta_kl/3)/2.
All fermion quartics still act BEFORE the native projection.
"""
from collections import defaultdict
from itertools import product
from rich_composite_geometry_20261015_model import mode,label,trim
from rich_composite_geometry_20261015_runtime import coordinate_operations


def electric(primitive,packet):
    lengths=primitive.graph['lengths'];path=[sum(lengths[:v]) for v in range(4)]
    groups=list(product(range(1,4),range(2),range(2)));out=defaultdict(complex)
    outer=primitive.p['electric']/2
    # -sum_tree ell N_colour_subtree^2/6 is exact diagonal in the public counts.
    for key,value in packet.items():
        lab=primitive.label(key) if hasattr(primitive,'local') else label(key[0])
        n=[lab[4*v]+lab[4*v+2] for v in range(4)]
        diagonal=-sum(lengths[e]*sum(n[e+1:])**2 for e in range(3))/6
        out[key]+=outer*diagonal*value
    for alpha,first in enumerate(groups):
        for beta in range(alpha,len(groups)):
            second=groups[beta];weight=path[min(first[0],second[0])]*(.5 if alpha==beta else 1.)
            for a,b in product(range(3),repeat=2):
                operations=[(mode(*first,a),True),(mode(*first,b),False),
                    (mode(*second,b),True),(mode(*second,a),False)]
                for key,value in packet.items():
                    for target,coefficient in coordinate_operations(primitive,key,operations).items():
                        out[target]+=outer*weight*value*coefficient
    # 2sum_site distance(root,site) Q_site dot R; actual non-Abelian matrices.
    for site,a in product(range(1,4),range(8)):
        terms=[(mode(site,f,s,c),mode(site,f,s,d),primitive.ts[a][c,d])
            for f,s,c,d in product(range(2),range(2),range(3),range(3)) if primitive.ts[a][c,d]!=0]
        image=primitive.field(primitive.fermion(packet,terms),primitive.field_maps['R',a])
        for key,value in image.items():out[key]+=2*outer*path[site]*value
    image=primitive.field(packet,primitive.field_maps['C'])
    for key,value in image.items():out[key]+=outer*sum(lengths)*value
    return trim(out)
