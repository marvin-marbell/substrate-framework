"""Explicit source operators on ARBITRARY states, not a state-only extension.
B, (2J.Rd)B, TrU B and D=[H0,B] quark-replacement/sea clouds.
Phi and Pphi source states lie in the FULL independentPhi7 matrix-unit release.
This module applies sources; it does NOT claim a generic dressed-source EOM.
"""
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_cloud_probe import source,add
from rich_composite_geometry_20261015_source_eom import source_operator,commutator_terms,apply_terms


def field_action(state,pw=None,rotation=None,phi=None):
    result={}
    for key,field in state.items():
        value=field if rotation is None else rotation@field
        if phi is not None:value=value@phi.T
        if pw is None:add(result,key,1.,value)
        else:
            for target in np.flatnonzero(abs(pw[:,key[-1]])>1e-14):add(result,key[:-1]+(int(target),),pw[target,key[-1]],value)
    return result


def vector_source(primitive,state,spec,rotor,direction,twice_j):
    positions,kind,m2=spec;j=twice_j/2;m=m2/2
    rotation=[sum((direction[a]*sparse.csr_matrix(rotor['coordinate'][i][a]).toarray() for a in range(3)),np.zeros((10,10),complex)) for i in range(3)]
    result=field_action(source(primitive,state,spec),rotation=2*m*rotation[2])
    for step,combination in ((-1,rotation[0]+1j*rotation[1]),(1,rotation[0]-1j*rotation[1])):
        if -j<=m+step<=j:
            coefficient=np.sqrt((j-step*m)*(j+step*m+1))
            other=field_action(source(primitive,state,(positions,kind,m2+2*step)),rotation=coefficient*combination)
            for key,value in other.items():add(result,key,1.,value)
    return result


def eom_source(primitive,state,spec,scalar,rotor):
    positions,kind,m2=spec
    if len(set(positions))!=1:raise ValueError('This bare D kernel is explicitly LOCAL; separated source has additional electric commutators')
    terms=commutator_terms(primitive,source_operator(positions[0],kind,m2,False))
    groups=apply_terms(primitive,state,terms,scalar,rotor);result={}
    while groups:
        _,group=groups.popitem()
        while group:
            key,value=group.popitem()
            if key in result:result[key]+=value
            else:result[key]=value
    return result


def descriptors(config):
    for j in (1,3):
        for positions in config['compact_anchors']:
            kind='half_triplet' if j==1 else 'threehalf'
            for family in ('bare','vector','wilson','eom'):yield {'twice_J':j,'spec':(positions,kind,j),'family':family}
        for positions in config['separated_anchors']:
            for kind in (('half_triplet','half_singlet') if j==1 else ('threehalf',)):
                for family in ('bare','vector','wilson'):yield {'twice_J':j,'spec':(positions,kind,j),'family':family}


def apply_source(primitive,state,descriptor,scalar,rotor):
    """Literal operator action, equally valid for Omega or actual HnOmega.

    vector creation index is (mu,m), not transposed. W=complex TrU, not ReTrU.
    D is the ZERO-EM bare local commutator, not [H,F B] for other families.
    """
    family=descriptor['family'];spec=descriptor['spec']
    if family=='vector':return vector_source(primitive,state,spec,rotor,np.asarray(primitive.graph['axes'])[0],descriptor['twice_J'])
    if family=='eom':return eom_source(primitive,state,spec,scalar,rotor)
    bare=source(primitive,state,spec)
    if family=='bare':return bare
    if family=='wilson':return field_action(bare,pw=sum((primitive.loop['U'][a,a] for a in range(3)),np.zeros((19,19),complex)))
    raise ValueError('Unknown explicit source family')
