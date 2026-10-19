"""One CAR64 source algebra on the original occupied N32 sea.
No tensor product of solved particles; all sources act on arbitrary same-Fock inputs.
"""
from collections import defaultdict
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_cloud_probe import add
from rich_composite_geometry_20261015_source_eom import canonical,bare_state,commutator_terms,apply_terms
from rich_composite_geometry_20261015_dressing_sources import field_action


def combine(terms):
    result=defaultdict(complex)
    for operations,value in terms:
        key,sign=canonical(operations)
        if key is not None:result[key]+=sign*value
    return [(list(key),value) for key,value in result.items() if abs(value)>2e-14]


def operator_terms(descriptor):
    if descriptor['sector'] in (31,33):
        return [([(model.mode(descriptor['site'],1,0,3),descriptor['sector']==33)],1.+0j)]
    if descriptor['sector']!=36:raise ValueError('Explicit N31/N33/N36 source sectors only')
    # Operator order Bdag_x Ldag_y: L acts FIRST on Omega. Both highest spin
    # components give total Noether J=1,M=1; no tensoring independently solved states.
    white=[(model.mode(descriptor['L_site'],1,0,3),True)]
    return combine([(operations+white,value) for operations,value in model.source_terms([descriptor['B_site']]*3,'half_triplet',1)])


def spin_commutator(terms,axis):
    result=[]
    for operations,value in terms:
        for slot,(index,create) in enumerate(operations):
            site,flavour,spin,member=model.unpack(index)
            for other in range(2):
                coefficient=model.PAULI[axis][other,spin]/2 if create else -model.PAULI[axis][spin,other]/2
                if coefficient:
                    changed=operations[:slot]+[(model.mode(site,flavour,other,member),create)]+operations[slot+1:]
                    result.append((changed,value*coefficient))
    return combine(result)


def algebra_receipt(descriptor):
    terms=operator_terms(descriptor);j=descriptor['twice_J']/2;m=(-.5 if descriptor['sector']==31 else j)
    def dictionary(items):return {tuple(ops):value for ops,value in combine(items)}
    z=dictionary(spin_commutator(terms,2));casimir=[]
    for axis in range(3):casimir.extend(spin_commutator(spin_commutator(terms,axis),axis))
    original=dictionary(terms);squared=dictionary(casimir)
    return {'spin_INPUT':True,'source_M':m,'source_j':j,
        'exact_CAR_operator_Sz_residual2':float(sum(abs(z.get(key,0)-m*original.get(key,0))**2 for key in z.keys()|original.keys())),
        'exact_CAR_operator_S2_residual2':float(sum(abs(squared.get(key,0)-j*(j+1)*original.get(key,0))**2 for key in squared.keys()|original.keys())),
        'Noether':'Bare source field independent on actual properJ0Omega: source spin=j implies physical J=Lleft+SCAR=j. ScalarTrU and H commutator preserve rank; vector is 2 sum_i(Rd)_i[S_i,O], correct creation/annihilation indices. Count projectors/Phi slices commute J. NOT field-only spin origin or full continuum Noether.'}


def descriptors(config):
    if config['mode']=='carrier':
        for sector in (33,31):
            for site in config['carrier_sites']:
                for family in ('bare','vector','wilson','eom'):yield {'sector':sector,'site':site,'twice_J':1,'family':family}
    elif config['mode']=='joint':
        for baryon,lepton in config['joint_pairs']:
            for family in ('bare','vector','wilson','eom'):yield {'sector':36,'B_site':baryon,'L_site':lepton,'twice_J':2,'family':family}
    else:raise ValueError('Unknown declared construction mode')


def apply_source(primitive,state,descriptor,scalar,rotor):
    terms=operator_terms(descriptor);family=descriptor['family']
    if family=='eom':
        # Bare local colour-singlet B and white L commute HE, Hphi, Hrot and
        # plaquette. This exact CAR commutator retains ALL scalar/vector q+white
        # replacement terms; source is unmasked on the whole occupied sea.
        groups=apply_terms(primitive,state,commutator_terms(primitive,terms),scalar,rotor);result={}
        while groups:
            _,group=groups.popitem()
            while group:
                key,value=group.popitem()
                if key in result:result[key]+=value
                else:result[key]=value
        return result
    if family=='vector':
        result={};direction=np.asarray(primitive.graph['axes'])[0]
        for axis in range(3):
            rotation=sum((direction[a]*sparse.csr_matrix(rotor['coordinate'][axis][a]).toarray() for a in range(3)),np.zeros((10,10),complex))
            part=field_action(bare_state(primitive,state,spin_commutator(terms,axis)),rotation=2*rotation)
            for key,value in part.items():add(result,key,1.,value)
        return result
    bare=bare_state(primitive,state,terms)
    if family=='bare':return bare
    if family=='wilson':return field_action(bare,pw=sum((primitive.loop['U'][a,a] for a in range(3)),np.zeros((19,19),complex)))
    raise ValueError('Unknown physical source family')
