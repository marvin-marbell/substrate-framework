"""Exact factor snapshots of THIS extended H, independent of the fermion representation.
No G/D insertion, old producer replay, fixed-M adapter or CAR-coordinate assumption.
"""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from rich_spatial_colour_em_20261015_model import FactorOperator,MatterMap
from rich_spatial_colour_em_20261015_fields import field_shell,endpoint_connections
from rich_spatial_colour_root_20261015_model import so3,graph
from rich_colour_composite_20261014_model import loop_field


def save_sparse(stored,prefix,matrix):
    matrix=sparse.csr_matrix(matrix)
    for key in ['data','indices','indptr']:stored[prefix+'_'+key]=getattr(matrix,key)
    stored[prefix+'_shape']=matrix.shape


def matrix(data,prefix):
    return sparse.csr_matrix((data[prefix+'_data'],data[prefix+'_indices'],data[prefix+'_indptr']),shape=tuple(data[prefix+'_shape']))


def save_factor(stored,prefix,operator):
    stored[prefix+'_terms']=len(operator.terms);stored[prefix+'_include_scalar']=operator.include_scalar
    for index,(matter,rq,phi) in enumerate(operator.terms):
        save_sparse(stored,prefix+'_A'+str(index),matter);save_sparse(stored,prefix+'_B'+str(index),rq);stored[prefix+'_phi'+str(index)]=phi


def factor(data,prefix,nm,scalar,nrq):
    terms=[(matrix(data,prefix+'_A'+str(index)),matrix(data,prefix+'_B'+str(index)),bool(data[prefix+'_phi'+str(index)])) for index in range(int(data[prefix+'_terms']))]
    return FactorOperator(nm,scalar,nrq,terms,include_scalar=bool(data[prefix+'_include_scalar']))


def save_body_primitives(stored,name,op):
    for key in ['Q','rho0','rho1','density_delta','density_delta2','electric','C','B','N','PW_boundary']:
        save_sparse(stored,name+'_'+key,op[key])
    for axis in range(3):save_sparse(stored,name+'_S'+str(axis),op['S'][axis])
    for edge in range(2):
        for channel,(charge,hops) in enumerate(op['hops'][edge]):
            stored[name+'_q_'+str(edge)+'_'+str(channel)]=charge
            for spin,hop in enumerate(hops):save_sparse(stored,name+'_hop_'+str(edge)+'_'+str(channel)+'_'+str(spin),hop)


def load(physics_json,physics_npz):
    metadata=json.loads(Path(physics_json).read_text());data=np.load(physics_npz);p=metadata['parameters'];grid=data['scalar_grid']
    scalar=dict(grid=grid,H=data['scalar_H'],phi=np.diag(grid),boundary=np.diag([1.]+[0.]*(len(grid)-2)+[1.]))
    rotor=so3(p);geometry=graph(p,rotor);field=field_shell(p['em_cut'],p['em_basis_scale'])
    field['H_E']=data['field_HE'];field['C']=data['field_C'];field['boundary']=data['field_boundary']
    for axis in range(3):field['F'][axis]=data['field_F'+str(axis)]
    for axis in range(6):field['Q'][axis]=data['field_Q'+str(axis)];field['Pi'][axis]=data['field_Pi'+str(axis)]
    operators={};laws={}
    for name,entry in metadata['sectors'].items():
        op={key:matrix(data,name+'_'+key) for key in ['Q','rho0','rho1','density_delta','density_delta2','electric','C','B','N','PW_boundary']}
        op['S']=[matrix(data,name+'_S'+str(axis)) for axis in range(3)];op['hops']=[]
        for edge in range(2):
            channels=[]
            for channel in range(4):
                charge=float(data[name+'_q_'+str(edge)+'_'+str(channel)])
                hops=[matrix(data,name+'_hop_'+str(edge)+'_'+str(channel)+'_'+str(spin)) for spin in range(4)]
                channels.append((charge,hops))
            op['hops'].append(channels)
        operators[name]=op;laws[name]=factor(data,name+'_H',entry['matter_dimension'],scalar,rotor['dimension']*field['dimension'])
    base=dict(operators=operators,scalar=scalar,rotor=rotor,geometry=geometry,loop=loop_field())
    model=dict(base=base,field=field,parameters=p,coefficients=dict(body=data['route_body_coefficients'],values=data['route_coefficients'],raw_electric_gram=data['electric_raw_Gram']),connections=endpoint_connections(geometry,rotor,p),V_self=metadata['V_self'],V_cross=metadata['V_cross'],phase_cache={},mechanical_cache={})
    return metadata,data,model,laws


def source(data,model,name,key):
    base=model['base'];field=model['field'];return MatterMap(matrix(data,name+'_source_'+key+'_matter'),len(base['scalar']['grid']),base['rotor']['dimension'],field['dimension'])


def charge_receiver(data,model,name):
    labels=data[name+'_charge_labels'].tolist()
    effects=[matrix(data,name+'_charge_effect_'+str(index)) for index in range(len(labels))]
    return labels,effects
