"""Trusted local exact numeric/native-isometry snapshot, not a producer replay."""
import pickle
from pathlib import Path
from scipy import sparse
from rich_spatial_colour_charge_20261015_model import (
    loop_field,scalar_field,so3,graph,connection_data,clean,BodyLift,source_operator)


def save(data,parameters,path):
    operators={name:{k:v for k,v in op.items() if k not in ['connection','J','J2']} for name,op in data['operators'].items()}
    payload=dict(parameters=parameters,operators=operators,source_matter=data['source_matter'],source_metric_full=data['source_metric_full'],C_matter=data['C_matter'],native=data['native'],source_definitions=data['source_definitions'],root=data['root'])
    with Path(path).open('xb') as handle:pickle.dump(payload,handle,protocol=5)


def load(path):
    # ONLY exact locally produced/pinned snapshots; pickle is not an untrusted-data parser.
    with Path(path).open('rb') as handle:data=pickle.load(handle)
    p=data['parameters'];rotor=so3(p);scalar=scalar_field(p);geometry=graph(p,rotor);sectors={}
    for name,basis in data['native'].items():
        op=data['operators'][name];op['connection']=connection_data(op,rotor,p)
        op['J']=[clean(sparse.kron(s,sparse.eye(10))+sparse.kron(sparse.eye(basis.dimension),l)) for s,l in zip(op['S'],rotor['L'])];op['J2']=clean(sum(j@j for j in op['J']))
        sectors[name]=dict(dimension=basis.dimension,native_isometry=basis,charge3_site0=basis.h,twice_Sz=basis.m,charge_PVM={h:sparse.diags((basis.h==h).astype(float),format='csr') for h in sorted(set(basis.h))},P=None,masks=None)
    data.update(loop=loop_field(),rotor=rotor,scalar=scalar,geometry=geometry,sectors=sectors,
        sources={name:{label:BodyLift(mat,10*len(scalar['grid'])) for label,mat in values.items()} for name,values in data['source_matter'].items()},
        C={name:source_operator(mat,rotor,scalar) for name,mat in data['C_matter'].items()})
    return data
