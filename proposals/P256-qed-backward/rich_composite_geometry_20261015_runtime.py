"""Explicit serialized coordinate dispatch, never a physical-state/cut fallback."""
from rich_composite_geometry_20261015_model import Primitives,string,conjugate


def primitive_for(data,canonical=None):
    if 'primitive' not in data:
        if canonical is not None and hasattr(canonical,'local'):raise ValueError('Different physical coordinate representations')
        return Primitives(data['p'])
    primitive=data['primitive']
    if canonical is not None:
        previous=primitive.local.vectors;current=canonical.local.vectors
        if len(previous)>len(current) or any(a!=b for a,b in zip(previous,current)):
            raise ValueError('Local orthonormal coordinate dictionary prefix changed')
        # Later exact local-coordinate extensions leave every earlier isometry unchanged.
        data['basis'].primitive=canonical;primitive=canonical
    return primitive


def conjugate_packet(primitive,packet):
    if hasattr(primitive,'local'):return primitive.conjugate(packet)
    return conjugate(packet)


def coordinate_operations(primitive,key,operations):
    if hasattr(primitive,'local'):return primitive.coordinate_operations(key,operations)
    mask,pw=key;step=string(mask,operations)
    return {} if step is None else {(step[0],pw):step[1]}
