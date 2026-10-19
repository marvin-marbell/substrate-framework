"""Primary join with NEW raw physical-site-charge orbit body, not bandpair control.
Only native finite-body operators/source metrics/C/PVMs cross this interface.
"""
from rich_spatial_colour_charge_20261015_model import build as native_build
from rich_spatial_colour_em_20261015_fields import field_shell,route_coefficients,endpoint_connections,gaussian_green
from rich_spatial_colour_em_20261015_model import assemble,angular,observable,MatterMap


def from_body(body,p):
    field=field_shell(p['em_cut'],p['em_basis_scale'])
    coefficients=route_coefficients(body['geometry'],body['rotor'],p)
    connections=endpoint_connections(body['geometry'],body['rotor'],p)
    return dict(base=body,field=field,coefficients=coefficients,connections=connections,parameters=p,
        V_self=float(gaussian_green(0.,p['charge_sigma'],p['epsilon'])),
        V_cross=float(gaussian_green(p['vertex_separation'],p['charge_sigma'],p['epsilon'])),
        phase_cache={},mechanical_cache={})


def build(p,checkpoint=None):
    return from_body(native_build(p,checkpoint=checkpoint),p)


def native_source(model,name,key):
    body=model['base'];return MatterMap(body['source_matter'][name][key],len(body['scalar']['grid']),body['rotor']['dimension'],model['field']['dimension'])


def native_C(model,name):
    body=model['base'];return MatterMap(body['C_matter'][name],len(body['scalar']['grid']),body['rotor']['dimension'],model['field']['dimension'],model['field']['C'])


def native_charge_receiver(model,name):
    measures=model['base']['sectors'][name]['charge_PVM'];labels=sorted(measures)
    return [h/3 for h in labels],[measures[h] for h in labels]
