"""Safe standalone carrier field accessors; immutable model is not monkeypatched."""
import numpy as np
from scipy import sparse
from rich_common_carrier_motion_20261015_model import observable
from rich_common_frame_orbit_20261015_model import FormOperator
from rich_spatial_colour_em_20261015_fields import gaussian_green


def field_observable(model,name,kind,axis=None):
    """Projected physical field Q/Pi/F/HE or material L, in explicit rq basis."""
    nf=model['field']['dimension']
    if kind=='material_L':matrix=sparse.kron(model['physical_rotor']['L'][axis],sparse.eye(nf))
    else:
        value=model['field']['H_E'] if kind=='HE' else model['field'][kind][axis]
        matrix=sparse.kron(sparse.eye(10),value)
    return observable(model,name,rq=model['rq'](matrix))


def transverse_off(model,name,bridge=True):
    """Remove BOTH transverse Peierls and charged mechanical feedback.

    Coulomb, colour electric/loop, radial, all declared primitive hopping,
    material kinetic and field electric kinetic remain. NOT bridge-off.
    """
    p=model['parameters'];s=model['sectors'][name];op=s['op'];rot=model['physical_rotor'];nf=model['field']['dimension'];rq=model['rq'];eye=np.eye(10*nf);terms=[]
    matter=op['electric']-p['plaquette']*op['B']
    for a in range(4):
        for b in range(4):
            v=float(gaussian_green(np.linalg.norm(model['geometry']['vertices'][a]-model['geometry']['vertices'][b]),p['charge_sigma'],p['epsilon']))
            matter=matter+op['rho'][a]@op['rho'][b]*v/2
    terms.append((matter,eye,0));terms.append((sparse.eye(s['dimension']),rq(sparse.kron(rot['L2']/(2*p['orientation_inertia']),sparse.eye(nf))+sparse.kron(sparse.eye(10),model['field']['H_E'])),0))
    for edge,channels in enumerate(op['hops']):
        if edge==2 and not bridge:continue
        for charge,matrices in channels:
            for axis in [None,0,1,2]:
                angle=eye if axis is None else rq(sparse.kron(rot['integrate'](model['geometry']['tangents'][:,edge,axis]),sparse.eye(nf)))
                a=(p['scalar_coupling']*matrices[0] if axis is None else 1j*p['vector_velocity']*matrices[axis+1])*model['geometry']['weights'][edge]
                terms.append((a,angle,1));terms.append((a.conj().T,angle.conj().T,1))
    return FormOperator(s,s,model['rotor'],model['radial'],terms,{float(j):model['radial'].operators(float(j))['H'] for j in set(s['spin']['j'])})
