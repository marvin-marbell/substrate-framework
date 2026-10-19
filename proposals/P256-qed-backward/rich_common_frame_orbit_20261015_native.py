"""Explicit NEW frame-H rebuild from trusted pinned native BODY operators only.

No primary neutral state, H0, scalar envelope, spectrum or EM field is loaded.
The published native isometry/Gram and physical source matrices are consumed
through its versioned ordinary snapshot API. All old prefixes stay untouched.
"""
import hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from rich_spatial_colour_charge_20261015_snapshot import load
from rich_spatial_colour_root_20261015_model import so3,graph,connection_data
from rich_rotcov_gauge_20261015_sparse import SparseRoundoff
from rich_common_frame_orbit_20261015_frame import RadialForm,spin_basis,gauss_frame,material_spin


def build(p):
    root=Path(__file__).resolve().parent;path=(root/p['native_body']).resolve()
    if path.parent!=root:raise ValueError('Only trusted pinned local project BODY snapshots are accepted')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=p['native_body_sha256']:raise ValueError('Trusted native BODY hash does not match exact selected artifact')
    data=load(path)
    for key in ['vertex_separation','path_height','electric']:
        if p[key]!=data['parameters'][key]:raise ValueError(f'Native BODY physical parameter differs: {key}')
    # Throw away old scalar/lifted Noether constructs, not filesystem data.
    # Only the actual BODY kernels/isometry/raw source metrics enter this H.
    for old in data['operators'].values():
        for key in ['J','J2','connection']:old.pop(key,None)
    for key in ['scalar','sources','C']:data.pop(key,None)
    factory=SparseRoundoff();radial=RadialForm(p);native_rotor=so3(p);material=material_spin(native_rotor);um=material['U']
    def integrate(values):return factory.csr(um.conj().T@native_rotor['integrate'](values)@um)
    rotor=dict(native_rotor,L=material['S'],L2=sum(a@a for a in material['S']),basis=native_rotor['basis']@um,integrate=integrate);geometry=graph(p,rotor)
    sectors={}
    for name,native in data['sectors'].items():
        old=data['operators'][name];spin=spin_basis(old['S'],native['charge3_site0'])
        if not spin['completion']:raise ValueError('Chosen RIGHT Spin requires a complete native packet representation')
        def convert(matrix):return factory.csr(spin['U'].conj().T@matrix@spin['U'])
        op={key:convert(old[key]) for key in ['electric','B','Q','N','density_delta','density_delta2','rho0','rho1','PW_boundary','matter_octet'] if key in old}
        op['hops']=[[(q,[convert(a) for a in matrices]) for q,matrices in edge] for edge in old['hops']];op['S']=spin['S'];op['connection']=connection_data(op,rotor,p)
        sectors[name]=dict(native=native,native_isometry=data['native'][name],spin=spin,op=op,representation='native_charge_orbit',colour_Gauss=None,colour_Gauss_provenance='Pinned native raw-seed/exact-charge Gauss construction; not a fabricated raw P/masks isometry or independently reconstructed colour-G matrix here',right_spin_completion=spin['completion'],native_hop_Gauss=None)
    largest=max(max(s['spin']['j']) for s in sectors.values());frame_js=[x/2 for x in range(int(round(2*largest))+1)]
    for s in sectors.values():s['frame']=gauss_frame(s['spin'],frame_js)
    return dict(parameters=p,loop=data['loop'],radial=radial,rotor=rotor,material=material,geometry=geometry,sectors=sectors,sparse=factory,frame_js=frame_js,native_source_matter=data['source_matter'],native_source_metric_full=data['source_metric_full'],native_source_definitions=data['source_definitions'],native_payload=dict(file=path.name,sha256=digest,source_parameters=data['parameters'],meaning='Numeric native BODY, raw source metrics/isometry only. NEW r³/frame/SU2-gauged H/reference/current/source costs must be recomputed; no old state/H0/spectrum.'))
