"""Publish a NEW prepared neutral input and immutable consumer binding early."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import resource
import time
import zipfile
import numpy as np
from rich_local_frame_carrier_20261015_model import Model, DEFAULTS, prepared_reference

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_local_frame_carrier_20261015_'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    (ROOT/(PREFIX+name+'.json')).write_text(json.dumps(data, indent=2)+'\n')


def main():
    started = time.monotonic()
    files = [PREFIX+'model.py', PREFIX+'equations.json', PREFIX+'prepare.py',
             'rich_frame_holonomy_20261014_model.py', 'rich_frame_gauss_20261014_model.py',
             'rich_frame_gauss_join_20261015_model.py']
    save('input', dict(parameters=DEFAULTS, source_site=0, source_component=0, source_orbital=0,
                       branches=['addition', 'removal'], times=[0., .25, .5],
                       owning_reference='NEW_PREPARED_NONSTATIONARY', reference_number=6))
    sources = {name: sha(ROOT/name) for name in files+[PREFIX+'input.json']}
    with zipfile.ZipFile(ROOT/(PREFIX+'source_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as capsule:
        for name in sources:
            capsule.write(ROOT/name, name)
    save('reference_start', dict(source_SHA256=sources, source_capsule_SHA256=sha(ROOT/(PREFIX+'source_capsule.zip'))))
    model = Model(6, DEFAULTS)
    reference = prepared_reference(model)
    h, _, _ = model.law()
    hpsi = h@reference
    energy = float(np.vdot(reference, hpsi).real)
    variance = float(np.vdot(hpsi-energy*reference, hpsi-energy*reference).real)
    np.savez_compressed(ROOT/(PREFIX+'neutral_reference.npz'), psi=reference,
                        gauge_basis=np.array(model.sector.states, np.int16),
                        radial_nodes=model.r, shape=np.array(model.shape))
    reference_hash = sha(ROOT/(PREFIX+'neutral_reference.npz'))
    interface = dict(kind='FROZEN_EXECUTABLE_OWNING_LOCAL_MODEL_AND_NEW_REFERENCE',
        module=PREFIX+'model', model_api='Model(number, parameters), number5/6/7; full sparse scipy CSR operators',
        parameters=DEFAULTS, tensor_order=['literal_Gauss_CAR12', 'material_SO3_10', 'r0', 'r1', 'r2'],
        neutral_shape=model.shape, neutral_dimension=model.dimension, neutral_basis_SHA256=model.basis_hash,
        gauge_basis_SHA256=model.sector.basis_hash, source_SHA256=sources,
        neutral_reference=dict(path=PREFIX+'neutral_reference.npz', key='psi', SHA256=reference_hash,
                               status='NEW owning prepared nonstationary INPUT; consumer does not claim independent ground solve',
                               energy=energy, energy_variance=variance),
        law_api='law(variant=full/path_cut/vector_off/all_hop_off,phase=0,phase_edge=0)->(H,current[3],contact[3]); q=-1, exp(-i phase)',
        source_api='target.source_from(reference,site,branch=addition/removal,component=0/1,orbital=0/1)->CSR; preserves radial/material, literal boundary tree and CAR',
        density_api='n[v], Q[v]=2-n[v]; count_projector(v,count0..4); sector.density/electric/states/lookup',
        local_noether_api='S_local[v][axis], S2_local[v]; S[axis]=sum_v S_v; local_spin_projector(v,j=0/.5/1)',
        field_api='r2_local[v], E[e], HE, F=Wilson_sine, Wcos; lift_gauge(A), lift_material(A), lift_radial(A)',
        angular_api='material_L[axis], J[axis]=S+L; no invented U1 field angular momentum',
        root_api='Hroot,Hframe,Hradial,Hmaterial; root_band_count(site,signed=+1/-1); architecture raw Gauss/LEFT/source and Gamma constitutive matrices',
        transport_api='scalar_transport[e],vector_transport[e] actual forwards; all products preprojection',
        primitive_provenance=model.architecture['provenance'],
        neutral_local_n=[float(np.vdot(reference,a@reference).real) for a in model.n],
        neutral_local_S2=[float(np.vdot(reference,a@reference).real) for a in model.S2_local],
        neutral_local_r2=[float(np.vdot(reference,a@reference).real) for a in model.r2_local],
        warning='Signed density difference is NOT added-particle PVM; r² is NOT spatial radius; finite material triangle no Nature/COM/Maxwell photon theorem')
    save('interface', interface)
    save('reference_exit', dict(natural_exit=True, exit_code=0, elapsed_seconds=time.monotonic()-started,
                               max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                               reference_SHA256=reference_hash, model_SHA256=sources[PREFIX+'model.py'],
                               dimension=model.dimension, energy_variance=variance))
    print(json.dumps(dict(interface=PREFIX+'interface.json', dimension=model.dimension,
                          reference_SHA256=reference_hash, model_SHA256=sources[PREFIX+'model.py'],
                          variance=variance)), flush=True)


if __name__ == '__main__':
    main()
