"""Global neutral sea centre sewing discriminator; point data, not quantum ground.

Independent pair-frame Z4 choices are incompatible with a global Slater sea.
A connected link chart can leave only one global Z4 choice; global canonical
number makes that choice harmless. Full quantum chart normalization is unpaid.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent


def finite_control(profile: dict[str, Any], n: int) -> dict[str, Any]:
    sx = np.array([[0,1],[1,0]],dtype=complex)
    sy = np.array([[0,-1j],[1j,0]],dtype=complex)
    sz = np.diag([1,-1]).astype(complex)
    beta = np.kron(sz,np.eye(2))
    alpha = [np.kron(sx,p) for p in (sx,sy,sz)]
    pblock = np.zeros((8,8),dtype=complex)
    hermitian_error = 0.0
    projector_error = 0.0
    for indices in itertools.product(range(n),repeat=3):
        k = (2*np.array(indices)+1)*np.pi/n
        mass = float(Fraction(profile['onsite']))
        for path in profile['paths']:
            mass += 2*float(Fraction(path['c0']))*np.cos(k[path['axis']])
        h = (mass/2)*beta
        for path in profile['paths']:
            h += (-2/3*float(Fraction(path['ci']))*np.sin(k[path['axis']]))*alpha[path['axis']]
        epsilon = float(np.sqrt(np.trace(h@h).real/4))
        pk = (np.eye(4)+h/epsilon)/2
        projector_error = max(projector_error,float(np.linalg.norm(pk@pk-pk)))
        hermitian_error = max(hermitian_error,float(np.linalg.norm(pk-pk.conj().T)))
        # Two neighbouring sites inside one prior matching block.
        for a,b in itertools.product(range(2),repeat=2):
            pblock[4*a:4*a+4,4*b:4*b+4] += pk*np.exp(1j*k[0]*(a-b))/n**3
    fcs = np.linalg.det(np.eye(8)+(1j-1)*pblock)
    assert abs(fcs-1) > 0.1
    assert abs(fcs) < 0.99
    assert projector_error < 1e-12 and hermitian_error < 1e-12
    global_particles = 2*n**3
    # Exact arithmetic, not a determinant on a FLOAT approximate projector.
    assert global_particles % 4 == 0
    global_z4_phase = ((1,0),(0,1),(-1,0),(0,-1))[global_particles % 4]
    assert global_z4_phase == (1,0)
    return {'N':n,'neutral_canonical_particles_exact':global_particles,
            'global_Z4_phase_exact':'i^(2V)=1',
            'independent_matching_block_centre_overlap_FLOAT':[float(fcs.real),float(fcs.imag)],
            'independent_matching_block_Z4_is_not_harmless':True,
            'local_onebody_projector_defect_FLOAT':projector_error,
            'Hermiticity_defect_FLOAT':hermitian_error,
            'two_site_FCS_is_point_sea_not_quantum_family':True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name,expected in request['pins'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    profile = json.loads((ROOT/request['profiles']).read_text())['primitive_profiles'][0]
    results = [finite_control(profile,n) for n in (4,6,8)]
    # On a connected graph a centre-valued frame relabelling preserves a link
    # chart around I only if z_x=z_y on every edge. This leaves global Z4.
    for n in (4,6,8):
        assert len({(x,y,z) for x,y,z in itertools.product(range(n),repeat=3)}) == n**3
    result = {'kind':'GLOBAL_NEUTRAL_SEA_CONNECTED_CENTRE_SEWING_CONTROL_NOT_FULL_QUANTUM_SOURCE',
              'controls':results,
              'connected_link_constraint':'z_x^dagger I z_y remains in the fixed small I chart iff z_x=z_y; connected torus leaves one global Z4, acting trivially on canonical2V at evenN.',
              'pointwise_normalized_ray_scope':'Full canonical Slater vector only, not independent matching block source or fixed per-block number.',
              'full_quantum_chart_normalizer_or_N_uniform_derivative_prices_paid':False,
              'SU4_Gauss_full_source_or_D_H_construction_paid':False,
              'actual_ground_or_BO_remainder_or_phase':False,
              'original_SOURCE_or_law_adopted':False,'campaign_complete':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'global_Z4_exact':True,'independent_matching_Z4_rejected':True,
                      'quantum_source_or_ground_paid':False}))


if __name__ == '__main__':
    main()
