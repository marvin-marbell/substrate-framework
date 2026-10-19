"""New physical fork: remove explicit spin-orbit, retain moving rotational loading."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from rich_composite_rot_20261014_consumer import (COORDS, EDGES, summary, current_contact,
    expectation, spectrum, polarized_ensemble_response)


def evaluate(p, output):
    selected, (h, m, psi, minus) = summary(p)
    magnetic, mc = current_contact(m, [0, 0, 0], [0, 0, 0], magnetic=True)
    no_spin, (nh, nm, npsi, _) = summary(dict(p, loading_g=0.))
    nj, _ = current_contact(nm, [0, 0, 0], [0, 0, 0], magnetic=True)
    reversed_orientation, (rh, rm, rpsi, _) = summary(dict(p, axial_bias=-p['axial_bias']), fragments=False)
    rj, _ = current_contact(rm, [0, 0, 0], [0, 0, 0], magnetic=True)
    isotropic, (ih, im, ipsi, _) = summary(dict(p, axial_bias=0.), fragments=False)
    ij, _ = current_contact(im, [0, 0, 0], [0, 0, 0], magnetic=True)
    response = []
    for q in ([0., 0., 0.], [1.1, .7, 0.]):
        js, cs = zip(*(current_contact(m, q, pol) for pol in np.eye(3)))
        sources = np.stack([j @ psi for j in js], axis=1); gram = sources.conj().T @ sources
        charge = np.exp(1j*(COORDS[m['positions']] @ q)) @ m['charges']; rho = np.repeat(charge, m['ns']*m['nf'])
        longitudinal = sum(q[a]*js[a] for a in range(3))
        continuity = float(np.linalg.norm(h @ (rho*psi)-selected['energy']*rho*psi-longitudinal @ psi))
        response.append(dict(q=q, current_gram_real=gram.real.tolist(), current_gram_imag=gram.imag.tolist(),
                             contact_diagonal=[expectation(psi, c) for c in cs],
                             spectral_diagonal=[spectrum(h, selected['energy'], sources[:, a], p['krylov_steps']) for a in range(3)],
                             full_density_continuity_source_error=continuity))
    bonds = []
    for v, w in EDGES:
        operator = sparse.csr_matrix(h.shape, dtype=complex)
        for term, charge, start, end in m['hops']:
            if np.array_equal(start, COORDS[v]) and np.array_equal(end, COORDS[w]):
                operator += -1j*charge*term
            elif np.array_equal(start, COORDS[w]) and np.array_equal(end, COORDS[v]):
                operator += 1j*charge*term
        bonds.append(dict(edge=[v, w], polarization=polarized_ensemble_response(m, operator)))
    result = dict(candidate='AXIAL_PRETWIST_MOVING_SPIN_DIFFERENCE_LOADING_NO_EXPLICIT_SO_V1', inputs=p,
                  selected=selected, magnetic_polarization=polarized_ensemble_response(m, magnetic),
                  magnetic_contact=expectation(psi, mc), magnetic_spectrum=spectrum(h, selected['energy'], magnetic @ psi, p['krylov_steps']),
                  no_spin_loading=dict(state=no_spin, magnetic=polarized_ensemble_response(nm, nj)),
                  reverse_axial_pretwist=dict(state=reversed_orientation, magnetic=polarized_ensemble_response(rm, rj)),
                  zero_axial_pretwist=dict(state=isotropic, magnetic=polarized_ensemble_response(im, ij)),
                  physical_bond_currents=bonds, response=response,
                  mechanism='Bare hopping is spin independent. Spin-dependent orbital current arises from transporting position-conditioned neutral rotational dressing with -g P.(S_v-S_w), under an explicit time-even axial material pre-twist. No external J, bare Pauli moment or explicit spin-orbit hopping.',
                  boundary='Only internal Jz conserved on axial square material. Measured J2 is not .75 and its transverse generators fail to commute with H. This is not a rotationally invariant spin-half free particle.')
    Path(output+'.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(output+'.npz', ground_plus=psi, ground_minus=minus, ground_Jz_block=m['ground_basis'],
                        ground_Jz_values=m['ground_magnetic_numbers'], positions=m['positions'], charges=m['charges'])
    print(json.dumps(dict(selected=selected, magnetic=result['magnetic_polarization'], no_spin=result['no_spin_loading'],
                          reversed_magnetic=result['reverse_axial_pretwist']['magnetic'], zero_bias_magnetic=result['zero_axial_pretwist']['magnetic'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); evaluate(json.loads(Path(args.request).read_text()), args.output)
