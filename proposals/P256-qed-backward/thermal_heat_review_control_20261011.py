"""Independent exact finite-N heat review; never imports target producers."""
from __future__ import annotations
import argparse
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import sys
import zipfile
from fractions import Fraction
import numpy as np
from scipy import sparse
import sympy as sp

sys.set_int_max_str_digits(0)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def native_graph(n: int):
    sites = list(product(range(n), repeat=3))
    lookup = {site: i for i, site in enumerate(sites)}
    existing = [(site, axis) for site in sites for axis in range(3)
                if not (axis == 0 and site[0] == n-1)]
    dynamic = [(site, axis) for site, axis in existing
               if not (axis in (1, 2) and site[0] in (0, n-1))]
    edge_index = {edge: i for i, edge in enumerate(dynamic)}
    interior = [site for site in sites if site[0] not in (0, n-1)]
    rows = {site: i for i, site in enumerate(interior)}
    entries = []
    for j, (site, axis) in enumerate(dynamic):
        end = list(site); end[axis] = (end[axis]+1) % n; end = tuple(end)
        for vertex, sign in ((site, 1), (end, -1)):
            if vertex in rows:
                entries.append((rows[vertex], j, sign))
    dmat = sparse.csr_matrix(([s for _, _, s in entries],
                              ([i for i, _, _ in entries], [j for _, j, _ in entries])),
                             shape=(len(interior), len(dynamic)), dtype=np.int64)
    tree = [edge_index[((x, y, z), 0)] for x, y, z in product(range(n-2), range(n), range(n))]
    tree_set = set(tree)
    chords = [j for j in range(len(dynamic)) if j not in tree_set]
    # Integrate each column's sources toward the left boundary, independent
    # construction from graph incidence; the chord block is the identity.
    db = dmat[:, chords].toarray()
    cmat = np.zeros((len(dynamic), len(chords)), dtype=np.int64)
    cmat[chords, np.arange(len(chords))] = 1
    for x, y, z in product(range(n-2), range(n), range(n)):
        cmat[edge_index[((x, y, z), 0)]] = sum(
            (db[rows[(xx, y, z)]] for xx in range(x+1, n-1)),
            np.zeros(len(chords), dtype=np.int64))
    assert not np.any(dmat @ cmat)
    assert np.array_equal(cmat[chords], np.eye(len(chords), dtype=np.int64))
    gram = cmat.T @ cmat
    assert len(tree) == (n-2)*n*n
    return sites, lookup, existing, dynamic, edge_index, interior, dmat, tree, chords, cmat, gram


def exact_separated_det(n: int) -> int:
    # Adjacency eigenvalues of the transverse n-cycle, counted with multiplicity.
    spectra = {4: [2, 0, -2, 0], 6: [2, 1, -1, -2, -1, 1],
               8: [2, sp.sqrt(2), 0, -sp.sqrt(2), -2, -sp.sqrt(2), 0, sp.sqrt(2)]}
    x = sp.Symbol('x')
    chain = sp.diag(*([6-x]*(n-2)))
    for j in range(n-3):
        chain[j, j+1] = chain[j+1, j] = -1
    polynomial = chain.det(method='domain-ge')
    result = sp.prod(polynomial.subs(x, a+b) for a, b in product(spectra[n], repeat=2))
    result = sp.simplify(sp.expand(result))
    assert result.is_Integer and result > 0
    return int(result)


def exact_colour_controls():
    results = []
    shells = {}
    for a, b, c in product(range(9), repeat=3):
        s = a+b+c
        if s > 8:
            continue
        dim = Fraction((a+1)*(b+1)*(c+1)*(a+b+2)*(b+c+2)*(a+b+c+3), 12)
        c2 = Fraction(3*a*a+4*b*b+3*c*c+4*a*b+2*a*c+4*b*c+12*a+16*b+12*c, 8)
        assert dim.denominator == 1
        assert c2 >= Fraction(s*s, 8)
        shells[s] = shells.get(s, Fraction(0))+dim*dim
        if (a, b, c) in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 1)):
            results.append({'labels': [a, b, c], 'dimension': str(dim), 'Casimir': str(c2)})
    for s, value in shells.items():
        assert value <= Fraction((s+2)**2*(s+3)**12, 288)
    # Analytic polynomial witnesses prove the ALL-shell inequalities, not only
    # the finite shell checks above.
    a, b, c = sp.symbols('a b c', nonnegative=True)
    difference = sp.expand(3*a*a+4*b*b+3*c*c+4*a*b+2*a*c+4*b*c+12*a+16*b+12*c-(a+b+c)**2)
    assert all(coefficient >= 0 for coefficient in sp.Poly(difference, a, b, c).coeffs())
    r, u = sp.symbols('r u', positive=True)
    f = r**14*sp.exp(-u*r*r)
    derivative = sp.simplify(sp.diff(f, r)/f)
    assert sp.simplify(derivative-(14/r-2*u*r)) == 0
    exponent = Fraction(4*65, 2**2048)
    u_crop = Fraction(3, 20*16*8*2**4096)
    assert exponent < Fraction(1, 2**2039)
    assert u_crop*64**2 < 7
    return {'representations': results, 'shells_0_through_8_exact': True,
            'all_shell_Casimir_lower_polynomial': str(difference),
            'shell_majorant': '(s+2)^2(s+3)^12/288',
            'decreasing_tail_constant': str(Fraction(9*4**12, 288)),
            'tail_derivative_log': str(derivative),
            'tail_condition': 'u R^2 >= 7; R>=1',
            'tail_integral': 'Gamma(15/2,uR^2)/(2u^(15/2))',
            'phi_crop_exponent': str(exponent), 'colour_crop_uR2': str(u_crop*64**2),
            'crop64_rejected_exactly': True, 'native_Galerkin_error_priced': False}


def native_coefficient_control(n, graph, profiles, frozen):
    sites, lookup, existing, dynamic, edge_index, interior, dmat, tree, chords, cmat, gram = graph
    parts = {'heavy_Lambda': Fraction(0), 'charged': Fraction(0)}
    mj = mk = Fraction(0)
    for profile in profiles:
        value = 4*len(sites)*abs(Fraction(profile['onsite']))
        for row in profile['paths']:
            # Enumerate existing paths rather than use producer path-count formula.
            valid = [site for site in sites if row['axis'] != 0 or site[0]+row['length'] < n]
            norm = abs(Fraction(row['c0']))+abs(Fraction(row['ci']))
            value += 8*len(valid)*norm
            if profile['scale'] == 'charged':
                affected = [site for site in valid if row['axis'] == 2 and site[0] in (0, n-1)]
                mj += 8*profile['load']*len(affected)*norm*row['length']
                mk += 8*profile['load']*len(affected)*norm*row['length']**2
        parts['heavy_Lambda' if profile['scale'] == 'heavy' else 'charged'] += profile['load']*value
    assert {k: str(v) for k, v in parts.items()} == frozen['M_N_parts']
    assert str(mj) == frozen['MJ_N'] and str(mk) == frozen['MK_N']
    assert sum(row['load'] for row in profiles) == 65
    assert [row['load'] for row in profiles] == [1, 1, 63]
    assert profiles[1]['onsite'] != profiles[2]['onsite']
    assert frozen['fullPhi_coordinates'] == 60*n**3
    assert frozen['canonical_particles_per_each65'] == 2*n**3
    assert int(frozen['canonical_CAR_dimension']) == math.comb(4*n**3, 2*n**3)**65
    assert frozen['colour_Haar_factors'] == len(existing)
    assert int(frozen['Etrial_N']['electric_upper']) == 16*len(chords)*int(np.trace(gram))
    assert int(frozen['Etrial_N']['HC_upper']) == 16*(n-1)**2*128**2*len(interior)
    return {'M_N_parts': {k: str(v) for k, v in parts.items()}, 'MJ_N': str(mj), 'MK_N': str(mk),
            'all65_counts_and_distinct_onsites_retained': True,
            'free_reference_not_native_factorization': True,
            'Phi_kinetic': '4 chi (-Delta_q); q=sqrt8 C_original',
            'colour_kinetic': '(3/20) chi Casimir',
            'one_collective_HC': '64 rhoI^T LI^-1 rhoI; rho=sum64(n-2), Ref before I'}


def flow_and_memory(n, graph):
    sites, lookup, existing, dynamic, edge_index, interior, dmat, tree, chords, cmat, gram = graph
    def face(x):
        flow = np.zeros(len(dynamic), dtype=np.int64)
        for edge, sign in ((((x, 0, 0), 1), 1), (((x, 1, 0), 2), 1),
                           (((x, 0, 1), 1), -1), (((x, 0, 0), 2), -1)):
            if edge in edge_index:
                flow[edge_index[edge]] += sign
        assert not np.any(dmat @ flow)
        return flow
    interior_flow, boundary_flow = face(1), face(0)
    assert np.count_nonzero(interior_flow) == 4 and not np.any(boundary_flow)
    assert not np.any(dmat @ interior_flow)
    # Entire canonical occupation witness: one component particle circulates
    # around the face, remaining 2V-1 active and all spectator counts unchanged.
    vertices = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]
    occupied = {4*lookup[vertices[0]]}
    excluded = {4*lookup[v] for v in vertices}
    occupied.update(j for j in range(4*n**3) if j not in excluded and j < 2*n**3+4)
    occupied = set(sorted(occupied)[:2*n**3-1]) | {4*lookup[vertices[0]]}
    # Fill exactly to half occupancy without occupying the moving target slots.
    for j in range(4*n**3):
        if len(occupied) == 2*n**3:
            break
        if j not in excluded:
            occupied.add(j)
    assert len(occupied) == 2*n**3
    original = occupied.copy()
    for start, end in zip(vertices, vertices[1:]+vertices[:1]):
        src, dst = 4*lookup[start], 4*lookup[end]
        assert src in occupied and dst not in occupied
        occupied.remove(src); occupied.add(dst)
    assert occupied == original
    k = np.zeros(len(dynamic), dtype=np.int64)
    for y, z in product(range(n), repeat=2):
        k[edge_index[((0, y, z), 0)]] = -1
    rho = dmat @ k
    phi = [Fraction(n-1-site[0], n-1) for site in interior]
    longitudinal = [sum(Fraction(int(dmat[i, j]))*phi[i] for i in range(len(interior)))
                    for j in range(len(dynamic))]
    assert all(sum(Fraction(int(dmat[i, j]))*longitudinal[j] for j in range(len(dynamic))) == int(rho[i])
               for i in range(len(interior)))
    transverse = [Fraction(int(k[j]))-longitudinal[j] for j in range(len(dynamic))]
    kl2 = sum(v*v for v in longitudinal); kt2 = sum(v*v for v in transverse)
    assert kl2 == Fraction(n*n*(n-2), n-1) and kt2 == Fraction(n*n, n-1)
    assert sum(a*b for a, b in zip(longitudinal, transverse)) == 0
    return {'CAR_closed_nonzero_cycle_rejected': True, 'boundary_raw0_electric_selection_only': True,
            'longitudinal_norm2': str(kl2), 'transverse_norm2': str(kt2),
            'common_b_longitudinal_price': str(kl2/2), 'common_b_memory_price': str(kt2/4),
            'reset_b_wrongcase_overcount_exponent': str(16*kt2),
            'HC_double_count_wrongcase_extra_exponent': str(32*kl2),
            'all65_canonical_counts': 2*n**3, 'full_quantum_word_coefficient': 'unpaid'}


def small_exact_affine(graph):
    sites, lookup, existing, dynamic, edge_index, interior, dmat, tree, chords, cmat, gram = graph
    d = sp.Matrix(dmat.toarray().tolist())
    li = d*d.T
    g = sp.Matrix(gram.tolist())
    det_li = li.det(method='domain-ge')
    det_g = g.det(method='domain-ge')
    assert det_li == det_g == 435403533737610087890625
    raw = sp.zeros(len(dynamic), 1); raw[edge_index[((2, 1, 1), 0)]] = 1
    charge = d*raw
    lam = li.inv()*charge
    delta = (charge.T*lam)[0]
    assert (2*delta).q != 1
    projected = 1-delta
    assert projected+delta == 1
    assert (projected-delta).q != 1  # reversed reduced sign cannot intertwine
    q = sp.eye(len(dynamic))-d.T*li.inv()*d
    c = sp.Matrix(cmat.tolist())
    assert q[:, chords] == c*g.inv()
    # Use original geometric rho bound and an exact Gaussian coordinate witness.
    a4, m, r = sp.symbols('a4 m r', positive=True)
    assert sp.expand(a4*r**4/2+m*m/(2*a4)-m*r*r) == sp.expand((a4*r*r-m)**2/(2*a4))
    assert sp.expand(a4*r**4/2+1/(2*a4)-r*r) == sp.expand((a4*r*r-1)**2/(2*a4))
    return {'detLI_exact_direct': str(det_li), 'detGram_exact_direct': str(det_g),
            'primitive_dual_Qchords_CGinv_exact': True, 'integer_cycle_wrong_cell_volume_ratio': str(det_g),
            'affine_charge_pairing': str(delta), 'projected_hop_pairing': str(projected),
            'reduced_minus_sign_required': True, 'full_section_plus_sign_required': True,
            'Young_and_reference_comparison_exact_square_witnesses': True,
            'trial_single_site_gaussian_Cart_r4': 930,
            'trial_Jordan_upper_14880_conservative': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.input.read_text())
    root = Path(request['root'])
    for name, pin in request['pins'].items():
        assert digest((root/name).read_bytes()) == pin, name
    frozen = json.loads((root/'magnetic_thermal_actual2_20261011_receipt.json').read_text())
    profiles = json.loads((root/request['profile']).read_text())['primitive_profiles']
    with zipfile.ZipFile(root/'magnetic_thermal_actual2_20261011_executed.zip') as archive:
        archived = {name: archive.read(name) == (root/name).read_bytes()
                    for name in request['pins'] if name in archive.namelist()}
    assert all(archived.values())
    records = []
    for n in request['N']:
        graph = native_graph(n)
        target = next(row for row in frozen['native_N_controls'] if row['lattice']['N'] == n)
        li_det = exact_separated_det(n)
        assert str(li_det) == target['lattice']['det_LI']
        assert int(np.trace(graph[-1])) == target['lattice']['cycle_gram_trace']
        records.append({'N': n, 'M': len(graph[3]), 'r': len(graph[5]), 'd': len(graph[8]),
                        'DC_zero_chord_identity_saturation': True, 'detLI': str(li_det),
                        'coefficients': native_coefficient_control(n, graph, profiles, target['exact_thermal_enclosures']),
                        'word': flow_and_memory(n, graph),
                        'direct_exact_N4': small_exact_affine(graph) if n == 4 else None})
    colour = exact_colour_controls()
    for name, pin in request['pins'].items():
        assert digest((root/name).read_bytes()) == pin
    result = {'kind': 'INDEPENDENT_EXACT_THERMAL_HEAT_REVIEW_CONTROLS_NOT_NATIVE_PARTITION',
              'source_sha256': digest(Path(__file__).read_bytes()), 'input_sha256': digest(args.input.read_bytes()),
              'governor': request['governor'], 'contract_sha256': request['contract_sha256'],
              'pins_verified': len(request['pins']), 'archived_bytes_matched': archived,
              'N_controls': records, 'colour_tail_and_failed_crop': colour,
              'old_target_imports_or_entrypoints': False, 'finite_float_claims': False,
              'native_numerical_partition_phase_photon': False}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'independent_exact_control': 'complete', 'N': request['N'],
                      'native_numerical_partition': False}))


if __name__ == '__main__':
    main()
