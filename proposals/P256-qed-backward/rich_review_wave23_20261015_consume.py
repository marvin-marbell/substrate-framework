"""Independent physical preparation, action, inclusive receiver and bounded propagation.
Consumes corrected frozen actual4 states; never calls an author's solve/evolve/observe CLI.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import scipy
from scipy import sparse, linalg
from scipy.special import eval_genlaguerre, gammaln
from rich_frame_field_join_20261015_model import build, law, fibre, observable, source, compress, physical_phase
from rich_common_frame_orbit_20261015_model import FormOperator
from rich_spatial_colour_root_20261015_model import so3
from rich_spatial_colour_em_20261015_fields import phase_values, phase_kernel, mechanical_kernels
from rich_frame_gauss_20261014_model import rotation

ROOT = Path(__file__).resolve().parent


def norm(x):
    return float(np.linalg.norm(x))


def expectation(operator, state):
    return float(np.vdot(state, operator @ state).real)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=serial) + '\n')


def all_m(model, name, m):
    body = model['sectors'][name]['spin']
    angular = model['rqspin']
    selected = np.flatnonzero(np.abs((body['m'][:, None] + angular['m'][None, :]).ravel() - m) < 1e-9)
    base = sparse.csr_matrix((np.ones(len(selected)), (selected, np.arange(len(selected)))), shape=(len(body['m']) * len(angular['m']), len(selected)))
    return sparse.kron(base, sparse.eye(model['radial'].count), format='csr')


def arnoldi_step(operator, state, duration, tolerance, width=64):
    """Independent full-Hessenberg Arnoldi; conservative integrated-defect bound.

    Hessenberg bandwidth forces e_m^T (T-cI)^k e_1=0 for k<m-1.
    Bound the exponential tail by x^(m-1)/(m-1)!/(1-x/m), x=t||T-cI||2.
    Unitary exact H evolution integrates this residual without norm amplification.
    Include measured Arnoldi relation roundoff and adapt by interval bisection.
    No small projected Hamiltonian or physical H is symmetrized.
    """
    if duration == 0:
        return state.copy(), dict(vector_error_bound=0., stages=[], norm=norm(state))
    length = norm(state)
    basis = np.empty((width + 1, len(state)), complex)
    basis[0] = state / length
    small = np.zeros((width + 1, width), complex)
    relation_square = 0.
    used = width
    for k in range(width):
        action = operator @ basis[k]
        remainder = action.copy()
        for unused in range(2):
            for j in range(k + 1):
                coefficient = np.vdot(basis[j], remainder)
                small[j, k] += coefficient
                remainder -= coefficient * basis[j]
        beta = norm(remainder)
        small[k + 1, k] = beta
        if beta:
            basis[k + 1] = remainder / beta
        reconstructed = sum((small[j, k] * basis[j] for j in range(k + 1)), np.zeros_like(state))
        reconstructed += remainder
        relation_square += norm(action - reconstructed) ** 2
        if beta < 1e-13:
            used = k + 1
            break
    projected = small[:used, :used]
    centre = float(np.trace(projected).real / used)
    radius = float(np.linalg.norm(projected - centre * np.eye(used), 2))
    x = abs(duration) * radius
    if x == 0:
        tail = 0.
    elif x < used:
        tail = math.exp((used - 1) * math.log(x) - math.lgamma(used)) / (1 - x / used)
    else:
        tail = float('inf')
    last = abs(small[used, used - 1])
    nonhermitian = norm(projected - projected.conj().T) / 2
    floating_relation = math.sqrt(relation_square)
    bound = length * abs(duration) * (last * tail + floating_relation * math.exp(abs(duration) * nonhermitian))
    # Allow final dense exponential/matrix-vector rounding, explicitly not a microscopic-model bound.
    rounding = 64 * np.finfo(float).eps * used * length * (1 + abs(duration) * radius)
    bound += rounding
    if bound > tolerance:
        del basis
        first, left = arnoldi_step(operator, state, duration / 2, tolerance / 2, width)
        second, right = arnoldi_step(operator, first, duration / 2, tolerance / 2, width)
        return second, dict(vector_error_bound=left['vector_error_bound'] + right['vector_error_bound'], stages=left['stages'] + right['stages'], norm=norm(second))
    coefficients = linalg.expm(-1j * duration * projected)[:, 0]
    propagated = length * (coefficients @ basis[:used])
    stage = dict(duration=duration, dimension=used, projected_radius=radius, tail_upper_bound=tail, last_Arnoldi_beta=float(last), relation_Frobenius_defect=floating_relation, projected_nonhermiticity=nonhermitian, dense_roundoff_allowance=rounding, vector_error_bound=bound)
    return propagated, dict(vector_error_bound=bound, stages=[stage], norm=norm(propagated))


def field_apply(model, projector, state, matrix):
    sector_dimension = projector.shape[0] // (280 * model['radial'].count)
    expanded = (projector @ state).reshape(sector_dimension, 280, model['radial'].count)
    u = model['rqU']
    physical = (u @ expanded.transpose(1, 0, 2).reshape(280, -1)).reshape(280, sector_dimension, -1).transpose(1, 0, 2)
    physical = physical.reshape(sector_dimension, 10, 28, model['radial'].count)
    applied = np.einsum('ef,irfn->iren', matrix, physical, optimize=True).reshape(sector_dimension, 280, -1)
    spin = (u.conj().T @ applied.transpose(1, 0, 2).reshape(280, -1)).reshape(280, sector_dimension, -1).transpose(1, 0, 2).ravel()
    compressed = projector.conj().T @ spin
    return compressed, norm(spin - projector @ compressed)


def charge_conjugation(model, name):
    target_name = 'antilepton' if name == 'lepton' else name
    a = model['sectors'][name]
    b = model['sectors'][target_name]
    matter = b['spin']['U'].conj().T @ model['C_matter'][name] @ a['spin']['U']
    field = model['rq'](sparse.kron(sparse.eye(10), model['field']['C']))
    return FormOperator(a, b, model['rotor'], model['radial'], [(matter, field, 0)])


def coordinate_and_radial(model):
    output = dict(radial_mass=model['radial'].mass, derivative=[], radial=[])
    for name, number in [('neutral', 16), ('lepton', 17), ('antilepton', 15)]:
        sector = model['sectors'][name]
        frame = sector['frame']
        rng = np.random.default_rng(230015 + number)
        vector = rng.normal(size=len(sector['spin']['j'])) + 1j * rng.normal(size=len(sector['spin']['j']))
        vector /= norm(vector)
        def wave(a, v, centre=False):
            coefficients = (frame['P'] @ v).reshape(len(frame['basis']), len(v))
            matrices = {j: rotation(j, a, .61, -.44).conj() for j in set(x[0] for x in frame['basis'])}
            values = np.array([np.sqrt(2*j+1)*matrices[j][int(round(m+j)), int(round(n+j))] * ((-1)**int(round(2*j)) if centre else 1) for j, m, n in frame['basis']])
            return values @ coefficients
        step = 1e-4
        derivative = -1j * (-wave(.29+2*step, vector)+8*wave(.29+step, vector)-8*wave(.29-step, vector)+wave(.29-2*step, vector))/(12*step)
        expected = wave(.29, sector['spin']['S'][2] @ vector)
        output['derivative'].append(dict(name=name, RIGHT_Gauss=frame['Gauss'], actual_coordinate_LEFT_error=norm(derivative-expected), field_centre_sewing_error=norm(wave(.29, vector, True)-(-1)**number*wave(.29, vector)), isometry=frame['isometry'], LEFT_equals_reduced=frame['left_equals_reduced']))
    radial = model['radial']
    for j in sorted(set(model['sectors']['lepton']['spin']['j'])):
        beta = radial.beta(j)
        nodes, weights = np.polynomial.legendre.leggauss(320)
        maximum = math.sqrt((2*radial.count+beta+70)/radial.scale)
        r = (nodes+1)*maximum/2
        weights *= maximum/2
        f = radial.values(j, r)
        ns = np.arange(radial.count)
        normal = np.sqrt(2)*radial.scale**((beta+2)/2)*np.exp((gammaln(ns+1)-gammaln(ns+beta+2))/2)
        lag = np.array([eval_genlaguerre(n, beta+1, radial.scale*r*r) for n in ns]).T
        previous = np.array([eval_genlaguerre(n-1, beta+2, radial.scale*r*r) if n else np.zeros_like(r) for n in ns]).T
        df = normal[None, :]*r[:, None]**(beta-1)*np.exp(-radial.scale*r[:, None]**2/2)*((beta-radial.scale*r[:, None]**2)*lag-2*radial.scale*r[:, None]**2*previous)
        direct = df.T @ ((weights*r**3)[:, None]*df)/(2*radial.mass) + j*(j+1)/radial.mass*f.T @ ((weights*r)[:, None]*f)
        output['radial'].append(dict(j=j, beta=beta, independently_integrated_r3_Gram_error=norm(f.T@((weights*r**3)[:, None]*f)-np.eye(radial.count)), independent_gradient_plus_frame_error=norm(direct-radial.operators(j)['kinetic']), positive_domain='r>0 Friedrichs; I(r)=Mrad*r^2/2; no origin epsilon'))
    return output


def basis_audit(model):
    rot = model['physical_rotor']
    real = so3(model['parameters'])
    bridge = np.kron(model['material']['U'], np.eye(28))
    rows = []
    for edge, charge in [(0, -1.), (1, 2/3)]:
        for derivative in [(), (2,), (2, 5)]:
            for axis in [None, 2]:
                actual = physical_phase(rot, model['geometry'], model['coefficients'], model['field'], edge, charge, derivative, axis)
                other = phase_kernel(real, model['geometry'], model['coefficients'], model['field'], edge, charge, derivative, axis)
                rows.append(dict(edge=edge, charge=charge, derivative=derivative, spin_axis=axis, independently_real_integrated_bridge_error=norm(actual-bridge.conj().T@other@bridge)))
    mechanics = []
    for derivative in [(), (2,), (2, 5)]:
        a = mechanical_kernels(rot, model['field'], model['connections'], 1., derivative=derivative)
        b = mechanical_kernels(real, model['field'], model['connections'], 1., derivative=derivative)
        mechanics.append(dict(derivative=derivative, errors=[norm(x-bridge.conj().T@y@bridge) for key in ['linear', 'quadratic'] for x, y in zip(a[key], b[key])]))
    basis = rot['basis']
    return dict(true_complex_Gram_error=norm(basis.conj().T@(rot['weights'][:, None]*basis)-np.eye(10)), wrong_unconjugated_Gram_discriminator=norm(basis.T@(rot['weights'][:, None]*basis)-np.eye(10)), physical_bridge_isometry=norm(bridge.conj().T@bridge-np.eye(280)), projected_phase_current_contact=rows, mechanical_source_current_contact=mechanics, correction='Actual bra conjugation before BOTH cuts, not H symmetrization; frozen real-R10 helper remains valid on real basis only.')


def delta_external_law(model, name, external):
    sector = model['sectors'][name]
    op = sector['op']
    p = model['parameters']
    rotor = model['physical_rotor']
    before = mechanical_kernels(rotor, model['field'], model['connections'], p['orientation_inertia'])
    after = mechanical_kernels(rotor, model['field'], model['connections'], p['orientation_inertia'], external=external)
    terms = []
    for key, matrices in [('linear', [op['Q'], op['density_delta']]), ('quadratic', [op['Q']@op['Q'], op['Q']@op['density_delta'], op['density_delta2']])]:
        for matter, a, b in zip(matrices, after[key], before[key]):
            terms.append((matter, model['rq'](a-b), 0))
    for edge in range(2):
        coefficients = model['coefficients']['values'][:, edge]
        for charge, matrices in op['hops'][edge]:
            values = phase_values(coefficients, charge, model['field'])
            difference = np.expm1(1j*charge*(coefficients@external))
            for axis in [None, 0, 1, 2]:
                factor = difference if axis is None else difference*model['geometry']['tangents'][:, edge, axis]
                angle = np.einsum('qr,q,qef,qs->resf', rotor['basis'].conj(), rotor['weights'], values*factor[:, None, None], rotor['basis'], optimize=True).reshape(280, 280)
                angle = model['rq'](angle)
                matter = (p['scalar_coupling']*matrices[0] if axis is None else 1j*p['vector_velocity']*matrices[axis+1])*model['geometry']['weights'][edge]
                terms.extend([(matter, angle, 1), (matter.conj().T, angle.conj().T, 1)])
    return FormOperator(sector, sector, model['rotor'], model['radial'], terms)


def source_derivative_audit(model, name, full):
    direction = np.array([0., 0., .8, 0., 0., -.35])
    step = 1e-3
    positive = delta_external_law(model, name, step*direction) @ full
    negative = delta_external_law(model, name, -step*direction) @ full
    j = .8*(law(model, name, derivative=(2,))@full)-.35*(law(model, name, derivative=(5,))@full)
    k = .64*(law(model, name, derivative=(2, 2))@full)+.1225*(law(model, name, derivative=(5, 5))@full)-.56*(law(model, name, derivative=(2, 5))@full)
    return dict(step=step, direction=direction, current_relative_error=norm((positive-negative)/(2*step)+j)/max(norm(j), 1e-300), contact_relative_error=norm((positive+negative)/step**2-k)/max(norm(k), 1e-300), current_wrong_sign_relative_error=norm((positive-negative)/(2*step)-j)/max(norm(j), 1e-300), contract="J=-dH/dAexternal, K=+d2H/dAexternal2; actual same-Gaussian mechanical and BOTH r-weighted hopping routes, external shift BEFORE cuts; finite difference includes mixed TE/TM contact.")


def physical_receiver_setup(model, threshold):
    field = model['field']
    one = np.empty((3, 3))
    for m in range(3):
        for n in range(3):
            polynomial = np.polynomial.polynomial.polymul(np.polynomial.hermite.herm2poly(np.eye(3)[m]), np.polynomial.hermite.herm2poly(np.eye(3)[n]))
            integral = sum(c*math.gamma((k+1)/2)/2 for k, c in enumerate(polynomial))
            one[m, n] = integral/math.sqrt(2**(m+n)*math.factorial(m)*math.factorial(n)*math.pi)
    effects = {}
    for left in range(2):
        for right in range(2):
            a = one if left else np.eye(3)-one
            b = one if right else np.eye(3)-one
            matrix = np.zeros((28, 28))
            for row, m in enumerate(field['basis']):
                for col, n in enumerate(field['basis']):
                    if all(m[axis] == n[axis] for axis in [0, 1, 3, 4]):
                        matrix[row, col] = a[m[2], n[2]]*b[m[5], n[5]]
            effects[left, right] = matrix
    radial = {}
    nodes, weights = np.polynomial.legendre.leggauss(160)
    r = (nodes+1)*threshold/2
    weights = weights*threshold/2*r**3
    for j in set(model['sectors']['lepton']['spin']['j']):
        f = model['radial'].values(j, r)
        low = f.T@(weights[:, None]*f)
        radial[float(j)] = [low, np.eye(model['radial'].count)-low]
    diagnostics = dict(field_effect_sum_error=norm(sum(effects.values())-np.eye(28)), minimum_field_effect_eigenvalue=min(float(np.linalg.eigvalsh(e)[0]) for e in effects.values()), minimum_radial_effect_eigenvalue=min(float(np.linalg.eigvalsh(e)[0]) for pair in radial.values() for e in pair), interpretation='Direct compression of commuting underlying site-charge, positive matrix-amplitude interval, TE_z sign and TM_z sign PVM; not spectral PVM of PQP and not product of separately compressed sign effects. No postselection or renormalization.', matrix_amplitude_threshold=threshold)
    te = effects[1, 0]+effects[1, 1]
    tm = effects[0, 1]+effects[1, 1]
    diagnostics['wrong_product_of_compressed_effects_defect'] = norm(effects[1, 1]-te@tm)
    return effects, radial, diagnostics


def receiver(model, name, projector, state, effects, radial):
    sector = model['sectors'][name]
    spin = sector['spin']
    x = (projector@state).reshape(len(spin['j']), 280, model['radial'].count)
    x = (model['rqU']@x.transpose(1, 0, 2).reshape(280, -1)).reshape(280, len(spin['j']), -1).transpose(1, 0, 2).reshape(len(spin['j']), 10, 28, model['radial'].count)
    charges = np.array([label[0] for label in spin['labels']])
    labels = sorted(set(charges.tolist()))
    probabilities = np.zeros((len(labels), 2, 2, 2))
    for iq, charge in enumerate(labels):
        for side in range(2):
            density = np.zeros((28, 28), complex)
            for j in set(spin['j']):
                selected = np.flatnonzero((charges == charge)&(spin['j'] == j))
                if not len(selected):
                    continue
                block = x[selected]
                moved = np.einsum('iren,mn->irem', block, radial[float(j)][side], optimize=True)
                density += np.einsum('iren,irfn->ef', block.conj(), moved, optimize=True)
            for (left, right), effect in effects.items():
                probabilities[iq, side, left, right] = float(np.sum(density*effect).real)
    return dict(site_charge3_labels=labels, probabilities=probabilities, total=float(probabilities.sum()), minimum=float(probabilities.min()), norm_squared=norm(state)**2)


def angular_response(model, name, pole):
    initial = fibre(model, name, .5, .5)@pole
    targets = {(j, float(m)): fibre(model, name, j, float(m)) for j in [.5, 1.5] for m in np.arange(-j, j+1)}
    rows = []
    for axis in range(6):
        image = law(model, name, derivative=(axis,))@initial
        reconstructed = np.zeros_like(image)
        weights = []
        for (j, m), projector in targets.items():
            projected = projector.conj().T@image
            reconstructed += projector@projected
            weights.append(dict(J=j, M=m, weight=norm(projected)**2, dimension=projector.shape[1]))
        rows.append(dict(axis=axis, full_image_norm_squared=norm(image)**2, all_allowed_J_M_leak=norm(image-reconstructed), signed_virtual_image_weights=weights, contact=expectation(law(model, name, derivative=(axis, axis)), initial)))
    return rows


def main():
    inp, output = map(Path, sys.argv[1:3])
    p = json.loads(inp.read_text())
    started = time.time()
    statepath = ROOT/p['state_file']
    if hashlib.sha256(statepath.read_bytes()).hexdigest() != p['state_sha256']:
        raise ValueError('Pinned corrected actual4 state changed')
    if hashlib.sha256((ROOT/p['result_file']).read_bytes()).hexdigest() != p['result_sha256']:
        raise ValueError('Pinned owning metadata changed')
    states = np.load(statepath)
    metadata = json.loads((ROOT/p['result_file']).read_text())
    model = build(p)
    result = dict(kind='INDEPENDENT_NEW_COHERENT_TWO_SITE_TWO_FAMILY_FIELD_ONLY_PHYSICAL_CONSUMER', parameters=p, versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__), native_dimensions={name:len(s['spin']['j']) for name, s in model['sectors'].items()}, geometry=coordinate_and_radial(model), basis=basis_audit(model), source=[], events={})
    v0 = fibre(model, 'neutral', 0., 0.)
    ref = states['reference']
    fullref = v0@ref
    h0, h0audit = compress(law(model, 'neutral'), v0)
    result['reference'] = dict(saved_corrected_reference_dimension=len(ref), energy=metadata['reference']['energies'][0], own_residual=norm(h0@ref-metadata['reference']['energies'][0]*ref), factorization=h0audit, origin='Fresh corrected author actual4, independently consumed; not an old EM/Frame state and not a new full eigensolve by reviewer.')
    seeds = {}
    cache = dict(reference=ref)
    for name in ['lepton', 'antilepton']:
        pieces = {}
        spin0 = model['sectors']['neutral']['spin']
        for site in [0, 1]:
            for m in [-.5, .5]:
                operator, gauss = source(model, name, .5, m, site)
                raw = operator@fullref
                metric = next(a for label, a in model['native_source_metric_full'][name].items() if label.split(':')[0] == str(site) and float(label.split(':')[1]) == .5 and float(label.split(':')[2]) == m)
                gram = spin0['U'].conj().T@metric@spin0['U']
                rawweight = expectation(observable(model, 'neutral', matter=gram), fullref)
                q = observable(model, name, matter=model['sectors'][name]['op']['Q'])@raw
                result['source'].append(dict(name=name, site=site, M=m, raw_primitive_weight=rawweight, retained_weight=norm(raw)**2, projection_loss=rawweight-norm(raw)**2, physical_charge_eigen_error=norm(q-(-1 if name=='lepton' else 1)*raw), Gauss_and_wrong_bare_source=gauss))
                if m == .5:
                    pieces[site] = raw
        mixed = pieces[0]+complex(*p['site1_coefficient'])*pieces[1]
        weight = norm(mixed)**2
        seeds[name] = mixed/math.sqrt(weight)
        result.setdefault('own_preparation', {})[name] = dict(coherent_retained_weight=weight, unnormalized_definition='(B_site0 + (.31+.27i) B_site1) |corrected neutral J0>; spin M=+1/2 is actual Wigner/RIGHT-Gauss field LEFT source, not an assigned CAR physical spin', norm=norm(seeds[name]))
        result.setdefault('angular_current', {})[name] = angular_response(model, name, states[name+'_vectors'][:, 0])
    c = charge_conjugation(model, 'lepton')
    cn = charge_conjugation(model, 'neutral')
    cz = cn@fullref
    cx = c@seeds['lepton']
    phase = np.vdot(seeds['antilepton'], cx)
    result['C'] = dict(neutral_C_overlap=[float(np.vdot(fullref, cz).real), float(np.vdot(fullref, cz).imag)], independently_created_anti_vs_C_positive_up_to_phase=norm(cx-phase*seeds['antilepton']), phase=[float(phase.real), float(phase.imag)], H_intertwiner=norm(law(model, 'antilepton')@cx-c@(law(model, 'lepton')@seeds['lepton'])), current_odd=[], contact_even=[])
    for axis in [2, 5]:
        result['C']['current_odd'].append(norm(law(model, 'antilepton', derivative=(axis,))@cx+c@(law(model, 'lepton', derivative=(axis,))@seeds['lepton'])))
        result['C']['contact_even'].append(norm(law(model, 'antilepton', derivative=(axis, axis))@cx-c@(law(model, 'lepton', derivative=(axis, axis))@seeds['lepton'])))
    result['source_current_contact_variation'] = source_derivative_audit(model, 'lepton', seeds['lepton'])
    effects, radial, receiver_diagnostics = physical_receiver_setup(model, p['matrix_amplitude_receiver_threshold'])
    result['inclusive_receiver'] = receiver_diagnostics
    save(output, result)
    np.savez_compressed(output.with_name(output.stem+'_source_checkpoint.npz'), reference=ref, lepton=seeds['lepton'], antilepton=seeds['antilepton'])
    print('source_geometry_and_current_checkpoint', flush=True)
    generator = .8*model['field']['Q'][2]-.35*model['field']['Q'][5]
    direction = np.array([0., 0., .8, 0., 0., -.35])
    for name in ['lepton', 'antilepton']:
        vm = all_m(model, name, .5)
        seed = vm.conj().T@seeds[name]
        sign = 1 if name == 'lepton' else -1
        unitary = linalg.expm(-1j*sign*p['own_field_kick']*generator)
        kicked, kickleak = field_apply(model, vm, seed, unitary)
        continuous = phase_values(direction[None, :], -sign*p['own_field_kick'], model['field'])[0]
        truncated_continuous, unused = field_apply(model, vm, seed, continuous)
        h, ha = compress(law(model, name), vm)
        off, oa = compress(law(model, name, coupled=False), vm)
        sector = model['sectors'][name]
        operators = {}
        for key, matrix in [('LEFT_z', sector['op']['S'][2]), ('colour_loop_B', sector['op']['B']), ('colour_electric', sector['op']['electric']), ('rho0', sector['op']['rho0'])]:
            operators[key] = compress(observable(model, name, matter=matrix), vm)[0]
        for key, value in [('material_z', model['rqobs']['material'][2]), ('EM_z', model['rqobs']['field'][2]), ('HE', model['rqobs']['HE']), ('field_boundary', model['rqobs']['boundary'])]:
            operators[key] = compress(observable(model, name, rq=value), vm)[0]
        radial_operators = {float(j):model['radial'].operators(float(j))['r2'] for j in set(sector['spin']['j'])}
        operators['matrix_r2'] = compress(FormOperator(sector, sector, model['rotor'], model['radial'], [], radial_operators), vm)[0]
        for axis in [2, 5]:
            operators['current_'+str(axis)] = compress(law(model, name, derivative=(axis,)), vm)[0]
            operators['contact_'+str(axis)] = compress(law(model, name, derivative=(axis, axis)), vm)[0]
        def measure(state):
            values = {key:expectation(a, state) for key, a in operators.items()}
            values.update(norm=norm(state), energy=expectation(h, state), total_Jz=values['LEFT_z']+values['material_z']+values['EM_z'])
            return values
        before, after = measure(seed), measure(kicked)
        largest = max(sector['spin']['j'])+max(model['rqspin']['j'])
        js = np.arange(.5, largest+1e-8, 1.)
        spaces = [(float(j), fibre(model, name, float(j), .5)) for j in js]
        initial_j = [dict(J=j, dimension=v.shape[1], weight=norm(v.conj().T@(vm@kicked))**2) for j, v in spaces]
        event = dict(dimension=vm.shape[1], all_compatible_J_dimension=sum(v.shape[1] for j, v in spaces), angular_spaces=initial_j, factorization=ha, off_factorization=oa, H_full_outside_event_fibre=norm(law(model, name)@(vm@seed)-vm@(h@seed)), kick=dict(signed_strength=sign*p['own_field_kick'], finite_field_unitary_error=norm(unitary.conj().T@unitary-np.eye(28)), event_fibre_leakage=kickleak, omitted_continuous_preparation_norm_squared=norm(seed)**2-norm(truncated_continuous)**2, finite_cut_unitary_vs_compressed_continuum_preparation=norm(kicked-truncated_continuous)), t0_before=before, t0_after=after, t0_matter_differences={key:after[key]-before[key] for key in ['LEFT_z', 'colour_loop_B', 'rho0', 'matrix_r2']}, trajectories={})
        cache[name+'_seed'] = seed
        cache[name+'_kicked'] = kicked
        for tag, operator, initial in [('full_no_kick', h, seed), ('full_kicked', h, kicked), ('off_no_kick', off, seed), ('off_kicked', off, kicked)]:
            propagated, ledger = arnoldi_step(operator, initial, p['own_duration'], p['propagation_error_target'])
            event['trajectories'][tag] = dict(time=p['own_duration'], observables=measure(propagated), propagation=ledger, receiver=receiver(model, name, vm, propagated, effects, radial))
            cache[name+'_'+tag] = propagated
            result['events'][name] = event
            save(output, result)
            np.savez_compressed(output.with_suffix('.npz'), **cache)
            print(name, tag, 'bound', ledger['vector_error_bound'], 'norm', norm(propagated), flush=True)
        event['full_intervention'] = {key:event['trajectories']['full_kicked']['observables'][key]-event['trajectories']['full_no_kick']['observables'][key] for key in operators}
        event['proper_full_rotor_and_hop_off_intervention'] = {key:event['trajectories']['off_kicked']['observables'][key]-event['trajectories']['off_no_kick']['observables'][key] for key in operators}
        fulljoint = event['trajectories']['full_kicked']['receiver']['probabilities']-event['trajectories']['full_no_kick']['receiver']['probabilities']
        offjoint = event['trajectories']['off_kicked']['receiver']['probabilities']-event['trajectories']['off_no_kick']['receiver']['probabilities']
        event['full_joint_receiver_TV_intervention'] = float(np.abs(fulljoint).sum()/2)
        event['off_joint_receiver_TV_intervention'] = float(np.abs(offjoint).sum()/2)
        # Off field joint statistics can still differ; only the matter/radial marginal must not react.
        event['off_matter_radial_marginal_TV'] = float(np.abs(offjoint.sum(axis=(2, 3))).sum()/2)
        result['events'][name] = event
        save(output, result)
        del h, off, operators, spaces
    result['elapsed_seconds'] = time.time()-started
    result['scope'] = 'Actual finite occupied neutral/lepton/antilepton physical source and reaction. BODY2108 is available but no baryon quantum-EM state or event was computed. Conditional reference, finite Gaussian TE/TM shell2=28 and radial6, no COM/continuum/gauge-complete photons/Nature necessity.'
    save(output, result)
    print(json.dumps(dict(natural_completion=True, elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
