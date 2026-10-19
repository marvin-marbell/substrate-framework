"""Full 108-CAR MPS of the fixed spatial law, not its old Gaussian-span H.
The C3 charge is physical z translation, not an extra COM degree of freedom.
"""
from __future__ import annotations
from itertools import product
from typing import Any, cast
import numpy as np
from scipy.linalg import eigh
from tenpy.linalg import np_conserved as npc
from tenpy.models.lattice import Lattice, IrregularLattice
from tenpy.models.model import MPOModel
from tenpy.networks.site import Site
from tenpy.networks.mpo import MPO, MPOEnvironment
from tenpy.networks.mps import MPS
from rich_spatial_quantum_sea_20261015_model import geometry
from rich_spatial_translation_20261015_model import shell_fields


def single_particle(law: dict[str, Any]) -> dict[str, Any]:
    geo: Any = geometry(law)
    cells = geo['cells']
    orbitals = []
    energies = []
    labels = []
    blocks = []
    for b, (kx, ky) in enumerate(product(range(3), repeat=2)):
        w = np.zeros((108, 12), complex)
        for site, (x, y, z) in enumerate(cells):
            w[4*site:4*site+4, 4*z:4*z+4] = np.eye(4)*np.exp(2j*np.pi*(kx*x+ky*y)/3)/3
        local = []
        for kz in range(3):
            zfourier = np.kron(np.exp(2j*np.pi*kz*np.arange(3)/3)[:, None]/np.sqrt(3), np.eye(4))
            h = zfourier.conj().T @ w.conj().T @ geo['reference'] @ w @ zfourier
            values, vectors = eigh(h)
            if np.count_nonzero(values < 0) != 2:
                raise ValueError('The selected fixed-law sea must fill two bands per actual momentum')
            local.append(zfourier @ vectors)
            energies.extend(values.tolist())
            labels.extend([dict(block=b, kx=kx, ky=ky, kz=kz, band=a, occupied=a < 2) for a in range(4)])
        u = w @ np.column_stack(local)
        orbitals.append(u)
        components = np.concatenate([(3*law['wilson']*geo['components'][0])[None], geo['components']])
        blocks.append(np.array([u.conj().T @ h @ u for h in components]))
    u = np.column_stack(orbitals)
    return dict(geometry=geo, orbitals=u, energies=np.array(energies), labels=labels,
                components=np.array(blocks), occupied=np.array([x['occupied'] for x in labels]),
                momentum=np.array([x['kz'] for x in labels]),
                offset=float(np.sum(np.array(energies)[np.array([x['occupied'] for x in labels])])),
                orbital_unitarity=float(np.linalg.norm(u.conj().T @ u-np.eye(108))))


def field_representation(law: dict[str, Any]) -> dict[str, Any]:
    field: Any = shell_fields(law)
    angular, rotation = eigh(field['em_lz'])
    integers = np.rint(angular).astype(int)
    if np.max(abs(angular-integers)) > 1e-12:
        raise ValueError('Complete shell lost its exact physical field rotation')
    raw = np.kron(np.eye(field['nphi']), rotation)
    momenta = np.tile(integers % 3, field['nphi'])
    order = np.argsort(momenta, kind='stable')
    v = raw[:, order]
    momenta = momenta[order]
    transform = lambda a: v.conj().T @ (a.toarray() if hasattr(a, 'toarray') else a) @ v
    coefficients = [transform(x) for x in field['coupling']]
    derivatives = [[transform(x) for x in row] for row in field['derivatives']]
    contacts = [[[transform(x) for x in row] for row in matrix] for matrix in field['contacts']]
    phi_h = field['boson'].toarray().reshape(field['nphi'], field['shell_dimension'], field['nphi'], field['shell_dimension'])
    scalar_h = (np.einsum('peqe->pq', phi_h)-np.trace(field['em_kinetic'])*np.eye(field['nphi']))/field['shell_dimension']
    _, scalar_vectors = eigh(scalar_h)
    initial = np.kron(scalar_vectors[:, 0], np.eye(field['shell_dimension'])[:, 0])
    return dict(raw=field, transform=v, momentum=momenta, coupling=coefficients,
                derivatives=derivatives, contacts=contacts, boson=transform(field['boson']),
                initial=v.conj().T @ initial,
                coordinates=[transform(x) for x in field['coordinates']],
                phi=transform(field['phi_operator']), phi2=transform(field['phi2_operator']),
                field_translation=transform(field['field_translation']))


class Law:
    def __init__(self, law: dict[str, Any]):
        self.parameters = dict(law)
        self.one = single_particle(law)
        self.field = field_representation(law)
        self.charge_info = npc.ChargeInfo([1, 3], ['N', 'C3z'])
        momenta = self.field['momentum']
        leg = npc.LegCharge.from_qflat(self.charge_info, np.column_stack([np.zeros(len(momenta), int), momenta]))
        self.field_site = Site(leg, [str(i) for i in range(len(momenta))], sort_charge=False)
        self.sites = [self.field_site]
        for kz in self.one['momentum']:
            leg = npc.LegCharge.from_qflat(self.charge_info, [[0, 0], [1, int(kz)]])
            site = Site(leg, ['empty', 'full'], sort_charge=False,
                        N=np.diag([0., 1.]), JW=np.diag([1., -1.]))
            site.add_op('C', np.array([[0., 1.], [0., 0.]]), need_JW=True, hc=False)
            site.add_op('Cd', np.array([[0., 0.], [1., 0.]]), need_JW=True, hc='C')
            self.sites.append(site)
        regular = Lattice([9], self.sites[1:13], bc='open', bc_MPS='finite')
        self.lattice = IrregularLattice(regular, add=([[0, 12]], [-.5]), add_unit_cell=[self.field_site])
        self.sites = self.lattice.mps_sites()
        self.epochs: list[dict[str, Any]] = []
        self._operator_count = 0
        self.model = self.make_operator(self.field['coupling'], 'coupled', 0, True)
        self.emission = self.emission_operators()

    def charge_matrix(self, matrix: np.ndarray, charge: int, name: str) -> tuple[str, float]:
        r = self.field['momentum']
        allowed = (r[:, None]-r[None, :]) % 3 == charge % 3
        forbidden = float(np.linalg.norm(matrix[~allowed]))
        if forbidden > 2e-11*max(1., np.linalg.norm(matrix)):
            raise ValueError(f'Physical C3 charge failure {name}: {forbidden}')
        selected = np.where(allowed, matrix, 0.)
        self.field_site.add_op(name, npc.Array.from_ndarray(selected,
            [self.field_site.leg, self.field_site.leg.conj()], qtotal=[0, charge % 3], labels=['p', 'p*']), hc=False)
        self.field_site.add_op(name+'D', self.field_site.get_op(name).conj().transpose(), hc=name)
        return name, forbidden

    def make_operator(self, coefficients: list[np.ndarray], label: str, total_charge: int,
                      bare: bool = False) -> MPOModel:
        # Retain EVERY primitive coefficient, including operators outside H's span.
        # C3 splitting is exact; no field-operator SVD or small-channel threshold.
        r = self.field['momentum']
        fields = []
        bodies = []
        channels = []
        for c, coefficient in enumerate(coefficients):
            for charge in range(3):
                matrix = np.where((r[:, None]-r[None, :]) % 3 == charge, coefficient, 0.)
                if not np.any(matrix):
                    continue
                body = self.one['components'][:, c].copy()
                for block in range(9):
                    k = self.one['momentum'][12*block:12*block+12]
                    body[block] = np.where((k[:, None]-k[None, :]+charge) % 3 == total_charge, body[block], 0.)
                if not np.any(body):
                    continue
                self._operator_count += 1
                name, _ = self.charge_matrix(matrix, charge, f'F{self._operator_count}')
                fields.append(matrix)
                bodies.append(body)
                channels.append(dict(name=name, charge=charge, primitive_component=c))
        original = np.einsum('bcij,cuv->bijuv', self.one['components'], np.asarray(coefficients), optimize=True)
        reconstructed = np.einsum('cbij,cuv->bijuv', np.asarray(bodies), np.asarray(fields), optimize=True)
        coefficient_error = float(np.linalg.norm((original-reconstructed).ravel()))
        coefficient_norm = float(np.linalg.norm(original.ravel()))
        if coefficient_error > 2e-11*max(1., coefficient_norm):
            raise ValueError(f'Primitive/C3 coefficient reconstruction failure {label}: {coefficient_error}')
        del original, reconstructed
        field_bare = None
        if bare:
            self._operator_count += 1
            field_bare, _ = self.charge_matrix(self.field['boson']-self.one['offset']*np.eye(len(r)), 0, f'F{self._operator_count}')
        mpo = self.literal_one_body_mpo(channels, np.asarray(bodies), total_charge, field_bare)
        action = self.action_discriminator(mpo, coefficients, bare)
        self.epochs.append(dict(label=label, actual_total_C3_charge=total_charge,
            primitive_field_channels=channels, coefficient_reconstruction_error=coefficient_error,
            coefficient_norm=coefficient_norm, MPO_bonds=list(mpo.chi), direct_CAR_action=action,
            representation='All primitive field/C3 components; literal bounded Jordan-Wigner one-body automaton. No operator SVD and no current/matter/state-channel truncation.'))
        return MPOModel(self.lattice, mpo)

    def literal_one_body_mpo(self, channels: list[dict[str, Any]], bodies: np.ndarray,
                             total_charge: int, field_bare: str | None) -> MPO:
        # Carry only the future fermionic endpoint after multiplying its coefficient.
        def quantum(key):
            kind, index = key
            if kind == 'start':
                return [0, 0]
            if kind == 'done':
                return [0, total_charge]
            if kind == 'field':
                return [0, channels[index]['charge']]
            momentum = int(self.one['momentum'][index])
            return [1, (total_charge+momentum) % 3] if kind == 'ann' else [-1, (total_charge-momentum) % 3]
        def ordered(keys):
            return sorted(keys, key=lambda key: (*quantum(key), key))
        alive = [('field', j) for j in range(len(channels))]
        done = ('done', 0)
        keys = ordered(alive+[done])
        bonds = [[('start', 0)], keys]
        grid: list[list[Any]] = [[None for _ in keys]]
        for j, key in enumerate(keys):
            if key[0] == 'field':
                grid[0][j] = channels[key[1]]['name']
            elif field_bare is not None:
                grid[0][j] = field_bare
        grids = [grid]
        for i in range(108):
            block, local = divmod(i, 12)
            pending = {key for key in keys if key[0] in ['ann', 'cre'] and key[1] > i}
            for j in range(local+1, 12):
                if np.any(bodies[:, block, local, j]):
                    pending.add(('ann', 12*block+j))
                if np.any(bodies[:, block, j, local]):
                    pending.add(('cre', 12*block+j))
            right_keys: list[tuple[str, int]] = [done] if i == 107 else ordered(alive+[done]+list(pending))
            left_index = {key: j for j, key in enumerate(keys)}
            right_index = {key: j for j, key in enumerate(right_keys)}
            grid = [[None for _ in right_keys] for _ in keys]
            for key in keys:
                row = left_index[key]
                if key == done:
                    grid[row][right_index[done]] = 'Id'
                elif key[0] in ['ann', 'cre']:
                    if key[1] == i:
                        grid[row][right_index[done]] = 'C' if key[0] == 'ann' else 'Cd'
                    else:
                        grid[row][right_index[key]] = 'JW'
                else:
                    channel = key[1]
                    if key in right_index:
                        grid[row][right_index[key]] = 'Id'
                    strength = bodies[channel, block, local, local]
                    if strength != 0:
                        grid[row][right_index[done]] = [('N', strength)]
                    for j in range(local+1, 12):
                        forward = bodies[channel, block, local, j]
                        reverse = bodies[channel, block, j, local]
                        if forward != 0:
                            grid[row][right_index[('ann', 12*block+j)]] = [('Cd', forward)]
                        if reverse != 0:
                            grid[row][right_index[('cre', 12*block+j)]] = [('C', reverse)]
            grids.append(grid)
            bonds.append(right_keys)
            keys = right_keys
        legs = [npc.LegCharge.from_qflat(self.charge_info, [quantum(key) for key in bond]).bunch()[1] for bond in bonds]
        identity_right = [None]+[bond.index(done) for bond in bonds[1:]]
        return MPO.from_grids(self.sites, grids, 'finite', IdL=[0]+[None]*len(self.sites),
                              IdR=identity_right, legs=legs, max_range=len(self.sites),
                              mps_unit_cell_width=self.lattice.mps_unit_cell_width)

    def action_discriminator(self, mpo: MPO, coefficients: list[np.ndarray], bare: bool) -> dict[str, Any]:
        # Complex amplitudes, not just norms, catch actual CAR/JW signs.
        tensors = [mpo.get_W(i).transpose(['wL', 'wR', 'p', 'p*']).to_ndarray() for i in range(len(self.sites))]
        field = np.exp(.17j*np.arange(len(self.field['momentum'])))*(1+.03*np.arange(len(self.field['momentum'])))
        field /= np.linalg.norm(field)
        first = np.einsum('vpq,q->vp', tensors[0][0], field, optimize=True)
        base = sum(1 << int(i) for i in np.flatnonzero(self.one['occupied']))
        empty = int(np.flatnonzero(~self.one['occupied'])[0])
        filled = int(np.flatnonzero(self.one['occupied'])[0])
        errors = []
        def collapse(bra_mask, ket_mask):
            env = np.ones(1, complex)
            for i in reversed(range(108)):
                env = tensors[i+1][:, :, (bra_mask >> i) & 1, (ket_mask >> i) & 1] @ env
            return first.T @ env
        for number, mask in [(54, base), (55, base | (1 << empty)), (53, base ^ (1 << filled))]:
            diagonal = np.zeros((len(field), len(field)), complex)
            if bare:
                diagonal += self.field['boson']-self.one['offset']*np.eye(len(field))
            candidates = []
            for block in range(9):
                ranked = []
                for b in range(12):
                    gb = 12*block+b
                    if not mask & (1 << gb):
                        continue
                    diagonal += sum(self.one['components'][block, c, b, b]*f for c, f in enumerate(coefficients))
                    for a in range(12):
                        ga = 12*block+a
                        if mask & (1 << ga):
                            continue
                        matrix = sum(self.one['components'][block, c, a, b]*f for c, f in enumerate(coefficients))
                        ranked.append((float(np.linalg.norm(matrix @ field)), ga, gb, matrix))
                candidates.extend(sorted(ranked, key=lambda row: row[0], reverse=True)[:4])
            errors.append(float(np.linalg.norm(collapse(mask, mask)-diagonal @ field)))
            for _, a, b, matrix in candidates:
                intermediate = mask ^ (1 << b)
                parity = (-1)**((mask & ((1 << b)-1)).bit_count()+(intermediate & ((1 << a)-1)).bit_count())
                expected = parity*(matrix @ field)
                actual = collapse(intermediate | (1 << a), mask)
                errors.append(float(np.linalg.norm(actual-expected)))
        maximum = max(errors)
        if maximum > 2e-10:
            raise ValueError(f'Literal full-field/CAR/JW action mismatch: {maximum}')
        return dict(orders=[54, 55, 53], tested_complex_vector_actions=len(errors), maximum_absolute_error=maximum,
                    meaning='Full coefficient reconstruction and literal field-vector/Fock-transfer amplitudes, including JW signs and the owning signed C3 phase; not an old producer replay.')

    def emission_operators(self) -> list[dict[str, Any]]:
        # Q J is formed BEFORE the shell cut. Q only connects retained shell2 to
        # shell<=3, so this single larger intermediate shell is exact for the product.
        extended_law = dict(self.parameters, shell_cutoff=self.parameters['shell_cutoff']+1)
        extended: Any = shell_fields(extended_law)
        source_states = self.field['raw']['shell_states']
        target_states = extended['shell_states']
        indices = np.array([p*len(target_states)+target_states.index(s) for p in range(extended['nphi']) for s in source_states])
        v = self.field['transform']
        result = []
        k = self.one['momentum']
        r = self.field['momentum']
        for pol in range(2):
            for sign in [1, -1]:
                current = [-(extended['derivatives'][2*pol][c]+1j*sign*extended['derivatives'][2*pol+1][c])/np.sqrt(2) for c in range(16)]
                # Determine the physical finite-C3 phase, not a guessed momentum=current identity.
                small = [v.conj().T @ x.toarray()[np.ix_(indices, indices)] @ v for x in current]
                weights = np.zeros(3)
                for block in range(9):
                    for i in range(12):
                        for j in range(12):
                            m: Any = sum(self.one['components'][block, c, i, j]*f for c, f in enumerate(small))
                            delta = (r[:, None]-r[None, :]+k[12*block+i]-k[12*block+j]) % 3
                            for q in range(3):
                                weights[q] += np.sum(abs(m[delta == q])**2)
                charge = int(np.argmax(weights))
                if charge == 0 or np.sum(weights)-weights[charge] > 1e-18*max(1., weights[charge]):
                    raise ValueError(f'Signed current is not one nonzero actual C3 character: {weights}')
                qplus = (extended['coordinates'][2*pol]+1j*extended['coordinates'][2*pol+1])/np.sqrt(2)
                qsmall = v.conj().T @ qplus.toarray()[np.ix_(indices, indices)] @ v
                qweight = [np.linalg.norm(qsmall[(r[:, None]-r[None, :]) % 3 == z]) for z in range(3)]
                qcharge = int(np.argmax(qweight))
                counter = qplus if qcharge == (-charge) % 3 else qplus.conj().T
                products_qj = [v.conj().T @ (counter @ x).toarray()[np.ix_(indices, indices)] @ v for x in current]
                products_jq = [v.conj().T @ (x @ counter).toarray()[np.ix_(indices, indices)] @ v for x in current]
                qj = self.make_operator(products_qj, f'emission_QJ_pol{pol}_sign{sign}', 0)
                jq = self.make_operator(products_jq, f'emission_JQ_pol{pol}_sign{sign}', 0)
                current_model = self.make_operator(small, f'current_pol{pol}_sign{sign}', charge)
                result.append(dict(polarization=pol, profile_sign=sign, current_C3_charge=charge,
                    current_character=np.exp(-2j*np.pi*charge/3), current=current_model,
                    QJ=qj, JQ=jq, character_weights=weights))
        return result

    def packet(self, spin: int = 0) -> np.ndarray:
        x = self.one['geometry']['cells'].astype(float)
        dx = (x-np.array(self.parameters['packet_center'])+1.5) % 3-1.5
        wave = np.exp(-np.sum(dx*dx, axis=1)/(2*self.parameters['packet_width']**2)
                      +1j*x @ np.array(self.parameters['packet_momentum']))
        raw = np.zeros((27, 4), complex)
        raw[:, spin] = wave
        raw[:, spin+2] = .3j*wave
        raw = raw.ravel()/np.linalg.norm(raw)
        return self.one['orbitals'].conj().T @ raw

    def canonical_packet_mpo(self, coefficient: np.ndarray, create: bool, momentum: int) -> MPO:
        # Explicit charges resolve the inactive field/zero-coefficient prefix;
        # the generic wavepacket constructor cannot infer this dangling branch.
        sign = 1 if create else -1
        target = [sign, (sign*momentum) % 3]
        before_after = npc.LegCharge.from_qflat(self.charge_info, [[0, 0], target])
        legs = [npc.LegCharge.from_qflat(self.charge_info, [[0, 0]])]
        legs += [before_after]*(len(self.sites)-1)
        legs += [npc.LegCharge.from_qflat(self.charge_info, [target])]
        grids = []
        for site in range(len(self.sites)):
            local = None if site == 0 or coefficient[site-1] == 0 else [('Cd' if create else 'C', coefficient[site-1])]
            grid = [['JW', local], [None, 'Id']]
            if site == 0:
                grid = grid[:1]
            if site == len(self.sites)-1:
                grid = [[row[1]] for row in grid]
            grids.append(grid)
        return MPO.from_grids(self.sites, grids, 'finite', IdL=[0]+[None]*len(self.sites),
            IdR=[None]*len(self.sites)+[0], legs=legs, max_range=len(self.sites),
            mps_unit_cell_width=self.lattice.mps_unit_cell_width)

    def prepared(self, number: int, polarization: int | None = None) -> dict[int, MPS]:
        field_vector = self.field['initial'].copy()
        if polarization is not None:
            from scipy.linalg import expm
            field_vector = expm(1j*self.parameters['kick_strength']*self.field['coordinates'][2*polarization]) @ field_vector
        packet = self.packet()
        result: dict[int, MPS] = {}
        for field_charge in range(3):
            part = np.where(self.field['momentum'] == field_charge, field_vector, 0.)
            if np.linalg.norm(part) < 1e-14:
                continue
            sea = MPS.from_product_state(self.sites, [part]+['full' if b else 'empty' for b in self.one['occupied']],
                bc='finite', dtype=cast(Any, complex), unit_cell_width=self.lattice.mps_unit_cell_width)
            sea.canonical_form_finite(renormalize=False, cutoff=1e-14)
            if number == 54:
                pieces = [(field_charge, sea)]
            else:
                pieces = []
                for momentum in range(3):
                    coefficient = np.where(self.one['momentum'] == momentum, packet if number == 55 else packet.conj(), 0.)
                    mpo = self.canonical_packet_mpo(coefficient, number == 55, momentum)
                    state = sea.copy()
                    mpo.apply_naively(state)
                    state.canonical_form_finite(renormalize=False, cutoff=1e-14)
                    total = (field_charge+(momentum if number == 55 else -momentum)) % 3
                    pieces.append((total, state))
            for charge, state in pieces:
                result[charge] = state if charge not in result else result[charge].add(state, 1., 1., cutoff=1e-14)
        return result


def exact_image(operator: MPOModel, psi: MPS) -> MPS:
    image = psi.copy()
    operator.H_MPO.apply_naively(image)
    image.canonical_form_finite(renormalize=False, cutoff=1e-13)
    return image


def prepare_current_space(law: Law, psi: MPS, minimum_bond: int, duration: float) -> tuple[MPS, dict[str, Any]]:
    # Occupy the new directions by an actual short SAME-H evolution, rather than
    # padding the canonical Schmidt spectrum with zeros (undefined inverse form).
    vectors = [psi.copy(), exact_image(law.model, psi)]+[exact_image(op['QJ'], psi) for op in law.emission]
    gram = np.array([[a.overlap(b) for b in vectors] for a in vectors])
    values, basis = eigh(gram)
    threshold = 64*np.finfo(float).eps*max(1., float(values[-1]))
    retained = values > threshold
    whitening = basis[:, retained]/np.sqrt(values[retained])
    h = np.zeros(gram.shape, complex)
    for i, a in enumerate(vectors):
        for j in range(i, len(vectors)):
            b = vectors[j]
            entry = a.norm.conjugate()*b.norm*MPOEnvironment(a, law.model.H_MPO, b).full_contraction(0)
            h[i, j] = entry
            h[j, i] = entry.conjugate()
    projected = whitening.conj().T @ h @ whitening
    energies, modes = eigh(projected)
    initial = whitening.conj().T @ gram[:, 0]
    evolved = modes @ (np.exp(-1j*duration*energies)*(modes.conj().T @ initial))
    coefficients = whitening @ evolved
    state = None
    for coefficient, vector in zip(coefficients, vectors):
        if coefficient == 0:
            continue
        if state is None:
            state = vector.copy()
            state.norm *= coefficient
        else:
            state = state.add(vector, 1., coefficient, cutoff=1e-14)
    assert state is not None
    raw_h_residual = float(np.real(gram[1, 1]-np.vdot(whitening.conj().T @ gram[:, 1], whitening.conj().T @ gram[:, 1])))
    report = dict(before=list(psi.chi), after=list(state.chi), absolute_physical_start_duration=duration,
        raw_image_norm2=np.real(np.diag(gram)).tolist(), raw_current_Gram_eigenvalues=values.tolist(),
        machine_roundoff_Gram_threshold=threshold, retained_current_directions=int(np.count_nonzero(retained)),
        initial_H_image_outside_short_space_norm2_raw=raw_h_residual,
        initial_norm2=float(abs(psi.norm)**2), evolved_norm2=float(abs(state.norm)**2),
        short_space_H_antihermitian_roundoff=float(np.linalg.norm(projected-projected.conj().T)),
        coefficients_real_imag=np.stack([coefficients.real, coefficients.imag], axis=-1).tolist(),
        chi_max=max(minimum_bond, max(state.chi)+8),
        meaning='Actual short same-H evolution in the full-CAR psi,Hpsi,four before-cut QJpsi span, then occupied-Schmidt full-Fock TDVP. No zero padding, old-source-span projection or extra interaction term. All raw current Gram eigenvalues retained in evidence; only numerical metric nulls at the stated machine threshold are omitted.')
    return state, report
