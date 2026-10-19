"""Uniform laboratory A source of actual material charge motion plus intracell hops."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from rich_composite_rot_20261014_consumer import COORDS, current_contact
from rich_composite_free_rot_20261014_model import rotation_matrices, covariant_derivatives, vector_component


def body_dipole(mat):
    positions = COORDS[mat['body']['positions']]
    return [sparse.diags(np.repeat(positions[:, :, a] @ mat['body']['charges'], mat['body']['ns']*mat['body']['nf']), format='csr') for a in range(3)]


def dipole_projected(mat, jp, mp, j, m, axis):
    return vector_component(mat, jp, mp, j, m, body_dipole(mat), axis)


def current_projected(mat, p, jp, mp, j, m, axis, part='total'):
    rotations = rotation_matrices(jp, mp, j, m)
    dj, dp = covariant_derivatives(mat, j), covariant_derivatives(mat, jp)
    dipole = body_dipole(mat)
    shape = (int(2*jp+1)*mat['size'], int(2*j+1)*mat['size'])
    mechanical = sparse.csr_matrix(shape, dtype=complex)
    for a in range(3):
        b, c = (a+1) % 3, (a+2) % 3
        f = sparse.kron(sparse.csr_matrix(rotations[axis][c]), dipole[b], format='csr')-sparse.kron(sparse.csr_matrix(rotations[axis][b]), dipole[c], format='csr')
        mechanical += (dp[a] @ f+f @ dj[a])/(2*p['inertia'][a])
    components = [current_contact(mat['body'], [0, 0, 0], np.eye(3)[a])[0] for a in range(3)]
    hopping = vector_component(mat, jp, mp, j, m, components, axis)
    if part == 'mechanical':
        return mechanical
    if part == 'peierls':
        return hopping
    if part != 'total':
        raise ValueError(part)
    return mechanical+hopping
