"""Independent material-rotation current and Noether-vector physical consumer."""
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Any, Callable, cast
import numpy as np
from scipy.sparse.linalg import eigsh
from rich_composite_free_rot_20261014_model import material, sector, spin_matrices, magnetic_projected, magnetic_contact, rotation_matrices
from rich_composite_free_rot_20261014_current import current_projected, dipole_projected
from rich_review_wave4_20261014_tools import complex_pair, save


def main() -> None:
    inputs = json.loads(Path(sys.argv[1]).read_text())
    p = inputs['parameters']
    mat = material(p)
    j = .5
    h, derivatives = sector(mat, p, j)
    h = cast(Any, h)
    rng = np.random.default_rng(inputs['seed'])
    initial = rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0])
    solve_low = cast(Callable[..., tuple[np.ndarray, np.ndarray]], eigsh)
    energies, vectors = solve_low(h, k=2, which='SA', v0=initial, tol=1e-10)
    order = np.argsort(energies); energies = energies[order]; state = vectors[:,order[0]]
    energy = float(energies[0])
    moment_parts = {}
    for part in ['total','mechanical','peierls']:
        moments = np.zeros((3,2,2),complex)
        for axis in range(3):
            for row, mp in enumerate([-.5,.5]):
                for column, m in enumerate([-.5,.5]):
                    operator = magnetic_projected(mat,p,j,mp,j,m,axis,part)
                    moments[axis,row,column] = np.vdot(state,operator@state)
        moment_parts[part] = moments
    generators = spin_matrices(.5)
    errors = []
    for axis in range(3):
        for component in range(3):
            target = np.zeros((2,2),complex)
            if axis != component:
                third = 3-axis-component
                sign = 1 if (axis,component,third) in [(0,1,2),(1,2,0),(2,0,1)] else -1
                target = 1j*sign*moment_parts['total'][third]
            errors.append(float(np.linalg.norm(generators[axis]@moment_parts['total'][component]-moment_parts['total'][component]@generators[axis]-target)))
    receivers = []
    for jp in [.5,1.5]:
        hp,_ = sector(mat,p,jp)
        for mp in np.arange(-jp,jp+1):
            for axis in range(3):
                dipole = dipole_projected(mat,jp,float(mp),j,.5,axis)
                physical = current_projected(mat,p,jp,float(mp),j,.5,axis)
                dipole_source = dipole@state
                kinematic_source = 1j*(hp@dipole_source-dipole@(h@state))
                current_source = physical@state
                mech = current_projected(mat,p,jp,float(mp),j,.5,axis,'mechanical')@state
                hop = current_projected(mat,p,jp,float(mp),j,.5,axis,'peierls')@state
                receivers.append({'jp':jp,'mp':float(mp),'axis':axis,'dipole_current_identity_source_error':float(np.linalg.norm(kinematic_source-current_source)),
                                  'physical_current_source_norm':float(np.linalg.norm(current_source)), 'mechanical_norm':float(np.linalg.norm(mech)),
                                  'peierls_norm':float(np.linalg.norm(hop)), 'split_error':float(np.linalg.norm(current_source-mech-hop))})
    contacts = [float(np.vdot(state,magnetic_contact(mat,p,j,.5,axis)@state).real) for axis in range(3)]
    # A physically changed charge sign reverses the SAME actual material sources,
    # while Coulomb q_i q_j and the mechanically selected state are unchanged.
    reversed_mat = dict(mat)
    reversed_mat['C'] = [[-op for op in row] for row in mat['C']]
    reversed_mat['hop_moment'] = -mat['hop_moment']
    reversed_mat['body'] = dict(mat['body'],charges=-mat['body']['charges'],hops=[(term,-charge,start,end) for term,charge,start,end in mat['body']['hops']])
    reversal_errors = []
    for axis in range(3):
        normal = magnetic_projected(mat,p,j,.5,j,.5,axis)@state
        reversed_source = magnetic_projected(reversed_mat,p,j,.5,j,.5,axis)@state
        reversal_errors.append(float(np.linalg.norm(normal+reversed_source)))
    save('rotor', {'kind':'INDEPENDENT_NEW_FULL_BODY_PHYSICAL_ROTOR_CONSUMER','inputs':inputs,'body_dimension':mat['size'],'selected_lab_j':j,
                   'joint_dimension':h.shape[0],'new_lowest_sector_energies':energies.tolist(),'ground_norm':float(np.linalg.norm(state)),
                   'ground_residual':float(np.linalg.norm(h@state-energy*state)), 'magnetic_multiplet_matrices':{part:[[[complex_pair(x) for x in row] for row in component] for component in tensor] for part,tensor in moment_parts.items()},
                   'true_lab_vector_commutator_error_max':max(errors),'current_dipole_receivers':receivers,'magnetic_contacts':contacts,
                   'charge_reversal_source_errors':reversal_errors,'rotor_covariant_kinetic_energy':float(sum(np.vdot(d@state,d@state).real/(2*p['inertia'][axis]) for axis,d in enumerate(derivatives))),
                   'scope':'NEW ground solved in full jhalf K/internal space at unequal inertia. This probe does not alone establish global jhalf selection or composite binding. All internal body states retained; real material SO3 moment/current checked against dipole dynamics, not internal SU2 labels. Primitive assigned charges/spins/inertias/body geometry remain inputs.'})

if __name__ == '__main__':
    main()
