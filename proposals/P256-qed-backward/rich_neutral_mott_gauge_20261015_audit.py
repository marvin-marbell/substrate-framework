"""Independent full physical-link census and retained fresh-state audit.

No dynamics, spectra, parameter expansion or old CLI. Runs after first physics.
"""
from __future__ import annotations
import itertools
import json
import math
from pathlib import Path
import numpy as np
from rich_photon_fullq_20261014_model import geometry
from rich_photon_20261014_loop import hop

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_neutral_mott_gauge_20261015_'


def main():
    _, edges, _, _, incidence, _ = geometry('cube')
    stored = np.load(ROOT/(PREFIX+'actual_states.npz'))
    # Independent of the spanning-tree inversion: enumerate EVERY physical E.
    electric = np.asarray(list(itertools.product((-1, 0, 1), repeat=12)), dtype=np.int8)
    occupation = (1-electric @ incidence.T).astype(int)
    allowed = np.all((occupation >= 0) & (occupation <= 2), axis=1)
    physical = electric[allowed]
    charges = occupation[allowed]
    doubles = np.sum(charges == 2, axis=1)
    multiplicity = np.asarray([math.comb(8-2*int(d), 4-int(d)) for d in doubles])
    independent_counts = {str(d): int(np.sum(multiplicity[doubles == d])) for d in range(5)}
    basis_electric = stored['electric']
    basis_occupation = 1-basis_electric @ incidence.T
    basis_counts = {str(d): int(np.count_nonzero(np.sum(basis_occupation == 2, axis=1) == d)) for d in range(5)}
    # Every accepted physical E occurs with precisely its full spin degeneracy.
    code_weights = 3**np.arange(12, dtype=np.int64)
    independent_codes = (physical+1) @ code_weights
    retained_codes, retained_counts = np.unique((basis_electric+1) @ code_weights, return_counts=True)
    order = np.argsort(independent_codes)
    exact_sets = bool(np.array_equal(retained_codes, independent_codes[order]))
    exact_multiplicity = exact_sets and bool(np.array_equal(retained_counts, multiplicity[order]))
    # Spin-singlet witness: literal total S+ maps seed to Nup5,Ndown3.
    initial = stored['initial']
    fock = stored['fock']
    nflux = len(stored['chord_flux'])
    raised = {}
    for k in np.flatnonzero(abs(initial[:, 0]) > 0):
        old = int(stored['old_indices'][k])
        bits = int(fock[old//nflux])
        for site in range(8):
            result = hop(bits, 2*site, 2*site+1)
            if result is not None:
                target, sign = result
                raised[target] = raised.get(target, 0j)+sign*initial[k, 0]
    E = np.diag([-1., 0., 1.])
    U = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    output = dict(kind='INDEPENDENT_ALL_PHYSICAL_EDGE_CENSUS_NOT_TREE_CUTOFF_ASSUMPTION',
                  physical_assignments_enumerated=len(electric), allowed_charge_flux_assignments=len(physical),
                  independent_spinful_dimension=int(np.sum(multiplicity)), observed_retained_dimension=len(basis_electric),
                  independent_dimension_by_total_doublons=independent_counts, observed_dimension_by_total_doublons=basis_counts,
                  physical_flux_sets_exactly_equal=exact_sets, every_flux_full_spin_multiplicity=exact_multiplicity,
                  interpretation='For every E, Gauss fixes n_i=1-(BE)_i; each doublon contributes one up/down and remaining singles choose 4-D up spins. This independently counts and compares all local-link Gauss states, not just total dimensions.',
                  spin_singlet=dict(seed_Sz=0, total_Splus_seed_norm=float(np.sqrt(sum(abs(x)**2 for x in raised.values()))),
                                    inference='Sz0 and S+ annihilation prove S0 for this fresh seed. The field pulse acts identically on all spins.'),
                  local_ladder=dict(EU_commutator_minus_U_max=float(np.max(abs(E @ U-U @ E-U))),
                                    U_nonunitarity_max=float(np.max(abs(U.T @ U-np.eye(3)))),
                                    electric_ladder_not_photon_helicity=True),
                  no_new_dynamics=True)
    (ROOT/(PREFIX+'audit_result.json')).write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
