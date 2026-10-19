"""Compose the corrected current-N4 static certificates with the fixed source map.

No root search, old producer execution, source adoption or quantum-ground claim.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)
PINNED = {
    'induced_phi_current_n4_root_static_slow_certificate_actual_receipt.json': '79f49c752b4c2cc716d0291f1c38d448093a94f231412900be2c0e730c67dfee',
    'induced_phi_current_n4_root_static_slow_certificate_actual_natural_exit_capsule.json': '4f3b5dfb981aff87d0038a8bf5fce5e08f16f7ed0f4a7c794d21832b378abac8',
    'joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json': '2a1bbb46852707c82e477f28b5497e303b5ca66453db76d0655379f35cfb9a97',
}

def run() -> dict[str, Any]:
    start = time.monotonic()
    raws = {name: (ROOT/name).read_bytes() for name in PINNED}
    assert all(hashlib.sha256(raw).hexdigest() == PINNED[name] for name,raw in raws.items())
    root, capsule, action = [json.loads(raw) for raw in raws.values()]
    assert capsule['returncode'] == 0 and capsule['natural_exit'] and capsule['source_unchanged']
    assert root['source_unchanged'] and root['input_unchanged']
    assert action['all_inputs_and_sources_unchanged']
    assert root['exact_conditions']['input_hashes']['gaussian'] == action['input_sha256']['induced_phi_gaussian_action_certificate_receipt.json']
    assert root['Rootexist']['exists'] and root['Rootexist']['unique_in_fixed_box']
    radius = F(root['Rootexist']['radius_exact'])
    assert F(root['Rootexist']['contraction']['upper']) < 1
    assert all(F(row['upper']) < radius for row in root['Rootexist']['image_radii'])
    assert all(F(row['gap_physical_lower']) > 0 and row['rank_each'] == 128 for row in root['gaps'])
    assert sum(row['species_count']*row['rank_each'] for row in root['gaps'])+4 == action['source_weights']['fixed_total_particle_number']
    assert root['gaps'][0]['original_h_occupied_sheet'] == 'UPPER'
    slow = root['q0lower']; coupled = slow['full_coupled_q0_certificate']
    principal = F(slow['q0_static_lower_divided_Lambda'])
    full = F(coupled['q0_full90_static_stiffness_lower_divided_Lambda'])
    assert principal >= full > F(2,5) and slow['actual_Gphi'] == 8
    phi_frequency_lower = 8*principal*F(2)**4096
    full_frequency_lower = F(3,10)*full*F(2)**4096
    assert F(slow['q0_Phi_principal_frequency_squared_lower']) == phi_frequency_lower
    assert F(coupled['q0_full90_frequency_squared_lower']) == full_frequency_lower
    epsilon = F(action['action_input']['epsilon']['upper'])
    assert 0 < epsilon < F(2)**-400
    assert all((ROOT/name).read_bytes() == raw for name,raw in raws.items())
    return {
        'kind': 'CURRENT_N4_STATIC_SUPPLIERS_COMPOSED_WITH_DECLARED_COMPACT_SOURCE',
        'input_sha256': PINNED,
        'same_frozen_gaussian_epoch': root['exact_conditions']['input_hashes']['gaussian'],
        'discharged_at_exact_declared_scope': ['unique fixed-box implicit root exists', 'all45 normal forces vanish for classical plus leading all64q fast-ZPE feedback', '65 independent static rank128 seas with correctly signed heavy occupation', 'positive static Phi45 and full90 q0 pencils with actual cometric'],
        'root_mean_or_state_changed': False,
        'phi_q0_frequency_squared_lower': str(phi_frequency_lower),
        'full90_q0_frequency_squared_lower': str(full_frequency_lower),
        'uniform_source_intertwiner_epsilon_upper': str(epsilon),
        'source_map_scope': action['proof']['T'],
        'remaining': ['declared exact rational NEW source-law agreement, not adoption of binary Python literals or originalSOURCE', 'body floating first moments are not interval certified by this composition', 'no interacting HC or full quantum ground, no stationary photon pole, common cone or Compton admission'],
        'superseded_initial_root_price_preserved': root['superseded_initial_attempt'],
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'all_inputs_unchanged': True,
        'runtime': {'executable': sys.executable, 'python': sys.version},
        'elapsed_seconds': time.monotonic()-start,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'kind': result['kind'], 'discharged': result['discharged_at_exact_declared_scope'], 'elapsed_seconds': result['elapsed_seconds']}))
