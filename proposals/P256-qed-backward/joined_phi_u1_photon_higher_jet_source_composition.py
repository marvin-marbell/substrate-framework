"""Compose outward initial M3, full-H tails and compact-Phi transfer.

The NEW 2^-16 clock is an exploratory consumer, not adoption or evolution.
No root, old coefficient producer or old primary is executed.
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
import numpy as np
import joined_phi_u1_current_signal_interval_certificate as ar

ROOT = Path(__file__).resolve().parent
CLOCK = F(1, 2**16)
PINS = {
    'joined_phi_u1_photon_causal_response_verified_20261010_receipt.json': 'c7c3ec787e34d5aa58c806650db7603e9f06c92d388cf590672d3644efbe262f',
    'joined_phi_u1_initial_photon_cubic_jet_interval_receipt.json': '0518db967dbc2ae72f33a6c4bc7f7a794dca45cc8487369222be072858f23456',
    'joined_phi_u1_initial_photon_cubic_jet_interval_receipt.npz': 'e0acf1a8de056e4527c9e542c1c40784d7cf7ad0fc571d6db58b1ad04f9efeb2',
    'joined_phi_u1_current_signal_interval_certificate.py': '386c684f42aa3715dbf370d075cb1b4f5a1b0a645c2784fd9ace2c97ee491810',
}
PROOF = {
    'body': 'Actual BODY/preparation parity, not stationarity: M(t)=-128 C0 t+M3 t^3/6+R5; D(t)-128 mass t I=-8192 C0 t^3/3+128 M3 t^5/120+R7; Pi response-minus-mass I=-64 C0 t^2+M3 t^4/24+R6. Full nonstationary M3 retained. Uniform full-H future tails use native spectral Holder/positive moments, not future Wick.',
    'frame': 'Initial Gaussian on the full129 exact physical space is invariant under orthonormal frame changes. Qstar is the certified polar orthonormal frame anchored to Qobs; exact physical3 torons are separately probed. Subtract the NEW explicit ||Cstar-Cobs||HS frame repair from predecessor C0 lower. Free128 mass t I belongs to canonical frame; no numerical Gram defect is called a photon correction.',
    'matrix': 'For a129x129 entrywise l1 complex ball with uniform radius r, HS radius<=129r. Centre HS uses exact decoded binary squares and directed square-root upper. Operator norm<=HS. Thus the bound proves every real/complex directional restoring quadratic-form real part, without symmetrizing away nonstationary antisymmetric M3.',
    'physical_scaling': 'Let R=t/t0>=1. In the existing uniform HT=T(Hb+Ephi)+residual proof, d(t)=t epsilon n and h(t)=Erefabs d+t epsilon Hb+2epsilon n; both <=R times their t0 prices. Quartic current w=kappa*[d^2*(d*h+C*d^2)/quartic]^(1/4) is positive and homogeneous of degree1, so w<=R w0, also for Theta insertions. Mixed error 2[(w+endpoint*n)(bt+dt)+jbody*n*dt]<=R^2 oldError. Therefore D/Pi/mixed physical transfer is bounded by R^4/R^3/R^2 times old upper. This is a NEW exact clock composition, not running/changing old CLOCK or assuming bounded physical current.',
    'domain': 'Higher Taylor jets belong to smooth BODY initial vectors. Physical comparison uses existing compact-source D(H_F^2), Theta insertions and quartic graph norm. No D(H_F^8), physical full-Phi timeparity, ground or future Slater premise.',
    'scope': 'Same H/parameters/fixed rational mean/source and old clocks. NEW2^-16 not adopted. Declared compact-Phi prepared kernels, not originalSOURCE/CURRENT427, measured evolution, stationary pole, massless dispersion, commoncone or Compton.'}


def hs_upper(matrix: np.ndarray) -> F:
    squares = sum((F.from_float(float(z.real))**2+F.from_float(float(z.imag))**2 for z in matrix.flat), F(0))
    return ar.end(ar.iv.sqrt(ar.exact(squares)), True)


def run(output: Path) -> dict[str, Any]:
    started = time.monotonic()
    sys.set_int_max_str_digits(0)
    ar.iv.dps = 90
    captured = {Path(__file__).name: Path(__file__).read_bytes()}

    def take(name: str) -> bytes:
        raw = (ROOT/name).read_bytes()
        if name in PINS and hashlib.sha256(raw).hexdigest() != PINS[name]:
            raise ArithmeticError('Changed photon composition epoch: '+name)
        captured[name] = raw
        return raw

    old = json.loads(take('joined_phi_u1_photon_causal_response_verified_20261010_receipt.json'))
    cubic = json.loads(take('joined_phi_u1_initial_photon_cubic_jet_interval_receipt.json'))
    caps = json.loads(take('joined_phi_u1_initial_photon_cubic_jet_interval_natural_exit_capsule.json'))
    if caps['numeric_exit'] != 0 or not caps['natural_exit'] or not caps['source_stable']:
        raise ArithmeticError('Missing completed outward initial consumer')
    if hashlib.sha256(take('joined_phi_u1_initial_photon_cubic_jet_interval.py')).hexdigest() != caps['source_sha256']:
        raise ArithmeticError('Outward producer/capsule source mismatch')
    for name, sha in caps['outputs'].items():
        if hashlib.sha256(take(name)).hexdigest() != sha:
            raise ArithmeticError('Outward completed output mismatch: '+name)
    take('joined_phi_u1_current_signal_interval_certificate.py')
    tails_name = 'joined_phi_u1_higher_response_tail_composition_receipt_20261010.json'
    tails = json.loads(take(tails_name))
    if tails['natural_numeric_exit'] != 0 or F(tails['clock_exact']) != CLOCK:
        raise ArithmeticError('Future tail horizon mismatch')
    if hashlib.sha256(tails['producer_program'].encode()).hexdigest() != tails['producer_sha256']:
        raise ArithmeticError('Tail producer program binding mismatch')
    for name, sha in tails['input_sha256'].items():
        if hashlib.sha256(take(name)).hexdigest() != sha:
            raise ArithmeticError('Future tail input mismatch: '+name)
    ratio = CLOCK/F(old['clock_exact'])
    if ratio < 1 or cubic['measured_time_evolution']:
        raise ArithmeticError('Invalid cached-transfer composition regime')
    matrix_file = ROOT/'joined_phi_u1_initial_photon_cubic_jet_interval_receipt.npz'
    states: dict[str, Any] = {}
    with np.load(matrix_file, allow_pickle=False) as arrays:
        for label, mass in (('background', F(1)), ('raw_source', F(1,2))):
            before = old['all129_states'][label]
            new = cubic['all129_states'][label]
            if F(before['raw_norm_squared']) != mass:
                raise ArithmeticError('Raw source mass changed')
            matrix = arrays[label+'_M3_centre']
            if matrix.shape != (129,129) or not np.all(np.isfinite(matrix)):
                raise ArithmeticError('Full129 cubic coefficient incomplete')
            radius = F.from_float(new['M3']['entry_l1_radius'])
            centre_norm = hs_upper(matrix)
            coefficient_upper = centre_norm+129*radius
            repair = F(new['prior_literal_Q_C0_frame_HS_correction_upper_exact'])
            contact = F(before['contact_minimum_directional_outward_lower']['lower'])-repair
            contact_hs = F(before['contact_HS_lower']['lower'])-repair
            if contact <= 0 or contact_hs <= 0:
                raise ArithmeticError('New frame repair consumes contact certificate')
            body_tail = tails['photon_states'][label]
            physical = before['physical_transfer']
            rows = {}
            for name, power, leading, coefficient, tail_key, transfer_key, transfer_power in (
                ('D', 3, F(8192,3), F(128,120)*CLOCK**5, 'D_t7_remainder_upper_exact', 'D_transfer', 4),
                ('Pi', 2, F(64), CLOCK**4/24, 'Pi_t6_remainder_upper_exact', 'Pi_transfer', 3),
                ('mixed', 1, F(128), CLOCK**3/6, 'mixed_t5_remainder_upper_exact', 'mixed_transfer', 2),
            ):
                signal = leading*contact*CLOCK**power
                cubic_price = coefficient*coefficient_upper
                remainder = F(body_tail[tail_key])
                transfer = ratio**transfer_power*F(physical[transfer_key]['upper'])
                total_error = cubic_price+remainder+transfer
                lower = signal-total_error
                if lower <= 0:
                    raise ArithmeticError('Coupled all129 restoring response unresolved: '+label+'/'+name)
                rows[name] = {'leading_directional_signal_lower_exact': str(signal),
                    'initial_M3_contribution_upper_exact': str(cubic_price),
                    'initial_M3_uncertainty_upper_exact': str(coefficient*129*radius),
                    'future_remainder_upper_exact': str(remainder),
                    'physical_Phi_transfer_upper_exact': str(transfer),
                    'directional_restoring_lower_exact': str(lower),
                    'directional_restoring_lower_display': float(lower),
                    'error_over_signal_display': float(total_error/signal)}
            states[label] = {'raw_mass_exact': str(mass), 'canonical_contact_directional_lower_exact': str(contact),
                'canonical_contact_HS_lower_exact': str(contact_hs), 'C0_additional_frame_HS_error_exact': str(repair),
                'M3_centre_HS_upper_exact': str(centre_norm), 'M3_HS_uncertainty_upper_exact': str(129*radius),
                'M3_HS_upper_exact': str(coefficient_upper), 'responses': rows,
                'toron_probes': 'Exact3 uniform physical192 vectors retained in outward coefficient archive; all129 lower includes their directions.'}
    stable = all((ROOT/name).read_bytes() == raw for name, raw in captured.items())
    if not stable:
        raise ArithmeticError('Composition source/input mutated')
    result = {'status': 'ALL129_LONGER_CLOCK_PHYSICAL_RESTORING_RESPONSE_PROVED_AT_DECLARED_PREPARATION',
        'clock_exact': str(CLOCK), 'clock_adopted': False, 'old_clock_unchanged': old['clock_exact'],
        'clock_ratio_exact': str(ratio), 'proof': PROOF, 'all129_states': states,
        'physical_frame': cubic['physical_frame'], 'all_source_input_bytes_stable': stable,
        'source_sha256': hashlib.sha256(captured[Path(__file__).name]).hexdigest(),
        'input_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in captured.items()},
        'measured_time_evolution': False, 'originalSOURCE_adopted': False,
        'runtime': {'executable': sys.executable, 'python': sys.version}, 'elapsed_seconds': time.monotonic()-started}
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    receipt = run(parser.parse_args().output)
    print(json.dumps({'status': receipt['status'], 'clock': receipt['clock_exact'],
        'directional_lowers': {label: {name: r['directional_restoring_lower_display'] for name, r in state['responses'].items()} for label, state in receipt['all129_states'].items()}}))
