"""NEW scientific calculation; exact declared finite Hilbert space, no legacy run."""
import json
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from rich_electron_20261014_model import (Parameters, build_sector, solve,
    spectral_data, retarded_green, sector_sources, momentum_sources)

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_electron_20261014_'


def mean(state, matrix):
    return float(np.vdot(state, matrix @ state).real)


def packet_result(sectors, params):
    sea, charged = sectors[4], sectors[5]
    ground = sea['vectors'][:, 0]
    raw = sector_sources(4, 5, params)[0] @ ground
    source_weight = float(np.vdot(raw, raw).real)
    packet = raw / np.sqrt(source_weight)
    coeff = charged['vectors'].conj().T @ packet
    times = np.linspace(0, 12, 97)
    states = charged['vectors'] @ (coeff[:, None] * np.exp(-1j * charged['energies'][:, None] * times))
    reference_density = np.array([mean(ground, n) for n in sea['density']])
    densities = np.array([[mean(states[:, i], n) for n in charged['density']] for i in range(len(times))])
    current = [mean(states[:, i], charged['J']) for i in range(len(times))]
    photons = [mean(states[:, i], charged['nb']) for i in range(len(times))]
    pairs = [mean(states[:, i], charged['positive_band_number']) for i in range(len(times))]
    entropy = []
    for i in range(len(times)):
        wave = states[:, i].reshape(-1, params.boson_levels)
        eigenvalues = np.linalg.eigvalsh(wave.conj().T @ wave)
        eigenvalues = eigenvalues[eigenvalues > 1e-14]
        entropy.append(float(-np.dot(eigenvalues, np.log(eigenvalues))))
    spin_squared = sum(s @ s for s in charged['spin'])
    return {'times': times.tolist(), 'excess_number_density': (densities-reference_density).tolist(),
            'number_flux_current': current, 'charge_flux_current': (-np.array(current)).tolist(),
            'cavity_number': photons, 'positive_band_number': pairs, 'matter_field_entropy_nats': entropy,
            'source_mode': 'site0 spin up cdagger', 'source_weight': source_weight,
            'normalized_source_norm': float(np.vdot(packet, packet).real),
            'source_charge': -1, 'source_spin_z': mean(packet, charged['spin'][2]),
            'source_spin_squared': mean(packet, spin_squared),
            'fermion_exchange': 'exact CAR Fock representation; charged odd parity (-1)^5=-1',
            'sea_positive_band_number': mean(ground, sea['positive_band_number']),
            'sea_cavity_number': mean(ground, sea['nb']),
            'charge_excess_sum_max_error': float(np.max(np.abs(np.sum(densities-reference_density, axis=1)-1))),
            'max_site0_density_departure': float(np.max(np.abs(densities[:, 0]-densities[0, 0]))),
            'peak_absolute_current': float(np.max(np.abs(current))),
            'max_matter_field_entropy': max(entropy)}


def conduction_spectrum(sectors, params, k=np.pi):
    data = spectral_data(sectors, params, k)
    # h_AB(k=pi)=-(tstrong-tweak), so positive-band source is (A-B)/sqrt(2).
    spin_up = np.array([1, 0, -1, 0]) / np.sqrt(2)
    amps = data['plus_amplitudes'] @ spin_up
    weights = np.abs(amps)**2
    active = np.where(weights > 1e-10)[0]
    first_energy = float(data['plus_energies'][active[0]])
    low = np.abs(data['plus_energies']-first_energy) < 1e-7
    lowest_weight = float(weights[low].sum())
    summed = float(weights.sum())
    peaks = sorted([{'energy': float(data['plus_energies'][i]), 'weight': float(weights[i])}
                    for i in active], key=lambda x: x['weight'], reverse=True)[:12]
    v, w = data['plus_amplitudes'], data['minus_amplitudes']
    sumrule = v.conj().T @ v + (w.conj().T @ w).T
    spin_flip = []
    trace_green = []
    frequencies = np.linspace(-4, 4, 401)
    for frequency in frequencies:
        green = retarded_green(data, frequency)
        spin_flip.append(np.max(np.abs(green[np.ix_([0, 2], [1, 3])])))
        trace_green.append(float(-np.trace(green).imag / np.pi))
    return {'addition_threshold': first_energy, 'lowest_cluster_weight': lowest_weight,
            'conduction_source_weight': summed, 'lowest_cluster_fraction': lowest_weight/summed,
            'satellite_fraction': 1-lowest_weight/summed, 'largest_peaks': peaks,
            'car_matrix_sum_rule_max_error': float(np.max(np.abs(sumrule-np.eye(4)))),
            'max_spin_flip_green': float(max(spin_flip)),
            'frequencies': frequencies.tolist(), 'trace_spectral_function': trace_green}, data


def charged_flux_dispersion(params):
    angles = [-0.08, 0.0, 0.08]
    energies = []
    currents = []
    for theta in angles:
        n4, n5 = build_sector(params, 4, theta), build_sector(params, 5, theta)
        e4 = eigh(n4['H'], eigvals_only=True, subset_by_index=[0, 0], check_finite=False)[0]
        e5, v5 = eigh(n5['H'], subset_by_index=[0, 1], check_finite=False)
        energies.append(float(e5[0]-e4))
        currents.append(mean(v5[:, 0], n5['J']))
    curvature = (energies[0]+energies[2]-2*energies[1]) / 0.08**2
    # Four physical links: k shift is theta/4. Dimerized gap valley is k=pi/2.
    # d2E/dk2 = 16*d2E/dtheta2 for this uniform flux probe.
    return {'theta': angles, 'charged_addition_energy': energies,
            'number_current_N5': currents, 'flux_curvature': curvature,
            'band_curvature_d2E_dk2': 16*curvature,
            'inertial_band_mass': 1/(16*curvature) if curvature > 0 else None,
            'mass_scope': 'finite-ring twist curvature, not a relativistic pole mass'}


def spin_response(sectors, params, b=0.3):
    """Physical SU2 response to declared primitive field energy b, not derived g=2."""
    sources = sector_sources(4, 5, params)
    sea = sectors[4]['vectors'][:, 0]
    raw = (sources[0]+sources[1]) @ sea / np.sqrt(2)
    packet = raw / np.linalg.norm(raw)
    charged = build_sector(params, 5, spin_field=b)
    energies, vectors = eigh(charged['H'], check_finite=False)
    coefficients = vectors.conj().T @ packet
    times = np.linspace(0, 8, 33)
    spin_values = []
    for t in times:
        state = vectors @ (coefficients*np.exp(-1j*energies*t))
        spin_values.append([mean(state, s) for s in charged['spin']])
    return {'primitive_spin_field_energy': b, 'H_field': '-b*S_z',
            'initial_source': '(cdagger_site0_up+cdagger_site0_down)/sqrt(2), normalized after acting on Omega',
            'times': times.tolist(), 'physical_spin_xyz': spin_values,
            'scope': 'A physical spin-half source precesses while coupled charge/cavity dynamics continue; magnetic moment coefficient b/B is not supplied by orbital Peierls law.'}


def main():
    params = Parameters()
    result = {'candidate': 'spinful dimerized ring / exact quantum Peierls cavity v1',
              'units': {'hbar': 1, 'electron_charge': -1, 'link_length': 1, 'energy_unit': 'tstrong'},
              'scope': 'N3/4/5 all 8-mode CAR configurations, one oscillator with exact finite projected coordinate exponential',
              'parameters': asdict(params), 'branches': {}}
    serialized = {}
    for tag, p in [('dressed', params), ('undressed_wrong', replace(params, coupling=0.0))]:
        sectors = solve(p)
        spectrum, data_pi = conduction_spectrum(sectors, p)
        branch = {'parameters': asdict(p), 'sector_ground_energies': {str(n): float(s['energies'][0]) for n,s in sectors.items()},
                  'packet': packet_result(sectors, p), 'conduction_spectrum_pi': spectrum,
                  'charged_dispersion': charged_flux_dispersion(p),
                  'spin_response': spin_response(sectors, p)}
        result['branches'][tag] = branch
        for k, label in [(0, 'zero'), (np.pi, 'pi')]:
            data = data_pi if label == 'pi' else spectral_data(sectors, p, k)
            for key, value in data.items():
                serialized[f'{tag}_{label}_{key}'] = value
        for n, sector in sectors.items():
            serialized[f'{tag}_N{n}_energies'] = sector['energies']
    dressed, wrong = result['branches']['dressed'], result['branches']['undressed_wrong']
    result['decision_comparison'] = {
        'addition_energy_shift': dressed['conduction_spectrum_pi']['addition_threshold']-wrong['conduction_spectrum_pi']['addition_threshold'],
        'dressed_satellite_fraction': dressed['conduction_spectrum_pi']['satellite_fraction'],
        'undressed_satellite_fraction': wrong['conduction_spectrum_pi']['satellite_fraction'],
        'band_mass_ratio_dressed_over_undressed': dressed['charged_dispersion']['inertial_band_mass']/wrong['charged_dispersion']['inertial_band_mass'],
        'current_series_l2_difference': float(np.linalg.norm(np.array(dressed['packet']['number_flux_current'])-np.array(wrong['packet']['number_flux_current']))),
        'excess_density_series_l2_difference': float(np.linalg.norm(np.array(dressed['packet']['excess_number_density'])-np.array(wrong['packet']['excess_number_density']))),
        'interpretation': 'Dynamic cavity versus g=0 changes charge propagation, spectral sidebands and current; both share same CAR sea, hopping, source and oscillator cutoff.'}
    (ROOT/(PREFIX+'result.json')).write_text(json.dumps(result, indent=2))
    np.savez_compressed(ROOT/(PREFIX+'spectral.npz'), **serialized)
    print(json.dumps(result['decision_comparison'], indent=2))


if __name__ == '__main__':
    main()
