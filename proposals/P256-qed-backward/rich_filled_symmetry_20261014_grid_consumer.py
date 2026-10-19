"""NEW allowed-grid real-pair consumer on literal64-cell filled reference."""
import json
from pathlib import Path
import numpy as np
import rich_filled_symmetry_20261014_model as law
from rich_filled_symmetry_20261014_calculate import encode

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_filled_symmetry_20261014_actual3'


def main():
    request = json.loads((ROOT / 'rich_filled_symmetry_20261014_grid_input.json').read_text())
    model = law.build()
    channels = [law.pair_creation(model, request['kh'], request['qin'], request['epsin'], request['qout'], request['epsout'], request['omega'], particle=p, hole=h, eta=request['eta']) for p, h in request['channels']]
    kh = np.array(request['kh']); qi = np.array(request['qin']); qo = np.array(request['qout']); kp = kh + qi - qo
    momenta = [kh, kp, kh + qi, kp - qi, kh - qo, kp + qo]
    residue = np.array([p / (np.pi / 2) - np.rint(p / (np.pi / 2)) for p in momenta])
    reference = json.loads((ROOT / 'rich_filled_symmetry_20261014_actual1_result.json').read_text())['reference']
    result = {'law': model['law'], 'input': request, 'finite_reference': reference, 'Nsea': 18432, 'cells': 64, 'momentum_grid_residue_max': float(np.max(abs(residue))), 'actual_intermediate_momenta': momenta, 'channels': channels, 'largest_amplitude_magnitude': max(abs(x['amplitude']) for x in channels), 'scope': 'Every external and internal momentum is an allowed literal4^3 grid momentum modulo2pi; true same64-cell occupied reference, not off-grid integrand splice. One current on filled Slater sea accesses only one-pair states; both complete occupied hole and empty particle intermediate legs and contact retained. All other momentum sectors remain filled spectators, their Esea cancels from exact excitation denominators.'}
    (ROOT / (PREFIX + '_result.json')).write_text(json.dumps(result, default=encode, indent=2) + '\n')
    print(json.dumps(result, default=encode, indent=2))

if __name__ == '__main__':
    main()
