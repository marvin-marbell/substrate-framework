"""Min-max free exterior attempt; never retag a floating Ritz as ground."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    raw = Path(__file__).read_bytes()
    input_path = root/args.receipt
    input_raw = input_path.read_bytes()
    receipt = json.loads(input_raw)
    # Literal canonical4 matrices independently dGamma'd into wedge2.
    masks = [3, 5, 9, 6, 10, 12]
    a = np.diag([-1., 1., -1., 1.])
    b = np.kron([[0., 1.], [1., 0.]], np.diag([1., -1.]))
    x = np.array([[0., 1.], [1., 0.]])
    c = np.block([[np.zeros((2, 2)), 1j*x], [-1j*x, np.zeros((2, 2))]])
    minima = []
    thresholds = []
    profiles = []
    for role in receipt['parameters']:
        h = role['onsite_float']*a+role['c0']*b+role['ci']*c
        wedge = np.zeros((6, 6), dtype=complex)
        for j, mask in enumerate(masks):
            for q in range(4):
                if not mask & (1 << q):
                    continue
                reduced = mask ^ (1 << q)
                sign_q = (-1)**(mask & ((1 << q)-1)).bit_count()
                for p in range(4):
                    if reduced & (1 << p):
                        continue
                    sign_p = (-1)**(reduced & ((1 << p)-1)).bit_count()
                    wedge[masks.index(reduced | (1 << p)), j] += sign_q*sign_p*h[p, q]
        eigenvalues = np.linalg.eigvalsh(wedge)
        # Invariant singlet is zero; selected vector5 has the two nonzero pairs.
        omega1 = -float(eigenvalues[0]); omega2 = -float(eigenvalues[1])
        n = role['spin_replicas']
        minima.append(-n*omega1)
        thresholds.append(omega1-omega2)
        profiles.append({'role': role['role'], 'replicas': n, 'native_wedge_eigenvalues': eigenvalues.tolist(), 'harmonic_ground_free': -n*omega1, 'first_free_excitation': omega1-omega2})
    free_minimum = sum(minima)
    second_threshold = free_minimum+min(thresholds)
    energy = receipt['zero_u']['energy']
    passed = energy < second_threshold
    output = {'kind': 'EXPLICIT_FREE_MINMAX_EXTERIOR_CERTIFICATE_ATTEMPT_NOT_GROUND_ADMISSION', 'source_sha256': hashlib.sha256(raw).hexdigest(), 'input': input_path.name, 'input_sha256': hashlib.sha256(input_raw).hexdigest(), 'command': [sys.executable, *sys.argv], 'profiles': profiles, 'derivation': 'A generic SO5 generator in vector5 has weights {-omega1,-omega2,0,omega2,omega1}. Harmonic rankn ground weight is -n*omega1 and next weight is -(n-1)*omega1-omega2. Both are tracefree because contraction lands in degree n-2 with minimal energy -(n-2)*omega1. In product active2/aux126 the lowest free excitation is min of these two free gaps. H_full=H_free+64rho^2 implies minmax lambda2(full)>=lambda2(free). This is a genuine sufficient exterior comparison, not positivity of H-E.', 'free_product_minimum_numeric': free_minimum, 'second_free_threshold_numeric': second_threshold, 'actual_Ritz_energy': energy, 'margin_threshold_minus_Ritz': second_threshold-energy, 'Temple_separation_sufficient_test_passed': passed, 'ground_certificate_available': False, 'reason': 'Free comparison threshold does not separate Ritz from all exterior levels.' if not passed else 'Floating sufficient inequality survives, but rigorous outward spectral and residual bounds are absent; no ground label.', 'ritz_stage_retained': True, 'L_equals_H_minus_E_positive_assumed': False, 'source_and_input_unchanged': Path(__file__).read_bytes() == raw and input_path.read_bytes() == input_raw}
    path = root/'local_so5_resum_exterior_proof_20261011.json'
    path.write_text(json.dumps(output, indent=2)+'\n')
    with zipfile.ZipFile(root/'local_so5_resum_exterior_capsule_20261011.zip', 'w', zipfile.ZIP_DEFLATED) as capsule:
        capsule.writestr(Path(__file__).name, raw)
        capsule.writestr(input_path.name, input_raw)
        capsule.writestr(path.name, path.read_bytes())
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
