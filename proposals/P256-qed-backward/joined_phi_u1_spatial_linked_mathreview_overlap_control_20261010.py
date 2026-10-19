"""Reject multiplying the once-shared scalar weight once per species."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

path = Path(sys.argv[1])
raw = path.read_bytes()
receipt = json.loads(raw)
phis = 2*np.pi*np.arange(512)/512
w = (1+np.cos(phis)/4)**2/(33/32)
rows = []
for case in receipt['cases']:
    angles = np.asarray(case['raw_cell_phi'])
    weight = (1+np.cos(angles)/4)**2/(33/32)
    weighted_single_overlap = np.asarray(case['raw_cell_overlap_squared'])
    single_overlap = weighted_single_overlap/weight
    actual_shared = float(np.mean(weight*single_overlap**64))
    independent = float(np.mean(weighted_single_overlap)**64)
    rows.append({'N':case['N'], 'correct_E_shared_A2_overlap_to_64':actual_shared, 'independent_E_A2_overlap_to_64':independent, 'corrected_shared_minus_independent':actual_shared-independent, 'current_misweighted_shared':float(np.mean(weighted_single_overlap**64)), 'current_wrong_independent_field_defect':case['wrong_independent_field_defect'], 'single_overlap_range':[float(min(single_overlap)),float(max(single_overlap))]})
result = {
    'kind':'INDEPENDENT_ONCE_SHARED_WEIGHT_CONTRARY_CONTROL',
    'input_path':str(path),'input_sha256':hashlib.sha256(raw).hexdigest(),'python':sys.executable,
    'constant_overlap_positive_control':{'o':1,'correct_shared':float(np.mean(w)),'correct_independent':float(np.mean(w)**64),'current_misweighted_shared':float(np.mean(w**64))},
    'actual_receipt_overlap_controls':rows,
    'finding':'mean((A² o)^64) weights A² 64 times. For the normalized shared probability measure use mean(A² o^64). Constant o=1 must return1; misweighted expression fails even when there is no overlap variation.',
    'nonclaim':'This probability-correlation witness is not the overlap of two shifted complete A-weighted states; that overlap also requires the two scalar amplitudes at their respective fields. No source norm or energy is altered.'
}
print(json.dumps(result,indent=2))
if not (result['constant_overlap_positive_control']['current_misweighted_shared'] > 1e6 and abs(result['constant_overlap_positive_control']['correct_shared']-1) < 1e-12):
    sys.exit(2)
