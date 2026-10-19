"""NEW discriminator: spinor operator, occupied reference and carrier subspace differ."""
import json
from pathlib import Path
import numpy as np
from rich_holonomy_em_20261014_model import build, sector
from rich_frame_photon_20261014_model import spectrum
from rich_review_wave7_20261014_holonomy import norm, expectation, propagate

ROOT=Path(__file__).resolve().parent


def main():
    inputs=json.loads((ROOT/'rich_review_wave7_20261014_source_rotation_input.json').read_text())
    model=build(inputs['parameters'],external=inputs['external'])
    generators=model['matter_spin'];sources=model['matter_sources']
    source_casimir=[]
    for source in sources:
        adjoint=sum(s@(s@source-source@s)-(s@source-source@s)@s for s in generators)
        source_casimir.append(float(np.linalg.norm(adjoint-.75*source)))
    ix=sector(model,4);h=model['H'][ix][:,ix];energy,vectors,receipt=spectrum(h,count=6);reference=vectors[:,0]
    invariance=[]
    for spin in model['S']:
        restricted=spin[ix][:,ix]
        rotated=propagate(restricted,reference,-1j*inputs['rotation_angle'])
        invariance.append({'frame_rotation_fidelity':float(abs(np.vdot(reference,rotated))**2),
                           'reference_frame_spin_mean':expectation(restricted,reference),
                           'reference_frame_spin_variance':expectation(restricted@restricted,reference)-expectation(restricted,reference)**2,
                           'whole_H_frame_commutator':norm(model['H']@spin-spin@model['H'])})
    result={'kind':'NEW_PHYSICAL_SOURCE_OPERATOR_VERSUS_REFERENCE_AND_CARRIER_DISCRIMINATOR','inputs':inputs,
            'source_adjoint_Casimir_jhalf_residuals':source_casimir,'source_generator':'Actual total LEFT frame coordinate operators from two raw right-Gauss frames, not installed spin matrices.',
            'neutral_reference':{'N':4,'S2_frame':expectation(model['S2'][ix][:,ix],reference),**receipt},
            'frame_rotation_reference_and_H':invariance,
            'choice':'Dressed charge sources DO transform as jhalf under physical frame LEFT rotations. Neutral occupied reference is not invariant under those frame rotations in pinned nonisotropic geometry. Unfiltered charged total-frame jhalf leakage therefore refutes an exact closed frame-halfsector but is not a no-go for every spinor quasiparticle, since inclusive fields/pairs carry angular momentum.',
            'missing_full_rotation':'No dynamical orbital/geometry generator or rotation-closed spatial EM mode basis is supplied. No full spatial+orbital+field physical rotation/NoEther invariance or isolated spinor pole follows from source adjoint Casimir alone.',
            'not_imposed':'No state spin projector, spin filter, energy penalty, refitted coefficient, suppressed higher virtuals or old producer replay.'}
    (ROOT/'rich_review_wave7_20261014_source_rotation_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
