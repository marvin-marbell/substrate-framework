"""Preserve actual first-chain evidence before the material bridge correction."""
import json
from pathlib import Path
import zipfile
import rich_unitary_rotor_photon_20261016_capture as capture

ROOT=Path(__file__).resolve().parent
P='rich_unitary_rotor_photon_20261016_'


def main():
    first=json.loads((ROOT/(P+'exit.json')).read_text())
    names=['execution_input.json','exit.json','source_capsule.zip','stdout.txt','stderr.txt','result.json','small_result.json']
    files=[ROOT/(P+n) for n in names if (ROOT/(P+n)).exists()]
    files+=sorted(ROOT.glob(P+'b*_raw.npz'))+sorted(ROOT.glob(P+'b*_result.json'))
    with zipfile.ZipFile(ROOT/(P+'first_attempt_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
    correction=dict(first_natural_exit=first['natural_exit'],first_source_sha256=first['source_sha256'],
                    cause='Fixed temporal initialization plus sparse one-angle proposals leave most temporal coordinates unvisited; no trusted thermal estimator from first run.',
                    material_correction='Fresh compact Gaussian full-link temporal bridges with winding images, exact free-heatkernel Hastings ratio; full temporal initialization and Gaussian local holonomy.',
                    unchanged='Original law, PH, full Gauss projector, heatkernel and determinant weight, full Wick/current estimators. Old sampler removed from active module; exact old source preserved in first source capsule.')
    (ROOT/(P+'correction.json')).write_text(json.dumps(correction,indent=2)+'\n')
    capture.main()


if __name__=='__main__':main()
