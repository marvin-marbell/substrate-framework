"""Persist chosen primitive coefficients/source tensors, no evolution or replay."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import scipy
ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_composite_local_flux_binding_20261017_'


def main() -> None:
    spec = importlib.util.spec_from_file_location('binding_frozen_coefficient_model', ROOT/(PREFIX+'model.py'))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    for stamp in sys.argv[1:]:
        input_path = ROOT/(PREFIX+stamp+'_input.json'); inp = json.loads(input_path.read_text())
        law = module.Law(inp['parameters'])
        output = ROOT/(PREFIX+stamp+'_frozen_coefficients.npz')
        if output.exists():
            raise FileExistsError('Frozen coefficient image is immutable')
        np.savez_compressed(output, local_vectors=np.array(law.local_vectors), local_energies=np.array(law.local_energies),
            primitive_X=law.bare_X, primitive_X2=law.bare_X2,
            cross_load_X=np.array([[law.X[a, b] for b in range(19)] for a in range(19)]),
            cross_load_overlap=np.array([[law.overlap[a, b] for b in range(19)] for a in range(19)]),
            projected_X2=np.array(law.X2), left_Gauss=np.array(law.left), right_Gauss=np.array(law.right),
            fundamental_transport=np.array([[law.U[a, b, False] for b in range(3)] for a in range(3)]),
            positions=law.positions, edges=np.array(law.edges))
        source, raw = law.baryon(0)
        local_norm = float(np.vdot(law.local_vectors[0][:, 0], law.local_vectors[0][:, 0]).real)
        descriptor = dict(kind='EXACT_FACTORISED_PRIMITIVE_SOURCE_IMAGE_BEFORE_WORD_GALERKIN', epoch=stamp,
            input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
            model_sha256=hashlib.sha256((ROOT/(PREFIX+'model.py')).read_bytes()).hexdigest(),
            coefficient_file=output.name, coefficient_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            CAR_source=[dict(canonical_key=str(key), amplitude=[float(value.real), float(value.imag)]) for key, value in source.items()],
            literal_CAR_prefactor_norm2=raw, normalized_body_norm2=float(module.inner(source, source).real),
            canonical_bare_image='Same normalized9-term CAR body times FOUR primitive local_vectors[0][:,0] tensor factors, all6 links in |0>; this is the Bdag-on-empty-sea PREPARED reference image, before conditional/word projection.',
            canonical_bare_full_norm2=float(module.inner(source, source).real*local_norm**4),
            dressed_preparation='Same body, local_vectors[3][:,0] at site0 and local_vectors[0][:,0] at other3 sites; all6 link vacua. A DISTINCT physical prepared source, not renamed bare BdagOmega.',
            scope='Full exact tensor descriptor in the declared bare32 oscillator/PW19/CAR representation. No materialized huge tensor required, no unbounded-boson or full-H closure claim.',
            environment=dict(Python=sys.version, NumPy=np.__version__, SciPy=scipy.__version__))
        path = ROOT/(PREFIX+stamp+'_canonical_source_image.json')
        path.write_text(json.dumps(descriptor, indent=2)+'\n')
        print(json.dumps(dict(epoch=stamp, coefficient_file=output.name, source_full_norm2=descriptor['canonical_bare_full_norm2'])), flush=True)


if __name__ == '__main__':
    main()
