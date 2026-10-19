import json
from dataclasses import replace
from pathlib import Path
from rich_soft_field_20261014_model import Parameters, LAW
from rich_soft_field_20261014_density import DensityJoint
from rich_soft_field_20261014_calculate import summarize


def main():
    request = json.loads(Path('rich_soft_field_20261014_density_input.json').read_text())
    p = Parameters(**{**request['parameters'], 'shape':tuple(request['parameters']['shape'])})
    results = []
    for retain in request['retained_states']:
        for n in request['q_indices']:
            one = DensityJoint(p, n, suppliers=1, retain=retain)
            result = summarize(one, f'one_supplier_retain{retain}')
            result['supplier_count'], result['retained_per_supplier'] = 1, retain
            results.append(result)
            pp = replace(p, shape=tuple(request['density_preserving_shape']))
            two = DensityJoint(pp, n, suppliers=2, retain=retain)
            result = summarize(two, f'two_supplier_density_preserved_retain{retain}', full=retain==8 and n==1)
            result['supplier_count'], result['retained_per_supplier'] = 2, retain
            result['retained_energies'] = two.retained_energies.tolist()
            results.append(result)
    result = summarize(DensityJoint(p, 1, frozen=True), 'frozen_actual_cluster_state')
    results.append(result)
    payload = {'law':LAW+'-finite-density-retained-subspace-fork', 'input':request, 'cases':results,
               'projection':'Each supplier projected to lowest r=8 or16 actual phi=0 64-state Hubbard eigenvectors. H(phi),J(phi),K(phi) are projected microscopic operators. Electric amplitude and all products of retained supplier states remain fully quantum. This is neither full64 each supplier nor a Gaussian/stiffness/D approximation.',
               'geometry':'Volume doubled along z while exact supplier count doubles, q along x unchanged. Second xy supplier cluster at z=2,3, disjoint orbitals; each two-loop cluster internally connected. Both share the normalized phi mode. Opposite field-leg copy rotates the independent matter clusters.',
               'boundary':'All run amplitudes restricted to the same one-well pi Dirichlet box. Original actual1 wide-box control exposes periodic multiwell delocalization.',
               'bulk_scope':'Two-supplier finite-density check only; no thermodynamic homogeneous nonlinear compact phase, source Gauss, multi-q packet or spin-half electron claimed.'}
    Path('rich_soft_field_20261014_density_actual1_result.json').write_text(json.dumps(payload,indent=2))


if __name__=='__main__':
    main()
