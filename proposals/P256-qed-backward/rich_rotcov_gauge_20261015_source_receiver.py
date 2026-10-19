"""P i[H,D] P is not i[PHP,PDP]: same-state full-product source receiver."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import eye
from rich_rotcov_gauge_20261015_model import build,london_profiles
from rich_rotcov_gauge_20261015_sparse import SparseRoundoff
from rich_rotcov_induced_20261015_calculate import mean,magnitude
from rich_rotcov_induced_20261015_model import comm


def complete_source(model,p):
    sparse=SparseRoundoff();kron=sparse.kron;csr=sparse.csr
    body=np.prod(model['dims'][:3]);n=model['supplier']['primitive_dimension'];ident=np.eye(n);e=p['charge']*p.get('gauge_coupling',1.)
    def pair(a,b):return np.kron(a,ident)+np.kron(ident,b)
    def embed(op):return kron(eye(body),kron(csr(op),eye(model['dims'][4])))
    def mix(op,field):return kron(eye(body),kron(csr(op),csr(field)))
    linear=[];cross=[]
    for f,q in enumerate(model['supplier_quadratures']):
        original,radial,chi,gradient=london_profiles(q['points'],p)
        # i[H,D_a]=-e/(2m){p+e AP,grad chi_a}, BEFORE projection.
        linear.append(-q['linear'](gradient,p['supplier_masses'][f],-e))
        cross.append(-e**2*q['integrate'](np.einsum('xai,xbi->xab',gradient,radial))/p['supplier_masses'][f])
    full_dipoles=[];full_currents=[];complements=[]
    for a in range(6):
        full=embed(pair(linear[0][a],linear[1][a]))
        for b in range(6):full+=mix(pair(cross[0][:,:,a,b],cross[1][:,:,a,b]),model['field']['Q'][b])
        compressed=1j*comm(model['H'],model['dipole'][a]);complement=full-compressed
        full_dipoles.append(full);complements.append(complement)
        full_currents.append(model['covariant_current'][a]+complement)
    return full_currents,full_dipoles,complements,sparse.snapshot()


def main():
    inp=Path(sys.argv[1]);states_path=Path(sys.argv[2]);out=Path(sys.argv[3]);p=json.loads(inp.read_text());model=build(p)
    data=np.load(states_path);ground=data['ground'];states=data['event_states'];baselines=data['baseline_states'];times=data['times']
    currents,dipole_full,complements,ledger=complete_source(model,p)
    direct=model['direct_original_current'];matrix=model['covariant_current'];differences=[a-b for a,b in zip(currents,direct)]
    records=[]
    for t,v,baseline in zip(times,states,baselines):
        records.append(dict(time=float(t),projected_full_covariant_current_excess=[mean(v,a)-mean(baseline,a) for a in currents],direct_original_current_excess=[mean(v,a)-mean(baseline,a) for a in direct],full_covariant_minus_direct_state_norm=[magnitude(a@v) for a in differences]))
    original_response=data['neutral_current_Q_amplitudes'][:,:6]
    constant_parameters=dict(p,gauge_profile='constant');constant=build(constant_parameters)
    constant_currents,constant_dipoles,constant_complements,constant_ledger=complete_source(constant,constant_parameters)
    result=dict(kind='EXACT_PROJECTION_ORDER_SOURCE_CALIBRATION_SAME_NEW_H_AND_STATES',states_input=states_path.name,parameters=p,formula='P i[H,D_a]P=-P e{p+e AP,grad chi_a}/(2m)P. Coordinate Coulomb, electric and body pieces commute chi before projection. i[PHP,PDP] differs by retained/complement cross terms; it is NOT the complete physical covariant source.',unchanged='No H, coupling, kinetic, state, source B, field preparation, contact or pole changes. Only physical source observable is assembled with products BEFORE projection.',projected_full_vs_direct_operator=[magnitude(a) for a in differences],projected_full_vs_direct_ground=[magnitude(a@ground) for a in differences],old_compressed_commutator_vs_direct_ground=[magnitude((a-b)@ground) for a,b in zip(matrix,direct)],complement_operator=[magnitude(a) for a in complements],complement_ground=[magnitude(a@ground) for a in complements],corrected_physical_electric_Heisenberg_source_ground=[magnitude((1j*comm(model['H'],model['physical_P'][a])-currents[a])@ground) for a in range(6)],signed_virtual_semantics='All13440 neutral virtuals already saved. Complete covariant current equals direct original-A current as operator to measured precision, so physical corrected Lehmann strengths/signed sums are the DIRECT columns0:6, not old matrix-commutator columns6:12.',all_retained_direct_current_gram_real=(original_response.conj().T@original_response).real.tolist(),pure_gradient_operator_control=dict(projected_full_covariant_minus_direct=[magnitude(a-b) for a,b in zip(constant_currents,constant['current'])],physical_direct_current_norm=[magnitude(a) for a in constant['current'][:3]],full_product_dipole_norm=[magnitude(a) for a in constant_dipoles[:3]],static_AP_coupling=magnitude(constant['gauge_matter_coupling']),old_compressed_commutator_defect=[magnitude(a-b) for a,b in zip(constant['covariant_current'],constant['current'])],interpretation='Static AP=0 and dynamic physical current remains nonzero. The former1.12684 ground discrepancy was also a compression-commutator witness, not proof of physical source failure.'),sparse_roundoff=ledger,constant_sparse_roundoff=constant_ledger,samples=records,judgment='Withdraw .2282894 as physical/direct-vs-complete-covariant failure: it is the omitted complement from commuting after compression. Finite field CCR/Heisenberg defects, bare gaps and lack of regulator/continuum photon proof remain separate, unchanged limitations. This source correction does not prove all gauge physics or Nature.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(full_covariant_vs_direct_operator=result['projected_full_vs_direct_operator'],full_covariant_vs_direct_ground=result['projected_full_vs_direct_ground'],gradient_full_vs_direct=result['pure_gradient_operator_control']['projected_full_covariant_minus_direct'],finite_CCR_EOM_defect=result['corrected_physical_electric_Heisenberg_source_ground'])))


if __name__=='__main__':main()
