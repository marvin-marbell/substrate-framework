"""Same NEW London-state physical/invariant supplier receiver, no re-evolution."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import eye,kron,csr_matrix
from rich_rotcov_gauge_20261015_model import build,london_profiles
from rich_rotcov_induced_20261015_calculate import mean,magnitude
from rich_rotcov_induced_20261015_model import comm


def main():
    inp=Path(sys.argv[1]);states_path=Path(sys.argv[2]);out=Path(sys.argv[3]);p=json.loads(inp.read_text());model=build(p)
    data=np.load(states_path);states=data['event_states'];baseline=data['baseline_states'];times=data['times']
    body=np.prod(model['dims'][:3]);nf=model['dims'][4];n=model['supplier']['primitive_dimension'];ident=np.eye(n)
    physical_mechanical=model['supplier_energy']+model['gauge_matter_coupling']
    canonical_transformed=model['supplier_energy'].copy()
    linear=[];quadratic=[]
    for f,q in enumerate(model['supplier_quadratures']):
        original,radial,chi,gradient=london_profiles(q['points'],p)
        linear.append(q['linear'](gradient,p['supplier_masses'][f],p['charge']))
        quadratic.append(p['charge']**2*q['integrate'](np.einsum('xai,xbi->xab',gradient,gradient))/(2*p['supplier_masses'][f]))
    def pair(a,b):return np.kron(a,ident)+np.kron(ident,b)
    def mix(op,field):return kron(eye(body),kron(csr_matrix(op),csr_matrix(field)),format='csr')
    for a in range(6):
        canonical_transformed+=mix(pair(linear[0][a],linear[1][a]),model['field']['Q'][a])
        for b in range(6):canonical_transformed+=mix(pair(quadratic[0][:,:,a,b],quadratic[1][:,:,a,b]),model['field']['QQ'][a][b])
    records=[]
    for t,v,base in zip(times,states,baseline):
        records.append(dict(time=float(t),invariant_supplier_r2=[mean(v,a) for a in model['supplier_r2']],invariant_supplier_r2_excess=[mean(v,a)-mean(base,a) for a in model['supplier_r2']],physical_minimal_mechanical_energy=mean(v,physical_mechanical),physical_minimal_mechanical_energy_excess=mean(v,physical_mechanical)-mean(base,physical_mechanical),transformed_original_canonical_bare_Hs_excess=mean(v,canonical_transformed)-mean(base,canonical_transformed),London_coordinate_bare_Hs_excess_not_physical=mean(v,model['supplier_energy'])-mean(base,model['supplier_energy']),longitudinal_cross_energy_excess=mean(v,model['longitudinal_cross'])-mean(base,model['longitudinal_cross'])))
    result=dict(kind='SAME_NEW_QUANTUM_STATE_INVARIANT_SUPPLIER_PHYSICAL_RECEIVER',states_input=str(states_path.name),parameters=p,energy_semantics={'London_coordinate_bare_Hs':'Inherited zero-field p²+V operator in London coordinates, NOT the transformed original bare observable and NOT alone physical mechanical energy','physical_minimal_mechanical':'Hs(AP)=sum (p+e AP(q))²/(2m)+V_ion+V12; ALL continuum AP² products retained. This is Udag Hs_original(A) U. Shared body-supplier Coulomb separately reported, not silently allocated.','transformed_original_canonical_bare_Hs':'Udag[p²/(2m)+V]U=sum(p-e q grad chi)²/(2m)+V, computed with continuum gradient/qq products; canonical bare p² is gauge-dependent, not called kinetic-velocity energy','r2':'Supplier position r² commutes with London U exactly; invariant same-state receiver, no energy nomenclature ambiguity'},source_factor_closure={'B_commutes_all_dipoles_max':max(magnitude(comm(b,d)) for b in model['sources'] for d in model['dipole']),'argument':'All core N share identical supplier/field/geometry fibre; U depends only on supplier coordinates and q. U_Nplus1dag B U_N=B before projection; zero intersector basis leakage by tensor factorization.'},samples=records)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(final_invariant_supplier_r2_excess=records[-1]['invariant_supplier_r2_excess'],final_physical_mechanical_energy_excess=records[-1]['physical_minimal_mechanical_energy_excess'],source_closure=result['source_factor_closure']['B_commutes_all_dipoles_max'])))


if __name__=='__main__':main()
