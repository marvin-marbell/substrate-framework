"""Actual source/density/field/multiplication support on THIS extended-H state.
Component complements are exact within stated factors; never relabelled full-H closure.
"""
import hashlib
import json
from pathlib import Path
import resource
import sys
import numpy as np
from scipy import sparse
from rich_spatial_colour_em_20261015_snapshot import load,source,matrix
from rich_spatial_colour_em_20261015_model import observable,FactorOperator
from rich_spatial_colour_em_20261015_fields import field_shell,phase_kernel


def expectation(state,op):return float(np.vdot(state,op@state).real)


def main():
    inp,out=map(Path,sys.argv[1:3]);request=json.loads(inp.read_text());root=inp.parent;state_json=root/request['state_json'];state_npz=root/request['state_npz'];metadata,data,model,laws=load(state_json,state_npz)
    base=model['base'];p=model['parameters'];field=model['field'];rotor=base['rotor'];ne=field['dimension'];nr=rotor['dimension'];nf=len(base['scalar']['grid'])
    extended=field_shell(p['em_cut']+2,p['em_basis_scale']);lookup={tuple(n):index for index,n in enumerate(extended['basis'])}
    slots=[lookup[tuple(n)] for n in field['basis']];embedding=np.eye(extended['dimension'])[:,slots]
    electric_image=extended['H_E']@embedding-embedding@field['H_E'];electric_metric=electric_image.conj().T@electric_image
    results={}
    for case in request['cases']:
        name=case['name'];key=case.get('key');label=name if key is None else name+'_'+key;op=base['operators'][name]
        if key is None:state=data['neutral_reference']
        else:
            raw=source(data,model,name,key)@data['neutral_reference'];state=raw/np.linalg.norm(raw)
        density_metric=(op['density_delta2']-op['density_delta']@op['density_delta'])/4
        local=expectation(state,observable(model,name,matter=density_metric))
        density_operator_norm=float(sparse.linalg.norm(density_metric))
        electric=expectation(state,observable(model,name,rq=sparse.kron(sparse.eye(nr),electric_metric)))
        multiplications=[]
        for edge in range(2):
            for charge in [2/3,-1/3,-1.]:
                for spin in [None,0,1,2]:
                    kernel=phase_kernel(rotor,base['geometry'],model['coefficients'],field,edge,charge,spin_axis=spin)
                    multiplication=FactorOperator(op['Q'].shape[0],base['scalar'],nr*ne,[(sparse.eye(op['Q'].shape[0]),kernel,False)],hermitian=False)
                    retained=float(np.linalg.norm(multiplication@state)**2)
                    if spin is None:full=float(np.linalg.norm(state)**2)
                    else:
                        r=rotor['integrate'](base['geometry']['tangents'][:,edge,spin]**2)
                        full=expectation(state,observable(model,name,rq=sparse.kron(r,sparse.eye(ne))))
                    multiplications.append(dict(edge=edge,physical_charge=charge,spin_axis=spin,full_multiplication_norm_squared=full,retained_norm_squared=retained,omitted_norm_squared=full-retained))
        source_result={}
        if key is not None:
            insertion=source(data,model,name,key);raw=insertion@data['neutral_reference'];metric=matrix(data,name+'_source_'+key+'_full_metric')
            source_result=dict(full_strength=expectation(data['neutral_reference'],observable(model,'neutral',matter=metric)),retained_strength=float(np.linalg.norm(raw)**2))
        results[label]=dict(local_EM_density_omitted_norm_squared=local,local_density_complement_Frobenius_norm=density_operator_norm,primitive_Pi_squared_exact_omitted_norm_squared=electric,precut_Peierls_spin_multiplication_complements=multiplications,source=source_result,meaning='Density complement uses FULL raw D². Electric complement reaches every state Pi² can reach beyond the chosen field shell. Phase/spin complements use exact continuous-Q unitary multiplication and full R products, BEFORE either cut, but hold the native body basis fixed. These are exact component supports, NOT a total nonlinear H omitted-norm certificate, convergence estimate or full pair/photon probability.')
        canonical=[field['Q'][mu]@field['Pi'][mu]-field['Pi'][mu]@field['Q'][mu]-1j*np.eye(ne) for mu in range(6)]
        results[label]['finite_field_CCR_diagonal_defect_norms']=[float(np.linalg.norm(observable(model,name,rq=sparse.kron(sparse.eye(nr),a))@state)) for a in canonical]
        results[label]['finite_field_coordinate_commutator_norms']=[dict(mu=mu,nu=nu,norm=float(np.linalg.norm(observable(model,name,rq=sparse.kron(sparse.eye(nr),field['Q'][mu]@field['Q'][nu]-field['Q'][nu]@field['Q'][mu]))@state))) for mu in range(6) for nu in range(mu+1,6)]
        results[label]['canonical_scope']='Electric Gram normalization is exact BEFORE projection. Finite total-shell Q/Pi matrices cannot obey the full CCR or mutually commuting coordinate algebra. These measured cut defects do not alter the physical current/contacts and are not new forces. Receiver uses ONE continuous-coordinate sign effect with body/phi effects, not a false joint sharp PVM of all compressed Q matrices.'
    payload=dict(kind='OWN_EXTENDED_STATE_OMITTED_SUPPORT_NOT_CLOSURE_PROOF',request=request,input_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [state_json,state_npz]},results=results,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload),flush=True)


if __name__=='__main__':main()
