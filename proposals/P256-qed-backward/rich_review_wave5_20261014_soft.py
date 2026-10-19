"""NEW cubic-source/current consumer; finite projected modes, never bulk by label."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, expm_multiply, spsolve
from rich_soft_field_20261014_model import Joint, Parameters
from rich_review_wave5_20261014_consumer import encode, norm

ROOT=Path(__file__).resolve().parent
PREFIX='rich_review_wave5_20261014_soft'


def audit(joint):
    values,vectors=eigsh(joint.H,k=14,which='SA',tol=1e-11,v0=np.linspace(.3,1.7,joint.H.shape[0]).astype(complex))
    order=np.argsort(values); values=values[order]; vectors=vectors[:,order]
    g=vectors[:,0]; gaps=values-values[0]
    operators=[joint.F,joint.electric_leg,joint.J,joint.Rho]
    sources=np.column_stack([op@g-g*np.vdot(g,op@g) for op in operators])
    weights=np.abs(vectors.conj().T@sources)**2
    bright=np.flatnonzero((gaps>1e-7)&(weights[:,0]>1e-8))
    b=int(bright[0]); phi=len(joint.phi)
    reduced=g.reshape(phi,joint.md).conj().T@g.reshape(phi,joint.md)
    point_probability=np.sum(abs(g.reshape(phi,joint.md))**2,axis=1)
    result={'q':joint.geometry['q'],'electric_metric':joint.metric,'dimension':len(g),'gap':gaps[b],'bright_weights_flux_electric_current_density':weights[b],'ground_residual':norm(joint.H@g-values[0]*g),'source_Gram':sources.conj().T@sources,'matter_purity':np.trace(reduced@reduced).real,'outer_tenth_probability':point_probability[abs(joint.phi)>.9*joint.p.halfwidth].sum(),'gap_per_sqrt_curl':gaps[b]/np.sqrt(joint.geometry['lambda'])}
    return result,g,sources,values


def main():
    request=json.loads((ROOT/(PREFIX+'_input.json')).read_text())
    p=Parameters(**{**request['parameters'],'shape':tuple(request['parameters']['shape'])})
    cases={}; saved={}; full_model=None
    for n in request['q_indices']:
        model=Joint(p,n); result,g,sources,values=audit(model)
        cases['joint_n'+str(n)]=result
        if n==request['q_indices'][0]:
            full_model=model; full_g=g; full_sources=sources; full_values=values
        bare=Joint(p,n,no_matter=True)
        bare_result,_,_,_=audit(bare); cases['bare_n'+str(n)]=bare_result
    for label,model in [('charge_zero',Joint(p,request['q_indices'][0],charge=0.)),('loop_closure_cut',Joint(p,request['q_indices'][0],cut=True)),('diluted',Joint(replace(p,shape=(p.shape[0],2*p.shape[1],p.shape[2])),request['q_indices'][0]))]:
        result,_,_,_=audit(model); cases[label]=result
    model=full_model; g=full_g; sources=full_sources; energy=full_values[0]
    # Reconstruct the chosen 3D cochain and canonical phi conversion independently.
    volume=np.prod(p.shape); x=np.arange(p.shape[0])[:,None,None]
    profile=np.broadcast_to(np.sqrt(2/volume)*np.cos(model.geometry['q']*x),p.shape)
    dprofile=np.roll(profile,-1,axis=0)-profile
    electric_a=np.zeros((*p.shape,3)); electric_b=np.zeros_like(electric_a)
    electric_a[...,1]=-dprofile; electric_b[...,2]=dprofile
    gram=np.array([[np.vdot(a,b).real for b in [electric_a,electric_b]] for a in [electric_a,electric_b]])
    local=profile[0,0,0]; independent_metric=p.electric**2*local**2*gram[0,0]
    wrong_metric=p.electric**2*gram[0,0]
    # Spectral residue of the physical E source in the exact disjoint-copy law.
    # Htotal=H_a x I + I x H_b; source E_a g_a x g_b and vice versa.
    # Factorized contraction is exact, not a fictitious two-field source label.
    electric_source=sources[:,1]
    product_source_gram=np.eye(2)*np.vdot(electric_source,electric_source).real
    z=complex(*request['response_z']); identity=sparse.eye(len(g),format='csr')
    plus=spsolve((z+energy)*identity-model.H,sources)
    minus=spsolve((z-energy)*identity+model.H,sources)
    response=sources.conj().T@plus-(sources.conj().T@minus).T
    two_copy_response=np.eye(2,dtype=complex)*response[1,1]
    positive_only=sources.conj().T@plus
    # Fully frozen matter is a diagnostic wrong-root protocol, not installed Pi.
    hm0=model.matter.operators(0.)[0].toarray(); ev,uv=eigh(hm0); frozen=uv[:,0]
    static=np.array([np.vdot(frozen,model.matter.operators(phi)[0]@frozen).real for phi in model.phi])
    kinetic=model.HE[::model.md,::model.md]
    frozen_values,frozen_vectors=eigh(kinetic.toarray()+np.diag(static))
    frozen_gap=frozen_values[1]-frozen_values[0]
    step=1e-4; hm,j,k=model.matter.operators(.29)
    hp=model.matter.operators(.29+step)[0]; hn=model.matter.operators(.29-step)[0]
    j_residual=norm(((hp-hn)/(2*step)-j).toarray())
    k_residual=norm(((hp+hn-2*hm)/step**2-k).toarray())
    state=expm_multiply(-1j*request['pulse_eta']*model.J,g,traceA=-1j*request['pulse_eta']*model.J.diagonal().sum())
    events=[]
    for t in request['times']:
        psi=expm_multiply(-1j*t*model.H,state,traceA=-1j*t*model.H.diagonal().sum())*np.exp(1j*t*energy)
        events.append({'time':t,'electric_energy':np.vdot(psi,model.HE@psi).real,'matter_energy':np.vdot(psi,model.HM@psi).real,'actual_loop_source_current':np.vdot(psi,model.J@psi).real,'norm':np.vdot(psi,psi).real})
    result={'new_request':request,'cases':cases,'independent_3D_electric_Gram':gram,'metric_projection_residual':abs(independent_metric-model.metric),'wrong_omit_local_source_conversion_metric_ratio':wrong_metric/independent_metric,'two_disjoint_copy_physical_electric_source_Gram':product_source_gram,'two_copy_full_retarded_electric_response':two_copy_response,'full_current_field_density_retarded_response':response,'wrong_positive_only_response_distance':norm(response-positive_only),'same_parameters_frozen_matter_gap':frozen_gap,'dynamic_vs_frozen_gap_difference':cases['joint_n'+str(model.n)]['gap']-frozen_gap,'current_derivative_residual':j_residual,'contact_derivative_residual':k_residual,'current_pulse_actual_energy_exchange':events,'meaning':'Two-copy contraction exact ONLY for supplied disjoint-cluster product law. Dynamic matter response full joint H; spectrum reports finite eigsh window. J is local Wilson-flux source derivative, not full Fourier Cartesian current. Metric from chosen 3D curl/electric law; no universal continuum/bulk photon claim.'}
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(encode(result),indent=2))
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),ground=g,sources=sources,physical_electric_a=electric_a,physical_electric_b=electric_b,response=response)
    print(json.dumps(encode({'cases':{k:{a:v[a] for a in ['gap','bright_weights_flux_electric_current_density','matter_purity']} for k,v in cases.items()},'metric_residual':result['metric_projection_residual'],'frozen_gap':frozen_gap,'positive_only_distance':result['wrong_positive_only_response_distance']})))


if __name__=='__main__':
    main()
