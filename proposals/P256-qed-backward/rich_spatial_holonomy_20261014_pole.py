"""Resolve actual degenerate spatial poles: source spin and flux branch inertia."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import eye
from rich_spatial_holonomy_20261014_model import build, momentum_basis, fourier_source
from rich_spatial_holonomy_20261014_calculate import norm, expect, evolve


def branch(model,number):
    p=model['parameters']; u=momentum_basis(model,number,0)
    h=(u.conj().T@model['H']@u).toarray(); e,v=eigh(h,check_finite=False)
    zero=np.flatnonzero(abs(e-e[0])<1e-8); excited=np.flatnonzero(e-e[0]>1e-8)
    b=v[:,zero]; w=-p['charge']/p['sites']*sum(model['edgeJ'])
    contact=-p['charge']**2/p['sites']**2*sum(link+link.conj().T for link in model['links'])
    wb=u.conj().T@(w@(u@b)); first=b.conj().T@wb
    a=v[:,excited].conj().T@wb
    second=b.conj().T@(u.conj().T@(contact@(u@b)))+2*a.conj().T@(a/(e[0]-e[excited])[:,None])
    slopes,rotation=eigh(first,check_finite=False); rows=[]
    seen=set()
    for i,slope in enumerate(slopes):
        if i in seen: continue
        group=np.flatnonzero(abs(slopes-slope)<1e-9); seen.update(group.tolist())
        basis=rotation[:,group]
        curvatures,r=eigh(basis.conj().T@second@basis,check_finite=False)
        for j,c in enumerate(curvatures):
            state=u@(b@(basis@r[:,j])); rows.append({'dE_dphi':float(slope),'d2E_dphi2_all_virtual':float(c),'S2':expect(model['S2'],state),'field_boundary_probability':float(np.sum(abs(state.reshape(model['matter'],model['nf']))[:,[0,-1]]**2))})
    return {'N':number,'m':0,'E':float(e[0]),'degeneracy':len(zero),'smooth_branches':rows,'ground_full':u@b,'all_virtual_states':len(excited),'U':u,'e':e,'v':v}


def run(p,primary):
    model=build(p); sea=2*p['sites']; data={n:branch(model,n) for n in (sea-1,sea,sea+1)}
    reference=data[sea]['ground_full']; ground=reference[:,0]; refcurve=data[sea]['smooth_branches'][0]['d2E_dphi2_all_virtual']; length=model['geometry']['circumference']
    s2=model['S2']; ident=eye(model['H'].shape[0],format='csr')
    # All odd N5/N7 spin irreps are 1/2,3/2,5/2: exact polynomial receiver.
    half=(s2-3.75*ident)@(s2-8.75*ident)/24.
    poles=[]
    for n in (sea-1,sea+1):
        low=data[n]['ground_full']; d=fourier_source(model,0)
        source=d if n>sea else [a.conj().T for a in d]
        columns=np.column_stack([a@ground for a in source]); amp=low.conj().T@columns
        density=amp@amp.conj().T; weight=float(np.trace(density).real)
        source_spin=float(np.trace(density@(low.conj().T@(s2@low))).real/weight)
        source_half=float(np.trace(density@(low.conj().T@(half@low))).real/weight)
        rows=[]
        for b in data[n]['smooth_branches']:
            curvature=(b['d2E_dphi2_all_virtual']-refcurve)*length**2/p['charge']**2
            rows.append({**b,'excess_inverse_inertia_smooth_branch':curvature,'conditional_band_inertia':1/curvature if curvature>0 else None,'mechanical_group_velocity':-length/p['charge']*b['dE_dphi']})
        poles.append({'N':n,'m':0,'gap_same_occupied_reference':data[n]['E']-data[sea]['E'],'lowest_degeneracy':low.shape[1],'full_source_pole_weight':weight,'source_weighted_S2':source_spin,'source_weighted_physical_jhalf_receiver':source_half,'smooth_branches':rows,'all_excited_states_in_same_m_used_in_curvature':data[n]['all_virtual_states']})
    states=np.load(primary); seed=states['source_packet']; indices=states['N7_indices']; kicked=states['field_kicked_N7']; hc=model['H'][indices][:,indices]
    event={}
    for label,initial in [('undriven',seed[indices]),('field_only_pulse',kicked)]:
        event[label]=[]
        for t in p['times']:
            local=evolve(hc,initial,t); whole=np.zeros(model['H'].shape[0],complex); whole[indices]=local
            event[label].append({'time':t,'physical_jhalf_inclusive_probability':expect(half,whole),'S2':expect(s2,whole)})
    f=model['frame']; adj=[]
    for s in f['S']:
        matrix=np.array([[np.trace(f['source'][b].conj().T@(s@f['source'][a]-f['source'][a]@s))/np.trace(f['source'][b].conj().T@f['source'][b]) for b in range(4)] for a in range(4)])
        adj.append(matrix)
    source_half_error=norm(sum(a@a for a in adj)-.75*np.eye(4))
    result={'candidate':p['candidate'],'pole_consumer':'NEW exact degenerate perturbation using physical flux current/contact and every same-m virtual state; actual full Fock reference, NOT one-cell Fourier ratio','reference':{k:v for k,v in data[sea].items() if k not in ('ground_full','U','e','v')},'charged_poles':poles,'source_adjoint_spin_casimir_error':source_half_error,'physical_spin_full_H_commutators':[norm(model['H']@s-s@model['H']) for s in model['S']],'physical_S2_full_H_commutator':norm(model['H']@s2-s2@model['H']),'source_packet_half_probability':expect(half,seed),'events':event,'inertia_meaning':'Finite-ring analytic eigenbranch flux curvature after SAME neutralreference subtraction. Nonzero opposite group velocities at m0 mean no rest minimum there; lowest-energy envelope has cusp, whose negative finite-difference quotient is NOT a mass. Reciprocal positive branch curvature is local band inertia only, not freeCOM/Lorentz mass.','spin_meaning':'Source transforms in actual raw LEFT-frame adjoint half representation, but occupied reference and charged pole are not pure totalframehalfspin. Inclusive receiver records real leakage without projection or gamma repair.','scope':p['scope']}
    return result

if __name__=='__main__':
    inp=Path(sys.argv[1]); primary=Path(sys.argv[2]); output=Path(sys.argv[3])
    result=run(json.loads(inp.read_text()),primary)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('reference','charged_poles','source_adjoint_spin_casimir_error','source_packet_half_probability','physical_S2_full_H_commutator')}))
