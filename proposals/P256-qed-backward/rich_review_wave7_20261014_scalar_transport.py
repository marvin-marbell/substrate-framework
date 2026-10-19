"""NEW actual spatial Fock/source/scalar-field/current consumer, no Bloch replay."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from rich_scalar_transport_20261014_model import build, sector, ground
from rich_frame_photon_20261014_model import full_resolvent
from rich_review_wave7_20261014_mobile import norm, expectation, evolve

ROOT=Path(__file__).resolve().parent


def main():
    inputs=json.loads((ROOT/'rich_review_wave7_20261014_scalar_transport_input.json').read_text())
    p=inputs['parameters'];model=build(p)
    i4,h4,e4,u4,reference=ground(model,4,count=8)
    ensemble=reference/np.sqrt(reference.shape[1]);e0=e4[0]
    i3,h3,e3,_,_=ground(model,3,count=8);i5,h5,e5,_,_=ground(model,5,count=8)
    source_data={};grams=[];residuals=[]
    z=complex(*inputs['frequency'])
    for number,addition,ix,h in ((5,True,i5,h5),(3,False,i3,h3)):
        operators=[a if addition else a.getH() for a in model['source']]
        columns=[a[ix][:,i4]@ensemble for a in operators]
        vectors=np.concatenate(columns,axis=1)
        solved,errors=full_resolvent(h,e0+(z if addition else -z),vectors)
        residuals+=errors
        ng=ensemble.shape[1]
        g=np.array([[np.vdot(columns[a],solved[:,b*ng:(b+1)*ng])for b in range(8)]for a in range(8)])
        gram=np.array([[np.vdot(columns[a],columns[b])for b in range(8)]for a in range(8)])
        grams.append(gram if addition else gram.T)
        if not addition:g=-g.T
        source_data[str(number)]={'lowest_gap':float((e5 if addition else e3)[0]-e0),'G_complete':[[str(value)for value in row]for row in g], 'total_source_weight':float(np.linalg.norm(vectors)**2)}
    number=model['N'][0]
    continuity=norm(1j*(model['H']@number-number@model['H'])-model['current'][0]+model['current'][1])
    wrong_continuity=norm(1j*(model['H']@number-number@model['H'])-model['S'][0])
    step=inputs['derivative_step'];derivatives=[]
    for axis in range(2):
        delta=np.eye(2)[axis]*step
        hp=build(p,angles=np.array(p['angles'])+delta)['H'];hm=build(p,angles=np.array(p['angles'])-delta)['H']
        derivatives.append({'J':norm((hp-hm)/(2*step)+model['current'][axis]),'K':norm((hp+hm-2*model['H'])/step**2-model['contact'][axis])})
    conjugate=build(p,angles=-np.array(p['angles']))
    c=model['C'];covariance=norm(c@model['H']@c.getH()-conjugate['H'])
    s2=sum(s@s for s in model['S']);spin_commutator=max(norm(model['H']@s-s@model['H'])for s in model['S'])
    # Two independent site fields, not body-conditioned shared coordinate copies.
    field_generator=model['Q'][0]
    coherence=model['source'][0]@model['source'][4].getH()+model['source'][4]@model['source'][0].getH()
    field_independence=norm(field_generator@coherence-coherence@field_generator)
    conditioner=model['N'][0]@field_generator
    conditioner_error=norm(conditioner@coherence-coherence@conditioner)
    probability=np.sum(abs(ensemble.reshape(70,p['scalar_points'],p['scalar_points'],-1))**2,axis=(0,3))
    marginals=[probability.sum(axis=1),probability.sum(axis=0)]
    field_means=[expectation(a[i4][:,i4],ensemble)for a in model['Q']]
    field_variances=[expectation((a@a)[i4][:,i4],ensemble)for a in model['Q']]
    added=sum(value*a[i5][:,i4]for value,a in zip(inputs['source_coefficients'],model['source']))@ensemble
    added/=np.linalg.norm(added)
    # Neutral reference has global scalar spin; do not project the added source.
    source_s2=expectation(s2[i5][:,i5],added)
    total_spin=sum(value*a for value,a in zip(inputs['spin_axis'],model['S']))[i5][:,i5]
    internal_spin=sum(value*(model['frame']['S'][0][axis]+model['frame']['S'][1][axis]).toarray()for axis,value in enumerate(inputs['spin_axis']))
    bodybits=[state for state in range(256)if state.bit_count()==5]
    sv,su=eigh(internal_spin[np.ix_(bodybits,bodybits)])
    spinplus=sp.kron(su[:,sv>0]@su[:,sv>0].conj().T,sp.eye(model['nfield']),format='csr')
    ident=sp.eye(len(i5),format='csr')
    fieldmask=(np.repeat(model['phi'],p['scalar_points'])>0).astype(float)
    fieldplus=sp.kron(sp.eye(56),sp.diags(fieldmask),format='csr')
    q1=model['Q'][0][i5][:,i5]
    current=sum(value*a for value,a in zip(inputs['current_direction'],model['current']))[i5][:,i5]
    events={}
    for label,operator in (('no_pulse',None),('independent_site_field',q1),('actual_charge_current',current),('wrong_body_conditioner',model['N'][0][i5][:,i5]@q1)):
        incoming=added if operator is None else evolve(operator,added,-1j*inputs['pulse'])
        outgoing=evolve(h5,incoming,-1j*inputs['time'])
        probabilities=[float(np.linalg.norm(sproj@fproj@outgoing)**2)for sproj in (spinplus,ident-spinplus)for fproj in (fieldplus,ident-fieldplus)]
        events[label]={'site0_number':expectation(model['N'][0][i5][:,i5],outgoing),'spin':expectation(total_spin,outgoing),'S2':expectation(s2[i5][:,i5],outgoing),
                       'field_mean':[expectation(a[i5][:,i5],outgoing)for a in model['Q']], 'field_Q2':[expectation((a@a)[i5][:,i5],outgoing)for a in model['Q']],
                       'charge_current':[expectation(a[i5][:,i5],outgoing)for a in model['current']], 'probabilities_spin_scalar':probabilities,'sum':sum(probabilities)}
    controls={}
    for label,law in (('coupling_off',build(p,coupling=0.)),('hopping_off',build(p,hop=0.))):
        _,hc=sector(law,5);q=law['Q'][0][i5][:,i5]
        driven=evolve(hc,evolve(q,added,-1j*inputs['pulse']),-1j*inputs['time']);baseline=evolve(hc,added,-1j*inputs['time'])
        controls[label]={'same_prepared_initial_state':True,'site0_field_reaction':expectation(law['N'][0][i5][:,i5],driven)-expectation(law['N'][0][i5][:,i5],baseline),
                         'current_norms':[norm(a)for a in law['current']]}
    result={'kind':'NEW_TWO_CELL_COMPLETE_FOCK_DYNAMIC_QUANTUM_SCALAR_TRANSPORT','inputs':inputs,'dimensions':[len(i3),len(i4),len(i5)],
            'reference_N4_ground_multiplicity':reference.shape[1],'reference_residual':float(np.linalg.norm(h4@ensemble-e0*ensemble)),
            'source':source_data,'CAR_source_Gram_error':float(np.linalg.norm(sum(grams)-np.eye(8))), 'full_adjacent_resolvent_residuals':residuals,
            'twoedge_charge_continuity':continuity,'wrong_spin_continuity':wrong_continuity,'A_current_contact_derivatives':derivatives,
            'C_whole_H_source_covariance':covariance,'global_spin_commutator':spin_commutator,
            'reference_S2':expectation(s2[i4][:,i4],ensemble),'unfiltered_added_source_S2':source_s2,
            'scalar_field_means':field_means,'scalar_field_variances':field_variances,'scalar_marginal_asymmetries':[float(np.linalg.norm(a-a[::-1]))for a in marginals],
            'field_only_body_coherence_commutator':field_independence,'wrong_conditioner_body_coherence_commutator':conditioner_error,
            'events':events,'controls':controls,
            'meaning':'Two actual finite periodic spatial charge cells with TWO independent scalar-site field factors and positive gradient; full8mode Fock N4reference/N3N5 virtuals. Quantum scalar means0 is not a mass failure. No EM/D/current photon join, continuum freeCOM/selfbinding or Lorentz mass claim. Source state not filtered for spin.'}
    (ROOT/'rich_review_wave7_20261014_scalar_transport_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items()if key not in ('source','inputs')},indent=2))


if __name__=='__main__':
    main()
