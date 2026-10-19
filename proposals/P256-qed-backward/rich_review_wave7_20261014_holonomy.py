"""NEW two-local-frame full-Fock spatial EM/source/current/quantum receiver."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply
from rich_holonomy_em_20261014_model import build, sector, spatial_current
from rich_frame_photon_20261014_model import spectrum, full_resolvent

ROOT=Path(__file__).resolve().parent


def norm(operator):
    return float(sp.linalg.norm(operator))


def expectation(operator,state):
    return float(np.vdot(state,operator@state).real)


def propagate(operator,state,scale):
    return expm_multiply(scale*operator,state,traceA=scale*operator.diagonal().sum())


def low_spectra(model):
    result={}
    for number in range(9):
        ix=sector(model,number)
        energy,vectors,receipt=spectrum(model['H'][ix][:,ix],count=4)
        result[number]=(ix,energy,vectors,receipt)
    return result


def main():
    inputs=json.loads((ROOT/'rich_review_wave7_20261014_holonomy_input.json').read_text())
    p=inputs['parameters'];model=build(p,external=inputs['external'])
    allspec=low_spectra(model);ix,e,v,reference_receipt=allspec[4];ground=v[:,0];reference_energy=e[0]
    source_data={};car=[];residuals=[]
    for mode in range(8):
        plus=model['source'][mode][allspec[5][0]][:,ix]
        minus=model['source'][mode].getH()[allspec[3][0]][:,ix]
        car.append(norm(plus.getH()@plus+minus.getH()@minus-sp.eye(len(ix))))
    for number,addition in ((5,True),(3,False)):
        target,energy,_,_=allspec[number]
        operators=[source if addition else source.getH() for source in model['source']]
        vectors=np.column_stack([operator[target][:,ix]@ground for operator in operators])
        z=complex(*inputs['frequency'])
        solved,errors=full_resolvent(model['H'][target][:,target],reference_energy+(z if addition else -z),vectors)
        residuals+=errors
        g=vectors.conj().T@solved
        if not addition:g=-g.T
        source_data[str(number)]={'lowest_gap':float(energy[0]-reference_energy),'total_source_weight':float(np.linalg.norm(vectors)**2),
                                  'Gram':[[str(value) for value in row] for row in vectors.conj().T@vectors],
                                  'G_full_virtual':[[str(value) for value in row] for row in g], 'resolvent_dimension':len(target)}
    q=np.array(inputs['wavevector']);nf=model['nf'];form=np.exp(-p['charge_width']**2*(q@q)/2)
    rho=sum(form*np.exp(-1j*centre@q)*sp.kron(sp.csr_matrix(charge),sp.eye(nf),format='csr') for centre,charge in zip(model['geometry']['centres'],model['rho']))
    spatial=spatial_current(model,q)
    continuity=norm(model['H']@rho-rho@model['H']+sum(q[a]*spatial[a] for a in range(3)))
    route=model['geometry']['routes'][0]
    coefficient=form*np.einsum('n,na->a',np.exp(-1j*route['middle']@q)*np.sinc(route['steps']@q/(2*np.pi)),route['steps'])
    omitted=p['charge']*sum(q[a]*coefficient[a]*model['edgeJ'][0] for a in range(3))
    wrong_continuity=norm(model['H']@rho-rho@model['H']+sum(q[a]*spatial[a] for a in range(3))-omitted)
    step=inputs['derivative_step'];derivatives=[]
    for a in range(2):
        delta=np.eye(2)[a]*step
        plus=build(p,frame=model['frame'],external=np.array(inputs['external'])+delta)
        minus=build(p,frame=model['frame'],external=np.array(inputs['external'])-delta)
        derivatives.append({'J':norm((plus['H']-minus['H'])/(2*step)+model['J'][a]),'K':norm((plus['H']+minus['H']-2*model['H'])/step**2-model['K'][a][a])})
    uniform=[]
    for a in range(3):
        delta=np.eye(3)[a]*step
        plus=build(p,frame=model['frame'],external=inputs['external'],spatial_external=delta)
        minus=build(p,frame=model['frame'],external=inputs['external'],spatial_external=-delta)
        uniform.append(norm(-(plus['H']-minus['H'])/(2*step)-spatial_current(model,np.zeros(3))[a]))
    covariant=build(p,frame=model['frame'],external=-np.array(inputs['external']))
    c=model['CF'];c_covariance=norm(c@model['H']@c.getH()-covariant['H'])
    # No spin filtering: every j in the full odd sector remains in event and G.
    target=allspec[5][0];h=model['H'][target][:,target]
    source=sum(value*operator[target][:,ix] for value,operator in zip(inputs['source_coefficients'],model['source']))@ground
    source/=np.linalg.norm(source)
    spin=sum(value*operator for value,operator in zip(inputs['spin_axis'],model['S']))[target][:,target]
    internal_spin=sum(value*operator for value,operator in zip(inputs['spin_axis'],model['matter_spin']))
    oddbits=[left*16+right for left in range(16) for right in range(16) if left.bit_count()+right.bit_count()==5]
    sval,su=eigh(internal_spin[np.ix_(oddbits,oddbits)])
    spinplus=sp.kron(su[:,sval>0]@su[:,sval>0].conj().T,sp.eye(nf),format='csr')
    internal_s2=sum(a@a for a in model['matter_spin'])
    jvals,ju=eigh(internal_s2[np.ix_(oddbits,oddbits)])
    halfproject=sp.kron(ju[:,abs(jvals-.75)<1e-8]@ju[:,abs(jvals-.75)<1e-8].conj().T,sp.eye(nf),format='csr')
    fieldcoords=np.array(model['field']['q']);fieldmask=(np.asarray(inputs['field_direction'])@fieldcoords>0).astype(float)
    fieldplus=sp.kron(sp.eye(len(oddbits)),sp.diags(fieldmask),format='csr');identity=sp.eye(len(target),format='csr')
    fieldpulse=sum(value*operator for value,operator in zip(inputs['field_direction'],model['P']))[target][:,target]
    current=sum(value*operator for value,operator in zip(inputs['current_direction'],model['J']))[target][:,target]
    incoming=propagate(fieldpulse,source,-1j*inputs['pulse']);events={}
    for label,insert in (('no_insertion',identity),('actual_current',current),('wrong_Noether_spin',spin)):
        state=insert@incoming;strength=float(np.linalg.norm(state)**2);state/=np.linalg.norm(state)
        outgoing=propagate(h,state,-1j*inputs['time'])
        probabilities=[float(np.linalg.norm(sproj@fproj@outgoing)**2) for sproj in (spinplus,identity-spinplus) for fproj in (fieldplus,identity-fieldplus)]
        events[label]={'insertion_strength':strength,'spin':expectation(spin,outgoing),'S2':expectation(model['S2'][target][:,target],outgoing),
                       'unfiltered_jhalf_weight':expectation(halfproject,outgoing),'mode_current':[expectation(operator[target][:,target],outgoing) for operator in model['J']],
                       'field':[expectation(operator[target][:,target],outgoing) for operator in model['Q']], 'probabilities_spin_field':probabilities,'sum':sum(probabilities)}
    off=build(p,frame=model['frame'],external=inputs['external'],induction=0.)
    offix=sector(off,4);offe,offv,offreceipt=spectrum(off['H'][offix][:,offix],count=4);offground=offv[:,0]
    field_energy=eigh(model['field']['kinetic'].toarray(),check_finite=False,eigvals_only=True)
    statistics=[]
    for label,which,indices,psi,energies in (('coupled',model,ix,ground,e),('induction_off',off,offix,offground,offe)):
        covariance=np.array([[expectation((a@b)[indices][:,indices],psi) for b in which['Q']] for a in which['Q']])
        statistics.append({'label':label,'Q2':covariance.tolist(),'lowest_gap':float(energies[1]-energies[0])})
    raw=model['frame'];gauss=max(float(np.linalg.norm(g@raw['P'])) for g in raw['G'])
    bare_leak=max(float(np.linalg.norm((np.eye(224)-raw['P']@raw['P'].conj().T)@d@raw['P'])) for d in raw['bare'])
    positive_frame=float(np.min(eigh(sum(s@s for s in raw['S']),eigvals_only=True)))
    rho_ops=[sp.kron(sp.csr_matrix(a),sp.eye(nf),format='csr') for a in model['rho']]
    source_car_gram=np.array([[np.vdot(model['source'][a][allspec[5][0]][:,ix]@ground,model['source'][b][allspec[5][0]][:,ix]@ground)+np.vdot(model['source'][b].getH()[allspec[3][0]][:,ix]@ground,model['source'][a].getH()[allspec[3][0]][:,ix]@ground) for b in range(8)] for a in range(8)])
    result={'kind':'NEW_TWO_LOCAL_FRAMES_FULL_FOCK_SPATIAL_EM_EVENT','inputs':inputs,
            'dimensions':{str(number):len(data[0]) for number,data in allspec.items()},'energies_N0_to8':[float(allspec[number][1][0]) for number in range(9)],
            'reference':{'N':4,'S2':expectation(model['S2'][ix][:,ix],ground),**reference_receipt},
            'source':source_data,'CAR_residuals':car,'source_Gram_CAR_error':float(np.linalg.norm(source_car_gram-np.eye(8))),
            'full_virtual_resolvent_residuals':residuals,'Gauss':gauss,'wrong_bare_source_leakage':bare_leak,'positive_local_frame_S2_minimum':positive_frame,
            'C_unitarity':norm(c.getH()@c-sp.eye(model['H'].shape[0])),'C_whole_H_source_covariance':c_covariance,
            'C_local_charge_conjugation':[norm(c@a@c.getH()+a) for a in rho_ops], 'C_frame_S_preservation':[norm(c@s@c.getH()-s) for s in model['S']],
            'spatial_continuity':continuity,'wrong_omit_scalar_bridge_continuity':wrong_continuity,'whole_H_derivatives':derivatives,'uniform_derivative_errors':uniform,
            'events':events,'source_unfiltered_jhalf_weight':expectation(halfproject,source),'field_statistics':statistics,
            'uncoupled_field_kinetic_gap':float(field_energy[1]-field_energy[0]),'field_momentum_rms':model['geometry']['momentum_rms'],
            'mode_line_Gram':(model['geometry']['line'].T@model['geometry']['line']).tolist(),
            'scope':'TWO local positive-frame physical16 factors; eight complete CAR modes, N4 filled reference and ONE added N5 charge. Whole adjacentN3/N5 resolvents retain all virtuals. All j retained; sign-spin receiver does not prove halfspin. Two spatial Gaussian field profiles, all scalar/frame paths, no supplied B/oscillator restoration. Fixed positions/global profiles/finite box, not free3D COM, helicity, locality or Lorentz. Local internal mass is supplied input.'}
    (ROOT/'rich_review_wave7_20261014_holonomy_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('source','inputs')},indent=2))


if __name__=='__main__':
    main()
