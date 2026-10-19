"""Additional observables, canonical debt and field-decoupled/hopping-OFF control.
Consumes only this owner's actual model/state, not another producer's states.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
import numpy as np
from rich_local_frame_spatial_carrier_20261016_model import PREFIX, physical_element
from rich_local_frame_spatial_carrier_20261016_run import apply, load_state, normalize, save, step, store
ROOT=Path(__file__).resolve().parent


def load(name,model=None):
    if model is not None:return load_state(name,model)
    with (ROOT/(PREFIX+name+'.pkl')).open('rb') as f:return pickle.load(f)


def local_mean(m,psi,site,A):
    return float(physical_element(psi,m.observable(site,A),psi).real)


def pi_matrices(m):
    out=[];lookup={tuple(n):i for i,n in enumerate(m.basis)}
    for a in range(8):
        p=np.zeros((m.nf,m.nf),complex)
        for col,n in enumerate(m.basis):
            for direction in (-1,1):
                last=n.copy();last[a]+=direction
                row=lookup.get(tuple(last))
                if row is not None:p[row,col]+=1j*direction*np.sqrt((n[a]+(direction==1))/2)
        out.append(p)
    return out


def extra(m,psi):
    M=m.parameters['radial_mass'];r=m.rnodes
    result=dict(local_LEFT_Casimir=[local_mean(m,psi,i,m.g['casimir'].toarray()) for i in m.gsites],material_LEFT=[local_mean(m,psi,m.rfsite,np.kron(a.toarray(),np.eye(m.nf))) for a in m.material['L']],material_Casimir=local_mean(m,psi,m.rfsite,np.kron(m.material['casimir'].toarray(),np.eye(m.nf))),field_Pi=[local_mean(m,psi,m.rfsite,np.kron(np.eye(m.nR),p)) for p in pi_matrices(m)],field_Q2=[local_mean(m,psi,m.rfsite,np.kron(np.eye(m.nR),m.QQ[a,a])) for a in range(8)],radial_force=[])
    for v in range(4):
        terms=[m.term(1.,{m.rsites[v]:np.diag(-3*(r-.9)+3/(4*M*r**3))}),m.term(1.,{m.rsites[v]:np.diag(1/(M*r**3)),m.gsites[v]:m.g['casimir'].toarray()})]
        terms += [m.term(-.7*m.geo['beta'][s,s],{m.fsites[8*v+s]:np.diag([0.,1.])}) for s in range(8)]
        result['radial_force'].append(float(physical_element(psi,m.mpo(terms),psi).real))
    return result


def main():
    m=load('model');result={'extra':{},'finite_cut_canonical_debt':{},'physical_source_joint':{}}
    for branch in ('neutral','addition','removal','addition_field_kick','addition_unpolarized'):
        result['extra'][branch]=[extra(m,load(branch+'_t'+str(k),m)) for k in range(3)]
    psi=load('addition_t2',m);Hpsi,quality=apply(m.H,psi,20)
    result['H_image']=quality
    result['H_image']['meaning']='Compressed actual full-H image, not exact uncompressed norm/variance.'
    Pi=pi_matrices(m);images={}
    for a in (0,4):
        source=m.lab_source(a);contact=m.lab_source(a,a)
        jp,jquality=apply(source,psi,20);images[a]=jp;store('physical_J'+str(a)+'_image',jp)
        pop=m.observable(m.rfsite,np.kron(np.eye(m.nR),Pi[a]));qop=m.observable(m.rfsite,np.kron(np.eye(m.nR),m.Q[a]))
        pp,_=apply(pop,psi,20);qp,_=apply(qop,psi,20)
        force=2*float(pp.overlap(Hpsi).imag);velocity=2*float(qp.overlap(Hpsi).imag)
        J=float(physical_element(psi,source,psi).real);P=float(physical_element(psi,pop,psi).real)
        result['finite_cut_canonical_debt'][str(a)]=dict(actual_dPi_dt=force,raw_physical_source_expectation=J,dPi_minus_J=force-J,actual_dQ_dt=velocity,raw_Pi_over_M=P/m.parameters['electric_inertia'],dQ_minus_Pi_over_M=velocity-P/m.parameters['electric_inertia'],source_image_quality=jquality,contact=float(physical_element(psi,contact,psi).real),meaning='Actual projected commutator versus projection of raw canonical law; differences include finite-shell boundary and disclosed compressed-image errors, not extra physical terms.')
    G=np.array([[images[a].overlap(images[b]) for b in (0,4)] for a in (0,4)])
    mixed=m.lab_source(0,4)
    result['physical_source_joint']=dict(mode_indices=[0,4],complex_Gram_real=G.real.tolist(),complex_Gram_imag=G.imag.tolist(),mixed_contact=float(physical_element(psi,mixed,psi).real),meaning='Nonparallel-q physical-current insertion joint covariance on one actual charged state, not a response pole or two independent models.')
    # Explicit norm-metadata discriminator on this actual source image.
    value=physical_element(images[0],m.H,psi);scaled=images[0].copy();scaled.norm*=3.25
    check=physical_element(scaled,m.H,psi)
    result['mixed_norm_metric']=dict(scale=3.25,mixed_H_real=value.real,mixed_H_imag=value.imag,scaled_H_real=check.real,scaled_H_imag=check.imag,scaling_error=abs(check-3.25*value),no_variance_claim=True)
    rotated=images[0].copy();angle=.37
    rotated.set_B(0,rotated._B[0]*np.exp(1j*angle),form=rotated.form[0])
    phased=physical_element(rotated,m.H,psi)
    result['mixed_norm_metric'].update(physical_tensor_phase_angle=angle,phase_covariance_error=abs(phased-np.exp(-1j*angle)*value),overlap_phase_error=abs(rotated.overlap(psi)-np.exp(-1j*angle)*images[0].overlap(psi)),phase_storage='Physical tensors; norm metadata realpositive, never a complex phase container.')
    # Explicit field-decoupled/hopping-disabled control. R kinetic remains:
    # physical x(R) still moves, so this is NOT literally all charge motion
    # frozen and its zero A derivative is NOT a material-current claim.
    terms=[]
    for coefficient,factors in m.terms:
        rf=next((name for name,i in factors if i==m.rfsite),None)
        key=next((k for k,v in m.ops[m.rfsite].items() if v==rf),None) if rf else None
        if key is not None and key in m.rfrecipes:continue
        terms.append((coefficient,factors))
    fullH=m.H;fullterms=m.terms;m.terms=terms;m.H=m.mpo(terms)
    controls={}
    for branch in ('neutral','addition','removal','addition_field_kick'):
        initial=load(branch+'_t0',m);out=initial.copy();ledgers=[]
        for k in range(2):
            out,ledger=step(m,out,.04,12);ledgers.append(ledger)
        controls[branch]=dict(initial=m.measure(initial),final=m.measure(out),ledgers=ledgers)
        store('field_decoupled_hopping_OFF_'+branch,out)
    result['field_decoupled_hopping_OFF']=dict(terms=len(terms),branches=controls,carrier_field_kick_difference={key:(np.asarray(controls['addition_field_kick']['final'][key])-np.asarray(controls['addition']['final'][key])).tolist() for key in ('n_cells','local_LEFT','local_r2','field_Q')},meaning='Field-decoupled/hopping-disabled: all microscopic Peierls hopping and convection A-couplings/contact removed, but R kinetic RETAINED and physical x=Rxbody still rotates. Field/body Hamiltonian factorizes; bare field echo subtraction valid. d_A H_control=0 is NOT a physical lab material-current or all-charge-motion-zero claim. [Hcontrol,Nv]=[Hcontrol,Lgv]=0; same RK2/bond as coupled case.')
    full=json.loads((ROOT/(PREFIX+'result.json')).read_text())
    calibrated={}
    for key in ('n_cells','local_LEFT','local_r2','field_Q'):
        f=np.asarray(full['field_to_carrier'][-1][key])-np.asarray(full['field_to_carrier'][0][key])
        o=np.asarray(controls['addition_field_kick']['final'][key])-np.asarray(controls['addition']['final'][key])
        o-=np.asarray(controls['addition_field_kick']['initial'][key])-np.asarray(controls['addition']['initial'][key])
        calibrated[key]=dict(full_dynamical=f.tolist(),OFF_dynamical=o.tolist(),full_minus_OFF=(f-o).tolist())
    result['dynamical_field_kick_calibrated']=calibrated
    result['dynamical_carrier_to_field_calibrated']={}
    for branch in ('addition','removal'):
        f=np.asarray(full['trajectories'][branch]['excess_matched_background'][-1]['field_Q'])-np.asarray(full['trajectories'][branch]['excess_matched_background'][0]['field_Q'])
        o=np.asarray(controls[branch]['final']['field_Q'])-np.asarray(controls['neutral']['final']['field_Q'])
        o-=np.asarray(controls[branch]['initial']['field_Q'])-np.asarray(controls['neutral']['initial']['field_Q'])
        result['dynamical_carrier_to_field_calibrated'][branch]=dict(full_dynamical=f.tolist(),OFF_dynamical=o.tolist(),full_minus_OFF=(f-o).tolist())
    m.H=fullH;m.terms=fullterms
    save('analysis',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
