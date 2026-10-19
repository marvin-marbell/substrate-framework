"""NEW literal CAR24/C-related removals, selected paircut gap and physical receiver."""
from itertools import product
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh, expm
from rich_colour_sea_20261014_model import build, law, mode, string, project
from rich_colour_composite_20261014_model import epsilon

ROOT=Path(__file__).resolve().parent
PREFIX='rich_review_wave10_20261014_sea'


def mean(op,state):
    return float(np.vdot(state,op@state).real)


def completed(model,name,angle,parameters=None):
    p=model['parameters'] if parameters is None else parameters
    scalar=model['scalar'];ops=model['operators']
    if name=='neutral':
        hp,jp,kp=law(ops[name],scalar,p,angle);hm,jm,km=law(ops[name],scalar,p,-angle);c=model['Cneutral']
        return (hp+c@hm@c.conj().T)/2,(jp-c@jm@c.conj().T)/2,(kp+c@km@c.conj().T)/2
    hp,jp,kp=law(ops['baryon'],scalar,p,angle);hm,jm,km=law(ops['antibaryon'],scalar,p,-angle);c=model['Cbaryon']
    if name=='baryon':
        return (hp+c.conj().T@hm@c)/2,(jp-c.conj().T@jm@c)/2,(kp+c.conj().T@km@c)/2
    hb,jb,kb=completed(model,'baryon',-angle,p)
    return c@hb@c.conj().T,-c@jb@c.conj().T,c@kb@c.conj().T


def literal_hole_source(model,spin):
    neutral=model['sectors']['neutral'];anti=model['sectors']['antibaryon'];lookup={mask:i for i,mask in enumerate(anti['masks'])}
    amplitudes={(0,0,1):np.sqrt(2/3),(0,1,0):-np.sqrt(1/6),(1,0,0):-np.sqrt(1/6)} if spin==.5 else {(0,0,1):1/np.sqrt(3),(0,1,0):1/np.sqrt(3),(1,0,0):1/np.sqrt(3)}
    rows=[];cols=[];data=[]
    for column,mask in enumerate(neutral['masks']):
        for colours in product(range(3),repeat=3):
            colour_sign=epsilon(*colours)
            if not colour_sign:
                continue
            for spins,value in amplitudes.items():
                operations=[(mode(f,1,1-s,c),False) for f,s,c in zip((0,0,1),spins,colours)]
                sign=np.prod([1 if s==0 else -1 for s in spins])
                result=string(mask,operations)
                if result is not None and result[0] in lookup:
                    rows.append(lookup[result[0]]);cols.append(column);data.append(colour_sign*value*sign*result[1]/np.sqrt(6))
    raw=sp.csr_matrix((data,(rows,cols)),shape=(len(anti['masks']),len(neutral['masks'])))
    projected=anti['P'].conj().T@sp.kron(raw,sp.eye(19),format='csr')@neutral['P']
    return np.kron(projected.toarray(),np.eye(len(model['scalar']['grid'])))


def charge_operator(model,name):
    sector=model['sectors'][name]
    charges=[]
    for mask in sector['masks']:
        nu=(mask&((1<<12)-1)).bit_count();nd=(mask>>12).bit_count()
        charges.append((2*(nu-6)-(nd-6))/3)
    return np.kron(project(sector,sp.kron(sp.diags(charges),sp.eye(19))),np.eye(len(model['scalar']['grid'])))


def projector_groups(op):
    e,v=eigh(op);groups=[]
    for value in sorted(set(np.rint(e*12).astype(int))):
        ix=abs(e-value/12)<1e-7
        groups.append((value/12,v[:,ix]@v[:,ix].conj().T))
    return groups


def receiver(model,name,state):
    op=model['operators'][name];nf=len(model['scalar']['grid']);tensor=state.reshape(op['S2'].shape[0],nf)
    pair=op['positive']-(3*np.eye(len(op['positive'])) if name=='baryon' else 0)
    output=[]
    for pv,pp in projector_groups(pair):
        for lv,lp in projector_groups(op['C']):
            for sv,ps in projector_groups(op['S2']):
                transformed=pp@lp@ps@tensor
                for field,label in [(model['scalar']['grid']<0,'negative'),(abs(model['scalar']['grid'])<1e-12,'zero'),(model['scalar']['grid']>0,'positive')]:
                    output.append(dict(pair_number=pv,loop_C=lv,S2=sv,phi_sign=label,probability=float(np.sum(abs(transformed[:,field])**2))))
    return dict(outcomes=output,sum=sum(row['probability'] for row in output),norm=float(np.vdot(state,state).real),meaning='Actual commuting pair-count x retainedloopCasimir x physicalmaterialS2 x quantumphi-sign inclusive PVM, every remaining state traced.')


def main():
    p=json.loads((ROOT/(PREFIX+'_input.json')).read_text());m=build(p);nf=len(m['scalar']['grid']);spectra={};records={}
    for name,(h,j,k) in m['laws'].items():
        e,v=eigh(h);spectra[name]=(e,v);sector=m['sectors'][name];op=m['operators'][name]
        positive_counts=[sum(bool(mask&(1<<mode(f,0,s,c))) for f,s,c in product(range(2),range(2),range(3))) for mask in sector['masks']]
        negative_holes=[12-sum(bool(mask&(1<<mode(f,1,s,c))) for f,s,c in product(range(2),range(2),range(3))) for mask in sector['masks']]
        q=charge_operator(m,name)
        records[name]=dict(number_values=sorted(set(mask.bit_count() for mask in sector['masks'])),charge_values=sorted(set(np.rint(np.linalg.eigvalsh(q)*6).astype(int)/6)),
                           raw_CAR_mask_count=len(sector['masks']),Gauss_dimension=sector['dimension'],dimension=len(e),Gauss_error=sector['gauge_error'],
                           positive_occupation_values=sorted(set(positive_counts)),negative_hole_values=sorted(set(negative_holes)),
                           energy=float(e[0]),S2_ground=mean(np.kron(op['S2'],np.eye(nf)),v[:,0]),
                           current_norm=float(np.linalg.norm(j)),contact_norm=float(np.linalg.norm(k)),
                           paircut_boundary_probability=mean(np.kron(op['positive']-(3*np.eye(sector['dimension']) if name=='baryon' else 0),np.eye(nf)),v[:,0]),
                           phi=mean(np.kron(np.eye(sector['dimension']),m['scalar']['phi']),v[:,0]),phi2=mean(np.kron(np.eye(sector['dimension']),m['scalar']['phi']@m['scalar']['phi']),v[:,0]))
    en,vn=spectra['neutral'];g=vn[:,0];eg=float(en[0]);cn=m['Cneutral'];c=m['Cbaryon'];source={};saved={'neutral_reference':g}
    for spin in (.5,1.5):
        key=str(spin);b=m['sources'][key];a=literal_hole_source(m,spin)
        source[key]=dict(literal_hole_operator_C_sewing_error=float(np.linalg.norm(a-m['anti_sources'][key])),baryon={},antibaryon={})
        for name,operator in [('baryon',b),('antibaryon',a)]:
            h,j,k=m['laws'][name];e,v=spectra[name];state=operator@g;amplitude=v.conj().T@state;weights=abs(amplitude)**2
            bright=np.flatnonzero(weights>1e-10);first=int(bright[0]);pole=abs(e-e[first])<1e-8;residue=float(sum(weights[pole]));total=float(sum(weights))
            source[key][name]=dict(actual_source_charge_commutator=float(np.linalg.norm(charge_operator(m,name)@operator-operator@charge_operator(m,'neutral')-(1 if name=='baryon' else -1)*operator)),
                                   lowest_source_bright_gap=float(e[first]-eg),pole_residue=residue,source_norm=total,normalized_pole_weight=residue/total,
                                   spectral_sum_error=float(abs(total-np.vdot(state,state).real)),all_virtual_states=len(e),all_gaps=(e-eg).tolist(),all_weights=weights.tolist(),
                                   source_S2_residual=float(np.linalg.norm(np.kron(m['operators'][name]['S2'],np.eye(nf))@state-spin*(spin+1)*state)))
            if spin==.5:
                reference=v[:,pole]@amplitude[pole];reference/=np.linalg.norm(reference);saved[name+'_source_bright_reference']=reference
            saved[name+'_literal_source_'+key]=state
    ha,ja,ka=m['laws']['antibaryon'];hb,jb,kb=m['laws']['baryon'];h0,j0,k0=m['laws']['neutral']
    step=2e-5;derivatives={};C_at_nonzero={}
    for name,(h,j,k) in m['laws'].items():
        hp=completed(m,name,step)[0];hm=completed(m,name,-step)[0]
        derivatives[name]=dict(current=float(np.linalg.norm((hp-hm)/(2*step)+j)),contact=float(np.linalg.norm((hp-2*h+hm)/step**2-k)))
    angle=.27
    C_at_nonzero['H']=float(np.linalg.norm(completed(m,'antibaryon',-angle)[0]@c-c@completed(m,'baryon',angle)[0]))
    C_at_nonzero['J']=float(np.linalg.norm(completed(m,'antibaryon',-angle)[1]@c+c@completed(m,'baryon',angle)[1]))
    rawhb,rawjb,rawkb=law(m['operators']['baryon'],m['scalar'],p)
    rawha,rawja,rawka=law(m['operators']['antibaryon'],m['scalar'],p)
    wrongly_even_current=(rawjb+c.conj().T@rawja@c)/2
    z=p['frequency']+1j*p['eta'];events={};responses={}
    for name in ('baryon','antibaryon'):
        h,j,k=m['laws'][name];e,v=spectra[name];ref=saved[name+'_source_bright_reference'];energy=mean(h,ref);gap=e-energy
        columns=j@ref;amplitude=v.conj().T@columns
        spectral=np.sum(abs(amplitude)**2*(1/(z-gap)-1/(z+gap)))
        xp=np.linalg.solve((z+energy)*np.eye(len(h))-h,columns);xm=np.linalg.solve((z-energy)*np.eye(len(h))+h,columns)
        direct=np.vdot(columns,xp)-np.vdot(columns,xm)
        responses[name]=dict(actual_reference_energy=energy,all_retained_current_virtual_states=len(e),actual_current_residue=float(np.vdot(columns,columns).real),
                             retarded_direct_error=float(abs(direct-spectral)),wrong_positive_only_error=float(abs(spectral-np.vdot(columns,xp))),
                             contact=mean(k,ref),NoEther_not_current=float(np.linalg.norm(j-.5*np.eye(len(h)))))
        op=m['operators'][name];obs={'pair':np.kron(op['positive']-(3*np.eye(len(op['positive'])) if name=='baryon' else 0),np.eye(nf)),
             'S2':np.kron(op['S2'],np.eye(nf)),'loop_C':np.kron(op['C'],np.eye(nf)),'octet':np.kron(op['octet'],np.eye(nf)),
             'phi':np.kron(np.eye(len(op['S2'])),m['scalar']['phi']),'current':j}
        scalar_u=np.kron(np.eye(len(op['S2'])),expm(-1j*p['field_kick']*m['scalar']['P']))
        phi_p=np.kron(np.eye(len(op['S2'])),m['scalar']['P']);wrong_phi_p=phi_p@obs['pair']
        mirror=np.kron(sum(d+d.conj().T for d,_,_ in op['transfers']),np.eye(nf))
        external=expm(-1j*p['pulse_duration']*completed(m,name,p['pulse_angle']*(1 if name=='baryon' else -1))[0])@ref
        scalar_seed=scalar_u@ref;wrong_seed=expm(-1j*p['field_kick']*wrong_phi_p)@ref
        events[name]=dict(scalar_only_body_coherence_commutator=float(np.linalg.norm(phi_p@mirror-mirror@phi_p)),wrong_body_conditioner_commutator=float(np.linalg.norm(wrong_phi_p@mirror-mirror@wrong_phi_p)),rows=[])
        for time in p['times']:
            row={'time':time,'states':{}}
            for label,initial in [('reference',ref),('external_EM_Wilson_probe',external),('quantum_scalar_FIELD_only',scalar_seed),('wrong_pair_conditioned_scalar',wrong_seed)]:
                state=v@(np.exp(-1j*time*e)*(v.conj().T@initial))
                row['states'][label]=dict(**{key:mean(operator,state) for key,operator in obs.items()},inclusive=receiver(m,name,state))
            events[name]['rows'].append(row)
        saved[name+'_scalar_initial']=scalar_seed;saved[name+'_EM_initial']=external
    decoupled={name:float(eigh(completed(m,name,0.,dict(p,scalar_coupling=0.))[0],eigvals_only=True)[0]) for name in m['laws']}
    decoupled['baryon_gap']=decoupled['baryon']-decoupled['neutral'];decoupled['antibaryon_gap']=decoupled['antibaryon']-decoupled['neutral']
    result=dict(kind='NEW_LITERAL_CAR24_SELECTED_PAIRCUT_QUANTUM_GAP_C_AND_CURRENT_AUDIT',parameters=p,frame=m['frame']['derivation'],sectors=records,
                source_data=source,C=dict(neutral_unitarity=float(np.linalg.norm(cn.conj().T@cn-np.eye(len(cn)))),charge_flip=float(np.linalg.norm(charge_operator(m,'antibaryon')@c+c@charge_operator(m,'baryon'))),
                                        neutral_reference_projector_leakage=float(np.linalg.norm(cn@g-g*np.vdot(g,cn@g))),H=float(np.linalg.norm(ha@c-c@hb)),J=float(np.linalg.norm(ja@c+c@jb)),K=float(np.linalg.norm(ka@c-c@kb)),raw_path=m['raw_C_defects'],nonzero_external=C_at_nonzero),
                positive_selected_number_gaps={name:records[name]['energy']-eg for name in ('baryon','antibaryon')},
                actual_H_A_derivatives=derivatives,wrong_C_even_current_norm=float(np.linalg.norm(wrongly_even_current)),actual_baryon_current_norm=float(np.linalg.norm(jb)),
                responses=responses,events=events,decoupled_quantum_scalar_control=decoupled,
                scope='One actual CAR24 occupied N12 and N15/N9 C-related creation/removal sectors. Paircut1 is variational: neutral/B have <=1 extra hole, anti <=1 extra particle; boundary mean~1/2 is not full allN vacuum/Dirac stability. Physical scalar FIELD unitary versus external classical EM Wilson pulse and conditional source poles distinct. All retained virtuals/current/contact, no additive3m/recentering, no photon/freeCOM/QCD/species result; primitive frame/multiplicity/CAR/flavour/charges/loopGalerkincut and scalar law remain inputs.')
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(ROOT/(PREFIX+'_states.npz'),**saved)
    print(json.dumps({key:result[key] for key in ('positive_selected_number_gaps','C','actual_H_A_derivatives','wrong_C_even_current_norm','actual_baryon_current_norm','responses','decoupled_quantum_scalar_control')},indent=2))


if __name__=='__main__':
    main()
