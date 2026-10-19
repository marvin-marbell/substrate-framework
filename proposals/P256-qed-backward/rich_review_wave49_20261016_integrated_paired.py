"""Own calibrated local-carrier field response, frozen operator input and dt/halfdt."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.linalg import expm
import rich_local_frame_spatial_carrier_20261016_model as law
import rich_spatial_frame_photon_20261016_fields as fields
from rich_review_wave49_20261016_integrated_consumer import advance,choose_terms,measure,save_state,load_compiled,norm2,ADVANCE_LEDGERS,restore,source_operator

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'




def kick(m,psi,coefficients):
    out=psi.copy();unitary=np.kron(np.eye(m.nR),expm(1j*sum(a*q for a,q in zip(coefficients,m.Q))))
    out.apply_local_op(m.rfsite,m.register(m.rfsite,unitary),unitary=True,renormalize=False)
    out.canonical_form(renormalize=False)
    return out


def all_mode_momenta(m,psi):
    basis=[tuple(v) for v in m.basis];values=[]
    for axis in range(8):
        operator=m.observable(m.rfsite,np.kron(np.eye(m.nR),fields.primitive(basis,basis,axis,'p').toarray()))
        values.append(float(law.physical_element(psi,operator,psi).real/norm2(psi)))
    return values


def transport_resolution(m,raw,previous):
    observers={}
    for label in ('coarse','fine'):
        state=restore(m,raw,'neutral_'+label);state.norm/=np.sqrt(norm2(state))
        observers[label]=m.measure(state,energy=False)
    results={}
    for branch,record in previous['records'].items():
        fine=record['final'];coarse=record['coarse_final']
        charge_delta=-((np.array(fine['n_cells'])-observers['fine']['n_cells'])-(np.array(coarse['n_cells'])-observers['coarse']['n_cells']))
        spin_delta=(np.array(fine['local_LEFT'])-observers['fine']['local_LEFT'])-(np.array(coarse['local_LEFT'])-observers['coarse']['local_LEFT'])
        results[branch]=dict(other_cell_occupation_charge_motion_norm=float(np.linalg.norm(record['excess_charge_change'][1:])),other_cell_occupation_charge_dt_difference_norm=float(np.linalg.norm(charge_delta[1:])),other_cell_Noether_motion_norm=float(np.linalg.norm(np.array(record['physical_Noether_vector_density_change'])[1:])),other_cell_Noether_dt_difference_norm=float(np.linalg.norm(spin_delta[1:])),charge_dt_difference=charge_delta.tolist(),Noether_dt_difference=spin_delta.tolist(),meaning='Same-H evolving-neutral subtraction at BOTH temporal resolutions; step difference is measured, not a rigorous error bound or finite-chi convergence proof. Occupation-charge is not lab density for rotating R.')
    return results


def run():
    source=ROOT/(P+'integrated_consumer_raw.npz');raw=np.load(source)
    previous=json.loads((ROOT/(P+'integrated_consumer_result.json')).read_text())
    m,input_record=load_compiled(sys.argv[1],sys.argv[2])
    assert input_record['sha256']==previous['compiled_operator_input']['sha256'] and m.signature==previous['model_signature']
    source_transport_resolution=transport_resolution(m,raw,previous)
    whole=m.mpo(choose_terms(m,whole=True));J=[source_operator(m,a) for a in (0,4)]
    basis=[tuple(v) for v in m.basis];outP={a:m.observable(m.rfsite,np.kron(np.eye(m.nR),fields.primitive(basis,basis,a,'p').toarray())) for a in (4,6)}
    records={};arrays={};initial_preparations={}
    for branch in ('neutral','addition','removal'):
        state=restore(m,raw,'neutral' if branch=='neutral' else branch+'_initial')
        reproduced=measure(m,state,J)
        saved_initial_Pi=all_mode_momenta(m,state)
        if branch=='neutral':
            assert abs(sum(reproduced['n_cells'])-16)<1e-7;initial=state
        else:
            expected=previous['records'][branch]['initial']
            assert np.max(abs(np.array(reproduced['n_cells'])-expected['n_cells']))<1e-7
            assert abs(reproduced['norm2']-expected['norm2'])<1e-7
            inverse=np.zeros(8);inverse[0]=-.19;inverse[4]=-.13;initial=kick(m,state,inverse)
        reference_Pi=all_mode_momenta(m,initial)
        initial_preparations[branch]=dict(reference_all_mode_Pi=reference_Pi,saved_main_pulse_all_mode_Pi=saved_initial_Pi,main_pulse_crossmode_changes=(np.array(saved_initial_Pi)-reference_Pi).tolist(),pulse_kind='Finite retained-field-space unitary, not exact unbounded mode-pure Weyl kick')
        references={}
        for resolution,dt,steps in (('coarse',.02,1),('fine',.01,2)):
            if branch=='neutral':
                # Exact same OWN initial neutral, H, age, dt and cap already ran
                # in the main consumer. Reuse its states, not producer output.
                unc=restore(m,raw,'neutral_'+resolution)
                ue=dict(previous['neutral_dt_control'][resolution])
                assert ue['dt']==dt and ue['steps']==steps and ue['chi_cap']==16
                ue.update(reused_own_state=True,raw_file=source.name,raw_label='neutral_'+resolution,ledger_origin=P+'integrated_consumer_evolution_ledger.json')
            else:
                unc,ue=advance(m,initial,dt=dt,steps=steps,label=branch+'_'+resolution+'_reference',initial_time=.004)
            unoff,oe=advance(m,initial,whole,dt=dt,steps=steps,label=branch+'_'+resolution+'_OFF_reference',initial_time=.004)
            references[resolution]=(unc,unoff,measure(m,unc,J),measure(m,unoff,J),ue,oe)
        rows=[]
        for incoming,outgoing in ((0,6),(2,4)):
            coefficients=np.zeros(8);coefficients[incoming]=.19;kicked=kick(m,initial,coefficients)
            kicked_Pi=all_mode_momenta(m,kicked);initial_shift=np.array(kicked_Pi)-reference_Pi
            answers={}
            for resolution,dt,steps in (('coarse',.02,1),('fine',.01,2)):
                unc,unoff,baseline,offbaseline,ue,oe=references[resolution]
                final,err=advance(m,kicked,dt=dt,steps=steps,label=f'{branch}_{incoming}_{resolution}',initial_time=.004)
                off,offerr=advance(m,kicked,whole,dt=dt,steps=steps,label=f'{branch}_{incoming}_{resolution}_OFF',initial_time=.004)
                full=measure(m,final,J);control=measure(m,off,J)
                def momentum(v):return float(law.physical_element(v,outP[outgoing],v).real/norm2(v))
                response=momentum(final)-momentum(unc);echo=momentum(off)-momentum(unoff)
                charge=-(np.array(full['n_cells'])-baseline['n_cells']);offcharge=-(np.array(control['n_cells'])-offbaseline['n_cells'])
                spin=np.array(full['local_LEFT'])-baseline['local_LEFT'];offspin=np.array(control['local_LEFT'])-offbaseline['local_LEFT']
                answer=dict(field_momentum_response=response,whole_charge_motion_OFF_echo=echo,calibrated_field_response=response-echo,physical_charge_response=charge.tolist(),whole_OFF_charge_response=offcharge.tolist(),calibrated_charge_response=(charge-offcharge).tolist(),physical_local_Noether_response=spin.tolist(),whole_OFF_local_Noether_response=offspin.tolist(),calibrated_local_Noether_response=(spin-offspin).tolist(),physical_current_response=(np.array(full['physical_lab_current'])-baseline['physical_lab_current']).tolist(),evolution=dict(full=err,whole=offerr,uncicked=ue,uncicked_whole=oe))
                answer.update(initial_reference_all_mode_Pi=reference_Pi,initial_kicked_all_mode_Pi=kicked_Pi,initial_crossmode_Pi_shift=float(initial_shift[outgoing]),temporal_full_response_difference=response-float(initial_shift[outgoing]),temporal_field_decoupled_response_difference=echo-float(initial_shift[outgoing]),pulse_kind='exp(i epsilon P*a_in*P), finite-space unitary, not exact full Weyl')
                if branch!='neutral':
                    n=next(v for v in records['neutral'] if v['incoming']==incoming)
                    n=n if resolution=='fine' else n['coarse']
                    answer['excess_carrier_charge_response']=(np.array(answer['calibrated_charge_response'])-n['calibrated_charge_response']).tolist()
                    answer['excess_carrier_Noether_response']=(np.array(answer['calibrated_local_Noether_response'])-n['calibrated_local_Noether_response']).tolist()
                    answer['excess_field_response']=answer['calibrated_field_response']-n['calibrated_field_response']
                answers[resolution]=answer
                for label,psi in dict(full=final,whole_OFF=off).items():save_state(arrays,f'{branch}_{incoming}_{outgoing}_{resolution}_'+label,psi)
            row=dict(answers['fine'],incoming=incoming,outgoing=outgoing,kick=.19,coarse=answers['coarse'],dt_halfdt=dict(time=.02,coarse_dt=.02,fine_dt=.01,calibrated_field_difference=answers['fine']['calibrated_field_response']-answers['coarse']['calibrated_field_response'],calibrated_charge_difference=(np.array(answers['fine']['calibrated_charge_response'])-answers['coarse']['calibrated_charge_response']).tolist(),calibrated_Noether_difference=(np.array(answers['fine']['calibrated_local_Noether_response'])-answers['coarse']['calibrated_local_Noether_response']).tolist()),kick_work=float(law.physical_element(kicked,m.H,kicked).real/norm2(kicked)-law.physical_element(initial,m.H,initial).real/norm2(initial)))
            rows.append(row)
        records[branch]=rows;print(branch,json.dumps(rows),flush=True)
        np.savez_compressed(ROOT/(P+'integrated_paired_raw.npz'),**arrays)
    result=dict(kind='OWN_ACTUAL47_CALIBRATED_FIELD_TO_LOCAL_CARRIER_FEEDBACK',compiled_operator_input=input_record,raw_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),model_signature=m.signature,records=records,source_transport_dt_assessment=source_transport_resolution,counterfactual_semantics_file=P+'control_semantics.json',evolution_ledger_file=P+'integrated_paired_evolution_ledger.json',limits='Reconstructs review49 OWN fresh source/neutral states from raw tensors, norm and n-cell readback; frozen pure operator input, no producer evolved output. Both T/nonforward fields; field-decoupled/hopping-disabled counterfactual and same-H evolving neutral subtraction. R kinetic and lab charge motion persist in the counterfactual; zero owning EM source is not a lab continuity-current claim. Actual localcharge/LEFT/current/field dt/halfdt values, not norm-only acceptance. Exact operator sum differs from approximate group/RK2/finite-chi dynamics; all normalization factors restored, scalar phase in tensors, real norm metadata. No converged photon/localization/mass or full-H exterior claim.')
    result['initial_all_mode_momentum_preparations']=initial_preparations
    result['pulse_semantics_file']=P+'pulse_semantics.json'
    (ROOT/(P+'integrated_paired_evolution_ledger.json')).write_text(json.dumps(ADVANCE_LEDGERS,indent=2)+'\n')
    (ROOT/(P+'integrated_paired_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
