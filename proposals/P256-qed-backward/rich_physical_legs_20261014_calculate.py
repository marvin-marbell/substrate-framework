"""NEW consumer: actual full mixed spectrum -> normalized legs -> same-H event."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import norm
from rich_frame_photon_20261014_model import Parameters
from rich_field_packet_20261014_model import PacketParameters, spatial_windows
from rich_field_packet_20261014_connected import ConnectedPacket
from rich_field_packet_20261014_calculate import attach_windows, snapshot, evolve
from rich_field_packet_20261014_localwell import ground, preparation
from rich_physical_legs_20261014_model import (encode, expectation, stationary_spectrum, spectral_response,
    normalized_leg, current_reaction, frozen_supplier_spectrum)


def physical_receiver(m, state, threshold):
    """Cell x actual physical S x actual right-region electric energy PVM.

    The energy threshold comes from the normalized outgoing leg, NOT a mode
    projection. Every supplier/residual field state is traced inclusively.
    """
    axis = np.array([.31,-.49,.815]); axis /= np.linalg.norm(axis)
    sm = sum(axis[a]*sum(s[a] for s in m.spin) for a in range(3))
    values, vectors = eigh(sm)
    spins = [vectors[:,values>0]@vectors[:,values>0].conj().T, vectors[:,values<0]@vectors[:,values<0].conj().T]
    mask = m.windows[2].diagonal().real >= threshold
    tensor = state.reshape(m.nfield,m.ns,64)
    columns=[]
    for cell in range(2):
        for spin in spins:
            for selected in (mask,~mask):
                z=np.zeros_like(tensor)
                z[selected]=np.einsum('ab,fbk->fak',m.number[cell]@spin,tensor[selected])
                columns.append(z.ravel())
    columns=np.column_stack(columns); gram=columns.conj().T@columns
    probability=np.diag(gram).real
    detection=float(sum(probability[::2]))
    conditional=probability[::2]/detection if detection>1e-15 else None
    return dict(probabilities=probability.tolist(), inclusive_Gram=encode(gram), sum=float(sum(probability)),
                detected_probability=detection, detected_body_spin_conditional=conditional.tolist() if conditional is not None else None,
                threshold=threshold, norm=float(np.vdot(state,state).real), order='cell0/1 x physical obliqueS+/- x rightE>=threshold/<threshold')


def leg_record(leg):
    return {key: encode(value) if isinstance(value,np.ndarray) and np.iscomplexobj(value) else value.tolist() if isinstance(value,np.ndarray) else value for key,value in leg.items() if key!='state'}


def response_record(response):
    return dict(poles=response['records'],retarded_mixed=encode(response['retarded']),source_Gram=encode(response['source_Gram']),
                contact=encode(response['contact']),canonical_contact=encode(response['canonical_contact']),
                spectral_Gram_residual=response['spectral_Gram_residual'],
                bright_poles=sum(r['bright_rank']>0 for r in response['records']),
                max_bright_rank=max(r['bright_rank'] for r in response['records']))


def run(inp, out):
    data=json.loads(inp.read_text())
    p=PacketParameters(base=Parameters(**data['base']),cutoff=data['cutoff'],qwall=data['qwall'])
    grams,first=spatial_windows(p)
    n0,m,n2=[ConnectedPacket(p,n) for n in range(3)]
    for c in (n0,m,n2):attach_windows(c,grams,first)
    print('Diagonalizing complete charged spectrum',flush=True)
    energies,vectors,g,ground_indices=stationary_spectrum(m)
    print('Diagonalizing independent neutral spectrum',flush=True)
    energies0,vectors0,g0,ground_indices0=stationary_spectrum(n0)
    charged=spectral_response(m,energies,vectors,g,data['frequency'],data['broadening'])
    neutral=spectral_response(n0,energies0,vectors0,g0,data['frequency'],data['broadening'])
    # Physical spatial smearings, not selecting coordinate0/coordinate5 poles.
    ci=(.6*grams[0,0]+.8*grams[0,1])/m.G[0,0]
    co=(.8*grams[2,4]-.6*grams[2,5])/m.G[4,4]
    incoming=normalized_leg(m,energies,vectors,g,ci,data['energy_window'])
    outgoing=normalized_leg(m,energies,vectors,g,co,data['energy_window'])
    leg_overlap=np.vdot(incoming['state'],outgoing['state'])
    leg_gram=np.array([[1,leg_overlap],[leg_overlap.conj(),1]])
    threshold=expectation(m.fop(m.windows[2]),outgoing['state'])
    reaction=current_reaction(m,energies,vectors,g,incoming,outgoing,data['broadening'])
    wrong=current_reaction(m,energies,vectors,g,incoming,outgoing,data['broadening'],wrong_noether=True)
    # Source-family input uses its OWN norm/residue/energy, not an unrelated pole.
    theta=data['family_angle']
    initial=np.cos(theta)*g+np.sin(theta)*incoming['state']
    records=[]; saved=dict(charged_reference=g,neutral_reference=g0,incoming=incoming['state'],outgoing=outgoing['state'],initial=initial,
                         charged_energies=energies,neutral_energies=energies0,charged_mixed_amplitudes=charged['amplitudes'],neutral_mixed_amplitudes=neutral['amplitudes'],
                         charged_source_columns=charged['source_columns'],neutral_source_columns=neutral['source_columns'],reaction=reaction['state'],wrong_noether_reaction=wrong['state'])
    amplitudes=vectors.conj().T@initial
    for t in data['times']:
        state=vectors@(np.exp(-1j*t*energies)*amplitudes)
        records.append(dict(time=t,packet=snapshot(m,state),reference=snapshot(m,g),receiver=physical_receiver(m,state,threshold)))
        saved['family_'+str(t)]=state
    reaction_receiver=physical_receiver(m,reaction['state']/np.sqrt(reaction['norm']),threshold) if reaction['norm']>1e-20 else None
    # Separate actual independent field-only pulse, no energy filter and no
    # source-spin-conditioning. Its physical probability is never renormalized
    # by a modal overlap. Well/bias are preparation only, removed in dynamics.
    wellg,welle,wellres=ground(m,data['well'])
    _,_,unitary=preparation(m,grams,data['pulse_strength'])
    pulse=(unitary@wellg.reshape(m.nfield,m.internal)).ravel()
    both=np.column_stack([pulse,wellg]);last=0.;physical=[]
    for t in data['times']:
        both=evolve(m.H,both,t-last);last=t
        pulse_snapshot,ref_snapshot=snapshot(m,both[:,0]),snapshot(m,both[:,1])
        physical.append(dict(time=t,packet=pulse_snapshot,reference=ref_snapshot,receiver=physical_receiver(m,both[:,0],threshold),reference_receiver=physical_receiver(m,both[:,1],threshold),
                             excess_electric=(np.array(pulse_snapshot['electric_regions'])-ref_snapshot['electric_regions']).tolist(),
                             physical_spin_contrast=(np.array(pulse_snapshot['spin'])-ref_snapshot['spin']).tolist(),
                             position_contrast=(np.array(pulse_snapshot['position'])-ref_snapshot['position']).tolist()))
        saved['physical_pulse_'+str(t)]=both[:,0];saved['physical_reference_'+str(t)]=both[:,1]
    saved.update(physical_initial=pulse,physical_reference_initial=wellg,field_unitary=unitary)
    # Full-space own charged-state canonical sources. No borrowed neutral vacuum.
    creators=[m.source_to(n2,a) for a in range(4)]
    annih=[n0.source_to(m,a).getH() for a in range(4)]
    plus=np.column_stack([c@initial for c in creators]);minus=np.column_stack([a@initial for a in annih])
    saved.update(addition_sources=plus,removal_sources=minus)
    add_return=evolve(n2.H,plus,data['source_time']);remove_return=evolve(n0.H,minus,data['source_time'])
    source=dict(addition_Gram=encode(plus.conj().T@plus),removal_RAW_Gram=encode(minus.conj().T@minus),
                canonical_sum_residual=float(np.linalg.norm(plus.conj().T@plus+(minus.conj().T@minus).T-np.eye(4))),
                addition_time_kernel=encode(plus.conj().T@add_return),removal_RAW_time_kernel=encode(minus.conj().T@remove_return),time=data['source_time'])
    k=np.array([.37,-.26,.19]);rho=m.density_fourier(k);j=m.current_fourier(k)
    continuity=float(norm(m.H@rho-rho@m.H+sum(k[a]*j[a] for a in range(3))))
    # NoEther control changes matrix columns rather than calling spin current.
    wrong_columns=np.column_stack([m.S[a%3]@g for a in range(m.nmodes)])
    wrong_columns-=g[:,None]*(g.conj()@wrong_columns)
    wrong_noether_gram=wrong_columns.conj().T@wrong_columns
    frozen=frozen_supplier_spectrum(m,g)
    bare=ConnectedPacket(p,0,induction=False);attach_windows(bare,grams,first)
    be,bv,bg,bgi=stationary_spectrum(bare)
    bare_response=spectral_response(bare,be,bv,bg,data['frequency'],data['broadening'])
    # Same-dimensional current_off contrast has its own prepared reference.
    off=ConnectedPacket(p,1,coupling=False);attach_windows(off,grams,first)
    offg,offe,offres=ground(off,data['well']);_,_,offu=preparation(off,grams,data['pulse_strength'])
    offp=(offu@offg.reshape(off.nfield,off.internal)).ravel()
    offboth=evolve(off.H,np.column_stack([offp,offg]),data['times'][-1])
    off_event=dict(packet=snapshot(off,offboth[:,0]),reference=snapshot(off,offboth[:,1]),ground_residual=offres,
                   current_columns_norm=float(np.linalg.norm(np.column_stack([a@offg for a in off.J]))))
    # Recompute normalized source legs using the wrong neutral reference ONLY
    # as a reported adverse comparator. Never feed it into charged receiver.
    incoming0=normalized_leg(n0,energies0,vectors0,g0,ci,data['energy_window'])
    outgoing0=normalized_leg(n0,energies0,vectors0,g0,co,data['energy_window'])
    reaction_record={key:encode(value) if isinstance(value,np.ndarray) else value for key,value in reaction.items() if key!='state'}
    eigen_residual=float(np.linalg.norm(m.H@vectors[:,:8]-vectors[:,:8]*energies[:8]))
    result=dict(kind='COMPLETE_MIXED_QUANTUM_FIELD_CURRENT_LEGS_AND_NORMALIZED_FINITE_EVENT',parameters=asdict(p),dimensions=[n0.size,m.size,n2.size],
                spectral_scope='COMPLETE 3328 charged and832 neutral finite spectra; all degeneracy partners; full virtual space; no low-state receiver truncation',
                canonical_metric=dict(G=m.G.tolist(),M=m.M.tolist(),Minv=m.Minv.tolist(),sqrt_M=charged['metric']['sqrt_M'].tolist(),inverse_sqrt_M=charged['metric']['inverse_sqrt_M'].tolist(),gram_quadrature_error=float(np.linalg.norm(sum(grams)-m.G))),
                reference=dict(charged_energy=float(energies[0]),neutral_energy=float(energies0[0]),charged_ground_degeneracy=len(ground_indices),neutral_ground_degeneracy=len(ground_indices0),charged_ground_spin=[expectation(s,g) for s in m.S],charged_eigen_residual=eigen_residual,physical_preparation_energy=welle,physical_preparation_residual=wellres),
                mixed_operator_order='6 M^.5 Q,6 M^-.5 P,6 M^-.5 J,3 physical S,3 physical CM chargecurrent',
                charged=response_record(charged),neutral=response_record(neutral),bare_neutral=response_record(bare_response),
                incoming=leg_record(incoming),outgoing=leg_record(outgoing),independent_leg_Gram=encode(leg_gram),independent_leg_Gram_eigenvalues=np.linalg.eigvalsh(leg_gram).tolist(),
                calibrated_physical_outgoing_threshold=threshold,reaction=reaction_record,reaction_receiver_conditional=reaction_receiver,
                wrong_NoEther=dict(current_Gram=encode(wrong_noether_gram),reaction_norm=wrong['norm'],relative_reaction_difference=float(np.linalg.norm(wrong['state']-reaction['state'])/max(np.linalg.norm(reaction['state']),1e-15))),
                frozen_supplier=frozen,current_off=off_event,
                wrong_neutral_reference=dict(incoming=leg_record(incoming0),outgoing=leg_record(outgoing0),charged_neutral_retarded_difference=float(np.linalg.norm(charged['retarded'][:18,:18]-neutral['retarded'][:18,:18]))),
                charged_source=source,continuum_continuity=continuity,physical_halfspin=float(norm(sum(s@s for s in m.S)-.75*m.I)),
                normalized_family_event=records,independent_physical_field_pulse_event=physical,
                source_energy_feedback=dict(family_initial_gap=expectation(m.H,initial)-energies[0],expected_gap=np.sin(theta)**2*incoming['energy'],incoming_residue=incoming['residue'],outgoing_residue=outgoing['residue']),
                limits=['Global bright spectral legs are not localized signals/helicity or asymptotic S-matrix.','Mode energy filters prepare a generally matter-entangled family; they are neither physical spin nor travel proof.','Independent physical spatial pulse and inclusive electric-region/physical-spin receiver remain separate; probability is never a modal-overlap normalization.','Reaction norm is a finite current/contact response strength; only normalized reaction receiver is conditional probability, not cross section.','Explicit finite Fourier-Q law has no assumed CCR/Maxwell Poynting/free-space or unique-vacuum claim.'])
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**saved)
    print(json.dumps(dict(dimensions=result['dimensions'],bright_charged=result['charged']['bright_poles'],bright_neutral=result['neutral']['bright_poles'],bare_bright=result['bare_neutral']['bright_poles'],independent_Gram=result['independent_leg_Gram_eigenvalues'],incoming_energy=incoming['energy'],outgoing_energy=outgoing['energy'],incoming_Z=incoming['residue'],outgoing_Z=outgoing['residue'],reaction_norm=reaction['norm'],wrong_NoEther_difference=result['wrong_NoEther']['relative_reaction_difference'],continuity=continuity,source_CAR=source['canonical_sum_residual'],physical_final=physical[-1])),flush=True)


if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
