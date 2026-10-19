"""Full charged stationary spectral legs, independently resolved current/contact."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import splu, norm, eigsh
from rich_field_packet_20261014_connected import ConnectedPacket
from rich_field_packet_20261014_model import PacketParameters, spatial_windows
from rich_frame_photon_20261014_model import Parameters
from rich_physical_legs_20261014_model import stationary_spectrum, spectral_response, normalized_leg, current_reaction, frozen_supplier_spectrum
from rich_review_wave8_20261014_packet import evolve, observe

ROOT=Path(__file__).resolve().parent
PREFIX='rich_review_wave8_20261014_legs'


def main():
    data=json.loads((ROOT/(PREFIX+'_input.json')).read_text())
    p=PacketParameters(base=Parameters(**data['base']),cutoff=1,qwall=data['qwall'])
    n0,m,n2=[ConnectedPacket(p,n) for n in range(3)]
    e,v,g,ground_indices=stationary_spectrum(m)
    response=spectral_response(m,e,v,g,data['frequency'],data['broadening'])
    columns=response['source_columns'];z=data['frequency']+1j*data['broadening']
    pos=(z+e[0])*m.I-m.H;neg=(z-e[0])*m.I+m.H
    xp=splu(pos.tocsc()).solve(columns);xm=splu(neg.tocsc()).solve(columns)
    direct=columns.conj().T@xp-(columns.conj().T@xm).T
    wrong_positive=columns.conj().T@xp
    grams,first=spatial_windows(p)
    incoming_coeff=(.6*grams[0,0]-.8*grams[0,1])/m.G[0,0]
    outgoing_coeff=(.8*grams[2,4]+.6*grams[2,5])/m.G[4,4]
    incoming=normalized_leg(m,e,v,g,incoming_coeff,data['window'])
    outgoing=normalized_leg(m,e,v,g,outgoing_coeff,data['window'])
    reaction=current_reaction(m,e,v,g,incoming,outgoing,data['broadening'])
    wrong=current_reaction(m,e,v,g,incoming,outgoing,data['broadening'],wrong_noether=True)
    ai=incoming['field_amplitude'];ao=outgoing['field_amplitude'].conj()
    ji=sum(ai[a]*m.J[a] for a in range(6));jo=sum(ao[a]*m.J[a] for a in range(6))
    rp=splu(((e[0]+incoming['energy']+1j*data['broadening'])*m.I-m.H).tocsc()).solve(ji@g)
    rm=splu(((e[0]-outgoing['energy']+1j*data['broadening'])*m.I-m.H).tocsc()).solve(jo@g)
    contact=sum(ao[a]*ai[b]*((m.link_derivative((a,b))+m.body_contact(a,b))@g) for a in range(6) for b in range(6))
    reacted=contact+jo@rp+ji@rm
    # Independently field-only pulse, not the global energy-window conditioned leg.
    q=sum(incoming_coeff[a]*m.Q[a] for a in range(6));qe,qv=eigh(q.toarray())
    u=(qv*np.exp(1j*data['pulse']*qe))@qv.conj().T
    packet=(u@g.reshape(m.nfield,m.internal)).ravel()
    pair=np.column_stack([packet,g]);events=[];last=0.
    for t in data['times']:
        pair=evolve(m.H,pair,t-last);last=t
        r=observe(m,pair,grams,first);r['time']=t;events.append(r)
    frozen=frozen_supplier_spectrum(m,g)
    plus=np.column_stack([m.source_to(n2,a)@g for a in range(4)])
    minus=np.column_stack([n0.source_to(m,a).getH()@g for a in range(4)])
    source_gram=float(np.linalg.norm(plus.conj().T@plus+(minus.conj().T@minus).T-np.eye(4)))
    adjacent={}
    for label,c,source,sign in [('addition',n2,plus,-1),('removal',n0,minus,1)]:
        matrix=(z-sign*e[0])*c.I+sign*c.H
        x=splu(matrix.tocsc()).solve(source)
        k=source.conj().T@x
        minimum=float(eigsh(c.H,k=1,which='SA',return_eigenvectors=False,tol=1e-10)[0])
        adjacent[label]=dict(reference_number=1,target_number=c.occupation,minimum_energy=minimum,
                             minimum_energy_minus_reference=float(minimum-e[0]),all_virtual_dimension=c.size,
                             residual=float(np.linalg.norm(matrix@x-source)),real=k.real.tolist(),imag=k.imag.tolist())
    sqrt_m=response['metric']['sqrt_M'];inverse=response['metric']['inverse_sqrt_M']
    bright=[r for r in response['records'] if r['bright_rank']]
    top=sorted(bright,key=lambda r:r['field_trace']*r['current_trace'],reverse=True)[:12]
    summary=lambda leg:{k:leg[k] for k in ('energy','energy_variance','residue','norm','field_current_overlap','window')}
    result=dict(kind='NEW_FULL_STATIONARY_CHARGED_CURRENT_BRIGHT_NORMALIZED_LEG_AUDIT',parameters=data,
                dimension=m.size,charged_ground_rank=len(ground_indices),reference_energy=float(e[0]),
                reference_residual=float(np.linalg.norm(m.H@g-e[0]*g)),full_virtual_count=len(e),
                actual_Gram_whitening_error=float(np.linalg.norm(inverse@m.M@inverse-np.eye(6))),
                wrong_unwhitened_metric_error=float(np.linalg.norm(m.M-np.eye(6))),
                all_spectral_Gram_residual=response['spectral_Gram_residual'],
                direct_retarded_resolvent_error=float(np.linalg.norm(response['retarded']-direct)),
                wrong_positive_only_retarded_error=float(np.linalg.norm(response['retarded']-wrong_positive)),
                direct_resolvent_residuals=[float(np.linalg.norm(pos@xp-columns)),float(np.linalg.norm(neg@xm-columns))],
                bright_block_count=len(bright),largest_bright_blocks=top,
                incoming=summary(incoming),outgoing=summary(outgoing),
                actual_current_reaction_norm=reaction['norm'],actual_current_reaction_direct_error=float(np.linalg.norm(reacted-reaction['state'])),
                omitted_contact_state_error=float(np.linalg.norm(contact)),wrong_Noether_reaction_norm=wrong['norm'],
                wrong_Noether_reaction_state_error=float(np.linalg.norm(wrong['state']-reaction['state'])),
                source_CAR_Gram=source_gram,adjacent_source=adjacent,
                frozen_supplier=frozen,independent_field_only_events=events,
                spin_projector_never_used=True,
                scope='Entire stationary N1 spectrum/current-contact with normalized finite spectral-window legs. Physical FIELD pulse/inclusive body-spin-region event distinct from entangled energy-window prep. N1 reference not neutral vacuum; adjacent G reference retained. These are finite retarded/prepared reaction amplitudes, not asymptotic Compton S-matrix or bulk photon modes.')
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),reference=g,incoming=incoming['state'],outgoing=outgoing['state'],reaction=reacted,field_prepared=packet,field_event_final=pair[:,0])
    print(json.dumps({k:result[k] for k in ('dimension','reference_residual','full_virtual_count','all_spectral_Gram_residual','direct_retarded_resolvent_error','wrong_positive_only_retarded_error','bright_block_count','incoming','outgoing','actual_current_reaction_norm','actual_current_reaction_direct_error','wrong_Noether_reaction_state_error','source_CAR_Gram')},indent=2))


if __name__=='__main__':
    main()
