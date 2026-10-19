"""Same-root independent global-field packet times localized occupied body event."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import expm_multiply
from rich_parity_sea_field_20261014_model import build


def expect(a,v):
    return float(np.vdot(v,a@v).real)


def unit(v):
    return v/np.linalg.norm(v)


def evolve(h,v,time):
    return expm_multiply(-1j*time*h,v,traceA=-1j*time*h.diagonal().sum())


def run(config):
    root=Path(__file__).resolve().parent
    p=json.loads((root/config['law_input']).read_text());m=build(p)
    captured=np.load(root/config['source_states'])
    references=captured['neutral_reference'];g=references[:,0]
    number=2*p['sites']+1
    indices=np.flatnonzero(np.repeat([state.bit_count()==number for state in range(m['matter'])],m['nf']))
    h=m['H'][indices][:,indices].tocsr()
    # Local charged insertion and full-frame spin rotation form the body
    # independently of the field preparation; these commute with every Q.
    body=m['source'][0]+(.27+.31j)*m['source'][1]+(.23-.19j)*m['source'][5]
    source=body@g
    rotated=evolve(m['S'][1],source,config['spin_angle'])
    density_phase=evolve(m['nums'][0],rotated,config['body_phase'])
    body_seed=unit(density_phase[indices])
    qcols=[]
    for a in m['Q']:
        field=a@g-g*np.vdot(g,a@g)
        qcols.append(field)
    field_sources=[]
    for field in qcols:
        incoming=body@unit(field)
        incoming=evolve(m['S'][1],incoming,config['spin_angle'])
        incoming=evolve(m['nums'][0],incoming,config['body_phase'])
        field_sources.append(unit(incoming[indices]))
    # The receiver is the full source-family span at each site after an
    # independent outgoing physical Q insertion; no halfspin spectral filter.
    receivers={}
    receiver_rank={}
    for profile in range(2):
        for site in range(p['sites']):
            columns=np.column_stack([(m['source'][4*site+a]@qcols[profile])[indices] for a in range(4)])
            u,s,_=np.linalg.svd(columns,full_matrices=False);mask=s>1e-10
            receivers[(profile,site)]=u[:,mask]
            receiver_rank[f'profile{profile}_site{site}']=int(sum(mask))
    ops={key:[a[indices][:,indices] for a in m[key]] for key in ('Q','E','J','S','nums')}
    v=m['V'][indices][:,indices];electric=m['electric'][indices][:,indices]
    charged_reference=unit(captured['em_reference'][indices])
    unitary_seeds=[evolve(ops['Q'][a],charged_reference,-config['field_pulse']) for a in range(2)]
    unitary_body=[evolve(ops['Q'][a],body_seed,-config['field_pulse']) for a in range(2)]
    conditioned=evolve(ops['Q'][0]@ops['nums'][0],charged_reference,-config['field_pulse'])
    # Exact joint PVM of local occupation, physical Sz sign and both finite
    # field coordinates, tracing every other matter/spin state.
    ndiag=ops['nums'][0].diagonal().real; sdiag=ops['S'][2].diagonal().real
    from scipy.sparse import diags
    diagonal_error=max(float(np.linalg.norm((a-diags(a.diagonal())).data)) for a in (ops['nums'][0],ops['S'][2]))
    if diagonal_error>1e-8:
        raise ValueError('Joint diagonal occupation/physical spin PVM requires a new exact simultaneous spectral basis')
    nbin=np.rint(ndiag).astype(int)
    sbin=np.where(sdiag>1e-8,2,np.where(sdiag < -1e-8,0,1))
    fieldbin=np.arange(len(indices))%m['nf']
    detector_bin=(nbin*3+sbin)*m['nf']+fieldbin
    independent_commutator=float(np.linalg.norm((m['Q'][0]@body-body@m['Q'][0]).data))
    conditioner=m['Q'][0]@m['nums'][0]
    wrong_conditioner_commutator=float(np.linalg.norm((conditioner@body-body@conditioner).data))
    reference_density=[sum(expect(a,r) for r in references.T)/references.shape[1] for a in m['nums']]
    event={}
    for label,initial in [('localized_body_only',body_seed),('nonunitary_profile0_times_body',field_sources[0]),('nonunitary_profile1_times_body',field_sources[1]),
                          ('bright_charged_unpulsed',charged_reference),('bright_charged_field0_only_unitary',unitary_seeds[0]),('bright_charged_field1_only_unitary',unitary_seeds[1]),
                          ('localized_body_field0_only_unitary',unitary_body[0]),('localized_body_field1_only_unitary',unitary_body[1]),('wrong_body_conditioned_field_pulse',conditioned)]:
        records=[]
        for time in p['times']:
            state=evolve(h,initial,time)
            probability=np.sum(abs(state.reshape(-1,m['nf']))**2,axis=0)
            pvm=np.bincount(detector_bin,weights=abs(state)**2,minlength=5*3*m['nf']).reshape(5,3,p['field_grid'],p['field_grid'])
            records.append(dict(time=time,norm=float(np.linalg.norm(state)),energy=expect(h,state),excess_density=[expect(a,state)-reference_density[j] for j,a in enumerate(ops['nums'])],
                                spin=[expect(a,state) for a in ops['S']],velocity=expect(v,state),Q=[expect(a,state) for a in ops['Q']],E=[expect(a,state) for a in ops['E']],
                                current=[expect(a,state) for a in ops['J']],electric_energy=expect(electric,state),joint_field_distribution=probability.tolist(),
                                occupation_spin_field_PVM=pvm.tolist(),PVM_trace=float(np.sum(pvm)),
                                independent_outgoing_field_source_receiver={f'profile{a}_site{j}':float(np.linalg.norm(r.conj().T@state)**2) for (a,j),r in receivers.items()}))
        event[label]=records
    differences={}
    for profile in range(2):
        joint=event[f'localized_body_field{profile}_only_unitary'];body_records=event['localized_body_only']
        differences[str(profile)]={key:max(float(np.linalg.norm(np.array(a[key])-np.array(b[key]))) for a,b in zip(joint,body_records)) for key in ('excess_density','spin','current','joint_field_distribution')}
    charged_differences={}
    for profile in range(2):
        pulsed=event[f'bright_charged_field{profile}_only_unitary'];baseline=event['bright_charged_unpulsed']
        charged_differences[str(profile)]={key:max(float(np.linalg.norm(np.array(a[key])-np.array(b[key]))) for a,b in zip(pulsed,baseline)) for key in ('current','joint_field_distribution','occupation_spin_field_PVM')}
    return dict(candidate=p['candidate'],parameters=config,reference='first exact N6 occupied ground member captured by the full same-H spectral consumer; no borrowed state',
                charged_dimension=len(indices),receiver_ranks=receiver_rank,events=event,field_packet_changes_to_body=differences,field_only_charged_changes=charged_differences,
                physical_PVM=dict(meaning='joint local occupation N0, sign of actual full physical Sz, both field Q indices; all remaining matter states traced',diagonal_error=diagonal_error),
                independent_preparation=dict(unitary='exp(i field_pulse Q_alpha) on same source-bright charged eigenstate and separately localized body seed, no body conditioner',field_body_commutator=independent_commutator,wrong_QN_body_commutator=wrong_conditioner_commutator),
                nonunitary_comparator='Q|sea> then charged body insertion and source-span receiver are nonunitary spectral families, not independent pulse or inclusive physical PVM',
                limits=['Profiles are global field packets, not localized free-photon propagation.','Number/density/current transport is finite ring motion, not continuous 3D inertia.','Source-span comparator is not a detector; joint occupation/physical-spin-sign/field PVM is the inclusive finite detector.']),dict(body_seed=body_seed,field0_body=field_sources[0],field1_body=field_sources[1],N7_indices=indices,charged_reference=charged_reference,field0_unitary=unitary_seeds[0],field1_unitary=unitary_seeds[1])


if __name__=='__main__':
    config=json.loads(Path(sys.argv[1]).read_text());result,states=run(config);stem=Path(sys.argv[2])
    stem.with_name(stem.name+'_result.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(stem.with_name(stem.name+'_states.npz'),**states)
    print(json.dumps(dict(charged_dimension=result['charged_dimension'],receiver_ranks=result['receiver_ranks'],field_packet_changes_to_body=result['field_packet_changes_to_body'])))
