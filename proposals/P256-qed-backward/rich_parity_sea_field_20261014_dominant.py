"""Own full-H dominant stationary bright doublet: signed response/positive Q legs."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from rich_parity_sea_field_20261014_model import build, momentum_basis
from rich_parity_sea_field_20261014_calculate import blocks, enc, norm


def run(config):
    root=Path(__file__).resolve().parent;p=json.loads((root/config['law_input']).read_text());m=build(p)
    source=json.loads((root/config['source_result']).read_text());saved=np.load(root/config['source_states'])
    bright=source['source_spectra']['0']['addition']['bright_clusters']
    selected=max((b for b in bright if b['stationary_entire_block'] and b['target_m']==0),key=lambda b:b['total_residue'])
    e=saved['N7_m0_energies'];v=saved['N7_m0_vectors'];u=momentum_basis(m,7,0)
    eg=source['reference']['E'];cluster=np.flatnonzero(abs(e-eg-selected['gap'])<2e-8)
    spin=(u.getH()@m['S'][2]@u).tocsr();w=v[:,cluster]
    _,rotation=eigh(w.conj().T@(spin@w));g=w@rotation[:,-1]
    e0=float(e[cluster[0]]);gaps=e-e0
    q=[(u.getH()@a@u).tocsr() for a in m['Q']];j=[(u.getH()@a@u).tocsr() for a in m['J']]
    qcols=np.column_stack([a@g for a in q]);jcols=np.column_stack([a@g for a in j])
    qcols-=g[:,None]*(g.conj()@qcols);jcols-=g[:,None]*(g.conj()@jcols)
    values,axes=eigh(m['M']);inverse_root=(axes/np.sqrt(values))@axes.T
    columns=np.column_stack([qcols,jcols@inverse_root]);amp=v.conj().T@columns;z=complex(*p['frequency'])
    retarded=amp.conj().T@(amp/(z-gaps)[:,None])-(amp.conj().T@(amp/(z+gaps)[:,None])).T
    records=[]
    for start,end in blocks(e):
        a=amp[start:end];mixed=a[:,:2].conj().T@a[:,2:]
        records.append(dict(signed_gap=float(gaps[start]),degeneracy=end-start,field_residue=enc(a[:,:2].conj().T@a[:,:2]),current_residue=enc(a[:,2:].conj().T@a[:,2:]),mixed_Q_current=enc(mixed),mixed_singular_values=np.linalg.svd(mixed,compute_uv=False).tolist()))
    def leg(coefficients):
        all_amp=v.conj().T@(qcols@np.array(coefficients));positive=all_amp*(gaps>2e-8)
        residue=float(np.vdot(positive,positive).real);state=v@positive/np.sqrt(residue)
        omega=float(np.dot(abs(positive)**2,gaps)/residue)
        variance=float(np.dot(abs(positive)**2,(gaps-omega)**2)/residue)
        return dict(state=state,residue=residue,energy=omega,variance=variance,field=np.array([np.vdot(g,a@state) for a in q]),
                    current=np.array([np.vdot(g,a@state) for a in j]),downward_residue=float(np.sum(abs(all_amp[gaps < -2e-8])**2)))
    incoming=leg([1.,0.]);outgoing=leg([0.,1.])
    ai=incoming['field'];ao=outgoing['field'].conj();ji=sum(ai[a]*j[a] for a in range(2));jo=sum(ao[a]*j[a] for a in range(2))
    contacts=[[(u.getH()@m['K'][a][b]@u).tocsr() for b in range(2)] for a in range(2)]
    contact=sum(ao[a]*ai[b]*(contacts[a][b]@g) for a in range(2) for b in range(2))
    def resolvent(state,shift):
        return v@((v.conj().T@state)/(e0+shift+1j*p['frequency'][1]-e))
    positive=jo@resolvent(ji@g,incoming['energy']);negative=ji@resolvent(jo@g,-outgoing['energy']);reaction=contact+positive+negative
    induced=qcols.conj().T@(v@((v.conj().T@jcols)/(z-gaps)[:,None]))-(jcols.conj().T@(v@((v.conj().T@qcols)/(z+gaps)[:,None]))).T
    route_lines=np.array([r['line'] for r in m['geometry']['routes']])
    neutral=saved['neutral_reference'];raw_field_spin=norm(spin@((u.getH()@m['H']@u).tocsr())-((u.getH()@m['H']@u).tocsr())@spin)
    summary=dict(selected_dominant_stationary_carrier=selected,reference_E=e0,reference_Sz=float(np.vdot(g,spin@g).real),reference_is_excited_in_its_momentum_sector=bool(e0>e[0]+2e-8),
                 complete_virtual_states=len(e),retarded=enc(retarded),all_signed_field_current_poles=records,induced_Q_current=enc(induced),
                 physical_positive_frequency_field_legs={label:dict(residue=a['residue'],energy=a['energy'],energy_variance=a['variance'],downward_residue_retained_in_response=a['downward_residue'],field_amplitude=enc(a['field']),current_amplitude=enc(a['current']),norm=float(np.linalg.norm(a['state']))) for label,a in [('profile0_incoming',incoming),('profile1_outgoing',outgoing)]},
                 reaction=dict(all_virtual_inclusive_strength=float(np.linalg.norm(v.conj().T@reaction)**2),contact_norm=norm(contact),positive_norm=norm(positive),negative_norm=norm(negative),no_contact_strength=float(np.linalg.norm(positive+negative)**2),without_negative_strength=float(np.linalg.norm(contact+positive)**2),meaning='unnormalized finite current/contact second-order strength; not probability or cross section'),
                 independent_profile_current=dict(route_line_singular_values=np.linalg.svd(route_lines,compute_uv=False).tolist(),field_Gram_eigenvalues=np.linalg.eigvalsh(m['M']).tolist(),current_source_Gram_eigenvalues=np.linalg.eigvalsh(jcols.conj().T@jcols).tolist()),
                 spin_response=dict(Sz_H_norm=raw_field_spin,meaning='whole physical frame spin is conserved; mirror-paired vector holonomy cancels. No EM magnetic spin torque or complete spatial SO3 generator claim.'),
                 limits=['Dominant bright m0 doublet is stationary but excited: lower dark m0 and bright moving m±1 sectors remain.','Two independent transverse Ax profile shapes, not two orthogonal polarizations at common k.','All signed/downward virtual response retained; positive-frequency external legs explicitly prepared.','Global field packet and finite ring, no free photon/continuous inertia/species stability.'])
    return summary,dict(em_reference=u@g,neutral_reference=neutral,incoming=u@incoming['state'],outgoing=u@outgoing['state'],reaction=u@reaction)


if __name__=='__main__':
    config=json.loads(Path(sys.argv[1]).read_text());result,states=run(config);stem=Path(sys.argv[2])
    stem.with_name(stem.name+'_result.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(stem.with_name(stem.name+'_states.npz'),**states)
    print(json.dumps({key:value for key,value in result.items() if key not in ('all_signed_field_current_poles','retarded')}))
