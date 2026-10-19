"""Independent interacting-reference packet, all-neutral modes and source audit."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, expm_multiply, norm, splu
from rich_field_packet_20261014_connected import ConnectedPacket
from rich_field_packet_20261014_model import PacketParameters, spatial_windows
from rich_frame_photon_20261014_model import Parameters

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave8_20261014_packet'


def mean(op, state):
    return float(np.vdot(state, op @ state).real)


def evolve(h, states, time):
    return expm_multiply(-1j*time*h, states, traceA=-1j*time*h.diagonal().sum())


def prepared_reference(m, well):
    h = m.H - well*m.N[0] - .029*(.43*m.S[0]-.51*m.S[1]+.745*m.S[2])
    rng = np.random.default_rng(82221014)
    value, vectors = eigsh(h, k=1, which='SA', tol=2e-11,
                           v0=rng.normal(size=m.size)+1j*rng.normal(size=m.size))
    g = vectors[:, 0]
    return g, float(value[0]), float(np.linalg.norm(h@g-value[0]*g))


def observe(m, states, grams, first):
    result=[]
    regional=[m.electric_window(g) for g in grams]
    moment=m.electric_window(sum(first))
    jcm=m.cm_current()
    for state in states.T:
        tensor=state.reshape(m.nfield,m.internal)
        population=np.sum(abs(tensor)**2,axis=1)
        electric=np.array([float(population@w.diagonal().real) for w in regional])
        entries=dict(norm=float(np.vdot(state,state).real), electric=electric.tolist(),
                     first=float(population@moment.diagonal().real),
                     spin=[mean(s,state) for s in m.S], cell=[mean(n,state) for n in m.N],
                     cm_current=[mean(j,state) for j in jcm], current=[mean(j,state) for j in m.J])
        if m.occupation==1:
            axis=np.array([.43,-.51,.745]);axis/=np.linalg.norm(axis)
            sm=sum(axis[a]*sum(s[a] for s in m.spin) for a in range(3))
            values,vectors=eigh(sm)
            projectors=[vectors[:,values>0]@vectors[:,values>0].conj().T,
                        vectors[:,values<0]@vectors[:,values<0].conj().T]
            t=state.reshape(m.nfield,m.ns,64)
            mask=regional[2].diagonal().real>=regional[0].diagonal().real
            entries['inclusive']=[float(np.einsum('fmk,mn,fnk->',t[select].conj(),m.number[c]@ps,t[select]).real)
                                  for c in range(2) for ps in projectors for select in (mask,~mask)]
        result.append(entries)
    a,b=result
    excess=np.array(a['electric'])-b['electric']
    return dict(packet=a,reference=b,excess=excess.tolist(),
                excess_centroid=float((a['first']-b['first'])/sum(excess)) if abs(sum(excess))>1e-13 else None,
                cell_contrast=(np.array(a['cell'])-b['cell']).tolist(),
                spin_contrast=(np.array(a['spin'])-b['spin']).tolist())


def main():
    data=json.loads((ROOT/(PREFIX+'_input.json')).read_text())
    p=PacketParameters(base=Parameters(**data['base']),cutoff=data['cutoff'],qwall=data['qwall'],supplier_spacing=data['supplier_spacing'])
    grams,first=spatial_windows(p)
    n0,m,n2=[ConnectedPacket(p,n) for n in range(3)]
    g,energy,residual=prepared_reference(m,data['well'])
    coeff=(.8*grams[0,0]-.6*grams[0,1])/m.G[0,0]
    q=sum(coeff[a]*m.Q[a] for a in range(m.nmodes))
    vals,vec=eigh(q.toarray())
    u=(vec*np.exp(1j*data['strength']*vals))@vec.conj().T
    packet=(u@g.reshape(m.nfield,m.internal)).ravel()
    pair=np.column_stack([packet,g]);saved=dict(initial=packet,reference_initial=g)
    control_models={'same_state_cut':ConnectedPacket(p,1,bridge=False),'same_state_current_off':ConnectedPacket(p,1,coupling=False)}
    records={label:[] for label in ['interacting',*control_models]}
    for label,c in [('interacting',m),*control_models.items()]:
        both=pair.copy();last=0.
        for t in data['times']:
            both=evolve(c.H,both,t-last);last=t
            record=observe(c,both,grams,first);record['time']=t;records[label].append(record)
        saved[label+'_final_packet']=both[:,0];saved[label+'_final_reference']=both[:,1]
    # A cut changes quantum matter connectivity but retains projected field overlap/tails.
    cut_edges=[edge for edge in m.cube['edges'] if m.cube['sites'][edge[0],0]<0<m.cube['sites'][edge[1],0]]
    creators=[m.source_to(n2,a) for a in range(4)]
    annih=[n0.source_to(m,a).getH() for a in range(4)]
    plus=np.column_stack([c@g for c in creators]);minus=np.column_stack([c@g for c in annih])
    pg=plus.conj().T@plus;mg=minus.conj().T@minus
    z=.57+.39j
    kernels={};resolvent_residual=[]
    # Nonstationary prepared g: these are reference-anchored matrix resolvents, NOT physical poles.
    for label,c,source,sign in [('addition',n2,plus,-1),('removal',n0,minus,1)]:
        matrix=(z-sign*energy)*c.I+sign*c.H
        solution=splu(matrix.tocsc()).solve(source)
        kernels[label]={'real':(source.conj().T@solution).real.tolist(),'imag':(source.conj().T@solution).imag.tolist()}
        resolvent_residual.append(float(np.linalg.norm(matrix@solution-source)))
    k=np.array([.29,-.33,.21]);rho=m.density_fourier(k);js=m.current_fourier(k)
    continuity=float(norm(m.H@rho-rho@m.H+sum(k[a]*js[a] for a in range(3))))
    wrong_spin=float(norm(m.H@rho-rho@m.H+sum(k[a]*m.S[a] for a in range(3))))
    cm=[float(norm(m.cm_current()[a]+1j*(m.H@m.X[a]-m.X[a]@m.H))) for a in range(3)]
    # Literal complete neutral spectrum, actual same-current bright residues, no bare box inference.
    neutral_e,neutral_v=eigh(n0.H.toarray())
    ng=neutral_v[:,0];gap=neutral_e-neutral_e[0]
    bright=np.column_stack([j@ng-ng*np.vdot(ng,j@ng) for j in n0.J])
    spectral=neutral_v.conj().T@bright
    fullgram=bright.conj().T@bright
    bare=ConnectedPacket(p,0,induction=False)
    bare_e,bare_v=eigh(bare.H.toarray());bare_g=bare_v[:,0]
    bare_bright=np.column_stack([j@bare_g-bare_g*np.vdot(bare_g,j@bare_g) for j in bare.J])
    # Diagonal inverse is a plausible wrong electric density in a nonorthogonal basis.
    wrongv=m.momenta@np.diag(1/np.diag(m.M))
    wrongenergy=p.base.epsilon*np.einsum('fa,ab,fb->f',wrongv,grams[2],wrongv)/2
    far=m.electric_window(grams[2]).toarray()
    coherence=m.mop(m.cmhop.toarray()+m.cmhop.toarray().conj().T)
    field=m.fop(q)
    wrongbody=field@m.N[0]
    result=dict(kind='NEW_INDEPENDENT_PACKET_AND_NEUTRAL_BRIGHT_CURRENT_AUDIT',parameters=asdict(p),dimensions=[c.size for c in (n0,m,n2)],
                reference_prepared_energy=energy,prepared_residual=residual,
                reference_event_H_residual=float(np.linalg.norm(m.H@g-mean(m.H,g)*g)),
                reference='Interacting local well+bias reference; BOTH legs released under same H. Not stationary photon/pole vacuum.',
                Gram_eigenvalues=np.linalg.eigvalsh(m.G).tolist(),quadrature_error=float(np.linalg.norm(sum(grams)-m.G)),
                inverse_identity=float(np.linalg.norm(m.Minv@m.M-np.eye(m.nmodes))),
                wrong_diagonal_inverse_far_operator_error=float(np.linalg.norm(np.diag(wrongenergy)-far)),
                far_pulse_operator_change=float(np.linalg.norm(u.conj().T@far@u-far)),
                field_coherence_commutator=float(norm(field@coherence-coherence@field)),
                wrong_body_conditioner_commutator=float(norm(wrongbody@coherence-coherence@wrongbody)),
                CAR_source_Gram=float(np.linalg.norm(pg+mg.T-np.eye(4))),
                wrong_removal_no_transpose=float(np.linalg.norm(pg+mg-np.eye(4))),
                source_resolvent_residuals=resolvent_residual,source_resolvents=kernels,
                continuity=continuity,wrong_Noether_as_current=wrong_spin,centre_current_identity=cm,
                frame_halfspin_error=float(norm(sum(s@s for s in m.S)-.75*m.I)),
                frame_rotation_H_commutators=[float(norm(m.H@s-s@m.H)) for s in m.S],
                cut_physical_edges=cut_edges,trajectories=records,
                neutral=dict(energy=float(neutral_e[0]),residual=float(np.linalg.norm(n0.H@ng-neutral_e[0]*ng)),
                             all_dimension=n0.size,gaps=gap.tolist(),current_residue=spectral.real.tolist(),current_residue_imag=spectral.imag.tolist(),
                             current_Gram_eigenvalues=np.linalg.eigvalsh(fullgram).tolist(),
                             spectral_completeness=float(np.linalg.norm(spectral.conj().T@spectral-fullgram)),
                             bare_box_first_gap=float(bare_e[1]-bare_e[0]),bare_current_norm=float(np.linalg.norm(bare_bright))),
                scope='Finite projected field packet and all-neutral current spectral measure; no automatic bulk/helicity/Lorentz/full NoEther/mass claim. Source spinor transformation and halfsector invariance distinct.')
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),**saved,neutral_reference=ng,plus=plus,minus=minus)
    print(json.dumps({key:result[key] for key in ['dimensions','prepared_residual','reference_event_H_residual','CAR_source_Gram','continuity','wrong_Noether_as_current','centre_current_identity','field_coherence_commutator','wrong_body_conditioner_commutator']},indent=2))
    print(json.dumps({label:[dict(time=r['time'],centroid=r['excess_centroid'],far=r['excess'][2],spin=r['spin_contrast'],cell=r['cell_contrast']) for r in rows] for label,rows in records.items()},indent=2))


if __name__=='__main__':
    main()
