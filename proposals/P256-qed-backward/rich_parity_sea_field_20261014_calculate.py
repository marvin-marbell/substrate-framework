"""Natural full finite-H consumer: occupied source, stationary bright poles and EM.

No state cutoff, mass extraction, source-spin projection or producer replay.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply, eigsh
from rich_parity_sea_field_20261014_model import build, momentum_basis, fourier_source


def norm(a):
    return float(np.linalg.norm(a.data if hasattr(a, 'tocsr') else a))


def enc(a):
    a = np.asarray(a)
    return dict(real=a.real.tolist(), imag=a.imag.tolist())


def expect(a, v):
    return float(np.vdot(v, a @ v).real)


def blocks(e):
    start = 0
    while start < len(e):
        end = start + 1
        while end < len(e) and e[end] - e[start] < 2e-8:
            end += 1
        yield start, end
        start = end


def diagonalize(model, number):
    output = []
    for m in range(model['parameters']['sites']):
        u = momentum_basis(model, number, m)
        e, v = eigh((u.getH() @ model['H'] @ u).toarray(), driver='evd', overwrite_a=True, check_finite=False)
        output.append(dict(m=m, U=u, e=e, v=v))
    return output


def evolve(h, v, t):
    return expm_multiply(-1j*t*h, v, traceA=-1j*t*h.diagonal().sum())


def source_analysis(model, sectors, references, eg):
    sites = model['parameters']['sites']; sea = 2*sites; z = complex(*model['parameters']['frequency'])
    records = {}; greens = {}; saved = {}; bright = []
    for momentum in range(sites):
        sources = fourier_source(model, momentum)
        green = np.zeros((4,4), complex); car = green.copy(); entries = {}
        for label, number, operators in [('addition', sea+1, sources), ('removal', sea-1, [d.getH() for d in sources])]:
            poles = []; clusters = []
            for b in sectors[number]:
                amps = [b['v'].conj().T @ (b['U'].getH() @ np.column_stack([a @ r for a in operators])) for r in references]
                weight = sum(np.sum(abs(a)**2, axis=1) for a in amps) / len(references)
                gaps = b['e']-eg
                for i, gap in enumerate(gaps):
                    residue = sum(np.outer(a[i].conj(),a[i]) if label == 'addition' else np.outer(a[i],a[i].conj()) for a in amps) / len(references)
                    car += residue
                    green += residue / (z-gap if label == 'addition' else z+gap)
                    poles.append(dict(target_m=b['m'], gap=float(gap), weight=float(weight[i])))
                saved[f'{momentum}_{label}_m{b["m"]}_amplitudes'] = np.stack(amps)
                for start, end in blocks(b['e']):
                    residue = sum(a[start:end].conj().T @ a[start:end] for a in amps) / len(references)
                    total = float(np.trace(residue).real)
                    if total <= 1e-10:
                        continue
                    w = b['U'] @ b['v'][:,start:end]
                    vm = w.conj().T @ (model['V'] @ w)
                    jm = [w.conj().T @ (a @ w) for a in model['J']]
                    pm = w.conj().T @ (model['P'] @ w)
                    sm = [w.conj().T @ (a @ w) for a in model['S']]
                    item = dict(source_m=momentum, target_m=b['m'], N=number, gap=float(gaps[start]), degeneracy=end-start,
                                residue=enc(residue), total_residue=total, velocity_matrix=enc(vm), velocity_eigenvalues=np.linalg.eigvalsh(vm).tolist(),
                                current_matrices=[enc(a) for a in jm], current_operator_norms=[norm(a) for a in jm], parity_matrix=enc(pm),
                                physical_spin_matrices=[enc(a) for a in sm], S2_values=np.linalg.eigvalsh(w.conj().T @ (model['S2'] @ w)).tolist(),
                                stationary_entire_block=norm(vm)<1e-8 and max(norm(a) for a in jm)<1e-8)
                    clusters.append(item)
                    if not any(a['target_m']==b['m'] for a in bright if a['source_m']==momentum and a['N']==number):
                        bright.append(item)
            entries[label] = dict(all_poles=poles, bright_clusters=clusters, total_weight=sum(a['weight'] for a in poles))
        records[str(momentum)] = entries
        greens[str(momentum)] = dict(G=enc(green), CAR_error=norm(car-np.eye(4)))
    return records, greens, saved, bright


def em_response(model, block, reference_index, z, p):
    u = block['U']; v = block['v']; e = block['e']; index = reference_index
    g = v[:,index]; e0 = e[index]; gaps = e-e0
    operators = {key:[(u.getH() @ a @ u).tocsr() for a in model[key]] for key in ('Q','J','S')}
    velocity = (u.getH() @ model['V'] @ u).tocsr()
    root_values, root_vectors = eigh(model['M']); inverse_root = (root_vectors / np.sqrt(root_values)) @ root_vectors.T
    qcols = np.column_stack([a@g for a in operators['Q']]); jcols = np.column_stack([a@g for a in operators['J']]) @ inverse_root
    qcols -= g[:,None]*(g.conj()@qcols); jcols -= g[:,None]*(g.conj()@jcols)
    columns = np.column_stack([qcols,jcols]); amplitudes = v.conj().T@columns
    response = amplitudes.conj().T @ (amplitudes/(z-gaps)[:,None]) - (amplitudes.conj().T @ (amplitudes/(z+gaps)[:,None])).T
    poles = []
    for start,end in blocks(e):
        t = amplitudes[start:end]; cross = t[:,:2].conj().T@t[:,2:]
        poles.append(dict(gap=float(gaps[start]),degeneracy=end-start,field_residue=enc(t[:,:2].conj().T@t[:,:2]),current_residue=enc(t[:,2:].conj().T@t[:,2:]),mixed=enc(cross),mixed_singular_values=np.linalg.svd(cross,compute_uv=False).tolist()))
    def leg(coefficients):
        raw = qcols@np.array(coefficients); amp = v.conj().T@raw
        amp[abs(gaps)<2e-8] = 0
        residue = float(np.vdot(amp,amp).real)
        state = v@amp/np.sqrt(residue)
        return dict(state=state,residue=residue,energy=float(np.dot(abs(amp)**2,gaps)/residue),field=np.array([np.vdot(g,a@state) for a in operators['Q']]),current=np.array([np.vdot(g,a@state) for a in operators['J']]))
    incoming = leg([1.,0.]); outgoing = leg([.31,1.])
    ai = incoming['field']; ao = outgoing['field'].conj()
    ji = sum(ai[a]*operators['J'][a] for a in range(2)); jo = sum(ao[a]*operators['J'][a] for a in range(2))
    contacts = [[(u.getH() @ model['K'][a][b] @ u).tocsr() for b in range(2)] for a in range(2)]
    contact = sum(ao[a]*ai[b]*(contacts[a][b]@g) for a in range(2) for b in range(2))
    def resolvent(state, shift):
        return v@((v.conj().T@state)/(e0+shift+1j*p['frequency'][1]-e))
    positive = jo@resolvent(ji@g,incoming['energy'])
    negative = ji@resolvent(jo@g,-outgoing['energy'])
    reaction = contact+positive+negative
    # Independent initial field packet and body-motion/spin insertions act in
    # the same charged Hilbert space. No tensor factorization of its ground.
    field_seed = incoming['state']
    body_insert = velocity + .43*operators['S'][1] + .17*operators['S'][2]
    joint = body_insert@field_seed; joint -= g*np.vdot(g,joint); joint /= np.linalg.norm(joint)
    base = body_insert@g; base -= g*np.vdot(g,base); base /= np.linalg.norm(base)
    event = {}
    for label,initial in [('field_only',field_seed),('body_only',base),('independent_field_times_body',joint)]:
        event[label] = []
        for time in p['times']:
            state = evolve((u.getH()@model['H']@u).tocsr(),initial,time)
            event[label].append(dict(time=time,norm=float(np.linalg.norm(state)),energy=expect((u.getH()@model['H']@u).tocsr(),state),velocity=expect(velocity,state),Q=[expect(a,state) for a in operators['Q']],current=[expect(a,state) for a in operators['J']],spin=[expect(a,state) for a in operators['S']],inclusive_reaction_probability=float(abs(np.vdot(state,reaction))**2)))
    induced_j = qcols.conj().T @ (v @ ((v.conj().T@jcols)/(z-gaps)[:,None])) - (jcols.conj().T @ (v @ ((v.conj().T@qcols)/(z+gaps)[:,None]))).T
    return dict(reference_E=float(e0),reference_m=block['m'],dimension=len(e),retarded=enc(response),all_poles=poles,source_Gram=enc(columns.conj().T@columns),induced_Q_current=enc(induced_j),
                physical_field_legs={label:dict(residue=x['residue'],energy=x['energy'],field_amplitude=enc(x['field']),current_amplitude=enc(x['current']),norm=float(np.linalg.norm(x['state']))) for label,x in [('incoming',incoming),('outgoing',outgoing)]},
                full_virtual_reaction=dict(norm=float(np.vdot(reaction,reaction).real),contact_norm=norm(contact),positive_norm=norm(positive),negative_norm=norm(negative),no_contact_norm=norm(positive+negative),without_negative_norm=norm(contact+positive),virtual_states=len(e),inclusive_all_final_probability=float(np.linalg.norm(v.conj().T@reaction)**2)),
                event=event), dict(em_reference=u@g,field_incoming=u@incoming['state'],field_outgoing=u@outgoing['state'],inclusive_reaction=u@reaction,independent_field_body=u@joint)


def run(p):
    model = build(p); h = model['H']; sites = p['sites']; sea = 2*sites
    sectors = {n:diagonalize(model,n) for n in (sea-1,sea,sea+1)}
    eg = min(b['e'][0] for b in sectors[sea]); references=[]; reference_m=[]
    for b in sectors[sea]:
        for j in np.flatnonzero(abs(b['e']-eg)<2e-8):
            references.append(b['U']@b['v'][:,j]); reference_m.append(b['m'])
    refs = np.column_stack(references)
    f = model['frame']; ident = eye(h.shape[0]); parity=model['P']; c=model['CF']; at=model['AT']
    raw = dict(Gauss_P=max(norm(a@f['P']) for a in f['G']),positive_frame_eigenvalues=np.linalg.eigvalsh(sum(a@a for a in f['S'])).tolist(),
               bare_source_physical_gram=max(norm(f['P'].conj().T@a@f['P']) for a in f['bare']),
               source_CAR=max(norm(a.getH()@b+b@a.getH()-(ident if i==j else 0)) for i,a in enumerate(model['source']) for j,b in enumerate(model['source'])),
               P_squared=norm(parity@parity-ident),P_H=norm(parity@h@parity.getH()-h),P_velocity=norm(parity@model['V']@parity.getH()+model['V']),
               P_current=[norm(parity@a@parity.getH()+a) for a in model['J']],P_axial_spin=[norm(parity@a@parity.getH()-a) for a in model['S']],
               P_reference_projector=norm(parity@refs-refs@(refs.conj().T@(parity@refs))),C_H=norm(c@h@c.getH()-h),
               C_reference_projector=norm(c@refs-refs@(refs.conj().T@(c@refs))),T_H=norm(at@h.conjugate()@at.getH()-h),
               T_axial_spin=[norm(at@a.conjugate()@at.getH()+a) for a in model['S']],T_reference_projector=norm(at@refs.conj()-refs@(refs.conj().T@(at@refs.conj()))),
               spin_H_commutators=[norm(a@h-h@a) for a in model['S']],S2_H=norm(model['S2']@h-h@model['S2']),
               translation_H=norm(model['T']@h-h@model['T']),continuity=[norm(1j*(h@n-n@h)-(model['edgeJ'][(j-1)%sites]-model['edgeJ'][j])) for j,n in enumerate(model['nums'])])
    raw['P_source_sewing'] = max(norm(parity@model['source'][4*j+a]@parity.getH()-model['source'][4*((-j)%sites)+(a+2)%4]) for j in range(sites) for a in range(4))
    raw['C_source'] = max(norm(c@model['source'][4*j+a]@c.getH()-sum(model['source_C'][a,b]*model['source'][4*j+b].getH() for b in range(4))) for j in range(sites) for a in range(4))
    raw['source_adjoint_spin'] = [enc(np.array([[np.trace(f['source'][b].conj().T@(s@f['source'][a]-f['source'][a]@s))/np.trace(f['source'][b].conj().T@f['source'][b]) for b in range(4)] for a in range(4)])) for s in f['S']]
    delta=1e-5
    raw['current_derivative']=[]; raw['contact_derivative']=[]
    for a in range(2):
        shift=np.zeros(2);shift[a]=delta
        hp=build(p,external=shift,frame=f)['H'];hm=build(p,external=-shift,frame=f)['H']
        raw['current_derivative'].append(norm((hp-hm)/(2*delta)+model['J'][a]))
        raw['contact_derivative'].append(norm((hp-2*h+hm)/delta**2-model['K'][a][a]))
    spectra, greens, states, bright = source_analysis(model,sectors,references,eg)
    pair_records=[]; pair_retarded=np.zeros((2,2),complex); z=complex(*p['frequency'])
    for b in sectors[sea]:
        amps=[b['v'].conj().T@(b['U'].getH()@np.column_stack([j@r for j in model['J']])) for r in references]
        states[f'pair_current_m{b["m"]}']=np.stack(amps)
        for start,end in blocks(b['e']):
            matrix=sum(a[start:end].conj().T@a[start:end] for a in amps)/len(references);gap=float(b['e'][start]-eg)
            pair_records.append(dict(m=b['m'],gap=gap,degeneracy=end-start,residue=enc(matrix)))
            if gap>2e-8:pair_retarded+=matrix/(z-gap)-matrix.T/(z+gap)
    # Physical EM response owns a genuine source-bright stationary charged
    # eigenstate, not a dark floor or a different construction's reference.
    eligible=[a for a in bright if a['N']==sea+1 and a['target_m']==0]
    candidate=min(eligible,key=lambda a:a['gap']); charged=sectors[sea+1][0]
    reference_index=int(np.argmin(abs(charged['e']-eg-candidate['gap'])))
    em,em_states=em_response(model,charged,reference_index,z,p);states.update(em_states)
    states['neutral_reference']=refs
    vertex=[]
    source_columns=[charged['U'].getH()@np.column_stack([d@r for d in fourier_source(model,0)]) for r in references]
    for current in model['J']:
        current_eigen=charged['v'].conj().T@(charged['U'].getH()@current@charged['U']@charged['v'])
        value=np.zeros((4,4),complex)
        for cols in source_columns:
            amp=charged['v'].conj().T@cols
            denominator=z-(charged['e']-eg)
            value+=amp.conj().T@((current_eigen@(amp/denominator[:,None]))/denominator[:,None])/len(references)
        vertex.append(enc(value))
    for n,items in sectors.items():
        for b in items:
            states[f'N{n}_m{b["m"]}_energies']=b['e'];states[f'N{n}_m{b["m"]}_vectors']=b['v']
    adverse=build(p,mirror=False,frame=f)
    adverse_data=dict(P_H=norm(adverse['P']@adverse['H']@adverse['P'].getH()-adverse['H']),route_count=len(adverse['geometry']['routes']),m0_lowest={})
    for n in (sea,sea+1):
        u=momentum_basis(adverse,n,0);hh=u.getH()@adverse['H']@u
        ee,vv=eigsh(hh,k=6,which='SA',tol=1e-11);order=np.argsort(ee);ee=ee[order];vv=vv[:,order]
        adverse_data['m0_lowest'][str(n)]=dict(energies=ee.tolist(),velocity_matrix=enc(vv.conj().T@(u.getH()@adverse['V']@u@vv)))
    # Flux response is kept as flux response; no continuous-inertia relabel.
    flux=[]; phi=p['twist_step']
    for sign in (-1,1):
        mt=build(p,twist=sign*phi,frame=f)
        for n in (sea,sea+1):
            b=sectors[n][0];hh=b['U'].getH()@mt['H']@b['U']
            ee=eigsh(hh,k=4,which='SA',tol=1e-11,return_eigenvectors=False)
            flux.append(dict(twist=sign*phi,N=n,m=0,raw_low_energies=sorted(ee.tolist())))
    return dict(candidate=p['candidate'],parameters=p,reference=dict(N=sea,E=float(eg),degeneracy=len(references),momenta=reference_m,S2_ensemble=sum(expect(model['S2'],r) for r in references)/len(references),density_ensemble=[sum(expect(n,r) for r in references)/len(references) for n in model['nums']],ensemble='equal entire exact ground manifold; no parity/spin filter'),
                raw=raw,source_G=greens,source_spectra=spectra,first_bright_blocks=bright,selected_stationary_carrier=candidate,
                source_current_RJR_m0=vertex,
                allowed_momentum_floors={str(n):[dict(m=b['m'],energy=float(b['e'][0]),same_reference_gap=float(b['e'][0]-eg)) for b in sectors[n]] for n in (sea-1,sea+1)},
                occupied_particle_hole=dict(all_poles=pair_records,retarded=enc(pair_retarded),contact=enc(np.array([[sum(expect(model['K'][a][b],r) for r in references)/len(references) for b in range(2)] for a in range(2)]))),
                geometry=dict(route_count=len(model['geometry']['routes']),routes=[dict(points=r['points'].tolist(),axial=r['axial'].tolist(),line=r['line'].tolist()) for r in model['geometry']['routes']],Gram=model['geometry']['gram'].tolist(),electric_metric=model['M'].tolist(),inverse_metric=np.linalg.inv(model['M']).tolist(),electric_minimum=float(eigsh(model['electric'],k=1,which='SA',return_eigenvectors=False)[0])),
                same_H_charged_EM=em,adverse_uncompleted_routes=adverse_data,finite_boundary_flux_response=flux,
                limits=['Finite T1 x R2 route root, not continuous three-dimensional COM inertia.','Two global transverse profiles, not local travelling free photons.','Full frame generators measured; source adjoint spin half is not full-H SO3 invariance.','Finite Dirichlet field regulator, no exact CCR and no primitive B squared or oscillator potential.','No stability or nature/QED/species admission.']), states


def main():
    p=json.loads(Path(sys.argv[1]).read_text());result,states=run(p);stem=Path(sys.argv[2])
    stem.with_name(stem.name+'_result.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(stem.with_name(stem.name+'_states.npz'),**states)
    print(json.dumps(dict(candidate=result['candidate'],reference=result['reference'],raw=result['raw'],bright=result['first_bright_blocks'],selected=result['selected_stationary_carrier'],EM=result['same_H_charged_EM']['physical_field_legs'],reaction=result['same_H_charged_EM']['full_virtual_reaction'],adverse=result['adverse_uncompleted_routes'])))


if __name__ == '__main__':
    main()
