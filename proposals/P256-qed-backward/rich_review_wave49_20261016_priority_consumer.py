"""Ask physical transport FIRST; mixed-contact inventory is not a prerequisite."""
from pathlib import Path
import hashlib,inspect,json,sys
import numpy as np
from scipy.linalg import expm
import rich_review_wave49_20261016_integrated_consumer as core
import rich_review_wave49_20261016_priority_storage as storage
import rich_spatial_frame_photon_20261016_fields as fields

ROOT=Path(__file__).resolve().parent;P=core.P;EPOCH='transport_priority2';DT=.004;T0=.004
storage.EPOCH=EPOCH
core.save_stage=storage.save_stage;core.load_stage=storage.load_stage


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def load_saved(m,label,epoch=EPOCH):
    meta=ROOT/(P+epoch+'_'+label+'_stage.json');record=json.loads(meta.read_text());binding=record['binding']
    if binding['base_operator_sha256']!=m._review_base_sha or binding['law_signature']!=m.signature or binding['basis_order_sha256']!=storage.basis_binding(m):raise ValueError(('own stage/model mismatch',label))
    payload=ROOT/record['raw_file']
    if digest(payload)!=record['raw_sha256']:raise ValueError(('own tensor payload mismatch',label))
    with np.load(payload) as raw:
        state=storage.restore(m,raw,'state')
        certificate=ROOT/(P+EPOCH+'_recovered_'+epoch+'_'+label+'.json')
        if 'state_B_00_qtotal' not in raw and not certificate.exists():
            env=np.ones((1,1),complex)
            for i in range(state.L):
                B=raw[f'state_B_{i:02}'];env=np.einsum('ab,apr,bps->rs',env,B.conj(),B,optimize=True)
            direct=float(env[0,0].real*float(raw['state_norm'])**2);native=core.norm2(state)
            if abs(native-direct)>1e-12*max(1.,abs(direct)):raise ValueError('recovered physical raw norm differs from direct tensor contraction')
            certificate.write_text(json.dumps(dict(raw_file=payload.name,raw_sha256=record['raw_sha256'],all_B_entries_bitwise_preserved=True,no_canonicalization_normalization_or_phase_reset=True,native_raw_norm2=native,independent_plain_tensor_raw_norm2=direct,absolute_norm_error=abs(native-direct),physical_total_and_sewing_charge=state.get_total_charge(only_physical_legs=True).tolist(),method='Native exact nonzero-support leg inference cutoff0 followed by DIRECT MPS constructor; future payloads include explicit actual virtual charges and tensor qtotal.'),indent=2)+'\n')
    core.norm2(state)
    return state,record


def save_initial(m,label,state):
    binding=storage.stage_binding(m,m.H,state,0.,0,16,T0);storage.save_stage(m,label,binding,state,0.,[])


def observe(m,state,energy=False):
    unit=core.normalized(state);out=m.measure(unit,energy=energy);out['raw_norm2']=core.norm2(state)
    out['physical_charge_cells']=(np.asarray(m.ions).reshape(4,8).sum(axis=1)-np.asarray(out['n_cells'])).tolist()
    out['local_LEFT_Casimir']=[float(core.law.physical_element(unit,m.observable(site,m.g['casimir'].toarray()),unit).real) for site in m.gsites]
    basis=[tuple(v) for v in m.basis]
    out['physical_field_Pi']=[float(core.law.physical_element(unit,m.observable(m.rfsite,np.kron(np.eye(m.nR),fields.primitive(basis,basis,a,'p').toarray())),unit).real) for a in range(8)]
    out['means_divide_by_raw_physical_norm2']=True
    return out


def initial_pair(m):
    path=ROOT/(P+EPOCH+'_preparation.json')
    if path.exists():
        neutral,_=load_saved(m,'neutral_initial');charged,_=load_saved(m,'charged_initial');return neutral,charged,json.loads(path.read_text())
    neutral,old=load_saved(m,'neutral_preparation','zero_source_repair2')
    generator=-m.geo['spin'][2].T;values,vectors=np.linalg.eigh(generator);index=np.flatnonzero(values>.49)[0];polarization=vectors[:,index]
    charged,strength=m.source(neutral,0,polarization,True,chi=16);quality=dict(m.last_source_quality)
    if strength<=0:raise ValueError('actual polarized canonical Wilson source has zero strength')
    save_initial(m,'neutral_initial',neutral);save_initial(m,'charged_initial',charged)
    record=dict(origin='OWN committed fullH neutral at t.004, not producer state',neutral_source_receipt=old,source_cell=0,create=True,polarization=[[float(c.real),float(c.imag)] for c in polarization],generator_eigenvalue=float(values[index]),source_strength=strength,source_quality=quality,compiled_operator_sha256=m._review_base_sha,signature=m.signature,positions=m.positions.tolist(),finite_q_modes=m.modes.tolist(),field_basis=m.basis.tolist(),physical_time=T0,algorithm_epoch=EPOCH)
    path.write_text(json.dumps(record,indent=2)+'\n');return neutral,charged,record


def carrier(m,neutral,charged,preparation):
    initial=dict(neutral=observe(m,neutral),charged=observe(m,charged));charged_t,ci=core.advance(m,charged,dt=DT,label='charged_full',initial_time=T0)
    # Charged physical tensors/ledger are committed BEFORE baseline/readout work.
    neutral_t,ni=core.advance(m,neutral,dt=DT,label='neutral_full',initial_time=T0)
    final=dict(neutral=observe(m,neutral_t),charged=observe(m,charged_t))
    def excess(group,key):return np.asarray(group['charged'][key])-np.asarray(group['neutral'][key])
    charge0=excess(initial,'physical_charge_cells');charge=excess(final,'physical_charge_cells');spin0=excess(initial,'local_LEFT');spin=excess(final,'local_LEFT');deltaq=charge-charge0;deltag=spin-spin0
    field=(excess(final,'physical_field_Pi')-excess(initial,'physical_field_Pi'))
    assertions=dict(actual_source_LEFT_polarized=bool(abs(spin0[0,2])>.1),same_age_excess_charge_transports=bool(np.linalg.norm(deltaq[1:])>1e-12),same_age_physical_local_LEFT_density_transports=bool(np.linalg.norm(deltag[1:])>1e-12),wrong_static_global_spin_label_rejected=bool(np.linalg.norm(deltag[1:])>1e-12))
    record=dict(kind='OWN_SAME_FROZEN_H_CHARGE_AND_LOCAL_NOETHER_TRANSPORT',algorithm_epoch=EPOCH,preparation=preparation,initial_time=T0,final_time=T0+DT,initial=initial,final=final,physical_excess_charge_initial=charge0.tolist(),physical_excess_charge_final=charge.tolist(),physical_excess_charge_change=deltaq.tolist(),physical_excess_LEFT_initial=spin0.tolist(),physical_excess_LEFT_final=spin.tolist(),physical_Noether_density_change=deltag.tolist(),dynamic_charge_to_field_Pi_doublecontrast=field.tolist(),evolution=dict(charged=ci,neutral_same_age=ni),physics_assertions=assertions,limits=['Same-aged actual neutral baseline under SAME frozen full3529-term H, all32CAR,55g,35R,9field and3radial retained','No K04 construction prerequisite, no model/H/source law change or producer state','Quantum r is a cone coordinate, not charge-profile radius; shell radius.08 and exteriorB20 debt retained','Global conserved spin or n/N labels do not define local LEFT densities; actual g_v operators measured','Approximate native symmetric-group/RK2 MPS evolution chi16, not exponential, dt/basis/exterior convergence or quasiparticle/mass proof','Real source full projected Weyl dressing can prepare initialPi; reported feedback subtracts t0 and same-aged neutral, not raw outgoing echo'])
    path=ROOT/(P+EPOCH+'_carrier_result.json');path.write_text(json.dumps(record,indent=2)+'\n')
    lines=['OWN SAME-H CARRIER: source at.004, physical evolution to.008; source/canonical law unchanged.',f'Operator SHA {m._review_base_sha}; independent own committed preparation, not author state.',f'Actual excess charge initial {charge0.tolist()}; final {charge.tolist()}; change {deltaq.tolist()}.',f'Actual local LEFT density change {deltag.tolist()}; off-cell norm {np.linalg.norm(deltag[1:]):.12g}.',f'Dynamic charge-to-fieldPi doublecontrast (final charged-neutral MINUS initial charged-neutral) {field.tolist()}.',f'Physical assertions {assertions}.',f'Charged/neutral raw norm² {core.norm2(charged_t):.12g}/{core.norm2(neutral_t):.12g}; SVD eps sums {ci["svd_eps_sum"]:.12g}/{ni["svd_eps_sum"]:.12g}.','No transport claim from n/N, no initial dressing echo counted as propagation, no K04/perfect inventory prerequisite.','FullCAR32/virtuals and g55/R35/field9/r3 law retained; physical stage tensors/raw phases/ledgers committed separately.','Finite-step chi16 result only; no full-H continuum/field-shell/exterior convergence, localization theorem, photon or inertial mass claim.']
    (ROOT/(P+EPOCH+'_carrier_synthesis.txt')).write_text('\n'.join(lines)+'\n');print(json.dumps(record),flush=True)
    assert all(assertions.values()),assertions


def kick(m,state):
    out=state.copy();U=np.kron(np.eye(m.nR),expm(.19j*m.Q[0]));out.apply_local_op(m.rfsite,m.register(m.rfsite,U),unitary=True,renormalize=False);return out


def feedback(m,neutral,charged,preparation):
    charged_t,_=load_saved(m,'charged_full');neutral_t,_=load_saved(m,'neutral_full')
    nk=kick(m,neutral);ck=kick(m,charged);save_initial(m,'neutral_kicked_initial',nk);save_initial(m,'charged_kicked_initial',ck)
    ni=observe(m,nk,True);ci=observe(m,ck,True);un=observe(m,neutral,True);uc=observe(m,charged,True)
    ckt,ce=core.advance(m,ck,dt=DT,label='charged_kicked_full',initial_time=T0);nkt,ne=core.advance(m,nk,dt=DT,label='neutral_kicked_full',initial_time=T0)
    states=dict(neutral_initial=un,charged_initial=uc,neutral_kicked_initial=ni,charged_kicked_initial=ci,neutral_full=observe(m,neutral_t),charged_full=observe(m,charged_t),neutral_kicked_full=observe(m,nkt),charged_kicked_full=observe(m,ckt))
    def effect(kind,key):return (np.asarray(states[kind+'_kicked_full'][key])-states[kind+'_full'][key])-(np.asarray(states[kind+'_kicked_initial'][key])-states[kind+'_initial'][key])
    feedback_charge=effect('charged','physical_charge_cells');feedback_neutral=effect('neutral','physical_charge_cells')
    # This OFF variant kills field coupling and intercell hopping, NOT physical
    # lab charge motion: active R kinetic still convects rotating electrons/ions.
    off=m.mpo(core.choose_terms(m,whole=True));co,coe=core.advance(m,charged,off,dt=DT,label='charged_field_decoupled',initial_time=T0);cko,ckoe=core.advance(m,ck,off,dt=DT,label='charged_kicked_field_decoupled',initial_time=T0)
    control=(np.asarray(observe(m,cko)['physical_charge_cells'])-observe(m,co)['physical_charge_cells'])-(np.asarray(ci['physical_charge_cells'])-uc['physical_charge_cells'])
    result=dict(kind='OWN_PAIRED_INCOMING_FINITE_FIELD_KICK_RECIPROCAL_BODY_FEEDBACK',same_H_full_paths=True,initial_time=T0,final_time=T0+DT,protocol='exp(i*.19*P a0 P) native finite-Hilbert incoming-only control; NOT full-Weyl displacement. Initial all8Pi calibrated; source itself uses actual projected full-Weyl dressing.',states=states,kick_work_same_H=dict(neutral=ni['energy']-un['energy'],charged=ci['energy']-uc['energy']),dynamic_field_to_charged_cell_charge=feedback_charge.tolist(),dynamic_field_to_neutral_cell_charge=feedback_neutral.tolist(),charge_specific_kick_doublecontrast=(feedback_charge-feedback_neutral).tolist(),dynamic_field_to_charged_LEFT=effect('charged','local_LEFT').tolist(),dynamic_outgoing_Pi_response=effect('charged','physical_field_Pi').tolist(),field_decoupled_intercell_hopOFF_control=control.tolist(),OFF_semantics='Not whole physical charge-motion OFF: R still rotates actual positions/ions/electrons and convects lab charge. Canonical A source0 is not lab current0.',evolution=dict(charged_kicked=ce,neutral_kicked=ne,charged_OFF=coe,charged_kicked_OFF=ckoe),preparation=preparation,limits='Actual same-H finite-field/body trajectories and same-age controls. No mixed-contact K04 completeness, photon propagation, finite-cut convergence, full quantum rotor closure or scattering claim.')
    (ROOT/(P+EPOCH+'_feedback_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


def load_cached_current(m,mode):
    """Use committed RF arrays; skip ONLY native quadratic HC-name inference."""
    old=P+'zero_source_repair2_J'+str(mode);data=json.loads((ROOT/(old+'_operator.json')).read_text());path=ROOT/(old+'_operator.npz');binding=data['binding']
    expected=dict(base_operator_sha256=m._review_base_sha,law_signature=m.signature,basis_order_sha256=storage.basis_binding(m),source_function_sha256=hashlib.sha256(inspect.getsource(type(m).lab_source).encode()).hexdigest(),mode=mode,other=None,selected_terms_sha256=hashlib.sha256(repr(m.terms).encode()).hexdigest())
    if binding!=expected or digest(path)!=data['raw_sha256']:raise ValueError('canonical-current cache binding mismatch')
    replacements={}
    with np.load(path) as raw:
        for oldname,key in data['RF_arrays'].items():
            A=np.asarray(raw[key],complex).copy();A[abs(A)<2e-12]=0;h=hashlib.sha256(A.tobytes()).hexdigest()
            if h not in m.ops[m.rfsite]:
                name='X'+str(len(m.ops[m.rfsite]));m.sites[m.rfsite].add_op(name,A,need_JW=False,hc=False,permute_dense=True);m.ops[m.rfsite][h]=name
                # Native documented hc=False omits NAME inference only. Actual
                # arrays/chargelegs/permutation unchanged; all adjoints explicit.
            replacements[oldname]=m.ops[m.rfsite][h]
    terms=[(complex(*c),[(replacements.get(name,name) if site==m.rfsite else name,site) for name,site in factors]) for c,factors in data['terms']]
    return m.mpo(terms),dict(cache=old,raw_sha256=data['raw_sha256'],terms=len(terms),native_registration='hc=False disables ONLY name inference; no operator or source coefficients changed')


def currents(m,neutral,charged,preparation):
    cfinal,_=load_saved(m,'charged_full');nfinal,_=load_saved(m,'neutral_full');states=dict(neutral_initial=neutral,charged_initial=charged,neutral_final=nfinal,charged_final=cfinal);values={};receipts={}
    # Load one canonical-current RF cache at a time, no K04 build or Model clone.
    for mode in (0,4):
        operator,receipt=load_cached_current(m,mode);receipts[str(mode)]=receipt
        values[str(mode)]={name:float(core.law.physical_element(core.normalized(state),operator,core.normalized(state)).real) for name,state in states.items()}
        path=ROOT/(P+EPOCH+'_canonical_current_result.json');path.write_text(json.dumps(dict(kind='OWN_ACTUAL_FIXED_CANONICAL_CURRENT_ON_TRANSPORT_STATES',values=values,cache_receipts=receipts,preparation=preparation,limits='Finite canonical derivative, not whole lab convection0 when field-decoupled; no mixed-contact or full response claimed.'),indent=2)+'\n')
    print(json.dumps(dict(kind='OWN_ACTUAL_FIXED_CANONICAL_CURRENT_ON_TRANSPORT_STATES',values=values,cache_receipts=receipts)),flush=True)


def main():
    mode=sys.argv[3];m,compiled=core.load_compiled(sys.argv[1],sys.argv[2]);neutral,charged,preparation=initial_pair(m)
    if mode=='transport':carrier(m,neutral,charged,preparation)
    elif mode=='feedback':feedback(m,neutral,charged,preparation)
    elif mode=='currents':currents(m,neutral,charged,preparation)
    else:raise ValueError(mode)
    (ROOT/(P+EPOCH+'_'+mode+'_evolution_ledger.json')).write_text(json.dumps(dict(compiled_operator=compiled,ledgers=core.ADVANCE_LEDGERS),indent=2)+'\n')

if __name__=='__main__':main()
