"""Own physical paths first; deferred canonical readouts never gate trajectories."""
from pathlib import Path
from types import MethodType
import hashlib,json,os,subprocess,sys,time,zipfile
import numpy as np
import rich_review_wave49_20261016_priority_consumer as p
import rich_review_wave49_20261016_integrated_storage as oldstorage
from rich_review_wave49_20261016_execute import dependencies

ROOT=p.ROOT;P=p.P;PY='/home/administrator/substrate-framework/.venv/bin/python'


def packcomplex(value):return [float(value.real),float(value.imag)]


def fast_register(m,i,A):
    """Native documented HC-NAME inference opt-out; exact same dense matrix."""
    A=np.asarray(A,complex).copy();A[abs(A)<2e-12]=0;key=hashlib.sha256(A.tobytes()).hexdigest()
    if key not in m.ops[i]:
        name='X'+str(len(m.ops[i]));m.sites[i].add_op(name,A,need_JW=False,hc=False,permute_dense=True);m.ops[i][key]=name
    return m.ops[i][key]


def regulator_measurements(m,state):
    unit=p.core.normalized(state)
    def mean(site,A):return float(p.core.law.physical_element(unit,m.observable(site,A),unit).real)
    shell=np.diag([float(sum(row)==m.parameters['field_cap']) for row in m.basis]);C=m.Q[0]@m.Q[4]-m.Q[4]@m.Q[0]
    material=np.diag([float(label[0]==m.parameters['material_max']) for label in m.material['labels']])
    radial=[[mean(site,np.diag(np.arange(3)==node)) for node in range(3)] for site in m.rsites]
    return dict(actual_field_top_total_Fock_shell_weight=mean(m.rfsite,np.kron(np.eye(m.nR),shell)),cut_coordinate_commutator_04_action_norm2=mean(m.rfsite,np.kron(np.eye(m.nR),C.conj().T@C)),actual_material_top_j_weight=mean(m.rfsite,np.kron(material,np.eye(m.nf))),actual_quantum_cone_radial_node_probabilities=radial,meaning='State/operator-specific finite-cut diagnostics, NOT observable error bounds or convergence. Cone quantum r nodes are not radius.08 of charge shells; fieldQ cut noncommutativity is not a new continuum law. Actual g top-two-j weights separately in each physical ledger.')


def reciprocal(m,neutral,charged,preparation):
    p.feedback(m,neutral,charged,preparation)
    # Free initial Weyl dressing can itself move Pi. Subtract a SAME-STATE,
    # SAME-AGE field-decoupled law, rather than calling raw outgoingPi feedback.
    off=m.mpo(p.core.choose_terms(m,whole=True));no,ne=p.core.advance(m,neutral,off,dt=p.DT,label='neutral_field_decoupled',initial_time=p.T0)
    co,_=p.load_saved(m,'charged_field_decoupled');cf,_=p.load_saved(m,'charged_full');nf,_=p.load_saved(m,'neutral_full')
    q=lambda state:np.asarray(p.observe(m,state)['physical_field_Pi'])
    full=(q(cf)-q(nf))-(q(charged)-q(neutral));free=(q(co)-q(no))-(q(charged)-q(neutral));difference=full-free
    base=json.loads((ROOT/(P+p.EPOCH+'_feedback_result.json')).read_text());active=np.asarray(base['dynamic_field_to_charged_cell_charge']);control=np.asarray(base['field_decoupled_intercell_hopOFF_control']);matter=active-control
    record=dict(kind='OWN_RECIPROCAL_SAME_H_BODY_FIELD_RESPONSE_WITH_FREE_DRESSING_CONTROL',field_kick_consumer=base,dynamic_charge_to_Pi_full=full.tolist(),dynamic_charge_to_Pi_field_decoupled_same_initial_state=free.tolist(),charge_to_canonical_Pi_feedback_active_minus_free=difference.tolist(),field_to_charged_cell_charge_active_minus_decoupled=matter.tolist(),neutral_OFF_evolution=ne,physics_assertions=dict(charge_to_field_feedback_nonzero=bool(np.linalg.norm(difference)>1e-12),field_to_charge_feedback_nonzero=bool(np.linalg.norm(matter)>1e-12)),limits=['Pi is the CANONICAL mode momentum; not relabelled as full Gauss electric edge flux','OFF keeps rotatingR/ions/electrons and lab convection. It is field-coupling/intercell-hopOFF, NOT whole charge-motionOFF','Source/fullWeyl initial dressing, PaP incoming-control preparation, free propagation and coupled response explicitly separated','All active paths use SAME frozen full-H; finite regulator/dt/chi16 and incomplete cavity Noether remain unpaid'])
    kicked_final,_=p.load_saved(m,'charged_kicked_full')
    record['actual_regulator_measurements']={name:regulator_measurements(m,state) for name,state in dict(neutral_initial=neutral,charged_initial=charged,neutral_final=nf,charged_final=cf,charged_kicked_final=kicked_final).items()}
    (ROOT/(P+p.EPOCH+'_reciprocal_result.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    assert all(record['physics_assertions'].values()),record['physics_assertions']


def refine(m,neutral,charged,preparation):
    records={}
    for name,state in (('charged',charged),('neutral',neutral)):
        fine,ledger=p.core.advance(m,state,dt=p.DT/2,steps=2,label=name+'_full_halfdt',initial_time=p.T0);coarse,_=p.load_saved(m,name+'_full');a=p.observe(m,fine);b=p.observe(m,coarse)
        records[name]=dict(fine=a,coarse=b,half_dt_evolution=ledger,fine_minus_coarse={key:(np.asarray(a[key])-b[key]).tolist() for key in ('physical_charge_cells','local_LEFT','local_LEFT_Casimir','local_r2','field_Q','physical_field_Pi')})
        (ROOT/(P+p.EPOCH+'_dtcontrol_result.json')).write_text(json.dumps(dict(kind='OWN_FULL_H_SAME_AGE_DT_HALVDT_TRANSPORT_DATA',records=records,meaning='Actual finite dt control, NOT basis/bond/exterior/continuum convergence or an observable error bound from SVD eps.'),indent=2)+'\n')
    print(json.dumps(records),flush=True)


def ordered(m,neutral,charged,preparation):
    p.currents(m,neutral,charged,preparation)
    currents={};receipts={}
    for mode in (0,4):currents[mode],receipts[mode]=p.load_cached_current(m,mode)
    observer,oe=p.core.advance(m,charged,dt=.002,label='current_order_observer',initial_time=p.T0);images={};image_receipts={}
    for mode,operator in currents.items():
        image=charged.copy();error=operator.apply(image,{'trunc_params':{'chi_max':16,'svd_min':1e-12}});before=p.core.norm2(image)
        if before==0:
            # EXACT linear U*0=0; never normalize/canonicalize an exact zero.
            evolved=image;ledger=dict(exact_zero_image=True,dt=.002)
        else:evolved,ledger=p.core.advance(m,image,dt=.002,label='current_order_image_'+str(mode),initial_time=p.T0)
        images[mode]=evolved;image_receipts[mode]=dict(pre_evolution_norm2=before,operator_image_svd_eps=float(error.eps),evolution=ledger)
    records={}
    for a,b in ((0,4),(4,0)):
        forward=p.core.law.physical_element(observer,currents[a],images[b]);reverse=p.core.law.physical_element(images[b],currents[a],observer)
        records[str((a,b))]=dict(C_a_t_b_0=packcomplex(forward),C_b_0_a_t=packcomplex(reverse),minus_i_commutator=packcomplex(-1j*(forward-reverse)),Hermitian_reverse_conjugacy_residual=float(abs(reverse-forward.conjugate())))
    result=dict(kind='OWN_ACTUAL_FULL_H_CANONICAL_CURRENT_TEMPORAL_ORDERS',time_separation=.002,source_time=p.T0,orders=records,initial_state='OWN polarized dressed charged source, not assumed stationary',observer_evolution=oe,current_image_receipts=image_receipts,current_cache_receipts=receipts,formula='C_a(t)b(0)=<Upsi|Ja|U Jb psi>. Reverse=<U Jb psi|Ja|Upsi> INDEPENDENTLY contracted; C04(t) is NOT silently equated with C40(t).',limits='Bare canonical current commutators at finite dt/bond/field cut. Contact is separately evaluated; this is not a gauge-complete scattering amplitude, photon pole, current range/lifetime or full-H norm from sum-of-piece squares.')
    (ROOT/(P+p.EPOCH+'_timeorders_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


def contact(m,neutral,charged,preparation):
    original=m.register;m.register=MethodType(fast_register,m)
    try:
        # An actually non-Hermitian newly registered RF matrix, explicit complex
        # MPO action and independent direct-product contraction test HC opt-out.
        seed=m.seed();A=np.zeros((m.nf,m.nf),complex);A[1,0]=1;A=np.kron(np.eye(m.nR),A);name=m.register(m.rfsite,A);bra=seed.copy();bra.apply_local_op(m.rfsite,name,unitary=False,renormalize=False)
        terms=[(2+.3j,[(name,m.rfsite)])];actual=p.core.law.physical_element(bra,m.mpo_small(terms),seed);direct=p.core.product_element(m,terms,bra,seed)
        if abs(actual-direct)>1e-12 or abs(actual-(2+.3j))>1e-12:raise ValueError(('native explicit-HC-metadata representation smoke failed',actual,direct))
        oldstorage.EPOCH=p.EPOCH;operator=oldstorage.source_operator(m,0,4)
    finally:m.register=original
    states=dict(neutral_initial=neutral,charged_initial=charged)
    for name in ('neutral','charged'):states[name+'_final']=p.load_saved(m,name+'_full')[0]
    values={name:float(p.core.law.physical_element(p.core.normalized(state),operator,p.core.normalized(state)).real) for name,state in states.items()}
    result=dict(kind='OWN_COMPLETE_FIXED_CANONICAL_K04_CONTACT_ON_PHYSICAL_TRANSPORT_STATES',definition='+d2H/(da0 da4) at FIXED canonical variables, not derivative at fixed mechanical velocity, not uniform pure-gauge transformation for finiteq',values=values,term_count=len(operator.terms),new_registration_smoke=dict(actual_complex_MPO_element=packcomplex(actual),independent_direct_product_element=packcomplex(direct),error=float(abs(actual-direct)),native_option='hc=False skips only quadratic NAME inference; all dense matrices, chargelegs, permutation and explicit adjoints unchanged'),scope='Actual full frozen Model.lab_source analytic derivatives of every uncompressed phase and mechanical connection-square term, BEFORE projection. cross/R-connection first derivative has zero second derivative; all g/r/CAR factors retained. No altered H/source coefficients, shared-file edit, producer-state replay or new frame domain.',cache_reference=P+p.EPOCH+'_K0_4_operator.json',limits='Contact+separate temporal current data do not establish full continuum/bond/exterior or Noether-cavity closure, finite-basis photon gap, scattering or inertial mass.')
    (ROOT/(P+p.EPOCH+'_contact_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


def twolegs(m,neutral,charged,preparation):
    polarization=np.array([complex(re,im) for re,im in preparation['polarization']]);initial_retained=preparation['source_quality']['retained_source_norm2'];records={}
    for name,suffix in (('coarse','_full'),('halfdt','_full_halfdt')):
        nt,nrecord=p.load_saved(m,'neutral'+suffix);ct,crecord=p.load_saved(m,'charged'+suffix)
        observer,raw=m.source(nt,0,polarization,preparation['create'],chi=16);quality=dict(m.last_source_quality);weight=float(np.sqrt(initial_retained*quality['retained_source_norm2']))
        amplitude=weight*observer.overlap(ct)
        nc=sum(v['restored_scalar_phase_exponent'] for v in nrecord['ledgers']);cc=sum(v['restored_scalar_phase_exponent'] for v in crecord['ledgers'])
        wrong_bra=observer.copy();wrong_ket=ct.copy()
        wrong_bra.set_B(0,wrong_bra._B[0]*np.exp(1j*nc),form=wrong_bra.form[0]);wrong_ket.set_B(0,wrong_ket._B[0]*np.exp(1j*cc),form=wrong_ket.form[0])
        wrong=weight*wrong_bra.overlap(wrong_ket);prediction=amplitude*np.exp(1j*(cc-nc));error=float(abs(wrong-prediction))
        records[name]=dict(physical_source_weighted_twoleg=packcomplex(amplitude),raw_observer_source_strength=raw,retained_endpoint_weight_product_sqrt=weight,observer_source_quality=quality,neutral_scalar_phase_exponent=nc,charged_scalar_phase_exponent=cc,WRONG_explicitly_removed_fullH_centering_phases_twoleg=packcomplex(wrong),wrong_vs_physical_difference=float(abs(wrong-amplitude)),independent_phase_prediction_error=error,raw_charged_state_norm2=p.core.norm2(ct),raw_neutral_state_norm2=p.core.norm2(nt))
        if error>1e-12:raise ValueError(('actual fullH phase-removal control mismatch',name,error))
        (ROOT/(P+p.EPOCH+'_twolegs_result.json')).write_text(json.dumps(dict(kind='OWN_FULLCAR32_PHASE_SENSITIVE_SOURCE_WEIGHTED_TWOLEG',records=records,formula='<U neutral|Sdag|U S neutral> represented with RETAINED compressed source strengths at BOTH endpoints; normalized-source overlap alone is not G.',meaning='Own actual full-H states, global scalar phases in tensors and real norm metadata; no producer trajectory or eigenphase assumption. Raw endpoint strengths are not Born probabilities; finite dt/chi/source/g/R/field cuts remain, no mass/pole/lifetime conclusion.'),indent=2)+'\n')
    difference=np.array(records['halfdt']['physical_source_weighted_twoleg'])-records['coarse']['physical_source_weighted_twoleg'];print(json.dumps(dict(records=records,halfdt_minus_coarse=difference.tolist())),flush=True)


def worker(mode,modelpath,sha):
    m,compiled=p.core.load_compiled(modelpath,sha);neutral,charged,preparation=p.initial_pair(m)
    if mode=='reciprocal':reciprocal(m,neutral,charged,preparation)
    elif mode=='refine':refine(m,neutral,charged,preparation)
    elif mode=='ordered':ordered(m,neutral,charged,preparation)
    elif mode=='contact':contact(m,neutral,charged,preparation)
    elif mode=='twolegs':twolegs(m,neutral,charged,preparation)
    else:raise ValueError(mode)
    (ROOT/(P+p.EPOCH+'_'+mode+'_evolution_ledger.json')).write_text(json.dumps(dict(compiled_operator=compiled,ledgers=p.core.ADVANCE_LEDGERS),indent=2)+'\n')


def capture(mode,attempt,modelpath,sha):
    tag=P+attempt;deps=set();dependencies(Path(__file__).resolve(),deps);hashes={path.name:p.digest(path) for path in sorted(deps)}
    with zipfile.ZipFile(ROOT/(tag+'_source_capsule.zip'),'x',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(deps):archive.write(path,path.name)
    command=[PY,__file__,'worker',mode,modelpath,sha];inputs=dict(kind='OWN_TRANSPORT_FIRST_SAME_FROZEN_H_DEFERRED_OBSERVERS',command=command,capture_epoch=p.EPOCH,source_sha256=hashes,operator_input_sha256=sha,own_preparation=P+p.EPOCH+'_preparation.json',limits='One streamed frozen Model per sequential worker, no heavy live clone, no producer state or producerCLI replay.')
    (ROOT/(tag+'_input.json')).write_text(json.dumps(inputs,indent=2)+'\n');env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',REVIEW49_INPUT_RECEIPT=tag+'_input.json');started=time.time()
    with (ROOT/(tag+'_stdout.txt')).open('x') as stdout,(ROOT/(tag+'_stderr.txt')).open('x') as stderr:child=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    receipt=dict(natural_exit=child.returncode,elapsed_seconds=time.time()-started,sources_unchanged={path.name:p.digest(path)==hashes[path.name] for path in sorted(deps)})
    (ROOT/(tag+'_exit.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True);sys.exit(child.returncode)

if __name__=='__main__':
    if sys.argv[1]=='worker':worker(*sys.argv[2:])
    else:capture(*sys.argv[1:])
