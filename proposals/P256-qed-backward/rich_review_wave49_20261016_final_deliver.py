"""Finish only completed physical controls/cross-checks; no new experiment."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_';E='transport_priority2'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def load(name):return json.loads((ROOT/(P+name)).read_text())


def main():
    carrier=load(E+'_carrier_result.json');reciprocal=load(E+'_reciprocal_result.json');feedback=reciprocal['field_kick_consumer'];current=load('priority_stream_current_result.json');raw=load('priority_raw_audit_result.json');direct=load('priority_transport_raw_audit_result.json');typing=load('final_language_result.json');exits={name:load(name+'_exit.json') for name in ('priority_transport2','priority_reciprocal1','priority_raw_audit','priority_transport_raw_audit','priority_stream_current')}
    source=carrier['preparation'];assert len(feedback['states'])==8
    scope='Parent latest bounded finish instruction: ONLY already-running same-H field/free controls and compact synthesis/capsule; preserve current/readout as cross-check, not dynamics accuracy. No K catalogue, extra convergence campaign or new experiments.'
    verdict=dict(kind='BOUNDED_EXECUTED_LOCAL_FRAME_AND_FINITE_FIELD_REVIEW_NOT_FULL_H_CONVERGENCE',scope=scope,fullCAR32_operator_sha256=source['compiled_operator_sha256'],actual_carrier_assertions=carrier['physics_assertions'],actual_reciprocal_assertions=reciprocal['physics_assertions'],scientific_exits=exits,typing=dict(natural_exit=typing['natural_exit'],summary=typing['diagnostics'].get('summary')),accuracy='Observed legal approximate-MPS states and independent readout agreement; chi16/dt/field/R/g/exterior dynamics accuracy remains UNRESOLVED. SVD epsilon sums are not observable error bars.',deferred_NOT_approved=['Combined47 K04/contact catalogue','Extra dt/bond/field convergence experiments','Unrun historical combined current-timeorder and twoleg branches','Full-H/continuum/global-ground/free-carrier/photon/mass conclusions'])
    (ROOT/(P+'final_verdict.json')).write_text(json.dumps(verdict,indent=2)+'\n')
    sourcePi=np.asarray(raw['states']['charged']['canonical_field_Pi_initial']);charge=np.asarray(carrier['physical_excess_charge_change']);spin=np.asarray(carrier['physical_Noether_density_change']);rfield=np.asarray(reciprocal['charge_to_canonical_Pi_feedback_active_minus_free']);rbody=np.asarray(reciprocal['field_to_charged_cell_charge_active_minus_decoupled']);density_error=max(value for item in direct['native_MPO_vs_independent_raw_network_errors'].values() for value in item.values())
    lines=[
        'REVIEW49 — SEPARATE BOUNDED EXECUTED VERDICTS; NOT PHOTON/PARTICLE/FULL-H CONVERGENCE APPROVAL.',
        'Final scope is parent-directed bounded finish: completed47 carrier pair + already-running same-H field/free controls + readout cross-checks; NO K catalogue, extra convergence campaign or new experiments.',
        'No ancestor producerCLI, borrowed solved/evolved state, graph/plan/public/world mutation, nested agent or lower-Hilbert replacement.',
        'LOCAL PHYSICAL LEFT / CHARGE — own executed discrimination.',
        'Own two-cell source-image exact-H LEFT-z[.5,0]->[.380841487813,.119158512187], excess charge[1,0]->[.761682975626,.238317024374]; normdrift<5e-16.',
        'That early image law omits localN3/N6 and r/R/EM, NOT fullCAR32 acceptance. Wrong pinned-global-spin/n/N labels reached assertions; LEFT/source commutator1.76e-16, central seam7.62e-16.',
        'Own actual47 primitive audit retains all55g, source/LEFT commutators<9e-14, sewing2.45e-14; full-beforecut Q/QQ/PP exact, displacement4.36e-17. Angular source exterior up to.6 stays debt.',
        'ACTUAL48 — own fullCAR8 quantum g/r/R +8real finite-q fields, BOTH charged branches504000 each, all8source bands.',
        'Own fixed ORIGINAL canonical external-H rebuild<=1.91e-16; currentFD<=3.552e-9, complete contact<=5.201e-7, mixedcontact<=2.824e-7. Raw/source/kinetic products precede cuts.',
        'HoppingOFF convection-current image norm.137171; whole charge-motionOFF source0. Wrong omitted Rconnection and all-finiteq-puregauge reach assertions; finite-q loopnorm.397872, independent paths/derivatives/q0 checked.',
        'Fresh same-H kickwork~.0056982, energy drift<2.9e-15, both current timeorders actually executed. Hermite field error<2.45e-15; wrong cut exponential.912516, wrong cutPi²15.',
        'Own paired0->6(t=.037,kick.19): raw additionPi .0142471824 versus whole-motionOFF echo.0141468662; calibrated excess.000100316, NOT raw echo as photon.',
        'Corrected finite-q versus own same8-coordinate q0/matched evolving-neutral: full-minusOFF charged field differences +9.43430e-5(0->6), -3.03053e-5(2->4), removal agrees~1e-10; microsite charge~2-2.8e-6/spin~1e-9.',
        'q0 keeps4nonorthogonal cos profiles and sin spectators, NOT44 isotropic3-mode spectral equality. Field Pi-dot/current and qdot/Pi cut debts~1 remain: no canonical/IR/pole closure claim.',
        'NUMERICAL INTEGRATION CORRECTION — not angular exterior convergence.',
        'Own fixed47 R35/field9 Lambda² old7SO3-root vs24-grid difference.00252790;24vs28 difference1.80e-15. IntegerWigner check7.45e-16; LEFT commutator8.21e-15.',
        'Own actual47 Gaunt/direct Peierls*R/Lambda²/mixedcontact/radial-Wilson checks<1e-11. Auxiliaryl<=4 expands retainedl<=2 bra/ket PRODUCTS, not extra/dropped evolved multiplets.',
        'Uniformq0 polynomial positivecontrol1.01e-15 survives; half-integer/sewn SU2g keeps4pi/55labels. MaterialSO3 periodicity is not exchanged with g. Older finite-cubature results retain original matrices, not relabelled corrected.',
        'BARE/FREE PICTURE CHECKS — own full-W preparation/unbounded free propagation and wrong H0/PaP/crossprotocol controls executed, source/input/natural0 receipts retained.',
        'These are source-kernel/free-wave evidence ONLY, NOT coupled-body evolution or a substitute for the actual47 states below.',
        'ACTUAL47 / COMBINED47 — OWN finite-time one-H states, full formal CAR32/g55x4/r3x4/R35/field9 law retained.',
        f"Frozen OPERATOR-ONLY input SHA {source['compiled_operator_sha256']}; no duplicate compiler or Model/state substitution. Actual global quantumr/cone is not charge-shell radius.08.",
        'Own raw source at.004: physical excess charge[-1,0,0,0], local LEFT-z.499999471828; wrong globalspin*n/N density error.407540935464 reaches physical assertion, all4sewing1 within3e-16.',
        f"Source regulated raw norm²={source['source_quality']['exact_regulated_source_norm2']:.12g}, retained={source['source_quality']['retained_source_norm2']:.12g}, sourceSVD eps={source['source_quality']['svd_eps']:.6g}; RETAINED endpoint strength, not raw/unconditional probability.",
        f"Actual source.004->.008 under SAME fullH, SAME-age OWN evolving neutral: excess charge change={charge.tolist()}.",
        f"Actual physical local LEFT density change off-cell norm={np.linalg.norm(spin[1:]):.12g}; cell1 deltaLz={spin[1,2]:.12g}. Not inferred from n/N or a global spin label. Only0->1 charge route resolved at this finite setting; no absent-channel theorem.",
        f"Independent raw-tensor charge/LEFT/Pi contraction matches native on all4 states<={density_error:.3g}; direct physical norm check1.11e-16. Legacy/new checkpoint B actual values AND tobytes identical, no hidden phase normalization.",
        'MPS norm metadata REAL nonnegative; scalar phases in tensors; every completed OWN stage raw/ledger committed before later builders. Future payloads carry actual virtualqflat/qconj/qtotal and Schmidt data, not default-cutoff charge inference.',
        f"Carrier raw norms² charged/neutral={carrier['evolution']['charged']['raw_norm2']:.12g}/{carrier['evolution']['neutral_same_age']['raw_norm2']:.12g}; SVD eps sums={carrier['evolution']['charged']['svd_eps_sum']:.6g}/{carrier['evolution']['neutral_same_age']['svd_eps_sum']:.6g}. NOT observable error bars; chi16/dt/field accuracy UNRESOLVED.",
        'OWN SAME-H FIELD KICK / FREE-DRESSING CONTROLS — all actual states at.004/.008, not plot collage.',
        f"Incoming exp(i*.19*P Q0 P) only: finite projected-Q protocol, NOT full-W/pure photon. Own kickwork neutral/charged={feedback['kick_work_same_H']}.",
        f"Actual source ALREADY has field-shell weight{raw['states']['charged']['field_top_total_Fock_shell_weight']:.12g}, Pi3={sourcePi[3]:.12g}, Pi5={sourcePi[5]:.12g}, Pi covariance06=-.00694321529/24=.00219556338. Initial means/covariances measured, not receiver purity assumed.",
        f"Charge->CANONICAL modePi ACTIVE minus matched SAME-initial-state FREE/OFF doublecontrast={rfield.tolist()}, L2={np.linalg.norm(rfield):.12g}.",
        f"Incoming field->charged-cell physical-charge ACTIVE minus diagnostic OFF response={rbody.tolist()}, L2={np.linalg.norm(rbody):.12g}.",
        f"Paired full-H source-specific kick charge contrast (charged minus evolving-neutral)={feedback['charge_specific_kick_doublecontrast']}; local LEFT kick-response matrices retained separately.",
        f"Actual field/free physical assertions={reciprocal['physics_assertions']}. A failed predicate remains a bounded physical finding, not suppressed or labelled convergence.",
        'OFF removes field coupling/intercell-hop paths but R/ions/electrons still rotate/convect; NOT whole charge-motionOFF. Canonical A-source0 is not lab current0. Free initial dressing and finite-P-Q echoes are separately calibrated.',
        'Actual g/R/field-shell, cut coordinate-commutator, quantum cone-r marginal diagnostics retained for own initial/final/kicked states. State weights are not exterior or observable-error bounds.',
        'FULL CANONICAL CURRENT CROSS-CHECK — NOT a dynamics-accuracy verdict.',
        'J0/J4 each ALL2960terms/1808RF arrays, fixed-canonical mechanical+Peierls source on own4states. Read40authentic nonRF sites then stream RF one-at-time; NO Model clone, peakRSS616884KiB.',
        'Native nonzero complex primitive [3.61338863194e-9,-3.07248184496e-7] matches independent raw network4.31e-22; whole-sum imaginary residual<=1.1e-20.',
        'J0 charged initial/final=-1.23425e-8/-2.82373e-8 versus neutral=-8.96019e-9/-3.87546e-7; J4 charged=2.16554e-9/-5.02945e-9 versus neutral=-2.72228e-14/-6.21343e-8. These support operator/readout ownership, not resolved H accuracy.',
        'ADVERSE / REPAIR EPOCHS — all retained; setup failures are not physical exclusions.',
        'Own old corrected constructor SIGKILL-9 after6edges/3529terms, no returned Model/assertion; cause UNCONFIRMED. Earlier explicit cancellations distinct.47 owns the ONE lazy-sum compilation; no duplicate giant constructor.',
        'Old8h natural1: valid primitive/phase evidence, transient neutral.004 then empty source-MPO API failed; transient was NOT saved, no charged trajectory. Never relabelled current saved preparation.',
        'Parent repaired shared native-zero same-type ABI(c33...): own cachedModel OFFzero/nonzero/scalarphase/checkpoint smokes executed. Nativezero apply_naively norm0; exp0 via EXISTING PhysicalOperatorSum identity; never normalize/SVD exact0.',
        'Prior fields.py text epoch difference was DOCSTRING-only after AST comparison, not a physics-source change blamed for the empty-MPO error.',
        'Later6.48h cache-builder intentionally SIGINT(-2) ONLY own child, sourceguardsTRUE; traceback confirmed K04 RF HC-name inference, no charged motion. Prep/J0/J4 survived; not natural physics failure.',
        'priority_transport1 natural1/29s at legacy chi16 virtual-charge reconstruction, BEFORE source/trajectory. New exact-support/direct-native-MPS restore and explicit future charge metadata reused OWN physical prep, not an8h confirmation rerun.',
        f"TERMINALS: carrier natural{exits['priority_transport2']['natural_exit']}/{exits['priority_transport2']['elapsed_seconds']:.3f}s; field/free natural{exits['priority_reciprocal1']['natural_exit']}/{exits['priority_reciprocal1']['elapsed_seconds']:.3f}s; raw source/audit/current natural0, receipts and all source/input flags retained.",
        'UNEXERCISED by explicit final scope: K/contact catalogue, extra dt/bond/field convergence, historical combined47 current-timeorder/twoleg branches. No approval of those branches or catalogue completeness.',
        'METRIC / PHYSICAL LIMITS',
        'Declared cone r³dr with Cg=j(j+1) is dual-LEFT SU2 metric, NOT flat-quaternion/Frobenius cone4Cg. CAR/central sewing/quantization/topology are INPUTS, not derived classical QM.',
        'Finite8-q set is not closed globalSO3; grounded-sphere boundary20/charge shells.08 and all g/R/r/field/exterior debts remain. Canonical Pi/Pi+P amplitudes are not SI electric field units.',
        'No localization/stability/unique-species theorem, free continuous carrier/COM, global ground/full-H convergence, on-shell photon, Lorentz/inertial mass, current range, Pcm=J or finite-gap-as-mass claim.',
        'Separate completed Common C1(e740...), literal quantum loops(3e3a...) and full108 leading recoil(322b...) are DIFFERENT laws/states, not replacement evolved suppliers for this one-H47 consumer.',
        f"Final selected actual13-file Pyright exit{typing['natural_exit']}: {typing['diagnostics'].get('summary')}; real rootedLSP exit{typing['rooted_LSP']['natural_exit']}. Historical18-file/55error report retained, not falsely green or rerun for approval.",
        'Delivery is a compact COMPLETION-DELTA source/state/raw/exit capsule over the immutable earlier handoff capsule; giant compiled Model/RF operator caches are external hash-bound INPUTS, never copied into every evidence capsule.',
    ]
    assert len(lines)<=80,len(lines)
    (ROOT/(P+'final_synthesis.txt')).write_text('\n'.join(lines)+'\n')
    base=ROOT/(P+'handoff_capsule.zip');basehash=digest(base);capsule=ROOT/(P+'final_capsule.zip');excluded={base.name,capsule.name,P+'final_capsule_receipt.json',P+'final_manifest.json'};included=[];base_members={};external={}
    with zipfile.ZipFile(base) as archive:
        existing={info.filename for info in archive.infolist() if not info.is_dir()}
        for path in sorted(ROOT.glob(P+'*')):
            if not path.is_file() or path.name in excluded:continue
            sha=digest(path);info=dict(bytes=path.stat().st_size,sha256=sha)
            if path.name.endswith('_operator.npz'):
                external[path.name]=dict(**info,meaning='Hash-bound derivative OPERATOR input, not an evolved state; not duplicated.');continue
            if path.name in existing:
                with archive.open(path.name) as stream:same=hashlib.file_digest(stream,'sha256').hexdigest()==sha
                if same:base_members[path.name]=info;continue
            included.append((path,info))
    interface=dict(kind=verdict['kind'],scope=scope,base_handoff_capsule=dict(path=base.name,sha256=basehash,meaning='Historical immutable source/raw/exit layer; its old partial synthesis is NOT the current final verdict.'),completion_delta_capsule=capsule.name,final_synthesis=P+'final_synthesis.txt',final_verdict=P+'final_verdict.json',actual_47_carrier=P+E+'_carrier_result.json',actual_47_field_free=P+E+'_reciprocal_result.json',actual_canonical_current=P+'priority_stream_current_result.json',independent_raw_measurement=P+'priority_transport_raw_audit_result.json',frozen_compiled_operator=dict(path='rich_local_frame_spatial_carrier_20261016_model.pkl',sha256=source['compiled_operator_sha256']),external_operator_inputs=external,deferred_NOT_approved=verdict['deferred_NOT_approved'],typing=verdict['typing'])
    interface_path=ROOT/(P+'final_interface.json');interface_path.write_text(json.dumps(interface,indent=2)+'\n');included=[(path,info) for path,info in included if path!=interface_path];included.append((interface_path,dict(bytes=interface_path.stat().st_size,sha256=digest(interface_path))))
    manifest=dict(base_capsule=interface['base_handoff_capsule'],base_unchanged_members=base_members,delta_members={path.name:info for path,info in included},external_operator_inputs=external);manifest_path=ROOT/(P+'final_manifest.json');manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(capsule,'x',zipfile.ZIP_DEFLATED) as archive:
        for path,info in included:archive.write(path,path.name)
        archive.write(manifest_path,manifest_path.name)
    receipt=dict(capsule=capsule.name,sha256=digest(capsule),bytes=capsule.stat().st_size,synthesis_lines=len(lines),base_handoff_sha256=basehash,physical_carrier_exit=exits['priority_transport2']['natural_exit'],physical_field_free_exit=exits['priority_reciprocal1']['natural_exit'],typing_natural_exit=typing['natural_exit'],scope='Completed explicitly bounded physical/measurement review; convergence, K catalogue and other deferred branches NOT approved.')
    (ROOT/(P+'final_capsule_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
