"""Actual47 zero-derivative ABI and nonzero physical-path/source persistence smoke.
Scope all requires47zero-ABI repair; positive scope checks unchanged nonzero paths.
"""
from pathlib import Path
import json
import sys
import numpy as np
import rich_local_frame_spatial_carrier_20261016_model as law
from rich_review_wave49_20261016_integrated_consumer import load_compiled,choose_terms,norm2,advance
from rich_review_wave49_20261016_integrated_storage import source_operator,save_state,restore

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def run():
    m,binding=load_compiled(sys.argv[1],sys.argv[2]);seed=m.seed()
    scope=sys.argv[3] if len(sys.argv)>3 else 'all'
    if scope not in ('all','positive'):raise ValueError('unknown repair scope')
    name='integrated_repair_smoke' if scope=='all' else 'integrated_repair_positive'
    zero_results={}
    if scope=='all':
        # R kinetic remains: zero owning derivative is not lab charge freeze.
        off_terms=choose_terms(m,whole=True)
        zero=source_operator(m,0,terms=off_terms,label='field_decoupled_J0')
        initial=seed.copy();initial.norm*=2
        matrix=law.physical_element(initial,zero,initial)
        image=initial.copy();zero_action_return=zero.apply_naively(image)
        zero_norm=norm2(image)
        assert abs(matrix)<1e-24 and zero_norm<1e-24
        # NEVER normalize zero image. exp(-it*zero) acts on NONZERO input.
        zero_H=law.PhysicalOperatorSum(m,[])
        identity_state,identity_ledger=advance(m,initial,H=zero_H,dt=.017,label='repair_zero_exponential')
        identity_overlap=initial.overlap(identity_state)/norm2(initial)
        assert abs(identity_overlap-1)<1e-11 and abs(norm2(identity_state)-4)<1e-11
        zero_results=dict(zero_matrix_element=[float(matrix.real),float(matrix.imag)],zero_image_norm2=zero_norm,zero_apply_method='Exact native apply_naively, no zero SVD/canonicalization/normalization',zero_apply_return=str(zero_action_return),zero_exponential_overlap=[float(identity_overlap.real),float(identity_overlap.imag)],zero_exponential_ledger=identity_ledger)
        zero_results['exponential_representation']='Existing PhysicalOperatorSum(empty Hamiltonian); native zero-MPO current tested separately. Native MPO has no evolve by design.'
    # Actual canonical variation of thirteen retained microscopic path pairs.
    # Subset is only an operator ABI positivecontrol, never an evolved new law.
    pair=[entry[0] for entry in m.hop_terms[:26]]
    nonzero=source_operator(m,0,terms=pair,label='repair_paths_J0')
    packet,strength=m.source(seed,0,0,True,chi=16);quality=dict(m.last_source_quality)
    current_image=packet.copy();err=nonzero.apply(current_image,{'compression_method':'zip_up','trunc_params':{'chi_max':16,'svd_min':1e-12}})
    current_norm=norm2(current_image);assert current_norm>1e-12
    reloaded=source_operator(m,0,terms=pair,label='repair_paths_J0')
    reloaded_image=packet.copy();reloaded.apply(reloaded_image,{'compression_method':'zip_up','trunc_params':{'chi_max':16,'svd_min':1e-12}})
    operator_cache_error=abs(current_image.overlap(reloaded_image)/current_norm-1)
    assert operator_cache_error<1e-10
    packet.norm*=2;packet.set_B(0,np.exp(.31j)*packet.get_B(0,form='B'),form='B')
    arrays={};save_state(arrays,'packet',packet);np.savez_compressed(ROOT/(P+name+'_raw.npz'),**arrays)
    restored=restore(m,arrays,'packet');roundtrip=packet.overlap(restored)/norm2(packet)
    assert abs(roundtrip-1)<1e-11 and abs(norm2(restored)-norm2(packet))<1e-11
    scalar=law.PhysicalOperatorSum(m,[m.term(3.7,{m.rsites[0]:np.eye(3)})])
    phase_state,phase_ledger=advance(m,packet,H=scalar,dt=.02,label='repair_saved_scalar_phase')
    amplitude=packet.overlap(phase_state)/norm2(packet);expected=np.exp(-1j*3.7*.02)
    assert abs(amplitude-expected)<1e-10 and abs(norm2(phase_state)-norm2(packet))<1e-10
    # Exercise completed-stage receipt restore without executing that stage again.
    cached,cached_ledger=advance(m,packet,H=scalar,dt=.02,label='repair_saved_scalar_phase')
    cached_error=abs(phase_state.overlap(cached)/norm2(phase_state)-1)
    assert cached_error<1e-10
    result=dict(kind='ACTUAL47_OPERATOR_API_AND_OWN_PERSISTENCE_SMOKE_NOT_TRANSPORT_ACCEPTANCE',scope=scope,compiled_operator_input=binding,nonzero_actual_paths_current_image_norm2=current_norm,nonzero_apply_svd_eps=float(err.eps),derivative_only_operator_cache_roundtrip_error=float(operator_cache_error),source_raw_norm2=strength,source_quality=quality,raw_state_norm2=norm2(packet),restored_state_norm2=norm2(restored),raw_state_roundtrip_overlap=[float(roundtrip.real),float(roundtrip.imag)],scalar_centering_phase_error=float(abs(amplitude-expected)),physical_stage_cache_roundtrip_error=float(cached_error),physical_stage_ledger=phase_ledger,restored_physical_stage_ledger=cached_ledger,meaning='Nonzero current/source positivecontrol and own rawtensor/centering-phase persistence. Scope all additionally tests exact-zero derivative and exp(0)=identity; positive scope does not exercise/claim zero ABI. No zero source normalization; no charged finite-time motion or field feedback inferred. Actual field-decoupled body H retains moving R.')
    result.update(zero_results)
    (ROOT/(P+name+'_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
