"""Exercise original native F2 admission without changing F2, SOURCE or state."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from fractions import Fraction
import sys
import time
import numpy as np
from prepared_neutral_shared_photon_state import load_shared_native_cache
from fixed_lab_receiver_admission import admit_native

EXPECTED={'cache':'10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3',
          'manifest':'90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a',
          'frozen-effect':'3182b40a78e7eb6cfd26d4803fc9f57b67b1ea78d2a44757f773217fbf8b5b84'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legal_effect_binding(descriptor: Path, receipt: Path) -> dict:
    """Inspect original sealed STATIC LEGAL payment; never replay its science."""
    hashes={'descriptor':digest(descriptor),'receipt':digest(receipt)}
    if hashes['descriptor']!='020fb175c9231b607fb0e53b016cf97528af9065a634dedc7ddea82bdac3556c':
        raise ValueError('Different original LEGAL factory')
    desc=json.loads(descriptor.read_bytes());run=json.loads(receipt.read_bytes())
    if run.get('exit_code')!=0 or run.get('natural_exit') is not True or run.get('stderr')!='':
        raise ValueError('LEGAL receipt does not bind a natural successful execution')
    result=json.loads(run['stdout'])
    if result['input_sha256']['descriptor']!=hashes['descriptor']:
        raise ValueError('LEGAL receipt consumed another descriptor')
    if result['identity']!='PNFF-RAW64-coherent-Ref4-sea-Ref0-seaPlus4-ONE-globalfilter-v2':
        raise ValueError('Different original LEGAL preparation')
    # Correct Delta/2 on FULL Root sectors. This checks only that the old
    # STATIC coarse Root coefficient remains paid, not a driven-loop proof.
    root=result['autonomous_ordinary_Root_energy_proof']
    corrected_Q2=2*Fraction(root['Q2_Lambda_minus1over4_coefficient'])
    if corrected_Q2>=2**14:raise ValueError('Corrected FULL Root Delta/2 does not preserve static coarse payment')
    return dict(input_paths={'descriptor':str(descriptor),'receipt':str(receipt)},input_sha256=hashes,
        historical_natural_exit=run['exit_code'],historical_runtime_seconds=run['wall_seconds'],
        preparation_identity=result['identity'],N=desc['N'],physical_family=desc['physical_family'],
        receiver_definition='TWO original raw OUT functions Y in true physical L2; F=Y(YdaggerY)^(-1/2); E2=|F2><F2|, E0=I-E2. Joint Gram, never per-charge/column rescaling.',
        paid_scope=result['scope'],physical_Gram_and_inverse_root_payments={q:{k:result['physical_binary'][q][k] for k in ('physical_joint_Gram_lower','input_absolute_norm_error','endpoint_absolute_operator_error','global_inverse_root_error','frozen_F2_absolute_norm_error')} for q in ('plus','minus')},
        all64_all512_harmonic_R4_retained=result['ordinary_ALL64_ALL512_and_R4_energies_and_phases_retained'],
        corrected_Delta_over2_STATIC_Q2_coefficient=str(corrected_Q2),
        corrected_coarse_Q_norm_payment='2^7 Lambda^(-1/8) retained: corrected Q² coefficient <2^14',
        static_payment_NOT_driven_spatial_rotation_or_joined_particle_or_scattering=True,
        native_F2_artifact_NOT_a_LEGAL_factory_artifact=True,
        existing_proof_receipt_inspected_NOT_rerun=True)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in EXPECTED:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--legal-descriptor',type=Path,required=True)
    p.add_argument('--legal-receipt',type=Path,required=True)
    args=p.parse_args();start=time.monotonic()
    inputs={name:getattr(args,name.replace('-','_')) for name in EXPECTED}
    hashes={name:digest(path) for name,path in inputs.items()}
    if hashes!=EXPECTED:raise ValueError('Different artifact/factory: this admission is bound to unchanged original native bytes')
    root=Path(__file__).resolve().parent
    sources={}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if isinstance(name,str):
            path=Path(name).resolve()
            if path.parent==root and path.suffix=='.py':sources[str(path)]=digest(path)
    model,data=load_shared_native_cache(inputs['cache'],inputs['manifest'])
    with np.load(inputs['frozen-effect'],allow_pickle=False) as stored:f2=stored['frozen_E2_coefficients'].copy()
    result=admit_native(model,data,f2)
    result['original_LEGAL_effect_binding']=legal_effect_binding(args.legal_descriptor,args.legal_receipt)
    cached=float(np.vdot(f2,data['charged_Gram']@f2).real)
    if abs(cached-1)>1e-8:raise ValueError('Wrong native factory: historical cached normalized receiver is not this artifact')
    if any(digest(path)!=hashes[name] for name,path in inputs.items()):raise RuntimeError('Original inputs changed')
    if any(digest(Path(path))!=sha for path,sha in sources.items()):raise RuntimeError('Executed source changed')
    result.update(input_paths={name:str(path) for name,path in inputs.items()},input_sha256=hashes,
        source_sha256=sources,command=sys.argv,runtime_seconds=time.monotonic()-start,
        cached_metric_only_wrong_case={'cached_F2_norm_squared':cached,'cached_metric_would_admit':True,'actual_decision':result['decision']},
        native_factory='load_shared_native_cache N3 accepted seed, original sigma/cutoff/normal/hard15/all55/Ref4; inverse cachedCholesky F2 NOT regenerated',
        state_SOURCE_Z_filter_seed_receiver_changed=False,campaign_closed=False,
        governing_plan='issue-222-mercury-synthesis-continuation-20261007',
        governing_revision='f9a076e0785f47644962a38b973e6b87a329d6ad594a4345a4728f4c689aa3be')
    payload=json.dumps(result,sort_keys=True,indent=2)
    args.output.write_text(payload+'\n')
    print(payload)


if __name__=='__main__':main()
