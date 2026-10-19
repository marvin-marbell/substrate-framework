"""Actual compact native F2 operator-norm certificate, not a Born simulator."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from original_charged_principal import portable
from prepared_neutral_shared_photon_state import load_shared_native_cache
from native_receiver_norm_certificate import certificate
from fixed_lab_receiver_admission import admit_native


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','manifest','frozen-effect'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();paths={name:getattr(args,name.replace('-','_')) for name in ('cache','manifest','frozen-effect')}
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    model,data=load_shared_native_cache(args.cache,args.manifest)
    with np.load(args.frozen_effect,allow_pickle=False) as stored:f2=stored['frozen_E2_coefficients'].copy()
    result=certificate(model,data,f2)
    admission=admit_native(model,data,f2)
    result.update(inputs_sha256=hashes,unchanged_native_cached_Gal_F2_norm_squared=float(np.vdot(f2,data['charged_Gram']@f2).real),
        exact_E2_definition='E2=|unchangedF2><unchangedF2| under TRUE compact field Hilbert metric; ||E2||=||F2||². Cached Gal vector normalization is not a POVM proof.',
        actual_Code_effect_norm_M_NOT_assumed_one=True,
        LEGAL_receiver_contract='Original TWO jointly Loewdin-normalized OUT functions under original physical metric (or bounded wholeunionPi/cappedF) on original all64/harmonic/LEGAL factory; this N3 savedGalvector defect cannot disprove that distinct specification.',
        no_receiver_fitting_renormalization_filter_new_prepare_or_fake_Born=True,
        no_prior_H_G_scientific_supplier_rerun_or_closure=True)
    result.update(independent_interval_native_receiver_admission=admission,
        native_rankone_E2_NOT_POVM=admission['decision']=='REJECT_BINARY_EFFECT',
        actual_effect_norm_M_interval_upper=admission['actual_E2_operator_norm_upper'],
        admission_scope='Independent Arb128 original mean witness and ALL extras plus the analytic actual compact SOURCE theorem; floating candidate/gamma_n does not certify mean LAPACK. Native unchanged effect only, not LEGAL disproof.')
    if any(hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[name] for name,path in paths.items()):raise RuntimeError('Original input changed')
    print(json.dumps(portable(result),sort_keys=True))


if __name__=='__main__':main()
