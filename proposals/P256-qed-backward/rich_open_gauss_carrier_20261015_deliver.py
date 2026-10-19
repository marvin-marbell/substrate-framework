"""Bind actual outputs and immutable scientific inputs to a scoped delivery."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_open_gauss_carrier_20261015_'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    result=json.loads((ROOT/(P+'charged_result.json')).read_text())
    language=json.loads((ROOT/(P+'language_final.json')).read_text())
    branches={}
    for name,b in result['branches'].items():
        first,last=b['samples'][0],b['samples'][-1]
        branches[name]=dict(dimension=b['dimension'],raw_source_strength_anchor=b['raw_source_norm_squared'][0],raw_source_strength_other_sites=b['raw_source_norm_squared'][1:],initial_right_charge=first['right_charge'],delta_right_charge=last['delta_right_charge_own_t0'],delta_right_electric_energy=last['delta_right_electric_energy_own_t0'],delta_right_flux=last['delta_flux_own_t0'],signed_G_site4_t0=first['signed_green_from_anchor'][4],signed_G_site4_t2=last['signed_green_from_anchor'][4],path_cut_right_charge_change=b['controls']['path_cut']['delta_right_charge_own_t0'],off_right_charge_change=b['controls']['hopping_off']['delta_right_charge_own_t0'],path_cut_defeat_difference=b['controls']['path_cut']['full_minus_control_right_charge'],source_energy_minus_reference=b['source_energy_minus_reference'],boundary_cost=b['boundary_charging_cost'],algebra=b['algebra'],numerical_resolution=b['resolution'],norm_drift=b['max_norm_drift'],energy_drift=b['max_energy_drift'])
    exits={stage:json.loads((ROOT/(P+stage+'_exit.json')).read_text()) for stage in ('reference','charged')}
    summary=dict(kind='SCOPED_NEW_CHARGED_FIELD_MECHANISM_NOT_CAR36_REPLACEMENT',law='I16,t1,U4; ions1, full finite physical links plus real outward boundary; Hcharge2sum(n-1)^2',reference=result['reference'],branches=branches,raw_dressed_anticommutator=result['raw_local_dressed_anticommutator'],decision='Retain observed finite coupled charged matter/field mechanism only. Actual prepared source charge motion is compared to own t0 and same-state bridge-cut, never to an assumed zero offsite Green. Finite spectral charging energies are not particle masses.',current_and_feedback='Literal same-H primitive J=-i(T-T†) obeys exact all-basis flux equation and matter continuity. Complete charge/cloud/current samples retained for both branches. Flux changes and density changes satisfy finite-time Gauss relation; no imported field propagator.',source_semantics=result['green_convention'],spin_limit=result['spinor_scope'],controls='Same prepared charged state. QR is conserved exactly by cut and OFF. Right field energy need not be static after a conditioned entangled source; full-H right energy is explicitly marked a diagnostic, while separate control own-energy expectations are retained.',neutral_consumer='GaussPhotonPhase imports these frozen model bytes and newly computed shared neutral reference as declared INPUT; its new response is separately owned, not an independent ground solve.',language=dict(returncode=language['returncode'],summary=language['checker'].get('summary'),LSP_exit=language['LSP']['server_exit'],scope=language['scope']),execution=exits,no_claims=['massless photon','positron','physical-frame Noether spin embedding','free COM','Coulomb law','QCD/proton','Nature','global ground state','canonical/unitary truncated Wilson source','particle mass'],governor=json.loads((ROOT/(P+'input.json')).read_text())['governor'])
    (ROOT/(P+'synthesis.json')).write_text(json.dumps(summary,indent=2)+'\n')
    helpers=['rich_photon_fullq_20261014_model.py','rich_photon_20261014_loop.py','rich_review_wave12_20261015_diagnose.py']
    files=[p for p in ROOT.glob(P+'*') if p.is_file() and p.suffix!='.zip' and not p.name.endswith(('manifest.json','delivery_receipt.json'))]
    files += [ROOT/n for n in helpers]
    manifest=dict(owned_prefix=P,files={p.name:dict(SHA256=sha(p),bytes=p.stat().st_size) for p in files},helper_semantics='Only frozen pure geometry/fock/CAR functions and tooling transport imported; no producer CLI replay or old evolved state')
    (ROOT/(P+'manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    files.append(ROOT/(P+'manifest.json'))
    capsule=ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,p.name)
    receipt=dict(capsule=capsule.name,SHA256=sha(capsule),bytes=capsule.stat().st_size,manifest_SHA256=sha(ROOT/(P+'manifest.json')))
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(receipt=receipt,branches=branches)),flush=True)
if __name__=='__main__':
    main()
