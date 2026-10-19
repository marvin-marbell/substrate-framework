"""Publish completed natural species physics, <=80-line plain synthesis.
Raw JSON/NPZ and exact native/source capsules remain separate from parent readout.
"""
from pathlib import Path
import hashlib,json,shutil,sys,time,zipfile
ROOT=Path(__file__).resolve().parent
PREFIX='rich_common_native_species_20261016_'


def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb')as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def actual(run):
    if not run.startswith(PREFIX)or'/'in run or'\\'in run:raise ValueError('New owned run required')
    receipt=json.loads((ROOT/(run+'_exit.json')).read_text());result=json.loads((ROOT/(run+'_result.json')).read_text())
    if receipt['returncode']!=0 or not receipt['natural_exit']or receipt['source_input_before']!=receipt['source_input_after']:raise ValueError('Actual immutable natural0 required')
    return receipt,result


def main():
    config=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);started=time.time();carrier_name=config['carrier_run'];joint_name=config['joint_run'];carrier_exit,carrier=actual(carrier_name);joint_exit,joint=actual(joint_name)
    synthesis_path=Path(str(output)+'_synthesis.txt');manifest_path=Path(str(output)+'_manifest.json');capsule_path=Path(str(output)+'_capsule.zip');native_path=Path(str(output)+'_native.zip');receipt_path=Path(str(output)+'_receipt.json')
    for path in (synthesis_path,manifest_path,capsule_path,native_path,receipt_path):
        if path.exists():raise FileExistsError('Never overwrite scientific publication')
    electron=carrier['sectors']['33'];hole=carrier['sectors']['31'];control=joint['ALL_scalar_AND_vector_hops_OFF_control'];lines=[
        'COMMON NATIVE SPECIES: PROVISIONAL SAME-PRIMITIVE ELECTRON + THREE-QUARK INTERACTION CANDIDATE',
        'One actual CAR64 occupied N32Omega and one Phi7/R10/PW19 quantum substrate; geometry/p/H unchanged.',
        'Own N33/N31/N36 Galerkin representations are declared approximations, not identical old finite H by fiat.',
        'No separately evolved baryon/electron tensor, oldN17/body408/CURRENT427 or47/48 solved-state splice.',
        'Primitive spin/charges/SU3/3+1/gauging, negative scalar quadratic/volume and mechanical density are INPUTS.',
        'N33=c† DOWN-WHITE flavour,physical SPIN-UP atsite0 (NOT neutral UP-WHITE flavour); N31=ORIGINAL c of SAMEorbital,unit coefficients.',
        f'Raw addition/removal strengths: {carrier["original_CAR_complement"]["addition_raw"]:.15g}, {carrier["original_CAR_complement"]["removal_raw"]:.15g}; sum={carrier["original_CAR_complement"]["sum"]:.15g}, Omega norm={carrier["omega_norm"]:.15g}.',
        'N31 computed on OWN reference/order,not assumed C partner/proton. SourceG is not stationary by stipulation.',
        f'N33: Q=-1,J=.5,M=.5,body={electron["body_dimension"]},fullPhi={electron["full_Phi_dimension"]},E={electron["energies"][0]:.12g}.',
        f'N31: Q=+1,J=.5,M=-.5,body={hole["body_dimension"]},fullPhi={hole["full_Phi_dimension"]},E={hole["energies"][0]:.12g}.',
        f'Offsets to HOT selected N32 reference: {electron["offset_to_HOT_reference"]:.12g}, {hole["offset_to_HOT_reference"]:.12g}; NOT masses/positive gaps/poles.',
        f'N33 t4 whole netQsite: {electron["events"]["full"][-1]["Q_em_site_mean"]}.',
        f'N31 t4 whole netQsite: {hole["events"]["full"][-1]["Q_em_site_mean"]}.',
        'These show native charged-halfspin motion; rawwhite-count channels and netQ expectations remain DIFFERENT.',
        f'Carrier postmetric Wardmax={max(v for s in carrier["sectors"].values()for a in s["Ward_after_actual_G"].values()for v in a):.6g}; ALLhopsOFF PVM-TVmax={max(e["count_PVM_TV"]for s in carrier["sectors"].values()for e in s["events"]["ALL_scalar_AND_vector_hops_OFF"]):.6g}.',
        'N36=Bdag0 Ldag2 Omega: L is charged DOWN-WHITE,physical SPIN-UP and acts FIRST; B epsilon/sqrt6,L coefficient1,no compactmask.',
        'Four predeclared sourcepairs(0,2),(1,2),(0,3),(1,3); both sourcehalfspinUP give totalJ1,M1.',
        f'Joint actualN36,Q0,body={joint["body_dimension"]},fullPhi={joint["full_independent_Phi_dimension"]},raw sourceG={joint["canonical_joint_source_strength"]:.15g},E={joint["energies"][0]:.12g}.',
        'Whole quarksea charge+5/whitesea-5 sum0; this is one Fock state,not two reference seas.',
        f'Joint no-pulse absolute t4.4 quark compact-countPsite={joint["full_baseline"][-1]["quark_compact_site_probability"]}; white addition-countPsite={joint["full_baseline"][-1]["white_addition_site_probability"]}.',
        f'Joint no-pulse t4.4 net TOTALQsite={joint["full_baseline"][-1]["Q_em_site_mean"]}; countPsite is NOT this netQ.',
        f'Joint zero/finite-source postmetric Wardmax={max([v for a in joint["Ward_zero_source_after_actual_G"].values()for v in a]+[v for pulse in joint["pulses"].values()for a in pulse["finite_source_Ward"].values()for v in a]):.6g}; actual bare-source Noether operator residuals0.',
        'Both use rich bare/proper2R.[S,O]/complexTrU/bare[H,O] clouds,ALL7Phi slices times independent7outputs.',
        'Actual allscalar+vector quark/white hops/fullcolour-square/Hphi/Hrot/plaquette before target projection.',
        'Quantum quark-hole/white/core/field changes allowed; rigid return-to-Omega is NOT a particle definition.',
        'White-only pulseedge2→quark receiveredge0; reverse quark-onlyedge0→whiteedge2; all edges/sites reported.',
        'Theta=.3,duration=.4 fixed BEFORE outcomes; independently evolved SAME-AGE no-pulse JOINT baseline.',
        'During0,.025,.05,.1,.2,.4; thereafter .25,.5,1,2,4 afterpulse. Source state continuous at0.',
        'Actual finite-theta J=-dH/dtheta and contact=d²H/dtheta² include exp(iqtheta),physical fractionalq.',
        'An immediate source-current operator change is NOT instantaneous transfer into the commuting receiver.'
    ]
    for source in ('white','quark'):
        measured=joint['connected'][source];off=control['connected'][source];lines.extend([
            f'{source} pulse: max|receiver current delta|={measured["max_abs_receiver_current"]:.12g}; pinned={measured["max_abs_pinned_receiver_current"]:.12g}; max|receiver Qdelta|={measured["max_abs_receiver_charge"]:.12g}.',
            f'{source} t0 crosscurrent={measured["instantaneous_cross_current"]}; t0 crossQ={measured["instantaneous_cross_charge"]}.',
            f'{source} first sampled delay t={measured["rows"][1]["t_absolute"]}: pinnedreceiver current delta={measured["rows"][1]["delta_pinned_receiver_current"]:.12g}; not instantaneous cross-transfer.',
            f'{source} max sharedfield deltas [Phi,Phi2,Pphi,Hphi,Hrot,loopB,loopC,electric]={measured["max_abs_shared_fields"]}.',
            f'{source} ALLscalar+vectorhopsOFF: sourceDeltaHnorm={control["pulses"][source]["source_deltaH_body_norm"]:.6g},maxcrossJ={off["max_abs_receiver_current"]:.6g},maxcrossQ={off["max_abs_receiver_charge"]:.6g}.'
        ])
    lines.extend([
        f'Joint ALLhopsOFF materialPVM-TVmax={max(control["actual_count_PVM_TV"]):.6g}; fields Hphi/Hrot/HE/plaquette remain ACTIVE.',
        'This reaches the controlled-H physical assertion; PhiOFFalone would leave sharedR/link dynamics.',
        'No source/receiver/coupling/constant tuning. Explicit source/body/anchor/neutralonehop/R/PW cuts SEVERE.',
        'Connected finite joint response can involve sharedPhi kinetic backreaction,R/Hrot/finiteRcommutators and indirect colourlink.',
        'Observed field changes do NOT uniquely isolate channels from Galerkin projection; NOT Maxwell/Coulomb/lightcone.',
        'Native EM is an EXTERNAL MATERIAL current probe,not dynamical photon or labrotational convection.',
        'Hot-reference fullH RMS5.905 belongs to reference; no new full-primitive particle residual/lifetime proof.',
        'First missing joined goal: stationary/range-stable rich particles with derivedfield-onlyspin/freeCOM and dynamicalEM.',
        'Existing47localframe/48photon laws remain distinct/open. No boundproton/QCD/positivephysicalmass/wholeissue closure.',
        f'Carrier NATURAL0,unchangedepoch,RSS={carrier_exit["child_peak_RSS_KiB"]}KiB; joint NATURAL0,unchangedepoch,RSS={joint_exit["child_peak_RSS_KiB"]}KiB.',
        'Raw arrays/states/operators/contacts/source strengths and exact epochs are separate; do not call raw strength probability.',
        'Native archive supplies frozen code/metadata/ops/states for fresh review WITHOUT producers or block reconstruction.',
        'Actual rooted LSP+AST exercised,diagnostic debt disclosed; no build/lint/tests/formatters. Reviewer owns raw audit.'
    ])
    if len(lines)>80:raise ValueError('Parent synthesis must stay <=80plain lines')
    synthesis_path.write_text('\n'.join(lines)+'\n');native_manifest={}
    with zipfile.ZipFile(native_path,'x',compression=zipfile.ZIP_DEFLATED,allowZip64=True)as native:
        for run,receipt in ((carrier_name,carrier_exit),(joint_name,joint_exit)):
            names=[name for name in receipt['source_input_before']if name.endswith('.py')];names.extend(name for name in receipt['files']if name.endswith(('_body_basis.pkl','_result.json','_input.json'))or name.endswith('.npz'));names=list(dict.fromkeys(names));hashes={**receipt['source_input_before'],**{name:item['SHA256']for name,item in receipt['files'].items()}}
            with zipfile.ZipFile(ROOT/(run+'_capsule.zip'))as source:
                for name in names:
                    destination=run+'/'+name;native_manifest[destination]={'bytes':source.getinfo(name).file_size,'SHA256':hashes[name]};info=zipfile.ZipInfo(destination);info.compress_type=zipfile.ZIP_DEFLATED
                    with source.open(name)as incoming,native.open(info,'w',force_zip64=True)as outgoing:shutil.copyfileobj(incoming,outgoing,1048576)
        for suffix in ('interface.json','language1.json','language_joint1.json'):
            path=ROOT/(PREFIX+suffix);native.write(path,path.name);native_manifest[path.name]={'bytes':path.stat().st_size,'SHA256':sha(path)}
    members=[synthesis_path,native_path,Path(__file__),Path(sys.argv[1]),ROOT/(PREFIX+'interface.json'),ROOT/(PREFIX+'language1.json'),ROOT/(PREFIX+'language_joint1.json'),ROOT/(PREFIX+'language_publish1.json')];scientific=[]
    for run,receipt in ((carrier_name,carrier_exit),(joint_name,joint_exit)):
        path=ROOT/(run+'_capsule.zip');members.append(path);scientific.append({'run':run,'natural_exit':receipt['natural_exit'],'returncode':receipt['returncode'],'capsule':path.name,'bytes':path.stat().st_size,'SHA256':sha(path),'peak_RSS_KiB':receipt['child_peak_RSS_KiB']})
    manifest={'kind':'COMMON_NATIVE_SPECIES_EXACT_DELIVERY','synthesis_lines':len(lines),'native_entries':native_manifest,'scientific_capsules':scientific,'members':{path.name:{'bytes':path.stat().st_size,'SHA256':sha(path)}for path in members},'scope':'No physics replay/cap/model change; exact captured native artifacts and all raw blocks/states in scientific capsules. Ancestor failures stay immutable in their owning historical capsules,not retroactively erased.'};manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');members.append(manifest_path)
    with zipfile.ZipFile(capsule_path,'x',allowZip64=True)as archive:
        for path in members:archive.write(path,path.name,compress_type=zipfile.ZIP_STORED if path.suffix=='.zip'else zipfile.ZIP_DEFLATED)
    receipt={'kind':'ACTUAL_COMMON_NATIVE_SPECIES_PUBLICATION_NOT_NEW_PHYSICS','capsule':str(capsule_path),'SHA256':sha(capsule_path),'bytes':capsule_path.stat().st_size,'native':str(native_path),'native_SHA256':sha(native_path),'synthesis':str(synthesis_path),'synthesis_lines':len(lines),'manifest_SHA256':sha(manifest_path),'elapsed_seconds':time.time()-started};receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
