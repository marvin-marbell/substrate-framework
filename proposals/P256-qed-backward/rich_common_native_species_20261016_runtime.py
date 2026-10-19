"""Common occupied CAR64/Phi7/R10/PW19 execution, no ancestor mutation."""
from pathlib import Path
import gc,json,resource,time
import numpy as np
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_compile import save
from rich_composite_geometry_20261015_cloud_probe import neutral_state,inner
from rich_composite_geometry_20261015_dressing_operators import assemble
from rich_common_native_species_20261016_basis import construct


def execute(config,out,analyze):
    started=time.time()
    if 'body_basis_snapshot' in config:
        metadata=load(config['body_basis_snapshot']);p=metadata['p'];primitive=metadata['primitive'];storage=metadata['storage'];labels=metadata['labels'];probes=metadata['probes'];generators=metadata['generators'];basis=metadata['basis_receipt'];reference=metadata['reference_energy'];omega_norm=metadata['omega_norm'];storage.relocate(Path(config['body_basis_snapshot']).resolve().parent)
        captured={Path(value).name for key,value in config.items() if key.endswith('_snapshot')}
        if not set(storage.paths.values())<=captured:raise ValueError('Every flat block must be explicitly pinned for exact resume')
        del metadata
    else:
        data=load(config['neutral_basis_snapshot']);primitive=primitive_for(data);p=data['p'];coupling=load(config['neutral_coupling_snapshot']);eigen=np.load(config['neutral_states_snapshot']);omega=neutral_state(data,coupling,eigen['states'][:,0]);reference=float(eigen['energies'][0]);omega_norm=float(inner(omega,omega).real);del data,coupling,eigen;gc.collect();scalar,rotor=model.field_and_rotor(p)
        storage,labels,probes,generators,basis=construct(primitive,omega,config,scalar,rotor,out);del omega;gc.collect()
        save(out+'_body_basis.pkl',{'p':p,'primitive':primitive,'storage':storage,'labels':labels,'probes':probes,'generators':generators,'basis_receipt':basis,'reference_energy':reference,'omega_norm':omega_norm})
    scalar,rotor=model.field_and_rotor(p);operators,assembly=assemble(primitive,storage,labels,rotor,out);local=primitive.local.receipt();del primitive,storage;gc.collect()
    result={'kind':'SAME_PRIMITIVE_OCCUPIED_COMMON_NATIVE_SPECIES_CONDITIONAL_MODEL','mode':config['mode'],'parameters':p,'configuration':config,'basis':basis,'assembly':assembly,'local_receipt':local,'reference_energy':reference,'omega_norm':omega_norm,
        'law':'SAME geometry primitiveH,p and actualN32Omega as completed geometry:dressing3,not its solvedchargedstate. One shared quantumPhi7/R10/PW19 and originalCAR64; original allquark/white scalarANDvector hops,fullcolour-square/rotation/plaquette beforetargetprojection. N33/N31/N36 are source-sector-specific Galerkin approximations,not identicaloldfiniteH by fiat. No independently evolved particles tensored.',
        'limits':'Primitive CAR spin/charge/flavour/3+1/gauging,negative scalar quadratic/volume and pointmass density INPUTS. Severe selectedonehop neutral order known fullH RMS5.90502139924; no stationaryground/mass/gap/lifetime theorem. R/PW/body/source cuts severe. MaterialPeierls external probe NOT photon/Coulomb/Maxwell/labrotationalEM; other47/48laws remain separate. CountPVM not tagged-three-particle position. No oldbody408/CURRENT427/N17 or otherlawstate splice.'}
    Path(out+'_model_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n');result.update(analyze(operators,labels,probes,scalar,p,reference,config,out));result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started};Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)
