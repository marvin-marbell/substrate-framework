"""Selected full-H/connection/HI images on OWN new coupled charged endpoint."""
from pathlib import Path
import json
import resource
import time
import gc
import numpy as np
import rich_exact_electric_picture_20261016_model as picture
from rich_exact_electric_picture_20261016_images import image
ROOT=Path(__file__).resolve().parent
P='rich_exact_electric_review_20261016_'


def main():
    started=time.time();case=json.loads((ROOT/(P+'result.json')).read_text());model=picture.Model(5,case['parameters']);raw=np.load(ROOT/(P+'addition_case_raw.npz'));state=raw['baseline'];T=case['T'];result={}
    x=state.reshape(model.shape);rotation=np.einsum('...ja,ij->...ia',x,model.R['casimir'].toarray(),optimize=True)/(2*model.parameters['material_inertia'])
    result['mechanical_control_scope']=dict(hopping_disabled_Lambda_still_coupled_source_norm=float(np.linalg.norm(model.vertex(state,0,T,hopping=False,coupled=True))),fully_field_decoupled_source_norm=float(np.linalg.norm(model.vertex(state,0,T,hopping=False,coupled=False))),retained_free_R_kinetic_action_norm=float(np.linalg.norm(rotation)),meaning='Field-decoupled source derivative zero is not a claim that rotating charged microsites have no lab convection. HoppingOFF alone leaves the actual mechanical Lambda coupling.')
    for angular,name in ((False,'field_shell'),(True,'angular_shell')):
        receipt,hi,connection=image(model,state,T,angular)
        if max(receipt['HI_compression_error'],receipt['H0_connection_compression_error'],receipt['physical_H_compression_error'])>2e-9:raise AssertionError('Own-state exterior image does not compress to actual selected coupled H')
        result[name]=receipt
        np.savez_compressed(ROOT/(P+name+'_images_raw.npz'),H_I_image=hi,H0_basis_connection_image=connection)
        (ROOT/(P+'images_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(stage='OWN_COUPLED_STATE_EXTERIOR_IMAGE',case=name,receipt=receipt)),flush=True);del hi,connection;gc.collect()
    result['resource']=dict(elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result['scope']='Pure frozen rectangular-image operator helper reused on OWN new unforced N5 endpoint, not author states/images. Field and angular target shells are separate partial lower bounds; no full infinite outgoing equality. H0 exterior is exact moving-basis connection, not itself the HI dynamical residual or a cutoff error. No source/range/photon convergence claim.'
    (ROOT/(P+'images_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
