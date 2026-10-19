"""Read actual initial all-mode Pi on review49 OWN old PaP-pulse rays; no new evolution."""
from pathlib import Path
import hashlib
import json
import numpy as np
import rich_spatial_frame_photon_20261016_fields as fields

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def run():
    source=ROOT/(P+'spatial_paired_raw.npz');raw=np.load(source)
    recorded=json.loads((ROOT/(P+'spatial_paired_result.json')).read_text())
    basis=fields.basis(1);nf=len(basis);operators=[fields.primitive(basis,basis,a,'p').toarray() for a in range(8)]
    def momenta(state):
        norm=float(np.vdot(state,state).real);x=state.reshape(-1,nf)
        return np.array([float(np.vdot(x,x@op.T).real/norm) for op in operators])
    records={}
    for branch in ('addition','removal'):
        initial=momenta(raw[branch+'_initial']);basefull=momenta(raw[branch+'_uncicked_full']);basedec=momenta(raw[branch+'_uncicked_off']);rows=[]
        for incoming,outgoing in ((0,6),(2,4)):
            before=momenta(raw[f'{branch}_{incoming}_{outgoing}_kicked'])
            full=momenta(raw[f'{branch}_{incoming}_{outgoing}_full']);dec=momenta(raw[f'{branch}_{incoming}_{outgoing}_off'])
            offset=before-initial;fullresponse=full-basefull;decresponse=dec-basedec
            coupled=float(fullresponse[outgoing]-decresponse[outgoing])
            reference=next(v for v in recorded['records'][branch] if v['incoming']==incoming)
            assert abs(coupled-reference['calibrated_field_response'])<1e-12
            rows.append(dict(incoming=incoming,outgoing=outgoing,initial_reference_all_mode_Pi=initial.tolist(),initial_PaP_kicked_all_mode_Pi=before.tolist(),initial_all_mode_shift=offset.tolist(),initial_outgoing_preparation_shift=float(offset[outgoing]),full_final_crossprotocol_response=float(fullresponse[outgoing]),field_decoupled_final_crossprotocol_response=float(decresponse[outgoing]),full_temporal_response_difference=float(fullresponse[outgoing]-offset[outgoing]),field_decoupled_temporal_response_difference=float(decresponse[outgoing]-offset[outgoing]),calibrated_increment=coupled,calibration_identity_error=float(abs(coupled-((fullresponse[outgoing]-offset[outgoing])-(decresponse[outgoing]-offset[outgoing]))))))
        records[branch]=rows
    result=dict(kind='OWN_EXISTING_RAY_INITIAL_PULSE_VERSUS_TEMPORAL_ATTRIBUTION',raw_input_sha256=hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),records=records,limits='Only review49 own corrected-cubature old-protocol states are read. exp(i epsilon P*a*P) is a finite-space unitary, not an exact mode-pure full Weyl pulse. Large final raw response can be an initial crossmode Pi offset; report temporal increments separately. Calibrated full-minus-field-decoupled increment is unchanged because identical initial offsets cancel. This does not compare interacting old/new IP rays or establish photon/IR/continuum closure.')
    (ROOT/(P+'pulse_audit_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
