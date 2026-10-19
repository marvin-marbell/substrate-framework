"""Definition-level diagonal-P bare-electric discriminator on prior NEW own states."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import norm
from rich_field_packet_20261014_connected import ConnectedPacket
from rich_field_packet_20261014_model import PacketParameters, spatial_windows
from rich_frame_photon_20261014_model import Parameters
from rich_review_wave8_20261014_packet import evolve, observe

ROOT=Path(__file__).resolve().parent
PREFIX='rich_review_wave8_20261014_electric'


def main():
    data=json.loads((ROOT/(PREFIX+'_input.json')).read_text())
    physical=json.loads((ROOT/data['parameter_input']).read_text())
    p=PacketParameters(base=Parameters(**physical['base']),cutoff=physical['cutoff'],qwall=physical['qwall'],supplier_spacing=physical['supplier_spacing'])
    grams,first=spatial_windows(p)
    model=ConnectedPacket(p,1)
    state=np.load(ROOT/data['state_input'])
    pair=np.column_stack([state['initial'],state['reference_initial']])
    windows=[model.fop(model.electric_window(g)) for g in grams]
    commutators=[float(norm(model.E@w-w@model.E)) for w in windows]
    off=ConnectedPacket(p,1,coupling=False)
    records={}
    for label,h,observer in [('bare_electric_only',model.E,model),('matter_coupling_off_full_metric',off.H,off)]:
        before=observe(observer,pair,grams,first)
        after=observe(observer,evolve(h,pair,data['time']),grams,first)
        records[label]=dict(before=before,after=after,
                           packet_drift=(np.array(after['packet']['electric'])-before['packet']['electric']).tolist(),
                           reference_drift=(np.array(after['reference']['electric'])-before['reference']['electric']).tolist())
    result=dict(kind='NEW_DIAGONAL_FOURIER_MOMENTUM_BARE_ELECTRIC_CONTROL',joint_field_momentum_cutoff=p.cutoff,
                CAR_occupation=1,physical_field_momentum_indices=model.indices.tolist(),full_metric=model.M.tolist(),
                regional_E_commutators=commutators,records=records,
                interpretation='Joint field Fourier-momentum projection is not oscillator FockN. All P and Eregion commute exactly; no bare electric transport can be credited to projection. Same prepared reviewer pulse/reference bytes retained; matter-coupling-off retains full nonorthogonal metric.')
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(commutators=commutators,drifts={label:entry['packet_drift'] for label,entry in records.items()}),indent=2))


if __name__=='__main__':
    main()
