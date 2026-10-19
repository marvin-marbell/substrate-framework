"""NEW exact bare-electric discriminator on OWN previously prepared full state.

This is an additional physical consumer, not a replay of a producer. It binds
its exact same-H initial states to the own capsule output and compares full H
against bare H_E without changing reference, metric or inclusive detector.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse.linalg import norm
from rich_frame_photon_20261014_model import Parameters
from rich_field_packet_20261014_model import PacketParameters, spatial_windows
from rich_field_packet_20261014_connected import ConnectedPacket
from rich_field_packet_20261014_calculate import attach_windows, evolve, snapshot
from rich_physical_legs_20261014_calculate import physical_receiver


def run(inp,out):
    data=json.loads(inp.read_text());root=inp.parent
    source=root/data['state_npz'];receipt=json.loads((root/data['state_receipt']).read_text())
    actual=hashlib.sha256(source.read_bytes()).hexdigest()
    if receipt['outputs'].get(source.name)!=actual or not receipt['source_unchanged'] or receipt['returncode']:
        raise ValueError('Own prepared-state natural-exit/hash binding failed')
    for key, bindings in [('original_input', receipt['source_before']), ('original_result', receipt['outputs'])]:
        bound = root / data[key]
        if bindings.get(bound.name) != hashlib.sha256(bound.read_bytes()).hexdigest():
            raise ValueError('Own input/result hash binding failed: ' + bound.name)
    original=json.loads((root/data['original_input']).read_text())
    original_result=json.loads((root/data['original_result']).read_text())
    p=PacketParameters(base=Parameters(**original['base']),cutoff=original['cutoff'],qwall=original['qwall'])
    m=ConnectedPacket(p,1);grams,first=spatial_windows(p);attach_windows(m,grams,first)
    # Shared law is required, not assumed from matching class names.
    for name,expected in receipt['source_before'].items():
        if name in ['rich_field_packet_20261014_connected.py','rich_field_packet_20261014_model.py','rich_common_mobile_20261014_model.py','rich_frame_photon_20261014_model.py','rich_frame_gauss_20261014_model.py']:
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Original physical law changed: '+name)
    states=np.load(source);initial=np.column_stack([states['physical_initial'],states['physical_reference_initial']])
    threshold=original_result['calibrated_physical_outgoing_threshold'];time=data['time']
    full=evolve(m.H,initial,time);bare=evolve(m.E,initial,time)
    records={}
    for label,pair in [('initial',initial),('full',full),('bare_electric_same_prepared',bare)]:
        ps, rs = snapshot(m,pair[:,0]), snapshot(m,pair[:,1])
        records[label]=dict(packet=ps,reference=rs,
                            receiver=physical_receiver(m,pair[:,0],threshold),reference_receiver=physical_receiver(m,pair[:,1],threshold),
                            excess_electric=(np.array(ps['electric_regions'])-rs['electric_regions']).tolist())
    region_commutators=[float(norm(m.E@m.fop(w)-m.fop(w)@m.E)) for w in m.windows]
    probabilities=lambda label:np.array(records[label]['receiver']['probabilities'])
    result=dict(kind='SAME_PREPARED_REFERENCE_EXACT_ELECTRIC_CONTRAST',state_npz_hash=actual,time=time,
                all_bare_electric_region_commutators=region_commutators,records=records,
                bare_electric_probability_change=float(np.linalg.norm(probabilities('bare_electric_same_prepared')-probabilities('initial'))),
                full_probability_change=float(np.linalg.norm(probabilities('full')-probabilities('initial'))),
                bare_excess_region_change=(np.array(records['bare_electric_same_prepared']['excess_electric'])-records['initial']['excess_electric']).tolist(),
                full_excess_region_change=(np.array(records['full']['excess_electric'])-records['initial']['excess_electric']).tolist(),
                scope='Exact H_E comparator on same normalized prepared pulse/reference, same field metric and physical outgoing threshold/receiver. No matter input or neutral-vacuum splice. No autonomous freeelectric transport; full changes require actual mattercoupling. Full snapshot energy/current are delivered-H observables even during explicitly bare-E evolution.')
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),initial=initial,full=full,bare=bare)
    print(json.dumps({key:result[key] for key in ['all_bare_electric_region_commutators','bare_electric_probability_change','full_probability_change','bare_excess_region_change','full_excess_region_change']}))


if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
