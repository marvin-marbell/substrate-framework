"""Actual streaming neutral/source geometry constructor, not a test runner."""
from __future__ import annotations
import json,pickle,resource,sys,time
from pathlib import Path
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_compact import PacketBasis


def save(path,value):
    with open(path,'xb') as f:pickle.dump(value,f,protocol=5)


def main():
    config=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);phase=sys.argv[3]
    started=time.time();p=config['parameters'];primitive=model.Primitives(p)
    if phase=='neutral':
        packets=model.neutral_generators(primitive)
        basis=model.PacketBasis()
        for i,packet in enumerate(packets):
            basis.append(packet)
            if i%10==0:print(json.dumps({'stage':'neutral','generator':i,'dimension':len(basis.vectors)}),flush=True)
        save(str(output)+'_neutral.pkl',{'p':p,'generators':packets,'basis':basis})
        result={'kind':'ACTUAL_NEW_CAR64_OCCUPIED_NEUTRAL_NATIVE_CONSTRUCTOR','parameters':p,'neutral':basis.receipt(),'generator_count':len(packets)}
    elif phase=='charged':
        with open(config['neutral_snapshot'],'rb') as f:data=pickle.load(f)
        if data['p']!=p:raise ValueError('New law geometry/input mismatch')
        basis=PacketBasis();specs=list(model.source_specs())
        for j,spec in enumerate(specs):
            for packet in data['generators']:basis.append(model.source(packet,*spec))
            print(json.dumps({'stage':'charged','source_index':j,'source_count':len(specs),'dimension':len(basis.vectors),'max_block':max(map(len,basis.blocks.values()),default=0)}),flush=True)
            if (j+1)%24==0:
                save(str(output)+f'_source{j+1}_checkpoint.pkl',{'p':p,'basis':basis,'next_source':j+1,'neutral_snapshot':config['neutral_snapshot']})
        save(str(output)+'_charged.pkl',{'p':p,'basis':basis,'sources':specs,'neutral_snapshot':config['neutral_snapshot']})
        result={'kind':'ACTUAL_NEW_CAR64_UUD_ALL_INTERNAL_CONFIGURATIONS_CONSTRUCTOR','parameters':p,'charged':basis.receipt(),'source_count':len(specs)}
    else:raise ValueError('Unknown constructor phase')
    result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    Path(str(output)+'_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'completed':phase,**result['resource']}),flush=True)


if __name__=='__main__':main()
