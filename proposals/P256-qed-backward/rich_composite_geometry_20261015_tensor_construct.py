"""Actual material alternative constructor, SAME microscopic H, noninert sea."""
import json,pickle,resource,sys,time
from pathlib import Path
from rich_composite_geometry_20261015_model import source_specs,inner
import rich_composite_geometry_20261015_tensor as tensor
from rich_composite_geometry_20261015_compact import PacketBasis


def save(path,data):
    with open(path,'xb') as f:pickle.dump(data,f,protocol=5)


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time();phase=config['phase'];p=config['parameters']
    if phase=='neutral':
        primitive=tensor.Primitives(p);o={(0,0,0,0,0):1.+0j}
        root={'spin_norms':[float(inner(v,v).real)**.5 for i in range(3) for v in [primitive.spin(o,i)]],
            'Gauss_norms':[float(inner(v,v).real)**.5 for a in range(8) for v in [primitive.gauss(o,a)]],
            'scalar_constituent_hop_image_norm':float(inner(primitive.collective(o),primitive.collective(o)).real)**.5,
            'meaning':'Correlated localN8/Ncolour6/Nwhite2 spin/colour singlet sea. Nonzero primitive hop norm is necessary: NOT inert spectators/particle-only trimer.'}
        if max(root['spin_norms']+root['Gauss_norms'])>1e-9:raise ArithmeticError(f'Actual local seed symmetry failure {root}')
        if root['scalar_constituent_hop_image_norm']<1e-9:raise ArithmeticError('Actual reference sea is inert')
        packets=tensor.neutral_generators(primitive);basis=tensor.PacketBasis(primitive)
        for i,packet in enumerate(packets):
            basis.append(packet)
            if i%10==0:print(json.dumps({'stage':'correlated_sea_neutral','generator':i,'dimension':len(basis.vectors),'local_dimension':len(primitive.local.vectors)}),flush=True)
        save(out+'_neutral.pkl',{'p':p,'primitive':primitive,'basis':basis,'generators':packets})
        result={'kind':'ACTUAL_CORRELATED_OCCUPIED_LOCAL_TENSOR_SEA_CONSTRUCTOR','phase':phase,'root':root,'neutral':basis.receipt(),'generator_count':len(packets)}
    elif phase=='charged':
        with open(config['neutral_snapshot'],'rb') as f:data=pickle.load(f)
        if data['p']!=p:raise ValueError('New candidate law mismatch')
        primitive=data['primitive'];basis=PacketBasis(primitive);specs=list(source_specs())
        for i,spec in enumerate(specs):
            for packet in data['generators']:basis.append(primitive.source(packet,*spec))
            print(json.dumps({'stage':'correlated_sea_ALL_uud_positions','source':i,'source_count':len(specs),'dimension':len(basis.vectors),'local_dimension':len(primitive.local.vectors)}),flush=True)
            if (i+1)%24==0:
                save(out+f'_source{i+1}_checkpoint.pkl',{'p':p,'primitive':primitive,'basis':basis,'next_source':i+1,'neutral_snapshot':config['neutral_snapshot']})
        save(out+'_charged.pkl',{'p':p,'primitive':primitive,'basis':basis,'sources':specs,'neutral_snapshot':config['neutral_snapshot']})
        result={'kind':'ACTUAL_CORRELATED_OCCUPIED_LOCAL_TENSOR_UUD_CONSTRUCTOR','phase':phase,'charged':basis.receipt(),'source_count':len(specs)}
    else:raise ValueError('Unknown actual constructor phase')
    result['parameters']=p;result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
