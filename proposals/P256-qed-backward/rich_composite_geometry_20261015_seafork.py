"""Independent correlated occupied-order candidate: one-hop + full flux images.
NOT completion/fallback statistics of the cancelled correlated degree2 family.
ALL288 physical source configurations/spins, actual quantum sea/Phi/colour remain.
"""
from itertools import product
import json,pickle,resource,sys,time
from pathlib import Path
import rich_composite_geometry_20261015_model as raw
import rich_composite_geometry_20261015_tensor as tensor
from rich_composite_geometry_20261015_sparse_basis import PacketBasis
from rich_composite_geometry_20261015_resume import checkpoint


def generators(primitive):
    o={(0,0,0,0,0):1.+0j};x=primitive.collective(o)
    vector=[primitive.collective(o,i+1,a) for i,a in product(range(3),repeat=2)]
    b=primitive.field(o,primitive.field_maps['B'])
    result=[o,x,*vector,b,primitive.field(b,primitive.field_maps['B']),primitive.electric(b)]
    for value in [x,*vector]:
        result.extend([primitive.field(value,primitive.field_maps['B']),primitive.electric(value)])
    result.append(primitive.collective(b));result.extend(primitive.collective(b,i+1,a) for i,a in product(range(3),repeat=2))
    return [value for packet in result if packet for value in (packet,primitive.conjugate(packet))]


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time();p=config['parameters'];phase=config['phase']
    if phase=='neutral':
        primitive=tensor.Primitives(p);o={(0,0,0,0,0):1.+0j};packets=generators(primitive);basis=PacketBasis(primitive)
        for packet in packets:basis.append(packet)
        white=raw.add(*[(primitive.graph['weights'][e],primitive.hop(o,e,f,True,0,d)) for e,f,d in product(range(4),range(2),(False,True))])
        root={'spin_norms':[float(raw.inner(v,v).real)**.5 for i in range(3) for v in [primitive.spin(o,i)]],
            'Gauss_norms':[float(raw.inner(v,v).real)**.5 for a in range(8) for v in [primitive.gauss(o,a)]],
            'scalar_hop_norm':float(raw.inner(primitive.collective(o),primitive.collective(o)).real)**.5,
            'WHITE_sea_hop_norm':float(raw.inner(white,white).real)**.5,
            'meaning':'Actual occupied correlated localN8/colour6/white2 spin/colour singlet; white is NOT added uud valence, so nonzero WHITE hop proves the retained sea is not inert.'}
        if max(root['spin_norms']+root['Gauss_norms'])>1e-9 or root['WHITE_sea_hop_norm']<1e-9:raise ArithmeticError(f'Actual chosen rich sea failure {root}')
        data={'p':p,'primitive':primitive,'basis':basis,'generators':packets}
        with open(out+'_neutral.pkl','xb') as f:pickle.dump(data,f,protocol=5)
        result={'root':root,'neutral':basis.receipt(),'generator_count':len(packets)}
    elif phase=='charged':
        with open(config['neutral_snapshot'],'rb') as f:neutral=pickle.load(f)
        if p!=neutral['p']:raise ValueError('Different candidate law')
        primitive=neutral['primitive'];basis=PacketBasis(primitive);specs=list(raw.source_specs())
        for i,spec in enumerate(specs):
            for packet in neutral['generators']:basis.append(primitive.source(packet,*spec))
            print(json.dumps({'stage':'actual_correlated_onehop_ALL_sources','source':i,'source_count':len(specs),'dimension':len(basis.vectors)}),flush=True)
            if (i+1)%8==0:checkpoint(out+'_continuation_checkpoint.pkl',{'p':p,'primitive':primitive,'basis':basis,'next_source':i+1,'neutral_snapshot':config['neutral_snapshot']})
        with open(out+'_charged.pkl','xb') as f:pickle.dump({'p':p,'primitive':primitive,'basis':basis,'sources':specs},f,protocol=5)
        result={'charged':basis.receipt(),'source_count':len(specs)}
    else:raise ValueError('Unknown material candidate phase')
    result.update(kind='ACTUAL_INDEPENDENT_CORRELATED_OCCUPIED_ONEHOP_FLUX_CANDIDATE',phase=phase,parameters=p,
        limits='Explicit SEVERE finite sea-pair packet cut:one constituent transfer plus all chosen loop/electric images, actual full canonical localFock sea, allPhi7/PW19/SO3j0+j1 and ALL288sourcepositions/spins. Not cancelledK2 success or nestedvariational/spin improvement. No inert particle-only trimer, source/state/coupling fit or inherited energies.')
    result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
