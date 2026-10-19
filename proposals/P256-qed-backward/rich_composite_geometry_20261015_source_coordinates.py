"""Numerically evaluate C^-1 (DIRECT annihilator source) C in local Fock space.
This is an exact unitary anti-coordinate change, not a particle/state fallback.
It avoids expanding the tensor product C on every global raw coordinate.
"""
from functools import lru_cache
from collections import defaultdict
from rich_composite_geometry_20261015_model import apply,trim


@lru_cache(maxsize=262144)
def local_C(mask,site,inverse=False):
    def forward(original):
        target=(1<<16)-1;amplitude=1
        for index in reversed([i for i in range(16) if original&(1<<i)]):
            f=index//8;s=(index%8)//4;c=index%4
            target,sign=apply(target,8*f+4*(1-s)+c,False);amplitude*=sign*((-1)**(site+s))
        return target,amplitude
    target,value=forward(mask)
    if not inverse:return target,value
    _,value=forward(target)
    return target,complex(value).conjugate()


def dressed_local(mask,site,index,create):
    first,a=local_C(mask,site);step=apply(first,index,create)
    if step is None:return None
    target,b=local_C(step[0],site,True)
    return target,a*step[1]*b


def operations(primitive,key,terms):
    current={key:1.+0j}
    for global_index,create in reversed(terms):
        site,index=divmod(global_index,16);out=defaultdict(complex)
        for state,value in current.items():
            if hasattr(primitive,'local'):
                n=sum(sum(primitive.local.labels[state[v]][:4]) for v in range(site))
                cache_key=('Cinv_op_C',site,index,create,state[site])
                if cache_key not in primitive.local.cache:
                    local=defaultdict(complex)
                    for mask,a in primitive.local.vectors[state[site]].items():
                        step=dressed_local(mask,site,index,create)
                        if step is not None:local[step[0]]+=a*step[1]
                    primitive.local.cache[cache_key]=primitive.local.resolve(trim(local))
                for column,a in primitive.local.cache[cache_key].items():
                    target=list(state);target[site]=column;out[tuple(target)]+=(-1)**n*value*a
            else:
                mask,pw=state;local=(mask>>(16*site))&65535;n=(mask&((1<<(16*site))-1)).bit_count()
                step=dressed_local(local,site,index,create)
                if step is not None:
                    target=(mask&~(65535<<(16*site)))|(step[0]<<(16*site));out[target,pw]+=(-1)**n*value*step[1]
        current=trim(out)
    return current
