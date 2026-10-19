"""Import-stable trusted-local disk-backed operator mapping, one cached CSR."""
from collections.abc import Mapping
from pathlib import Path
from scipy import sparse


class OperatorStore(Mapping):
    def __init__(self,prefix,names):self.prefix=str(prefix);self.names=tuple(names);self.cached=None
    def __len__(self):return len(self.names)
    def __iter__(self):return iter(self.names)
    def __getitem__(self,name):
        if name not in self.names:raise KeyError(name)
        if self.cached is None or self.cached[0]!=name:self.cached=(name,sparse.load_npz(self.prefix+'_op_'+name+'.npz'))
        return self.cached[1]
    def __getstate__(self):return {'prefix':self.prefix,'names':self.names}
    def __setstate__(self,state):self.prefix=state['prefix'];self.names=state['names'];self.cached=None
    def relocate(self,directory):self.prefix=str(Path(directory)/Path(self.prefix).name);self.cached=None
