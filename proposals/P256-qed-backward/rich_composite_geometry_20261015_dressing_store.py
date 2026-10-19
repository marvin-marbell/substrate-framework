"""Exact disk-backed rich BODY count blocks; cache is NOT a physical cut.
Fixes observed final CSC->CSR scratch allocation after all68 generators.
Flat owned-prefix files are automatically included in the actual capture.
"""
from pathlib import Path
from typing import Any
import pickle
from rich_composite_geometry_20261015_science import load


def update_pickle(path,value):
    """Atomic MUTABLE internal data update; final artifact save stays exclusive."""
    path=Path(path);temporary=Path(str(path)+'.tmp')
    with temporary.open('wb') as stream:pickle.dump(value,stream,protocol=5)
    temporary.replace(path)


class DiskBlocks:
    def __init__(self,prefix):
        path=Path(prefix);self.root=str(path.parent);self.prefix=path.name;self.meta={};self.paths={};self.cache_key=None;self.cache:Any=None;self.dirty=False
    def __contains__(self,label):return label in self.meta
    def __len__(self):return len(self.meta)
    def __iter__(self):return iter(self.meta)
    def path(self,label):return Path(self.root)/self.paths[label]
    def summary(self,block):
        return {'rank':block.rank,'nnz':block.matrix.nnz,'numeric_bytes':block.matrix.data.nbytes+block.matrix.indices.nbytes+block.matrix.indptr.nbytes,
            'raw_rows':len(block.keys),'columns':block.columns,'tiny_discarded_norm2':block.tiny_discarded_norm2,'nulls':block.nulls,
            'null_residual_norm2':block.null_residual_norm2,'minimum_accepted':block.minimum_accepted}
    def flush(self,unload=False):
        if self.cache is not None and self.dirty:
            update_pickle(self.path(self.cache_key),self.cache);self.meta[self.cache_key]=self.summary(self.cache);self.dirty=False
        if unload:self.cache_key=None;self.cache=None
    def create(self,label,block):
        if label in self.meta:raise ValueError('Existing exact body block')
        self.flush(True);self.paths[label]=f'{self.prefix}_bodyblock_{len(self.meta):05d}.pkl';self.meta[label]=self.summary(block);self.cache_key=label;self.cache=block;self.dirty=True
        return block
    def __getitem__(self,label):
        if self.cache_key!=label:
            self.flush(True);self.cache=load(self.path(label));self.cache_key=label
        self.dirty=True;return self.cache
    def load_readonly(self,label):return load(self.path(label))
    def relocate(self,directory):
        self.flush(True);self.root=str(Path(directory).resolve());return self
    def stats(self):
        self.flush()
        return {'dimension':sum(value['rank'] for value in self.meta.values()),'nnz':sum(value['nnz'] for value in self.meta.values()),
            'numeric_bytes':sum(value['numeric_bytes'] for value in self.meta.values()),'raw_rows':sum(value['raw_rows'] for value in self.meta.values()),
            'max_rank':max((value['rank'] for value in self.meta.values()),default=0),'tiny_discarded_norm2':sum(value['tiny_discarded_norm2'] for value in self.meta.values()),
            'null_residual_norm2':sum(value['null_residual_norm2'] for value in self.meta.values()),'nulls':sum(value['nulls'] for value in self.meta.values()),
            'minimum_accepted':min((value['minimum_accepted'] for value in self.meta.values()),default=1.)}
    def __getstate__(self):
        self.flush(True);return {**self.__dict__,'cache_key':None,'cache':None,'dirty':False}
