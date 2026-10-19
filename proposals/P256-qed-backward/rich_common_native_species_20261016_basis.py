"""Exact blocked MGS of chosen species-source Phi slices, full independentPhi7.
Source/body/sea/R/PW/anchor Galerkin cuts are explicit, not full primitive closure.
"""
import gc,json
from pathlib import Path
import numpy as np
from rich_composite_geometry_20261015_cloud_probe import inner
from rich_composite_geometry_20261015_dressing_basis import OperatorBlock,source_coordinates
from rich_composite_geometry_20261015_dressing_store import DiskBlocks,update_pickle
from rich_common_native_species_20261016_sources import descriptors,apply_source,algebra_receipt


def construct(primitive,omega,config,scalar,rotor,out):
    storage=DiskBlocks(out);probes=[];generators=list(descriptors(config));nf=primitive.p['scalar_points'];algebra={}
    for index,descriptor in enumerate(generators):
        state=apply_source(primitive,omega,descriptor,scalar,rotor);raw=float(inner(state,state).real);pieces={};parts={}
        if descriptor['family']=='bare':algebra[str(descriptor)]=algebra_receipt(descriptor)
        for key,field in state.items():parts.setdefault(primitive.label(key)[:-1],{})[key]=field
        for counts,part in parts.items():
            label=(descriptor['twice_J'],counts);block=None
            for phi in range(nf):
                packet={}
                for key,field in part.items():
                    for r in np.flatnonzero(field[:,phi]):packet[key+(int(r),)]=complex(field[r,phi])
                if not packet:continue
                if block is None:block=storage[label] if label in storage else storage.create(label,OperatorBlock())
                pieces.setdefault(label,{})[phi]=block.append_operator(packet,(index,phi))
        probes.append({'descriptor':descriptor,'raw_strength':raw,'pieces':pieces});storage.flush(True);del state,parts;gc.collect();statistics=storage.stats()
        print(json.dumps({'stage':'COMMON_NATIVE_SPECIES_FULL_PHI_SOURCE','generator':index,'descriptor':descriptor,'raw_strength':raw,'body':statistics['dimension'],'nnz':statistics['nnz'],'count_blocks':len(storage)}),flush=True)
        if descriptor['family']=='eom':update_pickle(out+'_construction_progress.pkl',{'storage':storage,'next_generator':index+1,'probes':probes,'generators':generators,'config':config})
    offset=0;labels=[]
    for label in storage:
        block=storage[label];block.columns=list(range(offset,offset+block.rank));offset+=block.rank;block.matrix=block.matrix.tocsr();labels.extend({'twice_J':label[0],'counts':label[1],'N':sum(label[1])} for _ in range(block.rank));storage.flush(True)
    statistics=storage.stats();receipt={**statistics,'body_dimension':offset,'full_independent_Phi':nf,'full_coordinate_dimension':offset*nf,'count_blocks':len(storage),'source_generators':len(generators),'source_Noether':algebra,
        'space':'All7Phi input slices of literal B/L/hole/proper2R.[S,O]/complexTrU/[H,O] acting on SAME occupiedN32Omega; tensor ALL7 independentPhi outputs. QuantumR10/PW19 embedded in body. Entire available quark/white/sea source profiles retained, no mandatory rigid field/core return or compact mask.',
        'cuts':'NEW source/sector-specific finite Galerkin family,not physically identical to ancestor nativeN35 by fiat. Declared source anchors/bodyrank,neutralonehop order,Rj0+j1/PW19/Phi7 cuts severe; no full all-pair/globalvacuum/range closure. Null thresholds1e-26input/1e-20relative/1e-25absolute and2e-14coefficient trim inherited exactly; all discarded norms in statistics.',
        'lift':'OperatorBlock.lifts preserve actual MGS source/Phi-input coefficients; offOmega extension includes |phiout><phiin|P_counts. No sourceEOM from a state-only extension.'}
    Path(out+'_basis_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return storage,labels,source_coordinates(storage,probes,len(labels),nf),generators,receipt
