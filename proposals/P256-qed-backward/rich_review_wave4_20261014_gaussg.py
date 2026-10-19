"""Fresh occupied-reference full N0/N2 anchored CAR/tree-electric receiver."""
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Any, cast
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import splu
from rich_spin_flux_spatial_20261014_model import Parameters, Sector, charged_source, LAW
from rich_review_wave4_20261014_tools import complex_pair, save


def main() -> None:
    inputs = json.loads(Path(sys.argv[1]).read_text())
    p = Parameters(**inputs['parameters'])
    reference = Sector(p,up=1,down=0,supplier=1)
    energy,ground,_ = reference.ground()
    removal = Sector(p,up=0,down=0,supplier=1)
    same = Sector(p,up=2,down=0,supplier=1)
    opposite = Sector(p,up=1,down=1,supplier=1)
    sources = []
    for target,spin,create in [(removal,0,False),(same,0,True),(opposite,1,True)]:
        sources.append(np.column_stack([charged_source(reference,target,site,spin,create)@ground for site in range(reference.vertices)]))
    z = complex(*inputs['z'])
    greens = []
    spectral_errors = []
    low4_errors = []
    grams = []
    for index,(target,source) in enumerate(zip([removal,same,opposite],sources)):
        sign = 1 if index==0 else -1
        shifted = target.H-energy*sparse.eye(target.dimension,format='csc')
        direct = source.conj().T@splu((z*sparse.eye(target.dimension,format='csc')+sign*shifted).tocsc()).solve(source)
        poles,vectors = eigh(target.H.toarray())
        amplitudes = vectors.conj().T@source
        spectral = amplitudes.conj().T@((1/(z+sign*(poles-energy)))[:,None]*amplitudes)
        low = amplitudes[:4].conj().T@((1/(z+sign*(poles[:4]-energy)))[:,None]*amplitudes[:4])
        spectral_errors.append(float(np.linalg.norm(direct-spectral)))
        low4_errors.append(float(np.linalg.norm(direct-low)))
        greens.append(direct)
        grams.append(source.conj().T@source)
    same_green = greens[1]+greens[0].T
    closure = grams[1]+grams[0].T
    save('gaussg',{'kind':'INDEPENDENT_NEW_OCCUPIED_COMMON_GAUSS_FULL_SECTOR_RECEIVER','inputs':inputs,'law':LAW,
                    'occupied_reference':{'up':1,'down':0,'supplier':1,'countercharge':-2,'dimension':reference.dimension,'energy':energy,
                                           'local_flux':[float(np.vdot(ground,f@ground).real) for f in reference.F]},
                    'sector_dimensions':[target.dimension for target in [removal,same,opposite]],'countercharges':[-1,-3,-3],
                    'full_spectral_direct_errors':spectral_errors,'low4_G_channel_errors':low4_errors,
                    'same_spin_G':[[complex_pair(x) for x in row] for row in same_green],
                    'opposite_spin_G_add':[[complex_pair(x) for x in row] for row in greens[2]],
                    'actual_projected_CAR_closure_diagonal':cast(Any,np.diag(closure)).real.tolist(),'closure_eigenvalues':np.linalg.eigvalsh(closure).tolist(),
                    'Gauss_sector_errors':[target.constraints()['Gauss_max'] for target in [reference,removal,same,opposite]],
                    'scope':'Same microscopic law, bounded field domain and occupied common active/supplier/orientor reference. Physical tree-Wilson/CAR operation simultaneously changes explicit static countercharge at anchor0. N0 removal transpose uses z+H0-E1, additions use z-H2+E1. Missing closure is retained field-domain loss, not silently rescaled CAR. No holes-as-positrons claim.'})

if __name__=='__main__':
    main()
