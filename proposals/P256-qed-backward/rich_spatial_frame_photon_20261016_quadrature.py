"""Retained integer-R Wigner functions on independently oversampled SO3 nodes.
Band40's angular grid integrates its linear O_R law, NOT finite-q products.
No high-j state is inserted: only integration nodes change; LEFT labels stay.
"""
from functools import lru_cache
from itertools import product
import numpy as np
from scipy.linalg import expm
import rich_gauge_band_frame_20261016_model as band


@lru_cache(maxsize=8)
def material(level,azimuth=24,polar=16):
    old=band.angular(tuple(range(level+1)))
    z,w=np.polynomial.legendre.leggauss(polar)
    angles=np.arange(azimuth)*2*np.pi/azimuth
    points=np.array(list(product(angles,np.arccos(z),angles)))
    weights=np.tile(np.repeat(w/(2*azimuth**2),azimuth),azimuth)
    blocks=[]
    for j in range(level+1):
        ms=np.arange(-j,j+1);small=np.array([expm(-1j*b*band.jmat(j)[1]) for b in np.arccos(z)])
        left=np.exp(-1j*angles[:,None]*ms);right=left
        D=np.einsum('am,bmn,cn->abcmn',left,small,right,optimize=True)
        blocks.append(np.sqrt(2*j+1)*D.reshape(len(points),(2*j+1)**2))
    F=np.concatenate(blocks,axis=1)
    return dict(old,F=F,points=points,weights=weights,quadrature=dict(azimuth=azimuth,polar=polar),gram_error=float(np.linalg.norm(F.conj().T@(weights[:,None]*F)-np.eye(F.shape[1]))))
