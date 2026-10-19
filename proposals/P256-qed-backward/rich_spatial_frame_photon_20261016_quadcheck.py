"""Selected actual finite-q contact/Peierls blocks: old24/32 cubature distinction."""
import json
from pathlib import Path
import numpy as np
import rich_gauge_band_frame_20261016_model as band
import rich_spatial_frame_photon_20261016_quadrature as quad
import rich_spatial_frame_photon_20261016_kernel as kernel
import rich_spatial_frame_photon_20261016_fields as fields
from rich_spatial_frame_photon_20261016_model import rotations,mult,projection
ROOT=Path(__file__).resolve().parent
P='rich_spatial_frame_photon_20261016_'


def blocks(R,full_peierls=False):
    positions=np.einsum('tab,sb->tsa',rotations(R['points']),band.geometry()['sites']-.5)
    geom=[kernel.geometry(x,kernel.default_modes()) for x in positions]
    conn=np.array([g['connection'] for g in geom]);Q=-band.Sector(3).q[[0,17,55]]
    cq=np.einsum('fs,tsla->ftla',Q,conn,optimize=True)
    contacts=np.array([mult(R,np.sum(cq[f,:,a,:]*cq[f,:,b,:],axis=-1)) for f in range(3) for a,b in [(0,0),(0,6),(1,7),(2,4)]])
    phi=kernel.segment(positions[:,0],positions[:,4],kernel.default_modes())[0]
    fb=fields.basis(1);D=fields.displacement(fb,fb,-phi)
    # Full primary R1x9-field block; R2 selected actual R entries (not a new field cut).
    if full_peierls:peierls=mult(R,D)
    else:peierls=np.einsum('ti,t,tab,ti->iab',R['F'][:,[0,7,16,34]].conj(),R['weights'],D,R['F'][:,[1,8,20,33]],optimize=True)
    return contacts,peierls


def main():
    result={};raw={}
    for level in (1,2):
        old=band.angular(tuple(range(level+1)));a=quad.material(level,24,16);b=quad.material(level,32,20)
        co,po=blocks(old,level==1);ca,pa=blocks(a,level==1);cb,pb=blocks(b,level==1)
        record=dict(gram24=a['gram_error'],gram32=b['gram_error'],contact_old_vs24=float(np.linalg.norm(co-ca)),contact24vs32=float(np.linalg.norm(ca-cb)),Peierls_old_vs24=float(np.linalg.norm(po-pa)),Peierls24vs32=float(np.linalg.norm(pa-pb)),selection='three actual N3 Q configurations, four mechanical contact pairs; full primary9-field Hermite matrix, allR1 pairs/fourR2 pairs')
        raw.update({f'R{level}_contact_old':co,f'R{level}_contact24':ca,f'R{level}_contact32':cb,f'R{level}_Peierls_old':po,f'R{level}_Peierls24':pa,f'R{level}_Peierls32':pb})
        cg,pg=blocks(projection.material(level),level==1)
        record['contact_Gaunt_vs_direct24']=float(np.linalg.norm(cg-ca))
        record['Peierls_Gaunt_vs_direct24']=float(np.linalg.norm(pg-pa))
        result[str(level)]=record;print(level,json.dumps(record),flush=True)
    np.savez_compressed(ROOT/(P+'quadcheck_raw.npz'),**raw)
    (ROOT/(P+'quadcheck_result.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
