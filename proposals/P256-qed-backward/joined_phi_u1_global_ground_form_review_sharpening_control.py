"""Parent-proposed HS/KyFan sharpening: independent audit, no producer adoption."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import zipfile
from typing import Any
import numpy as np

ROOT=Path(__file__).resolve().parent
INPUT='joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json'
PIN='1c79b1a74c4236079645ebf9765a48c1c2263ec7b7209e9f103704c98752da94'


def run(output: Path) -> dict[str, Any]:
    source=Path(__file__).resolve(); own=source.read_bytes(); raw=(ROOT/INPUT).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==PIN
    archive_path=output.with_name(output.stem+'_preexecution.zip')
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('executed_source/'+source.name,own);archive.writestr('input/'+INPUT,raw)
    profiles=json.loads(raw)['primitive_profiles']; coefficients=[]
    for p in profiles:
        c0=F(p['onsite'])**2
        spatial=[F(0),F(0),F(0)]
        for row in p['paths']:
            length=int(row['length']);axis=int(row['axis'])
            assert length in (1,2)
            factor=2 if length==1 else 4
            c0+=factor*F(row['c0'])**2
            spatial[axis]+=factor*F(row['ci'])**2
        coefficients.append([c0,*spatial])
    assert max(coefficients[0])<26**2
    assert all(int(r['length'])==1 for r in profiles[0]['paths'])
    assert all(max(c)<64**2 for c in coefficients[1:])
    lam,chi,fast,delta=F(2)**8192,F(2)**-4096,F(2)**10240,F(2)**-2048
    matter=8*(26+F(4096)/lam);assert matter<256
    # Deterministic actual N4 straight length2 target collision, one axis.
    phi=np.diag([1.,-1.,0.,0.]);h=np.zeros((256,256))
    for x in range(64):
        y=16*((x//16+2)%4)+x%16
        xx,yy=slice(4*x,4*x+4),slice(4*y,4*y+4)
        h[xx,yy]+=phi;h[yy,xx]+=phi.T
    actual_hs2=float(np.sum(h*h));wrong_without_collision=2*64*float(np.trace(phi@phi))
    assert actual_hs2==2*wrong_without_collision==512
    # Endpoint-transformed Hermitian fields: imaginary cross trace vanishes.
    a=np.diag([1.,2.,-1.,-2.]);b=np.array([[0,1j,0,0],[-1j,1,0,0],[0,0,-1,0],[0,0,0,0]],complex)
    block=2*a+3j*b
    exact_hs2=float(4*np.trace(a@a).real+9*np.trace(b@b).real)
    assert abs(float(np.sum(abs(block)**2))-exact_hs2)<1e-12
    # GLOBAL tracezero,256-mode KyFan factor8 can be saturated.
    eigen=np.concatenate([np.ones(128),-np.ones(128)])
    top=float(np.sum(np.sort(eigen)[-128:]));hs=float(np.sqrt(np.sum(eigen*eigen)))
    assert top==8*hs==128
    assert top>np.sqrt(128)*hs/2
    # Half-Young threshold32 keeps all positive terms; endpoint witness exact.
    assert F(32)**3/128==256 and 256*32==2**13
    trial=16*chi*2**6720/delta**2+64*lam*delta**4+F(59,20)*fast*delta**2+F(2)**24/3+64
    x4=128*(2**13+trial/lam);w=(lam*2**13+trial)/fast
    assert x4<2**21 and w<F(2)**-2034
    cj,ck=F(2)**16,F(2)**17
    assert cj**4*x4<2**86 and ck**4*x4<2**92
    result={'stage':'INDEPENDENT_AUDIT_PARENT_PROPOSED_SHARPENING_NOT_NEW_PRODUCER_ACCEPTANCE',
        'source_sha256':hashlib.sha256(own).hexdigest(),'input_sha256':PIN,
        'archive':{'path':archive_path.name,'sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest()},
        'HS_coefficients_exact':[[str(v) for v in c] for c in coefficients],
        'global_matter_coefficient':'8*(26+4096/Lambda)<256',
        'consequences':{'shift':'Lambda2^13','HS_fourth':'<2^21','W':'<2^-2034','charged_J_coefficient':'<2^16','charged_contact_coefficient':'<2^17','ground_J_squared':'<2^43','ground_contact':'<2^23','positive_total_weight':'<=2^43/omega,omega>0'},
        'controls':{'actual_N4_length2_HS_squared':actual_hs2,'wrong_missing_collision':wrong_without_collision,'Hermitian_c0_ici_cross_HS_error':abs(float(np.sum(abs(block)**2))-exact_hs2),'KyFan_factor8_saturation':top},
        'scope':'Exact profile arithmetic and explicit collision/cross/KyFan controls supplement analytic proof review. No full ground, IR or pole, and no immutable future producer/source receipt validated.',
        'source_inputs_unchanged':source.read_bytes()==own and (ROOT/INPUT).read_bytes()==raw}
    assert result['source_inputs_unchanged'];output.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(run(parser.parse_args().output),indent=2))
