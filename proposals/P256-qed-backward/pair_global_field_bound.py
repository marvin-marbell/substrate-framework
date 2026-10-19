#!/usr/bin/env python3
"""All120 two-cell field global classical bound and explicit coercive remainder.

Actual same Wilson pair law; not a three-axis or quantum charged-gap theorem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.linalg import expm

from full_band_gauss_section import FullBandGaussSection
from full_band_packet_bound import anchor_probe, packet_point_controls
from rotation_band import RotationBand, MIXING
from unrestricted_field_minimum import (ALIGNMENT, BARE, STIFFNESS, check, matrices,
    pair_clifford_reference, source_hashes as inherited_source_hashes)

ROOT = Path(__file__).resolve().parent


def source_hashes():
    return {**inherited_source_hashes(), **{
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('pair_global_field_bound.py', 'pair_global_field_bound.md')}}


def coefficients(model):
    c = float(model.onsite[0])
    p = model.paths[0]
    if np.max(abs(p[2:]))>1e-12 or abs(p[0].imag)>1e-12 or abs(p[1].real)>1e-12:
        raise ValueError('Actual nearest Wilson temporal/spatial pair path required')
    d, t = -float(p[0].real), -float(p[1].imag)
    if not c>abs(d):
        raise ValueError('Global pair bound requires c>|d|; do not discard negative X remainder')
    return c,d,t


def point(model, fields, link, *, stiffness, alignment) -> dict[str, Any]:
    c,d,t=coefficients(model)
    # Actual gauge transform diag(I,U) makes the one link identity. This is
    # only a coordinate choice; all fields at the right endpoint transform too.
    fixed=np.array(fields,dtype=np.complex128,copy=True)
    fixed[1]=link@fixed[1]@link.conj().T
    h=model.onebody(fixed,np.eye(4),(0.,0.))[0]
    raw_h=model.onebody(fields,link,(0.,0.))[0]
    gauge=np.eye(8,dtype=complex); gauge[4:,4:]=link
    check(float(np.linalg.norm(h-gauge@raw_h@gauge.conj().T)), 'exact pair gauge fixing')
    eig,vec=np.linalg.eigh(h)
    projector=vec[:,4:]@vec[:,4:].conj().T
    q=2*projector-np.eye(8)
    a,b,z=q[:4,:4],q[:4,4:],q[4:,4:]
    x,y=(b+b.conj().T)/2,(b-b.conj().T)/(2j)
    cross=float(np.trace((a+z)@x).real)
    check(abs(cross),'KyFan involution temporal cross identity',3e-10)
    hs=lambda v: float(np.vdot(v,v).real)
    l0=(c*a-d*x)/2,(c*z-d*x)/2
    temporal=sum(hs(v) for v in l0)
    temporal_identity=2*c*c-(c*c-d*d)*hs(x)/2-c*c*hs(y)/2
    check(abs(temporal-temporal_identity),'actual temporal dual norm identity',3e-10)
    grams=np.einsum('xmij,xnji->xmn',fixed,fixed).real/4
    u,v=float(grams[:,0,0].mean()),float(grams[:,1,1].mean())
    radial=float(np.sqrt(c*c*u+t*t*v))
    top=float(eig[4:].sum())
    sea_slack=4*radial-top
    if sea_slack < -3e-10*max(1.,abs(top)):
        raise AssertionError(f'All120 spectral bound failed: {sea_slack}')
    probe=anchor_probe(model,fields,link,(0.,0.),stiffness,alignment)
    radius=((c**4+t**4)/stiffness)**(1/3)
    optimal=np.array([c*c,t*t])/(stiffness*radius)
    diagonal_kept=stiffness/2*float(np.sum(grams[:,0,0]**2+grams[:,1,1]**2))
    full_Gram_remainder=probe['elastic']-diagonal_kept
    site_spread=stiffness/2*float(np.sum((grams[:,0,0]-u)**2+(grams[:,1,1]-v)**2))
    radial_norm_deviation=stiffness*float(np.sum((np.array([u,v])-optimal)**2))
    radial_square_remainder=2*(radial-radius)**2/radius
    remainders=dict(full_Gram=full_Gram_remainder,endpoint_norm_spread=site_spread,
        radial_norm_deviation=radial_norm_deviation,radial_square=radial_square_remainder,
        KyFan_sea=sea_slack,full_field_alignment=probe['alignment'])
    observed_excess=probe['leading_coefficient']+3*radius
    check(abs(observed_excess-sum(remainders.values())), 'full global excess decomposition',3e-9)
    if min(remainders.values()) < -3e-10 or observed_excess < -3e-10:
        raise AssertionError('Actual global nonnegative remainder failed')
    unused_field_fourth=stiffness/2*float(np.sum(grams[:,2,2]**2+grams[:,3,3]**2))
    if full_Gram_remainder < unused_field_fourth-3e-10:
        raise AssertionError('Unused field quartic confinement remainder failed')
    return dict(actual_leading_coefficient=probe['leading_coefficient'],exact_global_minimum=-3*radius,
        excess=observed_excess,nonnegative_remainders=remainders,
        unused_field_fourth_control=unused_field_fourth,
        actual_top_four=top,all120_top_four_upper=4*radial,
        temporal_dual_identity_residual=abs(temporal-temporal_identity),
        temporal_cross_identity_residual=abs(cross),positive_modes=int(np.sum(eig>0)),
        half_filling_gap=float(eig[4]-eig[3]))


def run(*, stiffness=STIFFNESS, alignment=ALIGNMENT):
    if not np.isfinite(stiffness) or stiffness<=0 or not np.isfinite(alignment) or alignment<0:
        raise ValueError('Positive finite quartic and nonnegative finite alignment required')
    identities=source_hashes()
    band=RotationBand(**BARE)
    model=FullBandGaussSection(band,heavy_scale=1.)
    c,d,t=coefficients(model)
    frame,reference=pair_clifford_reference(model,stiffness=stiffness)
    anchor=np.tile(frame,(2,1,1,1))
    equal=point(model,anchor,np.eye(4),stiffness=stiffness,alignment=alignment)
    check(abs(equal['excess']),'all120 global minimum attained',3e-10)
    rng=np.random.default_rng(222120)
    probes=[]
    for number in range(96):
        coordinates=rng.normal(size=(2,4,15))*(.02,.08,.25,.7)[number%4]
        fields=matrices(coordinates,model.basis)
        link=expm(1j*np.einsum('a,aij->ij',rng.normal(size=15)*.3,model.generators))
        probes.append(point(model,fields,link,stiffness=stiffness,alignment=alignment))
    # A real wrong-domain operator, not a numerical echo of the inequality:
    # with c<|d|, uniform temporal fields have top4=4|d|b>4cb.
    bad=FullBandGaussSection(RotationBand(a=1.,m=-1.9,r=2.,v=.3,improved=False),heavy_scale=1.)
    bad_fields=np.zeros((2,4,4,4),dtype=complex); bad_fields[:,0]=.3*MIXING
    bad_c=float(bad.onsite[0]); bad_d=float(abs(bad.paths[0][0]))
    bad_top=float(np.linalg.eigvalsh(bad.onebody(bad_fields,np.eye(4),(0.,0.))[0])[4:].sum())
    bad_upper=4*abs(bad_c)*.3
    if bad_top<=bad_upper+1e-5:
        raise AssertionError('Missing c>|d| failed to reject a genuinely wrong global bound')
    rejected=False
    try:
        coefficients(bad)
    except ValueError:
        rejected=True
    if not rejected:
        raise AssertionError('Wrong-domain coefficient acceptance boundary failed')
    primitive=packet_point_controls(model,anchor,np.eye(4),(0.,0.))
    if source_hashes()!=identities:
        raise RuntimeError('Actual source/proof bytes changed during global-field execution')
    return dict(source_sha256=identities,source_bindings_unchanged=True,
        bare_coefficients={**BARE,'stiffness':stiffness,'alignment':alignment},
        actual_pair_coefficients=dict(c=c,d=d,t=t),reference=reference,attainment=equal,
        all120_nonuniform_noncentral_probes=probes,actual_primitive_attainment=primitive,
        adverse_domain=dict(c=bad_c,d=bad_d,actual_top_four=bad_top,
            invalid_upper=bad_upper,violated_by=bad_top-bad_upper,rejected=rejected),
        all_checks_pass=True,
        theorem='Exact global classical whole-heavy plus zero-preference quartic/alignment coefficient over all120 pair fields and SU4 link when c>|d|',
        quantum_statement='Pointwise operator lower bound extends to all120-field wavefunctions; no quantum ground concentration or charged spectral difference yet',
        limits=['Two-cell one-active-x-axis m+r/a owner, not periodic three-axis bulk.',
            'Classical minimum and exact scalar coercive remainder, not full quantum ground energy.',
            'No charged lower gap, physical spin/quartet, compact photon phase or campaign completion.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stiffness',type=float,default=STIFFNESS)
    parser.add_argument('--alignment',type=float,default=ALIGNMENT)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(stiffness=args.stiffness,alignment=args.alignment)
    text=json.dumps(result,indent=2,allow_nan=False)
    args.output.write_text(text+'\n')
    print(text)


if __name__=='__main__':
    main()
