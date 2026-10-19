"""Native normalized-influence gates and signed fullE history comparison.
No interacting Gibbs expectations, old entrypoints or eigensolvers are run.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, cast
import sympy as sp
from magnetic_boundary_slab_scientist_20261011 import FixedMagneticSlab, dirichlet_solve
from magnetic_boundary_scientist_20261011 import car_transfer, exact_matvec, generic_phi

ROOT = Path(__file__).resolve().parent
rational_matvec = cast(Callable[[Any, list[F | int]], list[F]], exact_matvec)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_form(slab: FixedMagneticSlab, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    n, v = slab.n, slab.v
    c = n//2-1
    start = slab.index[(c, 1, 1)]
    path = slab.path(start, 0, 1)
    assert path is not None
    target = path['target']
    periodic = slab.native.path(start, 0, 1)
    assert periodic['edges'] == path['edges'] and periodic['seam'] == path['seam'] == 1
    row = next(r for r in profiles[1]['paths'] if r['axis'] == 0 and r['length'] == 1)
    fx, fy = generic_phi(2), generic_phi(5)
    # Noncommuting exact SU4 probe; not a frozen-field thermal replacement.
    u = sp.Matrix([[0,1,0,0],[-1,0,0,0],[0,0,0,sp.I],[0,0,sp.I,0]])
    assert u.adjoint()*u == sp.eye(4) and u.det() == 1
    f0 = sp.Rational(row['c0'])*fx[0]+sp.I*sp.Rational(row['ci'])*fx[1]
    f1 = sp.Rational(row['c0'])*fy[0]+sp.I*sp.Rational(row['ci'])*fy[1]
    b = (f0*u+u*f1)/2
    radius2 = sum(sp.trace(a*a) for a in fx+fy)
    price = abs(F(row['c0']))+abs(F(row['ci']))
    cut = b/(8*sp.Rational(price)*sp.sqrt(1+radius2))
    assert cut != sp.zeros(4) and f0*u != u*f0
    # Full masks:32 complementary charged pairs,65 independent species,Ref4.
    even = sum(15 << (4*i) for i, xyz in enumerate(slab.sites) if xyz[0]%2 == 0)
    if even & (1 << (4*start)):
        even ^= 15 << (4*start)
        even ^= 15 << (4*target)
    complement = ((1 << (4*v))-1)^even
    masks = [even if s%2 == 0 else complement for s in range(64)]+[even]
    assert all(mask.bit_count() == 2*v for mask in masks)
    assert all(sum((mask >> (4*i)&15).bit_count()-2 for mask in masks[:64]) == 0 for i in range(v))
    initial = sum(mask << (4*v*s) for s, mask in enumerate(masks)) | (15 << (4*v*65))
    image = slab.apply_path(initial, 0, path, cut, {}, True)
    assert image and all((mask & ((1 << (4*v))-1)).bit_count() == 2*v for mask in image)
    local: list[F | int] = [F(0)]*len(slab.dynamic)
    local[slab.dynamic_lookup[path['edges'][0]]] = 1
    drho = rational_matvec(slab.DI, local)
    assert sum(abs(x) for x in drho) == 2
    # Actual active insertion changes fullrho, so its affine update is mandatory.
    action = car_transfer(initial, 4*target, 4*start)
    assert action is not None
    after, sign = action
    reverse = car_transfer(after, 4*start, 4*target)
    assert reverse is not None and reverse[0] == initial and reverse[1]*sign == 1
    boundary: list[F | int] = [F(0)]*len(local)
    bmask = initial
    bsign = 1
    for y in range(n):
        for z in range(n):
            p = slab.path(slab.index[(0,y,z)], 0, 1)
            assert p is not None
            action = car_transfer(bmask, 4*v+4*p['target'], 4*v+4*p['source'])
            assert action is not None
            bmask, sgn = action
            bsign *= sgn
            boundary[slab.dynamic_lookup[p['edges'][0]]] = 1
    assert (bmask >> (4*v) & ((1 << (4*v))-1)).bit_count() == 2*v
    # Exact 1D Poisson restriction for the plane; no native spectral solve.
    rho = rational_matvec(slab.DI, boundary)
    rowrho = [rho[j] for j, site in enumerate(slab.interior) if slab.sites[site][1:] == (0,0)]
    potential = dirichlet_solve(rowrho)
    phi = [potential[slab.sites[site][0]] for site in slab.interior]
    longitudinal = exact_matvec(slab.DI.T, phi)
    assert exact_matvec(slab.DI, longitudinal) == rho
    cross = sum((a*b for a,b in zip(local,longitudinal,strict=True)),F(0))
    assert cross == -F(1,n-1)
    rawcross = sum((a*b for a,b in zip(local,boundary,strict=True)),F(0))
    assert rawcross == 0
    qcross = rawcross-cross
    assert qcross == F(1,n-1)
    # Both excursions occupy the SAME half-time and return; fullrho starts0.
    # Longitudinal joint exponent cross=64*cross; transverse=-32*cross.
    hc_cross = 64*cross
    memory_cross = -32*cross
    total = hc_cross+memory_cross
    assert total == -F(32,n-1) and total != hc_cross
    return {'N':n,'support_vertices':[list(slab.sites[start]),list(slab.sites[target])],
            'support_edges':path['edges'],'bulk_distance_to_boundary':min(c,n-2-c),
            'physical_hopping':1,'canonical_species_counts':2*v,'all64_initial_fullrho_zero':True,
            'active_onsite':profiles[1]['onsite'],'aux63_onsite':profiles[2]['onsite'],
            'local_cutoff_transport_CAR_images':len(image),'local_norm_upper':1,
            'noncommuting_fullPhi_SU4_probe':True,'affine_local_charge_change':list(map(str,filter(None,drho))),
            'boundary_plane_aux1_native_excursions':n*n,'boundary_plane_is_tangential_magnetic_source':False,
            'longitudinal_cross_exact':str(cross),'transverse_cross_exact':str(qcross),
            'HC_cross_exponent_per_beta':str(hc_cross),'shared_b_memory_cross_per_beta':str(memory_cross),
            'joint_minus_independent_exponent_per_beta':str(total),
            'joint_over_independent_electric_factor':'exp(32*beta/(N-1))',
            'connected_electric_factor_relative':'expm1(32*beta/(N-1))',
            'conditional_factor_relative_beta1':math.expm1(32/(n-1)),
            'conditional_factor_beta_window':'0<beta<=b0 finite: <=(32*b0/(N-1))*exp(32*b0/(N-1))',
            'reset_b_wrongcase_exponent':str(hc_cross),'double_HC_wrongcase_exponent':str(total+hc_cross),
            'wrongcases_rejected':total != hc_cross and total+hc_cross != total,
            'full_quantum_word_coefficient_integrated':False,'native_Gibbs_expectation':False}


def native_tail(n: int, cj: F, ck: F) -> dict[str, Any]:
    v, boundary = n**3,2*n*n
    fourth = F(cj**4,4)*boundary**3*v
    # Crude normalized local covariance, not response divided by volume.
    # |d omega(A)/d alpha| <=beta sqrt(<J^2>) for ||A||<=1.
    phi_tail_gap = F(4*65,2**2048)
    colour_tail_gate = F(3*64**2,20*16*8*2**4096)
    assert phi_tail_gap < F(1,2**2039) and colour_tail_gate < 7
    stripe_delta_per_volume = -F(65536*n*n,n-1)
    return {'N':n,'crude_local_influence_cap_fourth_over_beta4_Cbeta':str(fourth),
            'crude_cap_scaling':'beta*CJ*2^(1/4)*Cbeta^(1/4)*N^(9/4)',
            'fixed_beta_epsilon_gate_pass':False,'contact_coefficient':str(ck),
            'boundary_sites':boundary,'Phi64_reference_tail_lower':'1-260*2^-2048',
            'Phi64_reference_tail_not_native_projection_error':True,
            'SU4_shell64_decreasing_tail_gate_uR2_less_than7':colour_tail_gate < 7,
            'native_Galerkin_Trotter_all_b_integration_paid':False,
            'canonical_stripe_HC_slab_minus_periodic_per_V':str(stripe_delta_per_volume),
            'surface_operator_gate_rejected':True}


def denominator_control() -> dict[str, Any]:
    # Exact superselection comparison witness, not a native Gibbs number.
    zp,zs,ap,ass = F(2),F(3),F(1,3),F(-2,5)
    s=F(2,5)
    z=(1-s)*zp+s*zs
    mean=((1-s)*zp*ap+s*zs*ass)/z
    derivative=zp*zs*(ass-ap)/z**2
    jp,js=1/(1-s),-1/s  # beta=1 sector source jets
    jmean=((1-s)*zp*jp+s*zs*js)/z
    aj=((1-s)*zp*ap*jp+s*zs*ass*js)/z
    assert derivative == -(aj-mean*jmean)
    assert derivative != (zs*ass-zp*ap)/z
    return {'scope':'METHOD_EXACT_RATIONAL_SUPERSELECTION_DENOMINATOR_NOT_NATIVE_TRACE_VALUES',
            'normalized_connected_derivative':str(derivative),'unnormalized_wrongcase_rejected':True,
            'local_state_from_pressure_rejected':True}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    for name,pin in request['pins'].items():
        assert digest(ROOT/name) == pin,name
    profiles=json.loads((ROOT/request['profile']).read_text())['primitive_profiles']
    assert [p['load'] for p in profiles] == [1,1,63]
    assert profiles[1]['onsite'] != profiles[2]['onsite']
    cj=sum((8*p['load']*(abs(F(r['c0']))+abs(F(r['ci'])))*r['length']
            for p in profiles if p['scale']=='charged' for r in p['paths'] if r['axis']==2),F(0))
    ck=sum((8*p['load']*(abs(F(r['c0']))+abs(F(r['ci'])))*r['length']**2
            for p in profiles if p['scale']=='charged' for r in p['paths'] if r['axis']==2),F(0))
    assert cj == F(15475017073321605,1099511627776)
    result={'kind':'NATIVE_CONNECTED_LOCAL_INSERTION_AND_GAUGE_COMPLETE_HISTORY_INFLUENCE_NOT_STATE_IDENTITY',
            'source_sha256':digest(Path(__file__)),'request_sha256':digest(args.request),
            'input_sha256':request['pins'],'physical_hopping':1,'CJ':str(cj),'CK':str(ck),
            'native_forms':[local_form(FixedMagneticSlab(n,1),profiles) for n in request['N']],
            'native_tails_and_failed_crude_gates':[native_tail(n,cj,ck) for n in request['N']],
            'normalized_denominator_control':denominator_control(),
            'comparison_common_closed_form_family':'physical slab source alpha on fixed domain; enlarged direct-sum endpoint comparison has common interior domain but is not a local microscopic geometry interpolation',
            'conditional_electric_epsilon_N_achieved':True,
            'full_normalized_magnetic_epsilon_N_achieved':False,
            'local_state_convergence':False,'common_local_dynamics_convergence':False,
            'mixed_spectral_convergence':False,'wholeP0_convergence':False,
            'controlled_native_state_temperature_window':None,'conditional_electric_temperature_window':'every bounded 0<beta<=b0',
            'native_numeric_Gibbs_expectation':False,'old_entrypoints_rerun':False}
    for name,pin in request['pins'].items():
        assert digest(ROOT/name) == pin,name
    with args.output.open('x') as out:
        json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps({'N':request['N'],'physical_hopping':1,'conditional_signed_fullE':'paid','native_state_identity':False}))


if __name__ == '__main__':
    main()
