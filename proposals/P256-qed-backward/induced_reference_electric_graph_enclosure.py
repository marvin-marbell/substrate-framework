#!/usr/bin/env python3
"""NEW electric-priced graph-source Gaussian energy enclosure, unchanged Href.

Exact closed Peierls loops plus uniform rational tails on the frozen orthogonal
Phi reference. This is neither a stationary photon vacuum nor a source cutover.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any
import numpy as np
from mpmath import iv as _interval_context
from induced_phi_backreaction import PhiConfig, SELECTED, models, scalar_paths
from induced_phi_em_gaussian_variational import SOURCE_BYTES as NATIVE_BYTES
from induced_phi_ir_well_transfer import rational

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**NATIVE_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
SOURCE_HASHES={name:hashlib.sha256(raw).hexdigest() for name,raw in SOURCE_BYTES.items()}
# mpmath1.3 constructs ivmpf arithmetic dynamically; its class declaration
# omits the installed operators and accepts misleading constructor types.
iv: Any=_interval_context
iv.dps=50


def exact(value):
    f=value if isinstance(value,Fraction) else Fraction(value)
    return iv.mpf(f.numerator)/f.denominator


def bounds(value):
    endpoints=[]
    for sign,mantissa,exponent,_ in value._mpi_:
        fraction=Fraction((-1)**sign*mantissa)*(Fraction(2)**exponent)
        endpoints.append(str(fraction))
    return dict(lower=endpoints[0],upper=endpoints[1],decimal_display=str(value))


def flow_key(flow):
    return tuple(sorted((site,axis,value) for (site,axis),value in flow.items() if value))


def change(flow,site,axis,value):
    key=(site,axis); flow[key]=flow.get(key,0)+value
    if not flow[key]: del flow[key]


def walk_coefficients(length):
    closed=defaultdict(int); mass_two=defaultdict(int); flow={}
    def visit(depth,position,mask,spin_sign,direction_sign):
        if depth==length:
            phase=(-1)**(length//2)*spin_sign*direction_sign
            if position==(0,0,0):
                if mask: raise ArithmeticError('Closed microscopic Clifford word not scalar')
                closed[flow_key(flow)]+=phase
            elif sum(abs(v) for v in position)==2 and sum(v!=0 for v in position)==1:
                if mask: raise ArithmeticError('Actual two-link closure Clifford word not scalar')
                axis=next(i for i,v in enumerate(position) if v); direction=-1 if position[axis]>0 else 1
                current=position; added=[]
                for _ in range(2):
                    target=list(current); target[axis]+=direction; target=tuple(target)
                    origin=current if direction==1 else target
                    change(flow,origin,axis,direction); added.append((origin,axis,direction)); current=target
                mass_two[flow_key(flow)]+=phase
                for site,axis,direction in reversed(added): change(flow,site,axis,-direction)
            return
        # Closure requires endpoint zero or a native mass two-link endpoint.
        if sum(abs(v) for v in position)>length-depth+2: return
        for axis in range(3):
            crosses=(mask>>(axis+1)).bit_count(); next_sign=spin_sign*(-1 if crosses%2 else 1)
            for direction in (-1,1):
                target=list(position); target[axis]+=direction; target=tuple(target)
                origin=position if direction==1 else target
                change(flow,origin,axis,direction)
                visit(depth+1,target,mask^(1<<axis),next_sign,direction_sign*direction)
                change(flow,origin,axis,-direction)
    visit(0,(0,0,0),0,1,1)
    return dict(closed),dict(mass_two)


def Gaussian_sum(coefficients,variance):
    return sum((count*iv.exp(-variance*sum(value**2 for _,_,value in key)/2)
                for key,count in coefficients.items()),iv.mpf(0))


def pair_Gaussian_sum(left,right,axis,variance):
    total=iv.mpf(0); shift=tuple(int(i==axis) for i in range(3))
    for key,count in left.items():
        first={(site,a):value for site,a,value in key}
        for other,number in right.items():
            combined=dict(first)
            for site,a,value in other:
                moved=tuple(site[i]+shift[i] for i in range(3)); combined[(moved,a)]=combined.get((moved,a),0)+value
            total+=count*number*iv.exp(-variance*sum(value**2 for value in combined.values())/2)
    return total


def photon_CAR_graph_price(t_loads: list[tuple[Any,int]], *, size: int,
                           density_variance: Any, electric_coefficient: Any) -> dict[str,Any]:
    """Bound ||u sum Pi_Q^2 Psi||, including ALL64 second-ray PH pairs.

    A unit physical direction has ||Y'||F,||Y''||F<=2t (sqrt2 more
    for the two opposite N2 links sharing an unordered site pair).
    Flat determinant rays have ||Omega'||^2=g and
    ||Omega''||^2<=3g^2+||C_PH||F^2. This includes dGamma(T)^2,
    not the zero one-body T@T. The linear first-ray map also prices
    the Gaussian cross derivative without a spurious extra sqrt(M).
    """
    modes=2*size**3+1; gcap=iv.mpf(0); curvature_squared=iv.mpf(0)
    per_profile=[]
    for t,load in t_loads:
        y=6*t; b=2*t*iv.sqrt(2 if size==2 else 1)
        ddR=8*y*y*b*b+2*y*b+2*b*b
        curvature=iv.sqrt(2)*((1+y)*ddR+b+4*y*b*b)
        gcap+=exact(load)*b*b
        curvature_squared+=exact(load)*curvature*curvature
        per_profile.append(dict(load=load,first_and_second_Y_Frobenius_cap=bounds(b),
            second_projector_curvature_Frobenius_cap=bounds(curvature)))
    second_ray=iv.sqrt(3*gcap*gcap+curvature_squared)
    bare=iv.sqrt(exact(modes*(modes+2)))/(4*density_variance)
    cross=iv.sqrt(gcap*modes/density_variance)
    ray=modes*second_ray
    coupled=electric_coefficient*(bare+cross+ray)
    old_packet=electric_coefficient*iv.sqrt(exact(Fraction(63*modes,2)+9*modes*(modes-1)))*64
    return dict(real_EM_coordinates=modes,all64_PH_first_metric_cap=bounds(gcap),
        all64_second_ray_norm_cap_per_unit_direction=bounds(second_ray),
        bare_Gaussian_Pi_squared_norm=bounds(electric_coefficient*bare),
        actual_Gaussian_cross_ray_norm_cap=bounds(electric_coefficient*cross),
        actual_all64_second_ray_norm_cap=bounds(electric_coefficient*ray),
        actual_EM_CAR_kinetic_graph_norm_upper=bounds(coupled),
        old_radius_one_eighth_packet_EM_graph=bounds(old_packet),
        new_graph_to_old_EM_graph_ratio_upper=bounds(coupled.b/old_packet.a),
        per_profile=per_profile,
        proof='Exact graph R derivatives, local native edge Frobenius supports, flat determinant line and all64 CAR PH orthogonality. Omega second-ray vacuum,1PH,2PH pieces priced together; 2PH norm squared<=2g^2. Gaussian cross derivative uses the full linear directional ray map. Coulomb/charged operator caps and Phiweighted/Heinz transfer require their original universal-domain theorem; no automatic Phi admission.')


def compute_enclosure(mean_path: str | Path, *, order: int=2, size: int=262144,
                      density_variance: float=2.) -> dict[str, Any]:
    """Exact Gaussian polynomial expectations and per-volume rational remainder.

    All harmonics remain. The returned source is distinct from both h_eta
    eigen-Slaters and momentum preparations. Phi quantum transfer is separate.
    """
    mean_path=Path(mean_path)
    if order<0 or size<=2*order+4 or density_variance<=0:
        raise ValueError('Need positive variance and N>2*order+4 for nonwrapping native loop proof')
    start=time.monotonic(); K=order; N=size; V=N**3; variance=exact(density_variance)
    mean = json.loads(mean_path.read_bytes())
    cfg = PhiConfig(Lambda=1.0, **mean['microscopic_config'])
    if tuple(int(load) for load,_,_ in models(cfg)[1:])!=(1,63):
        raise ValueError('Assigned source requires actual active1 plus auxiliary63 canonical CAR loads')
    g0lo = rational(mean['centre_enclosure']['g0_lower'])
    g0hi = rational(mean['centre_enclosure']['g0_upper'])
    gslo = rational(mean['centre_enclosure']['gs_lower'])
    gshi = rational(mean['centre_enclosure']['gs_upper'])
    g0 = iv.mpf([exact(g0lo).a, exact(g0hi).b])
    gs = iv.mpf([exact(gslo).a, exact(gshi).b])
    root0 = iv.sqrt(g0)
    roots = iv.sqrt(gs)
    gamma5 = SELECTED[0] @ SELECTED[1] @ SELECTED[2] @ SELECTED[3]
    plus = (np.eye(4) + SELECTED[0]) / 2
    L = [-1j * SELECTED[j] @ gamma5 for j in range(1, 4)]
    algebra_error = max((float(np.max(abs(L[j] @ L[k] + L[k] @ L[j] - 2 * (j == k) * np.eye(4)))) for j in range(3) for k in range(3)))
    if algebra_error:
        raise ArithmeticError('Actual primitive offdiagonal Clifford reduction failed')
    if np.trace(plus @ L[0] @ L[1] @ L[0] @ L[1]) != -2:
        raise ArithmeticError('Wrong literal primitive plaquette spin trace')
    coefficients = {n: walk_coefficients(n) for n in range(0, 2 * K + 3, 2)}
    pair_sums = {(p, q, a): pair_Gaussian_sum(coefficients[2 * p][0], coefficients[2 * q][0], a, variance) for p in range(K + 1) for q in range(K + 1) for a in range(3)}
    profiles = []
    total = iv.mpf(0)
    tail = iv.mpf(0)
    u = exact(1) / exact(cfg.a)
    graph_t_loads=[]
    for load, model, role in models(cfg)[1:]:
        onsite, paths = scalar_paths(model)
        path1 = next((c for d, c in paths if d.tolist() == [1, 0, 0]))
        path2 = next((c for d, c in paths if d.tolist() == [2, 0, 0]))
        mass0 = exact(float(onsite)) * root0
        mass1 = exact(float(path1[0].real)) * root0
        mass2 = exact(float(path2[0].real)) * root0
        denominator = 2 * mass0 + u
        t = exact(float(abs(path1[1].imag))) * roots / denominator
        graph_t_loads.append((t,int(load)))
        rho = (6 * t) ** 2
        if mass0.a<=0 or denominator.a<=0 or rho.b>=1:
            raise ValueError('Native graph onsite/gauge-pair positivity or rational contraction premise failed')
        traces = {n: 2 * t ** n * Gaussian_sum(coefficients[n][0], variance) for n in coefficients}
        mass_traces = {n: mass0 * traces[n] + 2 * mass2 * t ** n * Gaussian_sum(coefficients[n][1], variance) for n in coefficients}
        car = sum(((-1) ** n * (mass_traces[2 * n + 2] - mass_traces[2 * n] - 2 * denominator * traces[2 * n + 2]) for n in range(K + 1)), iv.mpf(0))
        metric = sum((4 * t ** 2 * (-1) ** (p + q) * t ** (2 * p + 2 * q) * pair_sums[p, q, a] for p in range(K + 1) for q in range(K + 1) for a in range(3)), iv.mpf(0))
        Mmax = mass0 + 6 * abs(mass1) + 6 * abs(mass2)
        Bmax = Mmax * (1 + rho) + 2 * denominator * rho
        delta = rho ** (K + 1) / (1 + rho)
        car_tail = 2 * Bmax * delta
        metric_tail = u * 24 * t ** 2 * delta
        total += exact(load) * (car + u * metric)
        tail += exact(load) * (car_tail + metric_tail)
        profiles.append(dict(role=role, load=load, graph_denominator=bounds(denominator), graph_norm_squared_upper_interval=bounds(rho), exact_polynomial_CAR_per_V=bounds(car), exact_polynomial_full_link_metric_per_V=bounds(metric), uniform_CAR_remainder_per_V=bounds(car_tail), uniform_Born_plus_Gauss_remainder_per_V=bounds(metric_tail)))
    kinetic = u * exact(2 + Fraction(1, V)) / (4 * variance)
    centre = total + kinetic
    result = dict(order=K, N=N, V=V, real_EM_coordinates=2 * V + 1, density_variance=2, source='NEW normalized pure Gaussian EM × all64 Hermitian graph Slaters Y=Kphysical/(2Monsite+UE/a), canonical positive projected-onsite ray; not h_eta eigen-Slater, not a cutover.', actual_Href_energy_per_V_enclosure=bounds(centre + iv.mpf([-tail.b, tail.b])), exact_polynomial_energy_per_V=bounds(centre), uniform_omitted_rational_tail_per_V=bounds(tail), photon_kinetic_per_V=bounds(kinetic), profiles=profiles, actual_primitive_Clifford_algebra_error=algebra_error, loop_flow_counts={str(n): dict(closed=len(c[0]), physical_mass_two_link_closures=len(c[1])) for n, c in coefficients.items()}, proof='R=(I+Y2)^-1; ||Y||<=6t<1; alternating R_K contraction <=1 and ||R-R_K||<=rho^(K+1)/(1+rho). Closed integer Peierls currents satisfy Dell=0, so Q Q^T ell=ell with ALLharmonics. Gaussian mean is exp[-variance||ell||2/2]. True Coulomb + coexact Born recombines FULLlink ray metric, not removed. CAR tail <=2||M(Y2-I)-2KphysY||delta/Vrankfactor; metric tail <=sumedges2delta||dYedge||F2. No low-order AP wrapping because N>2K+4. Exact scalar coefficients from actual scalar_paths API; full input centre intervals, not midpoint.', source_hashes=SOURCE_HASHES, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), script_snapshot=Path(__file__).read_text(), input_sha256=hashlib.sha256(mean_path.read_bytes()).hexdigest(), input_snapshot=mean, elapsed_seconds=time.monotonic() - start, limitations='Frozen orthogonal Phi reference only. All1+63 chargedCAR, realcoexact/harmonic EM and Gauss law priced; no fullPhi quantum correction, stationarity/groundstate/propagator/pole/macro-current admission. mpmath interval elementary arithmetic, scientific loop/tail derivation needs independent mathematical owner review.')
    onsite_car=iv.mpf(0)
    for load,model,_ in models(cfg)[1:]:
        onsite_scalar,_=scalar_paths(model)
        onsite_car-=2*exact(load)*exact(float(onsite_scalar))*root0
    prepared_enclosure=centre+iv.mpf([-tail.b,tail.b])
    result['actual_EM_CAR_source_graph_price']=photon_CAR_graph_price(graph_t_loads,size=N,density_variance=variance,electric_coefficient=u)
    result['density_variance']=density_variance
    result['same_Href_onsite_CAR_any_EM_product_infimum_per_V']=bounds(onsite_car)
    result['same_Href_onsite_CAR_sigma4_energy_per_V']=bounds(onsite_car+u*exact(2+Fraction(1,V))/16)
    result['gain_lower_vs_onsite_CAR_any_EM_product_infimum_per_V']=bounds(onsite_car.a-prepared_enclosure.b)
    result['source_snapshots']={name:raw.decode() for name,raw in SOURCE_BYTES.items()}
    result['canonical_CAR_scope']='All8 flat encodedP sectors only; Q penalty separately priced. Each1+63 flavour has global2V particles, not two per momentum.'
    result['assigned_scientific_epoch']='03b34911221ba1898315373edaa7c4c3bcac718462e7fe51a842560eb6933f87'
    for name,raw in SOURCE_BYTES.items():
        if (ROOT/name).read_bytes()!=raw:
            raise RuntimeError('Source drift: '+name)
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mean',type=Path,required=True)
    parser.add_argument('--order',type=int,default=2)
    parser.add_argument('--size',type=int,default=262144)
    parser.add_argument('--variance',type=float,default=2.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compute_enclosure(args.mean,order=args.order,size=args.size,density_variance=args.variance)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('actual_Href_energy_per_V_enclosure','gain_lower_vs_onsite_CAR_any_EM_product_infimum_per_V','elapsed_seconds')}))
    print('Natural exit:',args.output)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
