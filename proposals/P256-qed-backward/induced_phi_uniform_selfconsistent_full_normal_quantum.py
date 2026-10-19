"""NEW finiteN4 selfconsistent uniform nonlinear mean and full45 quantum tangent.

The unique positive implicit Gram root includes physical negative heavy -Lambda h
and all charged1+63. An exact matrix-functional Gaussian wave coefficient, NOT
NumPy observations, is declared for the next uniform scalar/CAR consumer.
Frozen2832/ebd and the macro/source laws are unchanged.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import importlib.abc
import importlib.machinery
from itertools import product
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
OWN_BYTES = Path(__file__).read_bytes()
CAPTURED: dict[str, bytes] = {Path(__file__).name:OWN_BYTES}


class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str) -> Any:
        raw = Path(self.path).read_bytes(); CAPTURED[Path(self.path).name] = raw
        return self.source_to_code(raw,self.path)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        spec = importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is not None and spec.origin is not None:
            p = Path(spec.origin)
            if p.parent==ROOT and p.suffix=='.py':
                spec.loader = Loader(fullname,str(p)); return spec
        return None


finder = Finder(); sys.meta_path.insert(0,finder)
import induced_phi_uniform_full_normal_quantum as full
sys.meta_path.remove(finder)
CAPTURED.update(full.SOURCE_BYTES)
import mpmath as mp
import numpy as np

iv: Any = mp.iv
exact,endpoint,enclosure,bounds = full.exact,full.endpoint,full.enclosure,full.bounds
FROZEN_SOURCE_SHA = '2832ab998db0bcef0f21fc984a40d63f4e953d36674a63058563a9612260db80'
FROZEN_RECEIPT_SHA = 'ebd076228091607848e98f859cb8c6621548363f8f3a9c261925725ac0749267'
ROOT_DIGITS = 2600
ENCLOSURE_DIGITS = 2150


def midpoint(value: Any) -> F:
    return (endpoint(value,0)+endpoint(value,1))/2


def number(value: F, ctx: Any) -> Any:
    return ctx.mpf(value.numerator)/value.denominator


def radial_data(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for p in profiles:
        nearest = [v for v in p['paths'] if v['length']==1]
        second = [v for v in p['paths'] if v['length']==2]
        if len(nearest)!=3 or len({v['c0'] for v in nearest})!=1 or len({v['ci'] for v in nearest})!=1 or any(v['ci']!=0 for v in second):
            raise ArithmeticError('Exact quarterAP cubic coefficient reduction failed')
        rows.append(dict(role=p['role'],load=p['load'],scale=p['scale'],onsite=p['onsite'],c0=nearest[0]['c0'],ci=nearest[0]['ci']))
    return rows


def radial_force_jacobian(g0: Any, gs: Any, rows: list[dict[str, Any]], ctx: Any) -> tuple[list[Any],list[list[Any]],Any]:
    a = number(F(1,64),ctx); Lambda = number(F(2)**8192,ctx)
    force = [2*g0/a,2*gs/a]
    jac = [[2/a,ctx.mpf(0)],[ctx.mpf(0),2/a]]
    energy = (g0*g0+3*gs*gs)/a
    sqrt2 = ctx.sqrt(2)
    for row in rows:
        scale = ctx.mpf(1) if row['scale']=='heavy' else ctx.mpf(row['load'])/Lambda
        onsite,c0,ci = (number(row[k],ctx) for k in ('onsite','c0','ci'))
        spatial = 6*ci*ci
        for z,weight in ((3,1),(1,3),(-1,3),(-3,1)):
            x = onsite+sqrt2*c0*z; mass = x*x
            E = ctx.sqrt(g0*mass+gs*spatial); load = scale*weight/8
            force[0] -= load*mass/E
            force[1] -= load*spatial/(3*E)
            jac[0][0] += load*mass*mass/(2*E**3)
            jac[0][1] += load*mass*spatial/(2*E**3)
            jac[1][0] += load*mass*spatial/(6*E**3)
            jac[1][1] += load*spatial*spatial/(6*E**3)
            energy -= 2*load*E
    return force,jac,energy


def stationary_root(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mp.mp.dps = ROOT_DIGITS
    def f(x: Any,y: Any) -> tuple[Any,Any]:
        force,_,_ = radial_force_jacobian(x,y,rows,mp.mp)
        return force[0],force[1]
    def jac(x: Any,y: Any) -> Any:
        _,J,_ = radial_force_jacobian(x,y,rows,mp.mp)
        return mp.matrix(J)
    root = mp.findroot(f,(mp.mpf('.289'),mp.mpf('.1048')),J=jac,tol=mp.mpf(10)**-2500,maxsteps=40,solver='mdnewton')
    centres = [F(str(mp.nstr(x,2250))) for x in root]; r = F(1,10**ENCLOSURE_DIGITS)
    iv.dps = 2300
    point = [exact(x) for x in centres]; box = [enclosure(x-r,x+r) for x in centres]
    fpoint,Jpoint,_ = radial_force_jacobian(point[0],point[1],rows,iv)
    _,Jbox,energy = radial_force_jacobian(box[0],box[1],rows,iv)
    # Exact rational inverse of the centre Jacobian midpoint is the fixed
    # preconditioner; no unpriced floating Newton coefficients enter proof.
    Jrat = [[midpoint(v) for v in row] for row in Jpoint]
    determinant = Jrat[0][0]*Jrat[1][1]-Jrat[0][1]*Jrat[1][0]
    C = [[Jrat[1][1]/determinant,-Jrat[0][1]/determinant],[-Jrat[1][0]/determinant,Jrat[0][0]/determinant]]
    contraction = F(0); inclusion = []
    for i in range(2):
        residual = sum(exact(C[i][j])*fpoint[j] for j in range(2))
        derivative = [exact(int(i==j))-sum(exact(C[i][k])*Jbox[k][j] for k in range(2)) for j in range(2)]
        lipschitz = sum(endpoint(abs(v),1) for v in derivative)
        contraction = max(contraction,lipschitz)
        image_radius = endpoint(abs(residual),1)+lipschitz*r
        inclusion.append(image_radius)
    if contraction>=1 or max(inclusion)>=r:
        raise ArithmeticError('Actual all65 stationary root fixed-point inclusion failed')
    forcebox,_,_ = radial_force_jacobian(box[0],box[1],rows,iv)
    # Only the four diagonal grade1 forces can survive. Cubic AP permutations
    # make spatial forces identical; reflections kill mixed Gram entries.
    # Bounds avoid interval-squaring dependency crossing zero by using abs.
    force45_upper = iv.sqrt((16*iv.sqrt(box[0])/iv.sqrt(8)*abs(forcebox[0]))**2+3*(16*iv.sqrt(box[1])/iv.sqrt(8)*abs(forcebox[1]))**2)
    parameters = dict(g0=bounds(box[0]),gs=bounds(box[1]))
    return dict(parameters=parameters,centres=centres,box=box,
                centre_exact_rational=[str(x) for x in centres],
                Newton_preconditioner_exact_rational=[[str(x) for x in row] for row in C],
                root_radius=bounds(exact(r)),contraction=bounds(exact(contraction)),
                image_radii=[bounds(exact(x)) for x in inclusion],
                radial_force_intervals=[bounds(x) for x in forcebox],
                ALL45_force_norm_divided_Lambda_upper=bounds(force45_upper),
                nonlinear_energy_per_site_divided_Lambda=bounds(energy),
                proof='The exact preconditioned Newton map T(g)=g-C F(g) maps this rational box strictly into itself and contracts in infinity norm. The unique root is positive. The Gram energy is strongly convex (primitive Hessian >=2/a=128); cubic AP reflections/permutations force diagonal cubic G and kill all offdiagonal Gram gradients. Clifford saturation attains the universal both-band sea bound in ALL60 uniform Cartesian fields. At the implicit root all45 normal forces vanish EXACTLY, not merely at printed midpoint. Grade2/3 force traces vanish and grade1 offdiagonal traces cancel by actual AP reflections. Box residual is a numerical-enclosure price, not a nonzero implicit-root force.')


def input_mean(old: dict[str, Any], centres: list[F]) -> dict[str, Any]:
    mean = dict(old)
    # build_native consumes a NUMERICAL display only. It is not the exact wave.
    mean['centre_enclosure'] = {k+'_'+edge:dict(numerator=str(c.numerator),denominator=str(c.denominator)) for k,c in zip(('g0','gs'),centres,strict=True) for edge in ('lower','upper')}
    return mean


def run(path: Path, root_output: Path) -> dict[str, Any]:
    started = time.monotonic(); raw = path.read_bytes(); frozen = json.loads(raw)
    if hashlib.sha256(raw).hexdigest()!=FROZEN_RECEIPT_SHA or hashlib.sha256(CAPTURED[Path(full.__file__).name]).hexdigest()!=FROZEN_SOURCE_SHA:
        raise ArithmeticError('Frozen2832/ebd actual full45 predecessor required')
    oldmean = frozen['input_descriptor']['actual_root_mean_input_UNCHANGED']
    profiles = full.scalar.fast.primitive_profiles(oldmean); rows = radial_data(profiles)
    root = stationary_root(rows)
    oldpoint = frozen['input_descriptor']['actual_point']
    _,_,old_energy = radial_force_jacobian(exact(F(oldpoint['g0'])),exact(F(oldpoint['gs'])),rows,iv)
    new_energy = enclosure(F(root['nonlinear_energy_per_site_divided_Lambda']['lower']),F(root['nonlinear_energy_per_site_divided_Lambda']['upper']))
    root['certified_nonlinear_old_to_NEW_energy_gain_divided_Lambda'] = bounds(64*(old_energy-new_energy))
    root_descriptor = dict(predecessor=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()),law='NEW finiteN4 ALL65 stationary implicit root; frozen2832/ebd unchanged',parameters=dict(a='1/64',Lambda='2^8192',chi='2^-4096',G='8chi',V=64,f='1/8'),exact_profile_paths=[dict(role=p['role'],load=p['load'],scale=p['scale'],onsite=str(p['onsite']),paths=[{k:(str(v) if isinstance(v,F) else v) for k,v in item.items()} for item in p['paths']]) for p in profiles])
    root_capsule: dict[str, Any] = dict(input_descriptor=root_descriptor,input_sha256=hashlib.sha256(json.dumps(root_descriptor,sort_keys=True).encode()).hexdigest(),actual_stationary_root={k:v for k,v in root.items() if k not in ('centres','box')},source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in CAPTURED.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in CAPTURED.items()},phase='Stationary nonlinear root proved; full45 quantum consumer follows in same CLI.')
    if not all(root_capsule['source_unchanged_since_import'].values()):
        raise ArithmeticError('Source changed during exact stationary root proof')
    root_output.write_text(json.dumps(root_capsule,indent=2)+'\n')
    # Root precision is proved above. Eighty-digit interval matrix evaluation
    # below only observes/bounds the EXACT implicit matrix-functional law.
    iv.dps = 80; mp.mp.dps = 80
    g0 = enclosure(F(root['parameters']['g0']['lower']),F(root['parameters']['g0']['upper']))
    gs = enclosure(F(root['parameters']['gs']['lower']),F(root['parameters']['gs']['upper']))
    g = [g0,gs,gs,gs]
    mean = input_mean(oldmean,root['centres']); native = full.supplier.quantum.build_native(mean,4)
    geom = full.normal_geometry(native,g); Kprim,forceprim = full.primitive_matrix(geom,g,exact(F(1,64)))
    tensors = full.trace_tensors(geom['M'],full.supplier.primitive.SELECTED)
    algebra = [full.algebraic_species(p,geom,g,tensors) for p in profiles]
    literal = full.literal_consumers(native,profiles,geom,algebra)
    # Reference scalar conditional belongs to old point, retained ONLY as a
    # comparator. This adapter does not make it a new stationary scalar law.
    comparison = dict(actual_uniform_coupled_quantum_backreaction_and_realtime_transfer=dict(actual_coupled_ground_Phi_variance=frozen['actual_full45_quantum_collective_consumer']['scalar_conditional_0c3_variance']))
    quantum = full.coupled_covariance(geom,Kprim,forceprim,profiles,algebra,comparison)
    quantum['comparison_variance_scope'] = 'Old0c3 point scalar conditional ONLY; not same-new-root conditional.'
    quantum['implicit_root_equilibrium_normal_coordinate_exact'] = [0]*45
    quantum['tangent_force_scope'] = 'The inherited 80-digit force/linear-mean fields are numerical displays only; ALL45 exact implicit-root forces and equilibrium coordinates are exactly zero by the proved nonlinear stationarity. Their rounded displays are NEVER wave coefficients.'
    quantum['Hamiltonian'] = 'Exact implicit-root coupled PH Gaussian: H=G Pi^2/2+Lambda q^T Kprim q/2+sum Delta bdag b+sum q_a(A_aj bdag_j+A_aj* b_j); the linear scalar force is EXACT ZERO by actual all65 stationarity. Matrix-functional coefficients at the implicit root, not NumPy arrays, define this law.'
    quantum['physical_limits'] = 'NEW finiteN4 global-uniform selfconsistent nonlinear mean with full45 Gaussian tangent and all65 PH; this is NOT a full nonlinear quantum ground, nonuniform/full90/EM vacuum, original SOURCE or common-cone/scattering admission.'
    quantum['actual_omitted_dynamical_EM_cross_price'] = full.electromagnetic_cross_price(profiles,algebra,g)
    oldq = frozen['actual_full45_quantum_collective_consumer']
    idx = geom['index35']; partner = geom['labels'].index('Phi3 trivector(1, 2, 3)')
    # Execute the actual finite nonlinear potential/sea at old and new means.
    values = []
    for label,frame in (('old0c3',full.supplier.quantum.build_native(oldmean,4)['frame']),('NEW_stationary_root_display',native['frame'])):
        gram = np.einsum('mij,nji->mn',frame,frame).real/4
        primitive = 64/native['a']*float(np.sum(gram**2)); sea = []
        for p,alg in zip(profiles,algebra,strict=True):
            sign = -1 if p['scale']=='heavy' else 1
            xi = np.asarray([r['x'] for r in alg['points']])
            spectrum = np.linalg.eigvalsh(sign*np.einsum('km,mij->kij',xi,frame))
            sea.append(float(np.sum(spectrum[:,:2])))
        values.append(dict(label=label,leading_energy_divided_Lambda=primitive+sea[0],primitive_divided_Lambda=primitive,heavy_divided_Lambda=sea[0],active_energy=sea[1],auxiliary63_energy=63*sea[2]))
    desc = dict(plan=frozen['input_descriptor']['plan'],scientific_execution_revision=full.REVISION,observed_governing_plan_revision=full.OBSERVED_REVISION,input_receipt=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()),new_law='FiniteN4 uniform primitive quartic plus both filled physical negative heavy -Lambda h and all64 charged; exact positive implicit Gram minimum. Full45 static-PH Gaussian wave is matrix-functional at that root, no old mean retag.',parameters=dict(N=4,V=64,a='1/64',UEphys=1,u=64,Lambda='2^8192',chi='2^-4096',G='8chi',f='1/8',charged_loads=[1,63],neutral_heavy_load=1,alignment_gradient=0),exact_radial_species=[{k:(str(v) if isinstance(v,F) else v) for k,v in row.items()} for row in rows],original_micro_config_UNCHANGED=oldmean['microscopic_config'],original_SOURCE_receiver_probability_UNCHANGED=True)
    root_public = {k:v for k,v in root.items() if k not in ('centres','box')}
    lower = exact(F(quantum['certified_eigenvalue_lower_divided_Lambda']['lower']))
    gaussian_coordinate_norm_upper = iv.sqrt(45*iv.sqrt(exact(8*F(2)**-4096)/exact(F(2)**8192))/(2*iv.sqrt(lower)))
    residual_force = enclosure(F(0),F(root_public['ALL45_force_norm_divided_Lambda_upper']['upper']))*exact(F(2)**8192)
    root_public['finite_root_box_force_times_Gaussian_coordinate_norm_upper'] = bounds(residual_force*gaussian_coordinate_norm_upper)
    root_public['exact_implicit_root_force_times_coordinate_norm'] = 'EXACT ZERO; preceding value only prices any midpoint selected within root box.'
    wave = dict(exact_root=root_public['parameters'],
                coordinate_definition='Phi_x,m=Phi*_m+(1/8) sum_a q_a deltaPhi_a,m; EXACT algebraic Q45 normal basis defined by frozen2832 normal_geometry at implicit g*. Components reconstructed in exact Clifford grade basis, not rounded Qcart display.',
                exact_Hessian='Keff=Lambda Kprim−sum_species load scale B0, B0_ab=sum_k 2 Re Tr(Pocc F_a Pempty F_b)/delta_k; F_a=sign sum_m xi_m deltaPhi_a,m/8; heavy sign−, occupied upper h, scaleLambda; charged sign+, BOTH negative h, scales1.',
                wave_coefficient='A=(Keff/G)^(1/2), EXACT positive matrix functional calculus, where G=8*2^-4096. Psi(q)=(det A/pi^45)^(1/4) exp(−q^T A q/2). Mean q=0 at exact stationary implicit root. NumPy matrices/eigenvectors/covariance displays NEVER define wave coefficients.',
                Gaussian_covariance='Cov(q)=(1/2) A^{-1} for this exact static-PH wave; coupled PH Gaussian covariance is separately enclosed by full quantum consumer.',
                full_nonlinear_scope='This exact static-PH Gaussian is a selected uniform preparation, not a proved full nonlinear ground. The full PH dynamical Gaussian and its heavy quantum correction prices are separate. No nonuniform/full90/EM/original427 promotion.')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_NEW_finiteN4_stationary_mean=root_public,actual_exact_functional_Gaussian_wave_ABI=wave,actual_full45_geometry=dict(labels=geom['labels'],Qgrade_display=geom['Q'].tolist(),Qcart_display=geom['Qcart'].tolist(),physical_primitive_embedding_display=geom['E'].tolist(),errors=geom['errors'],Phi35_index=idx),actual_full45_ALLspecies_literal_consumers=literal,actual_full45_coupled_quantum_consumer=quantum,actual_nonlinear_old_and_new_mean_energy=values,actual_nonlinear_energy_gain_divided_Lambda=values[0]['leading_energy_divided_Lambda']-values[1]['leading_energy_divided_Lambda'],actual_old_new_comparison=dict(old_force_norm_divided_Lambda=oldq['actual_force_norm_divided_Lambda'],new_implicit_ALL45_force='EXACT ZERO',new_display_force_norm_divided_Lambda=quantum['actual_force_norm_divided_Lambda'],old_Phi35_cross_divided_Lambda=oldq['Phi35_not_eigenmode_offdiagonal_Hessian_norm'],new_Phi35_to_Phi3_T123_cross_divided_Lambda=quantum['actual_static_Hessian_divided_Lambda'][idx][partner],old_Phi35_variance=oldq['actual_full45_Phi35_variance'],new_Phi35_variance=quantum['actual_full45_Phi35_variance']),source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in CAPTURED.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in CAPTURED.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=mp.__version__,root_work_digits=ROOT_DIGITS,root_enclosure_digits=ENCLOSURE_DIGITS,matrix_observation_digits=iv.dps),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--root-output',type=Path,required=True)
    args = parser.parse_args(); result = run(args.predecessor,args.root_output)
    if not all(result['source_unchanged_since_import'].values()):
        raise ArithmeticError('Source changed during selfconsistent full45 execution')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    root = result['actual_NEW_finiteN4_stationary_mean']; q = result['actual_full45_coupled_quantum_consumer']
    print(json.dumps(dict(input_sha256=result['input_sha256'],g0=root['parameters']['g0']['lower_decimal'],gs=root['parameters']['gs']['lower_decimal'],root_radius=root['root_radius']['upper_decimal'],root_contraction=root['contraction']['upper_decimal'],force_enclosure_divided_Lambda=root['ALL45_force_norm_divided_Lambda_upper']['upper_decimal'],stiffness_lower=q['certified_eigenvalue_lower_divided_Lambda']['lower_decimal'],Phi35_variance=[q['actual_full45_Phi35_variance']['lower_decimal'],q['actual_full45_Phi35_variance']['upper_decimal']],nonlinear_gain_divided_Lambda=result['actual_nonlinear_energy_gain_divided_Lambda'],elapsed_seconds=result['elapsed_seconds']),indent=2))
