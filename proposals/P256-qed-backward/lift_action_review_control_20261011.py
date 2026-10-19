"""Independent bounded controls for the pinned compact-lift action review.

Imports pure jet/profile helpers only. Never runs a historical main/solver.
Native 70/924 controls do not evaluate the N4 field wave or full H variance.
"""
from __future__ import annotations
import ast
from fractions import Fraction as Q
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import cast
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
import physical_ground_lift_source_20261011 as lift

ROOT = Path(__file__).resolve().parent
PINS = {
    'physical_ground_lift_source_20261011.py': '31b1c270bf341f75f2364580cd6f646f1d66fda54d93e9b3b227b6ab3985763e',
    'physical_ground_lift_actual3_20261011_proof.json': 'b5db95dd2c56822c32d8975232a7ca60d946a5c62a482164aeb4afb51fa07b17',
    'physical_ground_lift_domain_supplement_20261011.json': 'ad4b7399dfdf22c09b1d6f5abf6163c29bdfb479c16c4d6724f8f9e59acc0d58',
    'physical_ground_lift_actual3_20261011_natural_exit.json': '2bd90fe94474dbfb078e1e71a2bec6af51b7d18b7bc7f14751eed3accb636fe9',
}

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def basis() -> list[np.ndarray]:
    out = []
    for i in range(4):
        for j in range(i+1, 4):
            a = np.zeros((4,4), complex); a[i,j] = a[j,i] = .5; out.append(a)
            a = np.zeros((4,4), complex); a[i,j] = -.5j; a[j,i] = .5j; out.append(a)
    for k in range(1,4):
        out.append(np.diag([1.]*k+[-float(k)]+[0.]*(3-k))/math.sqrt(2*k*(k+1)))
    return out

def car_matrix(h: np.ndarray, particles: int) -> tuple[csr_matrix, list[int]]:
    masks = [sum(1 << j for j in row) for row in itertools.combinations(range(len(h)),particles)]
    index = {m:i for i,m in enumerate(masks)}
    rr, cc, vv = [], [], []
    for col,m in enumerate(masks):
        for p,q in zip(*np.nonzero(h)):
            result = lift.primitive.car(m,int(p),int(q))
            if result is not None:
                target,sign = result; rr.append(index[target]);cc.append(col);vv.append(sign*h[p,q])
    return csr_matrix((vv,(rr,cc)),shape=(len(masks),len(masks))), masks

def ceil_log2(n: int) -> int:
    return (n-1).bit_length()

def main() -> None:
    inherited = json.loads((ROOT/'physical_ground_lift_actual3_20261011_natural_exit.json').read_text())
    pins = {**inherited['source_input_sha256'], **PINS}
    hashes = {name:digest(ROOT/name) for name in pins}
    if hashes != pins: raise ArithmeticError('Pinned target/supplier bytes changed')
    proof = json.loads((ROOT/'physical_ground_lift_actual3_20261011_proof.json').read_text())
    centre = np.array([[[complex(*z) for z in row] for row in m] for m in proof['configuration_binding']['Phi_binary_components']])
    b = basis(); R = float(lift.R)
    # A bounded one-species transport fixture, not a replacement for the g fibre.
    trial = object.__new__(lift.LiftedGround)
    trial.centre = centre
    trial.wstar,_ = lift.frame_jet([lift.Jet(x) for x in centre])
    gamma = centre[1]/centre[1,0,3].real
    trial.spin = np.column_stack([np.eye(4)[:,0],gamma@np.eye(4)[:,0],np.eye(4)[:,1],gamma@np.eye(4)[:,1]])
    point = np.broadcast_to(centre,(2,4,4,4)).copy()
    point[0,0] += R/5*b[1]
    direction = np.zeros_like(point); direction[0,0] = b[1]
    fj = [[lift.Jet(point[x,m],direction[x,m]) for m in range(4)] for x in range(2)]
    f = lift.source_shape_jet(fj,lift.Jet(np.eye(4)),trial,R)
    w,_ = lift.frame_jet(fj[0]); left,right = trial.transport_jet(w,lift.Jet(np.eye(4)),lift.Jet(0.))
    U = lift.exterior_jet(lift.block_jet(left,right),4)
    joined = U*f
    independent_dd = f.dd*U.v+2*f.d*U.d+f.v*U.dd
    product_error = float(np.linalg.norm(joined.dd-independent_dd)/max(1.,np.linalg.norm(joined.dd)))
    cross = float(np.linalg.norm(2*f.d*U.d))
    missing_frame = float(np.linalg.norm(joined.dd-f.dd*U.v))
    if product_error > 1e-14 or min(cross,missing_frame) <= 1: raise ArithmeticError(dict(product_error=product_error,cross=cross,missing_frame=missing_frame))
    # Nonmatching point in the Haar direction makes both scalar and fibre vary.
    point[1,2] += R/6*1j*(b[6]@centre[2]-centre[2]@b[6])
    uj = lift.Jet(np.eye(4),1j*b[6],-b[6]@b[6])
    hf = lift.source_shape_jet([[lift.Jet(x) for x in side] for side in point],uj,trial,R)
    hw,_ = lift.frame_jet([lift.Jet(x) for x in point[0]])
    hl,hr = trial.transport_jet(hw,uj,lift.Jet(0.))
    hu = lift.exterior_jet(lift.block_jet(hl,hr),4)
    hj = hu*hf
    haar_cross = float(np.linalg.norm(2*hf.d*hu.d))
    haar_error = float(np.linalg.norm(hj.dd-(hf.dd*hu.v+2*hf.d*hu.d+hf.v*hu.dd))/max(1.,np.linalg.norm(hj.dd)))
    if haar_cross <= 1 or haar_error > 1e-14: raise ArithmeticError(dict(haar_cross=haar_cross,haar_error=haar_error))
    # Exact one-dimensional support crossing: root frame is fixed here.
    boundaries = []
    for t in [0., .5, 1.-2**-10, 1., 1.+2**-10, 1.5]:
        p = np.broadcast_to(centre,(2,4,4,4)).copy(); dp = np.zeros_like(p)
        dp[1,3] = math.sqrt(2)*R*b[0];p += t*dp
        z = lift.source_shape_jet([[lift.Jet(p[x,m],dp[x,m]) for m in range(4)] for x in range(2)],lift.Jet(np.eye(4)),trial,R)
        inside = t < 1
        a = max(1-t*t,0.)
        expected = [a**4, -8*t*a**3 if inside else 0., (-8*a**3+48*t*t*a*a) if inside else 0.]
        observed = [float(z.v.real),float(z.d.real),float(z.dd.real)]
        err = max(abs(v-e) for v,e in zip(observed,expected))
        if err > 1e-7 or (t>=1 and max(abs(v) for v in observed)>1e-20): raise ArithmeticError('Boundary control failed')
        boundaries.append(dict(t=t,jet=observed,analytic_jet=expected,error=err))
    # Electric full canonical matrix, background and beta cross paid once.
    occup = list(itertools.combinations(range(8),4))
    rho = np.array([sum(i<4 for i in row)-2 for row in occup])
    beta = Q(1,1024); r = rho.astype(float)
    electric = 64*(r+float(beta))**2
    electric_join = 64*r*r+128*float(beta)*r+64*float(beta)**2
    electric_error = float(np.linalg.norm(electric-electric_join))
    wrong_beta_gap = float(np.linalg.norm(electric-(64*r*r+64*float(beta)**2)))
    if electric_error != 0 or wrong_beta_gap <= 0: raise ArithmeticError('Electric join failed')
    # Independent literal profiles and intermediate ordered Wilson matrices.
    cfg = json.loads((ROOT/lift.primitive.INPUTS[1]).read_text())['input_descriptor']['original_micro_config_UNCHANGED']
    frame,scale = lift.primitive.frozen.exact_frame((ROOT/lift.primitive.INPUTS[0]).read_bytes())
    profiles = lift.primitive.frozen.profiles(cfg,frame,scale)
    neutral_band = lift.primitive.frozen.RotationBand(a=cfg['a'],m=cfg['heavy_m'],r=cfg['heavy_r'],v=cfg['heavy_v'],improved=False)
    neutral_on,neutral_paths = lift.primitive.frozen.scalar_paths(neutral_band)
    profile_caps = []
    for profile in profiles:
        onsite = profile['onsite']/scale
        sums = [sum(abs(re)+abs(im) for re,im in path['coefficients']) for path in profile['paths']]
        # op norm <= entrywise l1; |Phi_mu,ij|<4, Wilsons unitary.
        per_site = 16*4*abs(onsite)+sum(2*16*4*s for s in sums)
        profile_caps.append(dict(role=profile['role'],count=profile['count'],onsite=str(onsite),paths=len(sums),per_site_entrywise_CAR_upper=str(per_site)))
    charged_cap = sum(Q(p['per_site_entrywise_CAR_upper'])*p['count'] for p in profile_caps)
    neutral_cap = 16*4*Q(neutral_on)+sum(2*16*4*sum(Q(float(abs(z.real)))+Q(float(abs(z.imag))) for z in cs) for _,cs in neutral_paths)
    if charged_cap >= 2**30 or neutral_cap >= 2**30: raise ArithmeticError('Matter arithmetic failed')
    u1,u2 = expm(.37j*b[0]),expm(.29j*b[4])
    ordered_gap = float(np.linalg.norm(u1@u2-u2@u1))
    if ordered_gap <= 1e-4: raise ArithmeticError('Ordered link control failed')
    hroles = []
    for p in profiles:
        path = p['paths'][0]
        coeff = np.array([complex(float(re),float(im)) for re,im in path['coefficients']])
        tmat = cast(np.ndarray, sum(coeff[m]*(centre[m]@u1+u1@centre[m])/2 for m in range(4)))
        one = np.block([[float(p['onsite']/scale)*centre[0],tmat],[tmat.conj().T,float(p['onsite']/scale)*centre[0]]])
        hroles.append(car_matrix(one,4)[0])
    role_gap = float(np.linalg.norm((hroles[0]-hroles[1]).toarray()))
    if role_gap <= 1: raise ArithmeticError('Distinct role control failed')
    # Three-site canonical six-particle control reaches a genuine exterior.
    h3 = np.zeros((12,12),complex); h3[4:8,8:12] = centre[1]@u2;h3[8:12,4:8] = h3[4:8,8:12].conj().T
    H3,masks3 = car_matrix(h3,6)
    seedmask = sum(1<<j for j in [0,1,4,5,8,9])
    seed = np.zeros(len(masks3),complex);seed[masks3.index(seedmask)] = 1
    action = H3@seed
    exterior_indices = np.array([(m&255).bit_count()!=4 for m in masks3])
    exterior_norm = float(np.linalg.norm(action[exterior_indices]))
    if exterior_norm <= 0: raise ArithmeticError('Intercell exterior failed')
    # Exact rational arithmetic for geometry, normalizer and complete N4 cap.
    sf = Q(float(-centre[0,0,0].real));rf = Q(float(centre[1,0,3].real))
    gap = min(2*(sf-4*rf*rf),8*rf*rf)
    assert gap-32*lift.R > Q(1,16) and rf > Q(1,8)
    pi_upper = Q(22,7)
    assert 128*pi_upper*pi_upper < 36**2 and 72**15 < 2**93
    box_body_ratio = Q(135,2**11) # 135 * 2^40 * 2^-75 / R.
    assert box_body_ratio < Q(1,8) and box_body_ratio**2 < Q(1,4)
    v,e,m = 64,192,32
    terms = {
        'Phi_complete_scalar_frame_Berry': -4096+ceil_log2(4*60*v)+16384*v+240,
        'Haar_complete_transport_scalar_cross': -4096+ceil_log2(15*e)+16384*v+240,
        'fullE64_longitudinal_and_transverse': ceil_log2(64*e)+16384*v+240,
        'neutral_all_canonical2V_onsite_paths': 8192+ceil_log2(v)+30,
        'charged_actual1_aux63_all_onsite_one_two_paths': ceil_log2(v)+30,
        'Jordan_full_support':8192+ceil_log2(v)+30,
        'alignment_full_support':10240+ceil_log2(v)+30,
        'colour_plaquette_full_Haar_support':10240+ceil_log2(v)+30,
    }
    assert terms == proof['N4_complete_action_upper']['term_log2_upper']
    bound = max(terms.values())+ceil_log2(len(terms))
    assert bound == 1048833
    eta = Q(274877906944,2204801412260314559041)
    assert 0 < eta < 1 and (1-eta)**2 <= 1-eta
    own_ast = ast.parse(Path(__file__).read_bytes())
    functions = [n.name for n in ast.walk(own_ast) if isinstance(n,ast.FunctionDef)]
    result = dict(kind='INDEPENDENT_BOUNDED_NATIVE_ACTION_AND_EXACT_RATIONAL_CONTROL_NOT_FULLFIELD_NUMERICS',
        source_sha256=digest(Path(__file__)),input_sha256=hashes,runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),
        controls=dict(native70_product_error=product_error,native70_Phi_cross_norm=cross,native70_missing_frame_gap=missing_frame,
            native70_Haar_product_error=haar_error,native70_Haar_cross_norm=haar_cross,boundary=boundaries,
            native70_electric_join_error=electric_error,native70_wrong_drop_beta_cross_gap=wrong_beta_gap,
            ordered_two_Wilson_wrong_order_gap=ordered_gap,distinct_active_aux_native70_action_gap=role_gap,
            native924_intercell_exterior_action_norm=exterior_norm,neutral_path_count=len(neutral_paths),profiles=profile_caps),
        rational=dict(exact_frame_gap=str(gap),charged_per_site_CAR_upper=str(charged_cap),neutral_per_site_CAR_upper=str(neutral_cap),
            diameter_square_upper=str(128*pi_upper*pi_upper),total_volume_cube_bound='72^15 < 2^93',
            normalizer_box_body_ratio=str(box_body_ratio),Z_lower_exponent=-10094*m,
            sqrt_support_over_Z_exponent=str(Q(360*v+10094*m,2)),term_log2_upper=terms,norm_upper_exponent=bound,
            infidelity=str(eta),projected_x_normalizer_lower=str(1-eta),graph_vector_error='sqrt(2 eta), once; M=32'),
        AST_positive_functions=functions,scope='Pure helpers, native70 one-species jets, native924 three-site CAR, exact rational uppers. No g eigensolver, historical main, fullfield normalization, H residual or variance.')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
