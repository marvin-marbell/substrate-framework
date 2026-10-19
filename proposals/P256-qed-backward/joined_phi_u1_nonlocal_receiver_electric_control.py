"""NEW Wilson-dressed nonlocal E_x: bare Gauss split versus source expectation."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
from itertools import combinations

import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_body_word as body
import joined_phi_u1_current_moments as current
import joined_phi_u1_current_controls as controls

ROOT = Path(__file__).resolve().parent
G = F(64)
DERIVATION = {
    'law': 'Same NEW fullN4 body HE=64sum129Pi², HC=64rho_allchargedLplusrho, realGaussian densityvariance2 and fixedmean active128+63aux128 Slaters, Ref4 before source; not originalSOURCE or ground.',
    'operator': 'For any actual oriented incidence k, q=Q^T k, delta=Dk=row-col. The hopping partial isometry O=exp(iqTheta)c_rowdag c_col has Pi O=O(Pi+q), rho O=O(rho+delta). Thus [HE+HC,O]=O*g[2qPi+2deltaLplusrho+q²+deltaLplusdelta]. Orthogonal Gauss reduction Q Q^T+D^T Lplus D=I gives q²+deltaLplusdelta=k². With this native D orientation the full electric field is QPi+D^T Lplusrho.',
    'receiver': 'For col0, E_x=exp(iqTheta)*.5sum4 c_xalpha^dag R0alpha, where the Refhole remains at0 and scalar Wilson path transforms both endpoints. Source Ref4->3 and E_xdagE_x=M_x=(4-n_active,x)/4. Same commutator identity uses delta=delta_x-delta_0, not delta_x alone. Ordered path/AP seam convention inherited from microscopic paths; constant seam signs do not change electric prices.',
    'unreduced_lift': 'The reduced CAR variables are Coulomb-gauge fields. In unreduced A coordinates use lambdaC=Lplus D A with any FAR anchor; full Wilson phase is qTheta+delta lambdaC=inc A. Under A->A-D^Talpha, lambdaC->lambdaC-alpha+constant and bare c_xdag R0->exp(i delta alpha)c_xdag R0. The full receiver is invariant because delta sums tozero. The reduced exp(iqTheta) form is valid after the long dressing is included in the CAR coordinate; using it with UNREDUCED bare CAR would omit compensation.',
    'state': 'In fixedmean translation invariant background, meanrho=0 and mass=<M_x>=1/2. Initial Gaussian <Pi>=0. Density covariance K_xv=delta_xv TrPxx-Tr(P_xv P_vx) includes all4 active components. Aux63 mean terms are included in meanrho; their covariance with M_x vanishes at this initial product state only. Therefore <M_x rho_v>=mass*meanrho_v-K_xv/4. Raw source energy-commutator increment is g*mass*q² + g*[mass*deltaLdelta+2deltaL<M_xrho>]. This is NOT the bare selfcoefficient64/128, NOT a dressed energy eigenvalue.',
    'limits': 'Derived initial CAR/Gauss identities versus finite floating controls. No future evolved-Slater Wick, freechargedpole, propagation/finite-time kernel, compactPhi endpoint transfer, ground/commoncone or originalSOURCE admission. Nonlocal receiver is gauge complete within the NEW meanPhi Coulomb-gauge body; compactPhi Wilson/Noether endpoints require their own bound.',
}


def dense_reference_control() -> dict[str, Any]:
    """Nonlocal CAR/Ref boundary oracle, separate from fullN4 geometry."""
    rng = np.random.default_rng(22264012)
    w, _ = np.linalg.qr(rng.normal(size=(8,4))+1j*rng.normal(size=(8,4)))
    p = w@w.conj().T
    cs = controls.creators(8)
    psi = np.zeros(256,complex)
    for ids in combinations(range(8),4):
        psi[sum(1 << i for i in ids)] = np.linalg.det(w[list(ids)])
    eye = np.eye(256)
    n = [sum((c@c.conj().T for c in part),np.zeros_like(eye)) for part in (cs[:4],cs[4:])]
    rho = [n[0]-2*eye,n[1]-2*eye]
    bg = 16*(rho[0]-rho[1])@(rho[0]-rho[1])
    src = 16*(rho[0]-eye-rho[1])@(rho[0]-eye-rho[1])
    vectors = [.5*c@psi for c in cs[4:]]
    mass = float(sum(np.vdot(v,v).real for v in vectors))
    mx = (4*eye-n[1])/4
    direct = sum(np.vdot(v,src@v) for v in vectors)-np.vdot(psi,mx@bg@psi)
    pb = p.reshape(2,4,2,4).transpose(0,2,1,3)
    occupation = np.trace(pb[np.arange(2),np.arange(2)],axis1=1,axis2=2).real
    cov = np.diag(occupation)-np.einsum('xyij,yxji->xy',pb,pb).real
    delta = np.array([-1.,1.]);lp = np.array([[1.,-1.],[-1.,1.]])/4
    weighted = mass*(occupation-2)-cov[1]/4
    formula = 64*(mass*(delta@lp@delta)+2*(delta@lp@weighted))
    defect = float(abs(direct-formula))
    body.require(defect,1e-8)
    charge = rho[0]+rho[1]
    accepted = float(sum(np.linalg.norm((charge-eye)@v)**2 for v in vectors))
    omitted = float(sum(np.linalg.norm(charge@v)**2 for v in vectors))
    body.require(accepted)
    body.require(abs(omitted-mass))
    if omitted < .01:
        raise ArithmeticError('Omitted static Refhole charge control is blind')
    return dict(dense_nonlocal_CAR_HC_commutator_defect=defect,
                raw_source_norm_squared=mass,actual_total_charge_squared=accepted,
                wrong_omitted_Ref_total_charge_squared=omitted,
                dense_HC_commutator_increment_real=float(direct.real),
                covariance_formula_increment=float(formula),
                boundary='Four orthogonal Ref holes atsite0; creation atsite1. This finite8orbital oracle is not N4 mode geometry or a ground/pole claim.')


def run() -> dict[str, Any]:
    start = time.monotonic()
    own = Path(__file__).read_bytes()
    oracle = dense_reference_control()
    data, species, inputs, _ = body.full_inputs()
    geo = current.geometry(data, species[0])
    native = data['native']
    qbasis, lp, d = native['Q'], native['Lplus'], native['D']
    split = qbasis@qbasis.T+d.T@lp@d
    split_defect = float(np.max(abs(split-np.eye(192))))
    body.require(split_defect)
    rng = np.random.default_rng(222129192)
    potential = rng.normal(size=192)
    alpha = rng.normal(size=64)
    transformed = potential-d.T@alpha
    lam, lamp = lp@d@potential,lp@d@transformed
    lam -= lam[63];lamp -= lamp[63]  # FAR anchor cancels at neutral endpoints.
    lifted = geo['q']@(qbasis.T@potential)+geo['delta']@lam
    liftedp = geo['q']@(qbasis.T@transformed)+geo['delta']@lamp
    lift_defect = float(np.max(abs(lifted-geo['k']@potential)))
    gauge_defect = float(np.max(abs(np.exp(1j*(liftedp-lifted+geo['delta']@alpha))-1)))
    missing_long = float(np.max(abs(np.exp(1j*(geo['delta']@alpha))-1)))
    body.require(lift_defect);body.require(gauge_defect)
    if missing_long < .1:
        raise ArithmeticError('Missing long dressing with unreduced CAR is blind')
    p, aux = species[0].p, species[1].p
    pb = p.reshape(64,4,64,4).transpose(0,2,1,3)
    pa = aux.reshape(64,4,64,4).transpose(0,2,1,3)
    na = np.trace(pb[np.arange(64),np.arange(64)],axis1=1,axis2=2).real
    nb = np.trace(pa[np.arange(64),np.arange(64)],axis1=1,axis2=2).real
    density = na+63*nb-128
    covariance = np.diag(na)-np.einsum('xyij,yxji->xy',pb,pb).real
    body.require(float(np.max(abs(density))))
    body.require(float(np.max(abs(covariance.sum(axis=1)))))
    rows = []
    indices = np.flatnonzero(geo['y']==0)
    for r in indices:
        x = int(geo['x'][r]); k, q, delta = geo['k'][r],geo['q'][r],geo['delta'][r]
        mass = float((4-na[x])/4)
        body.require(abs(mass-.5))
        length = int(np.dot(k,k))
        q2 = float(q@q); c2 = float(delta@lp@delta)
        expected_q = F(43,64) if length==1 else F(129,80)
        expected_c = F(length)-expected_q
        body.require(abs(q2-float(expected_q)))
        body.require(abs(c2-float(expected_c)))
        weighted_density = mass*density-covariance[x]/4
        transverse = float(G)*mass*q2
        coulomb_self = float(G)*mass*c2
        coulomb_cross = 2*float(G)*float(delta@lp@weighted_density)
        coulomb = coulomb_self+coulomb_cross
        naive = float(G)*length
        actual_normalized_increment = (transverse+coulomb)/mass
        correction = actual_normalized_increment-naive
        if abs(correction)<1e-5:
            raise ArithmeticError('Prepared-source cross-density discriminator is blind')
        rows.append(dict(microscopic_oriented_path=int(r),row=x,col=0,ordered_link_length=length,
                         raw_source_norm_squared=mass,transverse_q_squared_exact=str(expected_q),
                         Coulomb_delta_Lplus_delta_exact=str(expected_c),
                         bare_transverse_self_coefficient_exact=str(G*expected_q),
                         bare_Coulomb_self_coefficient_exact=str(G*expected_c),
                         bare_total_self_coefficient_exact=str(G*length),
                         raw_transverse_source_increment=transverse,raw_Coulomb_self_increment=coulomb_self,
                         raw_Coulomb_density_cross_increment=coulomb_cross,raw_Coulomb_source_increment=coulomb,
                         raw_combined_source_increment=transverse+coulomb,
                         source_normalized_energy_commutator_increment=actual_normalized_increment,
                         wrong_naive_bare_as_energy_commutator_increment_defect=correction,
                         wrong_omit_Pi_bare_self_defect=float(G)*q2,
                         wrong_omit_Coulomb_bare_self_defect=float(G)*c2,
                         dropped_Refhole_charge_sum=float(np.eye(64)[x].sum()),
                         actual_receiver_total_charge_sum=float(delta.sum())))
    if len(rows)!=12 or {row['ordered_link_length'] for row in rows}!={1,2}:
        raise ArithmeticError('All actual signed one/two paths to the fixedRef0 required')
    if any(row['actual_receiver_total_charge_sum']!=0 or row['dropped_Refhole_charge_sum']!=1 for row in rows):
        raise ArithmeticError('Gauss complete-state accept boundary failed')
    hashes = {name:hashlib.sha256(raw).hexdigest() for name,raw in {**body.CAPTURED,**inputs}.items()}
    hashes[Path(current.__file__).name]=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest()
    hashes[Path(controls.__file__).name]=hashlib.sha256(Path(controls.__file__).read_bytes()).hexdigest()
    if own!=Path(__file__).read_bytes():
        raise ArithmeticError('Own source changed')
    return dict(derivation=DERIVATION,receiver_rows=rows,all_orthogonal_Gauss_split_defect=split_defect,
                dense_nonlocal_reference_oracle=oracle,
                full_unreduced_Wilson_lift_defect=lift_defect,
                arbitrary_local_U1_receiver_covariance_defect=gauge_defect,
                wrong_unreduced_CAR_without_long_dressing_defect=missing_long,
                full_background_charge_mean_defect=float(np.max(abs(density))),
                fixed_N_active_density_covariance_sum_defect=float(np.max(abs(covariance.sum(axis=1)))),
                background_active_density_covariance_min_eigenvalue=float(np.linalg.eigvalsh(covariance)[0]),
                input_hashes=hashes,own_source_sha256=hashlib.sha256(own).hexdigest(),own_source_utf8=own.decode(),
                conditional=['Selected mean/root/Clifford inputs; finite floating matrix controls, not interval certification'],
                governing_plan='issue-222-induced-photon-delta13-continuation-20261008',
                captured_parent_revision='d626209c92ceb6505cdbc3799deaee4be54297b9f52504d9d9b30ed3088676d5',
                template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
                runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),
                elapsed_seconds=time.monotonic()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        result=run()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(receipt_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                         rows=result['receiver_rows'],split_defect=result['all_orthogonal_Gauss_split_defect'],
                         elapsed_seconds=result['elapsed_seconds']),indent=2))
