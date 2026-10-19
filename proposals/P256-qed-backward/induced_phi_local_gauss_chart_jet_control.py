"""Body-framed physical local-Gauss chart jets, not a tube/source certificate.

Run with the project Python and --output. Exact envelope arithmetic proves the
stated conditional inequalities; finite matrix controls discriminate omissions.
No root solver, precision campaign or scientific supplier is modified or run.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from itertools import product
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from scipy.linalg import expm, expm_frechet
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_gauss_quantum as gauss
import induced_phi_nonuniform_fermionic_source_unitary as source

ROOT = Path(__file__).resolve().parent
PROJECTOR_SHA = 'e18a9ce4ece8dea57e8f739eb22ea4aebd21c0b4c033eba665a48b9cd108a376'
ABI_FILE = ROOT / 'induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json'
PROJECTOR_FILE = ROOT / 'induced_phi_local_gauss_orbit_projector_control_receipt.json'

DERIVATION = {
    'coordinates_and_metric': (
        'Use T_a=M_a/sqrt(8), Tr(T_a T_b)=delta_ab/2 and cart(X)_a=2Tr(T_a X). '
        'The physical tangent metric divided by the common 1/chi is diag(I_Phi/8,(10/3)I_link), '
        'not the bare Cartesian metric. This is the metric whose reduced inverse is '
        'G/chi=diag(8I45,.3I45+d d* tensor H_orbit^-1); H_orbit=O^T O/8. '
        'The site Fourier transform is unitary, with constant slow injection Q0 s/sqrt(V)=Q0 s/8. '
        'Fast Phi has zero site average and norm ||f|| in its 2835-dimensional canonical real subspace. '
        'All 2880 link C coordinates are retained. Angles have 960 coordinates. '
        'The actual source restricts ||(s,f,C)||<=R<=2^-20, NOT its full Haar angle support. '
        'For local coordinate controls choose U_x=Ubase_x exp(i T theta_x), ||theta||<=R, '
        'with arbitrary Ubase at every site; the estimates are uniform over all bases. '
        'Input tangent norm is unweighted Euclidean direct sum; output is physical-metric '
        'whitened. The intrinsic body Noether matrix below is independent of Ubase/angles.'),
    'admitted_hypotheses': (
        'V=64, g_mu>0, h=min(g_mu)/2>1/64, w=sum(g_mu)/2<=1. '
        'The exact positive root and its global consumer input certificate are ADMITTED, not computed here. '
        'Use the exact algebraic Q0 recipe from full.normal_geometry, and the actual polar adapter '
        'Q(s)=(I-P(s))Q0[Q0^T(I-P(s))Q0]^-1/2. The bound from the SHA-bound projector receipt '
        'is ||Q||<9/8, Q1<512,Q2<2^20,Q3<2^33 on the canonical slow ball. '
        'The exact recipe, not rounded cached Q, supplies the algebraic mean hypothesis. '
        'No occupied-fermion Kato256 frame enters these estimates.'),
    'mean_normal_recipe': (
        'Clifford monomials M have Tr(M_a M_b)=4delta_ab. Grade1 normal columns are four diagonal '
        'unit vectors and six symmetric weighted pairs (sqrt(g_j),sqrt(g_i))/sqrt(g_i+g_j). '
        'Their vertical partner is (sqrt(g_i),-sqrt(g_j)) up to overall sign: dot product zero. '
        'Grade1 thus has normal dimension10, vertical dimension6. In grade2, four grade1-generator '
        'orbit vectors v have mutually disjoint three-slot supports. The supplier n=(v1,-v0,0)/sqrt(v0^2+v1^2) '
        'is perpendicular to v and unit; z=v cross n/||v|| is perpendicular to both and unit. '
        'These eight columns plus twelve untouched unit slots give normal20, vertical4. '
        'Grade3 has one four-slot orbit v. With w=v-||v||e0, H=I-2ww^T/(w^Tw) is exactly orthogonal '
        'and Hv=||v||e0. Positivity of the other g ensures w!=0. Columns H e1,H e2,H e3 are '
        'normal; twelve untouched unit slots give normal15, vertical1. Grade4 has four vertical '
        'slots and no normal columns. Disjoint grade/support accounting gives45 orthonormal normal '
        'columns and15 independent orbit columns, hence exact Q0^TQ0=I and O0^TQ0=0. '
        'The Cart change R_ab=2Tr((M_a/sqrt8) Tcart_b) is orthogonal by completeness of two '
        'orthonormal Hermitian traceless bases; Qcart=(I4 tensor R^T)Q preserves both identities. '
        'The matrix-unit Cart basis includes radicals, so R is NOT claimed to be a signed permutation. '
        'Normal_geometry centre(interval) and NumPy sqrt/eigh/SVD only OBSERVE this exact recipe; '
        'their finite rounded realizations are not interval-certified by the observation.'),
    'field_chain': (
        'F_x(s,f)=Phi0+sum_a Q0_a T s_a/8+sum_a[Q(s)f_x]_a T, and '
        'Phi_x=U_x F_x U_x^dag. Here F below denotes its site Cartesian coefficients. '
        'The slow injection has total operator norm1. The block-diagonal fast injection has norm '
        '||Q||<9/8. Holder on site blocks bounds every mixed multilinear term by the corresponding '
        'global Euclidean norms, with NO factor sqrt(V). Thus F1<=1+9/8+512R<4; '
        'F2<=2*512+2^20 R<=1025; F3<=3*2^20+2^33 R<=3153920. '
        'The multiplicities2 and3 are exactly the choices of the differentiated fast argument. '
        'These are derivatives in the actual /8 uniform slow coordinate, not a site-uniform /1 proxy. '
        'At each site ||F0_x||=sqrt(8 sum g)<=4 and ||F_x-F0_x||<=F1 R<4R.'),
    'Maurer_and_graph': (
        'Let A(c) be the real Cartesian adjoint generator of i T c. Hilbert-Schmidt isometry gives '
        '||A(c)||<=sqrt2||c||<=2||c||. B(c)=Ad_exp(i T c)=exp(A(c)); '
        'J(c)=integral_0^1 exp(t A(c))dt is the RIGHT Maurer differential of exp(i T c). '
        'For real c, exponentials are orthogonal. Duhamel with ordered insertion simplexes gives '
        '||D^k B||<=2^k and ||D^k J||<=2^k/(k+1) for k=0,1,2. '
        'Use J0<=1,J1<=1,J2<=2 and B1<=2,B2<=4. These bounds include noncommuting angle/link '
        'derivatives, not only rays with a single commuting generator. At the mean, the periodic '
        'N4 graph difference d has Fourier eigenvalues sum_i4sin^2(q_i/2)<=12, so ||d||<=sqrt12<4. '
        'Away from the mean, source and target lifts each have norm sqrt3<2 because each site has '
        'three outgoing and three incoming links. Never replace these lifts by an unweighted norm1.'),
    'actual_body_D': (
        'Write a_x=J(-theta_x)delta theta_x (LEFT/body Maurer of U_x) and E_e=exp(i T C_e). Conjugate the field differential '
        'by U_x^dag and the right link Maurer differential dot L_e L_e^dag by U_x^dag. Then '
        'D_Phi delta=(dF delta+[i T a,F])/sqrt8 and '
        'D_link delta=sqrt(10/3)[J(C_e)delta C_e+a_x-Ad_Ee a_y]. '
        'This is for L_e=U_x E_e U_y^dag, with ALL local U retained. Phi commutator norm as an '
        'operator on angular site blocks is <=2 max_x||F_x||<9. The body framing removes only '
        'output conjugations, not these angle columns, their J derivatives, or Ad_E.'),
    'D_aggregation': (
        'Differentiate this matrix in arbitrary joint unit directions. Phi first derivative is '
        '<=(F2+2F1+9J1)/sqrt8 < (1025+8+9)/2=521. '
        'Phi second derivative is <=(F3+2F2+4F1 J1+9J2)/sqrt8 '
        '<(3153920+2050+16+18)/2=1578002. The two mixed Fprime/Jprime terms give4F1. '
        'Link first derivative <=sqrt(10/3)[J1+2sqrt3 J1+sqrt3 B1] '
        '<2[1+4+4]=18. Link second derivative <=sqrt(10/3)[J2+2sqrt3 J2 '
        '+sqrt3(B2+2B1 J1)] <2[2+8+16]=52. The latter formula accounts for both '
        'angle endpoints, mixed C/theta_y, and pure C derivatives of Ad_E. '
        'Stacking output rows is bounded by sum of the two row-operator norms. Therefore '
        '||Dprime||<539<4096 and ||Dsecond||<1578054<2^24. '
        'The supplied named constants survive with ample slack; no generic conditional escape '
        'hides an unpriced angle/link term. These estimates are dimension-free operator/multilinear '
        'bounds, not maximum-entry bounds or spectral claims from the floating controls.'),
    'global_Haar_alignment': (
        'In the actual product-Haar source, use left-invariant angular velocities a_x directly, '
        'rather than exponential coordinate velocities: set J(-theta)=I in the body D formula. '
        'This retains the full commutator column and source-minus-Ad_E target link columns. '
        'The resulting coefficient matrix depends only on s,f,C, not on any local U_x; its '
        'angular Noether derivatives vanish. Its compact-domain first derivative is bounded '
        'by (F2+2F1)/sqrt8+sqrt(10/3)(J1+sqrt3 B1)<1053/2=526.5; its second derivative '
        'is bounded by (F3+2F2)/sqrt8+sqrt(10/3)(J2+sqrt3 B2)<1578005. '
        'Both are below the larger local-coordinate 539/1578054 envelopes. Thus the named '
        '4096/2^24 D-jet prices also cover the intrinsic body coefficients at every Haar '
        'angle, without restricting the physical source to theta near identity. This does '
        'NOT by itself prove determinant/Haar potential, anholonomic kinetic action or '
        'global tube/injectivity; those separate obligations remain open.'),
    'scope': (
        'This ingredient proves conditional local BODY-FRAMED D jets in the explicitly specified '
        'physical metric and coordinate chart. It does not prove D inverse, reach, injectivity, '
        'global one-Z4 chart descent, determinant/Haar potential differentiation, Gaussian/Riccati '
        'certification, Hermite cancellation, complete action error or physical particle/photon '
        'propagation. Laboratory-frame D derivatives require additional output-frame derivatives; '
        'they are not silently identified with these body derivatives. Root existence and global '
        'input certification remain open/admitted in this artifact; no issue/campaign closure.')
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def exact_envelopes() -> dict[str, Any]:
    R = F(1, 1 << 20)
    F1 = 1+F(9,8)+512*R
    F2 = 2*512+(1 << 20)*R
    F3 = 3*(1 << 20)+(1 << 33)*R
    assert F1 < 4 and F2 <= 1025 and F3 <= 3153920
    assert 2*(4+4*R) < 9
    p1 = (F2+8+9)/2
    p2 = (F3+2*F2+16+18)/2
    l1, l2 = F(18), F(52)
    assert p1+l1 < 4096 and p2+l2 < 1 << 24
    return {k:str(v) for k,v in dict(R=R,F1=F1,F2=F2,F3=F3,
        body_Phi_D1_upper=p1,body_Phi_D2_upper=p2,
        body_link_D1_upper=l1,body_link_D2_upper=l2,
        body_D1_upper=p1+l1,body_D2_upper=p2+l2,
        intrinsic_Haar_body_D1_upper=F(1053,2),
        intrinsic_Haar_body_D2_upper=F(1578005),
        derived_D1_price=4096,derived_D2_price=1 << 24).items()}


def cart(matrices: np.ndarray, T: np.ndarray) -> np.ndarray:
    return 2*np.einsum('aij,...ji->...a',T,matrices).real


def control() -> dict[str, Any]:
    # Structural couplings only: never substituted into the campaign root.
    full = gauss.stationary.full
    g = [F(1,4), F(1,16), F(1,16), F(1,16)]
    Phi0 = np.sqrt(np.asarray(g,dtype=float))[:,None,None]*full.supplier.primitive.SELECTED
    geom = full.normal_geometry({'frame':Phi0,'a':1.,'g0':float(g[0]),'gs':float(g[1])},[full.exact(x) for x in g])
    T, Q0 = geom['M']/np.sqrt(8), geom['Q']
    Qcart = geom['Qcart']
    rotation = 2*np.einsum('aij,bji->ab',T,full.supplier.static.force.T).real
    assert np.linalg.norm(rotation.T@rotation-np.eye(15)) < 1e-13
    assert np.linalg.norm(Q0.T@Q0-np.eye(45)) < 1e-13
    O0 = gauss.orbit(Phi0,T)
    assert np.linalg.norm(O0.T@Q0) < 1e-13
    rng = np.random.default_rng(2221009)
    def unit(shape: tuple[int,...]) -> np.ndarray:
        z = rng.normal(size=shape)
        return z/np.linalg.norm(z)
    R = 2.**-20
    s, f, C, theta = [R/3*unit(shape) for shape in ((45,), (64,45), (192,15), (64,15))]
    f -= f.mean(axis=0)
    ds, df, dc, dt = [unit(shape)/2 for shape in ((45,), (64,45), (192,15), (64,15))]
    df -= df.mean(axis=0)
    slow_field = np.einsum('a,amij->mij',s,geom['deltas'])/8
    slow_dot = np.einsum('a,amij->mij',ds,geom['deltas'])/8
    Q, Qd = source.adapted_jet(Phi0+slow_field,slow_dot,T,Q0)
    fast_field = np.einsum('xma,aij->xmij',(f@Q.T).reshape(64,4,15),T)
    fast_dot = np.einsum('xma,aij->xmij',(df@Q.T+f@Qd.T).reshape(64,4,15),T)
    fields = Phi0[None]+slow_field[None]+fast_field
    fd = slow_dot[None]+fast_dot
    U, Ud, E, Ed = [], [], [], []
    base_theta = .6*rng.normal(size=(64,15))
    for x, dx, base_x in zip(theta,dt,base_theta,strict=True):
        base = expm(1j*np.einsum('a,aij->ij',base_x,T))
        u, ud = expm_frechet(1j*np.einsum('a,aij->ij',x,T),1j*np.einsum('a,aij->ij',dx,T))
        U.append(base@u); Ud.append(base@ud)
    for x, dx in zip(C,dc,strict=True):
        e, ed = expm_frechet(1j*np.einsum('a,aij->ij',x,T),1j*np.einsum('a,aij->ij',dx,T))
        E.append(e); Ed.append(ed)
    U, Ud, E, Ed = map(np.asarray,(U,Ud,E,Ed))
    dag = lambda x: x.conj().swapaxes(-1,-2)
    A = dag(U)@Ud  # left body Maurer; J(-theta), same norm/jet bounds
    # Udag dU uses LEFT J(-theta); Ed Edag uses RIGHT J(C).
    laboratory_phi_dot = Ud[:,None]@fields@dag(U)[:,None]+U[:,None]@fd@dag(U)[:,None]+U[:,None]@fields@dag(Ud)[:,None]
    physical_phi = cart(dag(U)[:,None]@laboratory_phi_dot@U[:,None],T)/np.sqrt(8)
    predicted_phi = cart(fd+A[:,None]@fields-fields@A[:,None],T)/np.sqrt(8)
    wrong_phi = cart(fd,T)/np.sqrt(8)
    wrong_normalization = cart(fd+A[:,None]@fields-fields@A[:,None],T)
    sites = np.asarray(list(product(range(4),repeat=3)))
    index = {tuple(site):x for x,site in enumerate(sites)}
    links_dot, predicted_links, omitted_target = [], [], []
    for x, site in enumerate(sites):
        for axis in range(3):
            y = index[tuple((site+np.eye(3,dtype=int)[axis])%4)]; e=3*x+axis
            L = U[x]@E[e]@dag(U[y])
            Ld = Ud[x]@E[e]@dag(U[y])+U[x]@Ed[e]@dag(U[y])+U[x]@E[e]@dag(Ud[y])
            links_dot.append(cart(-1j*dag(U[x])@Ld@dag(L)@U[x],T)*np.sqrt(10/3))
            predicted_links.append(cart(-1j*(A[x]+Ed[e]@dag(E[e])-E[e]@A[y]@dag(E[e])),T)*np.sqrt(10/3))
            omitted_target.append(cart(-1j*(A[x]+Ed[e]@dag(E[e])),T)*np.sqrt(10/3))
    links_dot, predicted_links, omitted_target = map(np.asarray,(links_dot,predicted_links,omitted_target))
    result: dict[str, Any] = dict(test_couplings_exact=[str(x) for x in g],test_radius=R,
        arbitrary_local_group_base_max_angle_norm=float(np.max(np.linalg.norm(base_theta,axis=1))),
        physical_Haar_angles_restricted=False,
        exact_recipe_floating_Q_metric_defect=float(np.linalg.norm(Q0.T@Q0-np.eye(45))),
        exact_recipe_floating_Qcart_metric_defect=float(np.linalg.norm(Qcart.T@Qcart-np.eye(45))),
        exact_recipe_floating_normal_orbit_defect=float(np.linalg.norm(O0.T@Q0)),
        Cart_rotation_metric_defect=float(np.linalg.norm(rotation.T@rotation-np.eye(15))),
        physical_body_Phi_differential_defect=float(np.linalg.norm(physical_phi-predicted_phi)),
        physical_body_link_differential_defect=float(np.linalg.norm(links_dot-predicted_links)),
        wrong_omitted_local_frame_error=float(np.linalg.norm(physical_phi-wrong_phi)),
        wrong_omitted_link_target_error=float(np.linalg.norm(links_dot-omitted_target)),
        wrong_Phi_metric_normalization_error=float(np.linalg.norm(physical_phi-wrong_normalization)),
        polar_fast_frame_derivative_term_norm=float(np.linalg.norm(f@Qd.T)),
        status='Floating noncommuting all64-site/all192-link structural controls, not interval or global input certificates')
    assert result['physical_body_Phi_differential_defect'] < 1e-12
    assert result['physical_body_link_differential_defect'] < 1e-12
    for name in ('wrong_omitted_local_frame_error','wrong_omitted_link_target_error','wrong_Phi_metric_normalization_error'):
        assert result[name] > 1e-3, (name,result[name])
    return result


def run() -> dict[str, Any]:
    start = time.monotonic()
    assert sha(PROJECTOR_FILE) == PROJECTOR_SHA
    # Only this small consumer selection is parsed. The 1.4GB quantum receipt is never opened.
    import subprocess
    selected = subprocess.run(['jq','-c','.actual_full5760_local960_corrected_quantum_CAR_source_proof.exact_joint_wave_ABI | {mean_and_frame,physical_G,physical_K}',str(ABI_FILE)],check=True,capture_output=True,text=True)
    abi = json.loads(selected.stdout)
    assert all(isinstance(value,str) for value in abi.values())
    with threadpool_limits(limits=1):
        finite = control()
    source_paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module,'__file__',None)
        if not isinstance(filename,str):
            continue
        path = Path(filename).resolve()
        if path.parent == ROOT and path.suffix == '.py':
            source_paths.add(path)
    return dict(scope='Conditional exact local body-framed chart-jet ingredient, campaign OPEN',
        governing_plan='issue-222-induced-photon-delta13-continuation-20261008',
        governing_revision='2de065e033f83c33591752f5778876d9e28acce0578dda17e5295a0abe572e15',
        template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        runtime=dict(executable=sys.executable,python=platform.python_version(),numpy=np.__version__),
        exact_consumer_selected_ABI=abi,derivation=DERIVATION,exact_envelopes=exact_envelopes(),
        finite_controls=finite,input_sha256={str(p):sha(p) for p in (ABI_FILE,PROJECTOR_FILE)},
        source_sha256={str(p):sha(p) for p in sorted(source_paths)},
        source_utf8=Path(__file__).read_text(),
        remaining_debts=[
            'Actual exact root existence/global input certification not independently established.',
            'Rounded Q0/polar matrix caches not certified by finite matrix observations.',
            'No local tube/reach/injectivity/global chart certificate or D-inverse bound derived.',
            'No determinant/Haar potential, full kinetic action or source propagation certification.',
            'No occupied fermion Kato256 proof, particle identity, photon/electron closure or parameter adoption.'
        ],elapsed_seconds=time.monotonic()-start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    receipt = run()
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),exact_envelopes=receipt['exact_envelopes'],finite_controls=receipt['finite_controls']),indent=2))
