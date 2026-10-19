"""Independent exact native graph controls; never import or execute frozen targets."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
from typing import Any, cast
import sympy as sp

ROOT = Path(__file__).resolve().parent
PREFIX = 'bulk_gauss_review_exact_20261011'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph(n: int):
    sites = [(x, y, z) for x in range(n) for y in range(n) for z in range(n)]
    interior = [s for s in sites if 0 < s[0] < n - 1]
    rows = {s: i for i, s in enumerate(interior)}
    edges = []
    for s in sites:
        for axis in range(3):
            if axis == 0 and s[0] == n - 1:
                continue
            if axis != 0 and s[0] in (0, n - 1):
                continue
            target = list(s)
            target[axis] = (target[axis] + 1) % n
            edges.append((s, tuple(target), axis))
    d = sp.zeros(len(rows), len(edges))
    for e, (source, target, _) in enumerate(edges):
        if source in rows:
            d[rows[source], e] = 1
        if target in rows:
            d[rows[target], e] = -1
    return edges, d


def deck_and_sign():
    n = 4
    edges, d = graph(n)
    tree = [j for j, (source, _, axis) in enumerate(edges)
            if axis == 0 and source[0] <= n - 3]
    chords = [j for j in range(len(edges)) if j not in tree]
    t = d[:, tree]
    b = d[:, chords]
    assert abs(t.det(method='domain-ge')) == 1
    x = t.inv(method='DM') * b
    c = sp.zeros(len(edges), len(chords))
    for i, e in enumerate(tree):
        c[e, :] = -x[i, :]
    for i, e in enumerate(chords):
        c[e, i] = 1
    assert d * c == sp.zeros(d.rows, c.cols)
    assert c[chords, :] == sp.eye(c.cols)
    li = d * d.T
    gram = c.T * c
    detli = li.det(method='domain-ge')
    detgram = gram.det(method='domain-ge')
    assert detli == detgram
    assert d.rows == (n - 2) * n**2 and d.cols == 3*n**3 - 5*n**2
    assert c.cols == 2*n**3 - 3*n**2
    k = sp.zeros(d.cols, 1)
    e = next(i for i, (s, _, axis) in enumerate(edges)
             if s == (n-2, 0, 0) and axis == 0)
    k[e] = 1
    drho = d*k
    lam = li.inv(method='DM') * drho
    longitudinal = d.T * lam
    projected = k - longitudinal
    h = sp.factor(lam.dot(drho))
    peierls_turns = sp.factor(k.dot(projected))
    assert h + peierls_turns == 1
    assert sp.simplify(peierls_turns + h - 1) == 0
    assert cast(Any, sp.simplify(2*h)).q != 1
    assert cast(Any, sp.simplify(peierls_turns + h)).q == 1
    assert cast(Any, sp.simplify(peierls_turns - h)).q != 1
    # For rho'=rho+Dk: A(y+s) U_rho = U_rho' A(y).
    # The phase in turns must equal U_rho'/U_rho = -h, modulo integers.
    return {
        'N': n, 'M': d.cols, 'I': d.rows, 'd': c.cols,
        'det_LI_exact': str(detli), 'det_CGram_exact': str(detgram),
        'unimodular_tree': True, 'saturated_identity_chord_rows': True,
        'cycle_kernel_exact': True,
        'primitive_deck': 'Q Z^M = C(C^T C)^-1 Z^d',
        'cell_radian_volume': '(2pi)^d/sqrt(det LI)',
        'cycle_only_cell_overcount': str(detli),
        'normal_chord_edge': e,
        'h_lambda_Dk_exact': str(h), 'k_Qn_exact': str(peierls_turns),
        'minus_reduced_gluing_intertwines': True,
        'plus_reduced_gluing_intertwines': False,
        'plus_wrong_residual_turns_mod_1': str(sp.Mod(peierls_turns-h, 1)),
        'plus_full_translation_total_turns': str(peierls_turns+h),
        'plus_full_section_and_translation_valid': True,
        'fibre_equation': 'f(y+2pi Qn)=exp(2pi i b(n))*exp(-2pi i lambda_n.rho) f(y)',
        'full_section_equation': 'Psi(y+D^T lambda)=exp(+i lambda.rho) f(y)',
        'sign_scope': 'Operator action PLUS; forward reduced wave gluing MINUS; inverse gluing PLUS.'
    }


def stripe(n: int):
    assert n >= 4 and n % 2 == 0
    m, c = n//2, 128
    rho = [Fraction(c if i < m else -c) for i in range(n)]
    slab = [Fraction(0)] * n
    for i in range(1, m):
        slab[i] = c * (-Fraction(i*i, 2) + Fraction(i*(2*m*m-2*m+1), 2*(2*m-1)))
        slab[n-1-i] = -slab[i]
    assert all(2*slab[i]-slab[i-1]-slab[i+1] == rho[i] for i in range(1,n-1))
    # Periodic solve from exact edge differences: v_(i+1)-v_i = d_i,
    # d_i-d_(i-1)=-rho_i; fix sum d_i=0, then mean v=0.
    cumulative = []
    running = Fraction(0)
    for r in rho:
        running -= r
        cumulative.append(running)
    offset = -sum(cumulative) / n
    differences = [offset + a for a in cumulative]
    torus = [Fraction(0)]
    for diff in differences[:-1]:
        torus.append(torus[-1] + diff)
    mean = sum(torus) / n
    torus = [u-mean for u in torus]
    assert sum(torus) == 0
    assert all(2*torus[i]-torus[(i-1)%n]-torus[(i+1)%n] == rho[i] for i in range(n))
    hc_s = 64 * sum(rho[i]*slab[i] for i in range(1,n-1)) / n
    hc_t = 64 * sum(r*u for r,u in zip(rho,torus)) / n
    assert hc_s == 64*c*c*Fraction((m-1)*(m*m-m+1), 6*(2*m-1))
    assert hc_t == 64*c*c*Fraction(m*m+2,12)
    difference = hc_s-hc_t
    assert difference == -65536*Fraction(n*n,n-1)
    return {'N':n, 'HC_slab_over_V':str(hc_s), 'HC_torus_over_V':str(hc_t),
            'difference_over_V':str(difference), 'dirichlet_and_periodic_poisson_exact':True,
            'charged_species':64, 'particles_each':2*n**3,
            'occupancy': 'wedge4 on first half; wedge0 on second half; Ref4; neutral2V',
            'interpretation':'Native admissible stripe, not thermal/ground typicality.'}


def dtn():
    # Symbolic polynomial transfer solve, independent of cosh implementation.
    lam = sp.Symbol('lambda', nonnegative=True)
    rows = []
    for n in (4,6,8):
        l = n-1
        a, b = [sp.Integer(0),sp.Integer(1)], [sp.Integer(1),sp.Integer(0)]
        for _ in range(1,l):
            a.append(sp.expand((2+lam)*a[-1]-a[-2]))
            b.append(sp.expand((2+lam)*b[-1]-b[-2]))
        u1 = sp.cancel((1-b[l])/a[l])
        u = [sp.cancel(a[i]*u1+b[i]) for i in range(n)]
        assert u[0] == u[-1] == 1
        assert all(sp.cancel((2+lam)*u[i]-u[i-1]-u[i+1]) == 0 for i in range(1,l))
        energy = sum((u[i+1]-u[i])**2 for i in range(l))+lam*sum(v*v for v in u[1:-1])
        assert sp.cancel(energy-2*(1-u1)) == 0
        native_lam = {4:sp.Integer(2),6:sp.Integer(1),8:2-sp.sqrt(2)}[n]
        rows.append({'N':n,'lambda_q_exact':str(native_lam),
                     'DtN_exact':str(sp.simplify((1-u1).subs(lam,native_lam))),
                     'centre_exact':str(sp.simplify(u[n//2].subs(lam,native_lam))),
                     'energy_equals_2DtN_symbolic':True})
    return {'finite_exact':rows,
            'analytic_macro_limit':'lambda_N=4sin^2(pi/N); gamma_N=2asinh(sin(pi/N)); N gamma_N -> 2pi; (N-1)gamma_N/2 -> pi; centre numerator ->1; transfer ->sech(pi)>0',
            'fixed_q_limit':'q nonzero mod2pi gives gamma>0 and exponential interior decay',
            'price_scope':'Longitudinal Gauss energy only, not magnetic J,K/Gibbs Hessian.'}


def spectral_controls():
    eps = sp.Symbol('epsilon', positive=True)
    h = sp.diag(0,eps,2*eps)
    p0 = sp.diag(1,0,0)
    a = sp.zeros(3);a[1,0] = 1
    omega = sp.Matrix([1,0,0])
    xi = (sp.eye(3)-p0)*a*omega
    assert xi.dot(xi) == 1 and h*xi == eps*xi
    limiting_p0 = sp.eye(3)
    assert (sp.eye(3)-limiting_p0)*a*omega == sp.zeros(3,1)
    beta,alpha = sp.symbols('beta alpha', positive=True)
    # Two exact finite Gibbs controls, not the target native law.
    z_plus = sp.exp(beta*alpha)+sp.exp(-beta*alpha)
    z_minus = sp.exp(-beta*alpha)+sp.exp(beta*alpha)
    assert z_plus == z_minus
    ex_plus = (sp.exp(beta*alpha)-sp.exp(-beta*alpha))/z_plus
    ex_minus = -ex_plus
    return {'wholeP0':{'finite_H':'diag(0,epsilon,2epsilon)', 'finite_Q_spectrum':'delta_epsilon, mass1',
                      'weak_limit_finite_Q_spectra':'delta0, mass1',
                      'limiting_H':'zero3x3', 'limiting_entire_P0':'identity3',
                      'limiting_Q_spectrum':'zero measure',
                      'deletion_of_zero_and_weak_limit_do_not_commute':True},
            'pressure_state':{'H_plus':'-alpha Z','H_minus':'+alpha Z',
                              'partition_equality_exact':str(z_plus),
                              'Z_expectation_plus':str(ex_plus), 'Z_expectation_minus':str(ex_minus),
                              'same_pressure_not_same_state':True},
            'signed_thermal':{'H':'diag(0,1)','A':'Pauli X',
                              'measure':'(delta_(+1)+exp(-beta)*delta_(-1))/(1+exp(-beta))',
                              'KMS_ratio_negative_over_positive':'exp(-beta)',
                              'no_ground_nonnegative_L_at_finite_beta':True},
            'averaging_counterexample':{'quartic_mass':'S_x=V at the selected central site, zero elsewhere',
                                       'S_over_V':1,'central_S':'V -> infinity',
                                       'scope':'Arithmetic falsifier of averaging implication, not a claimed native Gibbs state'},
            'bridge':'Only conditional: legal common representation, differentiated heat/forms and integrable connected Duhamel covariance plus common dynamics/continuity. No native estimate executed.'}


def child(request: Path, output: Path):
    inputs = json.loads(request.read_text())
    for name,pin in inputs['input_sha256'].items():
        assert sha(Path(name)) == pin, name
    result = {'kind':'INDEPENDENT_EXACT_NATIVE_BULK_GAUSS_REVIEW_NOT_STATE_OR_PHOTON',
              'source_sha256':sha(Path(__file__)), 'request_sha256':sha(request),
              'input_sha256':inputs['input_sha256'],
              'deck_and_sign':deck_and_sign(),
              'native_stripes':[stripe(n) for n in inputs['stripe_N']],
              'native_DtN':dtn(), 'spectral_and_pressure_controls':spectral_controls(),
              'legacy_entrypoints_executed':False, 'native_state_identity':False,
              'phase_stiffness_photon_admission':False}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'receipt':output.name,'exact_controls_completed':True,'native_state_identity':False}))


def immutable_archive(path: Path, files: list[Path]):
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        for i,f in enumerate(files):
            z.write(f, f'{i:03d}_{f.name}')
    path.chmod(0o444)


def execute(request: Path):
    data = json.loads(request.read_text())
    files = list(dict.fromkeys([Path(__file__).resolve(),request.resolve()]+[Path(p) for p in data['input_sha256']]))
    pins = {str(p):sha(p) for p in files}
    pre = ROOT/(PREFIX+'_preexecution.zip')
    immutable_archive(pre,files)
    receipt = ROOT/(PREFIX+'_receipt.json')
    argv = [data['python'],'-B',str(Path(__file__).resolve()),'--child','--request',str(request.resolve()),'--output',str(receipt)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    start = time.monotonic()
    process = subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,check=False)
    elapsed = time.monotonic()-start
    stdout,stderr = ROOT/(PREFIX+'_stdout.txt'),ROOT/(PREFIX+'_stderr.txt')
    stdout.write_bytes(process.stdout);stderr.write_bytes(process.stderr)
    unchanged = all(sha(p)==pins[str(p)] for p in files)
    exitpath = ROOT/(PREFIX+'_exit.json')
    exitpath.write_text(json.dumps({'argv':argv,'returncode':process.returncode,'natural_exit':True,
                                  'elapsed_seconds':elapsed,'source_inputs_unchanged':unchanged,
                                  'source_input_sha256':pins},indent=2)+'\n')
    outputs = [stdout,stderr,exitpath]+([receipt] if receipt.exists() else [])
    executed = ROOT/(PREFIX+'_executed.zip')
    immutable_archive(executed,files+outputs)
    capsule = {'kind':'IMMUTABLE_INDEPENDENT_EXACT_BULK_GAUSS_REVIEW_CAPSULE',
               'natural_exit':True,'returncode':process.returncode,'elapsed_seconds':elapsed,
               'source_inputs_unchanged':unchanged,'source_input_sha256':pins,
               'preexecution_zip':pre.name,'preexecution_sha256':sha(pre),
               'executed_zip':executed.name,'executed_sha256':sha(executed),
               'outputs_sha256':{p.name:sha(p) for p in outputs},
               'legacy_entrypoints_executed':False,'native_state_spectral_identity':False}
    path = ROOT/(PREFIX+'_capsule.json')
    path.write_text(json.dumps(capsule,indent=2)+'\n')
    for p in outputs+[path]:
        p.chmod(0o444)
    print(json.dumps(capsule))
    raise SystemExit(process.returncode if unchanged else 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--child',action='store_true')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.child:
        if args.output is None:
            parser.error('--child requires --output')
        child(args.request,args.output)
    else:
        execute(args.request)


if __name__ == '__main__':
    main()
