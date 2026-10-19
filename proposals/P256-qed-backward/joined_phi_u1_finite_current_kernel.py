"""NEW same-H all129 nonstationary current Sobolev remainder; no pole claim.

The accepted current-moment producer is an input, never rerun by this CLI.
Analytic envelopes are deliberately separate from floating signal observations.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import numpy as np
import sympy as sy
from threadpoolctl import threadpool_limits

import joined_phi_u1_body_word as body
import joined_phi_u1_current_moments as current

ROOT = Path(__file__).resolve().parent
G = 64.0
S = 2.0
PINNED = {
    'joined_phi_u1_current_moments.py': 'dae823744522b4bd5f540a0e7e906135876bb57ed772f3b6849caa64ed3ed89f',
    'joined_phi_u1_current_moments_receipt.json': 'd267a425e63be09c39200743ea75a2efec5d339f4387483507d42fe21688ec3a',
    'joined_phi_u1_current_moments_receipt.npz': 'e9e6b945e3d9a211d7aed24c4470bce6b2e9d0cdb34c6006e54c76612538a8c4',
    'joined_phi_u1_current_natural_exit_capsule.json': '681eb30095a0933cc6234305af242746090195a9714d27de116ec2235fc26845',
    'joined_phi_u1_body_word.py': '5ac906cb786f0e6283f3abd402548611f799d06bea77cb73fabec96505c3c7c4',
}
DERIVATION = {
    'units_sign': 'hbar=1, body/lattice energy units. T=64 sum Pi^2, HC=64 rho^T Lplus rho, M=sum charged dGamma(h). J=-dM, C=d2M. R(t)=-i<[J(t),J]>, F=Rprime(0)=<[[H,J],J]>. Both times refer to the initial, generally nonstationary preparation.',
    'domain': 'Finite fixed particle sectors, real unwrapped R129. HC is a constant positive finite-CAR matrix; M and every Theta derivative are bounded smooth finite Fourier multipliers. H=T+HC+M is self-adjoint on D(T); Gaussian initial vectors and current-inserted vectors are Schwartz. All displayed differentiated forms are valid on this core and extend by graph-norm bounds. No operator norm of Pi or T is asserted.',
    'neutral': 'At fixed meanPhi, charge0 heavy has no Theta or charged-rho dependence. H=Hcharged tensor I+I tensor Hneutral and J=Jcharged tensor I. Therefore exp(iHt)J exp(-iHt)=exp(iHcharged t)Jcharged exp(-iHcharged t) tensor I EXACTLY, for arbitrary neutral state. Neither heavy energy nor Lambda^(5/4) bosonic stiffness enters an energy norm below. This is not compactPhi current transfer.',
    'retarded_remainder': 'B_mu=ad_H^2(J_mu) is Hermitian. Rdoubleprime_mu,nu(t)=i<[B_mu(t),J_nu]>. In a pure-state purification/direct sum of raw Ref channels, ||Rdoubleprime(t)||F <=2 (sum_mu ||B_mu U(t)psi||^2)^1/2 (sum_nu ||J_nu psi||^2)^1/2. Integrating twice gives ||R(t)-R(0)-tF||F <= t^2 Bstar b0. b0^2=Tr accepted all129 <JJ>, NOT a current operator norm. The raw source has norm sqrt(.5), not one.',
    'energy': 'Write E=T+HC; T and HC commute and are positive. If ||M||<=m in the preserved charged particle sector, then <E>_t <= <E>_0+2m||psi||^2 and ||E U(t)psi|| <= ||Tpsi||+||HCpsi||+2m||psi||. Consequently ||grad phi||<=sqrt(E1/G), ||Delta phi||<=E2/G, ||HC phi||<=E2, ||HC^.5 phi||<=sqrt(E1), ||HC^.5 grad phi||<=E2/(2sqrt(G)). These are state-weighted Sobolev bounds under the SAME H, not stationary moments or a bounded kinetic operator.',
    'density_moments': 'For each initial fixed-number Slater, site covariance C has offdiagonal -||Pxy||HS^2, diagonal TrPxx-TrPxx^2<=1 (four orbitals), zero row sums, hence 0<=C<=2I. Sum active+63aux gives C<=128I, TrC<=4096. Background meanrho=0 by translation and half filling. Each Ref3 source channel has meanrho=u-delta0, u>=0,sumu=1, so a^T Lplus a<=1 using ||Lplus||=1/2. Ref density is NOT omitted. For X=z.rho, z real Gaussian covariance Lplus, E_z X^4=3 E(rho^T Lplus rho)^2. Slater cumulants from logdet(I-P+exp(tD)P): k4=TrD4P-4TrD3PDP-3TrD2PD2P+12TrD2PDPDP-6TrDPDPDPDP. Schatten Holder/compression bounds each trace by Tr|D|4P, giving |k4|<=26TrD4P; |k3|<=6Tr|D|3P. Thus HCsecond <=G^2/3 [3aL a squared+(6TrLC+12*128*.5)aL a+3((TrLC)^2+2*128^2 TrL^2)+24sqrt(15 aL a) Ntotal Ldiag^(3/2)+78 Ntotal Ldiag^2], with TrLC<=128TrL. Different flavours factor at fixed Theta only; the current b0 and F retain all sharedTheta disconnected terms from the accepted input.',
    'path_algebra': 'For an oriented charged hop A_r=c_xdag B_r c_y, delta_r=e_x-e_y, [HC,A_r]=A_r V_r, V_r=128 delta_r Lplus rho+64 delta_r Lplus delta_r. Nuclear bound ||A_r||<=4 b_r where b_r=|c0|+|ci| uses exact mean box ||frame_mu||<1 and AP seams have modulus one. V_r norm on phi <=2sqrt(G d_r) sqrt(E1)+G d_r||phi||; V_r^2 norm <=4G d_r E2+4(G d_r)^(3/2)sqrt(E1)+(G d_r)^2||phi||. Every actual one/two-link path, both endpoint frame factors, onsite, and both charged profiles are retained.',
    'electric_second': 'For J_r=-i q_r A_r exp(iq_r.Theta), [T,[T,J_r]]=G^2(Delta^2J_r+4 gradDeltaJ_r.grad+4 HessJ_r:Hess). The path graph norm is ||A_r||[G^2 |q_r|4 n+4G^2 |q_r|3 P+4G |q_r|2 E2]. The two electric-Coulomb cross terms are equal because [T,HC]=0; their bound is 2G||A_r||[|q_r|2 v1(d_r)+2|q_r|(sqrt(d_r) E2+Gd_r P)]. No integration-by-parts term is dropped.',
    'mixed_matter': 'For a same-flavour chain C_er=[A_e,A_r], with Fourier q_e+q_r and endpoint delta_er, electric/matter cross bound per chain is G(|q_r|2+|q_e+q_r|2)||C_er||n+2G|q_e+2q_r| ||C_er||P; an additional 2G |q_r.q_e| ||A_r|| ||A_e|| n spans ALL charged flavours. Coulomb/matter cross bound is ||C_er||[v1(d_r)+v1(d_er)]+2G||A_r||sum_e|delta_r Lplus delta_e| ||A_e|| n, again all64 flavours in the product. ad_M^2(A_r) is a one-body local commutator: ||dGamma(ad_h^2 a_r)||<=4 hmax^2 ||a_r||1; it is NOT priced by the square of the extensive matter energy.',
    'wholematrix': 'Use the vector triangle inequality (sum_mu||B_mu phi||^2)^1/2 <=sum_r |q_r| times the path bound, summing active and 63 auxiliaries. This bound covers the entire129x129 matrix, not a selected harmonic or response entry. Exact rational Lplus and Q Q^T=I-D^T Lplus D supply q Gram, density metrics and cross-flavour couplings without fitting Q or source.',
    'contact': 'The capsule retains all129 C(0)=<d2H>, separately from R(t,0); the explicit instantaneous current derivative is -C(0), since J(A)=-dH(A). Contact is not absorbed into, sign-flipped with, or discarded from the retarded commutator. C(t) at a later nonstationary insertion time is not inferred from C(0).',
    'limits': 'Analytic inequalities hold conditional on the pinned meanPhi/Clifford/orthogonal-Q preparation. Exact rational geometry is checked algebraically. Evaluated prices are floating with explicit upward cushion, not an interval arithmetic proof. The first-moment signal is an accepted floating observation; nonvacuity relative to it is stated with a 1% signal allowance, not an unproved exact spectral lower enclosure. Finite Galerkin controls separately have a proven finite-matrix norm bound; they do not certify global spectral convergence. No ground, physical pole, dispersion, cone, SOURCE promotion or fullPhi current transfer.',
}


def exact_geometry(native: dict[str, Any], geo: dict[str, Any]) -> dict[str, Any]:
    d = np.rint(native['D']).astype(np.int64)
    body.require(float(np.max(np.abs(d-native['D']))))
    lap = sy.Matrix(d@d.T)
    constant = sy.ones(64)/64
    lp = (lap+constant).inv()-constant
    if lap*lp != sy.eye(64)-constant:
        raise ArithmeticError('Exact all63-mode pseudoinverse identity failed')
    den = int(sy.ilcm(*[x.q for x in lp]))
    lint = np.asarray([[int(x*den) for x in row] for row in lp.tolist()], np.int64)
    actual = lint.astype(float)/den
    body.require(float(np.max(np.abs(actual-native['Lplus']))))
    delta = geo['delta'].astype(np.int64)
    k = geo['k'].astype(np.int64)
    density = delta@lint@delta.T
    gram = den*(k@k.T)-density
    body.require(float(np.max(np.abs(gram/den-geo['q']@geo['q'].T))))
    return dict(denominator=den, L_integer=lint, L=actual, density=density/den,
                gram=gram/den, q2=np.diag(gram)/den,
                exact_L_trace=str(sy.trace(lp)), exact_L2_trace=str(sy.trace(lp*lp)),
                exact_L_diagonal=str(lp[0, 0]), exact_identity=True)


def envelopes(sp: body.Species, native: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    paths = sp.profile['paths']
    lookup = {(int(p['axis']), int(p['length'])): float(abs(Fraction(p['c0']))+abs(Fraction(p['ci']))) for p in paths}
    # native_paths order is site-major, then profile path order; map from the
    # actual displacement rather than relying on an inferred ordering.
    native_paths = body.joined.old.native_paths(sp.profile, native)
    forward = np.array([lookup[(int(p['axis']), int(p['length']))] for p in native_paths])
    op = np.r_[forward, forward]
    return op, 4*op


def density_price(g: dict[str, Any], particles: int, source: bool) -> dict[str, float]:
    tr = float(Fraction(g['exact_L_trace']))
    tr2 = float(Fraction(g['exact_L2_trace']))
    diag = float(Fraction(g['exact_L_diagonal']))
    a = 1.0 if source else 0.0
    trlc = 128*tr
    fourth = (3*a*a+(6*trlc+12*128*.5)*a
              +3*(trlc**2+2*128**2*tr2)
              +24*math.sqrt(15*a)*particles*diag**1.5+78*particles*diag**2)
    return dict(HC_mean_upper=G*(trlc+a), HC_norm_upper=G*math.sqrt(fourth/3),
                rho_quadratic_second_upper=fourth/3,
                source_mean_quadratic_upper=a, particle_count=particles)


def chains(sp: body.Species, geo: dict[str, Any], op: np.ndarray,
           geom: dict[str, Any]) -> list[list[tuple[float, float, float, float]]]:
    x, y = geo['x'], geo['y']
    rx = np.r_[x, np.arange(64)]
    ry = np.r_[y, np.arange(64)]
    oo = np.r_[op, np.full(64, abs(float(sp.profile['onsite'])))]
    k = np.r_[geo['k'], np.zeros((64, 192))].astype(np.int64)
    delta = np.eye(64)[rx]-np.eye(64)[ry]
    lp = geom['L']
    result: list[list[tuple[float, float, float, float]]] = []
    for r in range(len(x)):
        entries = []
        for e in np.flatnonzero((ry == x[r])|(y[r] == rx)):
            # Both products matter when both endpoints coincide. Triangle
            # bounds do not pretend that their signs cancel.
            for xx, yy, present in ((rx[e], y[r], ry[e] == x[r]),
                                    (x[r], ry[e], y[r] == rx[e])):
                if not present:
                    continue
                de = np.eye(64)[xx]-np.eye(64)[yy]
                ds = float(de@lp@de)
                # Q Gram from exact rational pseudoinverse, including onsite.
                dk = k[e]+k[r]
                dd = delta[e]+delta[r]
                qs2 = float(dk@dk-dd@lp@dd)
                dk2 = k[e]+2*k[r]
                dd2 = delta[e]+2*delta[r]
                qd2 = float(dk2@dk2-dd2@lp@dd2)
                body.require(max(0., -qs2, -qd2, -ds))
                entries.append((4*oo[e]*op[r], max(0., ds), max(0., qs2), math.sqrt(max(0., qd2))))
        result.append(entries)
    return result


def commutator_price(geo: dict[str, Any], geom: dict[str, Any], species: list[body.Species],
                     ops: list[np.ndarray], bs: list[np.ndarray], cc: list[Any],
                     energy1: float, energy2: float, norm: float) -> dict[str, float]:
    p = math.sqrt(energy1/G)
    root = math.sqrt(energy1)
    q2 = geom['q2']
    qn = np.sqrt(q2)
    dens = np.diag(geom['density'])
    allb = bs[0]+63*bs[1]
    # These are shared-Theta and shared-rho product terms, not independent
    # per-flavour expectations. Full Gram matrices retain all charged copies.
    electric_product = np.abs(geom['gram'])@allb
    density_product = np.abs(geom['density'])@allb
    totals = dict(electric_electric=0., electric_coulomb=0., coulomb_coulomb=0.,
                  electric_matter=0., coulomb_matter=0., matter_matter=0.)
    def v1(d: float) -> float:
        return 2*math.sqrt(G*d)*root+G*d*norm
    for sp, op, b, rows, load in zip(species, ops, bs, cc, (1, 63)):
        for r in range(len(qn)):
            if qn[r] == 0:
                continue
            d = float(dens[r])
            values = dict(
                electric_electric=b[r]*(G**2*q2[r]**2*norm+4*G**2*qn[r]**3*p+4*G*q2[r]*energy2),
                electric_coulomb=2*G*b[r]*(q2[r]*v1(d)+2*qn[r]*(math.sqrt(d)*energy2+G*d*p)),
                coulomb_coulomb=b[r]*(4*G*d*energy2+4*(G*d)**1.5*root+(G*d)**2*norm),
                electric_matter=2*G*b[r]*electric_product[r]*norm,
                coulomb_matter=2*G*b[r]*density_product[r]*norm,
                matter_matter=4*sp.hmax**2*b[r]*norm)
            for c, dd, qs2, qd in rows[r]:
                values['electric_matter'] += c*(G*(q2[r]+qs2)*norm+2*G*qd*p)
                values['coulomb_matter'] += c*(v1(d)+v1(dd))
            for name, value in values.items():
                totals[name] += load*qn[r]*value
    # Not an interval claim: a visible cushion separates exact inequalities
    # from their floating evaluation, which is conditional in the receipt.
    totals = {name:float(np.nextafter(value*1.000001, math.inf)) for name,value in totals.items()}
    return {**totals, 'Bstar': sum(totals.values()), 'gradient_norm_upper': p,
            'state_norm': norm, 'energy_first_upper': energy1, 'energy_graph_norm_upper': energy2}


def run(output: Path, controls: bool = True) -> dict[str, Any]:
    started = time.monotonic()
    own = Path(__file__).read_bytes()
    accepted = {name:(ROOT/name).read_bytes() for name in PINNED}
    for name, raw in accepted.items():
        if hashlib.sha256(raw).hexdigest() != PINNED[name]:
            raise ArithmeticError('Accepted input bytes changed: '+name)
    data, species, inputs, _ = body.full_inputs()
    geo = current.geometry(data, species[0])
    geom = exact_geometry(data['native'], geo)
    ops, bs = zip(*(envelopes(sp, data['native']) for sp in species))
    cc = [chains(sp, geo, op, geom) for sp,op in zip(species,ops)]
    matrices = np.load(ROOT/'joined_phi_u1_current_moments_receipt.npz')
    raw_weights = matrices['source_channel_weights']
    body.require(abs(float(raw_weights.sum())-.5))
    tmean = G*129/(4*S)
    tnorm = G*math.sqrt(129*131)/(4*S)
    rows = {}
    for state, mass, source in (('background', 1., False), ('raw_source', .5, True)):
        count = 128+int(source)+63*128
        n = math.sqrt(mass)
        matter = (128+int(source))*species[0].hmax+63*128*species[1].hmax
        dp = density_price(geom, count, source)
        e1 = mass*(tmean+dp['HC_mean_upper']+2*matter)
        e2 = n*(tnorm+dp['HC_norm_upper']+2*matter)
        # A VALID loose route, still no kinetic operator norm: replace the
        # prepared density moment by its entire fixed-sector charged bound.
        hc_sector = G*.5*64*(128+int(source))**2
        loose1 = mass*(tmean+hc_sector+2*matter)
        loose2 = n*(tnorm+hc_sector+2*matter)
        tight = commutator_price(geo,geom,species,list(ops),list(bs),cc,e1,e2,n)
        loose = commutator_price(geo,geom,species,list(ops),list(bs),cc,loose1,loose2,n)
        jj = matrices[state+'_second']
        b0 = math.sqrt(float(np.trace(jj).real))
        f = matrices[state+'_F']
        signal = float(np.linalg.norm(f))
        lower_signal = .99*signal
        coefficient = tight['Bstar']*b0
        loose_coefficient = loose['Bstar']*b0
        # The same clock will be selected for both raw and normalized states.
        rows[state] = dict(raw_norm_squared=mass, density_price=dp, preserved_matter_sector_bound=matter,
            kinetic_mean=tmean*mass, kinetic_graph_norm=tnorm*n, Sobolev=tight,
            loose_Sobolev=loose, current_inserted_state_norm=b0,
            accepted_first_moment_frobenius=signal, observed_signal_with_one_percent_allowance=lower_signal,
            wholematrix_quadratic_remainder_coefficient=coefficient,
            loose_wholematrix_quadratic_coefficient=loose_coefficient,
            material_coefficient_improvement=loose_coefficient/coefficient,
            contact_frobenius=float(np.linalg.norm(matrices[state+'_contact'])),
            retarded_initial_frobenius=float(np.linalg.norm(matrices[state+'_retarded_initial'])))
    allowed = min(.25*r['observed_signal_with_one_percent_allowance']/r['wholematrix_quadratic_remainder_coefficient'] for r in rows.values())
    clock_power = math.ceil(-math.log2(allowed))
    clock = 2.**(-clock_power)
    for row in rows.values():
        signal_at_clock = clock*row['observed_signal_with_one_percent_allowance']
        error = clock**2*row['wholematrix_quadratic_remainder_coefficient']
        loose_error = clock**2*row['loose_wholematrix_quadratic_coefficient']
        row['clock'] = dict(power_of_two=-clock_power, t=clock, signal_lower_observed=signal_at_clock,
            remainder_upper=error, signal_minus_remainder=signal_at_clock-error,
            signal_to_remainder=signal_at_clock/error, remainder_to_signal=error/signal_at_clock,
            loose_remainder_to_signal=loose_error/signal_at_clock,
            whole129_matrix_nonvacuous_relative_to_observed_signal=error<signal_at_clock,
            no_exact_interval_signal_lower_claim=True)
        if error >= signal_at_clock or row['material_coefficient_improvement'] <= 2:
            raise ArithmeticError('All129 estimate remained vacuous or not materially improved')
    finite_arrays: dict[str, Any] = {}
    for state, row in rows.items():
        linear = matrices[state+'_retarded_initial']+clock*matrices[state+'_F']
        finite_arrays[state+'_retarded_linear_at_clock'] = linear
        finite_arrays[state+'_initial_contact_C'] = matrices[state+'_contact']
        finite_arrays[state+'_initial_explicit_current_derivative'] = -matrices[state+'_contact']
        row['clock']['full_matrix_linear_approximant_frobenius'] = float(np.linalg.norm(linear))
        row['clock']['full_matrix_error_ball_radius'] = row['clock']['remainder_upper']
        row['clock']['full_matrix_lower_observed_after_R0_and_allowance'] = .99*float(np.linalg.norm(linear))-row['clock']['remainder_upper']
    matrix_path = output.with_suffix('.npz')
    np.savez_compressed(matrix_path, **finite_arrays)
    control_result = None
    extra = {}
    if controls:
        import joined_phi_u1_finite_current_kernel_controls as independent
        extra[Path(independent.__file__).name] = Path(independent.__file__).read_bytes()
        control_result = independent.run()
    captured = {**body.CAPTURED, **inputs, **accepted, **extra, Path(__file__).name:own}
    capture_path = output.with_name('joined_phi_u1_finite_current_kernel_input_bytes.zip')
    with zipfile.ZipFile(capture_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(captured.items()):
            archive.writestr(name, raw)
    hashes = {name:hashlib.sha256(raw).hexdigest() for name,raw in captured.items()}
    for name, raw in captured.items():
        disk = ROOT/name
        if disk.exists() and disk.read_bytes() != raw:
            raise ArithmeticError('Captured source/input changed during producer: '+name)
    result = dict(status='NEW full129 same-H nonstationary current remainder; conditional analytic envelope and independent finite controls, no adoption',
        governing_plan='issue-222-induced-photon-delta13-continuation-20261008',
        assignment_revision='34c80e78e7fbc97c60c8ac8294df069a56f4dc459f43879d17c8862dce6bd246',
        work_items=['joined-explanatory-campaign','induced-photon'], derivation=DERIVATION,
        coordinates=129, signed_paths=len(geo['x']), charged_flavours=64, auxiliary_load=63,
        normalized_background_and_raw_fourRef=True, source_channel_weights=raw_weights.tolist(),
        geometry={key:value for key,value in geom.items() if key.startswith('exact_') or key=='denominator'},
        all129_states=rows, independent_finite_controls=control_result,
        finiteclock_matrix_capsule=dict(path=matrix_path.name,sha256=hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            arrays={name:list(value.shape) for name,value in finite_arrays.items()},
            interpretation='All129 R0+tF approximants and analytic wholematrix error balls, NOT measured full129 interacting evolution; initial full contact retained separately'),
        input_hashes=hashes, immutable_input_capture=dict(path=capture_path.name,sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),stable_bytes_at_producer_exit=True),
        accepted_initial_moment_production_rerun=False,
        exact=['All63-mode rational Lplus identity','Neutralheavy current cancellation','Nonstationary Duhamel/Sobolev inequalities and Slater cumulant envelope identities'],
        floating=['Accepted all129 R0/F/contact/current insertion norm','Evaluated global envelopes with upward cushion','Independent finite spectral consumer'],
        conditional=['Pinned fixed meanPhi/root/Clifford and exact orthogonal quotient hypotheses','No interval enclosure of accepted first-moment matrices or evaluated constants'],
        open=['Photon pole/dispersion/integrability','Joined body and compactPhi ground/source admission','FullPhi current transfer','Commoncone/electron-photon explanatory closure; SOURCE and CURRENT427 untouched'],
        runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('joined_phi_u1_finite_current_kernel_receipt.json'))
    args = parser.parse_args()
    with threadpool_limits(1):
        result = run(args.output)
    print(json.dumps(dict(status=result['status'],states=result['all129_states'],elapsed_seconds=result['elapsed_seconds'])),flush=True)
