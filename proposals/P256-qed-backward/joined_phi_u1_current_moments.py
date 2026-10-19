"""NEW exact initial CAR/Fourier-Gaussian current moments; no evolved Slater."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_body_word as body

ROOT = Path(__file__).resolve().parent
G = 64.0
S = 2.0
DERIVATION = {
 'convention': 'hbar=1, lattice/body energy units; J_mu=-partial_mu H, C_mu_nu=partial_mu partial_nu H. HE=64 sum Pi^2; HC=64 rho Lplus rho (no half). R_mu_nu(t)=-i <[J_mu(t),J_nu(0)]>; R(0)=-i commutator, Rprime(0)=F=<[[H,J_mu],J_nu]>. Nonstationary two-time response, not a stationary spectral kernel.',
 'preparation': 'Same fullN4 meanPhi, ordered scalarU1I4 one/two-link AP paths, active128+63 auxiliary128 fixed negative-mean Slaters, shared real129 Gaussian densityvariance2. Ref4 background. E0=.5 sum4 c0alpha dagger Ralpha. Tracing orthogonal Ref3 holes gives raw source sum_alpha .25 |v_alpha|^2 Slater(P+v_alpha v_alpha dagger/|v_alpha|^2), v=(I-P)e0alpha. This is an exact initial CAR identity ONLY, not an interacting-Slater evolution. Refhole charge -delta0 retained per channel.',
 'wick': 'For fixed initial projectors, <dGamma(A)dGamma(B)>=Tr(AP)Tr(BP)+Tr(A(I-P)BP). Different flavours factor at fixed Theta, not after averaging. All63 independent auxiliary copies give63 connected terms and square63 disconnected terms. Raw source weights retained.',
 'fourier': 'Every microscopic block carries signed actual path incidence k and q=Q^T k. Exact Gaussian expectation of any product is exp(-S |sum q|^2/2), with a SINGLE sharedTheta for all flavours. Finite double/triple Fourier sums use all paths and all129 quotient coordinates; floating implementation is not an interval certificate.',
 'electric': 'On real Gaussian amplitude exp(-Theta^2/(4S)), integration by parts gives <[[64Pi^2,J_mu],J_nu]>=-64 sum_a <{partial_a J_mu,partial_a J_nu}>. The DeltaJ and [partialJ,J]Pi terms cancel only after exact integration by parts; Pi contribution is NOT discarded. Gaussian Schwartz domain justifies all terms.',
 'coulomb': 'For hopping A_r with [rho_x,A_r]=delta_rx A_r, [[HC,A_r],A_s]=64 [A_r,A_s](2 delta_r L rho+delta_r L delta_r)+128(delta_r L delta_s) A_r A_s. Cov(dGamma(C),rho_v)=Tr(C(I-P)diag(v)P), summed over every charged flavour; source meanrho includes its Refhole. This derives the higher CAR terms, no free-band replacement.',
 'matter': '[[dGamma(h),dGamma(A)],dGamma(B)]=dGamma([[h,A],B]). All ordered block chains and onsite blocks are included with exact triple Fourier characters; cross-flavour matter commutators vanish but electric/Coulomb cross-flavour products do NOT.',
 'ward': 'For full192 link coordinates, D k=delta_x-delta_y on each actual path. Thus dGamma([n_x,h])=-i sum_e D_xe partial_e h (with this native D convention). Physical projection Q has DQ=0. Contact is the derivative of this same identity; Refdensity commutes matter but remains in HC and the source boundary.',
 'limits': 'Derived exact identities conditional on pinned meanPhi/root/Clifford inputs. Numerical matrices are finite floating observations. No interval enclosure, bodyground, stationary kernel, pole/dispersion, current-transfer to compactPhi, commoncone, originalSOURCE admission or finite-time evolution claim.'}


def traceblock(a: np.ndarray, p: np.ndarray) -> complex:
    return complex(np.einsum('ij,ji->', a, p))


def geometry(data: dict[str, Any], sp: body.Species) -> dict[str, Any]:
    paths = body.joined.old.native_paths(sp.profile, data['native'])
    inc = np.zeros((len(paths), 192))
    for r, path in enumerate(paths):
        np.add.at(inc[r], np.asarray(path['edges'], int), 1.)
    k = np.r_[inc, -inc]
    x = np.r_[sp.x, sp.y]
    y = np.r_[sp.y, sp.x]
    q = k @ data['native']['Q']
    delta = np.eye(64)[x]-np.eye(64)[y]
    ward = data['native']['D']@k.T - delta.T
    body.require(float(np.max(np.abs(ward))))
    return dict(k=k, q=q, x=x, y=y, delta=delta,
                ward_defect=float(np.max(np.abs(ward))))


def blocks(sp: body.Species) -> np.ndarray:
    return np.concatenate((sp.hops, sp.hops.conj().transpose(0, 2, 1)))


def pair(spblocks: np.ndarray, p: np.ndarray, geo: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x, y = geo['x'], geo['y']
    pb = p.reshape(64, 4, 64, 4).transpose(0, 2, 1, 3)
    mean = np.einsum('rij,rji->r', spblocks, pb[y, x])
    cov = np.empty((len(x), len(x)), complex)
    for r in range(len(x)):
        subtract = np.einsum('ij,sjk,skl,sli->s', spblocks[r], pb[y[r], x], spblocks, pb[y, x[r]], optimize=True)
        direct = np.einsum('ij,sjk,ski->s', spblocks[r], spblocks, pb[y, x[r]], optimize=True)
        cov[r] = (y[r] == x)*direct-subtract
    return mean, cov


def conditional(p: np.ndarray, aux: np.ndarray, bs: list[np.ndarray], geo: dict[str, Any], lp: np.ndarray,
                refhole: bool, load: int = 63) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ma, ca = pair(bs[0], p, geo)
    mb, cb = pair(bs[1], aux, geo)
    mean = ma+load*mb
    second = np.outer(mean, mean)+ca+load*cb
    density = np.einsum('xixi->x', p.reshape(64, 4, 64, 4)).real
    density += load*np.einsum('xixi->x', aux.reshape(64, 4, 64, 4)).real-2*(1+load)
    if refhole:
        density[0] -= 1
    v = 2*geo['delta']@lp
    fmean = v@density + np.einsum('ri,ri->r', geo['delta']@lp, geo['delta'])
    hc = 2*(geo['delta']@lp@geo['delta'].T)*second
    x, y = geo['x'], geo['y']
    for b, pp, multiplicity in ((bs[0], p, 1), (bs[1], aux, load)):
        pb = pp.reshape(64, 4, 64, 4).transpose(0, 2, 1, 3)
        for r in range(len(x)):
            # Tr(C (I-P)diag(v_r)P) evaluates all64 density sites.
            weighted = (np.repeat(v[r], 4)[:, None]*pp)
            t = weighted-pp@weighted
            tb = t.reshape(64, 4, 64, 4).transpose(0, 2, 1, 3)
            idx = np.flatnonzero((y[r] == x)|(y == x[r]))
            for s in idx:
                cmean = 0j
                ccov = 0j
                if y[r] == x[s]:
                    c = b[r]@b[s]
                    cmean += traceblock(c, pb[y[s], x[r]])
                    ccov += traceblock(c, tb[y[s], x[r]])
                if y[s] == x[r]:
                    c = b[s]@b[r]
                    cmean -= traceblock(c, pb[y[r], x[s]])
                    ccov -= traceblock(c, tb[y[r], x[s]])
                hc[r, s] += multiplicity*(cmean*fmean[r]+ccov)
    return mean, second, G*hc


def matter_edges(sp: body.Species, effective: np.ndarray, geo: dict[str, Any]) -> np.ndarray:
    # Sparse ordered matrix multiplication: endpoints force finite chains.
    b = np.concatenate((blocks(sp), np.broadcast_to(sp.onsite, (64, 4, 4))))
    x = np.r_[geo['x'], np.arange(64)]
    y = np.r_[geo['y'], np.arange(64)]
    k = np.r_[geo['k'], np.zeros((64, 192))]
    q = np.r_[geo['q'], np.zeros((64, 129))]
    nz = [np.flatnonzero(row) for row in k]
    outgoing = [np.flatnonzero(x == site) for site in range(64)]
    pb = effective.reshape(64, 4, 64, 4).transpose(0, 2, 1, 3)
    out = np.zeros((192, 192), complex)
    for a in range(len(x)):
        for bb in outgoing[y[a]]:
            ab = b[a]@b[bb]
            for c in outgoing[y[bb]]:
                qq = q[a]+q[bb]+q[c]
                z = traceblock(ab@b[c], pb[y[c], x[a]])*np.exp(-S*(qq@qq)/2)
                for r, s, sign in ((bb, c, -1), (a, c, 1), (c, a, 1), (bb, a, -1)):
                    if len(nz[r]) and len(nz[s]):
                        out[np.ix_(nz[r], nz[s])] += sign*z*np.outer(k[r, nz[r]], k[s, nz[s]])
    return out


def matrices(mean: np.ndarray, second: np.ndarray, hc: np.ndarray, matter: np.ndarray,
             mass: float, geo: dict[str, Any], qbasis: np.ndarray) -> dict[str, np.ndarray]:
    k, q = geo['k'], geo['q']
    chars = np.exp(-S*np.sum((q[:, None]+q[None, :])**2, axis=2)/2)
    onechar = np.exp(-S*np.sum(q*q, axis=1)/2)
    jm = -1j*k.T@(onechar*mean)
    contact = -k.T@((onechar*mean)[:, None]*k)
    jj = -k.T@(chars*second)@k
    pi = -G*k.T@((q@q.T)*chars*(second+second.T))@k
    coul = -k.T@(chars*hc)@k
    edge = dict(current=jm, contact=contact, second=jj, covariance=jj-np.outer(jm, jm)/mass,
                commutator=jj-jj.T, F_Pi=pi, F_HC=coul, F_matter=matter,
                F=pi+coul+matter)
    out = {}
    for name, value in edge.items():
        out[name] = qbasis.T@value if value.ndim == 1 else qbasis.T@value@qbasis
    out['retarded_initial'] = -1j*out['commutator']
    out['retarded_first_moment'] = out['F']
    # Full192 Ward consumer arrays are retained, not just physical entries.
    out.update({'edge_'+name: value for name, value in edge.items()})
    return out


def run(output: Path) -> dict[str, Any]:
    start = time.monotonic()
    data, species, inputs, _ = body.full_inputs()
    geo = geometry(data, species[0])
    qbasis = data['native']['Q']
    bs = [blocks(sp) for sp in species]
    p, aux = [sp.p for sp in species]
    lp = data['native']['Lplus']
    mean, second, hc = conditional(p, aux, bs, geo, lp, False)
    print('background contractions complete', flush=True)
    mb = matter_edges(species[0], p, geo)+63*matter_edges(species[1], aux, geo)
    bg = matrices(mean, second, hc, mb, 1., geo, qbasis)
    empty = np.eye(256)-p
    srcmean = np.zeros_like(mean)
    srcsecond = np.zeros_like(second)
    srchc = np.zeros_like(hc)
    effective = np.zeros_like(p)
    weights = []
    for alpha in range(4):
        v = empty[:, alpha]
        norm = float(np.vdot(v, v).real)
        weight = norm/4
        pp = p+np.outer(v, v.conj())/norm
        mm, ss, cc = conditional(pp, aux, bs, geo, lp, True)
        srcmean += weight*mm
        srcsecond += weight*ss
        srchc += weight*cc
        effective += weight*pp
        weights.append(weight)
        print('Ref channel complete', alpha, weight, flush=True)
    mass = sum(weights)
    body.require(abs(mass-.5))
    ms = matter_edges(species[0], effective, geo)+mass*63*matter_edges(species[1], aux, geo)
    src = matrices(srcmean, srcsecond, srchc, ms, mass, geo, qbasis)
    arrays = {state+'_'+name: value for state, mat in (('background', bg), ('raw_source', src)) for name, value in mat.items()}
    # Wrong controls are independent calculations/actual dropped contributions.
    wrongmean, wrongsecond, wronghc = conditional(p, aux, bs, geo, lp, False, load=1)
    wrong = matrices(wrongmean, wrongsecond, wronghc, mb, 1., geo, qbasis)
    controls = {'Pi_drop_F_frobenius': float(np.linalg.norm(bg['F_Pi'])),
                'HC_drop_F_frobenius': float(np.linalg.norm(bg['F_HC'])),
                'omitted_contact_frobenius': float(np.linalg.norm(bg['contact'])),
                'wrong_aux_load_covariance_frobenius': float(np.linalg.norm(bg['covariance']-wrong['covariance'])),
                'wrong_aux_load_HC_frobenius': float(np.linalg.norm(bg['F_HC']-wrong['F_HC']))}
    # Omitted two-link response contact: same actual path mean, removing length2.
    mask = np.r_[species[0].lengths == 2, species[0].lengths == 2]
    char = np.exp(-S*np.sum(geo['q']**2, axis=1)/2)
    two = -geo['q'][mask].T@((char[mask]*mean[mask])[:, None]*geo['q'][mask])
    controls['two_link_drop_contact_frobenius'] = float(np.linalg.norm(two))
    # SharedTheta vs independent copies has a direct positive second-moment discriminator.
    ma, _ = pair(bs[0], p, geo)
    mbmean, _ = pair(bs[1], aux, geo)
    chars = np.exp(-S*np.sum((geo['q'][:, None]+geo['q'][None, :])**2, axis=2)/2)
    disconnected = np.outer(ma+63*mbmean, ma+63*mbmean)-np.outer(ma, ma)-63*np.outer(mbmean, mbmean)
    controls['independent_Theta_second_defect'] = float(np.linalg.norm(geo['q'].T@((chars-np.outer(char, char))*disconnected)@geo['q']))
    for name, value in controls.items():
        if value < 1e-8 and name != 'two_link_drop_contact_frobenius':
            raise ArithmeticError('Nontrivial wrong control blind: '+name)
    capsule = output.with_suffix('.npz')
    np.savez_compressed(capsule, **dict[str, Any](arrays), source_channel_weights=np.asarray(weights), Q=qbasis)
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in {**body.CAPTURED, **inputs, Path(__file__).name: Path(__file__).read_bytes()}.items()}
    result = dict(law=DERIVATION, status='NEW full all129 initial current/contact and first retarded moments; no adoption',
                  input_hashes=hashes, matrix_capsule=dict(path=str(capsule), sha256=hashlib.sha256(capsule.read_bytes()).hexdigest(), arrays={name:list(a.shape) for name,a in arrays.items()}),
                  coordinates=129, full_link_coordinates=192, charged_flavours=64, auxiliary_load=63,
                  source_raw_norm_squared=mass, source_four_channel_weights=weights, controls=controls,
                  retained_blind_probes=dict(two_link_initial_contact_observed_blind=bool(controls['two_link_drop_contact_frobenius'] < 1e-8),
                     independent_positive_discriminator='joined_phi_u1_current_controls_receipt.json actual_fullN4.two_link_operator_jet_drop_discriminator; actual all129 source/operator first/second jet columns retained',
                     parent_source_first_moment_two_link_cross_character='BLIND, retained; primitive-character and nonzeroTheta microscopic jet controls discriminate'),
                  matrix_summaries={state:{name:dict(frobenius=float(np.linalg.norm(value)), max_abs=float(np.max(np.abs(value)))) for name,value in mat.items() if not name.startswith('edge_')} for state,mat in (('background',bg),('raw_source',src))},
                  Ward_operator_path_defect=geo['ward_defect'],
                  conditional=['Pinned exact root/Clifford/meanPhi supplier truth; body.full_inputs hashes; finite floating matrix arithmetic not interval enclosed'],
                  remaining_debts=['Nonstationary finite-time current kernel and remainder bounds beyond first derivative','Dressed photon pole/dispersion and integrability','Joined compactPhi ground/source admission and current transfer','Commoncone/electron photon explanatory closure; originalSOURCE and CURRENT427 unchanged'],
                  runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__), elapsed_seconds=time.monotonic()-start)
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run(args.output)
    print(json.dumps(dict(status=result['status'], controls=result['controls'], elapsed_seconds=result['elapsed_seconds'])), flush=True)
