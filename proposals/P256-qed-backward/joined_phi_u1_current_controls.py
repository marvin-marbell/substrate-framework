"""Independent finite-CAR/Pi oracle and actual fullN4 source/operator jets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from itertools import combinations
from typing import Any
import numpy as np
from numpy.polynomial.hermite import hermgauss
from threadpoolctl import threadpool_limits
import joined_phi_u1_current_moments as current
import joined_phi_u1_body_word as body


def comm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a@b-b@a


def creators(n: int) -> list[np.ndarray]:
    out = []
    for i in range(n):
        a = np.zeros((2**n, 2**n), complex)
        for mask in range(2**n):
            if not mask & (1 << i):
                a[mask | (1 << i), mask] = (-1)**((mask & ((1 << i)-1)).bit_count())
        out.append(a)
    return out


def lift(a: np.ndarray, cs: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(cs[0])
    for i in range(len(cs)):
        for j in range(len(cs)):
            if a[i, j] != 0:
                out += a[i, j]*(cs[i]@cs[j].conj().T)
    return out


def finite_oracle() -> dict[str, float]:
    rng = np.random.default_rng(222129)
    t = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    a = np.zeros((8, 8), complex)
    a[:4, 4:] = t
    z = np.diag(np.r_[np.ones(4), -np.ones(4)])*.7
    h0 = a+a.conj().T+z
    _, u = np.linalg.eigh(h0)
    w = u[:, :4]
    p = w@w.conj().T
    cs = creators(8)
    psi = np.zeros(256, complex)
    for ids in combinations(range(8), 4):
        psi[sum(1 << i for i in ids)] = np.linalg.det(w[list(ids)])
    # Orthogonal Ref3 holes, no coherent collapse across labels.
    sources = [.5*c@psi for c in cs[:4]]
    n0 = sum((c@c.conj().T for c in cs[:4]), np.zeros((256, 256), complex))
    n1 = sum((c@c.conj().T for c in cs[4:]), np.zeros((256, 256), complex))
    eye = np.eye(256)
    rho0, rho1 = n0-2*eye, n1-2*eye
    hcs = 64*(rho0-eye-rho1)@(rho0-eye-rho1)/4
    source_rho_defect = max(float(np.linalg.norm((rho0-eye)@c-c@rho0)) for c in cs[:4])
    omitted_ref_defect = max(float(np.linalg.norm(rho0@c-c@rho0)) for c in cs[:4])
    q = .81
    nodes, weights = hermgauss(80)
    theta = np.sqrt(2*current.S)*nodes
    weights = weights/np.sqrt(np.pi)
    fa = lift(a, cs)
    fz = lift(z, cs)
    errors = dict(wick=0., source_first=0., source_second=0., source_HC=0., Pi=0., matter=0.)
    pi_original = 0j
    pi_reduced = 0j
    matter_direct = 0j
    matter_fourier = 0j
    for th, weight in zip(theta, weights):
        phase = np.exp(1j*q*th)
        j = -1j*q*(a*phase-a.conj().T/phase)
        fj = -1j*q*(fa*phase-fa.conj().T/phase)
        fdj = q*q*(fa*phase+fa.conj().T/phase)
        fddj = -q*q*fj
        def exp(op: np.ndarray) -> complex:
            return complex(np.vdot(psi, op@psi))
        mean = np.trace(j@p)
        second = mean*mean+np.trace(j@(np.eye(8)-p)@j@p)
        errors['wick'] = max(errors['wick'], float(abs(second-exp(fj@fj))))
        rawmean = 0j
        rawsecond = 0j
        rawhc = 0j
        for alpha in range(4):
            v = (np.eye(8)-p)[:, alpha]
            norm = float(np.vdot(v, v).real)
            pp = p+np.outer(v, v.conj())/norm
            mass = norm/4
            rawmean += mass*np.trace(j@pp)
            rawsecond += mass*(np.trace(j@pp)**2+np.trace(j@(np.eye(8)-pp)@j@pp))
            # General quadratic-density double commutator, with Refhole.
            rhos = [np.diag(np.r_[np.ones(4),np.zeros(4)]), np.diag(np.r_[np.zeros(4),np.ones(4)])]
            lp = np.array([[1.,-1.],[-1.,1.]])/4
            terms = [(a*phase,np.array([1.,-1.])),(a.conj().T/phase,np.array([-1.,1.]))]
            means = np.asarray([np.trace(rr@pp).real-2 for rr in rhos]);means[0] -= 1
            for ri,(ar,delta) in enumerate(terms):
                for si,(ass,ds) in enumerate(terms):
                    crs = comm(ar,ass)
                    vv = 2*delta@lp
                    dd = vv[0]*rhos[0]+vv[1]*rhos[1]
                    ef = np.trace(crs@pp)*(vv@means+delta@lp@delta)+np.trace(crs@(np.eye(8)-pp)@dd@pp)
                    eab = np.trace(ar@pp)*np.trace(ass@pp)+np.trace(ar@(np.eye(8)-pp)@ass@pp)
                    coef = (-1j*q if ri==0 else 1j*q)*(-1j*q if si==0 else 1j*q)
                    rawhc += mass*64*coef*(ef+2*(delta@lp@ds)*eab)
        exactmean = sum(np.vdot(s,fj@s) for s in sources)
        exactsecond = sum(np.vdot(s,fj@fj@s) for s in sources)
        exacthc = sum(np.vdot(s,comm(comm(hcs,fj),fj)@s) for s in sources)
        errors['source_first'] = max(errors['source_first'], float(abs(rawmean-exactmean)))
        errors['source_second'] = max(errors['source_second'], float(abs(rawsecond-exactsecond)))
        errors['source_HC'] = max(errors['source_HC'], float(abs(rawhc-exacthc)))
        # Direct differential commutator acts on real Gaussian amplitude.
        original = -64*comm(fddj,fj)+64*(th/current.S)*comm(fdj,fj)-128*fdj@fdj
        pi_original += weight*exp(original)
        pi_reduced += weight*(-128*exp(fdj@fdj))
        matter_direct += weight*exp(comm(comm(fa*phase+fa.conj().T/phase+fz,fj),fj))
    # Independent exact Fourier triple sum, no quadrature supplier.
    hterms = [(a,q),(a.conj().T,-q),(z,0.)]
    jterms = [(-1j*q*a,q),(1j*q*a.conj().T,-q)]
    for h, qh in hterms:
        for j, qj in jterms:
            for k, qk in jterms:
                matter_fourier += np.trace(comm(comm(h,j),k)@p)*np.exp(-current.S*(qh+qj+qk)**2/2)
    errors['Pi'] = float(abs(pi_original-pi_reduced))
    errors['matter'] = float(abs(matter_direct-matter_fourier))
    for value in errors.values():
        body.require(value, 3e-8)
    return dict(**errors, source_density_intertwining=source_rho_defect,
                wrong_omitted_Ref_density_defect=omitted_ref_defect,
                source_norm_squared=float(sum(np.vdot(s,s).real for s in sources)),
                electric_positive_discriminator=float(abs(pi_reduced)), matter_positive_discriminator=float(abs(matter_fourier)))


def actual_jets() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    data, species, inputs, _ = body.full_inputs()
    sp = species[0]
    geo = current.geometry(data, sp)
    rng = np.random.default_rng(427222)
    theta = rng.normal(size=129)*.17
    direction = rng.normal(size=129);direction /= np.linalg.norm(direction)
    qd = geo['q']@direction
    b = current.blocks(sp)
    phase = np.exp(1j*geo['q']@theta)
    x,y = geo['x'],geo['y']
    def assemble(values: np.ndarray) -> np.ndarray:
        out = np.zeros((256,256),complex)
        for r in range(len(x)):
            out[4*x[r]:4*x[r]+4,4*y[r]:4*y[r]+4] += values[r]*b[r]
        return out
    d1 = assemble(1j*qd*phase)
    d2 = assemble(-qd*qd*phase)
    eps = 1e-4
    h = sp.h(theta).toarray()
    hp = sp.h(theta+eps*direction).toarray()
    hm = sp.h(theta-eps*direction).toarray()
    error1 = float(np.linalg.norm((hp-hm)/(2*eps)-d1))
    error2 = float(np.linalg.norm((hp+hm-2*h)/eps**2-d2))
    body.require(error1,1e-7);body.require(error2,1e-5)
    empty = np.eye(256)-sp.p
    effective = np.zeros((256,256),complex)
    weights = []
    for alpha in range(4):
        v = empty[:,alpha];norm = float(np.vdot(v,v).real)
        weights.append(norm/4)
        effective += norm/4*sp.p+np.outer(v,v.conj())/4
    rawmass = sum(weights)
    # Source columns for every operator jet, raw fourRef E coefficient .5.
    # E is Theta-independent: dE=d2E=0; d(H E-E H) uses these actual columns.
    jets = np.zeros((129,256,4),complex)
    contact_source = np.zeros((129,129,256,4),complex)
    for r in range(len(x)):
        if y[r] == 0:
            jets[:,4*x[r]:4*x[r]+4,:] += .5j*phase[r]*geo['q'][r,:,None,None]*b[r]
            contact_source[:,:,4*x[r]:4*x[r]+4,:] -= .5*phase[r]*np.outer(geo['q'][r],geo['q'][r])[:,:,None,None]*b[r]
    omitted_two = assemble(1j*qd*phase*(np.r_[sp.lengths,sp.lengths] == 2))
    return dict(operator_directional_first_derivative_defect=error1,operator_directional_contact_defect=error2,
                two_link_operator_jet_drop_discriminator=float(np.linalg.norm(omitted_two)),
                wrong_generator_charge4_contact_discriminator=float(15*np.linalg.norm(d2)),
                raw_source_norm_squared=rawmass,wrong_single_Ref_channel_raw_norm=weights[0],
                collapsed_single_Slater_projector_defect=float(np.linalg.norm((effective/rawmass)@(effective/rawmass)-effective/rawmass)),
                source_jet_convention='dK=.5 sum cdag(dh e0alpha)Ralpha, d2K likewise; E derivative zero exactly. All129 and129x129 columns retained; source Refhole included in CAR oracle. These are operator/source jets at a retained nonzero blind-independent theta, not finite-time transfer.',
                factored_all_operator_jet_ABI='Both species, all768 signed microscopic hopping blocks: h_r=B_r exp(i q_r Theta), J_mu,r=-i q_rmu h_r, C_munu,r=-q_rmu q_rnu h_r, d_a J_mu,r=q_ra q_rmu h_r. Endpoints, q129, k192, blocks, onsite, lengths retained exactly as finite floating input representation. This is complete all129 operator jet data without allocating a redundant129x129x256x256 tensor; no selected-entry substitute.',
                input_hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in {**body.CAPTURED,**inputs}.items()}),dict(
                    source_first_operator_columns=jets, source_second_operator_columns=contact_source,
                    operator_signed_path_q129=geo['q'], operator_signed_path_k192=geo['k'],
                    operator_path_source=x, operator_path_target=y,
                    operator_path_lengths=np.r_[sp.lengths,sp.lengths],
                    operator_active_blocks=current.blocks(species[0]), operator_aux_blocks=current.blocks(species[1]),
                    operator_active_onsite=species[0].onsite, operator_aux_onsite=species[1].onsite,
                    retained_probe_theta=theta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    start = time.monotonic()
    with threadpool_limits(limits=1):
        oracle = finite_oracle()
        actual, arrays = actual_jets()
    capsule = args.output.with_suffix('.npz')
    np.savez_compressed(capsule, **dict[str, Any](arrays))
    result = dict(finite_CAR_oracle=oracle,actual_fullN4=actual,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  matrix_capsule=dict(path=str(capsule),sha256=hashlib.sha256(capsule.read_bytes()).hexdigest(),arrays={name:list(a.shape) for name,a in arrays.items()}),
                  evidence='Exact CAR/differential/Fourier identities; floating oracle with finite quadrature is not interval proof. No originalSOURCE admission.',
                  retained_blind_probe='Parent initial source first-moment wrong two-link-cross-character projection is BLIND; not discarded. Actual contact/jet/primitive-character controls provide independent positive discrimination.',
                  runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-start)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
