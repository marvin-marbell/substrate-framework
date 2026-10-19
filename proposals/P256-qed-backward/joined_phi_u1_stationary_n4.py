"""Exact real-cover Floquet descriptor and NEW correlated stationary Ritz probe.

The reduced probes are orthogonal fixed-body compressions, not full-Phi grounds.
All arithmetic observations are float64, separate from the derived domain law.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import sys
import time
import zipfile
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, diags, kron, eye
from scipy.sparse.linalg import eigsh, expm_multiply
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as old
import joined_phi_u1_finite_car as joined

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_stationary_n4'
G = [.28917434559291342, .10475219290309079, .10475219290309079, .10475219290309079]
PLAN = dict(id='issue-222-induced-photon-delta13-continuation-20261008', revision='3a1f07a384a7208fe0056b4d597a59b92bca6314d1746b9fa85a6b3fc7717721', template='general', template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe', work_item='induced-photon', requirements=['verification', 'outcome'])


def cycles(D: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Fundamental integer basis; the chord minor is exactly identity."""
    d = np.rint(D).astype(np.int64)
    if not np.array_equal(d, D):
        raise ArithmeticError('Not integer native incidence')
    roots = list(range(len(d)))
    def root(x: int) -> int:
        while roots[x] != x:
            x = roots[x]
        return x
    tree = []
    for e in range(d.shape[1]):
        x, y = np.flatnonzero(d[:, e])
        a, b = root(int(x)), root(int(y))
        if a != b:
            roots[a] = b
            tree.append(e)
    chords = [e for e in range(d.shape[1]) if e not in tree]
    c = np.zeros((d.shape[1], len(chords)), dtype=np.int64)
    c[chords, np.arange(len(chords))] = 1
    c[tree] = np.rint(np.linalg.solve(d[1:, tree], -d[1:, chords])).astype(np.int64)
    if len(tree) != 63 or np.any(d@c):
        raise ArithmeticError('Integer cycle derivation failed')
    return c, chords


def native_first_reference(data: dict[str, Any], c: np.ndarray) -> dict[str, Any]:
    """All actual charged paths acting on onsite sea, with exact CAR PH blocks.

    Dictionary characters C^T k identify projected momenta without float hashes.
    The three-hop moment includes surviving 1+1-2 path interference.
    """
    native = data['native']; pt = native['Q']@native['Q'].T
    f = native['frame']; _, u = np.linalg.eigh(f[0]); w, z = u[:, :2], u[:, 2:]
    rows = []; total_energy = total_var = total_m3 = 0.
    for profile in data['profiles'][1:]:
        mass = float(profile['onsite'])*np.sqrt(native['g0'])
        amps: dict[tuple[Any, ...], np.ndarray] = {}
        signed = []
        for path in old.native_paths(profile, native):
            k = np.zeros(192, dtype=np.int64); k[list(path['edges'])] = 1
            char = c.T@k
            b = path['seam']*np.einsum('m,mij->ij', path['coefficients'], f)
            for x, y, ch, kk, block in ((path['source'], path['target'], char, k, b), (path['target'], path['source'], -char, -k, b.conj().T)):
                # h_xy c_x^dag c_y: electron x, hole y.
                key = (int(y), int(x), tuple(ch.tolist()))
                amps[key] = amps.get(key, np.zeros((2, 2), complex))+z.conj().T@block@w
                signed.append((int(x), int(y), ch, kk, block))
        h0moment = variance = 0.
        for (hole, electron, char), a in amps.items():
            if np.linalg.norm(a) < 1e-14:
                continue
            # Obtain p from its actual path rather than invert a period metric.
            match = next(v for v in signed if v[0] == electron and v[1] == hole and tuple(v[2]) == char)
            k = match[3]; delta = np.eye(64)[electron]-np.eye(64)[hole]
            gap = 2*mass+64*float(k@pt@k)+64*float(delta@native['Lplus']@delta)
            weight = float(np.vdot(a, a).real)
            variance += weight; h0moment += gap*weight
        cubic = 0j
        for (hole, electron, char), a in amps.items():
            ch = np.asarray(char)
            for x, y, q, _, b in signed:
                if y == electron:
                    other = amps.get((hole, x, tuple((ch+q).tolist())))
                    if other is not None:
                        cubic += np.vdot(other, (z.conj().T@b@z)@a)
                if x == hole:
                    other = amps.get((y, electron, tuple((ch+q).tolist())))
                    if other is not None:
                        cubic -= np.vdot(other, a@(w.conj().T@b@w))
        if abs(cubic.imag) > 1e-7:
            raise ArithmeticError('Three-hop moment not real')
        load = profile['load']; energy = -128*mass*load
        total_energy += energy; total_var += load*variance
        total_m3 += load*(h0moment+cubic.real)
        rows.append(dict(role=profile['role'], load=load, onsite_mass=mass, energy=energy, residual_squared_per_copy=variance, diagonal_third_moment_per_copy=h0moment, actual_interfering_hopping_third_moment_per_copy=cubic.real, first_excitation_count=len(amps)))
    diagonal = total_m3/total_var
    ritz = np.linalg.eigh(np.array([[0., np.sqrt(total_var)], [np.sqrt(total_var), diagonal]]))
    electric_gap_lower = 64/float(np.linalg.eigvalsh(c.T@c)[-1])
    onsite_gap = 2*min(row['onsite_mass'] for row in rows)
    reference_gap = dict(lower=min(electric_gap_lower, onsite_gap), upper=min(onsite_gap, 64*float(pt[0,0])), onsite_particle_hole_gap=onsite_gap, electric_gap_lower=electric_gap_lower, kind='H0 Bloch0 gap enclosure, not full interacting gap', derivation='For nonzero reciprocal n=C^T p,1<=|n|²<=lambda_max(C^TC)|p|²; HE>=64/lambda_max. Any onsite fixedN excitation costs>=2minmass and HC>=0. Upper from onsite same-site PH rho0 or projected single-edge Fourier character with unchanged sea.')
    return dict(onsite_energy=total_energy, HE=0., HC=0., all_charged_rho_exactly_zero=True, reference_H0_gap_enclosure=reference_gap, reference_residual=np.sqrt(total_var), residual_squared=total_var, species=rows, two_vector_Ritz_shift=float(ritz[0][0]), two_vector_Ritz_energy=total_energy+float(ritz[0][0]), H_in_reference_residual_basis=[[total_energy, np.sqrt(total_var)], [np.sqrt(total_var), total_energy+diagonal]], definition='ref=all64 onsite negativeGamma0 occupancy2; Bloch0 Pi0; Ref4. r=(Hbody-Eref)ref; span(ref,r) retains all129 electric modes, all1+63 charged copies and native one/two/AP. HC on each PH is64 deltaLplusdelta; Hopping contribution to <r,(H-Eref)r> evaluated with electron and hole CAR signs, including surviving1+1-2 interference. Variational upper bound on fixed-body Bloch0 energy, not fullPhi unless a separate state transfer is proved.')


def fock(count: int) -> tuple[list[int], np.ndarray]:
    bits = [sum(1 << j for j in comb) for comb in combinations(range(6), count)]
    occ = np.asarray([[(b >> j)&1 for j in range(6)] for b in bits])
    return bits, occ


def car(bits: list[int], a: int, b: int) -> Any:
    lookup = {v: i for i, v in enumerate(bits)}; rows=[]; cols=[]; values=[]
    for j, v in enumerate(bits):
        if not (v >> b)&1:
            continue
        sign = (-1)**((v & ((1 << b)-1)).bit_count()); t = v ^ (1 << b)
        if (t >> a)&1:
            continue
        sign *= (-1)**((t & ((1 << a)-1)).bit_count()); t |= 1 << a
        rows.append(lookup[t]); cols.append(j); values.append(sign)
    return coo_matrix((values, (rows, cols)), shape=(len(bits), len(bits))).tocsr()


def compressed(data: dict[str, Any], radius: int, source: bool, variant: str = 'fullHC') -> tuple[Any, dict[str, Any], np.ndarray]:
    native = data['native']; f = native['frame']; _, u = np.linalg.eigh(f[0])
    v = u[:, 0]; positive = f[1]@v; positive /= np.linalg.norm(positive)
    spin = np.column_stack((v, positive))
    sites = [0, 16, 32]; edges = [0, 48]
    q = native['Q'][edges]; metric = q@q.T
    abits, ao = fock(4 if source else 3); bbits, bo = fock(3)
    na, nb = len(abits), len(bbits); nf = na*nb
    ns = np.asarray(list(product(range(-radius, radius+1), repeat=2)), dtype=int)
    nlookup = {tuple(n): j for j, n in enumerate(ns)}
    oa = ao.reshape(na, 3, 2).sum(axis=2); ob = bo.reshape(nb, 3, 2).sum(axis=2)
    rho = oa[:, None, :]+ob[None, :, :]-2
    refhole = np.array([1., 0., 0.]) if source else np.zeros(3)
    rho = rho-refhole
    lp = native['Lplus'][np.ix_(sites, sites)]
    hc = 64*np.einsum('abi,ij,abj->ab', rho, lp, rho)
    cross = 128*np.einsum('ai,ij,bj->ab', oa-1-refhole, lp, ob-1)
    if variant == 'omitHC':
        hc = np.zeros_like(hc)
    elif variant == 'omitCross':
        hc = hc-cross
    diag = hc.copy()
    onsite_reference = 0.
    for flavor, profile in enumerate(data['profiles'][1:]):
        m = float(profile['onsite'])*np.sqrt(native['g0'])
        onsite_reference += -128*m*profile['load']
        occupations = ao if flavor == 0 else bo
        diff = m*(occupations[:, 1::2].sum(axis=1)-occupations[:, 0::2].sum(axis=1)+3)
        diag += diff[:, None] if flavor == 0 else diff[None, :]
    ep = 64*np.einsum('ni,ij,nj->n', ns, metric, ns)
    h: Any = diags((ep[:, None]+diag.reshape(1, -1)).ravel(), format='csr')
    kept=[]
    for flavor, profile in enumerate(data['profiles'][1:]):
        for path in old.native_paths(profile, native):
            if path['source'] not in sites or path['target'] not in sites:
                continue
            incidence = np.zeros(192); incidence[list(path['edges'])] = 1
            qq = incidence@native['Q']
            coeff = np.linalg.solve(metric, q@qq)
            shift = np.rint(coeff).astype(int)
            if np.linalg.norm(shift@q-qq) > 1e-9:
                # Full native path has zero projection onto our Fourier subspace.
                continue
            x, y = sites.index(path['source']), sites.index(path['target'])
            block = path['seam']*spin.conj().T@np.einsum('m,mij->ij', path['coefficients'], f)@spin
            rows=[]; cols=[]
            for j, n in enumerate(ns):
                i = nlookup.get(tuple(n+shift))
                if i is not None:
                    rows.append(i); cols.append(j)
            shiftmat = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ns), len(ns))).tocsr()
            cm = sum((block[a,b]*car(abits if flavor == 0 else bbits, 2*x+a, 2*y+b) for a in range(2) for b in range(2)), start=diags(np.zeros(na if flavor == 0 else nb), format='csr'))
            cm = kron(cm, eye(nb), format='csr') if flavor == 0 else kron(eye(na), cm, format='csr')
            term = kron(shiftmat, cm, format='csr'); h = h+term+term.conj().T
            kept.append(dict(role=profile['role'], source=path['source'], target=path['target'], length=path['length'], seam=path['seam'], edges=list(path['edges']), Fourier_shift=shift.tolist(), block=dict(real=block.real.tolist(), imag=block.imag.tolist())))
    refa = 0b010101 | (0b10 if source else 0); refb = 0b010101
    initial = np.zeros(h.shape[0]); initial[nlookup[(0,0)]*nf+abits.index(refa)*nb+bbits.index(refb)] = 1.
    info = dict(dimension=h.shape[0], Fourier_radius=radius, source_Ref=3 if source else 4, active_count=4 if source else 3, aux_dynamic_copies=1, aux_spectator_copies=62, sites=sites, mode_edges=edges, Fourier_metric=metric.tolist(), retained_native_paths=kept, total_onsite_reference_energy=onsite_reference, source_refhole=refhole.tolist(), variant=variant, domain='Fixed body Phi=frame,U=I; one Gamma0-/Gamma1-connected+ channel per site/species dynamic, orthogonal negative channel occupied spectator. Allother sites and62aux copies onsite occupancy2. Exact orthogonal compression onto two projected-edge character Fourier sublattice |n1|,|n2|<=R. Complementary two-link route has distinct character and projects to zero; not deleted from microscopic law.', rho=rho.reshape(nf,3).tolist(), hc_diagonal=hc.ravel().tolist(), cross_diagonal=cross.ravel().tolist())
    return h, info, initial


def probe(data: dict[str, Any], radius: int, source: bool, variant: str = 'fullHC') -> dict[str, Any]:
    h, info, initial = compressed(data, radius, source, variant)
    seed = 2221010
    rng = np.random.default_rng(seed)
    all_state_start = rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0])
    # SciPy's unannotated tol=0 infers int; its documented API accepts float.
    solver_options: dict[str, Any] = {'tol': 1e-10}
    vals, vecs = eigsh(h, k=3, which='SA', v0=all_state_start, **solver_options)
    info['spectral_solver'] = dict(method='scipy sparse Hermitian eigsh SA', seed=seed, start='Nonzero random complex components in every CAR/Fourier basis state, not reference-only cyclic sector.', scope='Finite float64 returned Ritz values, not interval-certified global operator spectrum.')
    order = np.argsort(vals); vals=vals[order]; vecs=vecs[:,order]
    ground = vecs[:,0]; e0=float(np.vdot(initial,h@initial).real)
    r0=float(np.linalg.norm(h@initial-e0*initial)); residual=float(np.linalg.norm(h@ground-vals[0]*ground))
    wider, _, _ = compressed(data, radius+1, source, variant)
    narrow_ns = list(product(range(-radius,radius+1),repeat=2))
    wide_ns = list(product(range(-radius-1,radius+2),repeat=2))
    wide_lookup = {n:i for i,n in enumerate(wide_ns)}
    nf = len(info['rho'])
    embedded = np.zeros(wider.shape[0],complex)
    for i,n in enumerate(narrow_ns):
        j = wide_lookup[n]
        embedded[j*nf:(j+1)*nf] = ground[i*nf:(i+1)*nf]
    wide_residual = float(np.linalg.norm(wider@embedded-vals[0]*embedded))
    info['Ritz_residual_in_next_Fourier_shell'] = wide_residual
    info['Fourier_boundary_leakage'] = float(np.sqrt(max(0.,wide_residual**2-residual**2)))
    info['leakage_scope'] = 'Actual R+1 action on zero-embedded Ritz vector, retaining same spins/sites; bounds no discarded spatial/spin/Phi sector.'
    probabilities = np.abs(ground.reshape(-1, len(info['rho'])))**2
    p=probabilities.sum(axis=0); rho=np.asarray(info.pop('rho')); hc=np.asarray(info.pop('hc_diagonal')); cross=np.asarray(info.pop('cross_diagonal'))
    filter_rows=[]
    for tau in (.002, .01, .05):
        state = expm_multiply(-tau*(h-vals[0]*eye(h.shape[0], format='csr')), initial)
        state /= np.linalg.norm(state); energy=float(np.vdot(state,h@state).real)
        filter_rows.append(dict(imaginary_time=tau, energy=energy, residual=float(np.linalg.norm(h@state-energy*state)), overlap_squared_with_Ritz_ground=float(abs(np.vdot(ground,state))**2)))
    return dict(**info, energy_shift=float(vals[0]), absolute_fixed_body_variational_energy=info['total_onsite_reference_energy']+float(vals[0]), first_three_eigenvalues=vals.tolist(), compressed_gap=float(vals[1]-vals[0]), reference_energy_shift=e0, reference_residual=r0, Ritz_residual_inside_compression=residual, meanrho=(p@rho).tolist(), rho_second=np.einsum('n,ni,nj->ij',p,rho,rho).tolist(), Coulomb_expectation=float(p@hc), active_aux_Coulomb_cross_expectation=float(p@cross), imaginary_time_filter=filter_rows)


def run(output: Path) -> dict[str, Any]:
    start=time.monotonic(); data=old.load_mean(g_override=G, mean_descriptor={'law':'NEW stationary onsite/fiber candidate, fixed root display; not frozen mean-Slater Gaussian'})
    native=data['native']; c,chords=cycles(native['D'])
    if (native['N'],native['V'],native['Q'].shape)!=(4,64,(192,129)):
        raise ArithmeticError('Requires current N4')
    periods=[]; theta=native['Q']@np.linspace(-.27,.193,129)
    fields=np.broadcast_to(native['frame'],(64,4,4,4)); links=np.broadcast_to(np.eye(4),(192,4,4))
    for profile in data['profiles'][1:]:
        base=joined.assemble(profile,native,fields,links,theta)
        errors=[float(np.max(abs(joined.assemble(profile,native,fields,links,theta+2*np.pi*c[:,j])-base))) for j in range(129)]
        wrong=float(np.linalg.norm(joined.assemble(profile,native,fields,links,theta+np.pi*c[:,0])-base))
        if max(errors)>1e-10 or wrong<1:
            raise ArithmeticError('Wrong period control blind')
        periods.append(dict(role=profile['role'], all129_period_defects=errors, wrong_pi_period_defect=wrong))
    fullref=native_first_reference(data,c)
    results=[probe(data,r,s) for s in (False,True) for r in (1,2,3)]
    controls=[probe(data,2,False,v) for v in ('omitHC','omitCross')]
    domain_name='induced_phi_compact_source_global_domain_control_corrected_receipt_20261010.json'
    domain_raw=(ROOT/domain_name).read_bytes(); domain=json.loads(domain_raw)
    inputs={domain_name:domain_raw}
    for key in ('root_receipt','full_receipt'):
        path=Path(data['descriptor'][key]['path']); inputs[path.name]=path.read_bytes()
    sources={name:raw for name,raw in old.CAPTURED.items()}
    sources[Path(joined.__file__).name]=Path(joined.__file__).read_bytes(); sources[Path(__file__).name]=Path(__file__).read_bytes()
    equations=dict(periods='S=kerD subset R192, D unscaled integer64x192; strict lattice L=2pi(kerD intersect Z192)=2pi C Z129. Chord minor identity proves saturation and rank129, independent of orthonormal Q. Period in Q coordinates2pi Q^T C.', reciprocal='p=Q(C^T Q)^-1 n, n integer129; equivalent C^T Q p=n. Boundary psi(theta+2pi Cj)=exp(2pi i beta_j)psi(theta); p(beta)=(C^T Q)^-1(n+beta). HE64|p(beta)|². beta is configuration quasimomentum, NOT spatial photon q.', Hilbert='Original global L2(R3840 CartPhi x SU4^192 x R129_EM) tensor fixedN CAR+Ref, compatible Gauss sector. Equivalent direct integral over beta of fiber L2(S/L) with Bloch boundary; does not replace physical real-cover Hilbert or compactify fields. 5760normalPhi+960Haar is local chart, not ambient Cartesian count.', H='Friedrichs closure of actual positive form plus relatively form-bounded linear matter: chi/2[8pCart²+(3/10)sum E_SU4²]+64Pi²+Lambda/a sum||Gram_x||F²+(1/5)Lambda^(5/4)alignment/8+(7/10)Lambda^(5/4)sumfaces(4-ReTrUface)+64rho_allcharged Lplusrho_allcharged+dGamma(-Lambda hneutral)+sumcharged dGamma(hcharged). chi2^-4096 Lambda2^8192; no bare EM B². Onsite and one/two/AP ordered endpoint hxy=(FxU+UFy)/2 exp(i sumedge theta). All charged scales1, active1m1/5 aux63m2.', domain='Smooth compact Cartesian support x smoothHaar x smooth Bloch periodic EM with finiteCAR/Gauss projection is form core. Quartic >=Lambda/(64aV)||Phi||HS^4; subtract Young linear matter bound; compact fiber embedding and compact resolvent for full operator if compatible Gauss sector nonempty. Pi/current/contact periodic coefficient differential operators are fiber-defined on common graph cores; energies and stationary state are not identified by static root.', algebra='Fiber-preserving algebra generated by lattice-periodic coefficient functions, Pi exponentials/translations, CAR neutral/gauge-compatible observables, Hamiltonian resolvent, periodic Noether current/contact and Wilson characters exp(i k_edge.Theta), k_edge integer (C^T k integer). Raw Theta multiplication not fiber-preserving; generic exp(i a.Theta) maps beta to beta+C^T a mod Z. Full CCR/Weyl algebra requires direct-integral coherent state or extension, cannot be represented as fixed-fiber rawTheta/Pi canonical pair.', ground_boundary='Semibounded compact-resolvent full fibers admit lowest eigenstates; actual fullPhi ground NOT computed. No normalized real-cover stationary ground follows only with independent absence-of-flat-band proof; not asserted by this finite calculation. Our onsite reference is exact H0=onsite+HE+HC eigenstate but NOT H eigenstate. Improved states computed by genuine sparse CAR Fourier diagonalization/imaginary-time and fullnative2vector Ritz. Neither finite compressed gap nor Blochbeta is a photon pole.')
    equations['reciprocal'] = 'Let p_Q=(C^T Q)^-1(n+beta), p_edge=Q p_Q. C^T p_edge=n+beta. Boundary psi(theta+2pi Cj)=exp(2pi i beta_j)psi(theta); HE=64|p_Q|²=64|p_edge|². beta configuration-space quasimomentum is NOT spatial photon q.'
    data['profiles'] = [{k:v for k,v in p.items() if k != 'model'} for p in data['profiles']]
    report=dict(plan=PLAN, exact_law=dict(N=4,V=64,a='1/64',UEphys=1,HE=64,HC=64,diffusion=128,Ref_background=4,Ref_source=3,charged_profiles=data['profiles'][1:]), fixed_mean_display=G, exact_rational_law_vs_binary_float='Declared1/5,7/10 source law retained as descriptor; arrays are binary64 observations of fixed-root frame. No new parameter adoption.', equations=equations, inherited_domain_proof=domain['proof'], graph=dict(incidence=native['D'].astype(int).tolist(),integer_cycle_basis=c.tolist(),chord_edges=chords,integer_divergence_defect=int(np.max(abs(native['D'].astype(int)@c))),rank=129,period_controls=periods), fullN4_onsite_reference_and_two_vector_variational=fullref, correlated_compressed_probes=results, wrong_law_controls=controls, interpretation='Every reduced result is fixed-body variational upper bound in its explicit Fourier/CAR subspace; spectral gap/residual only compressed operator. Full charged covariance uses sum then quadratic, actual Refhole is subtracted. No interacting Wick. Orthogonal complement spatial/spin/Phi leakage not bounded; source/background energy difference not charged particle mass or photon pole.', next_material_construction='Expand sparse Fourier/CAR Krylov in allnative path characters and both spin channels, preserve multi-flavour shared density, then bind a normalizable fullPhi/Haar fiber trial via an independently checked uniform T action on these new vectors. FullCCR extension and beta minimization remain separate debts.', input_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in inputs.items()}, source_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in sources.items()}, runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__), elapsed_seconds=time.monotonic()-start)
    # JSON-friendly profile fields include primitive complex path arrays.
    def packed(v: Any) -> Any:
        if isinstance(v,Fraction):return str(v)
        if isinstance(v,np.ndarray):
            return dict(real=v.real.tolist(),imag=v.imag.tolist()) if np.iscomplexobj(v) else v.tolist()
        if isinstance(v,complex):return dict(real=v.real,imag=v.imag)
        if isinstance(v,(np.integer,np.floating)):return v.item()
        if isinstance(v,dict):return {str(k):packed(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [packed(x) for x in v]
        return v
    report=packed(report)
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in {**inputs,**sources}.items():archive.writestr(name,raw)
    report['immutable_source_input_archive']=dict(path=str(output.with_suffix('.zip')),sha256=hashlib.sha256(output.with_suffix('.zip').read_bytes()).hexdigest())
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(reference=fullref, compressed=[{k:r[k] for k in ('dimension','Fourier_radius','source_Ref','energy_shift','compressed_gap','Ritz_residual_inside_compression','Coulomb_expectation','active_aux_Coulomb_cross_expectation')} for r in results], elapsed_seconds=report['elapsed_seconds']),indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.output)
