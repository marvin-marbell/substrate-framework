"""Outward signed ACTUAL12 physical Wilson kernels at unchanged 2^-36.

Independent interval Bloch/CAR contraction; no floating first-moment premise.
Run with the project Python and --output; natural-exit driver captures execution.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_body_word as body
import joined_phi_u1_current_moments as current
import joined_phi_u1_finite_current_kernel as prices
import joined_phi_u1_nonlocal_body_transport as transport

ROOT = Path(__file__).resolve().parent
IV: Any = mp.iv
IV.dps = 60
CLOCK = F(1, 2**36)
PLAN = 'issue-222-induced-photon-delta13-continuation-20261008'
REVISION = '7d1bacd0f61d2373186458d14953515e817c7b863732d858571b4db0736387a5'
PIN = {
 'induced_phi_current_n4_root_static_slow_certificate_actual_receipt.json': '79f49c752b4c2cc716d0291f1c38d448093a94f231412900be2c0e730c67dfee',
 'joined_phi_u1_nonlocal_body_transport_verified_source_transfer_receipt_20261010.json': '442299e80be00a54ce32cf9482b5c70040398871cddfecb06afe5adfea5fe894',
 'joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json': '2a1bbb46852707c82e477f28b5497e303b5ca66453db76d0655379f35cfb9a97',
 'joined_phi_u1_nonlocal_body_transport_receipt.json': '354738d91318fbb048182f809a10c9a44fdbf8536710f46634dcaa5bc38eb537',
 body.ARCHIVE: body.ARCHIVE_SHA,
}


def exact(x: Any) -> Any:
    q = F(x)
    return IV.mpf(q.numerator)/IV.mpf(q.denominator)


def endpoint(x: Any, upper: bool) -> F:
    sign, mantissa, exponent, _ = x._mpi_[int(upper)]
    return F((-1 if sign else 1)*mantissa)*F(2)**exponent


def enclosure(x: Any) -> dict[str, str]:
    return dict(lower=str(endpoint(x, False)), upper=str(endpoint(x, True)))


def complex_enclosure(x: Any) -> dict[str, Any]:
    return dict(real=enclosure(x.real), imaginary=enclosure(x.imag))


def contains_zero(x: Any) -> bool:
    return endpoint(x, False) <= 0 <= endpoint(x, True)


def interval_box(lo: F, hi: F) -> Any:
    return IV.mpf([exact(lo).a, exact(hi).b])


class Bloch:
    """Clifford-vector contraction of the literal full four-colour trace.

    u(z)=1/64 sum_AP exp(ik.z) h(k)/E(k), E=(1+u)/2.
    The exact identity Tr(gamma_mu gamma_nu)/4=delta_mu_nu
    reduces every FULL Ref4 trace, not a selected colour or hole.
    """
    def __init__(self, profile: dict[str, Any], g: list[Any]):
        self.profile = profile
        self.frame = [IV.sqrt(v) for v in g]
        self.bands: list[list[Any]] = []
        self.ks = list(product(range(4), repeat=3))
        self.gaps = []
        self.cache: dict[tuple[int, ...], list[Any]] = {}
        for ks in self.ks:
            c = [exact(profile['onsite']), exact(0), exact(0), exact(0)]
            for p in profile['paths']:
                angle = IV.pi*(2*ks[p['axis']]+1)*p['length']/4
                c[0] += 2*exact(p['c0'])*IV.cos(angle)
                c[p['axis']+1] -= 2*exact(p['ci'])*IV.sin(angle)
            energy = IV.sqrt(sum((g[m]*c[m]**2 for m in range(4)), exact(0)))
            if endpoint(energy, False) <= 0:
                raise ArithmeticError('Current-root AP Bloch gap not enclosed')
            self.gaps.append(enclosure(energy))
            self.bands.append([self.frame[m]*c[m]/energy for m in range(4)])

    def u(self, displacement: Any) -> list[Any]:
        key = tuple(int(v) for v in displacement)
        if key not in self.cache:
            values = [IV.mpc(0) for _ in range(4)]
            for ks, band in zip(self.ks, self.bands):
                angle = IV.pi*sum((2*ks[a]+1)*key[a] for a in range(3))/4
                phase = IV.mpc(IV.cos(angle), IV.sin(angle))
                for m in range(4):
                    values[m] += phase*band[m]/64
            self.cache[key] = values
        return self.cache[key]


def hop_vectors(profile: dict[str, Any], paths: list[dict[str, Any]], frame: list[Any]) -> list[list[Any]]:
    lookup = {(p['axis'], p['length']): p for p in profile['paths']}
    result = []
    for p in paths:
        spec = lookup[p['axis'], p['length']]
        v = [IV.mpc(0) for _ in range(4)]
        v[0] = p['seam']*exact(spec['c0'])*frame[0]
        v[p['axis']+1] = IV.j*p['seam']*exact(spec['ci'])*frame[p['axis']+1]
        result.append(v)
    return result+[[IV.mpc(z.real, -z.imag) for z in v] for v in result]


def dot(a: list[Any], b: list[Any]) -> Any:
    return sum((x*y for x, y in zip(a, b)), IV.mpc(0))


def moments(r: int, geo: dict[str, Any], geom: dict[str, Any], native: dict[str, Any], bloch: Bloch,
            hops: list[list[Any]], variance: int = 2, omit_second: bool = False) -> Any:
    kr = geo['k'][r].astype(np.int64)
    dr = geo['delta'][r].astype(np.int64)
    lint, den = geom['L_integer'], geom['denominator']
    sites = native['sites']
    x = int(geo['x'][r])
    total = IV.mpc(0)
    for h in np.flatnonzero(geo['x'] == x):
        if omit_second and int(geo['k'][h]@geo['k'][h]) == 2:
            continue
        k = geo['k'][h].astype(np.int64)-kr
        d = geo['delta'][h].astype(np.int64)-dr
        q2 = F(int(k@k))-F(int(d@lint@d), den)
        attenuation = IV.exp(-exact(F(variance, 2)*q2))
        j = int(geo['y'][h])
        total += attenuation*dot(hops[h], bloch.u(sites[j]-sites[0]))/2
    q2 = F(int(kr@kr))-F(int(dr@lint@dr), den)
    onsite = [exact(bloch.profile['onsite'])*bloch.frame[0], exact(0), exact(0), exact(0)]
    return total+IV.exp(-exact(F(variance, 2)*q2))*dot(onsite, bloch.u(sites[x]-sites[0]))/2


PROOF = {
 'orientation': 'C=<Ex UBpsi,US E0psi>, Cprime=-i<Kx UBpsi,US E0psi>. Thus Im C=-t Re m1 plus paid remainder. Ex=exp(iqTheta)*sum4 cdag_xalpha R0alpha/2. Unwhitened raw E0 norm squared=1/2, four orthogonal Ref holes.',
 'root': 'Current corrected root79f49 unique fixed-box root; enclose BOTH root coordinates centre +/- corrected radius, not just floating centre. Literal rational native profiles checked against root antecedents. Conditional preparation is static AP rank128 seas, NOT interacting Coulomb ground.',
 'clifford': 'Native Gaussian-integer gamma matrices are checked exactly for all anticommutators and all Tr gamma=0, Tr gamma_mu gamma_nu=4delta. Full4 colour CAR contraction Tr[h_xj(1-P)_j0]/4 is dot(hvec,uvec)/2, retaining both Phi endpoint frame factors, every AP seam and both signed links.',
 'gaussian': 'S=2 DENSITY variance; shared real129 Theta. Exact physical Gram is k.k-delta Lplus delta, using rational full63-mode pseudoinverse. Gaussian character exp(-S gram/2) includes coexact modes and all3 torons. Full Q is captured; no subset of129 or independent auxiliaryTheta.',
 'C0_Pi': 'Offsite Tr(1-P)_x0=0 exactly because Bloch P has only identity on diagonal and traceless Clifford vectors offsite. C0=0. Initial Gaussian Pi IBP cancellation is exact, not suppression of electric dynamics.',
 'HC': 'Full initial HC=64 exp(-q2)[(2deltaLmeanrho+deltaLdelta)b - .5sum_v deltaL_v Tr(P_xv(1-P)_v0)]. Ref4 offset and active1+63aux density give meanrho=0 exactly. Each diagonal colour trace is2; connected active covariance retained, auxiliary connected part factorizes against b=0. Reflection v->x-v makes Clifford two-vector trace symmetric (including AP signs) and deltaL antisymmetric. Therefore HC=0 exactly. Direct interval covariance sum encloses zero independently; no future Wick assumption.',
 'second_remainder': transport.DERIVATION['domain']+' '+transport.DERIVATION['conservation']+' '+transport.DERIVATION['second_price']+' '+transport.DERIVATION['remainder'].split(' Any finiteclock')[0],
 'coefficient_authority': 'Every coefficient upper is recomputed through rational_energies/rational_caps from rational Lplus and pinned native profiles; hmax=sum abs rational onsite+2sum(abs c0+abs ci) because root-box sqrt(gmu)<1. No float-derived operator envelope, oldK0, observed m1 allowance or all129 F is used.',
 'physical': 'Compose receipt442 uniform physical Wilson endpoint and actualsource2a1 Hilbert-map action in both sectors. eta_x=(b1+length/sqrt2)radius and eta0=b1radius. Error=2t epsilon sqrt(1/2)+eta_x sqrt(1/2)+eta0, valid on evolved arbitrary bodyvectors by CAR spectator cancellation. Physical E0fullTpsi is compared to fixed Tsrc E0bodypsi, paying eta0, NOT equated or preparation-adopted. Physical Ex is not TExTdag. Common Ephi phases cancel. Initial physical kernel need not equal zero exactly: its endpoint discrepancy is separately priced.',
 'blind_controls': 'ROW/COLUMN, omittedRef and initialPi/HC omissions remain EXACTLY BLIND for this scalar Clifford preparation; no false discrimination. Second-link and wrong Gaussian variance controls change literal moment contraction. Electric half64->32 control changes conserved-energy/remainder law, not the blind initial Pi moment. Completed dense/Hermite controls are not rerun.',
}


def run(output: Path) -> dict[str, Any]:
    started = time.monotonic()
    sys.set_int_max_str_digits(0)
    captured = {**body.CAPTURED, Path(__file__).name: Path(__file__).read_bytes()}
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file:
            p = Path(file)
            if p.parent == ROOT and p.suffix == '.py':
                captured.setdefault(p.name, p.read_bytes())
    for name, sha in PIN.items():
        raw = (ROOT/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ArithmeticError('Pinned scientific input changed: '+name)
        captured[name] = raw
    cert = json.loads(captured[next(k for k in PIN if 'root_static' in k)])
    source = json.loads(captured['joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json'])
    transfer = json.loads(captured['joined_phi_u1_nonlocal_body_transport_verified_source_transfer_receipt_20261010.json'])
    saved = json.loads(captured[body.ARCHIVE])['input_descriptor']['exact_NEW_allq_BO_mean']
    radius = F(cert['Rootexist']['radius_exact'])
    centres = [F(z) for z in saved['centre_exact_rational']]
    bounds = [(c-radius, c+radius) for c in centres]
    if not cert['Rootexist']['exists'] or not cert['Rootexist']['unique_in_fixed_box']:
        raise ArithmeticError('Missing corrected root existence')
    if not all(0 < lo < hi < 1 for lo, hi in bounds):
        raise ArithmeticError('Root-frame rational upper1 not justified')
    g = [interval_box(*bounds[0])]+[interval_box(*bounds[1])]*3
    gamma = np.asarray(body.joined.old.full.supplier.primitive.SELECTED)
    if not np.all(np.isin(np.real(gamma), [-1,0,1])) or not np.all(np.isin(np.imag(gamma), [-1,0,1])):
        raise ArithmeticError('Clifford entries not exact Gaussian integers')
    for m in range(4):
        if np.trace(gamma[m]) != 0:
            raise ArithmeticError('Clifford scalar trace failed')
        for n in range(4):
            if not np.array_equal(gamma[m]@gamma[n]+gamma[n]@gamma[m], (2 if m==n else 0)*np.eye(4)):
                raise ArithmeticError('Exact Clifford anticommutator failed')
            if np.trace(gamma[m]@gamma[n]) != (4 if m==n else 0):
                raise ArithmeticError('Exact four-colour trace law failed')
    data, species, inputs, _ = body.full_inputs()
    captured.update(inputs)
    native = data['native']
    for actual, declared in zip(data['profiles'], cert['antecedents']['profiles']):
        if F(actual['onsite']) != F(declared['onsite']) or actual['load'] != declared['load']:
            raise ArithmeticError('Native/current-root profile mismatch')
        for a, d in zip(actual['paths'], declared['paths']):
            if (a['axis'], a['length'], F(a['c0']), F(a['ci'])) != (d['axis'], d['length'], F(d['c0']), F(d['ci'])):
                raise ArithmeticError('Native current-root hopping mismatch')
    geo = current.geometry(data, species[0])
    geom = prices.exact_geometry(native, geo)
    if not np.array_equal(native['D'], native['D'].astype(np.int64)):
        raise ArithmeticError('Nonintegral physical incidence')
    if not np.array_equal(native['D'].astype(np.int64)@geo['k'].astype(np.int64).T, geo['delta'].astype(np.int64).T):
        raise ArithmeticError('Actual full-Q ROW Ward identity failed')
    transport._NATIVE = native
    energy = transport.rational_energies(geom, species)
    paths = body.joined.old.native_paths(species[0].profile, native)
    bloch = Bloch(species[0].profile, g)
    auxiliary = Bloch(species[1].profile, g)
    hops = hop_vectors(species[0].profile, paths, bloch.frame)
    u0 = bloch.u([0,0,0])
    source_weights = [(1+sum((u0[m].real*int(gamma[m,a,a].real) for m in range(4)),exact(0)))/8 for a in range(4)]
    if not all(endpoint(w,False)>0 for w in source_weights):
        raise ArithmeticError('Four raw Ref-source weights not positive')
    if not contains_zero(sum(source_weights,exact(0))-exact(F(1,2))):
        raise ArithmeticError('Raw source norm sum does not enclose exact half')
    indices = np.flatnonzero(geo['y'] == 0)
    if len(indices) != 12:
        raise ArithmeticError('Actual twelve signed ROW receivers absent')
    epsilon = F(source['action_input']['epsilon']['upper'])
    packet_radius = F(source['source_weights']['prepared_radius_L2']['upper'])
    if epsilon != F(transfer['source_action_bound_exact']) or packet_radius != F(transfer['source_radius_L2_upper_exact']):
        raise ArithmeticError('Physical source transfer epoch mismatch')
    if not source['action_input']['uniform_body_Hilbert_map'] or not source['all_inputs_and_sources_unchanged']:
        raise ArithmeticError('Uniform physical source action absent')
    b1 = F(transfer['fundamental_active_W1_upper_exact'])
    root_half = transport.sqrt_upper(F(1,2))
    eta0 = b1*packet_radius
    rows = []
    for r0 in indices:
        r = int(r0)
        x = int(geo['x'][r])
        m1 = moments(r, geo, geom, native, bloch, hops)
        second_omitted = moments(r, geo, geom, native, bloch, hops, omit_second=True)
        wrong_variance = moments(r, geo, geom, native, bloch, hops, variance=1)
        caps = transport.rational_caps(r, geo, geom, species[0], species, energy)
        remainder = CLOCK**2*caps['quadratic_cap']
        length = int(geo['k'][r]@geo['k'][r])
        eta = (b1+length*root_half)*packet_radius
        physical_error = 2*CLOCK*epsilon*root_half+eta*root_half+eta0
        supplier_row = next(z for z in transfer['receiver_rows'] if z['path'] == r)
        if CLOCK != F(supplier_row['time_exact']) or physical_error != F(supplier_row['endpoint_Duhamel_kernel_error_exact']):
            raise ArithmeticError('Receipt442 endpoint composition differs')
        lo, hi = endpoint(m1.real, False), endpoint(m1.real, True)
        if not contains_zero(m1.imag):
            raise ArithmeticError('Imaginary moment symmetry violated')
        physical_imaginary = (-CLOCK*hi-remainder-physical_error, -CLOCK*lo+remainder+physical_error)
        lower = min(abs(v) for v in physical_imaginary) if physical_imaginary[0]*physical_imaginary[1] > 0 else F(0)
        if lower <= 0:
            raise ArithmeticError('Exact physical signed signal not separated')
        covariance_sum = IV.mpc(0)
        dl = geo['delta'][r].astype(np.int64)@geom['L_integer']
        for v in range(64):
            uv = bloch.u(native['sites'][x]-native['sites'][v])
            u0 = bloch.u(native['sites'][v]-native['sites'][0])
            covariance_sum -= exact(F(int(dl[v]), geom['denominator']))*dot(uv, u0)
        if not contains_zero(covariance_sum.real) or not contains_zero(covariance_sum.imag):
            raise ArithmeticError('Full connected HC cancellation witness failed')
        imaginary_absolute_upper = max(abs(endpoint(m1.imag,False)),abs(endpoint(m1.imag,True)))
        nonlocal_upper = CLOCK*(max(abs(lo), abs(hi))+imaginary_absolute_upper)+remainder+physical_error
        coloc_error = 2*CLOCK*epsilon*root_half+eta0*root_half+eta0
        coloc_change = CLOCK*species[0].hmax_exact*root_half+coloc_error
        separation = F(1,2)-coloc_change-nonlocal_upper
        initial_physical_error = eta*root_half+eta0
        if separation <= 0 or lower <= initial_physical_error:
            raise ArithmeticError('Co-local or own-initial physical separation failed')
        contrast = m1.real-second_omitted.real
        variance_contrast = m1.real-wrong_variance.real
        if contains_zero(contrast) or contains_zero(variance_contrast):
            raise ArithmeticError('Meaningful second-link/Gaussian-law controls blind')
        literal_float = transport.moments(r,geo,species[0],species[1],native)['m1']
        rows.append(dict(path=r,row=x,col=0,length=length,clock_exact=str(CLOCK),
            m1=complex_enclosure(m1),C0_exact='0',Pi_exact='0',HC_exact='0',
            HC_connected_full64_interval_witness=complex_enclosure(covariance_sum),
            allcharged_meanrho_exact='0; active2+63aux2-128 including Ref4',
            rational_caps={k:str(v) for k,v in caps.items()},remainder_upper_exact=str(remainder),
            eta_receiver_uniform_exact=str(eta),eta_physical_source_vs_fixed_Tsrc_E0body_exact=str(eta0),
            physical_transfer_upper_exact=str(physical_error),
            physical_imaginary_interval=dict(lower=str(physical_imaginary[0]),upper=str(physical_imaginary[1])),
            whole_kernel_lower_relative_initial_C0_exact=str(lower),
            physical_initial_kernel_upper_exact=str(initial_physical_error),
            change_from_own_physical_initial_lower_exact=str(lower-initial_physical_error),
            nonlocal_absolute_upper_exact=str(nonlocal_upper),co_local_change_upper_exact=str(coloc_change),
            co_local_real_separation_lower_exact=str(separation),
            controls=dict(omitted_second_link=complex_enclosure(second_omitted),second_link_contrast=enclosure(contrast),
                wrong_density_variance1=complex_enclosure(wrong_variance),variance_contrast=enclosure(variance_contrast),
                initial_ROW_COLUMN_omittedRef_Pi_HC='EXACTLY BLIND; retained'),
            floating_display=dict(m1=float((lo+hi)/2),literal_full_matrix_m1=dict(real=literal_float.real,imaginary=literal_float.imag),
                literal_vs_interval_midpoint_defect=abs(literal_float-complex(float((lo+hi)/2))),
                remainder=float(remainder),physical_lower=float(lower),co_local_separation=float(separation))))
    # Energy-law adverse control at identical preparation/clock: never infer
    # discrimination from the exactly blind initial Pi moment.
    half_E1 = energy['E1']-F(32)*F(129,8)
    half_E2 = energy['E2']-F(32)*transport.sqrt_upper(F(129*131))/8
    if half_E1 == energy['E1'] or half_E2 == energy['E2']:
        raise ArithmeticError('Wrong electric energy-law control failed')
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file:
            p = Path(file)
            if p.parent == ROOT and p.suffix == '.py':
                captured.setdefault(p.name, p.read_bytes())
    # Capture actual root/source antecedent bytes named by the receipts.
    for name, sha in transfer['input_sha256'].items():
        raw = (ROOT/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ArithmeticError('Receipt442 bound input changed: '+name)
        captured[name] = raw
    for name, raw in captured.items():
        if (ROOT/name).read_bytes() != raw:
            raise ArithmeticError('Source/input mutated during contraction: '+name)
    archive = output.with_name(output.stem+'_input_bytes.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, raw in sorted(captured.items()):
            z.writestr(name, raw)
        z.writestr('actual_full_Q_AP_Coulomb_geometry.json',json.dumps(dict(
            N=4,V=64,a='1/64',Q_float_display=native['Q'].tolist(),D=native['D'].tolist(),
            L_integer=geom['L_integer'].tolist(),denominator=geom['denominator'],
            signed_paths_k=geo['k'].astype(np.int64).tolist(),signed_ROW=geo['x'].tolist(),signed_COLUMN=geo['y'].tolist(),
            Clifford_real=np.real(gamma).tolist(),Clifford_imaginary=np.imag(gamma).tolist(),
            profiles=cert['antecedents']['profiles'],root_box=[dict(lower=str(lo),upper=str(hi)) for lo,hi in bounds]),default=body.packed))
    result = dict(status='CURRENT_N4_ACTUAL12_PHYSICAL_SIGNED_NONLOCAL_INTERVAL_PROVED',proof=PROOF,
        receiver_rows=rows,clock_exact=str(CLOCK),floating_premise_used=False,
        root_box_exact=[dict(lower=str(lo),upper=str(hi)) for lo,hi in bounds],
        active_all64_Bloch_gap_enclosures=bloch.gaps,auxiliary_all64_Bloch_gap_enclosures=auxiliary.gaps,
        four_source_norm_squared_enclosures=[enclosure(w) for w in source_weights],source_total_norm_squared_exact='1/2',
        conserved_energy_rational_prices={k:str(v) for k,v in energy.items()},
        wrong_half_electric_control=dict(correct_HE='64Pi^2',wrong_HE='32Pi^2',HC_unchanged='64rho Lplus rho',
            correct_E1=str(energy['E1']),wrong_E1=str(half_E1),
            correct_E2=str(energy['E2']),wrong_E2=str(half_E2),
            exact_E1_law_contrast=str(energy['E1']-half_E1),
            exact_E2_law_contrast=str(energy['E2']-half_E2),
            correct_dotTheta='128Pi',wrong_dotTheta='64Pi',
            preparation_and_time_unchanged=True,initial_m1_control='BLIND exact Pi cancellation; discrimination is in actual kinetic energy law, conserved prices and Heisenberg velocity'),
        source_bindings=dict(corrected_root=PIN[next(k for k in PIN if 'root_static' in k)],physical_transfer=PIN['joined_phi_u1_nonlocal_body_transport_verified_source_transfer_receipt_20261010.json'],
            actual_uniform_source=PIN['joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json']),
        input_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in captured.items()},
        immutable_inputs=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),
        all_inputs_and_source_unchanged=True,own_source_utf8=captured[Path(__file__).name].decode(),
        governing_plan=PLAN,governing_revision=REVISION,scoped_work_items=['induced-photon/verification'],
        template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
        exclusions=['OriginalSOURCE/CURRENT427 adoption','measured full129/fullfield evolution','future Wick/Slater','interacting quantum ground','charged pole','dispersion/cone','dense/Hermite control reruns','all129 current F substitution'],
        runtime=dict(executable=sys.executable,python=sys.version,mpmath=mp.__version__,numpy=np.__version__,interval_dps=IV.dps),
        elapsed_seconds=time.monotonic()-started)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        receipt = run(args.output)
    print(json.dumps(dict(status=receipt['status'],clock_exact=receipt['clock_exact'],
        signed_receivers=len(receipt['receiver_rows']),elapsed_seconds=receipt['elapsed_seconds'])),flush=True)
