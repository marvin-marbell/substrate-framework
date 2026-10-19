"""Native N4 AP charged retarded Green: explicit one-loop/GW Gaussian control.

Held Phi, actual one-active plus 63-auxiliary filled negative bands, primitive
u=UE/a=64. Not original Y/E/Ref source, not full interacting pole or g-2.
Physical Theta=a*A; scalar potential has energy per unit CAR charge.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import importlib.machinery
import json
from pathlib import Path
import sys
import time
from typing import Any
ROOT = Path(__file__).resolve().parent
OWN = Path(__file__).read_bytes()
SOURCES: dict[str, bytes] = {str(Path(__file__).resolve()): OWN}
_GET_CODE = importlib.machinery.SourceFileLoader.get_code

def capture_code(self: Any, fullname: str) -> Any:
    path = Path(self.path).resolve()
    local = path.parent == ROOT or (path.parent == Path('/tmp') and path.name.startswith('induced-'))
    if local:
        data = path.read_bytes()
        if str(path) in SOURCES and SOURCES[str(path)] != data:
            raise RuntimeError('Scientific source epoch changed at import: ' + str(path))
        SOURCES[str(path)] = data
    return _GET_CODE(self, fullname)

importlib.machinery.SourceFileLoader.get_code = capture_code
import numpy as np
from scipy.optimize import brentq
PHOTON = Path('/tmp/induced-native-photon-green.py')
PHOTON_SHA = '9806d2ee6d108f64a5887cedc16322326d7e22962d5529dc219856ab1e1f0962'
RECEIPT = ROOT / 'induced_native_photon_green_receipt.json'
RECEIPT_SHA = '7fb1abe8a04dcad8dd12409979667f77e9c7cf8144a96978486daadc2a0f262e'

def load_supplier() -> Any:
    raw = PHOTON.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PHOTON_SHA:
        raise RuntimeError('Frozen photon supplier changed')
    spec = importlib.util.spec_from_file_location('charged_actual_photon_supplier', PHOTON)
    if spec is None or spec.loader is None:
        raise RuntimeError('Photon supplier loader unavailable')
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.NativePhotonGreen(256)

def packed(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            return {'real': x.real.tolist(), 'imag': x.imag.tolist()}
        return x.tolist()
    if isinstance(x, complex):
        return {'real': x.real, 'imag': x.imag}
    if isinstance(x, dict):
        return {str(k): packed(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [packed(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x

def embedding(native: dict[str, Any], p: np.ndarray) -> np.ndarray:
    return np.kron((np.exp(1j * native['sites'] @ p) / np.sqrt(native['V']))[:, None], np.eye(4))

def vertices(native: dict[str, Any], profile: dict[str, Any], pout: np.ndarray,
             pin: np.ndarray, edge: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Literal path sums with AP seam hops, no guessed Fourier edge convention."""
    M = edge.shape[1]
    out = np.zeros((M + density.shape[1], 4, 4), complex)
    V = native['V']
    for path in profile['paths']:
        x, y = path['source'], path['target']
        phase = np.exp(1j * (pin @ native['sites'][y] - pout @ native['sites'][x])) / V
        reverse = np.exp(1j * (pin @ native['sites'][x] - pout @ native['sites'][y])) / V
        ell = native['a'] * np.sum(edge[list(path['edges'])], axis=0)
        out[:M] += ell[:, None, None] * (1j * phase * path['hop'] - 1j * reverse * path['hop'].conj().T)
    if density.shape[1]:
        amplitude = np.exp(1j * native['sites'] @ (pin - pout)) @ density / V
        out[M:] = amplitude[:, None, None] * np.eye(4)
    return out

def literal_coulomb(native: dict[str, Any], projectors: list[np.ndarray]) -> dict[str, Any]:
    V = native['V']; mean = np.zeros(V); cov = np.zeros((V, V))
    for profile, P in zip(native['Pprofiles'], projectors, strict=True):
        blocks = P.reshape(V, 4, V, 4).transpose(0, 2, 1, 3)
        rho = np.trace(blocks[np.arange(V), np.arange(V)], axis1=1, axis2=2).real
        mean += profile['load'] * (rho - 2)
        cov += profile['load'] * (np.diag(rho) - np.sum(abs(blocks)**2, axis=(2, 3)))
    u = native['UE'] / native['a']
    return {'energy': float(u * (mean @ native['Lplus'] @ mean + np.sum(native['Lplus'] * cov))),
            'Hartree': float(u * mean @ native['Lplus'] @ mean), 'mean': mean}

def ward_check(s: Any, native: dict[str, Any], profile: dict[str, Any], h: np.ndarray) -> list[dict[str, Any]]:
    rows = []; V = native['V']; theta = np.zeros(3*V)
    for q in (np.array([np.pi/2, 0., 0.]), np.array([np.pi, np.pi/2, np.pi/2])):
        for chi in (np.cos(native['sites'] @ q), np.sin(native['sites'] @ q)):
            edge = np.empty(3*V)
            for x, site in enumerate(native['sites']):
                for axis in range(3):
                    target = site.copy(); target[axis] = (target[axis]+1) % native['N']
                    y = int(np.flatnonzero(np.all(native['sites'] == target, axis=1))[0])
                    edge[3*x+axis] = chi[y]-chi[x]
            J = s.primitives.path_derivative(native, profile, theta, edge)
            rho = np.diag(np.repeat(chi, 4))
            rhs = 1j * (h @ rho - rho @ h)
            rows.append({'qtheta': q, 'literal_full_256_matrix_spatial_Ward_error': float(np.max(abs(J-rhs))),
                         'finite_frequency_Ward_error': float(np.max(abs((complex(3,.7)*np.eye(4*V)-h)@rho-rho@(complex(2,.2)*np.eye(4*V)-h)-(complex(1,.5)*rho+1j*J))))})
    return rows

def self_energy(B: np.ndarray, d: np.ndarray, z: complex) -> np.ndarray:
    return (B/(z-d)[None, :]) @ B.conj().T

def isolate_poles(h: np.ndarray, B: np.ndarray, d: np.ndarray) -> list[dict[str, Any]]:
    """Isolate actual roots in the bath-free central gap and two exterior gaps."""
    low = float(np.max(d[d<0])); high = float(np.min(d[d>0]))
    bound = float(np.max(abs(d)) + np.linalg.norm(h,2) + np.linalg.norm(B,2) + 10.)
    eps = 1e-7
    intervals = [(-bound, float(np.min(d))-eps), (low+eps, high-eps), (float(np.max(d))+eps, bound)]
    roots: list[dict[str, Any]] = []
    for left, right in intervals:
        def matrix(x: float) -> np.ndarray:
            F = x*np.eye(4)-h-self_energy(B,d,complex(x))
            return (F+F.conj().T)/2
        el = np.linalg.eigvalsh(matrix(left)); er = np.linalg.eigvalsh(matrix(right))
        for band in range(4):
            if not el[band] < 0 < er[band]:
                continue
            solved: Any = brentq(lambda x: float(np.linalg.eigvalsh(matrix(x))[band]), left, right, xtol=2e-11, full_output=False)
            root = float(solved)
            if any(abs(root-item['energy']) < 1e-7 for item in roots):
                continue
            ev, U = np.linalg.eigh(matrix(root)); keep = abs(ev)<1e-7
            if not np.any(keep):
                raise ArithmeticError('Root residual did not isolate an eigenvector')
            W = U[:,keep]
            metric = np.eye(4) + (B/((root-d)**2)[None,:])@B.conj().T
            R = W @ np.linalg.solve(W.conj().T@metric@W, W.conj().T)
            roots.append({'energy': root, 'multiplicity': int(np.sum(keep)), 'residue': R,
                          'residue_eigenvalues': np.linalg.eigvalsh(R), 'residue_trace': float(np.trace(R).real),
                          'root_matrix_residual': float(np.max(abs(matrix(root)@W))),
                          'isolating_bath_free_interval': [left,right], 'nearest_auxiliary_pole_distance': float(np.min(abs(root-d)))})
    return sorted(roots,key=lambda x:x['energy'])

def run() -> dict[str, Any]:
    started = time.monotonic(); receipt_raw = RECEIPT.read_bytes()
    if hashlib.sha256(receipt_raw).hexdigest() != RECEIPT_SHA:
        raise RuntimeError('Frozen photon receipt changed')
    supplier = load_supplier(); native = supplier.native; s = supplier.s
    V = native['V']; a = native['a']; u = native['UE']/a
    profile = native['Pprofiles'][0]; zero = np.zeros(3*V)
    mom = (2*np.pi*native['sites']+np.pi)/native['N']
    targets = [0, int(np.flatnonzero(np.all(native['sites']==[1,0,0],axis=1))[0])]
    mean_path = Path(s.primitives.MEAN)
    mean_at_import = mean_path.read_bytes()
    descriptor={'supplier_source_sha256':PHOTON_SHA,'supplier_receipt_sha256':RECEIPT_SHA,'photon_supplier_execution_epoch':supplier.full.ASSIGNED,
                'charged_consumer_assigned_revision':'d86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7',
                'plan_id':'issue-222-induced-photon-delta13-continuation-20261008','N':4,'g0':native['g0'],'gs':native['gs'],
                'a':a,'UE':native['UE'],'u':u,'charge':1,'Pprofile_loads':[p['load'] for p in native['Pprofiles']],
                'active_model':asdict(profile['model']),'AP_momenta':[mom[i] for i in targets],'spectral_bins':256,
                'actual_mean_input':{'path':str(mean_path),'sha256':hashlib.sha256(mean_at_import).hexdigest(),'bytes_utf8':mean_at_import.decode()},
                'source_hashes_at_import':{path:hashlib.sha256(data).hexdigest() for path,data in SOURCES.items()},
                'binding_phase':'immediately_after_actual_supplier_import_before_charged_calculation'}
    descriptor=packed(descriptor)
    input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest()
    hfull = s.primitives.matter_hamiltonian(native, profile, zero)
    embeds = [embedding(native,k) for k in mom]
    bands = [np.linalg.eigh(E.conj().T@hfull@E) for E in embeds]
    projectors = []
    for species in native['Pprofiles']:
        e,U = np.linalg.eigh(s.primitives.matter_hamiltonian(native,species,zero))
        projectors.append(U[:,:2*V]@U[:,:2*V].conj().T)
    base_C = literal_coulomb(native, projectors)
    Cfull = u*np.kron(np.diag(np.diag(native['Lplus'])),np.eye(4))-2*u*np.repeat(np.repeat(native['Lplus'],4,axis=0),4,axis=1)*projectors[0]
    receiver = []; Cchecks = []
    for index in targets:
        E = embeds[index]; eig,U = bands[index]; C = E.conj().T@Cfull@E
        for spin in (2,3):
            orbital = E@U[:,spin]; W = np.outer(orbital,orbital.conj())
            added = literal_coulomb(native,[projectors[0]+W,projectors[1]])
            predicted = float(np.vdot(orbital,Cfull@orbital).real)
            Cchecks.append({'p':mom[index], 'positive_band':spin, 'literal_all64_CAR_energy_change':added['energy']-base_C['energy'],
                            'bare_self_exchange_prediction':predicted, 'error':float(abs(added['energy']-base_C['energy']-predicted)),
                            'added_Hartree':added['Hartree'], 'active_occupation_before':float(np.vdot(orbital,projectors[0]@orbital).real)})
        receiver.append({'index':index,'p':mom[index],'bare':E.conj().T@hfull@E,'C':C,'DW':np.zeros((4,4),complex),'B':[],'d':[],'mixed_interference_Gram':np.zeros((4,4),complex),'omitted_holonomy_Gram':np.zeros((4,4),complex)})
    covariance = np.zeros((3*V,3*V)); qrows = []; vertex_errors = []; countA=0; countR=0
    for site in native['sites']:
        q = tuple(int(i) for i in site); neg = tuple((-i)%native['N'] for i in q)
        if q>neg:
            continue
        block = supplier.block(q); edge = block.data['physical_edgeframe']; density = block.data['physical_densityframe']
        amp = block.amplitudes; omega = block.omega
        countA += block.M; countR += block.R
        CA = np.real(amp[:block.M]@amp[:block.M].conj().T)
        covariance += a*a*edge@CA@edge.T
        allowed = {tuple((native['sites'][r['index']]+sign*np.asarray(q))%native['N']) for r in receiver for sign in (-1,1)}
        qrows.append({'q':q,'photon_modes':block.M,'density_modes':block.R,'positive_quantum_modes':len(omega),'frequency_min':float(np.min(omega)), 'spectral_bins':256})
        for rec in receiver:
            p=rec['p']; external=embeds[rec['index']]
            for kidx,ksite in enumerate(native['sites']):
                if tuple(ksite) not in allowed:
                    continue
                # Some indices in the union belong only to the other external p;
                # the literal Fourier contraction sets those vertices to zero.
                O = vertices(native,profile,p,mom[kidx],edge,density)
                if float(np.max(abs(O)))<1e-12:
                    continue
                ev,U = bands[kidx]
                for column in range(4):
                    positive = ev[column]>0
                    amplitudes = amp if positive else amp.conj()
                    BA = np.einsum('mij,ml,j->il',O[:block.M],amplitudes[:block.M],U[:,column],optimize=True)
                    BP = np.einsum('mij,ml,j->il',O[block.M:],amplitudes[block.M:],U[:,column],optimize=True)
                    B = BA+BP
                    rec['B'].append(B); rec['d'].append(ev[column]+omega if positive else ev[column]-omega)
                    rec['mixed_interference_Gram'] += B@B.conj().T-BA@BA.conj().T-BP@BP.conj().T
                    if q==(0,0,0):
                        rec['omitted_holonomy_Gram'] += B@B.conj().T
                if q in ((0,0,0),(1,0,0),(1,2,0)):
                    for m in range(block.M):
                        J = s.primitives.path_derivative(native,profile,zero,a*edge[:,m])
                        vertex_errors.append(float(np.max(abs(O[m]-external.conj().T@J@embeds[kidx]))))
                    for m in range(block.R):
                        rho=np.kron(np.diag(density[:,m]),np.eye(4))
                        vertex_errors.append(float(np.max(abs(O[block.M+m]-external.conj().T@rho@embeds[kidx]))))
        print('Actual qgroup',q,'positive modes',len(omega),flush=True)
    if (countA,countR,len(qrows))!=(129,63,36):
        raise ArithmeticError('Incomplete photon/density frame consumption')
    path_variances=[]; DWfull=np.zeros_like(hfull); average=hfull.copy()
    for path in profile['paths']:
        edges=list(path['edges']); var=float(np.sum(covariance[np.ix_(edges,edges)])); path_variances.append(var)
        x,y=path['source'],path['target']; hop=path['hop']
        DWfull[4*x:4*x+4,4*y:4*y+4] += -.5*var*hop
        DWfull[4*y:4*y+4,4*x:4*x+4] += -.5*var*hop.conj().T
        average[4*x:4*x+4,4*y:4*y+4] += np.expm1(-var/2)*hop
        average[4*y:4*y+4,4*x:4*x+4] += np.expm1(-var/2)*hop.conj().T
    # Independent literal second derivatives contract the entire covariance.
    vals,Ucov=np.linalg.eigh(covariance); DWcheck=np.zeros_like(hfull)
    for j in np.flatnonzero(vals>1e-12):
        direction=Ucov[:,j]*np.sqrt(vals[j])
        DWcheck += .5*s.primitives.path_derivative(native,profile,zero,direction,2,other=direction)
    DWerror=float(np.max(abs(DWfull-DWcheck)))
    outputs=[]; worst=0.; aux_error=0.
    for rec in receiver:
        E=embeds[rec['index']]; DW=E.conj().T@DWfull@E; h=rec['bare']+rec['C']+DW
        B=np.concatenate(rec['B'],axis=1); d=np.concatenate(rec['d']); probes=[]
        for z in (complex(0,.5),complex(10,.25),complex(30,1),complex(-25,.1),complex(200,2)):
            Sigma=self_energy(B,d,z); F=z*np.eye(4)-h-Sigma; G=np.linalg.solve(F,np.eye(4))
            minusImG=-(G-G.conj().T)/(2j); mineig=float(np.min(np.linalg.eigvalsh(minusImG))); worst=min(worst,mineig)
            identity=z.imag*(np.eye(4)+(B/(abs(z-d)**2)[None,:])@B.conj().T)
            spectral_error=float(np.max(abs(minusImG-G.conj().T@identity@G))); aux_error=max(aux_error,spectral_error)
            probes.append({'z':z,'Sigma':Sigma,'G':G,'minus_Im_G_eigenvalues':np.linalg.eigvalsh(minusImG),'positive_star_resolvent_identity_error':spectral_error})
        poles=isolate_poles(h,B,d)
        outputs.append({'p_dimensionless':rec['p'],'p_physical':rec['p']/a,'bare_band_energies':bands[rec['index']][0],
                        'bare_h':rec['bare'],'bare_C_self_exchange':rec['C'],'Debye_Waller':DW,'static_h_in_G':h,
                        'auxiliary_real_poles':len(d),'auxiliary_pole_range':[float(np.min(d)),float(np.max(d))],
                        'auxiliary_positive_Gram':B@B.conj().T,'coherent_mixed_A_phi_interference_Gram':rec['mixed_interference_Gram'],
                        'holonomy_q0_Gram_not_omitted':rec['omitted_holonomy_Gram'],'retarded_probes':probes,'actually_isolated_discrete_poles':poles,
                        'exact_Gaussian_averaged_h_minus_second_order_h_norm':float(np.linalg.norm(E.conj().T@(average-hfull-DWfull)@E,2))})
    ward=ward_check(s,native,profile,hfull)
    errors={'literal_source_vertex_max':max(vertex_errors),'seagull_literal_second_derivative_max':DWerror,
            'all64_CAR_C_energy_change_max':max(row['error'] for row in Cchecks),'Ward_max':max(row['literal_full_256_matrix_spatial_Ward_error'] for row in ward),
            'finite_frequency_Ward_max':max(row['finite_frequency_Ward_error'] for row in ward),'minus_Im_G_min':worst,'positive_auxiliary_identity_max':aux_error}
    if max(errors[k] for k in ('literal_source_vertex_max','seagull_literal_second_derivative_max','all64_CAR_C_energy_change_max','Ward_max','finite_frequency_Ward_max'))>1e-7 or worst < -1e-10 or aux_error>1e-8:
        raise ArithmeticError('Actual microscopic consumer identity failure: '+str(errors))
    supplier.s.assert_frozen()
    return {'status':'actual_native_charged_retarded_Green_complete_Gaussian_one_loop_control',
            'input_descriptor':descriptor,'input_sha256':input_sha256,
            'source_at_import':{Path(path).name:{'path':path,'sha256':hashlib.sha256(data).hexdigest(),'bytes_utf8':data.decode()} for path,data in SOURCES.items()},
            'source_unchanged_since_import':{Path(path).name:Path(path).read_bytes()==data for path,data in SOURCES.items()},
            'input_receipt_unchanged':RECEIPT.read_bytes()==receipt_raw and mean_path.read_bytes()==mean_at_import,'runtime':{'executable':sys.executable,'python':sys.version,'numpy':np.__version__},
            'errors':errors,'qgroups':qrows,'literal_all64_CAR_energy_addition_checks':Cchecks,'neutral_all64_C_energy':base_C['energy'],
            'neutral_all64_mean_max':float(np.max(abs(base_C['mean']))),'bare_lattice_Ward_source_checks':ward,
            'photon_covariance_link_phase_variance_range':[min(path_variances),max(path_variances)],'charged_receivers':outputs,
            'law':'G(z,p)=[z-hbare(p)-DeltaC(p)-DW(p)-Sigma(z,p)]^-1. Empty-band weights B=sum J_A*b_A+rho*b_phi, poles E+Omega; occupied weights B=sum J_A*b_A.conjugate+rho*b_phi.conjugate, poles E-Omega. Full coherent amplitudes squared only after addition. DeltaC=u Lplus_xx I-2u Lplus_xy Pactive_xy; other63 neutral Hartree0. DW=1/2<ThetaTheta> h_doubleprime. Scalar loop is correlation G_phi_phi only; instantaneous bareV already DeltaC. Hermitian star auxiliary [[h,B],[Bdag,diag(d)]] proves retarded positivity; all actual positive boson modes and both electron bands consumed.',
            'scope':['NEW explicit one-loop/GW positive Gaussian-screened microscopic control, finite N4 AP UV regulator, frozen Phi, real coexact+harmonic photons and density PH compressed at256 energy bins. Quantum law/parameters declared inputs.',
                     'Isolated poles are only the finite Hermitian auxiliary law poles; not bare Bloch retags, full primitive interacting poles, g-2, commoncone/eR, original Y/E/Ref source or vacuum admission.',
                     'Bare/source lattice Ward identities checked independently. A dressed conserving Ward vertex is not supplied by this one-shot GW approximation; no full QED Ward promotion.',
                     'Exact nonlinear Gaussian Peierls averaging is a validity diagnostic only, not substituted for the requested retarded Green. Large link variance may invalidate a controlled weak-field loop expansion. No fitted counterterm/charge/Schwinger value. Original probability/source/receiver bytes untouched.'],
            'elapsed_seconds':time.monotonic()-started}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    report=run()
    if not all(report['source_unchanged_since_import'].values()) or not report['input_receipt_unchanged']:
        raise RuntimeError('Scientific source/input changed during execution')
    args.output.write_text(json.dumps(packed(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'errors':report['errors'],'phase_variance_range':report['photon_covariance_link_phase_variance_range'],
                      'pole_summary':[[{'energy':p['energy'],'residue_trace':p['residue_trace']} for p in r['actually_isolated_discrete_poles']] for r in report['charged_receivers']],
                      'input_sha256':report['input_sha256'],'elapsed_seconds':report['elapsed_seconds']},indent=2),flush=True)
