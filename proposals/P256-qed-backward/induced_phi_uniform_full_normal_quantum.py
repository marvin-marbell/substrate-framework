"""Full45 uniform physical Cart-normal Phi coupled to both-band charged/heavy PH.

NEW linear Gaussian tangent law at the executed 0c3 binary point. Complex PH
couplings, primitive elasticity and the nonzero tangent force are retained.
No nonlinear quantum ground, original SOURCE, photon cone or clock admission.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import importlib.abc
import importlib.machinery
from itertools import combinations, product
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES: dict[str, bytes] = {Path(__file__).name: Path(__file__).read_bytes()}


class CapturedLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str) -> Any:
        raw = Path(self.path).read_bytes()
        SOURCE_BYTES[Path(self.path).name] = raw
        return self.source_to_code(raw, self.path)


class CapturedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.origin is not None:
            source = Path(spec.origin)
            if source.parent == ROOT and source.suffix == '.py':
                spec.loader = CapturedLoader(fullname, str(source))
                return spec
        return None


_finder = CapturedFinder()
sys.meta_path.insert(0, _finder)
# Load the scientific dependency tree under OUR capture before nested finders.
import induced_phi_quantum_oscillator_backreaction as supplier
import induced_phi_uniform_quantum_backreaction as scalar
sys.meta_path.remove(_finder)
import mpmath as mp
import numpy as np

iv: Any = mp.iv
exact, endpoint, enclosure, bounds = scalar.exact, scalar.endpoint, scalar.enclosure, scalar.bounds
SCALAR_SHA = '0c3d65dce2d22181073cc9bdd373a7ba7c99977f895ffab1a097e7ae070c9da7'
RECEIPT_SHA = 'ee8f8ebd6b20db73b7a5412ca828bc9d2687fc28b1c7bfb5fb3cd27eb5843166'
REVISION = 'd86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7'
OBSERVED_REVISION = '1c4f98f4e0e84bd598b150f73bb276ac882acd511d6af2de5848b4d15a80417a'
sys.set_int_max_str_digits(0)


def centre(x: Any) -> float:
    return float((endpoint(x, 0)+endpoint(x, 1))/2)


def encode(x: np.ndarray) -> dict[str, Any]:
    return dict(real=np.real(x).tolist(), imag=np.imag(x).tolist())


def normal_geometry(native: dict[str, Any], g: list[Any]) -> dict[str, Any]:
    """Algebraic Clifford-grade normal45, independently joined to physical Cart60."""
    gamma = supplier.primitive.SELECTED
    words = [(i,) for i in range(4)]+list(combinations(range(4), 2))+list(combinations(range(4), 3))+[(0,1,2,3)]
    matrices = []
    for word in words:
        m = np.eye(4, dtype=complex)
        for i in word:
            m = m@gamma[i]
        if len(word) in (2,3):
            m = 1j*m
        matrices.append(m)
    M = np.asarray(matrices)
    if np.max(abs(np.einsum('aij,bji->ab', M, M)-4*np.eye(15))) != 0:
        raise ArithmeticError('Exact Clifford-grade matrix-unit algebra changed')
    roots = [iv.sqrt(x) for x in g]
    columns: list[dict[int, Any]] = []
    labels: list[str] = []
    def add(entries: dict[int, Any], label: str) -> None:
        columns.append(entries); labels.append(label)
    for i in range(4):
        add({15*i+i:exact(1)}, f'Gram diagonal {i}')
    for i,j in combinations(range(4),2):
        den = iv.sqrt(g[i]+g[j])
        add({15*i+j:roots[j]/den, 15*j+i:roots[i]/den}, f'Gram symmetric {i}{j}')
    # Grade2 orbit of each grade1 generator has three disjoint entries.
    used = set()
    for j in range(4):
        entries = []
        for m in range(4):
            if m == j:
                continue
            comm = 1j*(gamma[j]@gamma[m]-gamma[m]@gamma[j])
            a = next(a for a in range(4,10) if abs(np.trace(M[a]@comm)) > 0)
            sign = int(np.trace(M[a]@comm).real/8)
            entries.append((15*m+a, exact(sign)*roots[m]))
        indices, v = zip(*entries, strict=True); used.update(indices)
        den = iv.sqrt(v[0]**2+v[1]**2); total = iv.sqrt(sum(x*x for x in v))
        n = [v[1]/den,-v[0]/den,exact(0)]
        z = [(v[1]*n[2]-v[2]*n[1])/total,
             (v[2]*n[0]-v[0]*n[2])/total,
             (v[0]*n[1]-v[1]*n[0])/total]
        add({i:x for i,x in zip(indices,n,strict=True)}, f'grade2 normal generator{j} first')
        add({i:x for i,x in zip(indices,z,strict=True)}, f'grade2 normal generator{j} second')
    for m in range(4):
        for a in range(4,10):
            if 15*m+a not in used:
                add({15*m+a:exact(1)}, f'Phi{m} bivector{words[a]}')
    # One grade4 generator supplies one grade3 orbit vector; all four
    # grade4 field components are vertical and are NOT normal coordinates.
    indices = []; v = []
    for m in range(4):
        comm = 1j*(M[14]@gamma[m]-gamma[m]@M[14])
        a = next(a for a in range(10,14) if abs(np.trace(M[a]@comm)) > 0)
        sign = int(np.trace(M[a]@comm).real/8)
        indices.append(15*m+a); v.append(exact(sign)*roots[m])
    norm = iv.sqrt(sum(x*x for x in v)); w = list(v); w[0] -= norm
    w2 = sum(x*x for x in w)
    for j in range(1,4):
        add({indices[i]:exact(int(i==j))-2*w[i]*w[j]/w2 for i in range(4)}, f'grade3 normal dual {j}')
    for m in range(4):
        for a in range(10,14):
            if 15*m+a not in indices:
                add({15*m+a:exact(1)}, f'Phi{m} trivector{words[a]}')
    if len(columns) != 45:
        raise ArithmeticError('Algebraic full45 dimension failed')
    Q = np.zeros((60,45))
    for a, column in enumerate(columns):
        for slot, value in column.items():
            Q[slot,a] = centre(value)
    Tgrade = M/np.sqrt(8)
    force = supplier.static.force
    R = 2*np.einsum('aij,bji->ab',Tgrade,force.T).real
    Qcart = np.kron(np.eye(4),R.T)@Q
    native_geometry = force.physical_normal_geometry(native)
    basis = supplier.primitive.matrix_unit_basis()
    S = np.kron(np.eye(4),np.einsum('bij,aji->ba',basis,force.T).real/4)
    E = S@Qcart
    deltas = np.einsum('mAa,Aij->amij',Q.reshape(4,15,45),Tgrade)
    K_literal = E.T@supplier.primitive.elastic_hessian(native['frame'],basis,2/native['a'],0.)@E
    errors = dict(grade_metric=float(np.max(abs(Q.T@Q-np.eye(45)))),
                  physical_Cart_metric=float(np.max(abs(Qcart.T@Qcart-np.eye(45)))),
                  physical_primitive_metric=float(np.max(abs(E.T@E-np.eye(45)/8))),
                  physical_normal_vertical=float(np.linalg.norm(native_geometry['orbit'].T@Qcart)),
                  actual_native_normal_span=float(np.max(abs(np.linalg.svd(native_geometry['normal'].T@Qcart,compute_uv=False)-1))))
    if max(errors.values()) > 1e-12:
        raise ArithmeticError('Actual physical Cart60 to algebraic45 join failed')
    slot35 = 15*0+words.index((0,1,2))
    index35 = next(a for a,c in enumerate(columns) if list(c)==[slot35])
    return dict(columns=columns,labels=labels,words=words,M=M,Q=Q,Qcart=Qcart,E=E,
                deltas=deltas,K_literal=K_literal,errors=errors,index35=index35)


def primitive_matrix(geom: dict[str, Any], g: list[Any], a: Any) -> tuple[list[list[Any]], list[Any]]:
    columns = geom['columns']; roots = [iv.sqrt(x) for x in g]; sqrt8 = iv.sqrt(exact(8))
    J = []
    for m,n in product(range(4),repeat=2):
        J.append([(roots[n]*c.get(15*m+n,exact(0))+roots[m]*c.get(15*n+m,exact(0)))/sqrt8 for c in columns])
    K = [[exact(0) for _ in range(45)] for _ in range(45)]
    for b in range(45):
        for c in range(b,45):
            value = 2/a*sum(row[b]*row[c] for row in J)
            value += sum(g[slot//15]/(2*a)*x*columns[c].get(slot,exact(0)) for slot,x in columns[b].items())
            K[b][c] = value; K[c][b] = value
    force = [2*8/a*sum(g[m]*J[5*m][b] for m in range(4)) for b in range(45)]
    return K, force


def trace_tensors(M: np.ndarray, gamma: np.ndarray) -> list[list[tuple[F,list[tuple[int,F,F]],list[tuple[int,int,F,F]]]]]:
    result = []
    for A in M:
        row = []
        for B in M:
            t0 = F.from_float(float(np.trace(A@B).real/32))
            linear = []; quadratic = []
            for i in range(4):
                z = np.trace(gamma[i]@A@B-A@gamma[i]@B)/32
                if z != 0:
                    linear.append((i,F.from_float(float(z.real)),F.from_float(float(z.imag))))
                for j in range(4):
                    z = -np.trace(gamma[i]@A@gamma[j]@B)/32
                    if z != 0:
                        quadratic.append((i,j,F.from_float(float(z.real)),F.from_float(float(z.imag))))
            row.append((t0,linear,quadratic))
        result.append(row)
    return result


def algebraic_species(profile: dict[str, Any], geom: dict[str, Any], g: list[Any], tensors: Any) -> dict[str, Any]:
    roots = [iv.sqrt(x) for x in g]; half = iv.sqrt(exact(2))/2
    cos1,sin1,sin2 = (1,-1,-1,1),(1,1,-1,-1),(1,-1,1,-1)
    B = [[exact(0) for _ in range(45)] for _ in range(45)]
    force = [exact(0) for _ in range(45)]; points = []; Cpoints = []; deltas = []
    signocc = 1 if profile['scale']=='heavy' else -1
    for k in product(range(4),repeat=3):
        x = [exact(profile['onsite']),exact(0),exact(0),exact(0)]
        for p in profile['paths']:
            axis,L = p['axis'],p['length']
            co = exact(cos1[k[axis]])*half if L==1 else exact(0)
            si = exact(sin1[k[axis]])*half if L==1 else exact(sin2[k[axis]])
            x[0] += 2*exact(p['c0'])*co
            x[axis+1] -= 2*exact(p['ci'])*si
        h = [r*y for r,y in zip(roots,x,strict=True)]; energy = iv.sqrt(sum(y*y for y in h)); n = [y/energy for y in h]
        Dreal = [[exact(0) for _ in range(15)] for _ in range(15)]
        Dimag = [[exact(0) for _ in range(15)] for _ in range(15)]
        for A in range(15):
            for Z in range(15):
                t0,linear,quad = tensors[A][Z]; re,im = exact(t0),exact(0)
                for i,r,s in linear:
                    re += signocc*exact(r)*n[i]; im += signocc*exact(s)*n[i]
                for i,j,r,s in quad:
                    re += exact(r)*n[i]*n[j]; im += exact(s)*n[i]*n[j]
                Dreal[A][Z] = re; Dimag[A][Z] = im
        C = np.zeros((45,45),dtype=complex); trace = exact(0)
        for a,col in enumerate(geom['columns']):
            for b in range(a,45):
                re,im = exact(0),exact(0)
                for slot,co in col.items():
                    m,A = divmod(slot,15)
                    for slot2,co2 in geom['columns'][b].items():
                        nfield,Z = divmod(slot2,15)
                        factor = co*co2*x[m]*x[nfield]/64
                        re += factor*Dreal[A][Z]; im += factor*Dimag[A][Z]
                C[a,b] = centre(re)+1j*centre(im); C[b,a] = C[a,b].conjugate()
                B[a][b] += re/energy
                if b != a:
                    B[b][a] += re/energy
                else:
                    trace += re
        for a,col in enumerate(geom['columns']):
            force[a] -= sum(co*x[slot//15]*n[slot%15]/(4*iv.sqrt(exact(8))) for slot,co in col.items() if slot%15<4)
        points.append(dict(k=list(k),x=[centre(y) for y in x],E=centre(energy),PH_trace=bounds(trace),unscaled_gap=bounds(2*energy)))
        Cpoints.append(C); deltas.append(2*energy)
    return dict(B=B,force=force,points=points,C=np.asarray(Cpoints),delta=deltas)


def literal_consumers(native: dict[str, Any], profiles: list[dict[str, Any]], geom: dict[str, Any], algebra: list[dict[str, Any]]) -> list[dict[str, Any]]:
    V = native['V']; zero = np.zeros(3*V); rows = []
    theta = (2*np.pi*np.asarray(list(product(range(4),repeat=3)))+np.pi)/4
    spatial = np.exp(1j*native['sites']@theta.T)/8; fourier = np.kron(spatial,np.eye(4))
    dv,du = np.linalg.eigh(native['Lplus']); rho_basis = du[:,dv>1e-12]*np.sqrt(128*dv[dv>1e-12])[None,:]
    rho = [np.eye(4*V)/8,*[np.diag(np.repeat(v,4)) for v in rho_basis.T]]
    directions = []
    for axis in range(3):
        d = np.zeros(3*V); d[axis::3] = 1/8; directions.append(d)
    for p,alg in zip(profiles,algebra,strict=True):
        sign = -1 if p['scale']=='heavy' else 1
        paths = []
        for displacement,coefficients in p['raw_paths']:
            support = [s for s in native['Pprofiles'][0]['paths'] if np.array_equal(s['displacement'],displacement)]
            if len(support) != V:
                raise ArithmeticError('Every native ordered one/two-link path is required')
            paths.extend(dict(s,scalars=coefficients,hop=s['seam']*np.einsum('m,mij->ij',coefficients,native['frame'])) for s in support)
        profile = dict(model=p['model'],onsite=float(p['onsite'])*native['frame'][0],paths=paths)
        derivative_profiles = [dict(profile,onsite=float(p['onsite'])*df[0]/8,paths=[dict(s,hop=s['seam']*np.einsum('m,mij->ij',s['scalars'],df)/8) for s in paths]) for df in geom['deltas']]
        h = sign*supplier.quantum.matter_hamiltonian(native,profile,zero)
        Fs = np.asarray([sign*supplier.quantum.matter_hamiltonian(native,dp,zero) for dp in derivative_profiles])
        energies,U = np.linalg.eigh(h)
        if not energies[127] < 0 < energies[128]:
            raise ArithmeticError('Literal physical negative BOTH bands must be filled')
        W,Z = U[:,:128],U[:,128:]; delta = energies[None,128:]-energies[:128,None]
        amps = W.conj().T@Fs@Z
        B = 2*np.einsum('aij,bij,ij->ab',amps,amps.conj(),1/delta).real
        static = np.asarray([[centre(x) for x in row] for row in alg['B']])
        # Independent Bloch diagonalization retains four actual PH amplitudes per k.
        Blochamps = []; herrors = []; Ferrors = []
        hfourier = fourier.conj().T@h@fourier
        expected_hfull = np.zeros_like(h)
        for j,point in enumerate(alg['points']):
            x = np.asarray(point['x']); expectedh = sign*np.einsum('m,mij->ij',x,native['frame'])
            expected_hfull[4*j:4*j+4,4*j:4*j+4] = expectedh
            ev,uv = np.linalg.eigh(expectedh)
            expectedF = sign*np.einsum('m,amij->aij',x,geom['deltas'])/8
            Blochamps.append(uv[:,:2].conj().T@expectedF@uv[:,2:])
            herrors.append(float(np.max(abs(hfourier[4*j:4*j+4,4*j:4*j+4]-expectedh))))
        herrors.append(float(np.max(abs(hfourier-expected_hfull))))
        expectedFs = np.zeros_like(Fs)
        for j,point in enumerate(alg['points']):
            expectedFs[:,4*j:4*j+4,4*j:4*j+4] = sign*np.einsum('m,amij->aij',point['x'],geom['deltas'])/8
        actualFs = fourier.conj().T@Fs@fourier
        Ferrors.append(float(np.max(abs(actualFs-expectedFs))))
        Cbloch = np.einsum('kaij,kbij->kab',np.asarray(Blochamps),np.asarray(Blochamps).conj())
        literal_force = np.einsum('ix,aij,jx->a',W.conj(),Fs,W).real
        forceerr = float(np.max(abs(literal_force-np.asarray([centre(x) for x in alg['force']]))))
        errors = dict(full256_h_Bloch=max(herrors),full45_full256_F_Bloch=max(Ferrors),full45_static_PH=float(np.max(abs(B-static))),full45_complex_projector_PH=float(np.max(abs(Cbloch-alg['C']))),full45_force=forceerr)
        if max(errors.values()) > 3e-10:
            raise ArithmeticError('Literal full45/Bloch/complex Clifford join failed')
        dynamic = []
        for om in (0,1,10,100):
            measured = np.einsum('aij,bij,ij->ab',amps,amps.conj(),1/(delta-1j*om))+np.einsum('aij,bij,ij->ab',amps.conj(),amps,1/(delta+1j*om))
            predicted = np.sum(alg['C']/(np.asarray([centre(d) for d in alg['delta']])[:,None,None]-1j*om)+alg['C'].conj()/(np.asarray([centre(d) for d in alg['delta']])[:,None,None]+1j*om),axis=0)
            err = float(np.max(abs(measured-predicted)))
            if err > 3e-10:
                raise ArithmeticError('Full odd-in-frequency PH response was not retained')
            dynamic.append(dict(unscaled_Euclidean_frequency=om,response=encode(predicted),literal_absolute_error=err,odd_antisymmetric_norm=float(np.linalg.norm((predicted-predicted.T)/2,2))))
        densityamps = W.conj().T@np.asarray(rho)@Z
        densitymix = []
        currentmix = []; contacts = np.zeros((45,3),complex); mixed_bound = 0.
        if sign == 1:
            Js = np.asarray([native['a']*supplier.quantum.path_derivative(native,profile,zero,d) for d in directions])
            Jamps = W.conj().T@Js@Z
            for b,d in enumerate(directions):
                DFs = np.asarray([native['a']*supplier.quantum.path_derivative(native,dp,zero,d) for dp in derivative_profiles])
                contacts[:,b] = np.einsum('ix,aij,jx->a',W.conj(),DFs,W)
            mixed_bound = float(2*np.linalg.norm((amps/np.sqrt(delta)).reshape(45,-1),2)*np.linalg.norm((Jamps/np.sqrt(delta)).reshape(3,-1),2))
        for om in (0,1,10,100):
            mix = np.einsum('aij,bij,ij->ab',amps,densityamps.conj(),1/(delta-1j*om))+np.einsum('aij,bij,ij->ab',amps.conj(),densityamps,1/(delta+1j*om))
            densitymix.append(dict(frequency=om,uniform_rho_max=float(np.max(abs(mix[:,0]))),ALL63_nonuniform_rho_max=float(np.max(abs(mix[:,1:])))))
            if sign == 1:
                mix = np.einsum('aij,bij,ij->ab',amps,Jamps.conj(),1/(delta-1j*om))+np.einsum('aij,bij,ij->ab',amps.conj(),Jamps,1/(delta+1j*om))
                currentmix.append(dict(frequency=om,PH_response=encode(mix),contact_minus_PH=encode(contacts-mix)))
        rows.append(dict(role=p['role'],load=p['load'],physical_scale='Lambda' if sign<0 else '1',physical_occupancy='negative -Lambda h=upper h BOTH bands' if sign<0 else 'negative h BOTH bands',literal_operator_shape=list(Fs.shape),literal_PH_shape=list(amps.shape),Bloch_PH_amplitudes=encode(np.asarray(Blochamps)),errors=errors,actual_static_matrix=static.tolist(),actual_dynamic_matrix=dynamic,actual_ALLdensity_cross=densitymix,actual_ALL3_physical_Aholonomy_cross=currentmix,actual_ALL45_Phi_Aholonomy_contact=encode(contacts),ALL_real_Euclidean_frequency_Phi_A_PH_cross_norm_upper=mixed_bound,ALL_real_Euclidean_frequency_total_cross_norm_upper=mixed_bound+float(np.linalg.norm(contacts,2)),neutralheavy_EM_law='charge0: no EM Peierls derivatives' if sign<0 else 'a*d/dTheta with native edge3*site+axis; all1/2-link paths'))
    return rows


def eigen_certificate(K: list[list[Any]]) -> tuple[np.ndarray,np.ndarray,Any,Any]:
    matrix = np.asarray([[centre(x) for x in row] for row in K]); values,U = np.linalg.eigh(matrix)
    residual = exact(0); orth = exact(0)
    for i in range(45):
        r = exact(0); o = exact(0)
        for j in range(45):
            reconstruction = sum(exact(F.from_float(float(U[i,k])))*exact(F.from_float(float(values[k])))*exact(F.from_float(float(U[j,k]))) for k in range(45))
            r += abs(K[i][j]-reconstruction)
            dot = sum(exact(F.from_float(float(U[k,i])))*exact(F.from_float(float(U[k,j]))) for k in range(45))
            o += abs(dot-int(i==j))
        residual = enclosure(F(0),max(endpoint(residual,1),endpoint(r,1)))
        orth = enclosure(F(0),max(endpoint(orth,1),endpoint(o,1)))
    error = residual+2*exact(F.from_float(float(max(abs(values)))))*orth/(1-orth)
    # Polar U=W(U^T U)^(1/2); the displayed conservative bound prices
    # noncommuting eigenvector normalization as well as the reconstruction.
    return values,U,error,orth


def fast_pole_certificate(values: np.ndarray, lower: Any, upper: Any, error: Any, profiles: list[dict[str, Any]], algebra: list[dict[str, Any]], Lambda: Any, G: Any) -> dict[str, Any]:
    xmin,xmax = lower/2,2*upper
    wmin,wmax = iv.sqrt(G*Lambda*xmin),iv.sqrt(G*Lambda*xmax)
    defect,slope = exact(0),exact(0)
    for p,alg in zip(profiles,algebra,strict=True):
        scale = Lambda if p['scale']=='heavy' else exact(1)
        for point,delta in zip(alg['points'],alg['delta'],strict=True):
            tr = enclosure(F(point['PH_trace']['lower']),F(point['PH_trace']['upper']))*exact(p['load'])*scale**2
            d = scale*delta; b = 2*tr/d
            if p['scale']=='heavy':
                denominator = d*d-wmax*wmax; even = b*wmax*wmax/denominator
            else:
                denominator = wmin*wmin-d*d; even = b*wmin*wmin/denominator
            if endpoint(denominator,0) <= 0:
                raise ArithmeticError('Actual charged/fast/heavy spectral separation failed')
            defect += (even+2*wmax*tr/denominator)/Lambda
            slope += G*(b*d*d+tr*(d*d+wmax*wmax)/wmin)/denominator**2
    if endpoint(slope,1)>=1 or endpoint(defect,1)>=endpoint(lower/2,0):
        raise ArithmeticError('Actual full45 complex fast-pole isolation failed')
    windows = []
    for j,value in enumerate(values):
        v = exact(F.from_float(float(value)))
        squared = enclosure(endpoint(v-error-defect,0),endpoint(v+error+defect,1))
        frequency = iv.sqrt(G*Lambda*squared)
        windows.append(dict(mode=j,squared_frequency_units_G_Lambda=bounds(squared),actual_frequency=bounds(frequency),phase_width_at_clock4096=bounds((frequency.b-frequency.a)*4096)))
    return dict(actual_number_of_coupled_fast_modes=45,actual_full_complex_retarded_pole_windows=windows,retarded_operator_defect_divided_Lambda=bounds(defect),retarded_defect_derivative_norm=bounds(slope),proof='For real x=omega^2/(G Lambda), M(x)=Kprim-xI-[C/(Delta-omega)+C* /(Delta+omega)]/Lambda is Hermitian. BOTH even real and odd imaginary gyroscopic pieces are retained. On [lambda_min/2,2lambda_max], all charged poles are below and all heavy poles above. ||M-(Keff/Lambda-xI)||<=defect; derivative defect norm<1 implies all eigenbranches strictly decreasing. Positive/negative endpoint inertia proves exactly45 roots counted with multiplicity, each within its Weyl window. Eigenvalue numerical/interval errors are added, not relabelled as exact high precision.')


def coupled_covariance(geom: dict[str, Any], Kprim: list[list[Any]], forceprim: list[Any], profiles: list[dict[str, Any]], algebra: list[dict[str, Any]], frozen: dict[str, Any]) -> dict[str, Any]:
    Lambda = exact(F(2)**8192); G = exact(8*F(2)**-4096)
    K = [[Kprim[a][b]-sum(exact(p['load'])*(exact(1) if p['scale']=='heavy' else 1/Lambda)*alg['B'][a][b] for p,alg in zip(profiles,algebra,strict=True)) for b in range(45)] for a in range(45)]
    force = [forceprim[a]+sum(exact(p['load'])*(exact(1) if p['scale']=='heavy' else 1/Lambda)*alg['force'][a] for p,alg in zip(profiles,algebra,strict=True)) for a in range(45)]
    values,U,error,orth = eigen_certificate(K)
    lower = exact(F.from_float(float(values[0])))-error; upper = exact(F.from_float(float(values[-1])))+error
    Kmid = np.asarray([[centre(x) for x in row] for row in K]); fmid = np.asarray([centre(x) for x in force])
    if endpoint(lower,0) <= 0:
        raise ArithmeticError('Physically adverse full45 instability: retain receipt and execute nonlinear direction before delivery')
    covariance = (U*(.5/np.sqrt(values)))@U.T
    scalar_variance = frozen['actual_uniform_coupled_quantum_backreaction_and_realtime_transfer']['actual_coupled_ground_Phi_variance']
    scale = iv.sqrt(G/Lambda); kphysical = Lambda*lower; omega = iv.sqrt(G*kphysical)
    even_eta = exact(0); odd_eta = exact(0); even_error = exact(0); odd_error = exact(0); energy_error = exact(0)
    price_rows = []
    for p,alg in zip(profiles,algebra,strict=True):
        physical_scale = Lambda if p['scale']=='heavy' else exact(1)
        eeta,oeta,eerr,oerr,ee = (exact(0) for _ in range(5))
        for point,delta in zip(alg['points'],alg['delta'],strict=True):
            tr = enclosure(F(point['PH_trace']['lower']),F(point['PH_trace']['upper']))*exact(p['load'])*physical_scale**2
            d = delta*physical_scale; b = 2*tr/d
            # C is positive Hermitian, ||Re C|| and ||Im C|| <= Tr C.
            eeta += G*b/(d+omega)**2
            oeta += G*tr/(d*d*omega)
            eerr += G*G*b/(4*omega*(d+omega)**2)
            oerr += G*G*tr/(iv.pi*d*d*omega**2)
            ee += G*b/(4*(d+omega))
        even_eta += eeta; odd_eta += oeta; even_error += eerr; odd_error += oerr; energy_error += ee
        price_rows.append(dict(role=p['role'],even_relative_resolvent_price=bounds(eeta),odd_relative_resolvent_price=bounds(oeta),even_covariance_absolute_price=bounds(eerr),odd_covariance_absolute_price=bounds(oerr),ground_energy_nonadiabatic_price=bounds(ee)))
    eta = even_eta+odd_eta
    if endpoint(eta,1) >= 1:
        raise ArithmeticError('Complex full45 quantum resolvent enclosure failed')
    dynamic_error = (even_error+odd_error)/(1-eta)
    function_rounding = exact(0)
    for a in range(45):
        row_error = exact(0)
        for b in range(45):
            entry = sum(exact(F.from_float(float(U[a,j])))*exact(F.from_float(float(U[b,j])))/(2*iv.sqrt(exact(F.from_float(float(values[j]))))) for j in range(45))
            row_error += abs(entry-exact(F.from_float(float(covariance[a,b]))))
        function_rounding = enclosure(F(0),max(endpoint(function_rounding,1),endpoint(row_error,1)))
    numeric_error = error/(4*lower*iv.sqrt(lower))+2*orth/(2*iv.sqrt(lower)*(1-orth))+function_rounding
    total_error = numeric_error+dynamic_error/scale
    idx = geom['index35']; phi = exact(F.from_float(float(covariance[idx,idx])))
    phi_bounds = enclosure(endpoint((phi-total_error)*scale,0),endpoint((phi+total_error)*scale,1))
    conditional = enclosure(F(scalar_variance['lower']),F(scalar_variance['upper']))
    equilibrium = -np.linalg.solve(Kmid,fmid)
    evbare,_,bare_error,_ = eigen_certificate(Kprim)
    energy_static_units = sum(iv.sqrt(exact(F.from_float(float(x)))) for x in values)/2
    energy_static_units -= sum(iv.sqrt(exact(F.from_float(float(x)))) for x in evbare)/2
    energy_lower = energy_static_units*iv.sqrt(G*Lambda)
    # For S>0 and real skew J, logdet(S+J)>=logdet(S). Also
    # logdet(I+S^-1/2 J S^-1/2)<=||S^-1/2 J S^-1/2||HS^2/2.
    # ||J||<=2y L, L=sum TrC/Delta^2; integral y^2/(y^2+w^2)^2=pi/(4w).
    odd_energy_price = 45*odd_eta**2*omega/4
    bare_lower = exact(F.from_float(float(evbare[0])))-bare_error
    eigenprice = 45*iv.sqrt(G*Lambda)*(error/iv.sqrt(lower)+bare_error/iv.sqrt(bare_lower))/4
    energy_interval = enclosure(endpoint(energy_lower-eigenprice,0),endpoint(energy_lower+energy_error+odd_energy_price+eigenprice,1))
    poles = fast_pole_certificate(values,lower,upper,error,profiles,algebra,Lambda,G)
    return dict(actual_static_Hessian_divided_Lambda=Kmid.tolist(),actual_full45_primitive_eigenvalues_divided_Lambda=evbare.tolist(),actual_full45_effective_eigenvalues_divided_Lambda=values.tolist(),certified_eigenvalue_lower_divided_Lambda=bounds(lower),eigen_reconstruction_operator_error=bounds(error),eigenvector_orthogonality_operator_error=bounds(orth),actual_full45_collective_eigenvectors=U.tolist(),actual_full45_static_collective_frequency_units_sqrt_G_Lambda=np.sqrt(values).tolist(),actual_full45_coupled_quantum_fast_modes=poles,actual_full45_quantum_covariance_units_sqrt_G_over_Lambda=covariance.tolist(),actual_full_coupled_Gaussian_covariance_uniform_operator_error_in_same_units=bounds(total_error),actual_full_coupled_Gaussian_covariance_physical_operator_error=bounds(total_error*scale),actual_full45_Phi35_variance=bounds(phi_bounds),scalar_conditional_0c3_variance=scalar_variance,actual_full_vs_scalar_Phi35_variance_ratio=bounds(phi_bounds/conditional),Phi35_not_eigenmode_offdiagonal_Hessian_norm=float(np.linalg.norm(Kmid[:,idx]-Kmid[idx,idx]*np.eye(45)[:,idx])),actual_primitive_plus_ALLspecies_tangent_force_divided_Lambda=fmid.tolist(),actual_linear_Gaussian_equilibrium_normal_coordinate=equilibrium.tolist(),actual_linear_Gaussian_mean_energy_drop_divided_Lambda=float(-.5*fmid@np.linalg.solve(Kmid,fmid)),actual_force_norm_divided_Lambda=float(np.linalg.norm(fmid)),actual_linear_shift_per_site_normal_norm=float(np.linalg.norm(equilibrium)/8),actual_quantum_zero_point_shift_relative_to_barePhi_and_same_PH_vacua=bounds(energy_interval),ALLspecies_even_and_odd_quantum_prices=price_rows,all45_quantum_covariance_derivation='C=(1/pi) integral_0^infty sym[(y^2/G)I+Lambda Kprim-B(iy)]^-1 dy. B_ab=sum[C_ab/(Delta-iy)+C_ab* /(Delta+iy)]; C_ab=A_a A_b*. No imaginary/gyroscopic part is deleted. Static reference C_ad=sqrt(G/Lambda) Kbar_eff^-1/2/2. For m=Lambda*certified lambda_min and w=sqrt(Gm), even eta<=G sum Btrace/(Delta+w)^2; odd eta<=G sum TrC/(Delta^2*w). Resolvent Neumann identity prices norm difference by (even_error+odd_error)/(1-eta); exact even integral=G^2 Btrace/[4w(Delta+w)^2], odd absolute integral<=G^2 TrC/[pi Delta^2 w^2]. These include ALL768 complex PH channels, both bands and load63. Matrix spectral residual and polar-orthogonality errors are independently interval-priced.',Hamiltonian='H=G Pi_a Pi_a/2+Lambda q^T Kprim q/2+f.q+sum Delta_j bdag_j b_j+sum_a,j q_a(A_aj bdag_j+A_aj* b_j). A contains all4 by45 endpoint derivatives with physical heavy sign and scale. Linear mean is displaced by -Keff^-1 f; connected covariance above. Gaussian PH bosonization is DECLARED NEW LAW, not exact interacting CAR/Phi vacuum.',physical_limits='Native N4 uniform q0 only; all nonuniform Phi/relativecolour and dynamical EM coordinates frozen. The executed nonlinear mean was NOT made stationary by fiat. Linear displacement is the mean of this Gaussian law, not a newly selfconsistent nonlinear root. Quantum covariance is an enclosure, not 45 copies of conditional scalar.',raw_clock_limits=[dict(time=t,raw_connected_Phi35_kernel_complex_disk_radius_upper=bounds(phi_bounds.b),phase_information='Only positivity/Cauchy-Schwarz |K(t)|<=C35 is certified. Pole windows explicitly price enormous physical-clock phase uncertainty; no falsely precise phase from numeric eigenvectors.') for t in ('1/4096','4096')])


def electromagnetic_cross_price(profiles: list[dict[str, Any]], algebra: list[dict[str, Any]], g: list[Any]) -> dict[str, Any]:
    """Outward ALL-frequency omitted EM cross price, independent of float SVD."""
    records = []; total = exact(0); a = exact(F(1,64))
    for p,alg in zip(profiles,algebra,strict=True):
        if p['scale']=='heavy':
            continue
        gap = exact(min(endpoint(d,0) for d in alg['delta']))
        phi_trace = sum(enclosure(F(point['PH_trace']['lower']),F(point['PH_trace']['upper'])) for point in alg['points'])
        current_norm = [exact(0),exact(0),exact(0)]; contact_sum = exact(0)
        for path in p['paths']:
            axis,L = path['axis'],path['length']
            c0,ci = exact(abs(path['c0'])),exact(abs(path['ci']))
            current_norm[axis] += a/4*L*(c0*iv.sqrt(g[0])+ci*iv.sqrt(g[axis+1]))
            contact_sum += L*(c0+ci)
        ph_bound = 2*iv.sqrt(phi_trace*128*sum(j*j for j in current_norm))/gap
        contact_bound = 128*a*contact_sum/(32*iv.sqrt(exact(2)))
        bound = ph_bound+contact_bound; total += exact(p['load'])*bound
        records.append(dict(role=p['role'],load=p['load'],ALL_real_Euclidean_frequencies_Phi45_A3_PH_cross_operator_norm_upper=bounds(ph_bound),ALL_Phi45_A3_mixed_contact_operator_norm_upper=bounds(contact_bound),ALL_frequency_total_cross_operator_norm_upper=bounds(bound)))
    return dict(species=records,ALL64_charged_total_cross_operator_norm_upper=bounds(total),proof='For every real y, each PH denominator magnitude>=actual minimum gap. Matrix Cauchy-Schwarz gives ||B_PhiA(iy)||<=2 sqrt(sum_j Tr C_Phi,j)*sqrt(sum_axis ||Pocc J_axis Pempty||HS^2)/gap. Occupied rank128 gives ||Pocc J_axis Pempty||HS^2<=128||J_axis||op^2. Exact one/two-link current row norm <=a/4 sum_axis L(|c0|sqrt(g0)+|ci|sqrt(gs)), with f=1/8. For unit normal Phi, sum_m ||deltaPhi_m||HS^2=1/2, hence each component opnorm<=1/sqrt2. The mixed contact trace has rank128 and row bound a sum L(|c0|+|ci|)/(32sqrt2). Load1+63 is retained. No photon propagator denominator, EM Schur feedback, common cone or EM-coupled vacuum is inferred from this cross price.')


def nonlinear_mean_response(native: dict[str, Any], profiles: list[dict[str, Any]], geom: dict[str, Any], algebra: list[dict[str, Any]], quantum: dict[str, Any]) -> dict[str, Any]:
    """Execute the retained tangent force on the actual nonlinear primitive/sea."""
    q = np.asarray(quantum['actual_linear_Gaussian_equilibrium_normal_coordinate'])
    direction = np.einsum('a,amij->mij',q,geom['deltas'])/8
    force = np.asarray(quantum['actual_primitive_plus_ALLspecies_tangent_force_divided_Lambda'])
    K = np.asarray(quantum['actual_static_Hessian_divided_Lambda'])
    rows = []; base = 0.; literal_errors = []
    for t in (-1.,0.,.25,.5,1.,1.5,2.):
        frame = native['frame']+t*direction
        Gram = np.einsum('mij,nji->mn',frame,frame).real/4
        primitive = native['V']/native['a']*float(np.sum(Gram**2))
        sea = []
        for p,alg in zip(profiles,algebra,strict=True):
            sign = -1 if p['scale']=='heavy' else 1
            x = np.asarray([point['x'] for point in alg['points']])
            values = np.linalg.eigvalsh(sign*np.einsum('km,mij->kij',x,frame))
            sea.append(float(np.sum(values[:,:2])))
            if t in (0.,1.):
                paths = []
                for displacement,coefficients in p['raw_paths']:
                    support = [s for s in native['Pprofiles'][0]['paths'] if np.array_equal(s['displacement'],displacement)]
                    paths.extend(dict(s,hop=s['seam']*np.einsum('m,mij->ij',coefficients,frame)) for s in support)
                profile = dict(onsite=float(p['onsite'])*frame[0],paths=paths)
                h = sign*supplier.quantum.matter_hamiltonian(native,profile,np.zeros(3*native['V']))
                literal = float(np.sum(np.linalg.eigvalsh(h)[:128]))
                literal_errors.append(abs(literal-sea[-1]))
        leading = primitive+sea[0]
        if t==0:
            base = leading
        rows.append(dict(t=t,primitive_energy_divided_Lambda=primitive,heavy_filled_physical_negative_BOTH_bands_divided_Lambda=sea[0],leading_total_energy_divided_Lambda=leading,charged_active_energy=sea[1],charged_auxiliary63_energy=63*sea[2],charged_correction_to_energy_divided_Lambda=bounds(exact(F.from_float(sea[1]+63*sea[2]))/exact(F(2)**8192)),tangent_predicted_energy_change_divided_Lambda=float(t*force@q+.5*t*t*q@K@q)))
    for row in rows:
        row['observed_nonlinear_energy_change_divided_Lambda'] = row['leading_total_energy_divided_Lambda']-base
    if max(literal_errors)>3e-10:
        raise ArithmeticError('Nonlinear selected mean direction lost literal256 join')
    return dict(candidate_frame=encode(native['frame']+direction),actual_all45_normal_q=q.tolist(),actual_per_site_normal_displacement=float(np.linalg.norm(q)/8),actual_literal256_vs_Bloch_energy_errors=literal_errors,actual_direction_energy_response=rows,interpretation='The native N4 quarter-AP point has a NONZERO full45 tangent force even though its Gaussian Hessian is stable. The force is retained, not removed by an invented counterterm. This is an executed nonlinear direction response to the Gaussian selected mean, NOT a selfconsistent nonlinear minimum or quantum vacuum. Printed heavy/primitive energies are finite numerical diagnostics; charged/Lambda coefficients are independently outward-priced.')


def run(path: Path) -> dict[str, Any]:
    started = time.monotonic(); raw = path.read_bytes(); frozen = json.loads(raw)
    if hashlib.sha256(raw).hexdigest()!=RECEIPT_SHA or hashlib.sha256(SOURCE_BYTES[Path(scalar.__file__).name]).hexdigest()!=SCALAR_SHA:
        raise ArithmeticError('Authoritative frozen executed0c3 source and ee8f receipt required')
    iv.dps = 80; mp.mp.dps = 80
    mean = frozen['input_descriptor']['actual_root_mean_input_UNCHANGED']
    g0,gs = scalar.fast.point_mean(mean); g = [exact(g0),exact(gs),exact(gs),exact(gs)]
    native = supplier.quantum.build_native(mean,4)
    geom = normal_geometry(native,g); Kprim,forceprim = primitive_matrix(geom,g,exact(F(1,64)))
    primmid = np.asarray([[centre(x) for x in row] for row in Kprim]); primerr = float(np.max(abs(primmid-geom['K_literal'])))
    if primerr > 1e-11:
        raise ArithmeticError('Full45 primitive literal Hessian disagrees with algebra')
    profiles = scalar.fast.primitive_profiles(mean); tensors = trace_tensors(geom['M'],supplier.primitive.SELECTED)
    algebra = [algebraic_species(p,geom,g,tensors) for p in profiles]
    literal = literal_consumers(native,profiles,geom,algebra)
    quantum = coupled_covariance(geom,Kprim,forceprim,profiles,algebra,frozen)
    quantum['actual_executed_nonlinear_mean_response'] = nonlinear_mean_response(native,profiles,geom,algebra,quantum)
    quantum['actual_omitted_dynamical_EM_cross_price'] = electromagnetic_cross_price(profiles,algebra,g)
    descriptor = dict(plan=frozen['input_descriptor']['plan'],scientific_execution_revision=REVISION,observed_governing_plan_revision=OBSERVED_REVISION,template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',input_receipts=dict(conditional0c3=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())),actual_root_mean_input_UNCHANGED=mean,actual_point=dict(g0=str(g0),gs=str(gs),g0_float=float(g0),gs_float=float(gs),interpretation='Exact executed0c3 binary point; NOT old .418725/.063222 Gram, NOT original microscopic true-mean interval'),actual_parameters=dict(N=4,V=64,f='1/8',q=[0,0,0],a='1/64',UEphys=1,u=64,Lambda='2^8192',chi='2^-4096',G='8chi',charged_loads=[1,63],neutral_heavy_load=1,alignment_gradient=0),new_law=quantum['Hamiltonian'],original_SOURCE_receiver_probability_UNCHANGED=True)
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),actual_physical60Cart_to45normal_embedding=dict(labels=geom['labels'],grade_words=geom['words'],grade_to_normal=geom['Q'].tolist(),physical_Cart_to_normal=geom['Qcart'].tolist(),physical_primitive_embedding=geom['E'].tolist(),errors=geom['errors'],full45_primitive_Hessian_divided_Lambda=primmid.tolist(),primitive_literal_algebraic_error=primerr,Phi35_index=geom['index35'],uniform_alignment_gradient=0,uniform_derivation='Each site has identical deltaPhi/8. Every endpoint and intermediate one/two-link path is translated identically; nearest periodic alignment bond differences vanish identically. No Lambda^(5/4) q0 stiffness.'),actual_full45_ALLspecies_literal_and_dynamic_consumers=literal,actual_full45_quantum_collective_consumer=quantum,density_selection_derivation='Uniform rho=I/8 has Poccupied I Pempty=0. Nonuniform rho modes carry q!=0; all45 uniform endpoint Phi derivatives carry only q0. Momentum conservation kills the PH cross for ALL63 modes regardless of Clifford grade. Literal full256 frequency rows retain numerical residuals, not zero substitutions. Phi-to-Aholonomy is NOT assumed to vanish: all45 actual currents and mixed contacts are retained and globally norm-priced, but dynamical EM is outside this Phi+PH Gaussian law.',exact_algebraic_ALLspecies_PH_rows=[dict(role=p['role'],load=p['load'],points=alg['points'],complex_PH_Gram=encode(alg['C'])) for p,alg in zip(profiles,algebra,strict=True)],source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=mp.__version__,interval_decimal_digits=iv.dps),elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scalar',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); result = run(args.scalar)
    if not all(result['source_unchanged_since_import'].values()):
        raise ArithmeticError('Scientific source changed during actual full45 execution')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    q = result['actual_full45_quantum_collective_consumer']
    print(json.dumps(dict(input_sha256=result['input_sha256'],stiffness_lower=q['certified_eigenvalue_lower_divided_Lambda'],full45_Phi35_variance=q['actual_full45_Phi35_variance'],full_vs_scalar=q['actual_full_vs_scalar_Phi35_variance_ratio'],Phi35_offdiagonal=q['Phi35_not_eigenmode_offdiagonal_Hessian_norm'],force_norm=q['actual_force_norm_divided_Lambda'],elapsed_seconds=result['elapsed_seconds']),indent=2))
