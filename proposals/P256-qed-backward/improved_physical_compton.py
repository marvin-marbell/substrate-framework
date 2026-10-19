#!/usr/bin/env python3
"""Range-two full-CAR Compton and actual coupled Phi/photon/source flight."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from fractions import Fraction as F
from typing import Any, Callable, cast
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse.linalg import eigsh, expm_multiply
from flint import arb, ctx
from rotation_band import RotationBand
from sea_selection import SELECTED
import noncompact_harmonic_scattering_fiber as fibre
import noncompact_legal_volume_bloch as legal
import noncompact_original_mic_photon as photon
from noncompact_native_current import exterior, displacement, accept, adverse
from noncompact_actual_receiver_gram import inverse_sqrt
from noncompact_scattering_flight import decimal_bound
from improved_phi_band import selected_band

ROOT = Path(__file__).resolve().parent
N = 5*2**96
UE = F.from_float(1.5429442570866347e-148)/2**4096
C = F(1130202988249777, 9007199254740992)
V = .4990360130634525
G0, GS = .4187253908699568, .0632221974274214


def encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return dict(real=value.real, imag=value.imag)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, F):
        return str(value)
    raise TypeError(type(value).__name__)


def decode(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {'real', 'imag'}:
        return complex(value['real'], value['imag'])
    if isinstance(value, dict):
        return {key:decode(item) for key,item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


class ImprovedBloch(fibre.SparseBlochFiber):
    """All inherited Schur, sea and Euclidean jets use these actual derivatives."""
    def __init__(self, points: np.ndarray, frame: np.ndarray, duration: float,
                 band: RotationBand, omit: str = ''):
        super().__init__(points, frame, duration)
        self.band, self.omit = band, omit
        self.endpoint = -band.r*frame[0]-1j*band.v*frame[3]/2
        self.second = band.r*frame[0]/4

    def info(self, shift: np.ndarray) -> dict[str, Any]:
        key = tuple(float(q) for q in shift)
        if key not in self.cache:
            phase = fibre.bounded_phase(self.points+shift)
            mass = self.band.m+self.band.r*np.sum((1-np.real(phase))**2, axis=1)
            coefficient = np.column_stack((mass, self.band.v*np.imag(phase)))
            h = np.einsum('pm,mab->pab', coefficient, self.frame)
            energy = np.sqrt(np.trace(h@h, axis1=1, axis2=2).real/4)
            if np.min(energy) <= 0:
                raise AssertionError('actual improved full-sea gap')
            q = (self.identity+h/energy[:,None,None])/2
            self.cache[key] = dict(h=h,E=energy,Q=q,P=self.identity-q)
        return self.cache[key]

    def vertex(self, row: dict[str, Any], shift: np.ndarray) -> np.ndarray:
        p = self.points[:,2]+shift[2]
        phase = fibre.bounded_phase(p)
        if np.linalg.norm(row['polar']-[0,0,1]) > 2e-12 or abs(row['k'][2]) > 1e-12:
            raise ValueError('same x/y hard photons, literal z polarization required')
        first = phase[:,None,None]*self.endpoint-phase.conj()[:,None,None]*self.endpoint.conj().T
        second = phase[:,None,None]**2*self.second-phase.conj()[:,None,None]**2*self.second.conj().T
        # Separate first insertion on each of the two links: 1+1, not one insertion.
        current = first+(0 if self.omit == 'two_current' else 2*second)
        return 1j*row['coefficient']/math.sqrt(row['omega'])*current

    def mixed(self, incoming: dict[str, Any], outgoing: dict[str, Any]) -> np.ndarray:
        zero = np.zeros(3)
        first, final = self.info(zero), self.info(incoming['k']+outgoing['k'])
        phase = fibre.bounded_phase(self.points[:,2])[:,None,None]
        contact = phase*self.endpoint+phase.conj()*self.endpoint.conj().T
        # Four ordered insertion pairs (11,12,21,22); every pair survives.
        contact += (0 if self.omit == 'two_contact' else 4)*(phase**2*self.second+phase.conj()**2*self.second.conj().T)
        contact *= -incoming['coefficient']*outgoing['coefficient']/math.sqrt(incoming['omega']*outgoing['omega'])
        if self.omit == 'contact':
            contact *= 0
        result = np.zeros_like(contact)
        for so, po in ((1.,final['Q']),(-1.,final['P'])):
            for si, pi in ((1.,first['Q']),(-1.,first['P'])):
                delta = so*final['E']-si*first['E']-incoming['frequency_sign']*incoming['omega']-outgoing['frequency_sign']*outgoing['omega']
                result += (-1j*fibre.entire_time_integral(delta,self.duration))[:,None,None]*(po@contact@pi)
        for early,late in ((incoming,outgoing),(outgoing,incoming)):
            middle = self.info(early['k'])
            je,jl = self.vertex(early,zero),self.vertex(late,early['k'])
            for so,po in ((1.,final['Q']),(-1.,final['P'])):
                for sm,pm in ((1.,middle['Q']),(-1.,middle['P'])):
                    for si,pi in ((1.,first['Q']),(-1.,first['P'])):
                        later = so*final['E']-sm*middle['E']-late['frequency_sign']*late['omega']
                        earlier = sm*middle['E']-si*first['E']-early['frequency_sign']*early['omega']
                        result -= fibre.entire_ordered_integral(later,earlier,self.duration)[:,None,None]*(po@jl@pm@je@pi)
        return result


def static_means(descriptor: dict[str,Any], band: RotationBand, q: int,
                 omit: str = '') -> tuple[complex,complex,complex]:
    incoming,outgoing = legal.represented_packets(float(C))
    first = dict(incoming['modes'][0],coefficient=sum(r['coefficient'] for r in incoming['modes']))
    second = dict(outgoing['modes'][0],coefficient=sum(r['coefficient'] for r in outgoing['modes']))
    static = dict(second,coefficient=second['coefficient']*complex(fibre.bounded_phase(np.asarray(-6*second['omega']))))
    forward = dict(first,coefficient=first['coefficient']*complex(fibre.bounded_phase(np.asarray(-first['k'][0]))))
    bare_photon = complex(fibre.bounded_phase(np.asarray(first['k'][0]-6*first['omega'])))
    frame = np.sqrt([G0,GS,GS,GS])[:,None,None]*SELECTED
    matrix = np.asarray(descriptor['source_matrix'])
    if omit == 'reference':
        matrix = matrix.copy(); matrix[:,1:] = 0
    ds,eta = np.asarray(descriptor['compact_displacement']),np.asarray(descriptor['source_eta'])
    totals = np.zeros(3,complex)
    count = q**3
    for start in range(0,count,2048):
        indices = np.arange(start,min(start+2048,count))
        points = 2*math.pi*np.column_stack(np.unravel_index(indices,(q,)*3))/q
        kernel = ImprovedBloch(points,frame,6.,band,omit)
        source = fibre.compact_fourier(points,ds,eta)
        detector = fibre.compact_fourier(points+first['k']+second['k'],ds,eta)
        final = kernel.info(first['k']+second['k'])
        free = fibre.bounded_phase(-6*final['E'])[:,None,None]*final['Q']+fibre.bounded_phase(6*final['E'])[:,None,None]*final['P']
        raw = kernel.connected_pair(first,static)+legal.cloud_pair(kernel,first,static)
        if omit == 'negative_final':
            raw = final['Q']@raw
        cloud = legal.cloud_pair(kernel,forward,second)
        initial = kernel.info(np.zeros(3))
        bare = fibre.bounded_phase(-6*initial['E'])[:,None,None]*initial['Q']
        for j,block in enumerate((free@raw,cloud,bare)):
            trace = np.trace(matrix.conj().T@block@matrix,axis1=1,axis2=2)/2
            weight = detector.conj()*source if j<2 else abs(source)**2*bare_photon
            totals[j] += np.sum(weight*trace)
    # Exact D/N=1/2 recoil multiplies C and g by -1 before any binary64 arithmetic.
    return -complex(totals[0]/count),-complex(totals[1]/count),complex(totals[2]/count)


class ImprovedFibre(photon.FullFieldFibre):
    """Actual arbitrary-Phi two-site multigraph; heavy BO law remains original."""
    def __init__(self, frame: np.ndarray, kappa: float, occupation: int = 4,
                 external: np.ndarray | None = None):
        super().__init__(frame,kappa,occupation,external)
        self.band = RotationBand(a=1.,m=.2,r=.12,v=V,improved=True)
        self.onsite,self.hops,self.paths = [],[],[]
        external = np.zeros(3) if external is None else external
        for phi in self.fields:
            local = block_diag((.2+4.5*.12)*phi[0,0],(.2+4.5*.12)*phi[1,0])
            self.onsite.append(exterior(local,self.rows))
            hops,paths = [],[]
            for axis in range(3):
                u = self.links[axis]
                endpoint = -.12*phi[:,0]-.5j*V*phi[:,axis+1]
                one = np.zeros((8,8),complex)
                one[:4,4:] = (endpoint[0]@u+u@endpoint[1])/2
                hops.append(exterior(one,self.rows)); paths.append((axis,))
                # On the directed two-vertex graph, second link goes back, with
                # separately owned SU4/U1 link equal to the same axis link.
                two = np.zeros((8,8),complex)
                for site in range(2):
                    two[4*site:4*site+4,4*site:4*site+4] = .12/8*(phi[site,0]@u@u+u@u@phi[site,0])
                hops.append(exterior(two,self.rows)); paths.append((axis,axis))
            self.hops.append(hops)
            self.paths = paths
        phases = []
        for path in self.paths:
            s = self.parameters[:,list(path)].sum(axis=1)
            nu = sum((1,1,-2)[edge] for edge in path)
            shift = np.zeros((self.l,self.l),complex)
            for j in range(self.l):
                if 0<=j+nu<self.l:
                    shift[j+nu,j] = 1
            # Compress the literal complete path exponential ONCE. Multiplying
            # truncated single-link matrices would introduce a fake boundary.
            phases.append(np.exp(1j*sum(external[list(path)]))*np.kron(np.kron(displacement(self.n,s[0]),displacement(self.n,s[1])),shift))
        self.phase = phases

    def current(self, v: np.ndarray, weights: np.ndarray, contact: bool = False) -> np.ndarray:
        wave,out = self.reshape(v),np.zeros_like(self.reshape(v))
        for s in range(self.nodes):
            for path,z,t in zip(self.paths,self.phase,self.hops[s]):
                w = float(sum(weights[list(path)]))
                if contact:
                    out[s] -= w*w*(z@wave[s]@t.T+z.conj().T@wave[s]@t.conj())
                else:
                    out[s] += 1j*w*(z@wave[s]@t.T-z.conj().T@wave[s]@t.conj())
        return out.ravel()


def quantum_flight() -> dict[str,Any]:
    frame = np.sqrt([G0,GS,GS,GS])[:,None,None]*SELECTED
    nodes,quadrature = np.polynomial.legendre.leggauss(4)
    weights = quadrature*15/16*(1-nodes*nodes)**2
    accept(abs(sum(weights)-1),'normalized paired-Bloch density',2e-14)
    cases = []
    global_gram = np.zeros((2,2),complex)
    global_cloud = np.zeros((2,2),complex)
    global_overlap = np.zeros(2,complex)
    for index in range(4):
        kappa = .08*float(nodes[index])
        neutral,charged = ImprovedFibre(frame,kappa),ImprovedFibre(frame,kappa,5)
        rng = np.random.default_rng(2221007+index)
        solver = cast(Callable[...,Any],eigsh)
        ev,vec = solver(neutral.operator,k=1,which='SA',tol=2e-10,v0=rng.normal(size=neutral.size))
        omega,E0 = vec[:,0],float(ev[0])
        residual = accept(np.linalg.norm(neutral.action(omega)-E0*omega),'improved quantumPhi fullCAR neutral residual',2e-7)
        matrices = photon.source_matrices(neutral,charged,frame)
        sources = [photon.apply_source(neutral,charged,matrices,omega,i==1) for i in range(2)]
        outputs = [np.array([charged.photon(w,i,True) for w in sources[i]])/math.sqrt(2) for i in range(2)]
        gram = np.array([[np.vdot(a,b) for b in outputs] for a in outputs])
        cloud = np.zeros((2,2),complex)
        for i in range(2):
            for j in range(2):
                left = photon.apply_source(neutral,charged,matrices,neutral.photon(omega,j),i==1)
                right = photon.apply_source(neutral,charged,matrices,neutral.photon(omega,i),j==1)
                cloud[i,j] = np.vdot(left,right)/2
        initial = outputs[0].T
        evolved = expm_multiply(-6j*charged.operator,initial,traceA=0.)
        accept(abs(np.linalg.norm(evolved)-np.linalg.norm(initial)),'improved T6 charged field/source norm',2e-7)
        overlap = np.array([np.vdot(w.T,evolved) for w in outputs])
        global_gram += weights[index]*gram
        global_cloud += weights[index]*cloud
        global_overlap += weights[index]*overlap
        row: dict[str,Any] = dict(kappa=kappa,E0=E0,residual=residual,source_references=4,outputs=2,cloud=cloud)
        if index == 2:
            w = np.array([.7,-.4,.2]); d = 2e-5
            plus,minus = ImprovedFibre(frame,kappa,external=d*w),ImprovedFibre(frame,kappa,external=-d*w)
            current = (plus.action(omega)-minus.action(omega))/(2*d)
            contact = (plus.action(omega)+minus.action(omega)-2*neutral.action(omega))/d**2
            row['current_error'] = accept(np.linalg.norm(current-neutral.current(omega,w)),'actual separate range2 current insertions',2e-8)
            row['contact_error'] = accept(np.linalg.norm(contact-neutral.current(omega,w,True)),'all four range2 contact insertions',3e-5)
            row['missing_current_rejected'] = adverse(np.linalg.norm(current),'actual improved current',2e-8)
            row['missing_contact_rejected'] = adverse(np.linalg.norm(contact),'actual improved contact',3e-5)
            row['missing_reference_rejected'] = adverse(abs(np.linalg.norm(sources[0])**2-np.linalg.norm(sources[0][0])**2),'allRef4 actual source',1e-5)
        cases.append(row)
    inverse = inverse_sqrt(global_gram)
    norm = math.sqrt(global_gram[0,0].real)
    amplitude = inverse@global_overlap/norm
    accept(np.linalg.norm(inverse@global_gram@inverse-np.eye(2)),'actual global2x2 Loewdin metric',2e-9)
    wrong = global_overlap/norm/np.sqrt(np.real(np.diag(global_gram)))
    rejected = adverse(np.linalg.norm(amplitude-wrong),'global versus diagonal receiver metric',1e-8)
    return dict(scope='executed changed-law coupled Phi Cartesian slice/fullCAR/two photons/harmonic fibres/source allRef4/T6; not astronomical native volume simulation',
        neutral_dimension=3*3*3*5*70,charged_dimension=3*3*3*5*56,
        envelope=dict(sigma=.08,nodes=.08*nodes,weights=weights,continuum_norm_squared=1,finite_energy=True,global_eigen_ground=False),
        fibres=cases,GLOBAL2x2_Gram=global_gram,crossed_cloud=global_cloud,
        actual_global_input_norm=norm,amplitude_T6=amplitude,probability_T6=abs(amplitude[1])**2,
        missing_globalmetric_rejected=rejected,per_fibre_rescaling=False,
        heavy_scalar='unchanged original heavy BO scalar on finite nonClifford Phi slice')


def legal_enclosure(descriptor: dict[str,Any], q: int,
                    means: tuple[complex,complex,complex]) -> dict[str,Any]:
    """Outward range-two strip, arithmetic, metric, splitting and source disk."""
    ctx.prec = 256
    sigma, distance = .3, 1.535245e-12
    g0,gs = arb(G0),arb(GS)
    mu = .2*math.sqrt(G0-distance-distance**2/(2*(GS-distance)))
    t1 = .12*math.sqrt(G0+distance)+V*math.sqrt(GS+distance)/2
    t2 = .12*math.sqrt(G0+distance)/4
    B = (.2+4.5*.12)*math.sqrt(G0+distance)+6*(t1+t2)
    incoming,outgoing = legal.represented_packets(float(C))
    fi,fo = (abs(sum(row['coefficient'] for row in packet['modes']))/math.sqrt(packet['modes'][0]['omega'])
        for packet in (incoming,outgoing))
    # Clifford norm of the literal z derivative: max|2sin z-sin2z|=3sqrt3/2;
    # max|2cos z-2cos2z|=4. Orthogonal beta/alpha components square-add.
    jcap = math.sqrt((.12*math.sqrt(G0+distance)*3*math.sqrt(3)/2)**2+V*V*(GS+distance))
    zcap = math.sqrt((4*.12*math.sqrt(G0+distance))**2+V*V*(GS+distance))
    ji,jo,k = jcap*fi,jcap*fo,zcap*fi*fo
    oi,oo = (min(row['omega'] for row in packet['modes']) for packet in (incoming,outgoing))
    limits = dict(P=1.,Q=1.,a=6*ji,b=6*jo,c=6*k+36*ji*jo,
        X=jo/(2*mu+oo),Y=ji/(2*mu+oi),Y0=ji/(2*mu+oi))
    ds,eta = np.asarray(descriptor['compact_displacement']),np.asarray(descriptor['source_eta'])
    l1 = float(np.sum(abs(eta)))
    majorant = 2*l1*l1*fibre.word_norms(limits)
    # Same explicit Horner phase/sinc arithmetic as the inherited entire kernels;
    # range two squares an already enclosed phase, paying an extra factor four.
    e,u = 4e-10,2.**-52
    eh = 8*B*e
    ee = B*eh/mu+100*u*B*B/mu+e
    ep = (eh+B*ee/mu)/(2*mu)+100*u*(1+B/mu)
    eji,ejo,ek = (j*(2*e+100*u) for j in (ji,jo,k))
    ef,eo = 24*e+36*ee,300*36*e+4*216*ee/3
    ea = 4*(6*(2*ep*ji+eji)+ji*ef+600*u*ji)
    eb = 4*(6*(2*ep*jo+ejo)+jo*ef+600*u*jo)
    ec = 4*(6*(2*ep*k+ek)+k*ef)+16*(18*(3*ep*ji*jo+eji*jo+ji*ejo)+ji*jo*eo+14400*u*ji*jo)
    ex = (2*ep*jo+ejo)/(2*mu+oo)+2*ee*jo/(2*mu+oo)**2+100*u*jo/(2*mu+oo)
    ey = (2*ep*ji+eji)/(2*mu+oi)+2*ee*ji/(2*mu+oi)**2+100*u*ji/(2*mu+oi)
    errors = dict(P=ep,Q=ep,a=ea,b=eb,c=ec,X=ex,Y=ey+13*ee*ji/(2*mu+oi),Y0=ey)
    arithmetic = 1.01*(2*l1*l1*fibre.word_error(limits,errors)+majorant*(1000*u+4*e+400*u*q**3))
    # beta*h is accretive on the full complex polytube. Re(1-cos z)^2
    # >=-sinh(sigma)^2; the imaginary kinetic vector has length <=sqrt3*sinh.
    sh,ch = arb(sigma).sinh(),arb(sigma).cosh()
    cbar = g0.sqrt()*(arb(.2)-3*arb(.12)*sh**2)-arb(V)*(3*gs).sqrt()*sh
    if not cbar > 0:
        raise AssertionError('improved Wilson complex full-band accretivity')
    bc = g0.sqrt()*(arb(.2)+3*arb(.12)*(1+ch)**2)+arb(V)*(3*gs).sqrt()*ch
    imh = 6*arb(.12)*g0.sqrt()*(1+ch)*sh+arb(V)*(3*gs).sqrt()*sh
    projector,flight = (1+bc/cbar)/2,(12*imh).exp()
    jc_i,jc_o = (2*(arb(t1)*arb(sigma).exp()+2*arb(t2)*(2*arb(sigma)).exp())*a for a in (fi,fo))
    kc = 2*(arb(t1)*arb(sigma).exp()+4*arb(t2)*(2*arb(sigma)).exp())*fi*fo
    strip_limits = dict(P=projector,Q=projector,a=6*flight*jc_i,b=6*flight*jc_o,
        c=6*flight*kc+36*flight**2*jc_i*jc_o,
        X=projector**2*jc_o/(2*cbar+oo),
        Y=projector**2*jc_i*flight/(2*cbar+oi),Y0=projector**2*jc_i/(2*cbar+oi))
    strip_l1 = sum((arb(float(abs(a)))*(arb(sigma)*int(np.sum(abs(d)))).exp() for a,d in zip(eta,ds)),arb(0))
    strip_majorant = 2*strip_l1**2*sum((arb(abs(scale))*math.prod(strip_limits[name] for name in word) for scale,word in fibre.CONNECTED_WORDS),arb(0))*flight
    periodic = strip_majorant*(legal.alias_factor((-arb(sigma)*q).exp())+legal.alias_factor(legal.capped_decay(arb(sigma)*N)))
    # Matrix, photon-transfer and harmonic seams: derivative telescoping on real
    # momenta, with all path lengths <=2 and all seven source jets explicit.
    rootlip = 1/(math.sqrt(GS)+math.sqrt(GS-distance))
    hchange = (.2+12*.12+3*V)*rootlip
    frame_error = 64*(1+B/mu)**3*majorant*(1+6*hchange)*distance
    splitting = arb(2)*arb.pi()/N*arb(256*(1+B/mu)**4*(1+majorant)*(1+l1*l1))
    profile = abs(arb(descriptor['source_raw_profile_log_c_interval']))
    profile_error = (4*profile).expm1()
    mark_error = (arb(descriptor['finite_alpha_input_normalized_matrix_error_interval'])
        +arb(descriptor['finite_alpha_output_whitened_matrix_error_interval']))
    mark_product = 2*arb(2).sqrt()*mark_error+mark_error**2
    operator_norm = arb(str(fibre.word_norms(limits)))*arb('1.01')
    source_error = operator_norm*mark_product+2*l1*l1*operator_norm*profile_error
    ec_total = periodic+arithmetic+frame_error+splitting+source_error
    eg = periodic+arithmetic+frame_error+splitting+2*limits['X']*limits['Y0']*mark_product
    em = periodic+arithmetic+frame_error+splitting+mark_product
    c,g,m = means
    radius = ec_total+arb('.5')*(arb(abs(g))*em+arb(abs(m))*eg+eg*em)
    reference = c-.5*g*m
    norm = (arb(reference.real)**2+arb(reference.imag)**2).sqrt()
    if not norm > radius:
        raise ArithmeticError(f'Improved disk nonpositive: periodic={periodic}, arithmetic={arithmetic}, metric={frame_error}, source={source_error}, radius={radius}')
    lower = legal.positive_lower(norm-radius)
    coefficient = legal.positive_lower(arb(lower)/(N**3))
    return dict(sigma=sigma,complex_accretivity_lower=cbar.str(65),
        changed_primitive_norms=limits,mu_lower=mu,B_upper=B,endpoint_one_upper=t1,endpoint_two_upper=t2,
        separate_two_current_insertions=2,separate_two_contact_insertions=4,
        errors=dict(periodic=periodic.str(65),arithmetic=arithmetic,full10_metric=frame_error,
            exact_adjacent_mode_splitting=splitting.str(65),screened_profile=source_error.str(65)),
        GLOBAL_disk_radius=radius.str(65),volume_rescaled_positive_lower=lower,
        exact_legalN_coefficient_positive_lower=coefficient,
        formula='mean(Cstatic)-.5*mean(offGram)*mean(bare_m); global products, noinputwhitening',
        numerical_precision_scope='explicit binary64 Horner kernels plus outward Arb disk',
        actual_probability_certified=False)


def renewed_transfer(descriptor: dict[str,Any], owner: dict[str,Any]) -> dict[str,Any]:
    """Changed Q/Phi pullthrough, SOURCE high tails, and whole flight payer."""
    from dataclasses import replace
    import noncompact_quantitative_normal_budget as normal
    import noncompact_scattering_flight as flight
    import noncompact_original_source_flight as source
    changed = owner['bounds']
    volume = N**3
    desc = flight.compact_descriptor(descriptor)
    # A single deliberately loose pathwise envelope dominates h/J/Z q<=6,
    # every endpoint commutator and spatial propagation sum of the new law.
    # It is a coefficient envelope, not a comparison of two ground states.
    desc.update(tau=F(8),mu=F(1294,10000),B=F(2))
    body = flight.normalized_flight_bound(desc,F(6))
    if body['volume_rescaled_error'] is None:
        raise AssertionError('renewed physical global Gram failed admission')
    old = normal.native_normal_budget(N)
    budget = replace(old,residual_light=512*old.residual_light,
        current_lipschitz=512*old.current_lipschitz,
        exterior_light_constant=8*old.exterior_light_constant,
        exterior_light_relative=F(changed['exterior_C1']))
    moment = changed['renewed_quartic']
    neutral = changed['renewed_actual_neutral']
    crho,cq = F(moment['Crho']),F(moment['CQ'])
    values = {key:F(value) for key,value in neutral['graph_values'].items()}
    shift = F(neutral['photon_form_shift_upper'])
    trial = F(neutral['sharp_trial_upper'])
    omega = F(neutral['omega_min_lower'])
    j_ground = F(neutral['current_ground_norm_upper'])
    mode_weight = source.sqrt_upper(F(2*volume)/omega)
    number = UE*mode_weight**2*j_ground**2/omega**2
    birth_norm = source.sqrt_upper(1+number)
    birth_first = C*source.sqrt_upper(F(12))*birth_norm+source.sqrt_upper(UE)*mode_weight*j_ground
    cd,a4 = F(changed['CD_changed']),F(changed['A4_unchanged'])
    birth_rho2 = source.sqrt_upper(2/a4*(birth_norm*birth_first+(cd+3)*birth_norm**2))
    field = desc['field_sum']*source.sqrt_upper(F(1,volume))
    # The literal path sums dominate J by2(t1+2t2) and Z by2(t1+4t2).
    # Uniform arbitrary Phi/Fock count is paid here without compact-frame substitution.
    jr = 16*volume*(1+F(8))*field
    zr = jr*field
    sources: dict[str,Any] = {}
    for label,support,word,relative in (('scattering19',19,N//2+2,N//2+1),
            ('RAW41',41**3,N//2+60,N//2+60)):
        jets = tuple(F(4)*F(2)**q for q in (4,427,850,1274,1698,2122,2547))
        jets = flight.free_conjugated(jets,F(6),F(16))
        k0,k1,k2 = jets[:3]
        dimension = 75*(support+1)*(word+1)
        graphs = {}
        for name,n0,q0,rho2 in (('cloud',values['n'],values['Q_full'],values['rho2']),
                ('birth',birth_norm,birth_first,birth_rho2)):
            half = source.sqrt_upper(n0*q0+shift*n0*n0)
            kinetic = source.sqrt_upper(2*n0*q0)+n0
            rho = source.sqrt_upper(n0*rho2)
            harmonic = 2*source.sqrt_upper(UE/volume)*source.sqrt_upper(F(3))*k0*relative*half
            harmonic += 3*UE/volume*k0*relative**2*n0
            comm = 2*k1*kinetic+dimension*k2*n0+36*k0*rho
            comm += (body['entire_Coulomb_neutral_upper']+body['entire_Coulomb_charged_upper'])*k0*n0+harmonic
            first = k0*q0+comm+shift*k0*n0
            graphs[name] = source.SourceGraph(source.dyadic_upper(k0*n0),
                source.dyadic_upper(first),source.dyadic_upper(source.sqrt_upper(k0*n0*first)),
                source.dyadic_upper(comm),tuple(source.dyadic_upper(j) for j in jets))
        sources[label] = dict(graphs=graphs,jets=jets,word=word,support=support)
    norm = max(g.norm for s in sources.values() for g in s['graphs'].values())
    first = max(g.first for s in sources.values() for g in s['graphs'].values())
    half = max(g.half for s in sources.values() for g in s['graphs'].values())
    jcap = jr*(crho*norm+cq*source.sqrt_upper(norm*half))
    zcap = zr*(crho*norm+cq*source.sqrt_upper(norm*half))
    # Full harmonic/source derivatives through6 cost <=2^2547; inverse Gram
    # spectrum>=1/8 costs<=2^6; complete real/PRE/POST/contact Schur words
    # with tau8,mu.1294 cost<2^512. The remaining2^1031 reserve dominates
    # every ordinary coefficient of the GLOBAL divided-difference maps.
    metric = F(2)**4096
    target = UE/(volume*2**40)
    observable = metric*(1+norm+first+jcap+zcap)**4*7**4
    delta_log = max(6,2*normal.ceil_log2(128*observable*(1+crho+cq+half)/target))
    delta = F(2)**-delta_log
    cutoff_log = max(1,2*normal.ceil_log2(16*first/delta))
    cutoff = F(2)**cutoff_log
    jet = max(sum(math.comb(6,q)*s['jets'][q] for q in range(7)) for s in sources.values())
    multiplier = source.dyadic_upper(metric*(1+jet)*(1+first+half)*(1+1/omega)**3)
    slope = F(changed['full_CAR_light_rho_coefficient'])
    deep_log = 8*normal.ceil_log2(128*multiplier*(1+trial+slope+crho+crho**2)/delta)
    dlog = deep_log+normal.ceil_log2(28*(1+norm)*multiplier)
    d,gap = F(2)**-dlog,8*cutoff
    neutral_tail = d/gap
    hem_tail = d*d/gap+(trial+d)*neutral_tail**2+slope*crho*source.sqrt_upper(neutral_tail**3)
    birth_error = source.sqrt_upper(neutral_tail**2+hem_tail/omega)
    current_error = jr*crho*source.sqrt_upper(neutral_tail)+F(2)**-deep_log
    cloud_error = (current_error+d*(1+first))/omega**2
    birth_error += 2*neutral_tail**2*(norm+birth_error)
    cloud_error += 2*neutral_tail**2*(1+norm+first)/omega**2
    cy,orbital = F(2)**64,F(2)**50
    boundary_coeff = max((jet+1)*cy**3*2*budget.chart_majorant,
        jet*2*budget.chart_majorant*(budget.soft_moment+budget.stiff_moment)*(1+first),orbital*(1+first))
    value_log = 5*normal.ceil_log2(64*boundary_coeff*multiplier/delta)
    required = max(180375,value_log,
        16*(normal.ceil_log2(budget.residual_kinetic)+dlog+3),
        (16*(normal.ceil_log2(budget.residual_light)+dlog+3)+4)//5,
        16*(budget.cutoff_log2+dlog+3),
        (16*(normal.ceil_log2(budget.current_lipschitz*budget.soft_moment)+deep_log+3)+4)//5,
        4*(cutoff_log+normal.ceil_log2(128/budget.frequency)))
    common = changed['common_Lambda']
    floor = max(int(common['local_log2_Lambda']),
        640*int(common['parameters']['coercivity_reciprocal_inner_exponent']))
    if floor < required:
        raise AssertionError('chosen improved common Lambda cannot pay source strengthening')
    boundary = 2*delta/(64*multiplier)+jet*cy**2*(1+crho**2)*neutral_tail+orbital*d*(1+first)
    initialization = multiplier*(birth_error+cloud_error+delta/(64*multiplier))
    error = initialization+6*d*multiplier+delta/8
    jerror = jr*(crho*error+cq*source.sqrt_upper(error*2*half))
    zerror = zr*(crho*error+cq*source.sqrt_upper(error*2*half))
    occ_error = 16*volume*(2*norm+error)*error
    full_error = metric*7**4*(1+norm+jcap+zcap)**3*(error+jerror+zerror+occ_error+boundary)
    if full_error >= target:
        raise AssertionError('renewed SOURCE/photon/normal fullflight exceeds target')
    for entry in sources.values():
        entry['tails'] = {name:source.spectral_tail(graph,cutoff) for name,graph in entry['graphs'].items()}
    # R3 even compact paired envelope, true finite energy, no eigen-ground claim.
    sigma = F(2)**-262144
    dispersion = 2*source.sqrt_upper(UE/volume*(shift+3*UE/volume*sigma*sigma))*source.sqrt_upper(F(3))*sigma+3*UE/volume*sigma*sigma
    history = F(2)**66048
    source_rho = 16*crho+cq*source.sqrt_upper(half)
    history_needed = 4096*volume*(1+F(changed['current_rho_coefficient'])**2)*(1+source_rho)**2/(C/2)
    if history_needed >= history:
        raise AssertionError('renewed opposite-source-leg current history exceeds supplied finite envelope')
    phase = 6*history*dispersion
    if phase >= F(2)**-80:
        raise AssertionError('actual changed-law paired packet phase too large')
    return dict(body_volume_rescaled_error_log2=normal.ceil_log2(body['volume_rescaled_error']),
        source_photon_combined_absolute_debt_log2=normal.ceil_log2(full_error),
        source_photon_target_log2=normal.ceil_log2(target),
        strengthened_residual_log2=-dlog,source_cutoff_log2=cutoff_log,
        required_log2_Lambda=required,chosen_common_floor_log2=str(floor),
        body_G0_radius_log2=normal.ceil_log2(body['actual_Gram_radius']),
        graph_scope='renewed changed Q/Phi moments and literal two-link coefficients; all true SOURCE tails retained',
        source_graphs_log2={name:{kind:dict(norm=normal.ceil_log2(g.norm),first=normal.ceil_log2(g.first),half=normal.ceil_log2(g.half))
            for kind,g in entry['graphs'].items()} for name,entry in sources.items()},
        all_mode_photon_number_log2=normal.ceil_log2(number),
        packet=dict(sigma='2^-262144',normalized_symmetric_R3=True,finite_energy_upper_log2=normal.ceil_log2(trial+3*UE/volume*sigma*sigma),
            actual_two_sided_dispersion_log2=normal.ceil_log2(dispersion),
            phase_volume_rescaled_log2=normal.ceil_log2(phase),
            actual_history_requirement_log2=normal.ceil_log2(history_needed),
            domain='balanced large-gauge Bloch boundary; entire quartic Phi form; smooth coexact CCR; fixed raw CAR/allRef4',
            global_eigen_ground_claimed=False),
        _body_error=body['volume_rescaled_error'],_source_error=full_error,_phase_error=phase)


def discriminants(descriptor: dict[str,Any], band: RotationBand) -> dict[str,Any]:
    good = static_means(descriptor,band,24)
    target = good[0]-.5*good[1]*good[2]
    rows = {}
    for omit in ('contact','two_contact','two_current','reference','negative_final'):
        wrong = static_means(descriptor,band,24,omit)
        actual = wrong[0]-.5*wrong[1]*wrong[2]
        rows[omit] = dict(wrong_coefficient=actual,
            rejected=adverse(abs(target-actual),'same improved nonforward STATIC coefficient: '+omit,1e-4))
    return dict(quadrature=24,scope='finite coupled Schur/CAR reduced-word adverse cases, not rigorous legal-volume positivity',
        target=target,cases=rows)


def source_hashes() -> dict[str,str]:
    modules = [fibre,legal,photon]
    paths = {Path(__file__).resolve(),ROOT/'improved_physical_compton.md'}
    for module in modules:
        assert module.__file__ is not None
        paths.add(Path(module.__file__).resolve())
    paths.update(ROOT/name for name in ('improved_phi_band.py','improved_phi_band.md',
        'noncompact_scattering_flight.py','noncompact_original_source_flight.py',
        'noncompact_quantitative_normal_budget.py','noncompact_actual_receiver_gram.py',
        'rotation_band.py','sea_selection.py'))
    return {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor',type=Path,required=True)
    parser.add_argument('--quadrature',type=int,default=128)
    parser.add_argument('--quantum-flight',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--enclose',action='store_true')
    parser.add_argument('--band-packet',type=Path)
    parser.add_argument('--discriminants',action='store_true')
    args = parser.parse_args()
    before = source_hashes()
    descriptor = decode(json.loads(args.descriptor.read_text()))
    band,_,calibration = selected_band()
    if band.v != V or band.m != .2 or band.r != .12 or not band.improved:
        raise AssertionError('primitive owner law changed; recompute this consumer')
    c,g,m = static_means(descriptor,band,args.quadrature)
    result: dict[str,Any] = dict(branch='improved Wilson range-two physical Compton',N=N,V=N**3,UE=str(UE),c=str(C),T=6,
        law=dict(m=.2,r=.12,v=V,kinetic='sin q',mass='m+r sum(1-cos q)^2',heavy_unchanged=True),
        source19_FAR_RIGHT=True,allRef4=True,static_outputs=2,input_whitened=False,
        raw_C=c,offGram=g,bare_m=m,GLOBAL_leading_rescaled=c-.5*g*m,
        primitive_owner_calibration=calibration,
        hardq=2*math.pi/5,controlled_spacetime_cutoff=.05,hard_Compton_continuum_accuracy_claimed=False,
        input_sha256=hashlib.sha256(args.descriptor.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes=before,quadrature=args.quadrature)
    if args.quantum_flight:
        result['executed_quantum_flight'] = quantum_flight()
    if args.discriminants:
        result['executed_discriminants'] = discriminants(descriptor,band)
    if args.enclose:
        result['legal_enclosure'] = legal_enclosure(descriptor,args.quadrature,(c,g,m))
        if args.band_packet:
            owner = json.loads(args.band_packet.read_text())
            transfer = renewed_transfer(descriptor,owner)
            enclosure = result['legal_enclosure']
            source_error = transfer.pop('_source_error')
            root_debt = F(2)**int(owner['bounds']['absolute_weighted_debt_log2_upper'])
            physical_debts = transfer.pop('_body_error')+transfer.pop('_phase_error')+N**3*(source_error+root_debt)/UE
            margin = F(enclosure['volume_rescaled_positive_lower'])-physical_debts
            if margin <= 0:
                raise AssertionError('improved actual physical probability lost its margin')
            amplitude = UE*margin/(N**3)
            result['renewed_transfer'] = transfer
            result['actual_probability'] = dict(volume_rescaled_margin_lower=decimal_bound(margin,False),
                amplitude_lower=decimal_bound(amplitude,False),probability_lower=decimal_bound(amplitude*amplitude,False),
                changed_law_supplier_sha256=hashlib.sha256(args.band_packet.read_bytes()).hexdigest(),
                legal_volume_probability_simulated=False,all_debts_counted_once=True)
            result['physical_receiver_probability_certified'] = True
    after = source_hashes()
    if before != after:
        raise RuntimeError('changed consumer dependencies during scientific execution')
    result['natural_exit'] = 0
    if args.output:
        args.output.write_text(json.dumps(result,default=encode,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output) if args.output else None,
        source_hashes=before,input_sha256=result['input_sha256'],
        leading_rescaled=result['GLOBAL_leading_rescaled'],
        actual_probability=result.get('actual_probability'),
        quantum_probability=result.get('executed_quantum_flight',{}).get('probability_T6'),
        natural_exit=0),default=encode,indent=2,allow_nan=False))

if __name__ == '__main__':
    main()
