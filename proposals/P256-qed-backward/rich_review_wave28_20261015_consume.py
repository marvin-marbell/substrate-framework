"""New charged/remote-field experiment and complete ordered connected consumer.

Imports only frozen microscopic construction and declared neutral input, never
producer evolution, spectral states, or producer consumer/checker entrypoints.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.special import jv
from rich_open_gauss_carrier_20261015_model import NeutralCube, LAW_ID

ROOT = Path(__file__).resolve().parent
P = 'rich_review_wave28_20261015_'


def save(name: str, value: Any) -> None:
    (ROOT/(P+name+'.json')).write_text(json.dumps(value, indent=2)+'\n')


def z(value: complex) -> list[float]:
    return [float(np.real(value)), float(np.imag(value))]


def maximum(a: Any) -> float:
    return float(np.max(np.abs(a.data))) if a.nnz else 0.0


def comm(a: Any, b: Any) -> Any:
    return a @ b-b @ a


def evolve(h: Any, state: Any, duration: float) -> Any:
    if duration == 0:
        return state.copy().astype(complex)
    return expm_multiply((-1j*duration)*h, state, traceA=(-1j*duration)*h.diagonal().sum())


class Polynomial:
    """Independent Chebyshev propagation; Gershgorin includes entire block."""
    def __init__(self, degree: int):
        self.degree = degree
        self.enclosures: dict[int, tuple[float, float]] = {}
        self.max_tail = 0.0

    def __call__(self, h: Any, state: Any, duration: float) -> Any:
        if duration == 0:
            return state.copy().astype(complex)
        if id(h) not in self.enclosures:
            d = h.diagonal()
            r = np.asarray(abs(h).sum(axis=1)).ravel()-np.abs(d)
            self.enclosures[id(h)] = (float(np.min(d-r)), float(np.max(d+r)))
        low, high = self.enclosures[id(h)]
        c, a = (high+low)/2, (high-low)/2
        v0 = state.astype(complex)
        v1 = (h @ v0-c*v0)/a
        out = jv(0,a*duration)*v0-2j*jv(1,a*duration)*v1
        for k in range(2,self.degree+1):
            v2 = 2*(h @ v1-c*v1)/a-v0
            out += 2*(-1j)**k*jv(k,a*duration)*v2
            v0,v1 = v1,v2
        self.max_tail = max(self.max_tail,float(2*np.sum(np.abs(jv(np.arange(self.degree+1,self.degree+501),a*duration)))))
        return np.exp(-1j*c*duration)*out


def variant(model: Any, cut: bool) -> tuple[Any, Any, list[Any], list[Any]]:
    active = set(range(12))-set(model.bridges) if cut else set(range(12))
    zero = sparse.csr_matrix(model.H.shape,dtype=complex)
    return (model.Hcut if cut else model.H, model.HRcut if cut else model.HR,
            [j if e in active else zero for e,j in enumerate(model.currents)],
            [k if e in active else zero for e,k in enumerate(model.contacts)])


def measure(m: Any, states: Any, cut: bool) -> dict[str, Any]:
    h,hr,js,ks = variant(m,cut)
    p = np.abs(states)**2
    def exp(a: Any) -> Any:
        return np.sum(states.conj()*(a @ states),axis=0).real.tolist()
    qr = np.sum(m.rho[:,m.right_sites],axis=1)
    jvals = np.asarray([exp(j) for j in js]).T
    powers = np.asarray([np.sum(states.conj()*m.electric[:,e,None]*(j @ states),axis=0).real/16 for e,j in enumerate(js)]).T
    return dict(norm=p.sum(axis=0).tolist(), H=exp(h), own_HR=exp(hr),
                rho=(m.rho.T @ p).T.tolist(), rho_right=(qr @ p).tolist(),
                physical_Q_right=(-qr @ p).tolist(),
                ER=(m.ER @ p).tolist(), EL=((np.sum(m.electric[:,m.left_edges].astype(float)**2,axis=1)/32) @ p).tolist(),
                flux=(m.electric.T @ p).T.tolist(), physical_field=(m.electric.T @ p/16).T.tolist(),
                flux_current=jvals.tolist(), physical_charge_current=(-jvals).tolist(),
                contact=np.asarray([exp(k) for k in ks]).T.tolist(),
                J2=np.asarray([np.sum(abs(j @ states)**2,axis=0) for j in js]).T.tolist(),
                electric_power=powers.tolist(), right_electric_power=powers[:,m.right_edges].sum(axis=1).tolist(),
                own_right_energy_power=(-2*np.sum((h @ states).conj()*(hr @ states),axis=0).imag).tolist())


def ordered_connected(n: Any, c: Any, omega: Any, source: Any, sink: Any,
                      times: list[float], cut: bool, prop: Callable[..., Any]) -> dict[str, Any]:
    """Absolute-time two-sided chain, also valid for nonstationary cut Ω."""
    _,t1,t2,t3 = times
    hn,_,jn,_ = variant(n,cut)
    hc,_,jc,_ = variant(c,cut)
    face = next(f for f in n.faces if np.all(n.positions[np.flatnonzero(np.any(n.incidence[:,np.flatnonzero(f)]!=0,axis=1)),0]==1))
    fn = sparse.diags(n.electric[:,:12] @ face/16,format='csr')
    fc = sparse.diags(c.electric[:,:12] @ face/16,format='csr')
    edge = int(c.bridges[0])
    qn,qc = -jn[edge],-jc[edge]
    # Columns: no field insertion and F(t1) insertion. The J insertion is
    # applied independently, so complete and all charge-allowed partitions
    # are retained, not merely the spectral-source projection.
    ct1 = prop(hc,source @ omega,t1)
    at2 = prop(hc,np.column_stack((ct1,fc @ ct1)),t2-t1)
    at3 = prop(hc,np.column_stack((at2[:,0],qc @ at2[:,0],at2[:,1],qc @ at2[:,1])),t3-t2)
    nt = prop(hn,omega,t3)
    charged = np.asarray([np.vdot(nt,sink.T @ at3[:,k]) for k in range(4)])
    g,gj,gf,raw = charged
    nt1 = prop(hn,omega,t1)
    nt2 = prop(hn,nt1,t2-t1)
    fmean = np.vdot(nt1,fn @ nt1)
    jmean = np.vdot(nt2,qn @ nt2)
    jf = np.vdot(nt2,qn @ prop(hn,fn @ nt1,t2-t1))
    subtraction = gj*fmean+gf*jmean+g*jf-2*g*jmean*fmean
    connected = raw-subtraction
    return dict(complete=z(raw), connected=z(connected), subtraction=z(subtraction),
                BBdag=z(g), BJBdag=z(gj), BFBdag=z(gf), background_JF=z(jf),
                background_F=z(fmean), background_J=z(jmean),
                charged_chain_norms=np.linalg.norm(at3,axis=0).tolist(), neutral_bra_norm=float(np.linalg.norm(nt)),
                physical_current_edge=edge, signed_branch='positive-frequency addition; raw B_sink(t3) jQ(t2) F(t1) B_source†(0); no -i, time-ordering or amputation',
                cumulant='C4=raw-GJ*f-GF*j-G*JF+2*G*j*f; all other partitions vanish by conserved total charge. Field and current means use absolute-time neutral evolution under this variant.',
                probe_units='F=sum face_orientation*E/16; jQ=-J on first +x bridge. Raw fourpoint and cumulant have field times charge-current units; B strengths not normalized.')


def identities(m: Any, generator: Any, cut: bool) -> dict[str, float]:
    h,_,js,ks = variant(m,cut)
    out = dict(gauss=float(np.max(abs(m.electric @ m.full_incidence.T+m.rho))), flux_current=0.0,
               electric_power=0.0, contact=0.0, physical_charge_continuity=0.0,
               pulse_matter_commutator=0.0, pulse_left_field_energy_commutator=0.0)
    for e,(j,k) in enumerate(zip(js,ks)):
        eop = sparse.diags(m.electric[:,e].astype(float),format='csr')
        out['flux_current'] = max(out['flux_current'],maximum(1j*comm(h,eop)-j))
        out['electric_power'] = max(out['electric_power'],maximum(1j*comm(h,eop @ eop/32)-(eop @ j+j @ eop)/32))
        # H(theta)=exp(i theta E) H exp(-i theta E): dH/dtheta=-J,
        # d²H/dtheta²=contact, also on absent edges where both vanish.
        out['contact'] = max(out['contact'],maximum(-comm(eop,comm(eop,h))-k))
    for s in range(8):
        rho = sparse.diags(m.rho[:,s].astype(float),format='csr')
        div = sum((m.incidence[s,e]*(-j) for e,j in enumerate(js)),sparse.csr_matrix(h.shape))
        out['physical_charge_continuity'] = max(out['physical_charge_continuity'],maximum(1j*comm(h,-rho)+div))
        out['pulse_matter_commutator'] = max(out['pulse_matter_commutator'],maximum(comm(generator,rho)))
    qr = sparse.diags(np.sum(m.rho[:,m.right_sites],axis=1).astype(float),format='csr')
    el = sparse.diags(np.sum(m.electric[:,m.left_edges].astype(float)**2,axis=1)/32,format='csr')
    out['regional_charge_commutator'] = maximum(comm(h,qr))
    out['pulse_left_field_energy_commutator'] = maximum(comm(generator,el))
    return out


def main() -> None:
    start = time.monotonic()
    settings = json.loads((ROOT/(P+'input.json')).read_text())
    for key in ('model','reference'):
        if hashlib.sha256((ROOT/settings[key]).read_bytes()).hexdigest()!=settings[key+'_SHA256']:
            raise ValueError('Frozen input hash mismatch: '+key)
    n,c = NeutralCube(),NeutralCube(5,4)
    ref = np.load(ROOT/settings['reference'])
    omega = ref['psi'].astype(complex)
    if str(ref['basis_SHA256'])!=n.basis_hash or not np.array_equal(ref['electric'],n.electric) or not np.array_equal(ref['old_indices'],n.old_indices) or not np.array_equal(ref['fock'],n.fock):
        raise ValueError('Owning neutral basis mismatch')
    source = n.charged_source(c,settings['creation_site'],settings['primitive_component'],True)
    sink = n.charged_source(c,settings['sink_site'],settings['primitive_component'],True)
    raw = source @ omega
    weight = float(np.vdot(raw,raw).real)
    psi = raw/np.sqrt(weight)
    face = next(f for f in c.faces if np.all(c.positions[np.flatnonzero(np.any(c.incidence[:,np.flatnonzero(f)]!=0,axis=1)),0]==1))
    w = c.loop(face)
    generator = (w-w.T)/(2j)
    pulse = expm_multiply(-1j*settings['field_pulse']['alpha']*generator,psi,traceA=0)
    initial = np.column_stack((psi,pulse))
    samples: dict[str, Any] = {}
    endpoints: dict[str, Any] = {}
    for name,cut in (('full',False),('cut',True)):
        h,_,_,_ = variant(c,cut)
        state = initial.copy()
        rows = []
        previous = 0.0
        for t in settings['operational_times']:
            state = evolve(h,state,t-previous)
            rows.append(dict(time=t,observables=measure(c,state,cut)))
            previous = t
        samples[name] = rows
        endpoints[name] = state
    correlations = {name:ordered_connected(n,c,omega,source,sink,settings['correlation_times'],cut,evolve)
                    for name,cut in (('full',False),('cut',True))}
    early = dict(kind='FIRST_NEW_CHARGED_FIELD_OPERATION_AND_COMPLETE_CONNECTED_RESULT',law_id=LAW_ID,
                 source_weight=weight,operational=samples,correlation=correlations,
                 interpretation='Independent new same-H charged source, disjoint right pure-field pulse, own-t0 matter motion; raw addition ordered field/current fourpoint with charge-allowed connected partitions. No event or on-shell assertion.')
    save('first_result',early)
    print(json.dumps(dict(event='FIRST_NEW_OPERATION_AND_CONNECTED',source_weight=weight,
                         full_final=samples['full'][-1],cut_final=samples['cut'][-1],correlation=correlations)),flush=True)
    algebra = {name:identities(c,generator,cut) for name,cut in (('full',False),('cut',True))}
    poly = Polynomial(settings['chebyshev_degree'])
    numerical: dict[str, Any] = {}
    alt_ends: dict[str, Any] = {}
    for name,cut in (('full',False),('cut',True)):
        h,_,_,_ = variant(c,cut)
        alt = poly(h,initial,settings['operational_times'][-1])
        alt_ends[name] = alt
        alternate_corr = ordered_connected(n,c,omega,source,sink,settings['correlation_times'],cut,poly)
        numerical[name] = dict(endpoint_l2_difference=np.linalg.norm(alt-endpoints[name],axis=0).tolist(),
                               alternate_endpoint=measure(c,alt,cut), alternate_correlation=alternate_corr,
                               correlation_complete_abs_difference=float(abs(complex(*alternate_corr['complete'])-complex(*correlations[name]['complete']))),
                               correlation_connected_abs_difference=float(abs(complex(*alternate_corr['connected'])-complex(*correlations[name]['connected']))),
                               max_norm_drift=max(max(abs(np.asarray(row['observables']['norm'])-1)) for row in samples[name]),
                               max_energy_drift=max(float(np.max(abs(np.asarray(row['observables']['H'])-samples[name][0]['observables']['H']))) for row in samples[name]))
    numerical['method'] = dict(degree=poly.degree, enclosures=list(poly.enclosures.values()), max_Bessel_tail_500_terms=poly.max_tail,
                               interpretation='Taylor expm_multiply against independent Chebyshev entire-block enclosures; finite numerical estimate excludes roundoff. No producer evolution repeated.')
    result = dict(**early,settings=settings,dimensions=dict(neutral=n.dimension,addition=c.dimension),
                  basis_hash=dict(neutral=n.basis_hash,addition=c.basis_hash),
                  reference=dict(energy=float(ref['energy']),residual=float(ref['residual']),declared_shared_input_not_independent_solve=True),
                  geometry=dict(positions=c.positions.tolist(),edges=c.edges,incidence=c.incidence.tolist(),right_face=face.tolist(),bridges=c.bridges.tolist()),
                  identities=algebra,numerical_resolution=numerical,
                  elapsed_seconds=time.monotonic()-start,
                  signs='Incidence +tail -head; incidence E+rho=0; rho=n-1=-Qphysical. dotE=J; dotrho=-incidence J; jQ=-J; dotF=J/16; dotHE_edge={E,J}/32. H(theta)=e^(i theta E)H e^(-i theta E): first derivative -J, second contact.',
                  limitations='One complete finite microscopic addition block. No physical Noether spin, event probability, on-shell Compton, photons, irreducible vertex, global ground or Nature conclusion. Cut Ω evolved on both sides; full-H stationary eigenphase never substituted. Cut current/contact/power masked at primitive active-edge boundary.')
    save('result',result)
    np.savez_compressed(ROOT/(P+'states.npz'),initial=initial,full_endpoint=endpoints['full'],cut_endpoint=endpoints['cut'],
                        chebyshev_full=alt_ends['full'],chebyshev_cut=alt_ends['cut'],charged_old_indices=c.old_indices,
                        neutral_reference_SHA256=settings['reference_SHA256'])
    print(json.dumps(dict(event='COMPLETED',algebra=algebra,numerical={name:{k:v for k,v in value.items() if k not in ('alternate_endpoint','alternate_correlation')} for name,value in numerical.items()},elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    main()
