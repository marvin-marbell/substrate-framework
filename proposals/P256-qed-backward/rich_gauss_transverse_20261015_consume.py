"""Exact full-boundary Hodge discrimination and a new same-profile two-field process."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.special import jv
import sympy as sy
from rich_open_gauss_carrier_20261015_model import NeutralCube, LAW_ID

ROOT = Path(__file__).resolve().parent
P = 'rich_gauss_transverse_20261015_'


def save(name: str, value: Any) -> None:
    (ROOT/(P+name+'.json')).write_text(json.dumps(value, indent=2)+'\n')


def encode(a: Any) -> dict[str, Any]:
    a = np.asarray(a)
    return dict(real=a.real.tolist(), imag=a.imag.tolist())


def decode(a: dict[str, Any]) -> Any:
    return np.asarray(a['real'])+1j*np.asarray(a['imag'])


def maximum(a: Any) -> float:
    return float(np.max(np.abs(a.data))) if a.nnz else 0.0


def evolve(h: Any, v: Any, t: float) -> Any:
    if t == 0:
        return v.astype(complex).copy()
    return expm_multiply(-1j*t*h, v, traceA=-1j*t*h.diagonal().sum())


class Chebyshev:
    def __init__(self, degree: int):
        self.degree = degree
        self.enclosures: dict[int, tuple[float, float]] = {}
        self.tail = 0.0

    def __call__(self, h: Any, v: Any, t: float) -> Any:
        if t == 0:
            return v.astype(complex).copy()
        if id(h) not in self.enclosures:
            d = h.diagonal()
            r = np.asarray(abs(h).sum(axis=1)).ravel()-abs(d)
            self.enclosures[id(h)] = (float(min(d-r)), float(max(d+r)))
        lo, hi = self.enclosures[id(h)]
        center, scale = (lo+hi)/2, (hi-lo)/2
        v0 = v.astype(complex)
        v1 = (h @ v0-center*v0)/scale
        out = jv(0,scale*t)*v0-2j*jv(1,scale*t)*v1
        for k in range(2,self.degree+1):
            v2 = 2*(h @ v1-center*v1)/scale-v0
            out += 2*(-1j)**k*jv(k,scale*t)*v2
            v0,v1 = v1,v2
        self.tail = max(self.tail,float(2*np.sum(abs(jv(np.arange(self.degree+1,self.degree+501),scale*t)))))
        return np.exp(-1j*center*t)*out


def current(m: Any, weights: Any, cut: bool) -> Any:
    # Boundary current is zero: no Hamiltonian term hops through the outward link.
    active = set(range(12))-set(m.bridges) if cut else set(range(12))
    return sum((-weights[e]*m.currents[e] for e in active),sparse.csr_matrix(m.H.shape,dtype=complex))


def field(m: Any, weights: Any) -> Any:
    return sparse.diags(m.electric @ weights/16,format='csr')


def chain(n: Any, c: Any, omega: Any, source: Any, sink: Any,
          first: list[Any], second: list[Any], times: list[float], cut: bool,
          prop: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """All charge-allowed partitions, with actual absolute-time two-sided evolution."""
    _, t1, t2, t3 = times
    hn, hc = (n.Hcut,c.Hcut) if cut else (n.H,c.H)
    fns,fcs = first
    qns,qcs = second
    v1 = prop(hc,source @ omega,t1)
    v2 = prop(hc,np.column_stack([v1]+[f @ v1 for f in fcs]),t2-t1)
    # Order is second-insertion major, first-insertion minor; index0 means absent.
    all_q = [sparse.eye(c.dimension,format='csr')]+qcs
    v3 = prop(hc,np.column_stack([q @ v2[:,k] for q in all_q for k in range(len(fcs)+1)]),t3-t2)
    bra = prop(hn,omega,t3)
    charged = np.asarray([np.vdot(bra,sink.T @ v3[:,k]) for k in range(v3.shape[1])]).reshape(len(qcs)+1,len(fcs)+1)
    n1 = prop(hn,omega,t1)
    n2 = prop(hn,np.column_stack([n1]+[f @ n1 for f in fns]),t2-t1)
    fmean = np.asarray([np.vdot(n1,f @ n1) for f in fns])
    qmean = np.asarray([np.vdot(n2[:,0],q @ n2[:,0]) for q in qns])
    qf = np.asarray([[np.vdot(n2[:,0],q @ n2[:,k+1]) for k in range(len(fns))] for q in qns])
    g = charged[0,0]
    subtraction = charged[1:,0,None]*fmean[None,:]+qmean[:,None]*charged[0,None,1:]+g*qf-2*g*qmean[:,None]*fmean[None,:]
    result = dict(complete=encode(charged[1:,1:]),connected=encode(charged[1:,1:]-subtraction),subtraction=encode(subtraction),
                  BBdag=encode(g), B_second_Bdag=encode(charged[1:,0]), B_first_Bdag=encode(charged[0,1:]),
                  first_mean=encode(fmean),second_mean=encode(qmean),neutral_second_first=encode(qf),
                  times=times, cut=cut, final_charged_chain_norms=np.linalg.norm(v3,axis=0).tolist(),
                  raw_source_strength=float(np.linalg.norm(source @ omega)**2),raw_sink_strength=float(np.linalg.norm(sink @ omega)**2),
                  normalized_neutral_bra_norm=float(np.linalg.norm(bra)),
                  cumulant='raw-GQ*f-GF*q-G*QF+2*G*q*f; other partitions vanish by total charge',
                  interpretation='Raw positive-frequency addition, no -i, timeordering, amputation or probability; full generated N8/N9 states, no selected-mode truncation')
    arrays = dict(charged_t1=v1,charged_t2=v2,charged_t3=v3,neutral_bra_t3=bra,neutral_t1=n1,neutral_t2=n2)
    return result, arrays


def resolution(primary: dict[str, Any], alternate: dict[str, Any], a: Any, b: Any) -> dict[str, Any]:
    return dict(complete_max_abs_difference=float(np.max(abs(decode(primary['complete'])-decode(alternate['complete'])))),
                connected_max_abs_difference=float(np.max(abs(decode(primary['connected'])-decode(alternate['connected'])))),
                endpoint_column_l2_difference=np.linalg.norm(a['charged_t3']-b['charged_t3'],axis=0).tolist(),
                neutral_bra_l2_difference=float(np.linalg.norm(a['neutral_bra_t3']-b['neutral_bra_t3'])),
                alternate_complete=alternate['complete'],alternate_connected=alternate['connected'])


def hodge(m: Any) -> tuple[Any, Any, dict[str, Any]]:
    b = sy.Matrix(m.full_incidence.astype(int))
    gram = b*b.T
    pl = b.T*gram.inv()*b
    pt = sy.eye(13)-pl
    assert b.rank() == 8 and pt.rank() == 5
    assert pt*pt == pt and pl*pl == pl and b*pt == sy.zeros(8,13) and pt*pl == sy.zeros(13,13)
    ptn,pln = np.asarray(pt,dtype=float),np.asarray(pl,dtype=float)
    fixed = -np.asarray(b.T*gram.inv(),dtype=float)
    residual = float(np.max(abs(m.electric @ pln.T-m.rho @ fixed.T)))
    return ptn,pln,dict(incidence=m.full_incidence.tolist(),incidence_rank=8,cycle_rank=5,
                       PT_exact=[[str(x) for x in row] for row in pt.tolist()],PL_exact=[[str(x) for x in row] for row in pl.tolist()],
                       exact_rational_idempotence_orthogonality_and_BPT=True,all_basis_EL_equals_minus_BT_inverse_BBT_rho_residual=residual,
                       boundary='8 physical sites x13 edges; outward edge retained; boundary row/column of PT exactly zero. Hcut changes hopping, not field graph.',
                       meaning='ET is independent graph-cycle coordinate; EL is Gauss-slaved density. Cycle rank is not helicity/polarization count.')


def main() -> None:
    start = time.monotonic()
    settings = json.loads((ROOT/(P+'input.json')).read_text())
    for key in ('model','reference'):
        if hashlib.sha256((ROOT/settings[key]).read_bytes()).hexdigest()!=settings[key+'_SHA256']:
            raise ValueError('Frozen input mismatch: '+key)
    n,c = NeutralCube(),NeutralCube(5,4)
    ref = np.load(ROOT/settings['reference'])
    if str(ref['basis_SHA256'])!=n.basis_hash or not np.array_equal(ref['electric'],n.electric):
        raise ValueError('Shared reference basis mismatch')
    omega = ref['psi'].astype(complex)
    source,sink = n.charged_source(c,0,0,True),n.charged_source(c,4,0,True)
    pt,pl,geo = hodge(n)
    _,_,cg = hodge(c)
    save('equations',dict(hodge=geo,charged_Gauss_residual=cg['all_basis_EL_equals_minus_BT_inverse_BBT_rho_residual'],
        law=LAW_ID,dimensions=[n.dimension,c.dimension],basis_SHA256=[n.basis_hash,c.basis_hash],
        reference_energy=float(ref['energy']),reference_residual=float(np.linalg.norm(n.H @ omega-float(ref['energy'])*omega)),
        shared_reference='Declared INPUT, not an independent ground solve',physical_signs='B E+rho=0; rho=-Qphysical; jQ=-Jflux; F=E/16; dot F=-jQ/16',
        current_Hodge='Project the actual active-edge physical current vector, with outward current=0; jT+jL=j. No requirement that decomposed cut pieces individually vanish.',
        nonstationary_cut='Both neutral and charged branches evolved under Hcut, including bra and all neutral disconnected terms; no full-H eigenphase shortcut.'))
    e = np.eye(13)[int(n.right_edges[0])]
    j = np.eye(13)[int(n.bridges[0])]
    fs,js = [e,pt @ e,pl @ e],[j,pt @ j,pl @ j]
    fields = [[field(m,w) for w in fs] for m in (n,c)]
    poly = Chebyshev(settings['chebyshev_degree'])
    results: dict[str, Any] = {}
    saved: dict[str, Any] = dict(PT=pt,PL=pl,Omega=omega)
    for name,cut in [('full',False),('cut',True)]:
        currents = [[current(m,w,cut) for w in js] for m in (n,c)]
        primary,arrays = chain(n,c,omega,source,sink,fields,currents,settings['times'],cut,evolve)
        for label in ('complete','connected','subtraction'):
            a = decode(primary[label])
            primary[label+'_reconstruction_error'] = float(abs(a[0,0]-a[1:,1:].sum()))
        primary['axis_order'] = 'rows physical current [edge,T,L], columns F [edge,T,L]; retain all four TT,TL,LT,LL including complex interference'
        primary['profiles'] = dict(field=[w.tolist() for w in fs],current=[w.tolist() for w in js],edge=n.edges[int(n.right_edges[0])],bridge=n.edges[int(n.bridges[0])])
        if not cut:
            save('first_channel',dict(kind='FIRST_ACTUAL_EXACT_HODGE_PHYSICAL_CHANNEL',result=primary,hodge=geo,
                original_review28_complete=[0.00003238122288106869,0.0006578625592134861],
                original_review28_connected=[-0.0005753916977985557,0.0011249881411240087],
                retained_symmetry_null='Review28 first right-circulation/bridge channel ~2e-17; unchanged null. New TWO same-profile field channel explicitly allowed and not yet reported here.'))
            print(json.dumps(dict(event='FIRST_HODGE_CHANNEL',complete=primary['complete'],connected=primary['connected'],reconstruction=primary['connected_reconstruction_error'])),flush=True)
        alternate,other = chain(n,c,omega,source,sink,fields,currents,settings['times'],cut,poly)
        primary['numerical_resolution'] = resolution(primary,alternate,arrays,other)
        primary['actual_active_H_current_split_residual'] = maximum(currents[1][0]-currents[1][1]-currents[1][2])
        results[name+'_mixed_hodge'] = primary
        saved.update({name+'_mixed_'+k:v for k,v in arrays.items()})
    # Explicit new reflection-even protocol: identical normalized left circulation.
    profile = np.r_[n.left_cycle/np.linalg.norm(n.left_cycle),0.0]
    ff = [[field(m,profile)] for m in (n,c)]
    collective: dict[str, Any] = dict(profile=profile.tolist(),divergence_max=float(np.max(abs(n.full_incidence @ profile))),
        protocol='B4(t3) F_leftcycle(t2) F_leftcycle(t1) B0dag(0), F=sum normalized_cycle E/16. New even protocol, not old odd channel.',
        microscopic='Same diagonal physical E profile in EVERY actual N8/N9 basis; no neutral rank1 ladder transplant. Full generated states, no evolved data imported.',
        mode_relation='Exact same normalized electric profile as source of observed gap .81420752854. That earlier selected ray carried .24465991185 of source weight, not a complete group; present calculation keeps ALL weight.',
        neutral_F_raw_strength=float(np.linalg.norm(ff[0][0] @ omega)**2),charged_F_on_raw_source_strength=float(np.linalg.norm(ff[1][0] @ (source @ omega))**2),
        H_F_commutator_max=maximum(n.H @ ff[0][0]-ff[0][0] @ n.H),
        source_commutator_max=maximum(ff[1][0] @ source-source @ ff[0][0]),sink_commutator_max=maximum(ff[1][0] @ sink-sink @ ff[0][0]))
    for name,cut in [('full',False),('cut',True)]:
        primary,arrays = chain(n,c,omega,source,sink,ff,ff,settings['times'],cut,evolve)
        alternate,other = chain(n,c,omega,source,sink,ff,ff,settings['times'],cut,poly)
        primary['numerical_resolution'] = resolution(primary,alternate,arrays,other)
        equal,_ = chain(n,c,omega,source,sink,ff,ff,[0.0]*4,cut,evolve)
        primary['equal_time_all_zero'] = equal
        primary['distinct_minus_equal_connected'] = encode(decode(primary['connected'])-decode(equal['connected']))
        collective[name] = primary
        saved.update({name+'_collective_'+k:v for k,v in arrays.items()})
        print(json.dumps(dict(event='NEW_COLLECTIVE_RESULT',variant=name,result=primary)),flush=True)
    results['collective'] = collective
    results['numerical_method'] = dict(primary='SciPy adaptive Taylor expm_multiply; full sparse H and all generated columns',alternate='Independent full-block Chebyshev recurrence',degree=poly.degree,enclosures=list(poly.enclosures.values()),max_500term_Bessel_tail=poly.tail)
    results['scope'] = dict(law=LAW_ID,no_new_law=True,no_borrowed_evolved_states=True,
        first_odd_channel='Review28 initial circulation/bridge amplitude ~2e-17 retained as reflection-null, not no-interaction.',
        causal='Review28 operational purefield-to-charge/remote-energy contrast and path-cut defeat are prior independent evidence. The cut full bridge-current zero is tautological. Nonlocal T/L pieces can be nonzero and cancel; two-field correlations under cut need not vanish.',
        limits='Exact finite independent-cycle decomposition, not Maxwell photons/polarizations/helicities, event probabilities, on-shell scattering or irreducible Dyson object. No source-state-law splice; frozen I16 cube, not triangle frame law.')
    results['elapsed_seconds'] = time.monotonic()-start
    np.savez_compressed(ROOT/(P+'states.npz'),**saved)
    save('result',results)
    print(json.dumps(dict(event='COMPLETE',elapsed=results['elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    main()
