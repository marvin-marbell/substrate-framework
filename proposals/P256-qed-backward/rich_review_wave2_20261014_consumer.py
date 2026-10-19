"""Independent finite physical consumers; no archived entrypoint is replayed.
All observables are dimensionless with hbar=charge=link spacing=1.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import solve
from scipy.sparse.linalg import expm_multiply, splu
import rich_composite_local3d_20261014_consumer as local
import rich_photon_fullq_20261014_model as photon
import rich_charged_gauss_20261014_model as charged

ROOT = Path(__file__).resolve().parent

def scalar(x):
    return float(np.real(x))

def cj(x):
    return dict(real=np.asarray(x).real.tolist(), imag=np.asarray(x).imag.tolist())

def local_consumer():
    p = json.loads((ROOT/'rich_composite_local3d_20261014_request.json').read_text())
    receipt = json.loads((ROOT/'rich_composite_local3d_20261014_selection1_receipt.json').read_text())
    saved = np.load(ROOT/'rich_composite_local3d_20261014_selection1_receipt.npz')
    h, m = local.build(p)
    g = saved['ground']; energy = scalar(np.vdot(g, h@g))
    q = np.array([.61, .87, 1.19]); pol = q/np.linalg.norm(q)
    rho = local.density(q, m); j, c = local.current_contact(p, q, pol, m)
    source = rho*g
    ward = np.linalg.norm(h@source-energy*source-np.linalg.norm(q)*(j@g))
    wrong = np.repeat(np.exp(1j*(local.COORDS[m['positions']]@q)).sum(axis=1), m['ns']*256)*g
    parity_rows = []
    for parity in itertools.product((0,1), repeat=3):
        u = local.sector_embedding(m, parity)
        pg = u.T@g; ps = u.T@source
        parity_rows.append(dict(parity=list(parity), ground_weight=scalar(np.vdot(pg,pg)), source_weight=scalar(np.vdot(ps,ps))))
    # An actual sudden removal of local feedback acts on the SAME bound state.
    # It is not confused with the separately selected g=0 ground or threshold.
    hfree, _ = local.build(dict(p, binding_g=0.))
    times = np.linspace(0, .6, 4)
    motion = expm_multiply(-1j*hfree, g.astype(complex), start=0, stop=.6, num=4, traceA=-1j*hfree.diagonal().sum())
    loads = np.repeat(m['load'].sum(axis=1), m['ns']*256)
    observations = [dict(time=scalar(t), norm=scalar(np.vdot(v,v)), no_feedback_energy=scalar(np.vdot(v,hfree@v)), survival=scalar(abs(np.vdot(g,v))**2), local_load=scalar(np.vdot(v,loads*v))) for t,v in zip(times,motion)]
    # New fragment calculation: diagonalize each singleton in ALL cube sectors.
    single_rows = []
    for flavor in range(3):
        hs, ms = local.build(p, (flavor,))
        sectors = [dict(parity=list(parity), energy=scalar(local.sector_ground(hs,ms,parity)[0][0])) for parity in itertools.product((0,1),repeat=3)]
        single_rows.append(dict(flavor=flavor, selected_energy=min(row['energy'] for row in sectors), sectors=sectors))
    breakup = []
    for row in receipt['all_representation_breakup']:
        missing = next(i for i in range(3) if i not in row['pair'])
        threshold = row['ground_energy']+single_rows[missing]['selected_energy']
        breakup.append(dict(pair=row['pair'], independent_singleton_plus_author_all_parity_pair=threshold, binding_margin=threshold-energy, pair_evidence='Frozen selected actual1/selection1 pair spectra; singleton newly solved'))
    freeg = saved['no_feedback_ground']; efree = scalar(np.vdot(freeg,hfree@freeg))
    cappedg = saved['capped_ground']
    return dict(energy=energy, eigen_residual=scalar(np.linalg.norm(h@g-energy*g)), norm=scalar(np.vdot(g,g)), q=q.tolist(), ward_source_error=scalar(ward), current_source_norm=scalar(np.linalg.norm(j@g)), contact=scalar(np.vdot(g,c@g)), wrong_unit_charge_source_distance=scalar(np.linalg.norm(wrong-source)), parity=parity_rows, parity_source_completeness=scalar(sum(r['source_weight'] for r in parity_rows)-np.vdot(source,source)), fragmentation=breakup, singleton=single_rows, removed_feedback_quench=observations, frozen_removed_feedback_eigen_residual=scalar(np.linalg.norm(hfree@freeg-efree*freeg)), removed_feedback_ground_energy=efree, removed_feedback_binding_margins=[r['binding_margin'] for r in receipt['no_field_feedback_breakup']], capped_bound_state_overlap=scalar(abs(np.vdot(g,cappedg))**2), scope='Independent graph-copy fragment energies, not same-cube asymptotic separation; spin, CAR and fractional charges supplied. Quench is NOT a new ground.')

def photon_consumer():
    p = photon.Parameters(background='integer_ions', cutoff=3)
    s = photon.Sector(p,1,1); ev, states = s.solve(3); g = states[:,0]; e = scalar(ev[0])
    plus = photon.Sector(p,2,1); minus = photon.Sector(p,0,1)
    a, ar = photon.charged_source(s,plus,4,0)
    bare, br = photon.charged_source(s,plus,4,0,dressed=False)
    alt, tr = photon.charged_source(s,plus,4,0,string_override=photon.path(s,0,4)-s.faces[1])
    q=np.array([.41,.63,.82]); eps=np.cross(q,[1.,0.,0.]);eps/=np.linalg.norm(eps)
    profile=photon.edge_profile(s,q,eps)
    jc=photon.current(s,profile.real); js=photon.current(s,profile.imag)
    flux=[]
    for cycle in s.faces:
        u=s.flux_operator(cycle);flux.append((u-u.T)/(2j))
    ops=flux+[jc,js,sparse.diags(s.electric[:,0])]
    responses=[dict(omega=w, tensor=cj(photon.retarded_matrix(s,g,e,ops,w+.073j))) for w in (.23,.93,1.71)]
    # Remove matter/field transport, keep actual electric and repulsion law;
    # choose its own unique lowest vector, not the full-H state under new labels.
    off=photon.Sector(photon.Parameters(hopping=0,background='integer_ions',cutoff=3),1,1)
    offg=np.zeros(off.dimension);offg[np.argmin(off.H.diagonal())]=1
    offe=scalar(np.min(off.H.diagonal()))
    offresponse=photon.retarded_matrix(off,offg,offe,[(off.flux_operator(c)-off.flux_operator(c).T)/(2j) for c in off.faces],.93+.073j)
    fullresponse=photon.retarded_matrix(s,g,e,flux,.93+.073j)
    # Direct finite Peierls source derivatives, on literal transfer matrices.
    f=profile.real;step=1e-4
    def sourced(a):
        return sparse.diags(s.electric_energy+p.repulsion*np.repeat(np.array([sum((b>>(2*i)&1)*(b>>(2*i+1)&1) for i in range(s.vertices)) for b in s.fock]),s.nflux))+sum((np.exp(1j*a*v)*t+np.exp(-1j*a*v)*t.T for v,t in zip(f,s.transfers)),sparse.csr_matrix(s.H.shape))
    hp,hm=sourced(step),sourced(-step)
    first=-(hp-hm)/(2*step); second=(hp+hm-2*s.H)/step**2
    derivative_errors=dict(current=scalar(np.linalg.norm((first-jc)@g)),contact=scalar(np.linalg.norm((second-photon.contact(s,f,f))@g)))
    # Actual finite charge sector G at new frequency; no cached isolated G/D.
    ep=plus.solve(2)[0][0];em=minus.solve(2)[0][0]
    mu=scalar((ep-e+e-em)/2)
    green=photon.charged_green(s,plus,minus,g,e,mu,.43+.073j)
    return dict(parameters=p.__dict__, gauss_error=s.gauss_error, integer_electric_error=s.integer_electric_error, continuity=s.continuity_error(), ground_energy=e, eigen_residual=scalar(np.linalg.norm(s.H@g-e*g)), proper_source_norm=scalar(np.linalg.norm(a@g)**2), bare_source_norm=scalar(np.linalg.norm(bare@g)**2), alternate_source_distance=scalar(np.linalg.norm((a-alt)@g)), source_receipts=[ar,br,tr], responses=responses, removed_hopping_flux_response=cj(offresponse), full_flux_response=cj(fullresponse), removed_coupling_response_distance=scalar(np.linalg.norm(fullresponse-offresponse)), peierls_derivative_errors=derivative_errors, charged_green=cj(green), charged_spectral_positivity_min=scalar(np.min(np.linalg.eigvalsh(-(green-green.conj().T)/(2j)))), mu=mu, charge_gap=scalar(ep+em-2*e), scope='Two finite flux channels on integer electric/Gauss sector, not massless propagating polarizations. Positive residue/Green probes use same H and physical string.')

def charged_consumer():
    sectors={n:charged.ChargedGaussMolecule(n,4) for n in (2,3,4)}
    ref,plus,minus=sectors[3],sectors[4],sectors[2]
    data=charged.green_spectral(ref,plus,minus);z=.67+.081j
    spectral=charged.retarded_green(data,z)
    aa=np.column_stack([charged.charged_source(ref,plus,a)@ref.ground for a in range(8)])
    rr=np.column_stack([charged.charged_source(ref,minus,a,False)@ref.ground for a in range(8)])
    direct=aa.conj().T@solve((z+ref.energies[0])*np.eye(len(plus.basis))-plus.H,aa)+(rr.conj().T@solve((z-ref.energies[0])*np.eye(len(minus.basis))+minus.H,rr)).T
    equal=aa.conj().T@aa+(rr.conj().T@rr).T
    mode=4;proper=charged.charged_source(minus,ref,mode)@minus.ground
    wrong=np.zeros(len(ref.basis),complex)
    for col,(bits,fields) in enumerate(minus.basis):
        result=charged.car(bits,mode,True)
        if result is not None:
            row=ref.index.get((result[0],fields))
            if row is not None:wrong[row]+=result[1]*minus.ground[col]
    # Resolve electric response directly, without spectral helper or isolated D.
    src=ref.electric*ref.ground[:,None]
    hd=ref.H-ref.energies[0]*np.eye(len(ref.basis))
    d=src.conj().T@solve(z*np.eye(len(ref.basis))-hd,src)-(src.conj().T@solve(z*np.eye(len(ref.basis))+hd,src)).T
    qin=np.array([.72,.31,.43]);qout=np.array([-.21,.63,.52])
    tensor=charged.physical(ref.elastic(qout,qin,2.73,.081))
    frozen=charged.ChargedGaussMolecule(3,4,frozen=True)
    ft=charged.physical(frozen.elastic(qout,qin,2.73,.081))
    gt=charged.physical(ref.elastic(qout,qin,2.73,.081,ground_only=True))
    gauss=[]
    for n,s in sectors.items():
        rho=np.array([[((b>>(2*v))&1)+((b>>(2*v+1))&1)-.5-(n-2)*int(v==0) for v in range(4)] for b,_ in s.basis])
        gauss.append(dict(N=n, residual=scalar(np.max(abs(s.electric-np.roll(s.electric,1,axis=1)+rho))), integer_spectrum_error=scalar(np.max(abs(s.electric-np.rint(s.electric)))), offset_mod_integer=(s.electric[0]-np.floor(s.electric[0])).tolist(), orthonormality=scalar(np.linalg.norm(s.states.conj().T@s.states-np.eye(len(s.basis))))))
    return dict(gauss=gauss, primitive_charge='One fermion changes number by1; supplied uniform half-ion background fixes alternating half/integer link offsets. Integer Wilson shift does NOT make all electric eigenvalues integers.', reference_state_id=ref.state_id, spin=ref.spin_properties(), proper_source_norm=scalar(np.linalg.norm(proper)**2), bare_source_norm=scalar(np.linalg.norm(wrong)**2), spectral_direct_G_error=scalar(np.linalg.norm(spectral-direct)), equal_time_error=scalar(np.linalg.norm(equal-np.eye(8))), electric_response_direct_error=scalar(np.linalg.norm(d-ref.field_response(z))), G=cj(direct), D_EE=cj(d), elastic_tensor=cj(tensor), frozen_electric_tensor_distance=scalar(np.linalg.norm(tensor-ft)), ground_only_tensor_distance=scalar(np.linalg.norm(tensor-gt)), scope='Declared twisted/background-electric plaquette sector. N3 maxSz ground G/current/contact/E response same state; frozen law is explicit comparator, not a receiver splice.')

def main():
    report={}
    for name,fn in [('local3D',local_consumer),('fullq_integer_ions',photon_consumer),('chargedGauss',charged_consumer)]:
        report[name]=fn();print(json.dumps(dict(completed=name)),flush=True)
    files=['rich_review_wave2_20261014_consumer.py','rich_composite_local3d_20261014_consumer.py','rich_composite_local3d_20261014_selection1_receipt.json','rich_composite_local3d_20261014_selection1_receipt.npz','rich_photon_fullq_20261014_model.py','rich_charged_gauss_20261014_model.py']
    report['input_source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (ROOT/'rich_review_wave2_20261014_result.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
    main()
