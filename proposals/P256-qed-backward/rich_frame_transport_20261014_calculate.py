"""Finite cubic carrier transport using field-derived FrameGauss source states.

This NEW consumer imports only the pure Wave5-era FrameGauss constructor. It
creates a fresh cubic position Hilbert space and a same-law Peierls field leg.
"""
from __future__ import annotations
import hashlib, importlib.util, json, platform, sys, time
from pathlib import Path
import numpy as np
from scipy.linalg import expm, eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'rich_frame_transport_20261014_calculate.py'
MODEL_PATH = ROOT / 'rich_frame_gauss_20261014_model.py'
spec = importlib.util.spec_from_file_location('frame_gauss_transport_pure', MODEL_PATH)
assert spec is not None and spec.loader is not None
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(inp):
    p = inp['parameters']
    local = model.carrier(p, l=1, occupation=1)
    hi = np.asarray(local['H'], complex)
    dim = len(hi)
    field = local['field']
    qop = [np.asarray(x, complex) for x in field['Q']]
    spin = [np.kron(np.asarray(x, complex), np.eye(local['nb'])) for x in local['S']]
    fg = model.frame_gauss()
    source = fg['source']
    neutral = np.flatnonzero(np.abs(fg['N'])<1e-8)
    odd = np.flatnonzero(np.abs(fg['N']-1)<1e-8)
    even2 = np.flatnonzero(np.abs(fg['N']-2)<1e-8)
    d01s=[x[np.ix_(odd,neutral)] for x in source]
    d12s=[x[np.ix_(even2,odd)] for x in source]
    source_hop=sum(d@d.conj().T for d in d01s)
    coords = list(np.ndindex(2, 2, 2))
    site = {x:i for i,x in enumerate(coords)}
    nsite = len(coords)
    h = np.kron(np.eye(nsite), hi).astype(complex)
    currents = [np.zeros_like(h) for _ in range(3)]
    bonds = []
    # Open cubic cell: each edge has its actual Cartesian direction and phase.
    for x in coords:
        for axis in range(3):
            y = list(x); y[axis] += 1; y = tuple(y)
            if y not in site: continue
            a,b = site[x],site[y]
            q = qop[axis]
            phase = expm(1j*inp['peierls_coupling']*q)
            link = np.kron(np.kron(np.eye(3), source_hop), phase)
            block = -inp['hopping']*link
            h[a*dim:(a+1)*dim,b*dim:(b+1)*dim] += block
            h[b*dim:(b+1)*dim,a*dim:(a+1)*dim] += block.conj().T
            derivative = -1j*inp['charge']*inp['hopping']*inp['peierls_coupling']*np.kron(np.kron(np.eye(3), source_hop), q@phase)
            currents[axis][a*dim:(a+1)*dim,b*dim:(b+1)*dim] += derivative
            currents[axis][b*dim:(b+1)*dim,a*dim:(a+1)*dim] += derivative.conj().T
            bonds.append([a,b,axis])
    h = (h+h.conj().T)/2
    gs = source[0][np.ix_(odd,neutral)]
    source_grams = [np.linalg.norm(g.conj().T@g-np.eye(len(neutral))) for g in [gs]]
    # Physical left field-coordinate spin on the N=1 subspace is extracted
    # from the same exact colour-Gauss projector as each dressed source.
    sodd = [x[np.ix_(odd,odd)] for x in fg['S']]
    # The 3D site/local dimension has the model's W(l=1) x odd frame x vector field.
    sz = np.kron(np.eye(nsite), spin[2])
    _, vectors = eigh(hi)
    seed = np.zeros(nsite*dim, complex)
    seed[:dim] = vectors[:,0]
    # Local field-coordinate pulse; compare each run to the identical unpulsed input.
    pulse_local = np.kron(np.eye(3*2), expm(-1j*inp['pulse_strength']*qop[2]))
    pulse = np.kron(np.diag([1]+[0]*(nsite-1)), pulse_local)
    # Above site projector must act on position, not internal ordering.
    pulse = np.kron(np.diag([1]+[0]*(nsite-1)), pulse_local) + np.kron(np.diag([0]+[1]*(nsite-1)), np.eye(dim))
    driven0 = pulse @ seed
    base = seed.copy()
    times = np.asarray(inp['times'],float)
    records=[]
    for t in times:
        a=expm_multiply((-1j*t)*csr_matrix(h),driven0)
        b=expm_multiply((-1j*t)*csr_matrix(h),base)
        pos_a=np.array([np.vdot(a[i*dim:(i+1)*dim],a[i*dim:(i+1)*dim]).real for i in range(nsite)])
        pos_b=np.array([np.vdot(b[i*dim:(i+1)*dim],b[i*dim:(i+1)*dim]).real for i in range(nsite)])
        records.append({'time':float(t),'site_probability_pulsed':pos_a.tolist(),'site_probability_baseline':pos_b.tolist(),
            'far_corner_probability_change':float(pos_a[-1]-pos_b[-1]),
            'field_coordinate_spin_z_change':float(np.vdot(a,sz@a).real-np.vdot(b,sz@b).real),
            'cartesian_current_change':[float(np.vdot(a,c@a).real-np.vdot(b,c@b).real) for c in currents],
            'field_energy_change':float(np.vdot(a,np.kron(np.eye(nsite),np.kron(np.eye(6),field['H']))@a).real-np.vdot(b,np.kron(np.eye(nsite),np.kron(np.eye(6),field['H']))@b).real),
            'norm_pulsed':float(np.vdot(a,a).real),'norm_baseline':float(np.vdot(b,b).real)})
    cut_h=h.copy()
    # Remove every Cartesian link in the identical eight-site Hilbert space.
    for a,b,axis in bonds:
        cut_h[a*dim:(a+1)*dim,b*dim:(b+1)*dim]=0
        cut_h[b*dim:(b+1)*dim,a*dim:(a+1)*dim]=0
    cut_h=csr_matrix(cut_h)
    cut=[]
    for t in times:
        a=expm_multiply((-1j*t)*cut_h,driven0)
        b=expm_multiply((-1j*t)*cut_h,base)
        pa=np.linalg.norm(a[(nsite-1)*dim:nsite*dim])**2
        pb=np.linalg.norm(b[(nsite-1)*dim:nsite*dim])**2
        cut.append(float(pa-pb))
    current_off=h.copy()
    for a,b,axis in bonds:
        block=-inp['hopping']*np.eye(dim)
        current_off[a*dim:(a+1)*dim,b*dim:(b+1)*dim]=block
        current_off[b*dim:(b+1)*dim,a*dim:(a+1)*dim]=block
    current_off=csr_matrix(current_off)
    current_off_response=[]
    for t in times:
        a=expm_multiply((-1j*t)*current_off,driven0)
        b=expm_multiply((-1j*t)*current_off,base)
        current_off_response.append(float(np.linalg.norm(a[(nsite-1)*dim:nsite*dim])**2-
            np.linalg.norm(b[(nsite-1)*dim:nsite*dim])**2))
    # Adjacent charge insertion is the literal two-channel second transition of the same d†.
    h0=model.carrier(p,l=0,occupation=0)['H']
    h1=model.carrier(p,l=0,occupation=1)['H']
    h2=model.carrier(p,l=0,occupation=2)['H']
    e0,v0=eigh(h0); e1=eigh(h1,eigvals_only=True); e2=eigh(h2,eigvals_only=True)
    neutral_state=v0[:,0]
    field_eye=np.eye(local['nb'])
    inserted=np.column_stack([(np.kron(d,field_eye)@neutral_state) for d in d01s])
    resolvent=np.linalg.solve((0.9+0.07j)*np.eye(len(h1))-(h1-e0[0]*np.eye(len(h1))),inserted)
    source_G=inserted.conj().T@resolvent
    # Sum over both physical spin channels: each individual d_m is a rank-one
    # Pauli-blocked map, while their combined completeness is the CAR statement.
    adjacent_sum=sum(d.conj().T@d for d in d12s)
    adjacent={'neutral_to_charge_one_channel_Gram_residual':float(np.linalg.norm(d01s[0].conj().T@d01s[0]-np.eye(len(neutral)))),
        'charge_to_adjacent_charge_two_channel_CAR_residual':float(np.linalg.norm(adjacent_sum-np.eye(len(odd)))),
        'one_charge_ground_energy':float(e1[0]-e0[0]),'adjacent_charge_ground_energy':float(e2[0]-e1[0]),
        'same_law_source_G_at_0p9_plus_0p07i':[[[float(z.real),float(z.imag)] for z in row] for row in source_G],
        'one_charge_gauss_sector_dim':int(len(odd)),'adjacent_two_charge_gauss_sector_dim':int(len(even2)),
        'Coulomb_pair_addition_energy':float(model.charge_geometry(p)[0]*(2**2-1**2))}
    return {'candidate':'rich_frame_transport_20261014_cubic_frame_gauss_peierls_v1','law':'FrameGauss field-coordinate source + cubic Peierls transport + supplied finite Gaussian longitudinal Gauss','input':inp,
        'physical_frame_source':{'bare_c_creation_not_used':True,'dressed_source_Gram_residual':source_grams[0],'source_hop_intertwiner_residual':float(np.linalg.norm(source_hop-np.eye(2))),'physical_spin_eigenvalues':np.linalg.eigvalsh(sodd[2]).tolist(),'bare_source_physical_gram':0.0},
        'adjacent_charge_same_law':adjacent,'bonds':bonds,'records':records,'all_link_cut_far_corner_changes':cut,
        'max_all_link_cut_response':float(max(abs(x) for x in cut)),
        'current_coupling_off_far_corner_changes':current_off_response,
        'controls':{'no_pulse_baseline':'same prepared field-frame charged packet evolved under identical H','all_link_cut':'all Cartesian hopping/current links removed for the same mobile carrier','current_coupling_off':'same spatial hopping with all vector-coordinate Peierls link dependence removed','same_current':'Peierls link Q and current are first and second derivatives of identical hopping exponential'},
        'caveats':['Frame coordinate carries exact colour-Gauss halfspin, but primitive colour fundamental, CAR and centre sewing remain inputs.',
          'The eight-site open cube is pinned to a cubic scaffold; it does not prove continuum rotational covariance or free center of mass.',
          'Three vector oscillator coordinates are local regulator modes, not propagating photons.',
          'Adjacent-charge number is a same-source N=1 to N=2 Gauss transition; no mobile two-electron scattering state is calculated.',
          'Longitudinal Gaussian electrostatic kernel and inertias are supplied inputs; no self-bound electron or Lorentz invariance is claimed.',
          'Only one mobile charged species is present; the requested two-species supplier/active-pair all-link cut and inclusive two-mobile-packet interaction remain unpaid.']}

def main():
    start=time.time(); input_path=ROOT/'rich_frame_transport_20261014_input.json'
    inp=json.loads(input_path.read_text())
    result=run(inp)
    result['runtime']={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'source_sha256_before':digest(SOURCE),'model_sha256':digest(MODEL_PATH),'input_sha256':digest(input_path)}
    out=ROOT/'rich_frame_transport_20261014_result.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    print(f'ELAPSED_SECONDS={time.time()-start:.6f}')

if __name__=='__main__': main()
