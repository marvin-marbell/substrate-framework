"""Independent PH/Gauss determinant benchmark and physical imaginary-time response.
No producer import, output replay or hard electric cutoff in physical open-link space.
"""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm, eigh

ROOT = Path(__file__).resolve().parent
P = 'rich_review_wave43_20261016_'


def annihilator(mode, modes=4):
    out = np.zeros((1 << modes, 1 << modes), complex)
    for n in range(1 << modes):
        if n >> mode & 1:
            out[n ^ (1 << mode), n] = (-1) ** ((n & ((1 << mode)-1)).bit_count())
    return out


def benchmark(cfg):
    """Open two-site tree: Gauss fixes E=n0-1, hence exact finite physical space."""
    U, I, beta, L, grid = (cfg[k] for k in ('U', 'I', 'beta', 'slices', 'angle_grid'))
    dt = beta / L
    assert L == 2
    c = [annihilator(a) for a in range(4)]
    n = [a.conj().T @ a for a in c]
    one = np.eye(16)
    N0, N1 = n[0]+n[1], n[2]+n[3]
    onsite = U/2*((N0-one)@(N0-one)+(N1-one)@(N1-one))
    physical = [b for b in range(16) if b.bit_count() == 2]
    ix = np.ix_(physical, physical)
    K = -sum((c[a].conj().T@c[a+2]+c[a+2].conj().T@c[a] for a in (0, 1)), np.zeros((16,16),complex))
    E = (N0-one)[ix]
    V = onsite[ix]
    Kp = K[ix]
    He = E@E/(2*I)
    H = Kp+He+V
    T = expm(-dt*He)@expm(-dt*Kp)@expm(-dt*V)
    Zt = np.trace(np.linalg.matrix_power(T,L)).real
    et = np.trace(E@E@np.linalg.matrix_power(T,L)).real/Zt
    evals, vec = eigh(H)
    weights = np.exp(-beta*evals)
    Z = weights.sum()
    Eg = vec.conj().T@E@vec
    J = -1j*(H@E-E@H)
    Jg = vec.conj().T@J@vec
    taus = np.array([0., beta/4, beta/2])
    def corr(A):
        return np.array([np.sum(np.exp(-(beta-t)*evals[:,None]-t*evals[None,:])*np.abs(A)**2).real/Z for t in taus])
    ce, cj = corr(Eg), corr(Jg)
    # PH implemented as a unitary on the ORIGINAL full Fock algebra.
    # Target d_up=c_up, d_h=epsilon*c_down^dagger; solve intertwiner by joint vacuum.
    d = [c[0], c[1].conj().T, c[2], -c[3].conj().T]
    vacuum = np.zeros(16,complex);vacuum[10]=1
    W = np.zeros((16,16),complex)
    for b in range(16):
        state = vacuum.copy()
        for a in reversed(range(4)):
            if b >> a & 1: state = d[a].conj().T@state
        W[:,b] = state
    nh = [a.conj().T@a for a in d]
    mapped_onsite = sum((-U*(nh[a]-.5*one)@(nh[a+1]-.5*one)+U/4*one for a in (0,2)),np.zeros((16,16),complex))
    rho = N0-one
    ph_rho = nh[0]-nh[1]
    wrong_charge = nh[0]+nh[1]-one
    # Complex temporal holonomy, genuine kinetic angles and nonuniform HS fields.
    # Compare grand trace of each path in original Fock vs conjugate determinants.
    theta = [0., .47]
    fields = np.array([[1,-1],[-1,-1]])
    alpha = np.array([.37,-.29])
    lam = np.arccosh(np.exp(dt*U/2))
    B = np.eye(2,dtype=complex)
    F = np.eye(16,dtype=complex)
    scalar = 1.
    for l in range(L):
        h = np.array([[0,-np.exp(1j*theta[l])],[-np.exp(-1j*theta[l]),0]],complex)
        B = expm(-dt*h)@np.diag(np.exp(lam*fields[l]))@B
        kup = -np.exp(1j*theta[l])*(c[0].conj().T@c[2]+c[1].conj().T@c[3])
        kfull = kup+kup.conj().T
        hs = lam*(fields[l,0]*(n[0]-n[1])+fields[l,1]*(n[2]-n[3]))
        F = expm(-dt*kfull)@expm(hs)@F
        scalar *= np.exp(-dt*U/2*2-lam*fields[l].sum())
    # HS in original variables has (nup-ndown), but PH nup+nh = nup-ndown+1.
    # Constant exp(+lambda*s) cancels scalar's exp(-lambda*s) in original path.
    scalar_original = np.exp(-dt*U/2*2*L)
    hol = expm(-1j*(alpha[0]*(N0-one)+alpha[1]*(N1-one)))
    original_path = scalar_original*np.trace(hol@F)
    det_path = scalar*abs(np.linalg.det(np.eye(2)+np.diag(np.exp(-1j*alpha))@B))**2
    # The PH hole kernel is conjugate only AFTER its staggered kinetic transform;
    # original down HS has opposite sign, exactly related by particle-hole trace.
    # Actual Gauss projected angle+HS integral. theta0=0 fixes a time-independent gauge.
    angles = 2*np.pi*np.arange(grid)/grid
    a0,a1 = np.meshgrid(angles, angles,indexing='ij')
    twists = np.stack((np.exp(-1j*a0),np.exp(-1j*a1)),axis=-1)
    m = np.arange(-cfg['heatkernel_terms'],cfg['heatkernel_terms']+1)
    def heat(x, e2=False):
        factors = np.exp(-dt*m*m/(2*I))*(m*m if e2 else 1)
        return np.real(np.sum(factors*np.exp(1j*x[...,None]*m),axis=-1))
    zq=0.;eq=0.;z_no_twist=0.;z_wrong_boundary=0.
    for th in angles:
        h = np.array([[0,-np.exp(1j*th)],[-np.exp(-1j*th),0]],complex)
        hop0 = expm(dt*np.array([[0,1],[1,0]],complex))
        hop1 = expm(-dt*h)
        k0 = heat(np.array(th))
        kb = heat(-th+a0-a1)
        eb = heat(-th+a0-a1,True)
        for ss in itertools.product((-1,1),repeat=4):
            s = np.array(ss).reshape(2,2)
            b = hop1@np.diag(np.exp(lam*s[1]))@hop0@np.diag(np.exp(lam*s[0]))
            scalar = np.exp(-dt*U/2*2*L-lam*s.sum())/(2**(2*L))
            mats = twists[..., :,None]*b
            dets = np.linalg.det(np.eye(2)+mats)
            wt = scalar*np.abs(dets)**2
            zq += float(k0*np.mean(kb*wt))/grid
            eq += float(k0*np.mean(eb*wt))/grid
            z_no_twist += float(k0*np.mean(kb)*scalar*abs(np.linalg.det(np.eye(2)+b))**2)/grid
            z_wrong_boundary += float(k0*np.mean(heat(-th+a0+a1)*wt))/grid
    # Actual determinant nonconjugate countercase: don't change physical down charge.
    bup=np.diag(np.exp(-1j*alpha))@B
    same_charge_hole_weight=np.exp(-dt*U/2*2*L-lam*fields.sum())*np.linalg.det(np.eye(2)+bup)**2
    result = dict(kind='EXACT_OPEN_TREE_GAUSS_PH_BENCHMARK_NOT_V64_THERMAL_STATE',input=cfg,
        physical_dimension=len(physical),electric_support=[-1,0,1],no_hard_E_cut=True,
        ph_unitary_error=float(np.linalg.norm(W.conj().T@W-one)),
        onsite_PH_error=float(np.linalg.norm(onsite-mapped_onsite)),
        rho_PH_error=float(np.linalg.norm(rho-ph_rho)),
        wrong_charge_error=float(np.linalg.norm(rho-wrong_charge)),
        original_path=[float(original_path.real),float(original_path.imag)],determinant_path=float(det_path),
        path_error=float(abs(original_path-det_path)),
        same_charge_hole_weight=[float(same_charge_hole_weight.real),float(same_charge_hole_weight.imag)],
        exact_Z=float(Z),trotter_Z=float(Zt),quadrature_Z=zq,
        quadrature_vs_same_trotter_error=abs(zq-Zt),trotter_bias=Zt-float(Z),
        E2_trotter=float(et),E2_quadrature=eq/zq,
        no_fermion_boundary_twist_Z=z_no_twist,wrong_rotor_boundary_Z=z_wrong_boundary,
        taus=taus.tolist(),E_imagtime=ce.tolist(),J_imagtime=cj.tolist(),
        physical_response_class='thermal imaginary-time full exact 6-state H, no analytic continuation or photon pole',
        source_phase_countercase='same-charge hole kernel produces complex determinant square, not conjugate pairing',
        onsite_constants='U4: +1 per site in centered attractive Hamiltonian; all Nup=Nh sectors, not fixed1/1',
        thermal_ensemble='Gauss only, original totalN2, all original spin populations')
    np.savez_compressed(ROOT/(P+'rotor_raw.npz'),H=H,E=E,J=J,energies=evals,taus=taus,Ecor=ce,Jcor=cj)
    (ROOT/(P+'rotor_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    assert result['onsite_PH_error'] < 1e-12
    assert result['path_error'] < 1e-9
    assert abs(zq-Zt) < 1e-8
    assert abs(eq/zq-et) < 1e-8
    assert abs(z_no_twist-zq) > .01
    assert abs(ce[-1]-ce[0]) > .001


if __name__ == '__main__':
    benchmark(json.loads((ROOT/(P+'rotor_input.json')).read_text()))
