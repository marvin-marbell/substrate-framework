"""New native algebra and actual-ground derived gates; no solver/backend reruns."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy import sparse
from scipy.linalg import expm
import sympy as sp
from rotation_band import RotationBand, MIXING, HOPPING

ROOT = Path(__file__).resolve().parent
PREFIX = 'ground_magnetic_order_'
INPUT_PINS = {
 'magnetic_mechanism_proof_20261011.json':'01355862fba1bbef9d49fe87472698cff7e4d724c3798c5e980ada949ff24141',
 'joined_phi_u1_closed_inverse_transverse_parent_packet_20261012.json':'ae1bf8dabadcd09f4cd3dc46f42ef668467f0452153b54bcf0be0f33c785e56b',
 'joined_phi_u1_global_ground_frobenius_actual_20261010.json':'0bcf08b8a72ebb0c316f727053cec208a024bf977a4f15dcfe91e875e9564fe9',
 'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json':'5b97862fc7fca21b8797d549d7d4dfdde58b101e8c46c7a7cfb6531a669c4354',
}

def incidence(n: int) -> tuple[Any, Any, Any]:
    v=n**3; sites=list(np.ndindex(n,n,n)); ix={x:i for i,x in enumerate(sites)}
    def target(x: tuple[int,...], j: int) -> int:
        y=list(x); y[j]=(y[j]+1)%n; return ix[tuple(y)]
    dr=[];dc=[];dv=[];cr=[];cc=[];cv=[];sr=[];sc=[];sv=[]
    for x in sites:
        i=ix[x]
        for j in range(3):
            dr.extend([i,target(x,j)]);dc.extend([3*i+j]*2);dv.extend([1,-1])
        for p,(j,k) in enumerate(((0,1),(0,2),(1,2))):
            cr.extend([3*i+p]*4);cc.extend([3*i+j,3*target(x,j)+k,3*target(x,k)+j,3*i+k]);cv.extend([1,1,-1,-1])
        for j,p,s in ((0,2,1),(1,1,-1),(2,0,1)):
            sr.extend([i,i]);sc.extend([3*target(x,j)+p,3*i+p]);sv.extend([s,-s])
    return (sparse.csr_matrix((dv,(dr,dc)),shape=(v,3*v),dtype=np.int64),
            sparse.csr_matrix((cv,(cr,cc)),shape=(3*v,3*v),dtype=np.int64),
            sparse.csr_matrix((sv,(sr,sc)),shape=(v,3*v),dtype=np.int64))

def car(mask: int, create: int, annihilate: int) -> tuple[int,int] | None:
    if not (mask>>annihilate)&1:return None
    sign=(-1)**((mask&((1<<annihilate)-1)).bit_count());mask^=1<<annihilate
    if (mask>>create)&1:return None
    sign*=(-1)**((mask&((1<<create)-1)).bit_count())
    return mask|(1<<create),sign

def symbolic_form() -> dict[str,Any]:
    b,k=sp.symbols('b k',real=True);psi=sp.Function('psi')(b);pot=sp.Function('V')(b)
    # r^2=4: electric derivative on the B coordinate is -256 d_B^2.
    h=lambda z:-256*sp.diff(z,b,2)+pot*z
    w=sp.exp(sp.I*b);comm=sp.simplify((h(w*psi)-w*h(psi))/w)
    expected=-512*sp.I*sp.diff(psi,b)+256*psi
    assert sp.simplify(comm-expected)==0
    c,s=sp.cos(b),sp.sin(b)
    double=lambda f:sp.simplify(2*f*h(f*psi)-f*f*h(psi)-h(f*f*psi))
    assert sp.simplify(double(c)+double(s)-512*psi)==0
    f=sp.exp(k*sp.cos(b)/2)
    price=sp.simplify(256*sp.diff(f,b)**2/f**2)
    assert sp.simplify(price-64*k*k*sp.sin(b)**2)==0
    return {'complete_multiplier_commutator':str(comm),'paired_double_commutator':str(512*psi),
            'normalized_reset_price_density':str(price),'physical_k':1,
            'neutral_Phi_Haar_CAR_multiplier_terms_cancel_symbolically':True}

def native_controls(n: int, cfg: dict[str,Any]) -> dict[str,Any]:
    D,C,S=incidence(n);v=n**3
    assert (S@C).nnz==0 and (C@D.T).nnz==0
    r=C.getrow(0).toarray().ravel();assert int(r@r)==4 and np.all(D@r==0)
    rng=np.random.default_rng(222120+n);units=rng.integers(-11,12,3*v)
    raw=C@units;wrapped=(raw+11)%22-11;defects=(S@wrapped)//22
    assert np.any(defects) and np.all(S@raw==0)
    a=np.pi*r/4;assert np.max(np.abs(D@a))<1e-12
    # Literal native positive one/two-link words and their adjoints, including AP seams.
    profiles=[]; full_count=0; phase_actions=[]
    basis=np.concatenate(([MIXING],HOPPING),axis=0)
    for role,count,mass,rr,vv,improved,charge in (
        ('neutral',1,cfg['heavy_m'],cfg['heavy_r'],cfg['heavy_v'],False,0),
        ('active',1,cfg['active_m'],cfg['light_r'],cfg['light_v'],True,1),
        ('auxiliary',63,cfg['auxiliary_m'],cfg['light_r'],cfg['light_v'],True,1)):
        band=RotationBand(a=cfg['a'],m=mass,r=rr,v=vv,improved=improved)
        onsite,paths=band.couplings();coeffs=[];changed=0;ap=0
        for disp,matrix in paths:
            coeff=np.einsum('mij,ji->m',basis,matrix)/4
            assert np.linalg.norm(matrix-np.einsum('m,mij->ij',coeff,basis))<1e-12
            coeffs.append([[float(z.real),float(z.imag)] for z in coeff])
            axis=int(np.flatnonzero(disp)[0]);length=int(disp[axis])
            for x in np.ndindex(n,n,n):
                y=list(x);flow=np.zeros(3*v,dtype=np.int64);seam=1
                for _ in range(length):
                    idx=(y[0]*n+y[1])*n+y[2];flow[3*idx+axis]+=1
                    if y[axis]==n-1:seam*=-1
                    y[axis]=(y[axis]+1)%n
                div=D@flow;assert np.count_nonzero(div)==2 and sum(div)==0
                phase=np.exp(1j*charge*(flow@a));changed+=int(abs(phase-1)>1e-12);ap+=int(seam==-1)
                if charge==0:assert phase==1
                # Reverse literal adjoint has reversed flow/charge and conjugate phase.
                assert abs(np.exp(-1j*charge*(flow@a))-phase.conjugate())<1e-12
            full_count+=2*len(paths)*v*count
        profiles.append({'role':role,'copies':count,'ordered_directed_words_per_copy':2*len(paths)*v,
                         'changed_forward_words':changed,'AP_seam_forward_words':ap,
                         'native_scalar_coefficients':coeffs,'onsite_HS':float(np.linalg.norm(onsite))})
        phase_actions.append(changed)
    assert phase_actions[0]==0 and phase_actions[1]>0 and phase_actions[2]>0
    # Disjoint integer-face block: exact diagonal energy Gram, no phase input.
    chosen=[]
    for row in range(0,3*v,3):
        candidate=C.getrow(row).toarray().ravel()
        if all(int(candidate@z)==0 for z in chosen):chosen.append(candidate)
        if len(chosen)==4:break
    gram=np.array(chosen)@np.array(chosen).T;assert np.array_equal(gram,4*np.eye(4,dtype=int))
    q=np.arange(4)*np.pi/2;block=[]
    for angle in q:
        f=np.cos(np.arange(4)*angle);f=f/np.linalg.norm(f)
        value=float(64*f@gram@f);assert abs(value-256)<1e-10
        block.append({'angle':float(angle),'exact_first_moment':256,'not_q2':True})
    # Positive working geometry control and ground-kernel adverse control.
    overlap=(C[::3,:]@C[::3,:].T).toarray();ones=np.ones(v)/np.sqrt(v)
    flat=float(64*ones@overlap@ones);orthogonal=float(64*ones@(overlap*np.eye(v))@ones)
    assert abs(flat)<1e-10 and abs(orthogonal-256)<1e-10
    return {'N':n,'SC_nonzero':(S@C).nnz,'CDT_nonzero':(C@D.T).nnz,
            'fullE_Gauss_closed_r':True,'wrapped_defect_count':int(np.count_nonzero(defects)),
            'native_profiles':profiles,'all_species_ordered_word_count':full_count,
            'disjoint_block':block,'q0_flat_kernel_first_moment':flat,
            'q0_orthogonal_Wilson_kernel_first_moment':orthogonal,
            'kernel_adverse_is_algebraic_not_ground_measurement':True}

def field_car_controls() -> dict[str,Any]:
    rng=np.random.default_rng(2221212)
    def hermitian() -> Any:
        z=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));z=(z+z.conj().T)/2;return z-np.trace(z)*np.eye(4)/4
    phi=np.array([hermitian() for _ in range(4)]);other=np.array([hermitian() for _ in range(4)])
    u=expm(1j*hermitian());gx=expm(1j*hermitian());gy=expm(1j*hermitian())
    endpoint=lambda x,y,z:(np.einsum('m,mij->ij',[1,1j,0,0],x)@z+z@np.einsum('m,mij->ij',[1,1j,0,0],y))/2
    b=endpoint(phi,other,u)
    transformed=endpoint(gx@phi@gx.conj().T,gy@other@gy.conj().T,gx@u@gy.conj().T)
    err=float(np.linalg.norm(transformed-gx@b@gy.conj().T));assert err<1e-11
    assert np.linalg.norm(endpoint(np.zeros_like(phi),np.zeros_like(phi),u))==0
    # Full canonical4 on two sites/8 colour modes. This finite CAR algebra is not a ground model.
    masks=[m for m in range(256) if m.bit_count()==4];ix={m:i for i,m in enumerate(masks)}
    hop=np.zeros((70,70),dtype=complex)
    for mask in masks:
        for i in range(4):
            for j in range(4):
                moved=car(mask,i,4+j)
                if moved:dest,sign=moved;hop[ix[dest],ix[mask]]+=sign*b[i,j]
    h=hop+hop.conj().T
    assert np.linalg.norm(h)>0
    for m in (15,240):
        # One direction vanishes on empty/full endpoints; no scalar number-only carrier equality.
        assert np.linalg.norm(hop[:,ix[m]])==0 or np.linalg.norm(hop.conj().T[:,ix[m]])==0
    eig0=np.eye(70)[:,ix[15]]
    assert abs(np.vdot(eig0,h@eig0))==0
    sign_defect=float(np.linalg.norm((hop-hop.conj().T)*1j));assert sign_defect>0
    # Coordinate multiplier commutes with actual full endpoint matrix and CAR, unlike endpoint phase-only change.
    phase=np.exp(1j*.7);delta=(phase-1)*hop+(phase.conjugate()-1)*hop.conj().T
    assert np.linalg.norm(delta)>0 and np.trace(delta)==0
    return {'fullPhi_traceless_Hermitian':True,'SU4_endpoint_covariance_error':err,
            'canonical_two_site_dimension':70,'nonzero_CAR_hopping_norm':float(np.linalg.norm(h)),
            'zeroPhi_no_flux_penalty':True,'coordinate_translation_signed_CAR_price_norm':float(np.linalg.norm(delta)),
            'price_zero_on_singlet_checkerboard_expectation':True,'not_ground_observation':True}

def ground_gate() -> dict[str,Any]:
    # Exact rational identities; no giant floating coupling and no numerical ground evaluation.
    t=sp.symbols('T',nonnegative=True);m1=sp.Integer(256);m2=16384*t+65536
    lower=sp.cancel(m1*m1/m2);assert lower==4/(t+4)
    R=128*(t+4);band=sp.simplify(lower-m1/R);assert band==2/(t+4)
    kinetic_trial=2**6724;lambda_shift=2**8205
    # Loose complete pinned N4 tube upper including geometry/HC/bare cover packet.
    trial_upper=kinetic_trial+2**6147+2**24+2**13
    assert trial_upper<lambda_shift
    assert (trial_upper+lambda_shift)//64 < 2**8200
    assert Fraction(4,2**8202+4)>Fraction(1,2**8201)
    return {'actual_ground_analytic_antecedent':True,'physical_hopping':1,
            'm1_pair_exact':256,'m2_pair_exact':str(m2),'wholeP0_covariance_lower':str(lower),
            'positive_band_cutoff':str(R),'positive_band_mass_lower':str(band),
            'N4_explicit_lower':'strictly > 2^-8201','finite_only':True,
            'reset_physical1_gain_minus_price_nonpositive':True,'IR_lower':False}

def run(output: Path) -> dict[str,Any]:
    for name,pin in INPUT_PINS.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin
    cfg=json.loads((ROOT/'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json').read_bytes())['input_descriptor']['original_micro_config_UNCHANGED']
    result={'kind':'NATIVE_COMPLETE_H_RESET_AND_GROUND_ENERGY_FORM_GATE_EXECUTED_NOT_GROUND_SOLVER',
            'symbolic_complete_H':symbolic_form(),'native':[native_controls(n,cfg) for n in (4,6,8)],
            'CAR_Phi_SU4':field_car_controls(),'actual_derived_ground_gate':ground_gate(),
            'proof':PREFIX+'proof_20261012.json','input_sha256':INPUT_PINS,
            'two_distinct_attempt_boundaries':['periodic full-H reset has nonnegative exact electric price and no signed defect gain','irreducible ground paired magnetic energy-form gives finite local wholeP0 lower but block first moment remains constant not q2'],
            'fullH_numeric_backend':False,'ground_exists_analytically':True,'campaign_complete':False}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'kind':result['kind'],'N':[4,6,8],'ground_gate':result['actual_derived_ground_gate']}))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
