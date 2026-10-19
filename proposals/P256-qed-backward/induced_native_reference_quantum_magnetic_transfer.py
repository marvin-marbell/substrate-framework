"""ActualWHOLElinearBose quantum Foldy-spin -> transversephoton transfer.

Samee961/7277 neutralE/Ref3/h/current/C/DW andALLboson-numberHamiltonian.
A_y(sin(pi x/2)) E0 versusphysicalcanonicalpositivebandS_z E0, unrescaled.
Exactpointmainchannelmoment pluspricedallotherq tails andALLbosonremainder.
No tree g2/anomaly/eR/originalSOURCE/fullPhi/SU4/Gauss admission.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from math import isqrt
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parent;OWN=Path(__file__).read_bytes()
import induced_native_reference_all_boson_unitary as whole
import induced_phi_electron_spin_magnetic as spin
NEUTRAL_SHA='0b0057da791093091cbad93f119689ad460f475f2f38eeefc418bf2b90ace98e'
WHOLE_SHA='23e7cd1c292ca6be0cd5c183f2106bf1bf0b34277b558bb8bd9e4436001e3b1f'
START={str(Path(__file__).resolve()):OWN,**whole.SOURCE_AT_START,**whole.neutral.SOURCES,**whole.neutral.charged.SOURCES}


def frac(v:dict[str,Any])->F:return F(int(v['numerator']),int(v['denominator']))
def cp(v:complex)->tuple[F,F]:return F.from_float(float(v.real)),F.from_float(float(v.imag))
def mul(a:tuple[F,F],b:tuple[F,F])->tuple[F,F]:return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
def sqrtupper(v:F)->F:
    if v<0:raise ArithmeticError('Negativephysicalnormenvelope')
    bits=256;return F(isqrt((v.numerator<<(2*bits))//v.denominator)+1,1<<bits)


def point_cross(v:np.ndarray,s:np.ndarray)->list[list[tuple[F,F]]]:
    n=v.shape[0];C=[]
    for i in range(n):
        row=[]
        for j in range(n):
            terms=[mul((cp(v[i,a])[0],-cp(v[i,a])[1]),cp(s[j,a])) for a in range(4)];row.append((sum((t[0] for t in terms),F(0))/4,sum((t[1] for t in terms),F(0))/4))
        C.append(row)
    return C


def bilinear_midpoint(O:np.ndarray,C:list[list[tuple[F,F]]])->tuple[F,F]:
    real=F(0);imag=F(0)
    for i in range(O.shape[0]):
        for j in range(O.shape[1]):
            a=cp(O[i,j]);b=cp(O[j,i]);mid=((a[0]+b[0])/2,(a[1]-b[1])/2);term=mul(mid,C[i][j]);real+=term[0];imag+=term[1]
    return real,imag


def foldy_spin_source(supplier:Any,Z:np.ndarray,v:np.ndarray)->dict[str,Any]:
    native=supplier.native;profile=native['Pprofiles'][0];gamma=native['frame']/np.sqrt([native['g0']]+[native['gs']]*3)[:,None,None];eb,Ub=np.linalg.eigh(gamma[0]);B0=Ub[:,eb>0];Srest=B0.conj().T@(-1j*gamma[1]@gamma[2]/2)@B0;terms=spin.volume.primitive(profile);full=np.zeros((4*native['V'],4*native['V']),dtype=np.complex128);controls=[]
    for site in native['sites']:
        ptheta=(2*np.pi*site+np.pi)/native['N'];E=whole.neutral.charged.embedding(native,ptheta);W=spin.positive(profile,terms,ptheta/native['a'],native['a'],B0);op=W@Srest@W.conj().T;full+=E@op@E.conj().T;controls.append(float(np.max(abs(W.conj().T@W-np.eye(2)))))
    Se=Z.conj().T@full@Z;source=Se@v;eigs=np.linalg.eigvalsh((Se+Se.conj().T)/2);defect=float(np.max(abs(abs(eigs)-.5)))
    if defect>1e-12:raise ArithmeticError('ActualcanonicalFoldyspin-halfsourceconstruction failed')
    return dict(source=source,operator=Se,eigenvalues=eigs,spinhalf_spectral_error=defect,canonical_overlap_error=max(controls),rest_spin=Srest,actual_field_definition='S_Foldy=Σk E_k Wplus(k) Srest Wplus(k)dag E_kdag, canonicalWplus fromFROZENmicroscopicspinmodule actualorderedone/twolinkBlochh. Notmaterialspin guessedfromtwo-levellabels; noSOURCEGramwhitening/normalization. All128bareemptybands,64APmomenta retained.')


def run(neutral_path:Path,whole_path:Path)->dict[str,Any]:
    started=time.monotonic();raw=neutral_path.read_bytes();wr=whole_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=NEUTRAL_SHA or hashlib.sha256(wr).hexdigest()!=WHOLE_SHA:raise ArithmeticError('Frozenactuale961/7277 source/law required')
    cap=json.loads(raw);quant=json.loads(wr);supplier=whole.neutral.charged.load_supplier();native=supplier.native;Z=whole.unpack(cap['bare_empty_Z']);v=whole.unpack(cap['source_all_four_v_alpha']);h=whole.unpack(cap['static_neutral_H_projected']);base=quant['actual_ALL_boson_compensated_source_unitary'];Genv=sum((frac(r['exact_coherent_operator_envelope']) for r in base['actual_physical_source_coupling_groups']),F(0));p=frac(base['source_norm_exact_binarypoint']);h2=frac(base['source_M2_static_exact_binarypoint']);Ss=foldy_spin_source(supplier,Z,v);s=Ss['source'];sMom=whole.hermitian_source_moments(h,s);M2S=sMom['m2_static']+sMom['p']*Genv;C=point_cross(v,s);JA,rho=whole.neutral.projected_operators(native,native['Pprofiles'][0],Z)
    # ActualrealL2smear, q=pi/2x, transversey, exactsinquartervalues.
    shape=np.zeros(3*native['V']);shape[1::3]=np.asarray([0.,1.,0.,-1.])[native['sites'][:,0]]/np.sqrt(32.)
    if np.max(abs(native['D']@shape))>1e-14:raise ArithmeticError('Actualmagneticphotonprobe nottransverse')
    Y=F(0);Ytail=F(0);W2=F(0);main=(F(0),F(0));numeric=0j;rows=[];countA=countR=countModes=0;wrong_density_only=0j
    for site in native['sites']:
        q=tuple(int(i) for i in site)
        if q>tuple((-i)%native['N'] for i in q):continue
        block=supplier.block(q);edge=block.data['physical_edgeframe'];density=block.data['physical_densityframe'];Oa=(edge.T@JA.reshape(3*native['V'],-1)).reshape(block.M,128,128);Op=(density.T@rho.reshape(native['V'],-1)).reshape(block.R,128,128)-density[0,:,None,None]*np.eye(128)[None];O=np.concatenate((Oa,Op));Aann=(shape@edge)@block.amplitudes[:block.M];qy=whole.frobenius_squared_exact(Aann);Y+=qy
        if q!=(1,0,0):Ytail+=qy
        W2+=sum(( (cp(a)[0]**2+cp(a)[1]**2)*F.from_float(float(om))**2 for a,om in zip(Aann,block.omega,strict=True)),F(0));Omid=(O+O.conj().transpose(0,2,1))/2;V=np.einsum('mij,ml->ijl',Omid,block.amplitudes,optimize=True);element=np.einsum('ia,jil,ja->l',v.conj(),V.conj(),s,optimize=True)/4;numeric+=np.sum(Aann*element);VP=np.einsum('mij,ml->ijl',Omid[block.M:],block.amplitudes[block.M:],optimize=True);wrong_density_only+=np.sum(Aann*np.einsum('ia,jil,ja->l',v.conj(),VP.conj(),s,optimize=True)/4)
        if q==(1,0,0):
            for m in range(block.M+block.R):
                sm=bilinear_midpoint(O[m],C);coefs=[mul(cp(a),(cp(b)[0],-cp(b)[1])) for a,b in zip(Aann,block.amplitudes[m],strict=True)];co=(sum((x[0] for x in coefs),F(0)),sum((x[1] for x in coefs),F(0)));term=mul(co,sm);main=(main[0]+term[0],main[1]+term[1])
        rows.append(dict(q=q,actual_photon_source_variance=whole.ratio(qy),physical_positive_modes=len(block.omega)));countA+=block.M;countR+=block.R;countModes+=len(block.omega)
        print('Actualquantumspin-photon group',q,'modes',len(block.omega),flush=True)
    if (len(rows),countA,countR,countModes)!=(36,129,63,3865):raise ArithmeticError('Missingfullactualquantumcoupledfieldmode')
    tail=sqrtupper(p*Ytail*M2S);M2A=3*p*Y*Genv+2*Y*h2+2*p*W2;graph=sqrtupper(M2A*M2S);times=[]
    for t in (F(0),F(1,4096),F(2)**-32,F(4096)):
        radius=t*tail+t*t*graph/2;re=t*main[1];im=-t*main[0];nonzero=abs(re)>radius or abs(im)>radius;times.append(dict(time=whole.ratio(t),raw_real_interval=dict(lower=whole.ratio(re-radius),upper=whole.ratio(re+radius)),raw_imag_interval=dict(lower=whole.ratio(im-radius),upper=whole.ratio(im+radius)),ALL_boson_Taylor_and_ALL_other_q_moment_radius=whole.ratio(radius),nonzero_quantum_magnetic_transfer_proved=nonzero))
    if not times[2]['nonzero_quantum_magnetic_transfer_proved']:raise ArithmeticError('Actualspin/photonquantumtransfer didnotpassfulluntruncatedbound; derivephysicalrepair ratherthanpromoteinitialmoment')
    descriptor=dict(plan_id='issue-222-induced-photon-delta13-continuation-20261008',assigned_scientific_revision='d86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7',observed_parent_current_revision='452ad8053dd6efcbc7d30b55443c1da57c4e6b7e6866b58b11fc61284c6d0961',neutral_receipt_sha256=NEUTRAL_SHA,whole_boson_receipt_sha256=WHOLE_SHA,actual_units=dict(UEphys=native['UE'],a=native['a'],u=native['UE']/native['a']),source='UnrescaledALL4E0Refhole neutral; S_Foldy E0 versus Aphys_y(sin_pi_x_over2)E0, rawsourcecolumns; original4096clockunchanged',exact_point_law=quant['input_descriptor']['exact_model'])
    physical=dict(actual_bare_band_Foldy_spin_half_source={k:value for k,value in Ss.items() if k!='source'},source_norm_spin_exact_point=whole.ratio(sMom['p']),source_norm_photon_exact_point=whole.ratio(p*Y),timezero_cross_exact=0,main_q_exact_M1=dict(real=whole.ratio(main[0]),imag=whole.ratio(main[1])),ALL_other_q_M1_tail_upper=whole.ratio(tail),ALL_q_numerical_mixed_initial_moment=numeric,WRONG_omit_material_Peierls_JA_density_only_moment=wrong_density_only,source_M2_spin_all_boson_upper=whole.ratio(M2S),source_M2_photon_all_boson_upper=whole.ratio(M2A),raw_quantum_spin_to_photon_transfer_time_enclosures=times,actual_quantum_photon_source_groups=rows,proof='SameWHOLElinearH of7277, sS=SFoldy E0|vac> andsA=Aphys_probe E0|vac> withorthogonalbosonnumber at0. Exacttargetq coherentM1 plusCauchypricesotherALLq; spectraltheorem mixedTaylor radius <=t² sqrt(<sA|H²|sA><sS|H²|sS>)/2. Forfieldsourceyλ v, HsA hasvacuum Σyλ Vannλ v, oneboson yλ(h+Ωλ)v, twoboson symΣyλVannμ†v. BoundsM2A<=3pYGramEnvelope+2Yh2source+2pΣ|yλ|²Ωλ² priceEVERYtwo-bosonpath andthenENTIREunitaryALLbosonsectors; M2S exacthSs+sourceNormSsGramEnvelope. Nooriginal-source/bath/privatephase reset.',scope='NEWsameconditionalpositiveelectron/RefholeFULLlinearquantumGaussianphoton/densityspintransfer, notbareg2 orfirstmoment-only. Realfiniteunitarynonzero specificallypricedat2^-32; original4096 evaluatedboundnotclaimednonzero. No nonlinearPeierls/fullPhi/SU4/Noether/originalcanonicalY SOURCE/eR/anomaly/commoncone/vacuum/pole/scatteringadmission.')
    captured={**whole.neutral.charged.SOURCES,**whole.neutral.SOURCES,**START};supplier.s.assert_frozen()
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(whole.neutral.charged.packed(descriptor),sort_keys=True).encode()).hexdigest(),actual_WHOLE_boson_quantum_magnetic_transfer=physical,source_at_import={Path(name).name:dict(path=name,sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in captured.items()},source_unchanged_since_import={name:Path(name).read_bytes()==data for name,data in captured.items()},input_unchanged=neutral_path.read_bytes()==raw and whole_path.read_bytes()==wr,runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--neutral',type=Path,required=True);p.add_argument('--whole',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.neutral,a.whole)
    if not all(r['source_unchanged_since_import'].values()) or not r['input_unchanged']:raise RuntimeError('Actualquantumspin/photon source/inputchanged')
    a.output.write_text(json.dumps(whole.neutral.charged.packed(r),indent=2)+'\n');print(json.dumps(whole.neutral.charged.packed(dict(input_sha256=r['input_sha256'],physical=r['actual_WHOLE_boson_quantum_magnetic_transfer'],elapsed_seconds=r['elapsed_seconds'])),indent=2))
