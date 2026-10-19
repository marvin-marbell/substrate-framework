#!/usr/bin/env python3
"""Actual flat Z4 seam sectors, full-ten Gram selection and Gauss competitor.

Finite N6 discriminator only. No bulk/global-vacuum or caller promotion.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import hashlib,itertools,json,math
from pathlib import Path
import platform,sys
import time
from typing import Any
import numpy as np
import mpmath as mp
from induced_phi_backreaction import (PhiConfig,models,periodic_grid,scalar_paths,
    SELECTED,IMPORT_BYTES,global_centre_projection)
ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**IMPORT_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
SOURCE_HASHES={name:hashlib.sha256(value).hexdigest() for name,value in SOURCE_BYTES.items()}
SYMMETRIC=tuple((i,j) for i in range(4) for j in range(i,4))
CENTRES=(1,1j,-1,-1j)


def matrix(vector):
    result=mp.zeros(4)
    for value,(i,j) in zip(vector,SYMMETRIC):
        result[i,j]=result[j,i]=value if i==j else value/mp.sqrt(2)
    return result


def vector(g):
    return mp.matrix([g[i,j]*(1 if i==j else mp.sqrt(2)) for i,j in SYMMETRIC])


def coefficient_rows(model,size,twist):
    onsite,paths=scalar_paths(model)
    rows=[]
    for site in periodic_grid(size):
        theta=[2*mp.pi*(int(n)+mp.mpf(t)/4)/size for n,t in zip(site,twist)]
        row=[mp.mpf(float(onsite)),mp.mpf(0),mp.mpf(0),mp.mpf(0)]
        for displacement,scalars in paths:
            phase=mp.exp(mp.j*sum(theta[j]*int(displacement[j]) for j in range(3)))
            for mu in range(4):
                row[mu]+=2*mp.re(mp.mpc(float(scalars[mu].real),float(scalars[mu].imag))*phase)
        rows.append(row)
    return rows


def data_for(cfg,size,twist,exponent):
    output=[]
    for load,model,role in models(replace(cfg,Lambda=1.)):
        weight=mp.mpf(1) if role.startswith('neutral') else mp.mpf(load)*mp.power(2,-exponent)
        rows=coefficient_rows(model,size,twist)
        features=[[row[i]*row[j]*(1 if i==j else mp.sqrt(2)) for i,j in SYMMETRIC] for row in rows]
        output.append((weight,rows,features,role))
    return output


def functional(g,data,stiffness):
    volume=len(data[0][1]); derivative=stiffness*g; hessian=stiffness*mp.eye(10)
    value=stiffness*sum(z*z for z in g)/2
    for weight,_,features,_ in data:
        for feature in features:
            squared=sum(z*x for z,x in zip(g,feature))
            if squared<=0:raise ArithmeticError('Physical band radius left positive Gram domain')
            radius=mp.sqrt(squared); value-=2*weight*radius/volume
            first=weight/(volume*radius); second=weight/(2*volume*radius**3)
            for t in range(10):
                derivative[t]-=first*feature[t]
                for u in range(t+1):
                    term=second*feature[t]*feature[u]
                    hessian[t,u]+=term
                    if t!=u:hessian[u,t]+=term
    return value,derivative,hessian


def solve_gram(data,stiffness,seed=None,tolerance=None):
    g=mp.matrix([mp.mpf('.3') if i==j==0 else mp.mpf('.1') if i==j else 0 for i,j in SYMMETRIC]) if seed is None else vector(seed)
    tolerance=mp.power(10,-mp.mp.dps+25) if tolerance is None else tolerance
    for iteration in range(24):
        value,gradient,hessian=functional(g,data,stiffness)
        if mp.norm(gradient)<tolerance:break
        step=mp.lu_solve(hessian,gradient); factor=mp.mpf(1)
        for _ in range(40):
            trial=g-factor*step
            eigenvalues=mp.eigsy(matrix(trial),eigvals_only=True)
            if eigenvalues[0]>0:
                trial_value,_,_=functional(trial,data,stiffness)
                # Near stationarity the true Newton energy decrement is below
                # arithmetic resolution; certify the gradient, not rounded descent.
                rounding_allowance=mp.eps*2**24*(1+abs(value))
                if trial_value<=value+rounding_allowance:break
            factor/=2
        else:raise ArithmeticError('Full-ten Newton step has no positive/descent continuation')
        g=trial
    else:raise ArithmeticError('Full-ten Gram stationarity did not converge')
    value,gradient,hessian=functional(g,data,stiffness)
    gram=matrix(g); minimum=mp.eigsy(gram,eigvals_only=True)[0]
    if minimum<=0:raise ArithmeticError('Selected Gram lies on/outside PSD boundary; no interior certificate')
    residual=mp.norm(gradient)
    return gram,value,residual,hessian


def class_transform(twist):
    absolute=[min(t,4-t) for t in twist]
    permutation=sorted(range(3),key=lambda i:absolute[i])
    representative=tuple(absolute[i] for i in permutation)
    transform=mp.zeros(4); transform[0,0]=1
    for j,i in enumerate(permutation):transform[i+1,j+1]=-1 if twist[i]==3 else 1
    return representative,transform


def seam_control(size,twist,gram):
    sites=periodic_grid(size); index={tuple(site):i for i,site in enumerate(sites)}
    links=np.empty((len(sites),3,4,4),complex); identity=np.eye(4,dtype=complex)
    for i,site in enumerate(sites):
        for axis in range(3):links[i,axis]=(CENTRES[twist[axis]] if site[axis]==size-1 else 1)*identity
    maximum=0.
    for i,site in enumerate(sites):
        for j,k in itertools.combinations(range(3),2):
            sj=site.copy();sj[j]=(sj[j]+1)%size
            sk=site.copy();sk[k]=(sk[k]+1)%size
            face=links[i,j]@links[index[tuple(sj)],k]@links[index[tuple(sk)],j].conj().T@links[i,k].conj().T
            maximum=max(maximum,float(np.linalg.norm(face-identity)))
    holonomy_error=0.
    for axis in range(3):
        cycle=identity.copy(); position=np.zeros(3,dtype=int)
        for _ in range(size):
            cycle=cycle@links[index[tuple(position)],axis]
            position[axis]=(position[axis]+1)%size
        holonomy_error=max(holonomy_error,float(np.linalg.norm(cycle-CENTRES[twist[axis]]*identity)))
    g=np.asarray([[float(z) for z in row] for row in gram.tolist()])
    eigen,vectors=np.linalg.eigh(g); root=(vectors*np.sqrt(eigen))@vectors.T
    frame=np.einsum('mn,nij->mij',root,SELECTED)
    alignment=max(float(np.linalg.norm(frame-u@frame@u.conj().T)) for u in links.reshape(-1,4,4))
    determinant_error=float(np.max(abs(np.linalg.det(links)-1)))
    wrong_root_determinant_error=max(abs(np.exp(2j*np.pi*t/size)-1) for t in twist)
    return dict(face_flux_error=maximum,link_SU4_determinant_error=determinant_error,
        fundamental_holonomy_error=holonomy_error,adjoint_alignment_error=alignment,
        excluded_uniform_scalar_Nthroot_determinant_error=float(wrong_root_determinant_error))


def native_band_control(cfg,size,twist,gram):
    sites=periodic_grid(size); rho=np.asarray([[float(z) for z in row] for row in gram.tolist()])
    eigen,vectors=np.linalg.eigh(rho); root=(vectors*np.sqrt(eigen))@vectors.T
    frame=np.einsum('mn,nij->mij',root,SELECTED); errors=[]
    for _,model,_ in models(replace(cfg,Lambda=1.)):
        onsite,paths=scalar_paths(model)
        for n in ((0,0,0),(2,3,5)):
            p=2*np.pi*(np.asarray(n)+np.asarray(twist)/4)/size
            Bloch=onsite*frame[0]
            for displacement,scalars in paths:
                hopping=np.einsum('m,mij->ij',scalars,frame)
                Bloch+=np.exp(1j*p@displacement)*hopping+np.exp(-1j*p@displacement)*hopping.conj().T
            for site in sites:
                actual=onsite*frame[0]*np.exp(1j*p@site)
                for displacement,scalars in paths:
                    hopping=np.einsum('m,mij->ij',scalars,frame)
                    target=(site+displacement)%size; previous=(site-displacement)%size
                    phase=np.prod([CENTRES[twist[j]]**int((site[j]+displacement[j])//size) for j in range(3)])
                    previous_phase=np.prod([CENTRES[twist[j]]**int((previous[j]+displacement[j])//size) for j in range(3)])
                    actual+=hopping*phase*np.exp(1j*p@target)+hopping.conj().T*np.conj(previous_phase)*np.exp(1j*p@previous)
                errors.append(float(np.linalg.norm(actual-np.exp(1j*p@site)*Bloch)))
    if max(errors)>1e-10:raise ArithmeticError('Literal SU4 seam hopping does not equal shifted fundamental Bloch bands')
    return dict(max_actual_seam_Bloch_action_error=max(errors),frame=frame,
        scope='Allthree actual profiles, one/two-link wraps and allfour orbital columns; scalarNthroot is only a forbidden physical-link alternative, never used as SU4 links')


def fast_pressure(gram,size):
    eigen,Q=mp.eigsy(gram); derivative=mp.zeros(4,1); coefficient=mp.mpf(0)
    for q in periodic_grid(size):
        lap=4*sum(mp.sin(mp.pi*int(z)/size)**2 for z in q)
        coefficient+=45*mp.sqrt(lap/5)/2
        for mask in range(1,16):
            indices=[i for i in range(4) if mask&(1<<i)]
            r=sum(eigen[i] for i in indices)/2
            transverse=3*r/25+21*lap/100; longitudinal=3*r/25+lap/5
            coefficient+=mp.sqrt(transverse)+mp.sqrt(longitudinal)/2
            slope=3/(100*mp.sqrt(transverse))+3/(200*mp.sqrt(longitudinal))
            for i in indices:derivative[i]+=slope
    volume=size**3
    return coefficient/volume,Q*mp.diag([z/volume for z in derivative])*Q.T


def preferred_quantum_mean(cfg,size,twist,exponent,seed):
    precision=max(320,math.ceil(exponent*math.log10(2))+120)
    with mp.workdps(precision):
        data=data_for(cfg,size,twist,exponent); stiffness=2/mp.mpf(cfg.a)
        gram,_,_,_=solve_gram(data,stiffness,mp.matrix(seed))
        g=vector(gram); pressure_scale=mp.power(2,-mp.mpf(5)*exponent/8)
        for _ in range(100):
            _,gradient,hessian=functional(g,data,stiffness)
            coefficient,force=fast_pressure(matrix(g),size)
            derivative=gradient+pressure_scale*vector(force)
            if mp.norm(derivative)<mp.power(10,-precision+40):break
            g-=mp.lu_solve(hessian,derivative)
        else:raise ArithmeticError('Preferred full-ten all-profile/ZPE mean failed precision closure')
        result=matrix(g); eigen=mp.eigsy(result,eigvals_only=True)
        if eigen[0]<=0:raise ArithmeticError('Quantum mean lost interior Gram positivity')
        return dict(precision=precision,G_decimal=[[mp.nstr(result[i,j],precision) for j in range(4)] for i in range(4)],
            quantum_uniform_gradient_decimal=mp.nstr(mp.norm(derivative),precision),
            fast_ZPE_coefficient_decimal=mp.nstr(coefficient,precision),
            scope='Actual three profiles on shifted N6 grid plus canonical leading full-Gauss fast pressure; not a new metastable-clock proof or bulk stationary mean')


def competitor_price(cfg,size,exponent,gram,periodic):
    with mp.workdps(200):
        scale=mp.power(2,exponent); chi=scale**(-mp.mpf(1)/2); volume=size**3; D=90*volume
        values=mp.eigsy(gram,eigvals_only=True); rmin=values[0]/2; rmax=sum(values)/2; rho=mp.sqrt(sum(gram[i,i] for i in range(4)))
        width=scale**(-mp.mpf(1)/4); radius=mp.sqrt(D)*width
        analytic=min(mp.mpf('.001'),rmin/(16*mp.sqrt(rmax)))
        if radius>=analytic/2:raise ArithmeticError('Competitor H2 product support does not fit the physical joint chart')
        caps=[]
        for load,model,role in models(replace(cfg,Lambda=1.)):
            onsite,paths=scalar_paths(model); cap=abs(float(onsite))+2*sum(float(np.sum(abs(s))) for _,s in paths)
            caps.append((load,mp.mpf(cap),role))
        heavy=caps[0][1]; light=sum(load*cap for load,cap,role in caps if not role.startswith('neutral'))
        metric=4+512/rmin; metric_first=8*metric/analytic; log_first=8/analytic; log_second=32/analytic**2
        measure=chi*(D**2*(metric_first*log_first+metric*log_second)/4+D*metric*log_first**2/8)
        charge=mp.sqrt(volume)*(3+cfg.auxiliary_count)*mp.sqrt(mp.mpf(5)/2)
        kinetic=chi*metric*3*D/width**2+4*chi*charge**2/rmin+measure
        heavy_change=32*scale*volume*heavy*(1+rho)*radius
        quartic_change=4*scale*volume/mp.mpf(cfg.a)*(rho+radius)**3*radius
        alignment_curl=128*volume*scale**(mp.mpf(5)/4)*(1+rho)**2*radius**2
        electric=mp.mpf(cfg.inertia)/mp.mpf(cfg.a)*3*(2*volume+1)/(mp.mpf(1)/8)**2
        coulomb=mp.mpf(cfg.inertia)/mp.mpf(cfg.a)*volume*(2*(1+cfg.auxiliary_count)+4)**2/(4*mp.sin(mp.pi/size)**2)
        light_change=16*volume*light*(1+rho+radius)
        cost=kinetic+heavy_change+quartic_change+alignment_curl+electric+coulomb+light_change
        periodic_rho=mp.sqrt(mp.mpf(periodic['quantum_mean']['g0_decimal'])+3*mp.mpf(periodic['quantum_mean']['gs_decimal']))
        lower_penalty=mp.mpf(periodic['prices']['mean_force_linear_action_on_chart'])+8*volume*light*(1+periodic_rho+mp.mpf(periodic['prices']['eta']))
        return dict(normalized_upper_extra=mp.nstr(cost/scale,100),normalized_periodic_lower_penalty=mp.nstr(lower_penalty/scale,100),
            actual_upper_extra_decimal=mp.nstr(cost,100),support_radius_decimal=mp.nstr(radius,100),width_decimal=mp.nstr(width,100),
            state='Normalized all90V-coordinate H2 product (1-(z/width)^2)^2_+, width=Lambda^-1/4, in exact normal Phi and near-central-seam link coordinates; joint SU4-equivariant lift with degree4/Haar normalization. Matter is exact positive-heavy and negative-charged Clifford spectral-projector Slater at the displayed mean; Ref4 onsite SU4 singlet, same total-neutral noncompact U1 sector. EM product H2 radius1/8 in all2V+1 real modes.',
            derivatives='No field-dependent Slater approximation: body matter is constant, its local Gauss charge enters the explicit affine kinetic square. <p_j^2>=3/width^2; homogeneous metric cap and charge-square bound price physical kinetic energy.',
            potential='Actual path Lipschitz bound prices heavy deviation; quartic mean-value bound prices Phi deviation; alignment/curl vanish at flat central seam and have displayed128 V Lambda^5/4 (1+rho)^2 radius^2 cap. Exact all-mode electric/Coulomb and unitary charged-path caps retained.',
            scope='Legitimate fixedNH=2V/full-centre Gauss variational competitor to the checked periodic local packet. Not its eigenstate, lifetime proof or global vacuum.')


def run(cfg,size,exponent,periodic_path):
    started=time.monotonic(); raw=Path(periodic_path).read_bytes(); periodic=json.loads(raw)
    if periodic['input_descriptor']['size']!=size or periodic['input_descriptor']['Lambda_exponent']!=exponent:raise ValueError('Periodic packet input mismatch')
    centre=global_centre_projection(cfg,size)
    if not centre['global_centre_admissible']:raise ArithmeticError('No full-centre Gauss competitor on this odd physical box')
    config={k:v for k,v in asdict(cfg).items() if k!='Lambda'}
    if any(periodic['input_descriptor']['config'][key]!=value for key,value in config.items()):
        raise ValueError('Periodic packet microscopic Hamiltonian/spacing/species differ')
    descriptor=dict(config={k:v for k,v in asdict(cfg).items() if k!='Lambda'},size=size,Lambda_exponent=exponent,
        EM_domain=periodic['input_descriptor']['reference_EM_packet']['Hilbert_domain'],periodic_receipt_sha256=hashlib.sha256(raw).hexdigest(),
        governing_plan=dict(id='issue-222-induced-photon-delta13-continuation-20261008',revision='cb2914f24b938d376070cc10a9681bd8dfb89b1c198628f01b8dffa7b4ca7bae',work_item='joined-explanatory-campaign'),
        sector='all65 filled species incl64charged/heavyneutral, fixed heavyNH=2V, Ref4 and totalcentreclass0',scan_precision=120)
    classes={}; sector_rows=[]; representatives=sorted({class_transform(t)[0] for t in itertools.product(range(4),repeat=3)})
    with mp.workdps(120):
        stiffness=2/mp.mpf(cfg.a); round_cap=mp.mpf('1e-90')
        for twist in representatives:
            data=data_for(cfg,size,twist,exponent); gram,value,residual,hessian=solve_gram(data,stiffness)
            certified_gradient=residual+round_cap; error=certified_gradient**2/(2*stiffness)
            classes[twist]=dict(G=gram,value=value,error=error,residual=residual,data=data)
        for twist in itertools.product(range(4),repeat=3):
            representative,transform=class_transform(twist); selected=classes[representative]
            gram=transform*selected['G']*transform.T
            value,gradient,_=functional(vector(gram),data_for(cfg,size,twist,exponent),stiffness)
            if mp.norm(gradient)>mp.mpf('1e-85') or abs(value-selected['value'])>mp.mpf('1e-85'):raise ArithmeticError('Actual all64 reflection/permutation equivalence failed')
            sector_rows.append(dict(twist=list(twist),representative=list(representative),energy=mp.nstr(value,110),
                full_ten_gradient=mp.nstr(mp.norm(gradient),110),seams=seam_control(size,twist,gram)))
        periodic_class=classes[(0,0,0)]; preferred=min(classes,key=lambda t:classes[t]['value']); chosen=classes[preferred]
        diagonal=mp.diag([chosen['G'][i,i] for i in range(4)])
        _,diagonal_gradient,_=functional(vector(diagonal),chosen['data'],stiffness)
        quarter=classes[(0,0,1)]; quarter_diag=mp.diag([quarter['G'][i,i] for i in range(4)])
        quarter_value,quarter_gradient,_=functional(vector(quarter_diag),quarter['data'],stiffness)
        class_rows=[dict(twist=list(t),multiplicity=sum(row['representative']==list(t) for row in sector_rows),
            G_decimal=[[mp.nstr(value,110) for value in row] for row in selected['G'].tolist()],
            minimum_Gram_eigenvalue=mp.nstr(mp.eigsy(selected['G'],eigvals_only=True)[0],110),
            energy_per_Lambda_per_site=mp.nstr(selected['value'],110),full_ten_gradient=mp.nstr(selected['residual'],110),
            strong_convex_optimizer_error=mp.nstr(selected['error'],110),
            delta_from_periodic_per_Lambda_box=mp.nstr(size**3*(selected['value']-periodic_class['value']),110)) for t,selected in classes.items()]
        native=native_band_control(cfg,size,(0,0,1),quarter['G']); native.pop('frame')
        quarter_adverse=dict(full_G_cross_component=mp.nstr(quarter['G'][0,3],110),
            forced_diagonal_full_ten_gradient=mp.nstr(mp.norm(quarter_gradient),110),
            forced_diagonal_energy_excess=mp.nstr(quarter_value-quarter['value'],110),
            meaning='Actual quarter-twist optimization uses allten Gram components. Diagonal-only select_gram is not an allowed optimizer.')
        chosen_seed=chosen['G'].tolist()
        difference=size**3*(chosen['value']-periodic_class['value'])
    mean=preferred_quantum_mean(cfg,size,preferred,exponent,chosen_seed)
    with mp.workdps(200):
        physical_gram=mp.matrix(mean['G_decimal'])
        price=competitor_price(cfg,size,exponent,physical_gram,periodic)
        margin=-(difference+mp.mpf(price['normalized_upper_extra'])+mp.mpf(price['normalized_periodic_lower_penalty'])+2*round_cap*size**3)
        if margin<=0:raise ArithmeticError('This parameter point does not prove a lower legitimate Gauss competitor')
        price['normalized_strict_energy_margin']=mp.nstr(margin,110)
        price['physical_mean_decimal']=mean['G_decimal']
    result=dict(branch='NEW all64 flat-Z4 finite-box full-Gram/physical-Gauss competitor discriminator; not global vacuum',
        input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        classes=class_rows,sectors=sector_rows,preferred_twist=list(preferred),preferred_quantum_mean=mean,
        quarter_cross_Gram_adverse=quarter_adverse,actual_shifted_Bloch_control=native,Gauss_competitor=price,
        centre=global_centre_projection(cfg,size),uniform_selection='For arbitrary traceless Herm4 fields, fixed-total-half-filled sea>=-Tr|h|/2>=-sqrt(Tr h^2)=-2sqrt(xGx). Clifford sqrt(G) frame saturates. Quadratic Gram term makes full-ten PSD objective strongly convex with constant2/a; interior positivity and full gradient certify global UNIFORM selection, not full spatial vacuum.',
        conclusion='Lower Gauss competitor to periodic prepared packet demonstrated with explicit normalized upper/lower energy prices; periodic checked metastable-clock result remains distinct. Preferred actual coupled full-Gram mean computed, no bulk/caller promotion.',
        source_at_import=SOURCE_HASHES,source_snapshot_at_import=dict(path=Path(__file__).name,sha256=SOURCE_HASHES[Path(__file__).name],text=SOURCE_BYTES[Path(__file__).name].decode()),
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,mpmath=mp.__version__,platform=platform.platform()),
        source_unchanged_since_import={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in SOURCE_HASHES.items()},elapsed_seconds=time.monotonic()-started)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--size',type=int,default=6);parser.add_argument('--Lambda-exponent',type=int,default=2048)
    parser.add_argument('--periodic-receipt',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run(PhiConfig(),args.size,args.Lambda_exponent,args.periodic_receipt)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=args.output,source_at_import=result['source_at_import'],input_sha256=result['input_sha256'],preferred=result['preferred_twist'],quarter=result['quarter_cross_Gram_adverse'],competitor=result['Gauss_competitor'],elapsed_seconds=result['elapsed_seconds']),indent=2))
