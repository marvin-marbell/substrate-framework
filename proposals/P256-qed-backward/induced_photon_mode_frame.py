#!/usr/bin/env python3
"""Physical two-mode RPA frame from the actual uncompressed filled-band kernel.

The frozen response source is a current/contact/density provider. Its independent
root eigenvectors are not used here. Exact cubic little groups give a common
root and irreducible transverse frame; generic q uses the nonlinear spectral
pencil and its full field-plus-particle-hole divided-difference norm.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from induced_photon_response import ResponseConfig, SpectralResponse

ROOT=Path(__file__).resolve().parent
LOADED_SOURCE_BYTES=Path(__file__).read_bytes()
LOADED_SOURCE_SHA256=hashlib.sha256(LOADED_SOURCE_BYTES).hexdigest()


def transverse_frame(q,qhat):
    """Only exact symmetry directions qualify; there is no angular tolerance."""
    occupied=np.flatnonzero(q)
    if len(occupied)==1:
        axis=int(occupied[0])
        return np.eye(3)[:,[j for j in range(3) if j!=axis]],'C4 axis'
    if len(occupied)==3 and np.all(np.abs(q)==abs(q[0])):
        signs=np.sign(q)
        direction=signs/math.sqrt(3)
        first=np.array([signs[1],-signs[0],0.])/math.sqrt(2)
        return np.column_stack((first,np.cross(direction,first))),'S3 signed body diagonal'
    direction=qhat/np.linalg.norm(qhat)
    return np.linalg.svd(direction[None,:])[2][1:].T,'generic q'


class ModeKernel:
    def __init__(self,cfg,data):
        self.cfg=cfg;self.data=data
        self.q=np.asarray(data['q']);self.qhat=np.asarray(data['qhat'])
        if not np.linalg.norm(self.qhat):
            raise ValueError('Nonzero lattice momentum required')
        self.t,self.symmetry=transverse_frame(self.q,self.qhat)
        self.delta=data['delta'];self.delta2=self.delta**2
        self.factor=cfg.flavors/cfg.a;self.bare=1/(2*cfg.inertia)
        self.kt=self.factor*(self.t.T@data['static']@self.t)
        self.wt=np.einsum('ia,bij,jc->bac',self.t,data['weights'],self.t).real
        self.mt=np.asarray(data['mixed_weights']@self.t).real
        self.density_delta=data['density']*self.delta
        self.qphysical2=float(self.qhat@self.qhat)/cfg.a**2
        self.stiffness=float(np.linalg.norm(self.kt,2))
        if np.min(np.linalg.eigvalsh(self.kt))<=0:
            raise ArithmeticError('Actual transverse stiffness is not positive')
        # Group averaging of the integral, not fitting or subtracting a mass.
        # Uniform isotropic selected Phi, isotropic Peierls paths and the full
        # signed cubical BZ rule are invariant under C4/S3. A real symmetric
        # operator on their transverse irreducible doublet is scalar; density
        # is the trivial representation and has no transverse matrix element.
        self.scalar_weight=np.trace(self.wt,axis1=1,axis2=2)/2
        self.scalar_stiffness=float(np.trace(self.kt)/2)

    def constraint(self,x):
        inv=1/(self.delta2-x)
        chi=self.bare*self.qphysical2+self.cfg.flavors/self.cfg.a**3*np.sum(self.density_delta*inv)
        d=self.cfg.flavors/self.cfg.a**2*np.einsum('bi,b->i',self.mt,inv)
        cp=self.cfg.flavors/self.cfg.a**3*np.sum(self.density_delta*inv**2)
        dp=self.cfg.flavors/self.cfg.a**2*np.einsum('bi,b->i',self.mt,inv**2)
        return float(chi),d,float(cp),dp

    def matrices(self,x):
        if not 0<=x<float(np.min(self.delta2)):
            raise ValueError('Real retarded kernel requires frequency below every sampled interband pole')
        inv=1/(self.delta2-x)
        chi,d,cp,dp=self.constraint(x)
        p=self.factor*np.einsum('bij,b->ij',self.wt,inv/self.delta)
        f=self.bare*x*np.eye(2)-self.kt+x*p-x*np.outer(d,d)/chi
        g=self.bare*np.eye(2)+self.factor*np.einsum('bij,b->ij',self.wt,self.delta*inv**2)
        g-=np.outer(d,d)/chi+x*((np.outer(dp,d)+np.outer(d,dp))/chi-np.outer(d,d)*cp/chi**2)
        return (f+f.T)/2,(g+g.T)/2

    def divided_norm(self,x,y):
        """Stable analytic [F(x)-F(y)]/(x-y), including signed A0 Schur term.

        For coincident roots this is Fprime, not an Euclidean polarization Gram.
        This eliminates cancellation even when the true splitting is unresolved.
        """
        invx=1/(self.delta2-x);invy=1/(self.delta2-y)
        g=self.bare*np.eye(2)+self.factor*np.einsum('bij,b->ij',self.wt,self.delta*invx*invy)
        cx,dx,_,_=self.constraint(x);cy,dy,_,_=self.constraint(y)
        dd=self.cfg.flavors/self.cfg.a**2*np.einsum('bi,b->i',self.mt,invx*invy)
        cd=self.cfg.flavors/self.cfg.a**3*np.sum(self.density_delta*invx*invy)
        nx=np.outer(dx,dx);ny=np.outer(dy,dy)
        nd=(np.outer(dd,dx+dy)+np.outer(dx+dy,dd))/2
        quotient_div=nd*(1/cx+1/cy)/2-(nx+ny)*cd/(2*cx*cy)
        g-=(nx/cx+ny/cy)/2+(x+y)*quotient_div/2
        return (g+g.T)/2

    def scalar(self,x):
        inv=1/(self.delta2-x)
        f=self.bare*x-self.scalar_stiffness+self.factor*x*np.sum(self.scalar_weight*inv/self.delta)
        g=self.bare+self.factor*np.sum(self.scalar_weight*self.delta*inv**2)
        return float(f),float(g)

    def norm_gram(self,x,vectors):
        return np.array([[vectors[:,i]@self.divided_norm(x[i],x[j])@vectors[:,j]
                          for j in range(2)] for i in range(2)])

    def solve(self):
        threshold2=float(np.min(self.delta2))
        scale=self.scalar_stiffness/(self.bare+self.factor*np.sum(self.scalar_weight/self.delta**3))
        upper=threshold2*(1-1e-10)/scale
        if self.symmetry!='generic q':
            root=scale*brentq(lambda y:self.scalar(y*scale)[0]/self.stiffness,0.,upper,xtol=1e-13,rtol=np.float64(1e-13))
            x=np.full(2,root)
            derivative=self.scalar(root)[1]
            vectors=np.eye(2)/math.sqrt(derivative)
            method='Exact cubic transverse irreducible doublet; group-averaged uncompressed integral, ONE common root'
        else:
            g0=self.matrices(0.)[1]
            x=np.asarray(eigh(self.kt,g0,eigvals_only=True))
            for _ in range(4):
                center=float(np.mean(x))
                f,g=self.matrices(center)
                offset,vectors=eigh(-f,g)
                x=center+offset
            if np.min(x)<=0 or np.max(x)>=threshold2:
                raise ArithmeticError('Clustered physical poles leave their positive interband domain')
            vectors=np.column_stack([vectors[:,i]/math.sqrt(vectors[:,i]@self.matrices(float(x[i]))[1]@vectors[:,i])
                                     for i in range(2)])
            residual=max(np.linalg.norm(self.matrices(float(x[i]))[0]@vectors[:,i])/np.linalg.norm(vectors[:,i])
                         for i in range(2))/self.stiffness
            gram=self.norm_gram(x,vectors)
            method='Common uncompressed F/Fprime generalized eigencluster; original nonlinear residual priced'
            if residual>1e-8 or np.max(abs(gram-np.eye(2)))>1e-7:
                # Resolved modes need their exact nonlinear eigenvectors, not a
                # common-frequency approximation and not Gram-Schmidt repair.
                roots=[];columns=[]
                for branch in (1,0):
                    root=scale*brentq(lambda y:np.linalg.eigvalsh(self.matrices(y*scale)[0])[branch]/self.stiffness,
                                     0.,upper,xtol=1e-13,rtol=np.float64(1e-13))
                    values,frame=np.linalg.eigh(self.matrices(root)[0])
                    vector=frame[:,int(np.argmin(abs(values)))]
                    vector/=math.sqrt(vector@self.matrices(root)[1]@vector)
                    roots.append(root);columns.append(vector)
                x=np.asarray(roots);vectors=np.column_stack(columns)
                method='Resolved exact uncompressed nonlinear poles with full field-plus-PH divided-difference norm'
        gram=self.norm_gram(x,vectors)
        error=float(np.max(abs(gram-np.eye(2))))
        if error>1e-7:
            raise ArithmeticError(f'Physical photon multiplet is not canonically complete: {gram}')
        return x,vectors,gram,method


def compute_mode(config,q,*,velocity=False,model=None,data=None):
    """Actual uncompressed pair. Optional model/data avoid repeated BZ evaluation."""
    model=SpectralResponse(config,config.orders[-1]) if model is None else model
    q=np.asarray(q,float)
    data=model.spectral(q) if data is None else data
    kernel=ModeKernel(config,data)
    x,amplitudes,gram,method=kernel.solve()
    poles=[];backward=[];symmetry_residual=[]
    gzero=kernel.matrices(0.)[1]
    for i in range(2):
        vector=amplitudes[:,i];z=float(vector@vector);unit=vector/math.sqrt(z)
        physical=kernel.t@unit
        f,g=kernel.matrices(float(x[i]))
        chi,d,_,_=kernel.constraint(float(x[i]))
        residual=float(np.linalg.norm(f@unit))
        price=residual/kernel.bare
        backward.append(price)
        scalar=(0. if kernel.symmetry!='generic q' else math.sqrt(x[i])*float(unit@d)/chi)
        poles.append(dict(omega=float(math.sqrt(x[i])),residue=z,polarization=physical.tolist(),
            scalar_polarization=scalar,local_omega=math.sqrt(float(unit@kernel.kt@unit)/(unit@gzero@unit)),
            epsilon=float(unit@gzero@unit),kappa=float(unit@kernel.kt@unit)/kernel.qphysical2,
            inverse_residual=residual,backward_error_omega_upper=price/(2*math.sqrt(x[i])),
            raw_scalar_polarization=float(math.sqrt(x[i])*float(unit@d)/chi)))
    for point in (0.,float(np.mean(x))/2,float(np.mean(x))):
        f,g=kernel.matrices(point)
        if kernel.symmetry!='generic q':
            fs,gs=kernel.scalar(point)
            symmetry_residual.append(dict(omega_squared=point,inverse_operator_error=float(np.linalg.norm(f-fs*np.eye(2),2)),
                                          norm_operator_error=float(np.linalg.norm(g-gs*np.eye(2),2))))
    result=dict(q=q.tolist(),omega=float(np.mean([p['omega'] for p in poles])),
        residue=float(np.mean([p['residue'] for p in poles])),two_polarizations=poles,
        pair_threshold=float(np.min(data['delta'])),pair_threshold_lower=2*config.m*math.sqrt(config.g0),
        pair_threshold_kind='Sampled actual interband minimum; independent analytic 2m sqrt(g0) lower bound also supplied',
        polarization_speed_splitting=float(math.sqrt(x[1]/x[0])-1),quadrature_order=model.order,
        quadrature_points=model.quadrature_points,canonical_mode_gram_error=float(np.max(abs(gram-np.eye(2)))),
        frame_proof=dict(method=method,little_group=kernel.symmetry,canonical_mode_gram=gram.tolist(),
            spatial_frame_gram=(np.asarray([p['polarization'] for p in poles])@np.asarray([p['polarization'] for p in poles]).T).tolist(),
            field_amplitude_gram=(amplitudes.T@amplitudes).tolist(),
            norm='Ai^T [F(xi)-F(xj)]/(xi-xj) Aj; diagonal Fprime. Analytic divided difference includes actual signed mixed-current/density A0 Schur term.',
            gap_bins=0,symmetry_residual=symmetry_residual,backward_error_omega_squared_upper=backward,
            backward_error_scope='Positive canonical PH realization gives Fprime >= eps_b below the first PH pole; raw-kernel residual / eps_b bounds pole perturbation. BZ quadrature and full-theory errors are separate.',
            rigorous_interband_window=bool(max(p['omega'] for p in poles)<2*config.m*math.sqrt(config.g0)),
            new_mode_source_sha256=LOADED_SOURCE_SHA256,
            old_independent_root_frames='Not used; old complete two-polarization claims withdrawn. Frozen kernel coefficients are unchanged.'))
    if velocity:
        # Ward-complete quadratic effective action: d(omega)/dq_a =
        # -Z e^T(dF/dq_a)e/(2omega). This is the one-quantum energy-current
        # expectation divided by its canonical norm, not a sorted-branch cusp.
        step=max(float(np.linalg.norm(q))*1e-3,1e-5)
        gradients=[];operators=[]
        physical_amplitudes=kernel.t@amplitudes
        for resolution in (step,step/2):
            velocity_values=np.zeros((2,3));velocity_operators=[]
            for axis in range(3):
                dq=np.eye(3)[axis]*resolution
                plus=ModeKernel(config,model.spectral(q+dq));minus=ModeKernel(config,model.spectral(q-dq))
                ap=plus.t.T@physical_amplitudes;am=minus.t.T@physical_amplitudes
                for i in range(2):
                    fp=plus.matrices(float(x[i]))[0];fm=minus.matrices(float(x[i]))[0]
                    velocity_values[i,axis]=-(ap[:,i]@fp@ap[:,i]-am[:,i]@fm@am[:,i])/(4*resolution*math.sqrt(x[i]))
                if kernel.symmetry!='generic q':
                    fp=plus.matrices(float(x[0]))[0];fm=minus.matrices(float(x[0]))[0]
                    velocity_operators.append((-(ap.T@fp@ap-am.T@fm@am)/(4*resolution*math.sqrt(x[0]))).tolist())
            gradients.append(velocity_values);operators.append(velocity_operators)
        corrected=(4*gradients[1]-gradients[0])/3
        result['group_velocity']=np.mean(corrected,axis=0).tolist()
        for i,pole in enumerate(poles):
            pole['group_velocity']=corrected[i].tolist()
        result['frame_proof']['energy_current']=dict(
            identity='Projected quadratic-action Noether energy flux / one-quantum energy = -Ai.T(dq F)Ai/(2 omega); Ai.T Fprime Ai=1',
            classical_plane_wave='Average energy density = omega^2*a.T Fprime a/2; average flux_i = -omega*a.T(dq_i F)a/4. Their ratio is the returned velocity; the PH and A0 terms are retained in F.',
            derivative='Actual neighbouring uncompressed kernels at FIXED omega and projected SAME physical field state; Richardson h^2 extrapolation',
            step=step,coarse=gradients[0].tolist(),half_step=gradients[1].tolist(),
            resolution_error=float(np.max(abs(gradients[0]-gradients[1]))),
            degenerate_velocity_operators=operators[-1],
            degeneracy_scope='At an exact doublet, these are prepared-frame energy-current expectations; velocity operators need not commute. No unique sorted-band Cartesian gradient is claimed.')
    return result


def compute_response(config=None):
    """Current physical response entry; all raw independent-root frames discarded."""
    from induced_photon_response import compute_response as raw_response
    cfg=config or ResponseConfig()
    result=raw_response(cfg)
    for row in result['raw_grids']:
        model=SpectralResponse(cfg,row['order'])
        for finite in row['finite_q']:
            finite['modes']=compute_mode(cfg,np.asarray(finite['q']),model=model)
    result['source_hashes']['induced_photon_mode_frame.py']=LOADED_SOURCE_SHA256
    result['physical_mode_scope']='Uncompressed physical paired-mode API; frozen raw provider used only for current/contact/density and scalar response statistics. Legacy independent-root two-pol frames are discarded.'
    return result


def current_frame_control(model,result,data,comparison=None) -> dict[str,Any]:
    q=np.asarray(result['q']);p=np.array([.13,-.21,.17])
    current=model.vertices(p[None,:],model.config.a*q)[0][0]*model.config.a
    fields=np.column_stack([np.r_[r['scalar_polarization'],r['polarization']]*math.sqrt(r['residue']/(2*r['omega']))
                            for r in result['two_polarizations']])
    def superoperator(columns):
        legs=[columns[0,i]*np.eye(4)-np.einsum('a,aij->ij',columns[1:,i],current) for i in range(2)]
        return sum(np.kron(leg.conj(),leg) for leg in legs)
    angle=.731;rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    good=superoperator(fields);rotated=superoperator(fields@rotation)
    bad=superoperator(np.column_stack((fields[:,0],fields[:,0])))
    kernel=ModeKernel(model.config,data)
    first=result['two_polarizations'][0]
    amplitude=kernel.t.T@np.asarray(first['polarization'])*math.sqrt(first['residue'])
    wrong_gram=np.full((2,2),float(amplitude@kernel.divided_norm(first['omega']**2,first['omega']**2)@amplitude))
    controls=dict(actual_current_unpolarized_rotation_error=float(np.linalg.norm(good-rotated)/np.linalg.norm(good)),
                  duplicate_state_current_error=float(np.linalg.norm(good-bad)/np.linalg.norm(good)),
                  duplicate_state_canonical_gram_error=float(np.max(abs(wrong_gram-np.eye(2)))),
                  scope='Actual literal Peierls current plus onsite density, quantum-normalized field legs. Rotation is a complete-state mixture identity, not a scattering-rate or Ref4 admission claim.')
    if comparison is not None:
        previous_fields=np.column_stack([np.r_[r['scalar_polarization'],r['polarization']]*math.sqrt(r['residue']/(2*r['omega']))
                                         for r in comparison])
        previous_amplitudes=np.column_stack([kernel.t.T@np.asarray(r['polarization'])*math.sqrt(r['residue'])
                                              for r in comparison])
        previous_gram=kernel.norm_gram(np.asarray([r['omega']**2 for r in comparison]),previous_amplitudes)
        controls['observed_legacy_canonical_gram']=previous_gram.tolist()
        controls['observed_legacy_canonical_gram_error']=float(np.max(abs(previous_gram-np.eye(2))))
        controls['observed_legacy_current_operator_error']=float(np.linalg.norm(good-superoperator(previous_fields))/np.linalg.norm(good))
    return controls


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supplier',type=Path,nargs='+',required=True)
    parser.add_argument('--observed-frame',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic();rows=[];inputs=[]
    for path in args.supplier:
        raw=path.read_bytes();capsule=json.loads(raw);cfg=ResponseConfig(**capsule['config'])
        inputs.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),config=asdict(cfg)))
        model=SpectralResponse(cfg,cfg.orders[-1])
        for q in (np.array([0.,0.,.1]),np.full(3,.22053099201074255),np.array([.07,.05,.03])):
            data=model.spectral(q)
            result=compute_mode(cfg,q,velocity=True,model=model,data=data)
            control=current_frame_control(model,result,data)
            if control['actual_current_unpolarized_rotation_error']>1e-12 or control['duplicate_state_current_error']<1e-3:
                raise AssertionError('Actual current does not discriminate incomplete/rotated physical state frames')
            rows.append(dict(supplier_sha256=inputs[-1]['sha256'],mode=result,controls=control))
            if np.all(q==.22053099201074255):
                reference_data=data
                reference_result=result
            print(json.dumps(dict(a=cfg.a,q=q.tolist(),Gram=result['canonical_mode_gram_error'],method=result['frame_proof']['method'])),flush=True)
    observed=None
    if args.observed_frame:
        raw=args.observed_frame.read_bytes();old=json.loads(raw)
        first=old['before']['two_polarizations'][0];second=old['separate']['two_polarizations'][1]
        if reference_result['quadrature_points']!=old['before']['quadrature_points']:
            raise ValueError('Observed legacy frame requires its actual mass-profile supplier as the final input')
        legacy_control=current_frame_control(model,reference_result,reference_data,(first,second))
        if legacy_control['observed_legacy_canonical_gram_error']<.1:
            raise AssertionError('Observed legacy frame must fail the full PH canonical-state discriminator')
        observed=dict(input_sha256=hashlib.sha256(raw).hexdigest(),source_hashes=old['source_hashes'],
            first=first,second=second,q0=old['q0'],q1=old['q1'],reported_old_overlap=old['separate_root_initial_overlap'],
            full_PH_and_literal_current_falsifier=legacy_control,
            scope='Previously observed legacy separate-root incident-frame failure; NOT rerun. Different printed q values are numerical splitting of the exact cubic degenerate beam, not a claim about Fock overlap of distinct exact momenta.')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=LOADED_SOURCE_SHA256:
        raise RuntimeError('Mode source changed during scientific proof')
    output=dict(branch='NEW uncompressed physical photon multiplet and fixed-state energy-current',inputs=inputs,
        input_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),rows=rows,
        observed_legacy_failure=observed,source_hashes={**capsule['source_hashes'],'induced_photon_mode_frame.py':LOADED_SOURCE_SHA256},
        source_snapshot=LOADED_SOURCE_BYTES.decode(),elapsed_seconds=time.monotonic()-started,
        quantum_scope='Same conditional quadratic large-N/RPA law; full finiteN/Root and cone closure are not promoted.')
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(f'Natural exit: {args.output}',flush=True)

if __name__=='__main__':
    main()
