"""Exact same CAR32 Hamiltonian as a lazy SUM of bounded-channel MPOs.
No full-MPO tensor is allocated. The operator sum is exact; MPS image/splitting
compressions are separate, explicit numerical approximations.
"""
from types import SimpleNamespace
import numpy as np
from tenpy.networks.mpo import MPOEnvironment


class PhysicalOperatorSum:
    def __init__(self,model,terms,group_size=24):
        self.model=model;self.terms=list(terms);self.group_size=group_size
        self.groups=[];first=0
        while first<len(self.terms):
            last=min(first+group_size,len(self.terms))
            # Hamiltonian paired terms are indivisible Hermitian units. Derived
            # source sums need no individual-group Hermiticity for linear action.
            if last<len(self.terms) and last-1 in getattr(model,'pair_starts',set()):last-=1
            self.groups.append(self.terms[first:last]);first=last
        self.chi=[1]*(len(model.sites)+1)
        for group in self.groups:
            sizes=model.graph_sizes(group)
            self.chi=[max(a,b) for a,b in zip(self.chi,sizes)]

    def pieces(self,reverse=False):
        groups=reversed(self.groups) if reverse else self.groups
        for terms in groups:yield self.model.mpo_small(terms)

    def matrix_element(self,bra,ket):
        factor=np.conj(bra.norm)*ket.norm
        return complex(sum(factor*MPOEnvironment(bra,piece,ket).full_contraction(0) for piece in self.pieces()))

    def expectation_value(self,psi):
        return self.matrix_element(psi,psi)/psi.overlap(psi)

    def apply(self,psi,options):
        """Full operator image via exact piece images and disclosed compression.

        Exact uncompressed norm of EACH bounded-MPO image is evaluated before
        compression. Those piece norms are NOT a total-H norm or variance.
        """
        original=psi.copy();total=None;records=[];eps_sum=0.
        trunc=options.get('trunc_params',{'chi_max':12,'svd_min':1e-12})
        for index,piece in enumerate(self.pieces()):
            image=original.copy();piece.apply_naively(image)
            n2=float(image.overlap(image).real)
            if n2<0:raise ArithmeticError(('negative piece-image Gram',index,n2))
            if n2==0:
                records.append(dict(group=index,uncompressed_norm2=0.,zero_image=True));continue
            error=image.compress_svd(trunc);eps_sum+=float(error.eps)
            image.canonical_form(renormalize=False)
            record=dict(group=index,uncompressed_norm2=n2,compressed_norm2=float(image.overlap(image).real),image_svd_eps=float(error.eps))
            if total is None:total=image
            else:
                total=total.add(image,1.,1.)
                addition=total.compress_svd(trunc);eps_sum+=float(addition.eps)
                record['partial_sum_svd_eps']=float(addition.eps)
            records.append(record)
        if total is None:
            psi.norm=0.
        else:
            total.canonical_form(renormalize=False)
            psi.__dict__.update(total.__dict__)
        return SimpleNamespace(eps=eps_sum,pieces=records,meaning='Sum of individual-image and partial-sum SVD eps values, NOT a global discarded probability or rigorous total-H error bound.')

    def evolve(self,psi,dt,chi):
        """Symmetric full-H group splitting, centred RK2 for each half group.

        (I+z2 K)(I+z1 K)=I-i*tau*K-tau²*K²/2 with
        z1=-i*tau*(1+i)/2, z2=-i*tau*(1-i)/2. Every K is a Hermitian
        Hamiltonian group minus a scalar phase; identity prevents zero-image
        SVDs. No local exponential of a huge RF tensor or source projection.
        """
        records=[];log_raw_norm=0.;scalar_phase=0.
        options={'compression_method':'zip_up','trunc_params':{'chi_max':chi,'svd_min':1e-12}}
        for reverse in (False,True):
            for index,piece in enumerate(self.pieces(reverse)):
                norm=float(psi.overlap(psi).real)
                mean=float((np.conj(psi.norm)*psi.norm*MPOEnvironment(psi,piece,psi).full_contraction(0)/norm).real)
                tau=dt/2
                for stage,z in enumerate((-.5j*tau*(1+1j),-.5j*tau*(1-1j))):
                    propagator=piece.plus_identity(1-z*mean,z)
                    error=propagator.apply(psi,options)
                    psi.canonical_form(renormalize=False)
                    n2=float(psi.overlap(psi).real)
                    records.append(dict(reverse=reverse,group=index,stage=stage,centering_energy=mean,svd_eps=float(error.eps),pre_normalization_norm2=n2))
                    log_raw_norm+=.5*np.log(n2)
                    psi.norm/=np.sqrt(n2)
                # TeNPy1.1.1 MPS.norm is REAL: MPSEnvironment multiplies
                # bra.norm*ket.norm without conjugation. Put phase in tensor.
                psi.set_B(0,psi._B[0]*np.exp(-1j*mean*tau),form=psi.form[0])
                scalar_phase+=mean*tau
        return dict(dt=dt,chi=chi,groups=len(self.groups),stages=records,log_removed_raw_norm_factor=float(log_raw_norm),restored_scalar_phase_exponent=float(scalar_phase),centering_phase_restored=True,method='Exact same operator SUM and exact RK2-polynomial factorization, NOT exact exponential evolution. Symmetric Hermitian-group splitting plus MPS compression; finite dt/bond errors. Scalar centering phases restored; removed raw norm product retained separately. No full-MPO/variance claim.')
