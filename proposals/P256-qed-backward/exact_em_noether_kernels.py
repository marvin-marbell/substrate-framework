#!/usr/bin/env python3
"""All52 Gaussian moments of the unchanged97 scalar Noether functions.

No photon/state/profile whitening. High precision retains cancellation of the
actual gamma Weyl terms. Sinusoidal blocks use stable analytic sinh identities.
"""
from __future__ import annotations

from typing import Any

import mpmath as mp
import numpy as np


class PolynomialMoments:
    """Literal finite polynomial/Weyl functions, with cached full52 dot products."""
    def __init__(self, bank: Any, digits: int = 60):
        self.digits = digits
        self.phases = np.asarray(bank.coefficients['phases'], float)
        self.flows = np.asarray(bank.coefficients['coexact'], float)
        self.functions = bank.gaussian_functions()
        self.directions: list[tuple[Any, ...]] = []
        self.indices: dict[tuple[complex, ...], int] = {}
        self.polynomials: list[list[tuple[dict[tuple[int, ...], Any], tuple[tuple[int, int], ...]]]] = []
        with mp.workdps(digits):
            self.k = [tuple(mp.mpf(float(x)) for x in row) for row in self.phases]
            self.flow_ids = [self.register(row) for row in self.flows]
            for function in self.functions:
                terms = []
                for term in function.terms:
                    ids = [self.register(row) for row in term.polynomial.projections]
                    polynomial = {tuple(sorted(ids[i] for i in factors)): mp.mpc(complex(value))
                                  for factors,value in term.polynomial.terms.items()}
                    terms.append((polynomial,self.phase_signature(term.phase)))
                self.polynomials.append(terms)
            self.conjugates = [self.register(np.asarray([complex(x).conjugate() for x in row]))
                               for row in list(self.directions)]
        self.dot_cache: dict[tuple[int,...], Any] = {}
        self.direction_phase_cache: dict[tuple[int,int], Any] = {}
        self.phase_dot_cache: dict[tuple[int,...], Any] = {}
        self.derivative_cache: dict[tuple[int,int,int], dict[tuple[int,...],Any]] = {}
        self.set_displacement(np.zeros(52))

    def register(self, row: Any) -> int:
        key = tuple(complex(x) for x in row)
        if key not in self.indices:
            self.indices[key] = len(self.directions)
            self.directions.append(tuple(mp.mpc(x) for x in key))
        return self.indices[key]

    def phase_signature(self, phase: np.ndarray) -> tuple[tuple[int,int], ...]:
        if not np.any(phase):
            return ()
        for sign in (1,-1):
            found = np.flatnonzero(np.all(self.phases*sign==phase,axis=1))
            if len(found):
                return ((int(found[0]),sign),)
        raise ValueError('Scalar library contains a phase outside its literal81 curl catalog')

    @staticmethod
    def combine(*signatures: tuple[tuple[int,int], ...]) -> tuple[tuple[int,int], ...]:
        values: dict[int,int] = {}
        for signature in signatures:
            for index,sign in signature:
                values[index] = values.get(index,0)+sign
        return tuple((index,sign) for index,sign in sorted(values.items()) if sign)

    @staticmethod
    def negative(signature: tuple[tuple[int,int], ...]) -> tuple[tuple[int,int], ...]:
        return tuple((index,-sign) for index,sign in signature)

    def dot(self, left: int, right: int) -> Any:
        key = tuple(sorted((left,right)))
        if key not in self.dot_cache:
            self.dot_cache[key] = mp.fsum(a*b for a,b in zip(self.directions[left],self.directions[right]))
        return self.dot_cache[key]

    def direction_phase(self, direction: int, phase: int) -> Any:
        key = direction,phase
        if key not in self.direction_phase_cache:
            self.direction_phase_cache[key] = mp.fsum(a*b for a,b in zip(self.directions[direction],self.k[phase]))
        return self.direction_phase_cache[key]

    def phase_dot(self, left: int, right: int) -> Any:
        key = tuple(sorted((left,right)))
        if key not in self.phase_dot_cache:
            self.phase_dot_cache[key] = mp.fsum(a*b for a,b in zip(self.k[left],self.k[right]))
        return self.phase_dot_cache[key]

    def set_displacement(self, displacement: np.ndarray) -> None:
        with mp.workdps(self.digits):
            d = tuple(mp.mpf(float(x)) for x in displacement)
            self.dd = mp.fsum(x*x for x in d)
            self.dk = [mp.fsum(x*y for x,y in zip(d,k)) for k in self.k]
            self.dv = [mp.fsum(x*y for x,y in zip(d,v)) for v in self.directions]
        self.characteristics: dict[tuple[tuple[int,int],...],Any] = {}

    def characteristic(self, signature: tuple[tuple[int,int],...]) -> Any:
        if signature not in self.characteristics:
            squared = self.dd+2*mp.fsum(sign*self.dk[index] for index,sign in signature)
            squared += mp.fsum(a*b*self.phase_dot(i,j) for i,a in signature for j,b in signature)
            self.characteristics[signature] = mp.exp(-squared/4)
        return self.characteristics[signature]

    def polynomial_expectation(self, polynomial: dict[tuple[int,...],Any],
                               signature: tuple[tuple[int,int],...]) -> Any:
        cache: dict[tuple[int,...],Any] = {():mp.mpc(1)}
        means: dict[int,Any] = {}
        def mean(index: int) -> Any:
            if index not in means:
                means[index] = .5j*(self.dv[index]+mp.fsum(sign*self.direction_phase(index,phase)
                                                        for phase,sign in signature))
            return means[index]
        def moment(factors: tuple[int,...]) -> Any:
            if factors not in cache:
                first,rest = factors[0],factors[1:]
                value = mean(first)*moment(rest)
                for position,second in enumerate(rest):
                    value += self.dot(first,second)/2*moment(rest[:position]+rest[position+1:])
                cache[factors] = value
            return cache[factors]
        return self.characteristic(signature)*mp.fsum(value*moment(factors) for factors,value in polynomial.items())

    def conjugate(self, polynomial: dict[tuple[int,...],Any]) -> dict[tuple[int,...],Any]:
        return {tuple(sorted(self.conjugates[i] for i in factors)):value.conjugate()
                for factors,value in polynomial.items()}

    @staticmethod
    def product(left: dict[tuple[int,...],Any],right: dict[tuple[int,...],Any]) -> dict[tuple[int,...],Any]:
        out: dict[tuple[int,...],Any] = {}
        for a,x in left.items():
            for b,y in right.items():
                key = tuple(sorted(a+b))
                out[key] = out.get(key,0)+x*y
        return out

    def kernel(self,left: int,right: int,extra: tuple[tuple[int,int],...]=()) -> complex:
        values = []
        for a,pa in self.polynomials[left]:
            ca = self.conjugate(a)
            for b,pb in self.polynomials[right]:
                signature = self.combine(self.negative(pa),pb,extra)
                values.append(self.polynomial_expectation(self.product(ca,b),signature))
        return complex(mp.fsum(values))

    def sine_kernel(self,left: int,right: int,phase: int) -> complex:
        return (self.kernel(left,right,((phase,1),))-self.kernel(left,right,((phase,-1),)))/(2j)

    def derivative(self,right: int,term_index: int,flow: int) -> dict[tuple[int,...],Any]:
        key = right,term_index,flow
        if key not in self.derivative_cache:
            polynomial,signature = self.polynomials[right][term_index]
            direction = self.flow_ids[flow]
            phase_dot = mp.fsum(sign*self.direction_phase(direction,index) for index,sign in signature)
            out = {factors:1j*phase_dot*value for factors,value in polynomial.items()}
            for factors,value in polynomial.items():
                for position,index in enumerate(factors):
                    remaining = factors[:position]+factors[position+1:]
                    out[remaining] = out.get(remaining,0)+value*self.dot(direction,index)
            self.derivative_cache[key] = out
        return self.derivative_cache[key]

    def cross(self,left: int,right: int) -> complex:
        """<f_left, flow.grad(f_right)>; common Gaussian cancels in SA cross."""
        values = []
        for a,pa in self.polynomials[left]:
            ca = self.conjugate(a)
            for term_index,(_,pb) in enumerate(self.polynomials[right]):
                for flow in range(len(self.k)):
                    derivative = self.derivative(right,term_index,flow)
                    if not any(derivative.values()):
                        continue
                    product = self.product(ca,derivative)
                    for sign in (-1,1):
                        signature = self.combine(self.negative(pa),pb,((flow,sign),))
                        values.append(sign/(2j)*self.polynomial_expectation(product,signature))
        return complex(mp.fsum(values))


class ExactNoetherKernels:
    """Actual97 scalar weak kernels; no SOURCE-based fitted matrix identities."""
    def __init__(self, model: Any, digits: int = 60):
        self.model,self.bank,self.digits = model,model.noether_bank,digits
        self.engine = PolynomialMoments(self.bank,digits)
        self.k = self.engine.phases
        self.u = self.engine.flows
        self.kk = self.k@self.k.T
        self.uk = self.u@self.k.T
        self.norms = np.diag(self.kk)
        self.hard = self.bank.hard
        self.labels = self.hard.labels
        self.odd_hard = [i for i,n in enumerate(self.labels) if sum(n)==1]
        self.even_hard = [i for i,n in enumerate(self.labels) if sum(n)%2==0]
        self.weyl_cache: dict[bytes,np.ndarray] = {}
        self.controls: list[dict[str,Any]] = []
        self._build()

    def hard_sine(self, displacement: np.ndarray) -> np.ndarray:
        """Literal hard polynomials, retaining native covariance-rounding terms."""
        with mp.workdps(self.digits):
            self.engine.set_displacement(displacement)
            return np.asarray([[self.engine.kernel(i,16+j) for j in range(len(self.k))]
                               for i in range(15)],complex)

    def sine_sine(self, displacement: np.ndarray) -> np.ndarray:
        d = self.k@displacement/2
        common = np.exp(-(float(displacement@displacement)+self.norms[:,None]+self.norms[None,:])/4)
        return common*(np.sinh(self.kk/2)*np.cosh(d)[:,None]*np.cosh(d)[None,:]
                       -np.cosh(self.kk/2)*np.sinh(d)[:,None]*np.sinh(d)[None,:])

    def triple(self) -> np.ndarray:
        """E[sin(k_i.y) sin(k_r.y) cos(k_j.y)], stable without exp differences."""
        a,b,c = self.kk[:,:,None]/2,self.kk[:,None,:]/2,self.kk[None,:,:]/2
        common = np.exp(-(self.norms[:,None,None]+self.norms[None,:,None]+self.norms[None,None,:])/4)
        return common*(np.sinh(a)*np.cosh(b)*np.cosh(c)-np.cosh(a)*np.sinh(b)*np.sinh(c))

    def _build(self) -> None:
        count = len(self.k)
        triple = self.triple()
        cross = np.zeros((97,97),complex)
        c_ss = np.einsum('irj,rj->ij',triple,self.uk,optimize=True)
        cross[16:,16:] = c_ss
        with mp.workdps(self.digits):
            self.engine.set_displacement(np.zeros(52))
            for index in self.odd_hard:
                for j in range(count):
                    cross[index,16+j] = self.engine.cross(index,16+j)
                    cross[16+j,index] = self.engine.cross(16+j,index)
        with mp.workdps(self.digits):
            self.engine.set_displacement(np.zeros(52))
            for i in range(15):
                for j in range(15):
                    if sum(self.labels[i])%2==sum(self.labels[j])%2:
                        cross[i,j] = self.engine.cross(i,j)
            for i in self.even_hard:
                cross[i,15] = self.engine.cross(i,15)
                cross[15,i] = self.engine.cross(15,i)
        # gamma is purely imaginary times a real wave; the SA diagonal is zero
        # algebraically, not a fitted eigenvalue or SOURCE commutator constraint.
        self.coexact = -.5j*(cross-cross.conj().T)
        sine = np.zeros((count,97,97),complex)
        with mp.workdps(self.digits):
            self.engine.set_displacement(np.zeros(52))
            for r in range(count):
                for i in range(15):
                    for j in range(15):
                        if (sum(self.labels[i])+sum(self.labels[j]))%2:
                            sine[r,i,j] = self.engine.sine_kernel(i,j,r)
                for i in self.even_hard:
                    for j in range(count):
                        value = self.engine.sine_kernel(i,16+j,r)
                        sine[r,i,16+j] = value
                        sine[r,16+j,i] = value.conjugate()
        # Actual gamma constants use native dot coefficients. Preserve their
        # tiny difference from exact full52 divergence; do not silently idealize.
        with mp.workdps(self.digits):
            residual = np.asarray([float(mp.re(self.engine.direction_phase(self.engine.flow_ids[r],r))
                                  -mp.mpf(float(self.u[r]@self.k[r]))) for r in range(count)])
            for r in range(count):
                for j in self.odd_hard:
                    sine[r,15,j] = self.engine.sine_kernel(15,j,r)
                    sine[r,j,15] = sine[r,15,j].conjugate()
        gamma_ss = -.5j*(c_ss+c_ss.T+np.einsum('s,irs->ir',residual,triple,optimize=True))
        sine[:,15,16:] = gamma_ss
        sine[:,16:,15] = gamma_ss.conj()
        self.harmonic = np.einsum('ra,rij->aij',self.bank.coefficients['harmonic'],sine,optimize=True)
        self.coulomb = np.einsum('rx,rij->xij',self.bank.coefficients['lambda_C'],sine,optimize=True)
        self.sine = sine
        self._controls()

    def _controls(self) -> None:
        gram = self.bank.scalar_gram()
        vacuum_error = float(np.linalg.norm(self.coexact[:,0]-gram[:,15]))
        self.controls.append(dict(name='derived_coexact_vacuum_gamma_column',error=vacuum_error,
                                  assigned_by_hand=False,scale=float(np.linalg.norm(gram[:,15]))))
        if vacuum_error>1e-9*max(float(np.linalg.norm(gram[:,15])),1e-30):
            raise ValueError('Derived exact coexact vacuum column disagrees with actual gamma functions')
        with mp.workdps(self.digits):
            self.engine.set_displacement(np.zeros(52))
            for left,right,phase in ((15,16,0),(15,23,7),(5,19,3),(0,16,0),(1,16,2)):
                exact = self.engine.sine_kernel(left,right,phase)
                observed = self.sine[phase,left,right]
                error = abs(exact-observed)
                scale = float(np.sqrt(gram[left,left].real*gram[right,right].real))
                self.controls.append(dict(name='literal_finite_wave_sine_kernel',left=left,right=right,phase=phase,
                    exact=[exact.real,exact.imag],observed=[observed.real,observed.imag],error=float(error),
                    diagonally_conditioned_error=float(error/scale)))
                if error>1e-10*scale:
                    raise ValueError('Stable sine formula fails literal high-precision finite-wave control')
            for left,right in ((16,17),(19,27)):
                a,b = self.engine.cross(left,right),self.engine.cross(right,left)
                exact = -.5j*(a-b.conjugate())
                observed = self.coexact[left,right]
                scale = float(np.sqrt(gram[left,left].real*gram[right,right].real))
                self.controls.append(dict(name='literal_coexact_sine_pair',left=left,right=right,
                    error=float(abs(exact-observed)),diagonally_conditioned_error=float(abs(exact-observed)/scale)))
                if abs(exact-observed)>1e-10*scale:
                    raise ValueError('Stable coexact formula fails literal polynomial/Weyl control')

    def weyl(self, displacement: np.ndarray) -> np.ndarray:
        key = np.asarray(displacement,float).tobytes()
        if key not in self.weyl_cache:
            out = np.zeros((97,97),complex)
            with mp.workdps(self.digits):
                self.engine.set_displacement(displacement)
                out[:15,:15] = np.asarray([[self.engine.kernel(i,j) for j in range(15)]
                                           for i in range(15)],complex)
            out[:15,16:] = self.hard_sine(displacement)
            out[16:,:15] = self.hard_sine(-displacement).conj().T
            out[16:,16:] = self.sine_sine(displacement)
            with mp.workdps(self.digits):
                self.engine.set_displacement(displacement)
                for j in range(97):
                    out[15,j] = self.engine.kernel(15,j)
                    out[j,15] = self.engine.kernel(j,15) if j!=15 else out[15,15]
            self.weyl_cache[key] = out
        return self.weyl_cache[key]

    def noether_kernels(self) -> dict[str,Any]:
        return dict(coexact=self.coexact,harmonic=self.harmonic,coulomb=self.coulomb,
                    all_coexact_coordinates=52,literal_sine=True,quadrature_nodes=0,
                    integration='analytic all52 polynomial/Weyl Gaussian moments; mp gamma60, stable sinh sin blocks',
                    generated_G_quadrature_debt=False,arithmetic_digits=self.digits,
                    vacuum_column_derived_not_assigned=True,controls=self.controls)
