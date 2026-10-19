#!/usr/bin/env python3
"""Complete Alpha0226 source Krylov actions; no spectral/source gate.

Projected matrix functions below are approximations Q f(Qdag A Q) Qdag,
not log of the true operator. Every solve is checked with the COMPLETE
midpoint callback. Raw range fractions use the variational projector identity
2 Re<m,x>-<x,Ax>, not the half-source norm as a raw-source denominator.
No original-field polar connection, moving-range convention or source
parameter derivatives are inferred from these midpoint function actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, cast

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator, cg
import mpmath as mp
import physical_alpha0226_midpoint_operator as core

IMPORT_EPOCH = time.time()
SOURCE_BYTES = Path(__file__).read_bytes()
CORE_BYTES = Path(core.__file__).read_bytes()


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def gram_json(v: np.ndarray) -> dict[str, Any]:
    return dict(real=np.real(v).tolist(), imaginary=np.imag(v).tolist())


def tensor_product(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                   v: np.ndarray) -> np.ndarray:
    """Fullcanonical tensor action with one packed RHS axis, without RHS loops."""
    tail = v.shape[3:]
    x = (a @ v.reshape(v.shape[0], -1)).reshape((a.shape[0], b.shape[1], c.shape[1]) + tail)
    order_b = (1, 0, 2) + tuple(range(3, x.ndim))
    x = (b @ x.transpose(order_b).reshape(b.shape[1], -1)).reshape((b.shape[0], a.shape[0], c.shape[1]) + tail).transpose(order_b)
    order_c = (2, 0, 1) + tuple(range(3, x.ndim))
    x = (c @ x.transpose(order_c).reshape(c.shape[1], -1)).reshape((c.shape[0], a.shape[0], b.shape[0]) + tail)
    return x.transpose((1, 2, 0) + tuple(range(3, x.ndim)))


class CompleteSFAction:
    """Exact P sum w (SF_a² tensor SF_u² tensor SF_u²) P action.

    The fixed complete Gauss projector acts only at both ends, exactly as in
    core.cross_gram. Squaring each small fullcanonical factor halves the
    contraction count without storing a field-node vector or large matrix.
    The authoritative residual continues to use core.apply.
    """
    def __init__(self, model: core.Alpha0226Midpoint):
        self.model = model
        self.squares = [tuple(p['SF'].conj().T @ p['SF'] for p in parts)
                        for parts in model.factors]
        self.diagonal = sum((w * (a.diagonal().real[:, None, None]
                                 * b.diagonal().real[None, :, None]
                                 * c.diagonal().real[None, None, :])
                             for (a, b, c), w in zip(self.squares, model.weights, strict=True)), np.zeros(model.shape))[model.pc]
        if np.any(self.diagonal <= 0):
            raise ArithmeticError('Nonpositive exact Jacobi diagonal; no floor is allowed')

    def apply(self, v: np.ndarray) -> np.ndarray:
        full = self.model.embed(v)
        result = np.zeros_like(full)
        for (a, b, c), w in zip(self.squares, self.model.weights, strict=True):
            result += w * tensor_product(a, b, c, full)
        return self.model.pack(result)

    def jacobi(self, v: np.ndarray) -> np.ndarray:
        return v / (self.diagonal if v.ndim == 1 else self.diagonal[:, None])

    def tensor_preconditioner(self) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
        """P (weighted factorwise mean Gram)^-1 P; strictly positive, no floor.

        This is a numerical preconditioner only, never an adopted operator.
        Full SF factors, not occupied factors, determine all three means.
        """
        inverses = []
        ranges = []
        for k in range(3):
            mean = cast(np.ndarray, sum((w * parts[k] for parts, w in zip(self.squares, self.model.weights, strict=True)), np.zeros_like(self.squares[0][k])))
            values, vectors = np.linalg.eigh((mean + mean.conj().T) / 2)
            ranges.append([float(values[0]), float(values[-1])])
            if values[0] <= 0:
                raise ArithmeticError(f'Full SF mean {k} has unresolved positivity; no spectral floor')
            inverses.append((vectors / values) @ vectors.conj().T)
        def apply(v: np.ndarray) -> np.ndarray:
            return self.model.pack(tensor_product(inverses[0], inverses[1], inverses[2], self.model.embed(v)))
        return apply, dict(factor_mean_eigenvalue_ranges=ranges,
                           convention='P inverse(tensor of weighted full SF Gram means) P; no floor')

    def spectral_bounds(self) -> dict[str, Any]:
        """Numerical small-factor extremal bounds; NOT interval certificates.

        In exact arithmetic alpha=sum w product lambda_min(SFdag SF)>0.
        Restriction by P cannot reduce this lower bound. Double eigensolvers
        cannot certify the sign/relative accuracy of an unresolved minimum.
        """
        lower = 0.
        upper = 0.
        resolved = True
        smallest_relative = 1.
        for parts, w in zip(self.squares, self.model.weights, strict=True):
            ranges = [np.linalg.eigvalsh((p + p.conj().T) / 2)[[0, -1]] for p in parts]
            resolved = resolved and all(v[0] > 100 * np.finfo(float).eps * v[-1] for v in ranges)
            smallest_relative = min(smallest_relative, *(float(v[0] / v[-1]) for v in ranges))
            lower += w * float(np.prod([v[0] for v in ranges]))
            upper += w * float(np.prod([v[-1] for v in ranges]))
        return dict(numerical_factor_lower_bound=lower if resolved else None,
                    numerical_factor_upper_bound=upper, factor_minimum_relative=smallest_relative,
                    positivity_resolved_double=bool(resolved), uniform_interval_certificate=False,
                    theorem='A >= sum_i w_i product_k lambda_min(SF_ikdag SF_ik) I on COMPLETE Gauss; numerical values not directed-rounded')


def represented_gram_lower_bound(model: core.Alpha0226Midpoint) -> dict[str, Any]:
    """Outward interval lower bound for exact SF/weight BYTE-defined Vdag V.

    Any approximate inverse Y gives sigma_min(S)>= (1-||I-YS||)/||Y||.
    Interval 1/infinity norms bound both spectral norms. ONE positive node
    suffices, since every other Haar-node Gram is positive semidefinite.
    This does NOT certify transcendental SF generation, floating callback
    contraction error, or the continuous-Haar operator.
    """
    # mpmath interval stubs omit arithmetic protocols and float constructors.
    iv = cast(Any, mp.iv)
    iv.dps = 40
    def upper(x: Any) -> float:
        return float(np.nextafter(float(x.b), np.inf))
    def interval(z: complex) -> Any:
        return iv.mpc(float(np.real(z)), float(np.imag(z)))
    cache: dict[bytes, dict[str, Any]] = {}
    receipts = []
    alpha = iv.mpf(float(model.weights[0]))
    for part in model.factors[0]:
        s = part['SF']
        key = s.tobytes()
        if key not in cache:
            y = np.linalg.inv(s)
            n = s.shape[0]
            si = [[interval(z) for z in row] for row in s]
            yi = [[interval(z) for z in row] for row in y]
            rows = [iv.mpf(0) for _ in range(n)]
            columns = [iv.mpf(0) for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    e = iv.mpc(int(i == j)) - sum((yi[i][k] * si[k][j] for k in range(n)), iv.mpc(0))
                    rows[i] += abs(e)
                    columns[j] += abs(e)
            er = max(upper(v) for v in rows)
            ec = max(upper(v) for v in columns)
            eb = iv.sqrt(iv.mpf(er) * iv.mpf(ec))
            yr = max(upper(sum((abs(z) for z in row), iv.mpf(0))) for row in yi)
            yc = max(upper(sum((abs(yi[i][j]) for i in range(n)), iv.mpf(0))) for j in range(n))
            yb = iv.sqrt(iv.mpf(yr) * iv.mpf(yc))
            lower = float(np.nextafter(float(((1 - eb) / yb).a), -np.inf))
            if lower <= 0:
                raise ArithmeticError('Interval inverse-defect proof does not resolve SF injectivity')
            cache[key] = dict(SF_sha256=hashlib.sha256(key).hexdigest(),
                              inverse_residual_spectral_upper=upper(eb),
                              inverse_norm_spectral_upper=upper(yb), singular_min_lower=lower)
        receipt = cache[key]
        receipts.append(receipt)
        alpha *= iv.mpf(receipt['singular_min_lower']) ** 2
    return dict(exact_byte_Gram_alpha_lower=float(np.nextafter(float(alpha.a), -np.inf)),
                node=0, weight=float(model.weights[0]), interval_decimal_precision=40,
                factor_receipts=receipts,
                scope='Certified BYTE-defined VdagV lower bound only; original mathematical SF/transcendentals, callback residual rounding and continuous Haar NOT certified')


class BlockKrylov:
    """Fully reorthogonalized complex block Arnoldi for Hermitian A.

    No physical mode is removed. SVD rank decisions concern only floating
    point linear dependence in a Krylov block; discarded block residuals are
    recorded. The source is neither truncated nor whitened into a new state.
    """
    def __init__(self, apply: Callable[[np.ndarray], np.ndarray], source: np.ndarray):
        self.apply = apply
        self.source = source
        self.q = np.empty((source.shape[0], 0), complex)
        self.aq = np.empty_like(self.q)
        self.frontier = source
        self.level = 0
        self.rank_receipts: list[dict[str, Any]] = []

    def extend(self, levels: int) -> None:
        while self.level < levels and self.frontier.shape[1]:
            remainder = self.frontier.copy()
            for _ in range(2):
                remainder -= self.q @ (self.q.conj().T @ remainder)
            u, s, _ = np.linalg.svd(remainder, full_matrices=False)
            threshold = 100 * np.finfo(float).eps * max(norm(self.frontier), np.finfo(float).tiny)
            rank = int(np.count_nonzero(s > threshold))
            self.rank_receipts.append(dict(level=self.level + 1, singular_values=s.tolist(),
                                           numerical_dependence_threshold=threshold, retained_rank=rank,
                                           discarded_residual_norm=norm(s[rank:])))
            if not rank:
                break
            block = u[:, :rank]
            ablock = self.apply(block)
            self.q = np.column_stack((self.q, block))
            self.aq = np.column_stack((self.aq, ablock))
            self.frontier = ablock
            self.level += 1

    def projection(self) -> tuple[np.ndarray, dict[str, Any]]:
        h = self.q.conj().T @ self.aq
        defect = norm(h - h.conj().T) / max(norm(h), np.finfo(float).tiny)
        h = (h + h.conj().T) / 2
        return h, dict(level=self.level, dimension=h.shape[0],
                       projected_Hermitian_relative_defect=defect,
                       basis_orthogonality_defect=norm(self.q.conj().T @ self.q - np.eye(h.shape[0])),
                       Arnoldi_FULLoperator_remainder=norm(self.aq - self.q @ h),
                       approximation='Q f(Qdag A Q) Qdag source; NOT exact full log(A)')

    def function_action(self, kind: str, rhs: np.ndarray | None = None) -> np.ndarray:
        h, _ = self.projection()
        values, vectors = np.linalg.eigh(h)
        if values[0] <= 0:
            raise ArithmeticError('Nonpositive Ritz value; positive function unresolved, no floor')
        if kind == 'inverse':
            f = 1 / values
        elif kind == 'log':
            f = np.log(values)
        elif kind == 'sqrt':
            f = np.sqrt(values)
        elif kind == 'inverse_sqrt':
            f = 1 / np.sqrt(values)
        else:
            raise ValueError('Unknown operator function')
        b = self.source if rhs is None else rhs
        coefficients = vectors.conj().T @ (self.q.conj().T @ b)
        if coefficients.ndim == 1:
            coefficients *= f
        else:
            coefficients *= f[:, None]
        return self.q @ (vectors @ coefficients)

    def resolvent(self, shift: float, rhs: np.ndarray | None = None) -> np.ndarray:
        if shift < 0:
            raise ValueError('Positive-range resolvent requires nonnegative shift')
        h, _ = self.projection()
        b = self.source if rhs is None else rhs
        return self.q @ np.linalg.solve(h + shift * np.eye(h.shape[0]), self.q.conj().T @ b)


def full_residual(model: core.Alpha0226Midpoint, x: np.ndarray, rhs: np.ndarray,
                  shift: float = 0.) -> dict[str, Any]:
    r = rhs - model.apply(x) - shift * x
    rn = norm(r)
    return dict(absolute=rn, relative=rn / max(norm(rhs), np.finfo(float).tiny),
                per_rhs_relative=(np.linalg.norm(r, axis=0) / np.linalg.norm(rhs, axis=0)).tolist() if r.ndim == 2 else None,
                residual=r, shifted_action_error_upper=rn / shift if shift > 0 else None,
                metric='COMPLETE44336 original core.apply, not reduced matrix residual')


def inverse_solve(model: core.Alpha0226Midpoint, action: CompleteSFAction,
                  rhs: np.ndarray, *, shift: float = 0., maxiter: int = 128,
                  rtol: float = 1e-12, preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
                  initial: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Independent complex PCG columns with authoritative true residuals."""
    b = rhs[:, None] if rhs.ndim == 1 else rhs
    operator_factory = cast(Any, LinearOperator)
    a = operator_factory((model.dimension, model.dimension), matvec=lambda v: action.apply(v) + shift * v, dtype=np.complex128)
    p = None if preconditioner is None else operator_factory(a.shape, matvec=preconditioner, dtype=np.complex128)
    results = []
    infos = []
    for j in range(b.shape[1]):
        iterations = [0]
        def count(_: np.ndarray) -> None:
            iterations[0] += 1
        start = None if initial is None else (initial if initial.ndim == 1 else initial[:, j])
        x, info = cg(a, b[:, j], x0=start, M=p, maxiter=maxiter, rtol=rtol, atol=0., callback=count)
        results.append(x)
        infos.append(dict(rhs=j, cg_info=int(info), iterations=iterations[0]))
    result = np.column_stack(results)
    if rhs.ndim == 1:
        result = result[:, 0]
    residual = full_residual(model, result, rhs, shift)
    residual.pop('residual')
    return result, dict(columns=infos, maxiter=maxiter, rtol=rtol, shift=shift, full_residual=residual)


def range_price(model: core.Alpha0226Midpoint, x: np.ndarray,
                lower_bound: float | None = None) -> dict[str, Any]:
    """Raw field projector variational price, unchanged all4Ref source.

    For m=Vdag raw, q(x)=2 Re<m,x>-<x,Ax> <= <m,A^-1m>.
    raw-q=||raw-Vx||². If A>=alpha I, missed range weight <=||r||²/alpha.
    No zero-kernel log/energy extension is introduced.
    """
    m = model.source_mid
    ax = model.apply(x)
    mx = m.conj().T @ x
    xa = x.conj().T @ ax
    q = float(2 * np.trace(mx).real - np.trace(xa).real)
    r2 = norm(m - ax) ** 2
    error = r2 / lower_bound if lower_bound is not None and lower_bound > 0 else None
    return dict(raw_CAR_denominator=model.raw_Gram,
                inverse_pairing_numeric=float(np.trace(mx).real),
                inverse_pairing_imaginary=float(np.trace(mx).imag),
                inverse_Gram=gram_json(mx), variational_range_Gram=gram_json(mx + mx.conj().T - xa),
                positive_range_RAW_fraction_variational_lower=q / model.raw_Gram,
                positive_range_RAW_fraction_upper=min(1., (q + error) / model.raw_Gram) if error is not None else 1.,
                RAW_range_complement_variational_upper=(model.raw_Gram - q) / model.raw_Gram,
                RAW_range_complement_lower=max(0., (model.raw_Gram - q - error) / model.raw_Gram) if error is not None else 0.,
                missed_range_weight_numeric_upper=error,
                residual_squared=r2, lower_bound_used=lower_bound,
                claim='Variational lower range weight and upper complement in exact arithmetic; double arithmetic, not interval certification. Inverse pairing is approximate until priced. raw-q includes solver error, not just true kernel complement.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--order', type=int, default=64)
    parser.add_argument('--levels', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--iterations', type=int, nargs='+', default=[32, 96])
    parser.add_argument('--shift', type=float, default=.01)
    parser.add_argument('--rtol', type=float, default=1e-12)
    args = parser.parse_args()
    if len(set(args.levels)) < 2:
        parser.error('At least two distinct approximation orders are required')
    started = time.time()
    model = core.Alpha0226Midpoint(order=args.order, jets=False)
    action = CompleteSFAction(model)
    source = model.source_mid
    source_action = model.apply(source)
    callback_equivalence = norm(action.apply(source) - source_action) / norm(source_action)
    bounds = action.spectral_bounds()
    represented_lower = represented_gram_lower_bound(model)
    lower = represented_lower['exact_byte_Gram_alpha_lower']
    report: dict[str, Any] = dict(descriptor=model.descriptor(), callback_equivalence_relative=callback_equivalence,
                                spectral_bounds=bounds, Krylov=[], inverse_repairs=[])
    report['represented_Gram_lower_bound'] = represented_lower
    report['correction_to_initial_capsule'] = 'Initial squared projected sqrt consistency is algebraically satisfied when A source lies in Q; it is NOT sqrt action convergence evidence and is withdrawn. Use approximation-order changes; full log/sqrt certification remains open.'
    krylov = BlockKrylov(action.apply, source)
    previous: dict[str, np.ndarray] = {}
    for level in sorted(set(args.levels)):
        krylov.extend(level)
        h, diagnostics = krylov.projection()
        eigenvalues = np.linalg.eigvalsh(h)
        row: dict[str, Any] = dict(diagnostics=diagnostics, Ritz_range=eigenvalues[[0, -1]].tolist(),
                                  Ritz_condition=float(eigenvalues[-1] / eigenvalues[0]), functions={})
        for kind in ('inverse', 'sqrt', 'inverse_sqrt', 'log'):
            x = krylov.function_action(kind)
            entry: dict[str, Any] = dict(action_norm=norm(x), source_pairing=gram_json(source.conj().T @ x),
                                        relative_change_from_previous=norm(x - previous[kind]) / max(norm(x), np.finfo(float).tiny) if kind in previous else None)
            if kind == 'inverse':
                residual = full_residual(model, x, source)
                residual.pop('residual')
                entry.update(full_residual=residual, range_price=range_price(model, x, lower))
            elif kind in ('sqrt', 'inverse_sqrt'):
                entry['warning'] = 'Approximation-order change, not a true sqrt/inverse-sqrt equation residual or uniform certificate'
                if kind == 'inverse_sqrt':
                    entry['polar_source_Gram_numeric'] = gram_json(x.conj().T @ x)
            else:
                entry['FULLoperator_commutation_defect_relative'] = norm(model.apply(x) - krylov.function_action('log', source_action)) / max(norm(model.apply(x)), np.finfo(float).tiny)
                entry['warning'] = 'Commutation/change diagnostics are not a uniform log action certificate'
            row['functions'][kind] = entry
            previous[kind] = x
        shifted = krylov.resolvent(args.shift)
        sr = full_residual(model, shifted, source, args.shift)
        sr.pop('residual')
        row['shifted_resolvent'] = dict(shift=args.shift, full_residual=sr,
                                       source_pairing=gram_json(source.conj().T @ shifted))
        report['Krylov'].append(row)
        print(json.dumps(dict(phase='Krylov', level=level, inverse_residual=row['functions']['inverse']['full_residual']['relative'], range_lower=row['functions']['inverse']['range_price']['positive_range_RAW_fraction_variational_lower'], shift_residual=sr['relative'])), flush=True)
    report['Krylov_rank_receipts'] = krylov.rank_receipts
    # Material numerical repair: exact Jacobi followed by fullcanonical
    # tensor mean inverse. A failed repair is retained, never floored.
    preconditioners: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [('exact_SF_Jacobi', action.jacobi)]
    try:
        tensor, metadata = action.tensor_preconditioner()
        report['tensor_preconditioner'] = metadata
        preconditioners.append(('full_SF_tensor_mean_inverse', tensor))
    except ArithmeticError as exc:
        report['tensor_preconditioner_failure'] = str(exc)
    best = previous['inverse']
    best_residual = full_residual(model, best, source)['relative']
    for name, preconditioner in preconditioners:
        for iterations in args.iterations:
            x, receipt = inverse_solve(model, action, source, maxiter=iterations,
                                       preconditioner=preconditioner, initial=best, rtol=args.rtol)
            receipt.update(preconditioner=name, range_price=range_price(model, x, lower))
            report['inverse_repairs'].append(receipt)
            if receipt['full_residual']['relative'] < best_residual:
                best = x
                best_residual = receipt['full_residual']['relative']
            print(json.dumps(dict(phase='inverse_repair', preconditioner=name, iterations=iterations,
                                  residual=receipt['full_residual']['relative'], range_lower=receipt['range_price']['positive_range_RAW_fraction_variational_lower'])), flush=True)
            if best_residual < args.rtol:
                break
    shifted_diagonal = action.diagonal + args.shift
    shifted, shifted_receipt = inverse_solve(model, action, source, shift=args.shift,
                                             maxiter=max(args.iterations), rtol=args.rtol,
                                             preconditioner=lambda v: v / shifted_diagonal,
                                             initial=krylov.resolvent(args.shift))
    shifted_receipt['preconditioner'] = 'Exact SF diagonal PLUS physical resolvent shift; no floor'
    report['shifted_solve'] = shifted_receipt
    report['best_inverse_range_price'] = range_price(model, best, lower)
    vector_path = str(Path(args.output).with_suffix('.npz'))
    projected, _ = krylov.projection()
    np.savez(vector_path, basis=krylov.q, A_basis=krylov.aq, projected=projected,
             source=source, inverse=best, shifted_resolvent=shifted,
             log_action=previous['log'], sqrt_action=previous['sqrt'],
             inverse_sqrt_action=previous['inverse_sqrt'],
             source_sha256=hashlib.sha256(SOURCE_BYTES).hexdigest(),
             midpoint_sha256=hashlib.sha256(CORE_BYTES).hexdigest())
    report['vectors'] = dict(path=vector_path, sha256=hashlib.sha256(Path(vector_path).read_bytes()).hexdigest(),
                            basis_shape=list(krylov.q.shape),
                            scope='Actual complex COMPLETE midpoint basis/source and approximate function actions, no full field-node allocation')
    report['honest_scope'] = dict(quadrature='EXACT_M64_frozen_source' if args.order == 64 else 'LOWER_OR_OTHER_ORDER_QUADRATURE_SMOKE_NOT_ALPHA0226_PROOF',
                                 numerical_inverse_converged=best_residual < args.rtol,
                                 uniform_function_certification=False,
                                 physical_logK_current=False,
                                 source_derivatives_computed=False,
                                 contact='No kernel log/energy extension or moving positive-range transport convention chosen',
                                 log_mixed_requirement='Integral R Axy R - R Ax R Ay R - R Ay R Ax R, both orders; original field polarconnection still required',
                                 unpriced='Continuous Haar, uniform lower spectral certification, original-field polarconnection/source derivatives, physical current/contact and Compton remain open')
    inputs = dict(argv=sys.argv, descriptor=model.descriptor(), levels=args.levels, iterations=args.iterations, shift=args.shift, rtol=args.rtol)
    input_bytes = json.dumps(inputs, sort_keys=True).encode()
    report['execution'] = dict(import_epoch=IMPORT_EPOCH, source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),
                               source_bytes_at_IMPORT=SOURCE_BYTES.decode(), midpoint_at_IMPORT=hashlib.sha256(CORE_BYTES).hexdigest(),
                               midpoint_bytes_at_IMPORT=CORE_BYTES.decode(), field_at_IMPORT=hashlib.sha256(core.FIELD_BYTES).hexdigest(),
                               consumer_at_IMPORT=hashlib.sha256(core.primitive.CONSUMER_BYTES).hexdigest(),
                               input_json=inputs, input_sha256=hashlib.sha256(input_bytes).hexdigest(),
                               python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__, mpmath=mp.__version__, runtime=platform.platform(),
                               startup_unchanged=Path(__file__).read_bytes() == SOURCE_BYTES and Path(core.__file__).read_bytes() == CORE_BYTES,
                               elapsed_seconds=time.time() - started)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=args.output, inverse_residual=best_residual,
                          range_price=report['best_inverse_range_price'], shifted_residual=shifted_receipt['full_residual']['relative'], elapsed_seconds=time.time() - started)), flush=True)


if __name__ == '__main__':
    main()
