#!/usr/bin/env python3
"""Exact Haar projection of four-field SU(4) alignment, not a product of PUP.

The CLI constructs and checks link multiplication only; it runs no charged
Hamiltonian, and finite alignment eigenvalues are not an infinite Higgs gap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import eigh, expm

from quantum_su4_gauss import (CUTOFFS, HaarLink, RotationBand, matrix_unit_basis,
                               norm, require_small, source_hashes, uniform_data)


CG_TOLERANCE = 2e-9
# Complete multiplicities among the seven canonical representations. Other
# irreps of each tensor product are deliberately not link cutoff states.
EXPECTED_MULTIPLICITIES = {
    ('15', '1'): 1, ('1', '15'): 1, ('15', '15'): 2,
    ('4', '4'): 1, ('bar4', 'bar4'): 1, ('6', '6'): 1,
    ('10', '6'): 1, ('bar10', '6'): 1,
    ('6', '10'): 1, ('10', '10'): 1,
    ('6', 'bar10'): 1, ('bar10', 'bar10'): 1,
}


def _spectral_groups(values, tolerance=2e-8):
    groups = []
    for value in values:
        value = float(value)
        if groups and abs(value - groups[-1]['eigenvalue']) <= tolerance:
            groups[-1]['multiplicity'] += 1
            groups[-1]['maximum'] = value
        else:
            groups.append(dict(eigenvalue=value, multiplicity=1, maximum=value))
    return groups


def _coefficients(link, frame):
    fields = np.asarray(frame, dtype=np.complex128)
    if fields.shape != (4, 4, 4) or not np.all(np.isfinite(fields)):
        raise ValueError('frame must be four finite traceless Hermitian 4x4 fields')
    for mu, field in enumerate(fields):
        scale = max(1., float(np.max(np.abs(field))))
        if np.max(np.abs(field - field.conj().T)) > 2e-11 * scale:
            raise ValueError(f'frame[{mu}] is not Hermitian')
        if abs(np.trace(field)) > 2e-11 * scale:
            raise ValueError(f'frame[{mu}] is not traceless')
    coordinates = 2 * np.einsum('aij,mji->ma', link.T, fields)
    if np.max(np.abs(coordinates.imag)) > 2e-10 * max(1., float(np.max(np.abs(coordinates)))):
        raise ValueError('Hermitian frame has non-real adjoint coefficients')
    p = coordinates.real
    with np.errstate(over='ignore', invalid='ignore'):
        constant = float(np.sum(p * p))
    if not np.isfinite(constant):
        raise ValueError('quadratic field norm must be finite in float64 arithmetic')
    return fields.copy(), p, constant


class _AdjointIntertwiners:
    """Casimir-restricted complete Hom kernels, including multiplicities."""

    def __init__(self, link):
        self.link = link
        self.products: dict[str, Any] = {}
        self.embeddings: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}
        self.records: dict[tuple[str, str], dict[str, Any]] = {}

    def _product(self, source):
        if source not in self.products:
            rep = self.link.reps[source]
            ds = rep.shape[1]
            generators = np.array([np.kron(adj, np.eye(ds)) + np.kron(np.eye(15), t)
                                   for adj, t in zip(self.link.reps['15'], rep)])
            casimir = sum(t @ t for t in generators)
            values, vectors = eigh(casimir, check_finite=False)
            self.products[source] = (generators, values, vectors)
        return self.products[source]

    def get(self, target, source):
        key = (target, source)
        if key in self.embeddings:
            return self.embeddings[key]
        dr, ds = self.link.reps[target].shape[1], self.link.reps[source].shape[1]
        record: dict[str, Any] = dict(target=target, source=source,
            target_dimension=dr, source_dimension=ds, multiplicity=0,
            centre_allowed=self.link.centre[target] == self.link.centre[source])
        self.records[key] = record
        if not record['centre_allowed']:
            self.embeddings[key] = ()
            return ()
        generators, values, vectors = self._product(source)
        cr = self.link.casimir[target]
        selected = np.abs(values - cr) <= CG_TOLERANCE * max(1., abs(cr))
        q = vectors[:, selected]
        restricted_dimension = q.shape[1]
        record.update(target_casimir=cr, casimir_eigenspace_dimension=restricted_dimension,
                      hom_dimension=restricted_dimension * dr)
        if not restricted_dimension:
            self.embeddings[key] = ()
            return ()
        product = np.array([q.conj().T @ t @ q for t in generators])
        # vec_F(Tprod C - C Tr) = (I_r tensor Tprod - Tr^T tensor I) vec_F(C).
        laplacian = np.zeros((restricted_dimension * dr,) * 2, complex)
        for tp, tr in zip(product, self.link.reps[target]):
            difference = np.kron(np.eye(dr), tp) - np.kron(tr.T, np.eye(restricted_dimension))
            laplacian += difference.conj().T @ difference
        hom_values, hom_vectors = eigh(laplacian, check_finite=False)
        threshold = CG_TOLERANCE * max(1., float(hom_values[-1]))
        zero = np.abs(hom_values) <= threshold
        embeddings = tuple((np.sqrt(dr) * q @ vector.reshape(restricted_dimension, dr, order='F'))
                           .reshape(15, ds, dr) for vector in hom_vectors[:, zero].T)
        residual = isometry = mutual = 0.
        for t, embedding in enumerate(embeddings):
            c = embedding.reshape(15 * ds, dr)
            isometry = max(isometry, norm(c.conj().T @ c - np.eye(dr)))
            for a, generator in enumerate(generators):
                residual = max(residual, norm(generator @ c - c @ self.link.reps[target][a]))
            for other in embeddings[:t]:
                mutual = max(mutual, norm(c.conj().T @ other.reshape(15 * ds, dr)))
        for value, label in ((residual, 'adjoint CG intertwining'), (isometry, 'adjoint CG isometry'),
                             (mutual, 'adjoint CG multiplicity orthogonality')):
            require_small(value, f'{target} <- 15 tensor {source}: {label}', 2e-7)
        record.update(multiplicity=len(embeddings), hom_zero_threshold=threshold,
            hom_lowest_eigenvalues=hom_values[:min(8, len(hom_values))].tolist(),
            hom_first_nonzero=float(hom_values[~zero][0]) if np.any(~zero) else None,
            intertwiner_residual=residual, isometry_residual=isometry,
            mutual_orthogonality_residual=mutual)
        self.embeddings[key] = embeddings
        return embeddings


def _weighted_block(embeddings, p, dr, ds, basis_change=None):
    result = np.zeros((dr, dr, ds, ds), complex)
    for c in embeddings:
        if basis_change is not None:
            c = np.einsum('ij,jca->ica', basis_change, c, optimize=True)
        # K_ij=sum_mu p_mu,i p_mu,j has rank at most four. Contract it
        # before forming a PW block: no family of 225 large MAdj matrices.
        weighted = np.einsum('mi,ica->mca', p, c, optimize=True)
        result += np.einsum('mca,mdb->abcd', weighted, weighted.conj(), optimize=True)
    return np.sqrt(ds / dr) * result.reshape(dr * dr, ds * ds)


def _assemble(link, blocks):
    rows, cols, data = [], [], []
    for (target, source), block in blocks.items():
        coo = block.tocoo()
        rows.append(coo.row + link.offset[target])
        cols.append(coo.col + link.offset[source])
        data.append(coo.data)
    if not data:
        return sparse.csr_matrix((link.dimension, link.dimension), dtype=complex)
    return sparse.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(link.dimension, link.dimension)).tocsr()


class ExactHaarAlignment:
    """P W(U) P in the supplied HaarLink.offset Peter-Weyl coefficient order.

    `frame` has shape (4,4,4), with arbitrary finite traceless Hermitian fields;
    zero, dependent and non-Clifford fields are allowed. `blocks[(r,s)]` is a
    CSR (d_r^2,d_s^2) block of W, not of U or D15. No coupling J is included.
    Evidence/spectra and independent product controls run only on evidence().
    """

    def __init__(self, link, frame):
        self.link = link
        self.frame, self.p, self.constant = _coefficients(link, frame)
        self.K = self.p.T @ self.p
        self._cg = _AdjointIntertwiners(link)
        self.matrix, self.blocks = self._project(self.p)

    def _project(self, p, basis_change=None):
        constant = float(np.sum(p * p))
        blocks = {}
        for target in self.link.names:
            dr = self.link.reps[target].shape[1]
            for source in self.link.names:
                ds = self.link.reps[source].shape[1]
                embeddings = self._cg.get(target, source)
                if not embeddings and target != source:
                    continue
                block = -_weighted_block(embeddings, p, dr, ds, basis_change)
                if target == source:
                    block += constant * np.eye(dr * dr)
                blocks[target, source] = sparse.csr_matrix(block)
        return _assemble(self.link, blocks), blocks

    def _known_controls(self):
        identity = np.eye(15)
        c151 = self._cg.get('15', '1')
        c115 = self._cg.get('1', '15')
        c1515 = self._cg.get('15', '15')
        if (len(c151), len(c115), len(c1515)) != (1, 1, 2):
            raise AssertionError('Known positive Hom controls require multiplicities 1,1,2')
        def span_residual(embeddings, known):
            dr = embeddings[0].shape[-1]
            vectors = np.column_stack([c.ravel() / np.sqrt(dr) for c in embeddings])
            return norm(known.ravel() - vectors @ (vectors.conj().T @ known.ravel()))
        singlet = identity.reshape(15, 15, 1) / np.sqrt(15)
        d = np.array([[2 * np.einsum('aij,ji->a', self.link.T, ti @ tj + tj @ ti)
                       for tj in self.link.T] for ti in self.link.T])
        cf, cd = self.link.f / 2, d / np.sqrt(3)
        residuals = dict(adjoint_tensor_1_to_15=span_residual(c151, identity.reshape(15, 1, 15)),
            adjoint_tensor_15_to_1=span_residual(c115, singlet),
            antisymmetric_15_span=span_residual(c1515, cf),
            symmetric_15_span=span_residual(c1515, cd),
            independent_15_isometries=norm(cf.reshape(225, 15).conj().T @ cf.reshape(225, 15) - identity),
            independent_symmetric_isometry=norm(cd.reshape(225, 15).conj().T @ cd.reshape(225, 15) - identity),
            independent_15_orthogonality=norm(cf.reshape(225, 15).conj().T @ cd.reshape(225, 15)))
        for name, value in residuals.items():
            require_small(value, name, 2e-7)
        return dict(multiplicities={'15<-15x1':1, '1<-15x15':1, '15<-15x15':2}, **residuals)

    def _representation_controls(self):
        link = self.link
        h = np.linspace(-.173, .219, 15)
        g = expm(1j * np.einsum('a,aij->ij', h, link.T))
        representation = {name: expm(1j * np.einsum('a,aij->ij', h, rep))
                          for name, rep in link.reps.items()}
        adjoint = 2 * np.einsum('aij,bji->ab', link.T,
                               np.array([g @ t @ g.conj().T for t in link.T]))
        canonical = norm(2 * np.einsum('aij,bji->ab', link.T, link.T) - np.eye(15))
        adjoint_exponential = norm(adjoint - representation['15'])
        finite_group = 0.
        for (r, s), embeddings in self._cg.embeddings.items():
            for embedding in embeddings:
                c = embedding.reshape(15 * link.reps[s].shape[1], link.reps[r].shape[1])
                finite_group = max(finite_group,
                    norm(np.kron(adjoint, representation[s]) @ c - c @ representation[r]))
        centre_log = np.pi / 2 * np.diag([1., 1., 1., -3.])
        centre_p = 2 * np.einsum('aij,ji->a', link.T, centre_log)
        centre = max(norm(expm(1j * np.einsum('a,aij->ij', centre_p, rep)) -
                          1j ** link.centre[name] * np.eye(rep.shape[1]))
                     for name, rep in link.reps.items())
        rotated = np.array([g @ field @ g.conj().T for field in self.frame])
        _, rotated_p, _ = _coefficients(link, rotated)
        rotated_matrix, _ = self._project(rotated_p)
        # V_g f(U)=f(g U g^dagger); W[g Phi g^dagger]=V_g^dagger W[Phi] V_g.
        conjugation = sparse.block_diag([sparse.csr_matrix(
            np.kron(representation[n].T, representation[n].conj().T))
            for n in link.names], format='csr')
        scalar_covariance = norm(rotated_matrix - conjugation.conj().T @ self.matrix @ conjugation)
        rng = np.random.default_rng(222)
        orthogonal, _ = np.linalg.qr(rng.normal(size=(15, 15)))
        basis_matrix, _ = self._project(self.p @ orthogonal.T, orthogonal)
        basis_covariance = norm(basis_matrix - self.matrix)
        scalar = float(np.sum(np.abs(self.frame - np.array([g @ field @ g.conj().T
                                                           for field in self.frame])) ** 2))
        expansion = self.constant - float(np.einsum('ij,ij->', self.K, adjoint).real)
        scalar_expansion = abs(scalar - expansion)
        scale = max(1., self.constant)
        for value, name in ((canonical, 'canonical trace normalization'),
                            (adjoint_exponential, 'full adjoint exponential'),
                            (finite_group, 'full representation CG covariance'),
                            (centre, 'representation centre'),
                            (scalar_covariance / scale, 'scalar conjugation covariance'),
                            (basis_covariance / scale, 'adjoint canonical basis covariance'),
                            (scalar_expansion / scale, 'continuous scalar expansion')):
            require_small(value, name, 3e-7)
        return dict(canonical_trace_residual=canonical, adjoint_exponential_residual=adjoint_exponential,
            finite_group_cg_residual=finite_group, centre_exponential_residual=centre,
            scalar_conjugation_covariance_residual=scalar_covariance,
            canonical_basis_covariance_residual=basis_covariance,
            continuous_scalar_expansion_residual=scalar_expansion,
            continuous_sample_W=scalar, generic_group_coordinates=h.tolist())

    def _minimal_product_control(self, frame=None, matrix=None):
        if tuple(self.link.names) != CUTOFFS['minimal']:
            return dict(executed=False, reason='complete fundamental intermediate check is minimal-only')
        fields = self.frame if frame is None else frame
        exact = self.matrix if matrix is None else matrix
        full = HaarLink('full')
        rows, cols = [], []
        for name in self.link.names:
            count = self.link.reps[name].shape[1] ** 2
            rows.extend(full.offset[name] + np.arange(count))
            cols.extend(self.link.offset[name] + np.arange(count))
        embedding = sparse.coo_matrix((np.ones(self.link.dimension), (rows, cols)),
                                      shape=(full.dimension, self.link.dimension)).tocsr()
        udagger = [[full.U[i][j].getH() @ embedding for j in range(4)] for i in range(4)]
        u = [[full.U[i][j] @ embedding for j in range(4)] for i in range(4)]
        contraction = sparse.csr_matrix(exact.shape, dtype=complex)
        naive_contraction = sparse.csr_matrix(exact.shape, dtype=complex)
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    for d in range(4):
                        weight = np.dot(fields[:, a, b], fields[:, c, d])
                        if weight != 0:
                            # P U_bc U_ad^dagger P, with a complete intermediate
                            # space for U_ad^dagger P. It is NOT PUP PUdaggerP.
                            contraction += weight * (udagger[b][c].getH() @ udagger[a][d])
                            naive_contraction += weight * (self.link.U[b][c] @ self.link.U[a][d].getH())
        constant = float((2 * np.einsum('mij,mji->', fields, fields)).real)
        independent = constant * sparse.eye(self.link.dimension, format='csr') - 2 * contraction
        naive = constant * sparse.eye(self.link.dimension, format='csr') - 2 * naive_contraction
        identity = sparse.eye(self.link.dimension, format='csr')
        zero = sparse.csr_matrix(identity.shape)
        complete_unitarity = naive_unitarity = 0.
        for i in range(4):
            for j in range(4):
                expected = identity if i == j else zero
                complete_unitarity = max(complete_unitarity,
                    norm(sum(u[k][i].getH() @ u[k][j] for k in range(4)) - expected))
                naive_unitarity = max(naive_unitarity,
                    norm(sum(self.link.U[k][i].getH() @ self.link.U[k][j] for k in range(4)) - expected))
        agreement = norm(independent - exact)
        require_small(agreement / max(1., constant), 'minimal independent Haar product', 3e-7)
        require_small(complete_unitarity, 'minimal full-intermediate unitarity', 3e-7)
        if naive_unitarity < 1e-5:
            raise AssertionError('Naive minimal compressed-unitarity adverse control vanished')
        return dict(executed=True, intermediate_cutoff='full', intermediate_dimension=full.dimension,
            required_Udagger_intermediates={'1':['bar4'], '4':['1','15'], 'bar4':['6','bar10']},
            required_U_intermediates={'1':['4'], '4':['6','10'], 'bar4':['1','15']},
            exact_product_agreement_residual=agreement, complete_intermediate_unitarity_residual=complete_unitarity,
            naive_minimal_unitarity_defect=naive_unitarity, naive_minimal_alignment_defect=norm(naive - exact),
            naive_alignment_defect_need_not_be_nonzero_for_zero_or_special_fields=True)

    def evidence(self):
        """Execute construction identities, complete link spectra and controls.

        Returns only JSON-native values. No matter/charged eigenproblem is run.
        """
        known = self._known_controls()
        multiplicity_correct = True
        for key, record in self._cg.records.items():
            if record['multiplicity'] != EXPECTED_MULTIPLICITIES.get(key, 0):
                multiplicity_correct = False
                raise AssertionError(f'Incomplete CG multiplicity for {key}: {record["multiplicity"]}')
        representations = self._representation_controls()
        hermitian = norm(self.matrix - self.matrix.conj().T)
        scale = max(1., self.constant)
        require_small(hermitian / scale, 'exact alignment Hermiticity', 3e-7)
        values = np.linalg.eigvalsh(self.matrix.toarray())
        require_small(max(0., -float(values[0])) / scale, 'positive projected alignment', 3e-7)
        require_small(max(0., float(values[-1]) - 2 * self.constant) / scale,
                      'alignment upper norm bound', 3e-7)
        centre_labels = np.empty(self.link.dimension, int)
        for name in self.link.names:
            start, count = self.link.offset[name], self.link.reps[name].shape[1] ** 2
            centre_labels[start:start + count] = self.link.centre[name]
        coo = self.matrix.tocoo()
        centre_defect = norm(coo.data[centre_labels[coo.row] != centre_labels[coo.col]])
        require_small(centre_defect / scale, 'alignment preserves centre', 3e-7)
        reconstruction = norm(self.frame - np.einsum('ma,aij->mij', self.p, self.link.T))
        require_small(reconstruction / max(1., norm(self.frame)), 'four-field adjoint reconstruction', 3e-7)
        minimal = self._minimal_product_control()
        # Arbitrary, non-Clifford/nonorthogonal fields and identically zero fields
        # are consumer boundaries, not hard-coded assumptions of the provider.
        generic_p = np.random.default_rng(223).normal(size=(4, 15)) / 3
        generic_frame = np.einsum('ma,aij->mij', generic_p, self.link.T)
        nonclifford_defect = max(norm(a @ b + b @ a - np.trace(a @ b) * np.eye(4) / 2)
                                for a in generic_frame for b in generic_frame)
        if nonclifford_defect <= 1e-5:
            raise AssertionError('Generic non-Clifford field adverse control vanished')
        generic, _ = self._project(generic_p)
        generic_values = np.linalg.eigvalsh(generic.toarray())
        require_small(max(0., -float(generic_values[0])), 'generic-field projected positivity', 3e-7)
        generic_hermitian = norm(generic - generic.conj().T)
        require_small(generic_hermitian, 'generic-field Hermiticity', 3e-7)
        zero_matrix, _ = self._project(np.zeros((4, 15)))
        require_small(norm(zero_matrix), 'zero-field alignment', 3e-7)
        invalid = {}
        for name, fields in (('nontraceless', np.broadcast_to(np.eye(4), (4, 4, 4))),
                             ('nonHermitian', np.broadcast_to(1j * np.eye(4), (4, 4, 4)))):
            try:
                _coefficients(self.link, fields)
            except ValueError:
                invalid[name] = True
            else:
                raise AssertionError(f'Invalid {name} fields were accepted')
        return dict(method='Exact adjoint Peter-Weyl multiplication via Casimir-restricted complete Hom kernels',
            cutoff=list(self.link.names), dimension=self.link.dimension,
            coefficients_p=self.p.tolist(), gram_K=self.K.tolist(), constant_sum_p_squared=self.constant,
            hermiticity_residual=hermitian, centre_block_residual=centre_defect,
            field_reconstruction_residual=reconstruction, complete_CG_multiplicities_correct=multiplicity_correct,
            cg_channels=[record for _, record in sorted(self._cg.records.items())],
            tensor_product_casimir_spectra={s:_spectral_groups(data[1]) for s,data in sorted(self._cg.products.items())},
            known_positive_controls=known, representation_controls=representations,
            eigenvalues=values.tolist(), minimum=float(values[0]), maximum=float(values[-1]),
            continuum_pointwise_bounds=[0., 2 * self.constant],
            spectrum_interpretation='finite Haar Ritz spectrum of multiplication; not an infinite Higgs or charged gap',
            minimal_independent_product=minimal,
            arbitrary_field_control=dict(nonClifford=True, anticommutator_nonscalar_defect=nonclifford_defect,
                coefficients_p=generic_p.tolist(),
                minimum=float(generic_values[0]), hermiticity_residual=generic_hermitian,
                minimal_independent_product=self._minimal_product_control(generic_frame, generic)),
            zero_field_residual=norm(zero_matrix), invalid_fields_rejected=invalid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs', nargs='+', choices=tuple(CUTOFFS), default=list(CUTOFFS))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stiffness', type=float, default=2.)
    parser.add_argument('--m', type=float, default=.2)
    parser.add_argument('--r', type=float, default=.12)
    parser.add_argument('--v', type=float, default=.3)
    args = parser.parse_args()
    if not all(np.isfinite(value) for value in (args.stiffness, args.m, args.r, args.v)) or args.stiffness <= 0:
        parser.error('finite coefficients and positive stiffness required')
    band = RotationBand(a=1., m=args.m, r=args.r, v=args.v, improved=False)
    data = uniform_data(band, 3, args.stiffness, 0., matrix_unit_basis())
    result: dict[str, Any] = dict(model='Exact continuous Haar alignment projection; selected fixed-orbit EFT frame',
        parameters={key:value for key,value in vars(args).items() if key != 'output'},
        selected_source='local_orbital_stability.uniform_data(band,size=3,preferred=0)',
        selected_frame_real=data['frame'].real.tolist(), selected_frame_imaginary=data['frame'].imag.tolist(),
        selected_gram_diagonal=np.diag(data['selected']['G']).tolist(),
        selected_gram_stationarity_residual=float(data['selected']['gradient_residual']),
        hypotheses=['arbitrary four traceless Hermitian 4x4 fields for the multiplication API',
            'CLI chooses the existing selected frame; this is a fixed-orbit EFT input, not a full60 quantum field solution',
            'canonical HaarLink representations, Tr(Ta Tb)=delta_ab/2 and exact Schur orthogonality',
            'finite Peter-Weyl projection is performed after continuous multiplication, not between its factors'],
        excluded_claims=['full60 quantum orbital law', 'infinite Higgs gap', 'charged gap', 'quantum3D phase',
                         'deconfined Z4 phase from the classical stabilizer alone', 'canonical QED'],
        alignments=[], nested_cutoff_controls=[])
    providers = {}
    for cutoff in args.cutoffs:
        provider = ExactHaarAlignment(HaarLink(cutoff), data['frame'])
        providers[cutoff] = provider
        result['alignments'].append(dict(cutoff_name=cutoff, **provider.evidence()))
    for small_name, small in providers.items():
        for large_name, large in providers.items():
            if small.link.dimension >= large.link.dimension or not set(small.link.names) <= set(large.link.names):
                continue
            indices = np.concatenate([large.link.offset[n] + np.arange(small.link.reps[n].shape[1] ** 2)
                                      for n in small.link.names])
            defect = norm(large.matrix[indices][:, indices] - small.matrix)
            require_small(defect / max(1., small.constant), 'nested exact Haar projections', 3e-7)
            result['nested_cutoff_controls'].append(dict(smaller=small_name, larger=large_name, residual=defect))
    result['source_sha256'] = source_hashes()
    here = Path(__file__).resolve()
    result['source_sha256'][here.name] = hashlib.sha256(here.read_bytes()).hexdigest()
    derivation = here.with_suffix('.md')
    result['source_sha256'][derivation.name] = hashlib.sha256(derivation.read_bytes()).hexdigest()
    result['runtime'] = dict(numpy=np.__version__)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()
