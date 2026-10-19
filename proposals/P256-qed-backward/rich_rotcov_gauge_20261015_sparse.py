"""Dimension-relative finite-precision CSR conversion before tensor expansion.

An ordinary explicitly passed factory, never a function/module-global rewrite.
The threshold scales with each operand in its own physical units; uniformly
weak operands are NOT compared with an arbitrary unit floor. All coefficient
perturbations, including tiny whole operands that are retained, are reported.
"""
import numpy as np
from scipy.sparse import csr_matrix as scipy_csr, kron as scipy_kron


class SparseRoundoff:
    def __init__(self):
        self.stats=dict(conversions=0,removed_entries=0,max_removed_abs=0.,removed_Frobenius_sum=0.,max_removed_relative_Frobenius=0.,max_Hermiticity_drift_Frobenius=0.,epsilon_multiplier=8.,threshold='8 epsilon_machine max(rows,cols) max_abs_operand; NO absolute/unit floor')

    def csr(self,value,*args,**kwargs):
        if isinstance(value,tuple) and len(value)==2 and all(isinstance(x,(int,np.integer)) for x in value):return scipy_csr(value,*args,**kwargs)
        matrix=scipy_csr(value,*args,**kwargs)
        if not matrix.nnz:return matrix
        scale=float(np.max(abs(matrix.data)));bound=8*np.finfo(float).eps*max(matrix.shape)*scale
        small=(abs(matrix.data)<=bound)&(matrix.data!=0);self.stats['conversions']+=1
        if np.any(small):
            original=matrix;removed=matrix.data[small].copy();matrix=matrix.copy();matrix.data[small]=0;matrix.eliminate_zeros()
            norm=float(np.linalg.norm(removed));original_norm=float(np.linalg.norm(original.data))
            self.stats['removed_entries']+=len(removed);self.stats['max_removed_abs']=max(self.stats['max_removed_abs'],float(np.max(abs(removed))))
            self.stats['removed_Frobenius_sum']+=norm
            self.stats['max_removed_relative_Frobenius']=max(self.stats['max_removed_relative_Frobenius'],norm/original_norm if original_norm else 0.)
            if matrix.shape[0]==matrix.shape[1]:
                change=matrix-original;drift=change-change.conj().T
                self.stats['max_Hermiticity_drift_Frobenius']=max(self.stats['max_Hermiticity_drift_Frobenius'],float(np.linalg.norm(drift.data)))
        return matrix

    def kron(self,left,right,format='csr'):
        return scipy_kron(self.csr(left),self.csr(right),format=format)

    def snapshot(self):return dict(self.stats)
