#!/usr/bin/env python3
"""Exact external-photon graph on grade0/1 states, colour vacuum, normal DVR.

This is NOT a photon dynamical operator. Infinite harmonic Wilson displacement
is handled by its exact Gaussian matrix elements and unitarity, not truncating
its norm. All incoming/outgoing choices here are coexactY/Z and commute with
SAME originalWilson source (which depends on harmonicX). Charged projection
occurs AFTER S a†. Full receiver Gram includes its omitted photon components.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from scipy.linalg import eigh
import physical_c_bare_electric_kernel as bare
import physical_c_spin_analyzer_packet as spin


def harmonic_wilson(p: bare.BareElectricParameters) -> np.ndarray:
    if p.photon_grade!=1 or p.colour_order!=1:
        raise ValueError('Exact external graph here is grade0/1 with identity colour vacuum')
    e=bare.alpha.charge_contract(bare.alpha.Parameters())['bare_e']
    displacement=1j*e/2;vacuum=math.exp(-e**2/8)
    matrix=np.eye(6,dtype=complex)*vacuum
    matrix[0,3]=matrix[3,0]=displacement*vacuum
    matrix[3,3]=vacuum*(1-e**2/4)
    return matrix


def exact_source_coefficients(p: bare.BareElectricParameters,centre: int) -> np.ndarray:
    fielddim=6*p.normal_order
    w=np.kron(harmonic_wilson(p),np.eye(p.normal_order))
    result=np.zeros((8,4,fielddim,fielddim),complex)
    for a in range(4):
        result[a,a]=np.exp(1j*np.pi*centre/2)*w/2
        result[4+a,a]=np.eye(fielddim)/2
    return result


def projected_joint_source(state: np.ndarray,mode: int,centre: int,creation: np.ndarray,p: bare.BareElectricParameters) -> np.ndarray:
    """P a† S P=P S a† P: source depends only on harmonicX, modeY/Z.

    Creation only raises Hermite degree, hence P a†=P a† P, while inserting
    P on the OTHER order S P a† is incorrect. No intermediate neutral gate.
    """
    if mode not in (0,1):raise ValueError('This exact joint identity uses source-orthogonal coexact modes')
    source=bare.reference.apply_source(exact_source_coefficients(p,centre),creation,state)
    tensor=source.reshape(4,6,p.normal_order,56)
    result=np.zeros_like(tensor);result[:,mode+1]=tensor[:,0]
    return result.reshape(4,-1)


def graph_parts(state: np.ndarray,mode: int,centre: int,creation: np.ndarray,p: bare.BareElectricParameters,transport: np.ndarray | None=None) -> tuple[np.ndarray,np.ndarray]:
    """Source components in six virtual image states of a†P; no phase norm loss.

    Image states are |mode>,sqrt2|2mode>,|mode,j>. They are orthogonal;
    their harmonicX occupations match the original six labels, so the exact
    Wilson compression is harmonic_wilson. Top source=W top, bottom unchanged.
    """
    if mode not in (0,1):raise ValueError('Virtual-image formula uses coexactY/Z')
    harmonic_wilson(p)
    image=state.reshape(6,p.normal_order,70).copy();image[mode+1]*=math.sqrt(2)
    top=np.empty((4,6,p.normal_order,56),complex);bottom=np.empty_like(top)
    for a in range(4):
        top[a]=np.einsum('pif,gf->pig',image,creation[a],optimize=True)*np.exp(1j*np.pi*centre/2)/2
        bottom[a]=np.einsum('pif,gf->pig',image,creation[4+a],optimize=True)/2
    top=top.reshape(4,-1);bottom=bottom.reshape(4,-1)
    if transport is not None:
        pr=bare.reference_parameters(p)
        top=spin.transport_Ref(top,transport,pr,adjoint=True)
        bottom=spin.transport_Ref(bottom,transport,pr,adjoint=True)
    return top,bottom


def full_cross_gram(parts1: tuple[np.ndarray,np.ndarray],parts0: tuple[np.ndarray,np.ndarray],p: bare.BareElectricParameters) -> np.ndarray:
    """<W t1+b1,W t0+b0>, exact WdagW=I BEFORE any photon projection."""
    t1,b1=parts1;t0,b0=parts0;w=harmonic_wilson(p)
    wt0=np.einsum('pq,aqig->apig',w,t0.reshape(4,6,p.normal_order,56),optimize=True).reshape(4,-1)
    wb0=np.einsum('pq,aqig->apig',w.conj().T,b0.reshape(4,6,p.normal_order,56),optimize=True).reshape(4,-1)
    return t1.conj()@t0.T+b1.conj()@b0.T+t1.conj()@wb0.T+b1.conj()@wt0.T


def full_frame_effect_price(gram1: np.ndarray,gram0: np.ndarray,cross: np.ndarray) -> dict[str,Any]:
    """Rank<=8 signed-Gram spectrum, independent of ambient photon dimension."""
    joined=np.block([[gram1,cross],[cross.conj().T,gram0]])
    values,vectors=eigh((joined+joined.conj().T)/2)
    if values[0]<-1e-10*max(1.,float(values[-1])):
        raise ArithmeticError('Actual uncompressed joined Gram is not positive')
    # Remove roundoff-negative Gram values only; no positive spectral floor.
    square=(vectors*np.sqrt(np.maximum(values,0)))@vectors.conj().T
    signed=(square*np.r_[np.ones(4),-np.ones(4)])@square
    delta=float(np.max(abs(eigh((signed+signed.conj().T)/2,eigvals_only=True))))
    eig0=eigh((gram0+gram0.conj().T)/2,eigvals_only=True)
    eig1=eigh((gram1+gram1.conj().T)/2,eigvals_only=True)
    mu0=float(eig0[-1]);mu1=float(eig1[-1])
    if min(mu0,mu1)<=0:raise ArithmeticError('Actual fullframe normalization vanished')
    return dict(full_raw_frame_difference_norm=delta,reference_full_mu=mu0,actual_full_mu=mu1,
        normalized_FULL_frame_effect_difference_upper=2*delta/mu0,
        actual_FULL_Gram_eigenweights=np.repeat(eig1,4).tolist(),reference_FULL_Gram_eigenweights=np.repeat(eig0,4).tolist(),
        actual_FULL_rank_relative1e_12=int(4*np.sum(eig1>mu1*1e-12)),
        no_Gmin_no_inverse_no_orthoprojector=True,joined_Gram_roundoff_min=float(values[0]))
