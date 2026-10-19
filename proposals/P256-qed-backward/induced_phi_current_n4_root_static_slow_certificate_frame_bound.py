"""Outward differentiation of the actual algebraic normal45 frame.

Only mathematical interval jets are evaluated; no scientific supplier executes.
The value/first-jet formulas reproduce the explicit orthogonal orbit complement.
"""
from __future__ import annotations
from fractions import Fraction as F
from itertools import combinations
from typing import Any


def derivative_bound(iv: Any, gamma: Any, matrices: Any, exact: Any, endpoint: Any) -> dict[str, Any]:
    # A jet is (outward value, four outward partial derivatives).
    def constant(x: Any) -> list[Any]:
        return [exact(x), *[exact(0) for _ in range(4)]]
    def add(a: list[Any], b: list[Any]) -> list[Any]:
        return [x+y for x,y in zip(a,b,strict=True)]
    def neg(a: list[Any]) -> list[Any]:
        return [-x for x in a]
    def sub(a: list[Any], b: list[Any]) -> list[Any]:
        return add(a,neg(b))
    def mul(a: list[Any], b: list[Any]) -> list[Any]:
        return [a[0]*b[0], *[a[j]*b[0]+a[0]*b[j] for j in range(1,5)]]
    def div(a: list[Any], b: list[Any]) -> list[Any]:
        return [a[0]/b[0], *[(a[j]*b[0]-a[0]*b[j])/b[0]**2 for j in range(1,5)]]
    def sqrt(a: list[Any]) -> list[Any]:
        r = iv.sqrt(a[0]); return [r, *[a[j]/(2*r) for j in range(1,5)]]
    def total(values: list[list[Any]]) -> list[Any]:
        result = constant(0)
        for value in values:
            result = add(result,value)
        return result
    def square(a: list[Any]) -> list[Any]:
        # Dependency-safe square for sign-definite values in this domain.
        assert endpoint(a[0],0)*endpoint(a[0],1) >= 0
        return [a[0]**2, *[2*a[0]*a[j] for j in range(1,5)]]
    g = [[iv.mpf([exact(F(1,10)).a,exact(F(3,10)).b]), *[exact(int(mu == j)) for j in range(4)]] for mu in range(4)]
    r = [sqrt(x) for x in g]
    columns: list[dict[int,list[Any]]] = [{15*j+j:constant(1)} for j in range(4)]
    for i,j in combinations(range(4),2):
        den = sqrt(add(g[i],g[j])); columns.append({15*i+j:div(r[j],den),15*j+i:div(r[i],den)})
    used: set[int] = set()
    for j in range(4):
        entries = []
        for mu in range(4):
            if mu == j:
                continue
            comm = 1j*(gamma[j]@gamma[mu]-gamma[mu]@gamma[j])
            a = next(a for a in range(4,10) if (matrices[a]@comm).trace() != 0)
            sign = int((matrices[a]@comm).trace().real/8)
            entries.append((15*mu+a,mul(constant(sign),r[mu])))
        ids,v = zip(*entries,strict=True); used.update(ids)
        den = sqrt(add(square(v[0]),square(v[1])))
        norm = sqrt(total([square(x) for x in v]))
        n = [div(v[1],den),neg(div(v[0],den)),constant(0)]
        z = [div(sub(mul(v[1],n[2]),mul(v[2],n[1])),norm),
             div(sub(mul(v[2],n[0]),mul(v[0],n[2])),norm),
             div(sub(mul(v[0],n[1]),mul(v[1],n[0])),norm)]
        columns.extend([{slot:value for slot,value in zip(ids,n,strict=True)}, {slot:value for slot,value in zip(ids,z,strict=True)}])
    for mu in range(4):
        for a in range(4,10):
            if 15*mu+a not in used:
                columns.append({15*mu+a:constant(1)})
    ids = []; v = []
    for mu in range(4):
        comm = 1j*(matrices[14]@gamma[mu]-gamma[mu]@matrices[14])
        a = next(a for a in range(10,14) if (matrices[a]@comm).trace() != 0)
        sign = int((matrices[a]@comm).trace().real/8)
        ids.append(15*mu+a); v.append(mul(constant(sign),r[mu]))
    norm = sqrt(total([square(x) for x in v])); w = list(v); w[0] = sub(w[0],norm)
    assert endpoint(w[0][0],1) < 0
    denominator = total([square(x) for x in w])
    assert endpoint(denominator[0],0) > 0
    for j in range(1,4):
        columns.append({ids[i]:sub(constant(int(i == j)),div(mul(constant(2),mul(w[i],w[j])),denominator)) for i in range(4)})
    for mu in range(4):
        for a in range(10,14):
            if 15*mu+a not in ids:
                columns.append({15*mu+a:constant(1)})
    assert len(columns) == 45
    derivatives = []
    for j in range(1,5):
        squared = sum(endpoint(abs(jet[j]),1)**2 for column in columns for jet in column.values())
        derivatives.append(endpoint(iv.sqrt(exact(squared)),1))
    price = sum(derivatives)
    return {'root_parameter_domain':['1/10','3/10'],
            'partial_Q_Frobenius_upper':[str(x) for x in derivatives],
            'Q_operator_Lipschitz_from_parameter_infinity_upper':str(price),
            'Householder_denominator_lower':str(endpoint(denominator[0],0)),
            'proof':'Outward first interval jets of every sparse algebraic Q entry on [1/10,3/10]^4. For each parameter, operator norm<=Frobenius norm; integrating along root segment gives ||Qnew-Qold||<=sum_mu partialFrobeniusUpper *||gnew-gold||infinity. Square operation uses sign-definite intervals; actual Householder first sign is negative so denominator stays positive.'}
