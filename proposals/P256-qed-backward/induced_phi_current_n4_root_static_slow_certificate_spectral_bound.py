"""Independently price the static q0 Phi/colour Schur complement.

This is a norm certificate of the actual all90 static PH pencil, not a
projection that deletes cross couplings or a quantum-ground assertion.
"""
from __future__ import annotations
from fractions import Fraction as F
from typing import Any


def full_q0_bound(profiles: list[dict[str,Any]], g: list[Any], mask: list[list[int]], phi_lower: F, iv: Any, exact: Any, endpoint: Any) -> dict[str,Any]:
    def upper(value: Any) -> F:
        return endpoint(abs(value),1)
    roots = [iv.sqrt(value) for value in g]
    rows = []; total = F(0)
    for p in profiles:
        nearest = next(path for path in p['paths'] if path['length'] == 1)
        gap = min(endpoint(2*iv.sqrt(g[0]*(exact(p['onsite'])+iv.sqrt(exact(2))*exact(nearest['c0'])*z)**2+6*g[1]*exact(nearest['ci'])**2),0) for z in (3,1,-1,-3))
        coefficients = F(0); weighted = exact(0); pc = F(0); cc = exact(0)
        for path in p['paths']:
            c0,ci,L = abs(F(path['c0'])),abs(F(path['ci'])),path['length']
            hop = exact(c0)*roots[0]+exact(ci)*roots[path['axis']+1]
            coefficients += c0+ci
            weighted += L*hop
            pc += L*(c0+ci)
            cc += exact(F(L*L,2))*hop
        # Componentwise Phi and colour vertices; factor2 deliberately prices
        # both endpoints/reverse edges even before taking the combined HS bound.
        vp = 2*(abs(exact(p['onsite']))+2*exact(coefficients))/(8*iv.sqrt(exact(2)))
        vc = 2*weighted/(8*iv.sqrt(exact(2)))
        # There is no PP contact because the endpoint matter law is affinePhi.
        # Rank128/64 Fourier factors and BOTH orientations are retained. Factor16
        # conservatively dominates all ordered same-link/adjacent-link products.
        contact = 45*iv.sqrt(2*exact(16*pc)**2+(16*cc)**2)
        bubble = exact(256*45)*(vp**2+vc**2)/exact(gap)
        cap = upper(contact+bubble)
        scale = F(1) if p['scale'] == 'heavy' else F(p['load'],2**8192)
        total += scale*cap
        rows.append({'role':p['role'],'gap_lower':str(gap),'component_Phi_vertex_upper':str(upper(vp)),
                     'component_colour_vertex_upper':str(upper(vc)),'contact90_upper':str(upper(contact)),
                     'both_PH90_upper':str(upper(bubble)),'sea90_operator_upper':str(cap),'weight_divided_Lambda':str(scale)})
    hmin = min(endpoint(sum(exact(F(bit,2))*value for bit,value in zip(row,g,strict=True)),0) for row in mask)
    fast_colour_lower = hmin/5
    colour_lower = 2**2048*fast_colour_lower-total
    assert colour_lower > 0
    schur_price = total**2/colour_lower
    schur_lower = phi_lower-schur_price
    # Young's inequality: 2|<x,B y>| <=2M²/d||x||²+d/2||y||².
    full_lower = min(phi_lower-2*schur_price,colour_lower/2)
    assert schur_lower > F(2,5) and full_lower > F(2,5)
    return {'q0_Phi_Schur_stiffness_lower_divided_Lambda':str(schur_lower),
            'q0_full90_static_stiffness_lower_divided_Lambda':str(full_lower),
            'q0_full90_frequency_squared_lower':str(F(3,10)*full_lower*F(2)**4096),
            'slow_Gphi':8,'colour_G_at_q0':'3/10','Horbit_lower':str(hmin),
            'q0_fast_colour_K_Lambda5over4_lower':str(fast_colour_lower),
            'q0_colour_block_lower_divided_Lambda':str(colour_lower),
            'matter_full90_operator_upper_divided_Lambda':str(total),
            'Phi_colour_Schur_price_divided_Lambda':str(schur_price),'species_prices':rows,
            'proof':'Actual q0 K/Lambda=[[A,B],[B*,2^2048 D+E]] with A independently certified, D=I3 tensor H/5, H=.5anticommSubset.g; ||B||,||E||<=M from the independently repriced full90 contact-minus-BOTHchannels sea. Ordered one/two links and reverse/AP remain in each norm price. Colour block>=d=2^2048 min(D)-M>0; Schur Phi stiffness>=a-M²/d. Young gives whole K/Lambda>=min(a-2M²/d,d/2)I90. The actual cometric is chi diag(8I45,.3I45) at q0, yielding whole squared-frequency lower .3chiLambda times wholeK lower. SlowPhi Schur coordinates retain Gphi8; no unit-metric or ignored mixed-term shortcut.',
            'nonclaim':'Static independent-sea harmonic pencil only, not interacting HC ground or stationary full quantum vacuum.'}
