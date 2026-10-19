"""Exact native contraction of the DECLARED operator, not weak coupling.
No physical coefficient threshold, source/sea/relative cut, or primitive H matrix.
The two-word key is exactly the original106-bit canonical state encoding.
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(inline='always')
def popcount(x):
    x = np.uint64(x)
    x -= (x >> np.uint64(1)) & np.uint64(0x5555555555555555)
    x = (x & np.uint64(0x3333333333333333))+((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x+(x >> np.uint64(4))) & np.uint64(0x0f0f0f0f0f0f0f0f)
    return np.int64((x*np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(inline='always')
def bit(lo, hi, mode):
    return np.int64((lo >> np.uint64(mode)) & np.uint64(1)) if mode < 64 else np.int64((hi >> np.uint64(mode-64)) & np.uint64(1))


@njit(inline='always')
def parity(lo, hi, mode):
    if mode < 64:
        n = popcount(lo & ((np.uint64(1) << np.uint64(mode))-np.uint64(1)))
    else:
        n = popcount(lo)+popcount(hi & ((np.uint64(1) << np.uint64(mode-64))-np.uint64(1)))
    return -1. if n % 2 else 1.


@njit(inline='always')
def toggle(lo, hi, mode):
    if mode < 64:
        return lo ^ (np.uint64(1) << np.uint64(mode)), hi
    return lo, hi ^ (np.uint64(1) << np.uint64(mode-64))


@njit(inline='always')
def local_bits(lo, hi, site):
    if site < 3:
        return (lo >> np.uint64(18*site)) & np.uint64(262143)
    return ((lo >> np.uint64(54)) | ((hi & np.uint64(255)) << np.uint64(10))) & np.uint64(262143)


@njit(inline='always')
def hash_key(lo, hi):
    x = lo ^ (hi*np.uint64(0x9e3779b97f4a7c15))
    x = (x ^ (x >> np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    return x ^ (x >> np.uint64(31))


@njit
def make_table(lows, highs):
    size = 1
    while size < 2*len(lows):
        size *= 2
    table = np.full(size, -1, dtype=np.int64)
    for row in range(len(lows)):
        slot = np.int64(hash_key(lows[row], highs[row]) & np.uint64(size-1))
        while table[slot] != -1:
            slot = (slot+1) & (size-1)
        table[slot] = row
    return table


@njit(inline='always')
def find(table, lows, highs, lo, hi):
    slot = np.int64(hash_key(lo, hi) & np.uint64(len(table)-1))
    while True:
        row = table[slot]
        if row == -1:
            return -1
        if lows[row] == lo and highs[row] == hi:
            return row
        slot = (slot+1) & (len(table)-1)


@njit(inline='always')
def contract(out, table, lows, highs, indptr, indices, values, source, lo, hi, amplitude):
    if amplitude == 0:
        return
    target = find(table, lows, highs, lo, hi)
    if target == -1:
        return
    # Only a weak matrix element is selected here. No claim H has no exterior.
    for a in range(indptr[target], indptr[target+1]):
        left = values[a].conjugate()*amplitude
        alpha = indices[a]
        for b in range(indptr[source], indptr[source+1]):
            out[alpha, indices[b]] += left*values[b]


@njit
def weak_h(lows, highs, indptr, indices, values, rank, table, energy, x, x2, overlap,
           hop_ptr, hop_info, hop_value, u_target, u_value, pairs, pair_value, mass, electric, gradient):
    out = np.zeros((rank, rank), dtype=np.complex128)
    counts = np.empty(4, dtype=np.int64)
    for source in range(len(lows)):
        lo, hi = lows[source], highs[source]
        total = 0
        diagonal = 0.
        for site in range(4):
            counts[site] = popcount(local_bits(lo, hi, site))
            total += counts[site]
            cloud = np.int64((hi >> np.uint64(38+site)) & np.uint64(1))
            diagonal += energy[counts[site], cloud]
        diagonal += mass*total
        for edge in range(6):
            if ((hi >> np.uint64(8+5*edge)) & np.uint64(31)) != 0:
                diagonal += electric*4/3
        contract(out, table, lows, highs, indptr, indices, values, source, lo, hi, diagonal)
        if gradient != 0:
            for i in range(4):
                for j in range(i+1, 4):
                    ci = np.int64((hi >> np.uint64(38+i)) & np.uint64(1))
                    cj = np.int64((hi >> np.uint64(38+j)) & np.uint64(1))
                    for target_cloud in range(2):
                        target_hi = hi ^ (np.uint64(ci ^ target_cloud) << np.uint64(38+i))
                        amplitude = gradient/2*x2[counts[i], target_cloud, ci]
                        contract(out, table, lows, highs, indptr, indices, values, source, lo, target_hi, amplitude)
                        target_hi = hi ^ (np.uint64(cj ^ target_cloud) << np.uint64(38+j))
                        amplitude = gradient/2*x2[counts[j], target_cloud, cj]
                        contract(out, table, lows, highs, indptr, indices, values, source, lo, target_hi, amplitude)
                    for a in range(2):
                        for b in range(2):
                            target_hi = hi ^ (np.uint64(ci ^ a) << np.uint64(38+i)) ^ (np.uint64(cj ^ b) << np.uint64(38+j))
                            amplitude = -gradient*x[counts[i], counts[i], a, ci]*x[counts[j], counts[j], b, cj]
                            contract(out, table, lows, highs, indptr, indices, values, source, lo, target_hi, amplitude)
        for pair in range(len(pairs)):
            amplitude = pair_value[pair]
            if amplitude == 0:
                continue
            q, anti, site = pairs[pair]
            for create in range(2):
                first, second = (anti, q) if create else (q, anti)
                if bit(lo, hi, first) == create or bit(lo, hi, second) == create:
                    continue
                sign = parity(lo, hi, first)
                newlo, newhi = toggle(lo, hi, first)
                sign *= parity(newlo, newhi, second)
                newlo, newhi = toggle(newlo, newhi, second)
                oldload = counts[site]; newload = oldload+(2 if create else -2)
                oldcloud = np.int64((hi >> np.uint64(38+site)) & np.uint64(1))
                for newcloud in range(2):
                    target_hi = newhi ^ (np.uint64(oldcloud ^ newcloud) << np.uint64(38+site))
                    coefficient = amplitude*sign*x[newload, oldload, newcloud, oldcloud]
                    contract(out, table, lows, highs, indptr, indices, values, source, newlo, target_hi, coefficient)
        # Iterate OCCUPIED canonical modes, never all vacant one-body terms.
        for half in range(2):
            occupied = lo if half == 0 else hi & np.uint64(255)
            while occupied:
                least = occupied & (~occupied+np.uint64(1))
                source_mode = popcount(least-np.uint64(1))+64*half
                occupied &= occupied-np.uint64(1)
                for term in range(hop_ptr[source_mode], hop_ptr[source_mode+1]):
                    target_mode, edge, a, b, adjoint, target_site, source_site = hop_info[term]
                    if bit(lo, hi, target_mode):
                        continue
                    link = np.int64((hi >> np.uint64(8+5*edge)) & np.uint64(31))
                    newlink = u_target[a, b, adjoint, link]
                    if newlink < 0:
                        continue
                    sign = parity(lo, hi, source_mode)
                    newlo, newhi = toggle(lo, hi, source_mode)
                    sign *= parity(newlo, newhi, target_mode)
                    newlo, newhi = toggle(newlo, newhi, target_mode)
                    newhi ^= np.uint64(link ^ newlink) << np.uint64(8+5*edge)
                    oldtarget = np.int64((hi >> np.uint64(38+target_site)) & np.uint64(1))
                    oldsource = np.int64((hi >> np.uint64(38+source_site)) & np.uint64(1))
                    coefficient = sign*hop_value[term]*u_value[a, b, adjoint, link]
                    for tc in range(2):
                        for sc in range(2):
                            target_hi = newhi ^ (np.uint64(oldtarget ^ tc) << np.uint64(38+target_site)) ^ (np.uint64(oldsource ^ sc) << np.uint64(38+source_site))
                            value = coefficient*overlap[counts[target_site]+1, counts[target_site], tc, oldtarget]*overlap[counts[source_site]-1, counts[source_site], sc, oldsource]
                            contract(out, table, lows, highs, indptr, indices, values, source, newlo, target_hi, value)
        if source % 250000 == 0:
            print('native exact weak rows', source, len(lows))
    return out


@njit
def physical_observe(lows, highs, table, state, x, top):
    density, anti_density, charge, phi, compact_centres = np.zeros((5, 4))
    norm, compact, separated, radius, two_pairs, casimir, sz, cloud_n, top_diag, mixed, zero_pair = (0.,)*11
    conditional_density = np.zeros(4)
    net = np.empty(4, dtype=np.int64)
    for row in range(len(lows)):
        probability = (state[row].conjugate()*state[row]).real
        lo, hi = lows[row], highs[row]
        total_anti = 0
        for site in range(4):
            bits = local_bits(lo, hi, site)
            u = popcount(bits & np.uint64(63)); d = popcount((bits >> np.uint64(6)) & np.uint64(63)); anti = popcount((bits >> np.uint64(12)) & np.uint64(63))
            n = u+d+anti; total_anti += anti; net[site] = u+d-anti
            density[site] += probability*n; anti_density[site] += probability*anti
            charge[site] += probability*(2/3*(u-anti)-d/3)
            cloud = np.int64((hi >> np.uint64(38+site)) & np.uint64(1))
            cloud_n += probability*cloud; top_diag += probability*top[n, cloud]
            for species in range(3):
                local = bits >> np.uint64(6*species)
                sz += probability*(popcount(local & np.uint64(7))-popcount((local >> np.uint64(3)) & np.uint64(7)))/2
            for target_cloud in range(2):
                target_hi = hi ^ (np.uint64(cloud ^ target_cloud) << np.uint64(38+site))
                target = find(table, lows, highs, lo, target_hi)
                if target != -1:
                    phi[site] += (state[target].conjugate()*x[n, n, target_cloud, cloud]*state[row]).real
        norm += probability
        if total_anti == 0:
            zero_pair += probability
            largest = max(net[0], net[1], net[2], net[3])
            if largest == 3:
                compact += probability
                for site in range(4):
                    if net[site] == 3:
                        compact_centres[site] += probability
            if largest == 1:
                separated += probability
            for site in range(4):
                conditional_density[site] += probability*net[site]/3
            for i in range(4):
                for j in range(i+1, 4):
                    radius += probability*net[i]*net[j]/3  # all tetra edges have length1
        if total_anti >= 2:
            two_pairs += probability
        if total_anti > 0 and min(net[0], net[1], net[2], net[3]) < 0:
            mixed += probability
        for edge in range(6):
            if ((hi >> np.uint64(8+5*edge)) & np.uint64(31)) != 0:
                casimir += probability*4/3
    return np.array([norm, compact, separated, radius, two_pairs, casimir, sz, cloud_n, top_diag, mixed, zero_pair]), density, anti_density, charge, phi, compact_centres, conditional_density


@njit
def physical_spin2(lows, highs, table, state):
    total = 0.+0j
    for row in range(len(lows)):
        lo, hi = lows[row], highs[row]
        sz = 0.
        for site in range(4):
            bits = local_bits(lo, hi, site)
            for species in range(3):
                local = bits >> np.uint64(6*species)
                sz += (popcount(local & np.uint64(7))-popcount((local >> np.uint64(3)) & np.uint64(7)))/2
        total += state[row].conjugate()*state[row]*sz*sz
        # BOTH full operator orders, with no intermediate union/Galerkin cut.
        for first_raise in range(2):
            for half in range(2):
                occupied = lo if half == 0 else hi & np.uint64(255)
                while occupied:
                    least = occupied & (~occupied+np.uint64(1))
                    source_mode = popcount(least-np.uint64(1))+64*half
                    occupied &= occupied-np.uint64(1)
                    old_spin = (source_mode % 6)//3
                    if old_spin != first_raise:
                        continue
                    target_mode = source_mode+(-3 if first_raise else 3)
                    if bit(lo, hi, target_mode):
                        continue
                    sign = parity(lo, hi, source_mode)
                    midlo, midhi = toggle(lo, hi, source_mode)
                    sign *= parity(midlo, midhi, target_mode)
                    midlo, midhi = toggle(midlo, midhi, target_mode)
                    for second_half in range(2):
                        second_occupied = midlo if second_half == 0 else midhi & np.uint64(255)
                        while second_occupied:
                            least_second = second_occupied & (~second_occupied+np.uint64(1))
                            second_source = popcount(least_second-np.uint64(1))+64*second_half
                            second_occupied &= second_occupied-np.uint64(1)
                            if (second_source % 6)//3 == first_raise:
                                continue
                            second_target = second_source+(3 if first_raise else -3)
                            if bit(midlo, midhi, second_target):
                                continue
                            second_sign = parity(midlo, midhi, second_source)
                            final_lo, final_hi = toggle(midlo, midhi, second_source)
                            second_sign *= parity(final_lo, final_hi, second_target)
                            final_lo, final_hi = toggle(final_lo, final_hi, second_target)
                            target = find(table, lows, highs, final_lo, final_hi)
                            if target != -1:
                                total += .5*sign*second_sign*state[target].conjugate()*state[row]
    return total
