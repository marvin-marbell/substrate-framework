"""Exact native all-matrix onsite strictness control, supplementary to OPEN/AP/BZ join."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from fractions import Fraction


def main() -> None:
    root = Path(__file__).parent
    profile = root/'neutral_body_parent_primitive_profiles_20261012.json'
    expected = '36f653729d91d0d8c021e708faed651098eb8e1e3e0a062d6f362e114fbf3c72'
    assert hashlib.sha256(profile.read_bytes()).hexdigest() == expected
    m = Fraction(json.loads(profile.read_text())['primitive_profiles'][0]['onsite'])
    phi = (3, -1, -1, -1)
    levels = sorted((m*t for t in phi),reverse=True)
    canonical = sum(levels[:2])
    positive = sum(t for t in levels if t > 0)
    assert sum(levels) == 0 and canonical == 2*m and positive == 3*m and canonical < positive
    x0 = Fraction(sum(t*t for t in phi),4)
    jordan = 16*sum(t**4 for t in phi)
    norm_lower = 64*x0*x0
    assert x0 == 3 and jordan == 1344 and norm_lower == 576 and jordan-norm_lower == 768
    # Same fixed particle number and a genuinely negative occupied level.
    mu = -m
    dual = 2*mu+sum(max(Fraction(),t-mu) for t in levels)
    assert dual == canonical
    receipt = {'kind':'EXACT_LITERAL_NATIVE_ONSITE_POSITIVEPART_STRICTNESS_NOT_RECTANGULAR_GATE',
               'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'profile_sha256':expected,'m':str(m),'field0':phi,
               'canonical_wedge2':str(canonical),'positive_part_upper':str(positive),
               'canonical_identity_rejected':True,'negative_level_occupied':True,
               'same_mu_compensated_dual':str(dual),'x0_quarter':str(x0),
               'ordered_Jordan':jordan,'norm_lower':str(norm_lower),'Jordan_defect':str(jordan-norm_lower),
               'rectangular_gate_replaced':False,'science_admission':False}
    (root/'open_box_body_strict_onsite_20261012_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Literal native onsite canonical strict positive-part inequality and Jordan defect768 executed')


if __name__ == '__main__':
    main()
