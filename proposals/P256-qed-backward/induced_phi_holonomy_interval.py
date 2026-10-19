#!/usr/bin/env python3
"""Independent outward arithmetic of frozen all64 holonomy and Gauss comparison.

No old receipt is repaired in place. The classical uniform subset and the
already specified physical H2 competitor remain distinct from a global vacuum.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any
import mpmath as mp
from mpmath.ctx_iv import MPIntervalContext
from induced_phi_centre_holonomy import (SOURCE_BYTES, SYMMETRIC, PhiConfig,
    models, scalar_paths, periodic_grid, class_transform)

ROOT = Path(__file__).resolve().parent
IMPORT_BYTES = {**SOURCE_BYTES, Path(__file__).name: Path(__file__).read_bytes()}
IMPORT_HASHES = {name: hashlib.sha256(data).hexdigest() for name, data in IMPORT_BYTES.items()}


def endpoints(value: Any) -> tuple[Fraction, Fraction]:
    values = []
    for sign, mantissa, exponent, _ in value._mpi_:
        number = Fraction((-1 if sign else 1)*int(mantissa))
        number *= Fraction(2)**int(exponent)
        values.append(number)
    return values[0], values[1]


def encoded(value: Fraction) -> dict[str, str]:
    with mp.workdps(80):
        display = str(mp.nstr(mp.mpf(value.numerator)/value.denominator, 60))
    return dict(numerator=str(value.numerator), denominator=str(value.denominator), decimal_display=display)


def interval_encoded(value: Any) -> dict[str, Any]:
    lower, upper = endpoints(value)
    return dict(lower=encoded(lower), upper=encoded(upper), width=encoded(upper-lower))


def exact(context: Any, value: Fraction | float | int | str) -> Any:
    rational = Fraction(value)
    return context.mpf(rational.numerator)/rational.denominator


class Arithmetic:
    def __init__(self, cfg: PhiConfig, size: int, exponent: int, digits: int):
        self.context: Any = MPIntervalContext()
        self.context.dps = digits
        self.cfg, self.size, self.exponent = cfg, size, exponent
        self.stiffness = Fraction(2)/Fraction(cfg.a)
        self.profiles = models(replace(cfg, Lambda=1.))
        self.trig: dict[tuple[int, int, int], tuple[Any, Any]] = {}
        self.rows_cache: dict[tuple[int, tuple[int, ...]], list[list[Any]]] = {}

    def rows(self, profile: int, twist: tuple[int, ...]) -> list[list[Any]]:
        key = profile, twist
        if key in self.rows_cache:
            return self.rows_cache[key]
        c = self.context
        onsite, paths = scalar_paths(self.profiles[profile][1])
        output = []
        for site in periodic_grid(self.size):
            row = [exact(c, float(onsite)), c.mpf(0), c.mpf(0), c.mpf(0)]
            for displacement, scalars in paths:
                axis = next(j for j in range(3) if displacement[j])
                trig_key = int(site[axis]), twist[axis], int(displacement[axis])
                if trig_key not in self.trig:
                    n, t, length = trig_key
                    angle = 2*c.pi*(n+exact(c, Fraction(t, 4)))*length/self.size
                    self.trig[trig_key] = c.cos(angle), c.sin(angle)
                cosine, sine = self.trig[trig_key]
                for mu in range(4):
                    row[mu] += 2*(exact(c, float(scalars[mu].real))*cosine-exact(c, float(scalars[mu].imag))*sine)
            output.append(row)
        self.rows_cache[key] = output
        return output

    def evaluate(self, gram: list[list[Fraction]], twist: tuple[int, ...], heavy_only: bool = False) -> dict[str, Any]:
        c = self.context
        floor = min(gram[i][i]-sum(abs(gram[i][j]) for j in range(4) if j != i) for i in range(4))
        if floor <= 0:
            raise ArithmeticError('Literal stored Gram lacks a positive Gershgorin certificate')
        g = [exact(c, gram[i][j])*(1 if i == j else c.sqrt(2)) for i, j in SYMMETRIC]
        value = exact(c, self.stiffness)*sum(z*z for z in g)/2
        derivative = [exact(c, self.stiffness)*z for z in g]
        for index, (load, _, role) in enumerate(self.profiles):
            if heavy_only and not role.startswith('neutral'):
                continue
            weight = Fraction(1) if role.startswith('neutral') else Fraction(load)*Fraction(2)**(-self.exponent)
            for row in self.rows(index, twist):
                features = [row[i]*row[j]*(1 if i == j else c.sqrt(2)) for i, j in SYMMETRIC]
                squared = sum(z*x for z, x in zip(g, features))
                if endpoints(squared)[0] <= 0:
                    raise ArithmeticError('Outward band radius is not strictly positive')
                radius = c.sqrt(squared)
                value -= 2*exact(c, weight)*radius/self.size**3
                first = exact(c, weight)/(self.size**3*radius)
                for j in range(10):
                    derivative[j] -= first*features[j]
        norm_squared = sum(max(abs(a), abs(b))**2 for a, b in map(endpoints, derivative))
        error = norm_squared/(2*self.stiffness)
        return dict(value=value, gradient=derivative, gradient_squared_upper=norm_squared,
            optimizer_error_upper=error, minimum_Gram_eigenvalue_lower=floor)

    def path_cap(self, profile: int) -> Any:
        c = self.context
        onsite, paths = scalar_paths(self.profiles[profile][1])
        cap = abs(exact(c, float(onsite)))
        for _, scalars in paths:
            for scalar in scalars:
                real, imag = exact(c, float(scalar.real)), exact(c, float(scalar.imag))
                cap += 2*c.sqrt(real*real+imag*imag)
        return cap

    def physical_price(self, gram: list[list[Fraction]], periodic_gram: list[list[Fraction]], eta: str) -> dict[str, Any]:
        c = self.context
        volume = self.size**3
        dimension = 90*volume
        scale = c.mpf(2)**self.exponent
        chi = 1/c.sqrt(scale)
        width = c.sqrt(c.sqrt(1/scale))
        radius = c.sqrt(dimension)*width
        rmin_fraction = min(gram[i][i]-sum(abs(gram[i][j]) for j in range(4) if j != i) for i in range(4))/2
        rmin = exact(c, rmin_fraction)
        trace = exact(c, sum(gram[i][i] for i in range(4)))
        rho = c.sqrt(trace)
        analytic_fraction = min(Fraction(1, 1000), endpoints(rmin/(16*c.sqrt(trace/2)))[0])
        analytic = exact(c, analytic_fraction)
        if endpoints(radius)[1] >= analytic_fraction/2:
            raise ArithmeticError('Outward H2 support does not fit the joint physical chart')
        metric = 4+512/rmin
        first, log_first, log_second = 8*metric/analytic, 8/analytic, 32/analytic**2
        measure = chi*(dimension**2*(first*log_first+metric*log_second)/4+dimension*metric*log_first**2/8)
        charge_squared = exact(c, Fraction(volume*(3+self.cfg.auxiliary_count)**2*5, 2))
        light = sum(exact(c, load)*self.path_cap(index) for index, (load, _, role) in enumerate(self.profiles) if not role.startswith('neutral'))
        components = dict(kinetic=chi*metric*3*dimension/width**2+4*chi*charge_squared/rmin+measure,
            heavy_change=32*scale*volume*self.path_cap(0)*(1+rho)*radius,
            quartic_change=4*scale*volume/exact(c, self.cfg.a)*(rho+radius)**3*radius,
            alignment_curl=128*volume*scale*c.sqrt(c.sqrt(scale))*(1+rho)**2*radius**2,
            electric=exact(c, self.cfg.inertia)/exact(c, self.cfg.a)*3*(2*volume+1)*64,
            coulomb=exact(c, self.cfg.inertia)/exact(c, self.cfg.a)*volume*(2*(1+self.cfg.auxiliary_count)+4)**2/(4*c.sin(c.pi/self.size)**2),
            light_change=16*volume*light*(1+rho+radius))
        # Recompute the actual heavy+quartic force, not the rounded old price.
        # At a Clifford mean every traceless colour-link first derivative is
        # identically zero; the site force is the uniform Cartesian gradient.
        periodic = self.evaluate(periodic_gram, (0, 0, 0), heavy_only=True)
        periodic_rho = c.sqrt(exact(c, sum(periodic_gram[i][i] for i in range(4))))
        force = 2*scale*volume*periodic_rho*c.sqrt(exact(c, periodic['gradient_squared_upper']))*exact(c, eta)
        lower_penalty = force+8*volume*light*(1+periodic_rho+exact(c, eta))
        return dict(normalized_upper_extra=sum(components.values())/scale,
            normalized_periodic_lower_penalty=lower_penalty/scale,
            components={key: interval_encoded(value/scale) for key, value in components.items()},
            support_radius=interval_encoded(radius), analytic_radius_lower=encoded(analytic_fraction),
            physical_state='The frozen literal737digit H2/fullGauss competitor, not an eigenstate or photon vacuum',
            periodic_lower='Accepted periodic local-well Hessian/convexity price; independent outward heavy+quartic uniform force, exact zero traceless-link first derivative, and retained absolute light path bound')


def run(holonomy_path: str, periodic_path: str, digits: int) -> dict[str, Any]:
    start = time.monotonic()
    holonomy_bytes, periodic_bytes = Path(holonomy_path).read_bytes(), Path(periodic_path).read_bytes()
    holonomy, periodic = json.loads(holonomy_bytes), json.loads(periodic_bytes)
    if hashlib.sha256(periodic_bytes).hexdigest() != holonomy['input_descriptor']['periodic_receipt_sha256']:
        raise ValueError('The checked periodic packet is not the frozen holonomy input')
    descriptor = dict(holonomy_receipt_sha256=hashlib.sha256(holonomy_bytes).hexdigest(),
        periodic_receipt_sha256=hashlib.sha256(periodic_bytes).hexdigest(),
        holonomy_input_sha256=holonomy['input_sha256'], working_digits=digits,
        governing_plan=dict(id='issue-222-induced-photon-delta13-continuation-20261008',
            revision='d13d7d28ebfd7cb8386e7442897c596481aeab95cd55def1a8b3a60c1e89f4dd',
            work_item='joined-explanatory-campaign'),
        microscopic_input=holonomy['input_descriptor'], scope='Independent exact-rational/outward continuation; no changes to either frozen receipt')
    cfg = PhiConfig(**holonomy['input_descriptor']['config'])
    size, exponent = holonomy['input_descriptor']['size'], holonomy['input_descriptor']['Lambda_exponent']
    arithmetic = Arithmetic(cfg, size, exponent, digits)
    classes = {tuple(row['twist']): [[Fraction(z) for z in line] for line in row['G_decimal']] for row in holonomy['classes']}
    sectors = []
    for twist in itertools.product(range(4), repeat=3):
        representative, transform = class_transform(twist)
        integer = [[int(transform[i, j]) for j in range(4)] for i in range(4)]
        old = classes[representative]
        gram = [[sum((integer[i][k]*old[k][l]*integer[j][l] for k in range(4) for l in range(4)), Fraction(0)) for j in range(4)] for i in range(4)]
        result = arithmetic.evaluate(gram, twist)
        lower, upper = endpoints(result['value'])
        sectors.append(dict(twist=list(twist), representative=list(representative),
            minimum_lower=lower-result['optimizer_error_upper'], minimum_upper=upper,
            gradient_squared_upper=result['gradient_squared_upper'], optimizer_error_upper=result['optimizer_error_upper'],
            minimum_Gram_eigenvalue_lower=result['minimum_Gram_eigenvalue_lower'], energy=result['value']))
    preferred = tuple(holonomy['preferred_twist'])
    selected = next(row for row in sectors if tuple(row['twist']) == preferred)
    selection_margin = min(row['minimum_lower']-selected['minimum_upper'] for row in sectors if tuple(row['twist']) != preferred)
    if selection_margin <= 0:
        raise ArithmeticError('Outward all64 uniform class selection is not strict')
    preferred_gram = [[Fraction(z) for z in line] for line in holonomy['preferred_quantum_mean']['G_decimal']]
    periodic_mean = periodic['quantum_mean']
    periodic_gram = [[Fraction(periodic_mean['g0_decimal'] if i == 0 else periodic_mean['gs_decimal']) if i == j else Fraction(0) for j in range(4)] for i in range(4)]
    preferred_value = arithmetic.evaluate(preferred_gram, preferred)['value']
    periodic_value = arithmetic.evaluate(periodic_gram, (0, 0, 0))['value']
    price = arithmetic.physical_price(preferred_gram, periodic_gram, periodic['prices']['eta'])
    difference = size**3*(preferred_value-periodic_value)
    margin = -(difference+price['normalized_upper_extra']+price['normalized_periodic_lower_penalty'])
    if endpoints(margin)[0] <= 0:
        raise ArithmeticError('Outward physical Gauss competitor comparison is not strict')
    price['normalized_upper_extra'] = interval_encoded(price['normalized_upper_extra'])
    price['normalized_periodic_lower_penalty'] = interval_encoded(price['normalized_periodic_lower_penalty'])
    output_sectors = []
    for row in sectors:
        output_sectors.append({key: interval_encoded(value) if key == 'energy' else encoded(value) if isinstance(value, Fraction) else value for key, value in row.items()})
    return dict(branch='NEW rigorous outward arithmetic continuation of actual finiteN6 all64 seam scan',
        input_descriptor=descriptor, input_sha256=hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(),
        preferred_twist=list(preferred), all64=output_sectors,
        strict_uniform_sector_selection_margin=encoded(selection_margin),
        literal_physical_mean_difference_per_Lambda_box=interval_encoded(difference),
        outward_physical_price=price, strict_physical_Gauss_competitor_margin_per_Lambda=interval_encoded(margin),
        method='Independent MPIntervalContext sqrt/sin/cos arithmetic; exact binary64 primitive coefficients, exact literal decimal centres, exact rational endpoints. Convex optimizer error is actual gradient bound squared/(2kappa), not a guessed rounding allowance.',
        limitations='Uniform-subset holonomy selection and lower physical competitor, not global vacuum, preferred-sector clock, bulk stationarity, photon or Compton transfer',
        source_at_import=IMPORT_HASHES, own_source_at_import=IMPORT_BYTES[Path(__file__).name].decode(),
        all_sources_unchanged_at_completion=all((ROOT/name).read_bytes() == data for name, data in IMPORT_BYTES.items()),
        runtime=dict(python=sys.version, executable=sys.executable, mpmath=mp.__version__, platform=platform.platform()),
        elapsed_seconds=time.monotonic()-start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--holonomy', required=True)
    parser.add_argument('--periodic', required=True)
    parser.add_argument('--digits', type=int, default=150)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run(args.holonomy, args.periodic, args.digits)
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('preferred_twist', 'strict_uniform_sector_selection_margin', 'strict_physical_Gauss_competitor_margin_per_Lambda', 'source_at_import', 'input_sha256', 'elapsed_seconds')}, indent=2))
