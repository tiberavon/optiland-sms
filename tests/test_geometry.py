"""
Python-specific module: test geometry
Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The surface: its shape against the profile, its trace on both backends, its
gradient against finite differences, and its registration by entry point.

The bounds are set just above what was measured on the fixture (measured in
brackets): the sag at the profile's samples (3e-16) and densely along the
curve (2e-16), the normal's angle (2.5e-14 rad), numpy against torch
(6e-12 on the landings with Optiland 0.6.3, 1e-16 with its later
development code: its solve stops when the residual is below its tolerance,
and the two backends may stop at different iterates), E1's rays from R2 (3.75e-3), and the gradient
against a central difference with step 1e-6 (1e-6 relative).
"""

import json
import math
import subprocess
import sys
import warnings

import numpy as np
import optiland.backend as be
import pytest
from optiland.coordinate_system import CoordinateSystem

from tests import rr_system as rr
from optiland_sms.geometry import SMSSplineGeometry
from optiland_sms.meridian import hermite_point


@pytest.fixture(autouse=True)
def numpy_backend():
    be.set_backend("numpy")
    yield
    be.set_backend("numpy")


def _geometry(label):
    return SMSSplineGeometry(CoordinateSystem(), rr.PROFILE, label, reverse=True)


@pytest.mark.parametrize("label", ["N", "X"])
def test_the_sag_and_normal_are_the_profiles_at_its_samples(label):
    g = _geometry(label)
    r = np.array([s[0] for s in g.meridian.samples])
    z = np.array([s[1] for s in g.meridian.samples])
    assert np.abs(g.sag(r, 0 * r) - z).max() < 1e-15
    nx, _, nz = g._surface_normal(r, 0 * r)
    exact = np.array(g.meridian.normals)
    angle = np.abs(np.arctan2(nx * exact[:, 1] - nz * exact[:, 0], nx * exact[:, 0] + nz * exact[:, 1]))
    assert np.minimum(angle, math.pi - angle).max() < 1e-13


@pytest.mark.parametrize("label", ["N", "X"])
def test_the_sag_is_the_profiles_curve_between_its_samples(label):
    g = _geometry(label)
    m = g.meridian
    pts = np.array([hermite_point(m.points, m.tangents, q.k, u)
                    for q in m.pieces for u in np.linspace(q.u_from, q.u_to, 101)])
    assert np.abs(g.sag(pts[:, 0], 0 * pts[:, 0]) - pts[:, 1]).max() < 1e-15


def test_the_surface_is_smooth_at_the_rim_and_beyond_it():
    g = _geometry("N")
    rim = g.rim_radius
    r = np.array([rim - 1e-7, rim, rim + 1e-7])
    z = g.sag(r, 0 * r)
    assert abs((z[2] - z[1]) - (z[1] - z[0])) < 1e-12


def test_a_trace_takes_e1s_rays_to_r2():
    y = np.asarray(rr.landings(rr.build(), rr.e1_fan()))
    assert np.isfinite(y).all()
    assert np.abs(y - rr.R2_X).max() < 5e-3


def test_numpy_and_torch_trace_the_same():
    torch = pytest.importorskip("torch")  # noqa: F841
    y_np = np.asarray(rr.landings(rr.build(), rr.e1_fan()))
    be.set_backend("torch")
    be.set_precision("float64")
    y_t = be.to_numpy(rr.landings(rr.build(), rr.e1_fan()))
    assert np.abs(y_np - y_t).max() < 1e-11


def test_the_gradient_with_respect_to_a_point_is_the_finite_difference():
    torch = pytest.importorskip("torch")
    be.set_backend("torch")
    be.set_precision("float64")
    be.grad_mode.enable()
    op = rr.build()
    g = op.surfaces.surfaces[1].geometry
    g.points = g.points.detach().clone().requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no ray rejected from the gradient
        rr.edge_merit(op).backward()
    k, h = 30, 1e-6
    for c in (0, 1):
        values = []
        for s in (1.0, -1.0):
            with torch.no_grad():
                op2 = rr.build()
                g2 = op2.surfaces.surfaces[1].geometry
                p = g2.points.clone()
                p[k, c] += s * h
                g2.points = p
                values.append(float(rr.edge_merit(op2)))
        fd = (values[0] - values[1]) / (2 * h)
        auto = float(g.points.grad[k, c])
        assert abs(auto - fd) < 1e-5 * abs(fd), (c, auto, fd)


def test_flip_turns_the_sag_round():
    g = _geometry("N")
    r = np.linspace(0.0, g.rim_radius, 7)
    before = g.sag(r, 0 * r)
    g.flip()
    assert np.abs(g.sag(r, 0 * r) + before).max() < 1e-15


def test_a_dict_round_trip_keeps_the_surface():
    g = _geometry("X")
    h = SMSSplineGeometry.from_dict(json.loads(json.dumps(g.to_dict())))
    r = np.linspace(0.0, g.rim_radius, 7)
    assert np.array_equal(g.sag(r, 0 * r), h.sag(r, 0 * r))


def test_optiland_finds_the_surface_type_by_entry_point():
    """In a fresh process, with no import of optiland_sms."""
    code = (
        "import sys\n"
        "from optiland.coordinate_system import CoordinateSystem\n"
        "from optiland.surfaces.factories.geometry_factory import GeometryFactory\n"
        "assert 'optiland_sms' not in sys.modules\n"
        f"g = GeometryFactory.create('sms_spline', CoordinateSystem(), profile={str(rr.PROFILE_PATH)!r}, "
        "span='N', reverse=True)\n"
        "print(type(g).__name__)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "SMSSplineGeometry"
