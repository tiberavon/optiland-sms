"""
Python-specific module: test meridian
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

The reader: a spline span in a surface's frame, and what it refuses."""

import copy

import pytest

from optiland_sms.meridian import read_meridian
from tests.rr_system import PROFILE


def test_both_faces_run_from_the_axis_to_the_rim_with_the_curves_own_vertex():
    for label in ("N", "X"):
        m = read_meridian(PROFILE, label, reverse=True)
        first, last = m.pieces[0], m.pieces[-1]
        assert first.u_from == 0.5  # the axis cuts the middle segment
        assert m.samples[0][0] == 0.0 or m.samples[-1][0] == 0.0
        # The vertex is the curve's, which the profile samples on the axis.
        on_axis = [p for p in PROFILE["points"] if p[0] == 0.0]
        assert any(abs(m.vertex[1] - p[1]) < 1e-15 for p in on_axis)
        assert m.rim_radius > 0.6
        assert last.u_to in (0.0, 1.0)


def test_reverse_changes_only_the_sign_of_z():
    a = read_meridian(PROFILE, "N", reverse=False)
    b = read_meridian(PROFILE, "N", reverse=True)
    assert all(pa[0] == pb[0] and pa[1] == -pb[1] for pa, pb in zip(a.points, b.points))
    assert a.vertex == b.vertex and a.vertex_along_axis == b.vertex_along_axis


def _with(change):
    profile = copy.deepcopy(PROFILE)
    change(profile)
    return profile


@pytest.mark.parametrize("label, change, message", [
    ("N", lambda p: p.update(schema_version="9.9"), "schema"),
    ("N", lambda p: p.update(symmetry={"kind": "extrude"}), "not a revolve"),
    ("N", lambda p: p["symmetry"].update(azimuth_deg=[-90, 90]), "part of a turn"),
    ("Q", lambda p: None, "0 spans labelled 'Q'"),
    ("RIM", lambda p: None, "not a spline"),
    ("N", lambda p: p["spans"][0]["curve"].update(domain=[30.0, 51.0]), "more than one span"),
])
def test_what_the_reader_refuses(label, change, message):
    with pytest.raises(ValueError, match=message):
        read_meridian(_with(change), label)


def test_a_meridian_whose_radius_turns_back_is_refused():
    profile = {
        "schema_version": "1.5",
        "symmetry": {"kind": "revolve", "axis": {"point": [0.0, 0.0], "direction": [0.0, 1.0]}},
        "points": [], "normals": [],
        "spans": [{"index_start": 0, "index_end": 0, "short_label": "S",
                   "curve": {"type": "spline", "points": [[0.0, 0.0], [-1.0, 0.2], [-0.6, 0.6]],
                             "tangents": [[-1.0, 0.0], [-0.6, 0.8], [0.6, 0.8]]}}],
    }
    with pytest.raises(ValueError, match="not strictly monotonic"):
        read_meridian(profile, "S")
