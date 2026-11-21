"""
Python-specific module: meridian
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

===============================================================================
MERIDIAN -- one spline span of a profile.json, in a surface's own frame
===============================================================================
A ``profile.json`` describes a 2D outline: points, exact outward normals, and
spans, each span one physical surface. A spline span is a cubic Hermite curve
through ``points`` with unit ``tangents``, each segment's tangents scaled by
its chord, over the parameter range ``domain`` (the segment index plus the
fraction along it). A revolve profile declares the axis it turns about.

``read_meridian`` takes one spline span and returns it in the frame of a
sequential surface of revolution:

- r: the distance from the revolve axis, on the side the half-section lies
  (the left of the axis direction);
- z: along the axis, from the span's vertex (where it meets the axis). With
  ``reverse`` it runs against the profile's axis direction, for light that
  travels that way.

The map is a rotation (and, with ``reverse``, a reflection), so the curve in
(r, z) is the same Hermite curve: its points and tangents are mapped, and its
chords are unchanged.

A surface is one span (one curve from the axis to its rim). The reader
refuses, with a ValueError that says why:

- a profile whose schema it does not know, or that is not a revolve;
- a label that names no span, or more than one;
- a span that is not a spline;
- a span with neither end on the axis (an annular zone: a surface made of
  more than one span);
- a span whose radius is not strictly monotonic along it, so that it is not
  single-valued as z(r).
===============================================================================
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

# Spline spans exist from schema 1.2. Later versions add fields this reader
# does not use (sources, rayfiles, aspheres, fit records).
KNOWN_SCHEMAS = ("1.2", "1.3", "1.4", "1.5")

# How close to the axis a span's end must be, relative to the span's size.
AXIS_TOL = 1e-9

XY = Tuple[float, float]
ProfileLike = Union[Dict[str, Any], str, Path]


@dataclass(frozen=True)
class Piece:
    """One cubic segment of the meridian, or the part of it inside the domain.

    Attributes:
        k: The spline segment's index (points k and k + 1).
        u_from: Where along the segment the piece starts, in [0, 1]: the end
            nearer the axis.
        u_to: Where it ends: the end nearer the rim.
    """

    k: int
    u_from: float
    u_to: float


@dataclass(frozen=True)
class Meridian:
    """A spline span in a surface's (r, z) frame.

    Attributes:
        points: The spline's points (r, z), all of them, z from the vertex.
        tangents: The unit tangents at the points, in the same frame.
        pieces: The parts of segments the domain covers, from the axis to
            the rim.
        samples: The profile's own samples of the span (r, z).
        normals: The profile's exact unit normals at the samples, in (r, z).
        label: The span's short label.
        reverse: Whether z runs against the profile's axis direction.
        vertex: The vertex (where the span meets the axis), in the profile's
            coordinates.
        vertex_along_axis: The vertex's distance from the axis point along
            the profile's axis direction (whatever ``reverse``). Two faces'
            values give the thickness between them: the vertex is the curve's,
            which lies between its points, not one of them.
    """

    points: Tuple[XY, ...]
    tangents: Tuple[XY, ...]
    pieces: Tuple[Piece, ...]
    samples: Tuple[XY, ...]
    normals: Tuple[XY, ...]
    label: str
    reverse: bool
    vertex: XY
    vertex_along_axis: float

    @property
    def rim_radius(self) -> float:
        """The radius of the span's end away from the axis."""
        last = self.pieces[-1]
        return hermite_point(self.points, self.tangents, last.k, last.u_to)[0]


def load_profile(profile: ProfileLike) -> Dict[str, Any]:
    """A profile as a dict: given as one, or read from a JSON file."""
    if isinstance(profile, dict):
        return profile
    with open(profile, encoding="utf-8") as fh:
        return json.load(fh)


def hermite_coefficients(points: Sequence[XY], tangents: Sequence[XY], k: int):
    """Segment k's cubic as (a, b, c, d) per coordinate: a + b u + c u^2 + d u^3.

    The same coefficients as the profile's writer: the tangents scaled by the
    chord between points k and k + 1.
    """
    p0, p1 = points[k], points[k + 1]
    h = math.dist(p0, p1)
    m0 = (tangents[k][0] * h, tangents[k][1] * h)
    m1 = (tangents[k + 1][0] * h, tangents[k + 1][1] * h)
    return tuple(
        (p0[i], m0[i], -3 * p0[i] - 2 * m0[i] + 3 * p1[i] - m1[i], 2 * p0[i] + m0[i] - 2 * p1[i] + m1[i])
        for i in (0, 1)
    )


def hermite_point(points: Sequence[XY], tangents: Sequence[XY], k: int, u: float) -> XY:
    """The point at fraction u of segment k."""
    out = []
    for a, b, c, d in hermite_coefficients(points, tangents, k):
        out.append(a + u * (b + u * (c + u * d)))
    return out[0], out[1]


def _span(profile: Dict[str, Any], label: str) -> Dict[str, Any]:
    found = [s for s in profile.get("spans", []) if s.get("short_label") == label]
    if len(found) != 1:
        labels = [s.get("short_label") for s in profile.get("spans", [])]
        raise ValueError(f"the profile has {len(found)} spans labelled {label!r}, not one; "
                         f"its labels are {labels}.")
    return found[0]


def _frame(profile: Dict[str, Any]) -> Tuple[XY, XY, XY]:
    symmetry = profile.get("symmetry") or {}
    if symmetry.get("kind") != "revolve":
        raise ValueError(f"not a revolve profile: symmetry kind is {symmetry.get('kind')!r}.")
    if symmetry.get("azimuth_deg") is not None:
        raise ValueError("a revolve over part of a turn (azimuth_deg) is not a sequential surface.")
    axis = symmetry.get("axis") or {}
    try:
        point = (float(axis["point"][0]), float(axis["point"][1]))
        ux, uy = float(axis["direction"][0]), float(axis["direction"][1])
    except (TypeError, KeyError, IndexError, ValueError):
        raise ValueError(f"symmetry needs axis = {{'point': [a, b], 'direction': [ux, uy]}}, "
                         f"got {axis!r}.") from None
    length = math.hypot(ux, uy)
    if length == 0.0:
        raise ValueError("the revolve axis direction is zero.")
    ux, uy = ux / length, uy / length
    return point, (ux, uy), (-uy, ux)


def _monotonic(points, tangents, piece: Piece) -> float:
    """The smallest of dr/du times its sign along a piece; > 0 if r is strictly monotonic."""
    (_, b, c, d), _ = hermite_coefficients(points, tangents, piece.k)
    lo, hi = sorted((piece.u_from, piece.u_to))
    us = [lo, hi]
    if d != 0.0:
        u_star = -c / (3.0 * d)  # the extremum of the quadratic dr/du
        if lo < u_star < hi:
            us.append(u_star)
    sign = 1.0 if piece.u_to > piece.u_from else -1.0
    return min(sign * (b + u * (2 * c + 3 * u * d)) for u in us)


def read_meridian(profile: ProfileLike, label: str, *, reverse: bool = False) -> Meridian:
    """One spline span of a revolve profile, in a sequential surface's (r, z) frame.

    Args:
        profile: The profile, as a dict or the path of its JSON file.
        label: The span's ``short_label``.
        reverse: z against the profile's axis direction (light travelling that way).

    Returns:
        The Meridian, its z measured from the vertex.

    Raises:
        ValueError: For anything the module docstring lists.
    """
    profile = load_profile(profile)
    version = str(profile.get("schema_version"))
    if version not in KNOWN_SCHEMAS:
        raise ValueError(f"profile schema {version!r} is not one this reader knows {KNOWN_SCHEMAS}.")
    point, direction, left = _frame(profile)
    span = _span(profile, label)
    curve = span.get("curve") or {}
    if curve.get("type") != "spline":
        raise ValueError(f"span {label!r} is a {curve.get('type')!r}, not a spline.")
    sign = -1.0 if reverse else 1.0

    def to_rz(p) -> XY:
        dx, dy = float(p[0]) - point[0], float(p[1]) - point[1]
        return dx * left[0] + dy * left[1], sign * (dx * direction[0] + dy * direction[1])

    def vec_to_rz(v) -> XY:
        x, y = float(v[0]), float(v[1])
        return x * left[0] + y * left[1], sign * (x * direction[0] + y * direction[1])

    points = [to_rz(p) for p in curve["points"]]
    tangents = []
    for t in curve["tangents"]:
        tr, tz = vec_to_rz(t)
        n = math.hypot(tr, tz)
        if n == 0.0 or not math.isfinite(n):
            raise ValueError(f"span {label!r} has a zero or non-finite tangent.")
        tangents.append((tr / n, tz / n))
    if len(points) < 2 or len(points) != len(tangents):
        raise ValueError(f"span {label!r} needs at least 2 points and one tangent per point.")
    last = len(points) - 1
    p0, p1 = (float(v) for v in curve.get("domain", (0.0, last)))
    if not 0.0 <= min(p0, p1) <= max(p0, p1) <= last or p0 == p1:
        raise ValueError(f"span {label!r} has a domain {[p0, p1]} outside [0, {last}].")

    def at(p: float) -> XY:
        k = min(int(math.floor(p)), last - 1)
        return hermite_point(points, tangents, k, p - k)

    size = max(math.dist(points[0], points[-1]), max(abs(c) for q in points for c in q), 1e-300)
    r0, r1 = at(p0)[0], at(p1)[0]
    if abs(r0) <= AXIS_TOL * size:
        start, end = p0, p1
    elif abs(r1) <= AXIS_TOL * size:
        start, end = p1, p0
    else:
        raise ValueError(f"span {label!r} does not meet the axis (r = {r0:.6g} and {r1:.6g} at its ends): "
                         "a surface made of more than one span is not supported.")
    # The pieces in increasing parameter, then ordered from the axis to the rim.
    lo, hi = min(start, end), max(start, end)
    pieces: List[Piece] = []
    for k in range(min(int(math.floor(lo)), last - 1), last):
        u_a, u_b = max(lo - k, 0.0), min(hi - k, 1.0)
        if u_b - u_a > 1e-15:
            pieces.append(Piece(k, u_a, u_b))
    if start > end:
        pieces = [Piece(q.k, q.u_to, q.u_from) for q in reversed(pieces)]
    for piece in pieces:
        if _monotonic(points, tangents, piece) <= 0.0:
            raise ValueError(f"span {label!r}: the radius is not strictly monotonic on segment "
                             f"{piece.k}, so the surface is not single-valued as z(r).")
    z_vertex = at(start)[1]
    vertex = (point[0] + sign * z_vertex * direction[0], point[1] + sign * z_vertex * direction[1])
    points = [(r, z - z_vertex) for r, z in points]
    k0, k1 = int(span["index_start"]), int(span["index_end"])
    samples = tuple((r, z - z_vertex) for r, z in (to_rz(q) for q in profile["points"][k0:k1]))
    normals = tuple(vec_to_rz(v) for v in profile["normals"][k0:k1])
    return Meridian(tuple(points), tuple(tangents), tuple(pieces), samples, normals, label, reverse,
                    vertex, sign * z_vertex)
