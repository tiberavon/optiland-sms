"""
Python-specific module: geometry
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
GEOMETRY -- a spline meridian as an Optiland sequential surface
===============================================================================
``SMSSplineGeometry`` is a surface of revolution about the local z axis whose
meridian is a profile's spline span (``meridian.read_meridian``). It is an
Optiland ``NewtonRaphsonGeometry``: it gives the sag and the normal, and
Optiland's own solve finds where a ray meets it. Under the torch backend that
solve gives first-order gradients through the converged intersection (its
one-step implicit correction).

The sag is the profile's own curve, not a second curve through its points.
At a radius r:

1. the piece of the meridian whose radius range holds r is found;
2. the piece's parameter u where r(u) = r is found by Newton's method from a
   linear guess, detached from the gradient; the radius is strictly monotonic
   on every piece (the reader checks it), so the root is unique;
3. one more Newton step, attached, makes u's derivative the exact first-order
   one (the implicit function theorem), with respect to r and to the points;
4. the sag is z(u).

The normal comes from the meridian's tangent at u: dz/dr = z'(u) / r'(u).
Beyond the rim the sag goes on along the rim's tangent, so a Newton iterate
that strays past the rim sees a smooth surface; the aperture limits the rays.

The spline's points are held as a backend array (``self.points``): under
torch it can be a leaf that requires a gradient, and a trace is then
differentiable with respect to the points. The tangents stay fixed.
===============================================================================
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry

from .meridian import Meridian, ProfileLike, hermite_point, load_profile, read_meridian

# Newton steps on r(u) = r before the attached one. The guess is linear on a
# piece of a cubic, and the convergence quadratic: six reach rounding.
DETACHED_STEPS = 6


def _detach(x):
    return x.detach() if hasattr(x, "detach") else x


class SMSSplineGeometry(NewtonRaphsonGeometry):
    """A surface of revolution whose meridian is a profile.json spline span.

    Args:
        coordinate_system: The surface's coordinate system; its vertex is
            where the span meets the axis.
        profile: The profile, as a dict or the path of its JSON file.
        span: The span's short label.
        reverse: Whether light travels against the profile's axis direction,
            so local +z runs against it.
        tol: Tolerance of Optiland's intersection solve.
        max_iter: Iteration limit of that solve.

    Attributes:
        meridian: The span as read (``meridian.Meridian``).
        points: The spline's points (n, 2) as (r, z), a backend array.
        tangents: The unit tangents (n, 2), a backend array.
        rim_radius: The radius of the span's end away from the axis.
    """

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        profile: ProfileLike,
        span: str,
        reverse: bool = False,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> None:
        self.profile = load_profile(profile)
        self.span = span
        self.reverse = bool(reverse)
        self.meridian: Meridian = read_meridian(self.profile, span, reverse=self.reverse)
        self.points = be.array(np.array(self.meridian.points, dtype=float))
        self.tangents = be.array(np.array(self.meridian.tangents, dtype=float))
        pieces = self.meridian.pieces
        self._k = np.array([q.k for q in pieces], dtype=np.int64)
        self._u_from = np.array([q.u_from for q in pieces], dtype=float)
        self._u_to = np.array([q.u_to for q in pieces], dtype=float)
        self.rim_radius = float(self.meridian.rim_radius)
        super().__init__(coordinate_system, self._seed_radius(), 0.0, tol, max_iter)
        self.is_symmetric = True

    def __str__(self) -> str:
        return "SMS Spline"

    # --- the meridian, on the backend ------------------------------------------

    def _coefficients(self):
        """Each piece's cubic in r and in z: four (pieces,) arrays each."""
        p, t = self.points, self.tangents
        k = self._k
        p0, p1 = p[k], p[k + 1]
        h = be.sqrt(be.sum((p1 - p0) ** 2, axis=1))
        m0, m1 = t[k] * h[:, None], t[k + 1] * h[:, None]
        a = p0
        b = m0
        c = -3 * p0 - 2 * m0 + 3 * p1 - m1
        d = 2 * p0 + m0 - 2 * p1 + m1
        return (a[:, 0], b[:, 0], c[:, 0], d[:, 0]), (a[:, 1], b[:, 1], c[:, 1], d[:, 1])

    @staticmethod
    def _cubic(co, u):
        a, b, c, d = co
        return a + u * (b + u * (c + u * d))

    @staticmethod
    def _cubic_d(co, u):
        _, b, c, d = co
        return b + u * (2 * c + 3 * u * d)

    def _knots(self, cr):
        """The radius at each piece's two ends, detached: (pieces,) arrays."""
        u_from, u_to = be.array(self._u_from), be.array(self._u_to)
        cr_d = tuple(_detach(x) for x in cr)
        return self._cubic(cr_d, u_from), self._cubic(cr_d, u_to)

    def _solve(self, r):
        """For each radius: the piece, the cubic's coefficients there, and u."""
        cr, cz = self._coefficients()
        r_from, r_to = self._knots(cr)
        r_d = _detach(r)
        n = len(self._k)
        idx = be.searchsorted(r_to, r_d)
        idx = be.clip(idx, 0, n - 1)
        u_from, u_to = be.array(self._u_from)[idx], be.array(self._u_to)[idx]
        ra, rb = r_from[idx], r_to[idx]
        crs = tuple(x[idx] for x in cr)
        czs = tuple(x[idx] for x in cz)
        crs_d = tuple(_detach(x) for x in crs)
        # The radius inside the piece (beyond the rim, the rim: the sag is
        # extended along the rim's tangent below).
        r_in = be.clip(r_d, 0.0, self.rim_radius)
        frac = (r_in - ra) / (rb - ra)
        u = u_from + frac * (u_to - u_from)
        lo, hi = be.minimum(u_from, u_to), be.maximum(u_from, u_to)
        for _ in range(DETACHED_STEPS):
            u = u - (self._cubic(crs_d, u) - r_in) / self._cubic_d(crs_d, u)
            u = be.minimum(be.maximum(u, lo), hi)
        u = _detach(u)
        # The attached step: u's first derivative is the implicit one.
        r_att = be.where(r_d > self.rim_radius, be.zeros_like(r) + self.rim_radius, r)
        u = u - (self._cubic(crs, u) - r_att) / self._cubic_d(crs, u)
        return crs, czs, u

    def _radius(self, x, y):
        # A tiny floor keeps the derivative of the radius finite on the axis,
        # where the meridian is flat and the sag's derivative is zero anyway.
        return be.sqrt(x**2 + y**2 + 1e-300)

    def _sag_and_slope(self, x, y):
        x, y = be.array(x), be.array(y)
        r = self._radius(x, y)
        shape = r.shape
        r = be.ravel(r)
        crs, czs, u = self._solve(r)
        z = self._cubic(czs, u)
        slope = self._cubic_d(czs, u) / self._cubic_d(crs, u)
        beyond = r > self.rim_radius
        z = be.where(beyond, z + slope * (r - self.rim_radius), z)
        return be.reshape(z, shape), be.reshape(slope, shape), be.reshape(r, shape)

    # --- NewtonRaphsonGeometry --------------------------------------------------

    def sag(self, x=0, y=0):
        """The sag at (x, y): the meridian's z at the radius hypot(x, y).

        Args:
            x: The x-coordinate(s).
            y: The y-coordinate(s).

        Returns:
            The sag, the shape of x and y.
        """
        return self._sag_and_slope(x, y)[0]

    def _surface_normal(self, x, y):
        """The unit normal (nx, ny, nz) at (x, y), with nz < 0 (Optiland's convention)."""
        _, slope, r = self._sag_and_slope(x, y)
        x, y = be.array(x), be.array(y)
        dfdx, dfdy = slope * x / r, slope * y / r
        mag = be.sqrt(dfdx**2 + dfdy**2 + 1)
        return dfdx / mag, dfdy / mag, -1 / mag

    def _seed_radius(self) -> float:
        """The radius of Optiland's seed sphere: the sphere through the vertex and the rim.

        Optiland starts its intersection solve where a ray meets the base
        sphere. A ray the sphere does not reach has no start, and some
        Optiland releases leave it non-finite. The sphere through the vertex
        and the rim point reaches the whole aperture (its radius is at least
        the rim's), which the vertex's own radius of curvature need not: the
        RR's front face curves more at its vertex than its rim radius allows.
        Infinite (a plane) for a flat meridian.
        """
        last = self.meridian.pieces[-1]
        r, z = hermite_point(self.meridian.points, self.meridian.tangents, last.k, last.u_to)
        if abs(z) < 1e-12 * max(r, 1e-300):
            return float("inf")
        return (r * r + z * z) / (2.0 * z)

    # --- Optiland's other hooks -------------------------------------------------

    def flip(self) -> None:
        """Turn the surface round: light now travels the other way along the axis."""
        self.__init__(self.cs, self.profile, self.span, not self.reverse, self.tol, self.max_iter)

    def scale(self, scale_factor: float) -> None:
        """Not supported: a profile's coordinates are its design's."""
        raise NotImplementedError("an SMS spline surface is not scaled; scale the profile instead.")

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({"profile": self.profile, "span": self.span, "reverse": self.reverse})
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SMSSplineGeometry":
        missing = {"cs", "profile", "span"} - data.keys()
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
        return cls(CoordinateSystem.from_dict(data["cs"]), data["profile"], data["span"],
                   data.get("reverse", False), data.get("tol", 1e-10), data.get("max_iter", 100))
