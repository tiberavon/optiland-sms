"""
Python-specific module: rr system
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
RR SYSTEM -- the fixture's RR lens as a sequential Optiland system
===============================================================================
The profile's 2D plane is the system's meridional plane: profile x is the
system's y, and light travels down the profile's y axis (from the emitter at
y = 2 to the receiver at y = -2), so the surfaces are ``reverse``. Global z is
0 at the N surface's vertex. Each surface's aperture is its rim.
===============================================================================
"""

from pathlib import Path

import numpy as np
import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture
from optiland.rays import RealRays

from optiland_sms.meridian import load_profile, read_meridian

PROFILE_PATH = Path(__file__).parent / "fixtures" / "rr_example_9_1.profile.json"
PROFILE = load_profile(PROFILE_PATH)
EMITTER_Y, RECEIVER_Y = 2.0, -2.0
E1_X, R2_X = -1.0, 0.5
N = read_meridian(PROFILE, "N", reverse=True)
X = read_meridian(PROFILE, "X", reverse=True)
YN, YX = N.vertex_along_axis, X.vertex_along_axis


def build(profile=PROFILE) -> Optic:
    """The RR: emitter plane, N, X, receiver plane."""
    op = Optic()
    op.surfaces.add(index=0, thickness=EMITTER_Y - YN)
    op.surfaces.add(index=1, surface_type="sms_spline", profile=profile, span="N", reverse=True,
                    thickness=YN - YX, material=IdealMaterial(n=1.5), is_stop=True,
                    aperture=RadialAperture(N.rim_radius))
    op.surfaces.add(index=2, surface_type="sms_spline", profile=profile, span="X", reverse=True,
                    thickness=YX - RECEIVER_Y, aperture=RadialAperture(X.rim_radius))
    op.surfaces.add(index=3)
    return op


def e1_fan(n: int = 21, aim_from: float = -0.6, aim_to: float = 0.45) -> RealRays:
    """Meridional rays from E1, aimed at points of the N vertex plane, all inside both rims."""
    aim = np.linspace(aim_from, aim_to, n)
    d = np.stack([np.zeros(n), aim - E1_X, np.full(n, EMITTER_Y - YN)])
    d /= np.linalg.norm(d, axis=0)
    values = (np.zeros(n), np.full(n, E1_X), np.full(n, YN - EMITTER_Y), d[0], d[1], d[2],
              np.ones(n), np.full(n, 0.55))
    return RealRays(*(be.array(v) for v in values))


def landings(op: Optic, rays: RealRays):
    """Where the rays meet the receiver plane (the system's y)."""
    op.surfaces.trace(rays)
    return rays.y


def edge_merit(op: Optic):
    """Mean squared distance of E1's rays from R2 on the receiver: the SMS condition, smooth."""
    rays = e1_fan()
    y = landings(op, rays)
    return ((y - R2_X) ** 2).mean()
