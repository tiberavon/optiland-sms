"""
Python-specific module: optiland_sms package
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

Python translation Copyright 2026 ray-tracing-shapely authors and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

===============================================================================
optiland_sms
===============================================================================
Registers ray_tracing_shapely's SMS (Minano-Benitez Simultaneous Multiple
Surface) lens surfaces with Optiland's sequential engine, as the surface type
"sms_spline", through Optiland's ``optiland.surfaces`` entry point: no import
of this package and no edit to Optiland is needed.

The input is a ``profile.json``: one spline span of a revolve profile becomes
a surface of revolution whose meridian is the profile's own curve
(``geometry.SMSSplineGeometry``). Under Optiland's torch backend a trace
through it is differentiable with respect to the spline's points.

What it is for: refraction-only lenses (an RR) in the sequential engine. A
mirror or a total internal reflection does not fit a sequential trace, and
Optiland's non-sequential engine has no plugin hook: those designs reach 3D
through the ray_tracing_shapely_optiland scene bridge instead.

Modules:
- meridian: one spline span, read and checked, in a surface's (r, z) frame.
- geometry: the surface, an Optiland NewtonRaphsonGeometry.
- register: the entry point.

Named for Optiland's audience (entry-point discovery is a channel into
Optiland's own users). See dev_doc/architecture/components/optiland_sms_plugin.md.
===============================================================================
"""

from .geometry import SMSSplineGeometry
from .meridian import Meridian, read_meridian
from .register import SURFACE_TYPE, register

__version__ = "0.1.0"

__all__ = ["SMSSplineGeometry", "Meridian", "read_meridian", "register", "SURFACE_TYPE", "__version__"]
