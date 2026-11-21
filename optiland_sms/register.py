"""
Python-specific module: register
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
REGISTER -- the entry point Optiland's plugin loader calls
===============================================================================
``register()`` adds the surface type ``"sms_spline"`` to Optiland's
``GeometryFactory``. The package declares it under the ``optiland.surfaces``
entry-point group, so Optiland calls it the first time it meets a surface
type it does not know; nothing has to import this package first.

A surface is then added as any other::

    optic.surfaces.add(index=1, surface_type="sms_spline", profile="lens.profile.json",
                       span="N", reverse=True, thickness=0.8, material=...)

``profile`` is the profile (a dict, or the path of its JSON file), ``span``
the spline span's short label, ``reverse`` whether light travels against the
profile's axis direction.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

SURFACE_TYPE = "sms_spline"


@dataclass
class SMSSplineConfig:
    """The arguments ``surface_type="sms_spline"`` accepts.

    Attributes:
        profile: The profile, as a dict or the path of its JSON file.
        span: The spline span's short label.
        reverse: Light travels against the profile's axis direction.
        tol: Tolerance of Optiland's intersection solve.
        max_iter: Iteration limit of that solve.
    """

    profile: Union[Dict[str, Any], str, None] = None
    span: str = ""
    reverse: bool = False
    tol: float = 1e-10
    max_iter: int = 100


def _create(cs: Any, config: SMSSplineConfig):
    from .geometry import SMSSplineGeometry  # noqa: PLC0415

    if config.profile is None or not config.span:
        raise ValueError('surface_type="sms_spline" needs profile= and span=.')
    return SMSSplineGeometry(cs, config.profile, config.span, config.reverse, config.tol, config.max_iter)


def register() -> None:
    """Register ``"sms_spline"`` with Optiland's GeometryFactory; calling it again does nothing."""
    from optiland.surfaces.factories.geometry_factory import GeometryFactory  # noqa: PLC0415
    from optiland.surfaces.factories.geometry_factory import geometry_mapper  # noqa: PLC0415

    if SURFACE_TYPE in geometry_mapper:
        return
    GeometryFactory.register(SURFACE_TYPE, _create, SMSSplineConfig)
