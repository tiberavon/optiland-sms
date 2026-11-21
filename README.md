# optiland-sms

Miñano–Benítez SMS (Simultaneous Multiple Surface) lens surfaces in
[Optiland](https://github.com/optiland/optiland)'s sequential engine: a spline span of a
[`ray_tracing_shapely`](https://github.com/tiberavon/ray-tracing-shapely) `profile.json` becomes a
surface of revolution, traced exactly and, under Optiland's torch backend, differentiable with
respect to the spline's points.

The surface type `sms_spline` is registered through Optiland's `optiland.surfaces` entry point:
once the package is installed, Optiland finds it without an import and without any change to
Optiland.

## Install

```bash
pip install optiland-sms            # numpy only
pip install "optiland-sms[torch]"   # with torch, for gradients
```

It needs Optiland 0.6.3 or later. It does not need `ray_tracing_shapely`: it reads that project's
profile files.

## Use

```python
from optiland.optic import Optic
from optiland.materials import IdealMaterial
from optiland.physical_apertures import RadialAperture
from optiland_sms import read_meridian

profile = "lens.profile.json"      # a revolve profile whose faces are spline spans
front = read_meridian(profile, "N", reverse=True)
back = read_meridian(profile, "X", reverse=True)

op = Optic()
op.surfaces.add(index=0, thickness=10.0)
op.surfaces.add(index=1, surface_type="sms_spline", profile=profile, span="N", reverse=True,
                thickness=front.vertex_along_axis - back.vertex_along_axis,
                material=IdealMaterial(n=1.5), is_stop=True,
                aperture=RadialAperture(front.rim_radius))
op.surfaces.add(index=2, surface_type="sms_spline", profile=profile, span="X", reverse=True,
                thickness=5.0, aperture=RadialAperture(back.rim_radius))
op.surfaces.add(index=3)
```

- `span` is the span's short label; `reverse` says that light travels against the profile's axis
  direction.
- Take thicknesses from `vertex_along_axis`: a spline's vertex is where the curve meets the axis,
  between its points, not one of them.
- Set each aperture to the surface's `rim_radius`. Beyond the rim the surface continues along the
  rim's tangent, only so that the intersection solve sees a smooth surface.

For gradients, select the torch backend in float64 and make a surface's `points` a leaf:

```python
import optiland.backend as be

be.set_backend("torch")
be.set_precision("float64")
be.grad_mode.enable()
geometry = op.surfaces.surfaces[1].geometry
geometry.points = geometry.points.detach().clone().requires_grad_(True)
# trace, compute a merit, merit.backward(); geometry.points.grad holds the gradient
```

## What it covers

- **Refraction-only lenses whose faces are surfaces of revolution**, one spline span per face,
  from the axis to the rim.
- **Not** mirrors or total internal reflection, which do not fit a sequential trace; not Optiland's
  non-sequential engine, which has no plugin hook; not faces made of several spans.

The reader refuses what it cannot represent, and says why.

## How it is checked

On the RR lens of Chaves, *Introduction to Nonimaging Optics*, Example 9.1 (`tests/fixtures/`):
- the surface reproduces the profile's samples (sag to 1e-15, normals to 1e-13 rad) and its curve
  between them;
- numpy and torch trace the same rays to 1e-11 (Optiland's solve stops at its own tolerance);
- the gradient agrees with a central finite difference to 1e-5 relative (1e-6 measured).

```bash
pip install -e ".[dev]"
pytest
```

## Licence

Apache-2.0. See `NOTICE.txt`.
