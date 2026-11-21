"""
Python-specific module: make rr profile
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
MAKE RR PROFILE -- writes the test fixture rr_example_9_1.profile.json
===============================================================================
The RR lens of Chaves, Introduction to Nonimaging Optics, 2nd ed., Example
9.1 (emitter E1 = (-1, 2), E2 = (1, 2); receiver R1 = (-0.5, -2),
R2 = (0.5, -2); n = 1.5; seed angles 303.3 and 42.7 degrees), built by
ray_tracing_shapely's RRLens with spline glass edges and exported as a
revolve profile about the y axis. Its spans N (facing the emitter) and X
(facing the receiver) are the two SMS surfaces.

Needs ray_tracing_shapely; the tests read only the JSON it writes. Run from
this folder:

    python make_rr_profile.py
===============================================================================
"""

import json
from pathlib import Path

from ray_tracing_shapely.core.scene import Scene
from ray_tracing_shapely.optical_elements.prisms.profile_export import export_profile
from ray_tracing_shapely.optical_elements.prisms.rr_lens import RRLens


def main() -> None:
    lens = RRLens(Scene(), (-1.0, 2.0), (1.0, 2.0), (-0.5, -2.0), (0.5, -2.0),
                  seed_n_angle_deg=303.3, seed_x_angle_deg=42.7)
    profile = export_profile(lens, symmetry={"kind": "revolve", "axis": {"point": [0.0, 0.0], "direction": [0.0, 1.0]}},
                             physical_params={"n": 1.5, "e1": [-1.0, 2.0], "e2": [1.0, 2.0],
                                              "r1": [-0.5, -2.0], "r2": [0.5, -2.0]},
                             numerical_params={})
    out = Path(__file__).with_name("rr_example_9_1.profile.json")
    out.write_text(json.dumps(profile, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
