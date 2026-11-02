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
Surface) nonimaging profiles as an Optiland `optiland.surfaces` entry-point
plugin -- a NewtonRaphsonGeometry subclass plus a register() entry point,
discovered automatically by Optiland's own plugin loader with no source edit
to Optiland itself.

Named for Optiland's audience deliberately, not this project's family: entry-
point discovery is a distribution channel into Optiland's own users, per
dev_doc/concepts/KD12_the_tool_list_is_audited_before_it_is_extended.md section 3.2.
Scene assembly, visualization, and ray persistence -- everything that is NOT
entry-point-discovered -- live in the separate ray_tracing_shapely_optiland
package instead.

Planning stage: no geometry class or register() target exists yet. See
dev_doc/architecture/components/optiland_sms_plugin.md. [KB7: the plugin
interface is one of the three reasons Optiland was chosen.]
===============================================================================
"""

__version__ = "0.0.0"
