# optiland-sms

Registers [`ray_tracing_shapely`](https://github.com/micheleben/ray_tracing_shapely)'s
Miñano–Benítez SMS nonimaging profiles as an Optiland `optiland.surfaces` entry-point
plugin — discovered automatically by [Optiland](https://github.com/optiland/optiland)'s
own plugin loader, no source edit to Optiland required.

**Planning stage — no code yet.** Named for Optiland's audience deliberately (entry-point
discovery is a distribution channel into Optiland's own users), scoped narrowly to the
geometry class and its `register()` target only — see
`dev_doc/architecture/components/optiland_sms_plugin.md`. Scene assembly, visualization, and ray persistence live in the
sibling `ray-tracing-shapely-optiland` package instead.

## Install (once there is code to install)

```bash
pip install -e ".[dev]"
```
