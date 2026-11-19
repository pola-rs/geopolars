# Installation

GeoPolars is a Python package that builds on Polars.
Install it with pip:

```bash
pip install geopolars
```

This also installs Polars, if you do not have it yet.
GeoPolars needs Python 3.10 or newer, and Polars 2.0 or newer.

You do not need to install anything else.
GeoPolars uses [GEOS](https://libgeos.org/) for operations such as `buffer` and `intersection`,
and [PROJ](https://proj.org/) for [reprojecting](concepts/crs.md#reprojecting).
GeoPolars comes with these built in, including PROJ's database of coordinate reference systems.
These copies are separate from any GEOS or PROJ on your system,
or the ones that come with shapely or pyproj.

!!! warning "Alpha"

```text
GeoPolars is in alpha: expect breaking changes between releases.
```

[Getting started](getting-started.md) shows how the two fit together.

## Building from source

To build GeoPolars yourself, for example to contribute,
follow *Compile GeoPolars from source* in the [README](https://github.com/pola-rs/geopolars#installation).

## Ready?

[Getting started](getting-started.md) gives you a quick introduction so you can get in on the action swiftly~
