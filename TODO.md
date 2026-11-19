# Before dev becomes main

## Licence, CI and metadata

- [x] Restore `LICENSE` from `main` (MIT, © 2022 Kyle Barron), possibly adding "GeoPolars contributors".
- [x] Add `.github/workflows`: tests (`make test-all`), lint (ruff, clippy, rustfmt) and a docs build. Wheels can wait.
- [ ] When adding a wheels workflow: on Windows, run `git config --global core.symlinks true` before checkout,
  as `geopolars/LICENSE` is a symlink.
- [x] Add `.github/dependabot.yml`.
- [x] Bump the Python package version above `0.1.0a4`, which is already on PyPI.
- [x] Check the `polars>=1.44` floor: the pinned Polars commit first appears in `py-2.0.0`.
- [x] Fill in description, readme, licence, authors and URLs in `geopolars/pyproject.toml`.
- [x] Point the GitHub link in `geopolars/docs/source/conf.py` at `pola-rs/geopolars`.

## README

Follow the order of the Polars README.

- [x] badges, and links to the API reference and user guide.
- [x] A two-sentence pitch, followed by a tightened feature list.
- [x] An end-to-end example: read GeoParquet, build points from lon/lat, `to_crs`, `area`, write.
  `examples/run_crs.py` is a good starting point.
- [x] Installation instructions that work, including building from source with `make install`.
- [x] One line on stability, instead of the "HIGHLY unstable" banner.
- [x] Contributing and licence sections.
- [x] Remove the empty "CRS awareness" and "Geometry Operations" headings, and the two TODOs.
- [x] Move the validity explanation and table to the user guide.
- [x] Move the UDF section to the user guide.
- [x] Move the winding order conventions to the user guide.
- [x] Move the developer doc links to the contributing guide.

## Docs

The user guide is embedded into the Polars user guide (zensical).
The API reference (Sphinx, in `geopolars/docs/`) is built separately,
but is also hosted on the Polars domain.

- [x] Decide on the tooling: the user guide goes into the Polars user guide, the API reference stays in Sphinx.
- [x] Mirror the Polars layout: `docs/source/user-guide/geo/` and `docs/source/development/`.
- [x] Move the developer docs into `development/`.
- [x] Rename doc files to kebab-case, which also fixes the broken README links.
- [x] Move the vendored GeoParquet specs out of the docs source, into `docs/ref/`.
- [x] Move the common arguments into the API reference (`reference/common_arguments.rst`).

### User guide

- [x] Rewrite the pages in the Polars conventions:
  code in `docs/source/src/python/user-guide/geo/`, included with `{{code_block(...)}}` so it runs during the build,
  Material admonitions, and links into the API reference (`docs/source/_build/API_REFERENCE_LINKS.yml`).
- [ ] Flesh out the stubs: `concepts/crs.md` and `concepts/winding-order.md`,
  and resolve the TODO in `udfs.md`.
- [ ] Write the missing pages: installation, geometry types (GeoArrow, dimensions), nulls, NaN and empty geometries,
  IO (GeoParquet, WKB/WKT), and migrating from GeoPandas.

### API reference

- [ ] Add a workflow that deploys the Sphinx build to the Polars domain, next to the Polars API reference.
- [ ] Decide on versioning (stable and dev), as the Polars API reference does.
- [ ] Match the look of the Polars API reference, and link it to and from the user guide.
- [ ] `common_arguments.rst`: drop the *Legacy* notes once the migration is done,
  as this is now a user-facing page.

### Developer docs

- [ ] Decide where the developer docs are read: on GitHub, or published somewhere.
- [ ] Render `expression-flow.dot` to SVG, or convert it to Mermaid.
- [ ] Turn `roadmap.md` into GitHub issues or milestones, or trim it to a user-facing roadmap.
- [ ] `view-types.md`: rewrite (it has a TODO saying so).
- [ ] `architecture.md`: fix the undefined `[Typical program flow]` link.

### Once deployed

- [ ] Point the README links to the user guide and API reference at the hosted pages, instead of GitHub.
- [ ] Replace or take down the 2023 mkdocs docs on `gh-pages`.

## Examples

- [ ] Move or drop `run_mean_coordinate_cost.py` and `run_mean_coordinate_manual.py`.
- [ ] Add a module docstring to `run_geo.py` and `run_line.py`, and remove the TODO in `run_geo.py`.
- [ ] Consider bringing back sample datasets (`geopolars.datasets` on `main`) for copy-paste examples.

## Contributor files

- [ ] A short root `CONTRIBUTING.md` pointing to the contributing guide.
- [ ] Issue templates (bug, feature) and a PR template.
- [ ] `CODE_OF_CONDUCT.md`.

## Switching the branch

`dev` and `main` share no history, so `main` gets replaced rather than merged.

- [ ] Check for open PRs targeting `main`, and for branch protection rules.
- [ ] Push `main` to `archive/legacy-main`.
- [ ] Force-push `dev` to `main`.
- [ ] Remove this file.
