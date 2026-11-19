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

- [ ] Logo (`static/logo.svg` from `main`), badges, and links to the API reference and user guide.
- [ ] A two-sentence pitch, followed by a tightened feature list.
- [ ] An end-to-end example: read GeoParquet, build points from lon/lat, `to_crs`, `area`, write.
  `examples/run_crs.py` is a good starting point.
- [ ] Installation instructions that work, including building from source with `make install`.
- [ ] One line on stability, instead of the "HIGHLY unstable" banner.
- [ ] Contributing and licence sections.
- [ ] Remove the empty "CRS awareness" and "Geometry Operations" headings, and the two TODOs.
- [ ] Move the validity explanation and table to the user guide.
- [ ] Move the UDF section to the user guide.
- [ ] Move the winding order conventions to the user guide.
- [ ] Move the developer doc links to the contributing guide.

## Docs

We want to have everything in sphinx.

OUTDATED:
Mirror Polars: a user guide and contributor guide under `docs/source/`,
and the Sphinx API reference in `geopolars/docs/` (already in place).

```text
docs/
  source/
    user-guide/
      getting-started.md, installation.md
      concepts/  geometry-types.md (GeoArrow, dimensions), crs.md,
                 validity.md (from README), nulls-nan-empty.md
      io/        geoparquet.md, wkb-wkt.md
      udfs.md (from README), migrating-from-geopandas.md
    development/
      contributing/  index.md (setup, make targets), test.md, code-style.md
      architecture/  architecture.md, expression-flow.md (+ rendered diagram),
                     view-types.md, common-arguments.md, tier-per-expression.md
      kernels/       kernel-guide.md, native-kernels.md, geos-kernels.md
  assets/          logo, rendered diagrams
```

- [ ] Decide whether the user guide gets its own tool (zensical, like Polars) or goes into the Sphinx site as MyST pages.
- [ ] Write the user guide pages.
- [ ] Move the developer docs into `development/`.
- [ ] Rename doc files to kebab-case, which also fixes the broken README links.
- [ ] Render `expression-flow.dot` to SVG, or convert it to Mermaid.
- [ ] Replace the vendored GeoParquet specs with links to upstream.
- [ ] Turn `Roadmap.md` into GitHub issues or milestones, or trim it to a user-facing roadmap.
- [ ] `Common arguments.md`: drop the legacy notes once the migration is done, or mark it as a design doc.
- [ ] `View Types.md`: rewrite (it has a TODO saying so).
- [ ] `Architecture.md`: fix the undefined `[Typical program flow]` link.
- [ ] Replace or take down the 2023 mkdocs docs on `gh-pages` once the new docs deploy.

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
