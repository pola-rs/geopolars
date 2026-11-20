"""Stage the user and developer guides in a tree laid out like the Polars repo, for zensical to build.

The zensical config (apart from its nav, which comes from `docs/nav.toml`) and `docs/source/_build` come from a Polars checkout,
so the build runs as it does in Polars.
Only our pages are staged, as building the Polars pages would run all of their examples.

Usage: `python3 stage_docs_guide.py <polars dir> <output dir>`
"""

from __future__ import annotations

import shutil
import sys
import tomllib
from pathlib import Path

import tomli_w
import yaml

ROOT = Path(__file__).resolve().parent.parent
LINKS = Path("docs/source/_build/API_REFERENCE_LINKS.yml")


def stage_guides(polars_dir: Path, out_dir: Path) -> None:
    links = stage_guides_merged_api_links(polars_dir)
    config = stage_guides_config(polars_dir)

    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.copytree(polars_dir / "docs/source/_build", out_dir / "docs/source/_build")
    shutil.copytree(
        ROOT / "docs/source",
        out_dir / "docs/source",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".*", "__pycache__"),
    )
    with (out_dir / LINKS).open("w") as f:
        yaml.safe_dump(links, f, sort_keys=False)
    with (out_dir / "zensical.toml").open("wb") as f:
        tomli_w.dump(config, f)
    # The examples write their data files here, as in Polars.
    (out_dir / "docs/assets/data").mkdir(parents=True)


def stage_guides_merged_api_links(polars_dir: Path) -> dict:
    # In Polars, our links go into its file, so a key in both must point to the same page.
    merged = yaml.safe_load((polars_dir / LINKS).read_text())
    conflicts = []
    for language, links in yaml.safe_load((ROOT / LINKS).read_text()).items():
        merged_links = merged.setdefault(language, {})
        for key, link in links.items():
            if merged_links.get(key, link) != link:
                conflicts.append(
                    f"{language}.{key}: {link} (Polars has {merged_links[key]})"
                )
            merged_links[key] = link
    if conflicts:
        sys.exit(f"{LINKS} conflicts with the Polars one:\n" + "\n".join(conflicts))
    return merged


def stage_guides_config(polars_dir: Path) -> dict:
    with (polars_dir / "zensical.toml").open("rb") as f:
        config = tomllib.load(f)
    with (ROOT / "docs/nav.toml").open("rb") as f:
        config["project"]["nav"] = tomllib.load(f)["nav"]
    # Its redirects lead to Polars pages, which are not staged.
    del config["project"]["plugins"]["redirects"]
    return config


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    stage_guides(Path(sys.argv[1]), Path(sys.argv[2]))
