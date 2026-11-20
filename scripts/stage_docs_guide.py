"""Stage the user and developer guides in a tree laid out like the Polars repo, for zensical to build.

The zensical config (apart from its nav, which comes from `docs/nav.toml`) and `docs/source/_build` come from a Polars checkout,
so the build runs as it does in Polars.
Only our pages are staged, as building the Polars pages would run all of their examples.

With `--watch`, it keeps an existing staged copy in sync with our pages until it is stopped,
for `zensical serve` to rebuild them as they change.

Usage: `python3 stage_docs_guide.py [--watch] <polars dir> <output dir>`
"""

from __future__ import annotations

import filecmp
import shutil
import sys
import time
import tomllib
from pathlib import Path

import tomli_w
import yaml

ROOT = Path(__file__).resolve().parent.parent
LINKS = Path("docs/source/_build/API_REFERENCE_LINKS.yml")


def stage_guides(polars_dir: Path, out_dir: Path) -> None:
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.copytree(polars_dir / "docs/source/_build", out_dir / "docs/source/_build")
    # The examples write their data files here, as in Polars.
    (out_dir / "docs/assets/data").mkdir(parents=True)
    sync_guides(polars_dir, out_dir)


def watch_guides(polars_dir: Path, out_dir: Path) -> None:
    last_error = None
    while True:
        time.sleep(0.5)
        try:
            sync_guides(polars_dir, out_dir)
            last_error = None
        except (ValueError, yaml.YAMLError) as error:
            # A file that is broken mid-edit should neither stop the watch, nor be reported on every pass.
            if str(error) != last_error:
                print(error, file=sys.stderr)
            last_error = str(error)


def sync_guides(polars_dir: Path, out_dir: Path) -> None:
    links = sync_guides_merged_api_links(polars_dir)
    config = sync_guides_config(polars_dir)
    # Only the files that changed are written, so that `zensical serve` only rebuilds those.
    shutil.copytree(
        ROOT / "docs/source",
        out_dir / "docs/source",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".*", "__pycache__", LINKS.name),
        copy_function=sync_guides_copy_if_changed,
    )
    sync_guides_write_if_changed(
        out_dir / LINKS, yaml.safe_dump(links, sort_keys=False)
    )
    sync_guides_write_if_changed(out_dir / "zensical.toml", tomli_w.dumps(config))


def sync_guides_copy_if_changed(src: str, dst: str) -> None:
    if not (Path(dst).exists() and filecmp.cmp(src, dst, shallow=False)):
        shutil.copy2(src, dst)


def sync_guides_write_if_changed(path: Path, text: str) -> None:
    if not (path.exists() and path.read_text() == text):
        path.write_text(text)


def sync_guides_merged_api_links(polars_dir: Path) -> dict:
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
        raise ValueError(
            f"{LINKS} conflicts with the Polars one:\n" + "\n".join(conflicts)
        )
    return merged


def sync_guides_config(polars_dir: Path) -> dict:
    with (polars_dir / "zensical.toml").open("rb") as f:
        config = tomllib.load(f)
    with (ROOT / "docs/nav.toml").open("rb") as f:
        config["project"]["nav"] = tomllib.load(f)["nav"]
    # Its redirects lead to Polars pages, which are not staged.
    del config["project"]["plugins"]["redirects"]
    return config


if __name__ == "__main__":
    args = sys.argv[1:]
    watch = args[:1] == ["--watch"]
    if watch:
        args = args[1:]
    if len(args) != 2:
        sys.exit(__doc__)
    polars_dir, out_dir = map(Path, args)
    try:
        if watch:
            watch_guides(polars_dir, out_dir)
        else:
            stage_guides(polars_dir, out_dir)
    except ValueError as error:
        sys.exit(str(error))
