"""Build or serve the user and developer guides with zensical, as the Polars docs build would.

The guides are staged in a tree laid out like the Polars repo, for zensical to build.
The zensical config (apart from its nav, which comes from `docs/nav.toml`) and `docs/source/_build` come from a Polars checkout,
so the build runs as it does in Polars.
Only our pages are staged, as building the Polars pages would run all of their examples.

Both commands fail on what the strict Polars build fails on,
and on what zensical only logs: a failing example, or an unknown API key in a `code_block`.
`build` reports every problem before it fails, and `serve` stops after the first rebuild with one.

Usage: `python3 docs_guide.py build|serve <polars dir> <output dir>`
"""

from __future__ import annotations

import filecmp
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import tomli_w
import yaml

ROOT = Path(__file__).resolve().parent.parent
LINKS = Path("docs/source/_build/API_REFERENCE_LINKS.yml")
ZENSICAL = Path(sys.executable).parent / "zensical"

COLOUR = re.compile(r"\x1b\[[0-9;]*m")
PROBLEM = re.compile(
    r"^(Warning|Error)\b|^\d+ issues? found"
    r"|Execution of python code block exited with errors|Could not find "
)
# zensical ends every build or rebuild with this line.
BUILD_SUMMARY = re.compile(r"^(No|\d+) issues? found")


def build_guides(polars_dir: Path, out_dir: Path) -> None:
    stage_guides(polars_dir, out_dir)
    zensical = start_zensical("build", out_dir)
    has_problem = False
    with (out_dir / "build.log").open("w") as log:
        for line in zensical.stdout:
            print(line, end="", flush=True)
            log.write(line)
            has_problem |= bool(PROBLEM.search(COLOUR.sub("", line)))
    if zensical.wait() or has_problem:
        sys.exit("The guides have problems, see the build log above.")


def serve_guides(polars_dir: Path, out_dir: Path) -> None:
    stage_guides(polars_dir, out_dir)
    zensical = start_zensical("serve", out_dir)
    sync_errors = []
    threading.Thread(
        target=serve_guides_sync_until_error,
        args=(polars_dir, out_dir, zensical, sync_errors),
        daemon=True,
    ).start()
    has_problem = False
    try:
        for line in zensical.stdout:
            print(line, end="", flush=True)
            plain = COLOUR.sub("", line)
            has_problem |= bool(PROBLEM.search(plain))
            if has_problem and BUILD_SUMMARY.search(plain):
                break
    finally:
        zensical.terminate()
        zensical.wait()
    if sync_errors:
        sys.exit(str(sync_errors[0]))
    if has_problem:
        sys.exit("Stopped serving, as the guides have problems: see above.")


def serve_guides_sync_until_error(
    polars_dir: Path, out_dir: Path, zensical: subprocess.Popen, errors: list
) -> None:
    while True:
        time.sleep(0.5)
        try:
            sync_guides(polars_dir, out_dir)
        except (ValueError, yaml.YAMLError) as error:
            errors.append(error)
            zensical.terminate()
            return


def start_zensical(command: str, out_dir: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [ZENSICAL, command],
        cwd=out_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stage_guides(polars_dir: Path, out_dir: Path) -> None:
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.copytree(polars_dir / "docs/source/_build", out_dir / "docs/source/_build")
    # The examples write their data files here, as in Polars.
    (out_dir / "docs/assets/data").mkdir(parents=True)
    sync_guides(polars_dir, out_dir)


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
    commands = {"build": build_guides, "serve": serve_guides}
    if len(sys.argv) != 4 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    try:
        commands[sys.argv[1]](Path(sys.argv[2]), Path(sys.argv[3]))
    except ValueError as error:
        sys.exit(str(error))
    except KeyboardInterrupt:
        sys.exit(130)
