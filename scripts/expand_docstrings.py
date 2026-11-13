"""Fill in the `@doc` docstrings of a package in its source, for the wheel.

Editors read docstrings from source without running decorators,
so they would otherwise show `{section}` rather than the shared text.
`make wheel` runs this on a copy of the source, as it rewrites the files in place.

Usage: `python3 expand_docstrings.py <package dir>`
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path


def load_fill_docstring(package_dir: Path):
    # Loaded by path, so that neither polars nor the built plugin is needed.
    spec = importlib.util.spec_from_file_location("_docs", package_dir / "_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fill_docstring


def doc_decorator(node: ast.AST) -> ast.Call | None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    for decorator in node.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "doc"
        ):
            return decorator
    return None


def docstring_literal(text: str, indent: str) -> str:
    text = text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    first, *rest = text.split("\n")
    body = "".join(f"\n{indent}{line}" if line else "\n" for line in rest)
    return f'"""{first}{body}\n{indent}"""'


def expand_file(path: Path, fill_docstring) -> int:
    lines = path.read_text().splitlines(keepends=True)
    # (first line, last line, replacement), with 0-based inclusive line numbers.
    edits = []
    filled_by_name = {}
    for node in ast.walk(ast.parse("".join(lines))):
        decorator = doc_decorator(node)
        if decorator is None:
            continue
        template = ast.get_docstring(node, clean=False)
        if decorator.args or template is None:
            sys.exit(f"{path}:{node.lineno}: expected `@doc(**kwargs)` and a docstring")
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in decorator.keywords}
        filled = fill_docstring(template, kwargs)
        docstring = node.body[0]
        indent = lines[docstring.lineno - 1][: docstring.col_offset]
        after = lines[docstring.end_lineno - 1][docstring.end_col_offset :]
        literal = indent + docstring_literal(filled, indent) + after
        edits.append((docstring.lineno - 1, docstring.end_lineno - 1, literal))
        edits.append((decorator.lineno - 1, decorator.end_lineno - 1, ""))
        filled_by_name[node.name] = filled

    for first, last, replacement in sorted(edits, reverse=True):
        lines[first : last + 1] = [replacement]
    expanded = "".join(lines)
    path.write_text(expanded)

    for node in ast.walk(ast.parse(expanded)):
        if doc_decorator(node) is not None:
            sys.exit(f"{path}:{node.lineno}: `@doc` left after expanding")
        name = getattr(node, "name", None)
        if name in filled_by_name and ast.get_docstring(node) != filled_by_name[name]:
            sys.exit(f"{path}:{node.lineno}: expanded docstring of `{name}` differs")
    return len(filled_by_name)


def main() -> None:
    package_dir = Path(sys.argv[1])
    fill_docstring = load_fill_docstring(package_dir)
    count = sum(expand_file(path, fill_docstring) for path in package_dir.rglob("*.py"))
    print(f"Expanded {count} docstrings in {package_dir}")


if __name__ == "__main__":
    main()
