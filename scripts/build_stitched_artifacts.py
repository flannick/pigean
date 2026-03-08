#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


COMMON_MODULES = (
    "pegs_cli_errors.py",
    "pegs_cli_utils.py",
    "pegs_sync_guard.py",
    "pegs_types.py",
    "pegs_utils.py",
    "pegs_utils_bundle.py",
    "pegs_utils_phewas.py",
)

COMMON_PACKAGES = (
    "pegs_shared",
)

PIGEAN_PROFILE = {
    "output_name": "pigean_stitched.py",
    "entry_module": "pigean.legacy_main",
    "entry_callable": "main",
    "modules": COMMON_MODULES
    + (
        "pigean_dispatch.py",
        "pigean_huge.py",
        "pigean_legacy_main.py",
        "pigean_outputs.py",
        "pigean_phewas.py",
        "pigean_pipeline.py",
    ),
    "packages": COMMON_PACKAGES + ("pigean",),
}

EAGGL_PROFILE = {
    "output_name": "eaggl_stitched.py",
    "entry_module": "eaggl.legacy_main",
    "entry_callable": "main",
    "modules": COMMON_MODULES,
    "packages": COMMON_PACKAGES + ("eaggl",),
}

PROFILES = {
    "pigean": PIGEAN_PROFILE,
    "eaggl": EAGGL_PROFILE,
}


def _iter_module_files(src_root: Path, profile_name: str) -> list[tuple[str, str, bool, str]]:
    profile = PROFILES[profile_name]
    module_entries: list[tuple[str, str, bool, str]] = []
    for rel_path in profile["modules"]:
        source_path = src_root / rel_path
        if not source_path.exists():
            raise FileNotFoundError("Missing source file for stitched artifact: %s" % source_path)
        module_name = rel_path[:-3]
        module_entries.append((module_name, "src/%s" % rel_path, False, source_path.read_text(encoding="utf-8")))
    for package_name in profile["packages"]:
        package_root = src_root / package_name
        if not package_root.exists():
            raise FileNotFoundError("Missing package directory for stitched artifact: %s" % package_root)
        init_path = package_root / "__init__.py"
        if init_path.exists():
            module_entries.append((package_name, "src/%s" % str(init_path.relative_to(src_root)), True, init_path.read_text(encoding="utf-8")))
        else:
            module_entries.append((package_name, "src/%s/__init__.py" % package_name, True, ""))
        for source_path in sorted(package_root.rglob("*.py")):
            if source_path.name == "__init__.py":
                continue
            rel_path = str(source_path.relative_to(src_root))
            module_name = rel_path[:-3].replace("/", ".")
            module_entries.append((module_name, "src/%s" % rel_path, False, source_path.read_text(encoding="utf-8")))
    return module_entries


def _format_module_entries(module_entries: list[tuple[str, str, bool, str]]) -> str:
    lines: list[str] = []
    for module_name, rel_path, is_package, source_text in module_entries:
        lines.append("# ===== BEGIN %s =====" % rel_path)
        lines.append(
            "_register_module(%r, %s, %r, %r)"
            % (module_name, "True" if is_package else "False", rel_path, source_text)
        )
        lines.append("# ===== END %s =====" % rel_path)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_artifact(profile_name: str, src_root: Path) -> str:
    profile = PROFILES[profile_name]
    module_entries = _iter_module_files(src_root, profile_name)
    module_payload = _format_module_entries(module_entries)
    return """#!/usr/bin/env python3
from __future__ import annotations

import importlib.abc
import importlib.util
import sys

PROFILE_NAME = {profile_name!r}
ENTRY_MODULE = {entry_module!r}
ENTRY_CALLABLE = {entry_callable!r}

MODULE_SOURCES = {{}}


def _register_module(module_name, is_package, source_path, source_text):
    MODULE_SOURCES[module_name] = {{
        "is_package": is_package,
        "source_path": source_path,
        "source_text": source_text,
    }}


{module_payload}


class _StitchedLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        module_info = MODULE_SOURCES.get(fullname)
        if module_info is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            self,
            origin="<stitched:%s>" % module_info["source_path"],
            is_package=module_info["is_package"],
        )

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module_info = MODULE_SOURCES[module.__name__]
        module.__file__ = "<stitched:%s>" % module_info["source_path"]
        if module_info["is_package"]:
            module.__path__ = []
        code = compile(module_info["source_text"], module.__file__, "exec")
        exec(code, module.__dict__)


def _install_loader():
    for finder in sys.meta_path:
        if isinstance(finder, _StitchedLoader):
            return finder
    loader = _StitchedLoader()
    sys.meta_path.insert(0, loader)
    return loader


def main(argv=None):
    _install_loader()
    entry_module = __import__(ENTRY_MODULE, fromlist=[ENTRY_CALLABLE])
    entry_callable = getattr(entry_module, ENTRY_CALLABLE)
    return entry_callable(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
""".format(
        profile_name=profile_name,
        entry_module=profile["entry_module"],
        entry_callable=profile["entry_callable"],
        module_payload=module_payload,
    )


def build_artifact(profile_name: str, repo_root: Path, out_dir: Path) -> Path:
    src_root = repo_root / "src"
    artifact_text = _render_artifact(profile_name, src_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / PROFILES[profile_name]["output_name"]
    output_path.write_text(artifact_text, encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build runnable stitched single-file artifacts from the canonical modular source tree.",
    )
    parser.add_argument(
        "--artifact",
        dest="artifacts",
        action="append",
        choices=("pigean", "eaggl"),
        help="Artifact to build. Can be specified multiple times. Defaults to both.",
    )
    parser.add_argument(
        "--out-dir",
        default="dist/stitched",
        help="Output directory for generated stitched artifacts.",
    )
    return parser


def main_cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    out_dir = (repo_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    artifact_names = args.artifacts or ["pigean", "eaggl"]
    for artifact_name in artifact_names:
        output_path = build_artifact(artifact_name, repo_root, out_dir)
        print("Wrote %s" % output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
