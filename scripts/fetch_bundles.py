#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def is_remote(path: str) -> bool:
    low = path.lower()
    return low.startswith("http://") or low.startswith("https://") or low.startswith("ftp://")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest_abs = dest.resolve()
    for member in tar.getmembers():
        out = (dest / member.name).resolve()
        if not str(out).startswith(str(dest_abs)):
            raise ValueError(f"unsafe path in tar: {member.name}")
    tar.extractall(dest)


def load_catalog(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        catalog = json.load(fh)
    if not isinstance(catalog, dict):
        raise ValueError("catalog must be a JSON object")
    catalog.setdefault("profiles", {})
    catalog.setdefault("modes", {})
    catalog.setdefault("bundles", {})
    return catalog


def resolve_bundle_names(catalog: dict, profile: str | None, modes: list[str], explicit: list[str]) -> list[str]:
    known = set(catalog.get("bundles", {}).keys())
    selected: set[str] = set()

    if profile:
        profile_items = catalog.get("profiles", {}).get(profile)
        if profile_items is None:
            raise ValueError(f"unknown profile: {profile}")
        if "*" in profile_items:
            selected.update(known)
        else:
            selected.update(profile_items)

    for mode in modes:
        mode_items = catalog.get("modes", {}).get(mode)
        if mode_items is None:
            raise ValueError(f"unknown mode: {mode}")
        selected.update(mode_items)

    selected.update(explicit)

    if not selected:
        selected.update(catalog.get("profiles", {}).get("minimal", []))

    unknown = sorted([x for x in selected if x not in known])
    if unknown:
        raise ValueError(f"unknown bundles in selection: {unknown}")

    return sorted(selected)


def download_to_tmp(url_or_path: str, tmpdir: Path) -> Path:
    if is_remote(url_or_path):
        out = tmpdir / "bundle.tar.gz"
        urllib.request.urlretrieve(url_or_path, out)
        return out
    p = Path(url_or_path)
    if not p.exists():
        raise FileNotFoundError(f"bundle path not found: {p}")
    return p


def install_bundle(name: str, spec: dict, dest: Path, force: bool = False, dry_run: bool = False) -> tuple[Path, Path]:
    version = str(spec.get("version", "unknown"))
    url = spec.get("url")
    sha = spec.get("sha256")
    if not url:
        raise ValueError(f"bundle '{name}' missing url")

    install_dir = dest / f"{name}-{version}"
    current_dir = dest / "current"
    current_link = current_dir / name

    if dry_run:
        return install_dir, current_link

    with tempfile.TemporaryDirectory() as t:
        tmpdir = Path(t)
        bundle_path = download_to_tmp(url, tmpdir)

        if sha:
            actual = sha256_file(bundle_path)
            if actual.lower() != str(sha).lower():
                raise ValueError(f"checksum mismatch for {name}: expected {sha}, got {actual}")

        extract_dir = tmpdir / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(bundle_path, "r:*") as tar:
            safe_extract(tar, extract_dir)

        dirs = [p for p in extract_dir.iterdir() if p.is_dir()]
        if len(dirs) == 1:
            source_root = dirs[0]
        else:
            source_root = extract_dir

        if install_dir.exists():
            if not force:
                raise FileExistsError(f"already exists: {install_dir} (use --force)")
            shutil.rmtree(install_dir)

        shutil.copytree(source_root, install_dir)

    current_dir.mkdir(parents=True, exist_ok=True)
    if current_link.exists() or current_link.is_symlink():
        current_link.unlink()
    rel_target = os.path.relpath(install_dir, current_dir)
    current_link.symlink_to(rel_target)

    return install_dir, current_link


def main() -> int:
    ap = argparse.ArgumentParser(description="Download/install bundle tarballs from a catalog")
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--dest", type=Path, default=Path("bundles"))
    ap.add_argument("--profile", type=str, default="minimal")
    ap.add_argument("--mode", action="append", default=[], choices=["gwas", "exomes", "huge_scores", "gene_list"])
    ap.add_argument("--bundle", action="append", default=[])
    ap.add_argument("--list", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--force", action="store_true", default=False)
    args = ap.parse_args()

    catalog = load_catalog(args.catalog)
    names = resolve_bundle_names(catalog, args.profile, args.mode, args.bundle)

    if args.list:
        for n in names:
            spec = catalog["bundles"][n]
            print(f"{n}\t{spec.get('version','unknown')}\t{spec.get('url','')}")
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)

    for name in names:
        spec = catalog["bundles"][name]
        install_dir, current_link = install_bundle(name, spec, args.dest, force=args.force, dry_run=args.dry_run)
        if args.dry_run:
            print(f"DRY-RUN install {name} -> {install_dir} ; link {current_link}")
        else:
            print(f"Installed {name} -> {install_dir}")
            print(f"Updated link {current_link} -> {current_link.readlink()}")

    if not args.dry_run:
        bundle_root = (args.dest / "current").resolve()
        print("")
        print("Next step:")
        print(f"- Edit config/profiles/common.factor.json and replace __BUNDLE_ROOT__ with:")
        print(f"  {bundle_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
