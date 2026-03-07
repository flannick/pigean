import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass, field


def _default_bail(message):
    raise ValueError(message)


EAGGL_BUNDLE_SCHEMA = "pigean_eaggl_bundle/v1"
EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS = set([
    "X_in",
    "gene_stats_in",
    "gene_set_stats_in",
    "gene_phewas_bfs_in",
    "gene_set_phewas_stats_in",
])


def get_tar_write_mode_for_bundle_path(bundle_path, option_name="--eaggl-bundle-out", bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    lower = bundle_path.lower()
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        return "w:gz"
    if lower.endswith(".tar"):
        return "w"
    bail_fn("Option %s must end with .tar, .tar.gz, or .tgz" % option_name)


def _is_unsafe_tar_member_path(member_name):
    if os.path.isabs(member_name):
        return True
    normalized_parts = member_name.replace("\\", "/").split("/")
    return ".." in normalized_parts


def safe_extract_tar_to_temp(bundle_path, temp_prefix="bundle_in_", bundle_flag_name="--bundle-in", bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    tmp_dir = tempfile.mkdtemp(prefix=temp_prefix)
    try:
        with tarfile.open(bundle_path, "r:*") as tar_fh:
            members = tar_fh.getmembers()
            for member in members:
                if _is_unsafe_tar_member_path(member.name):
                    bail_fn("Refusing to read suspicious path in %s bundle: %s" % (bundle_flag_name, member.name))
            tar_fh.extractall(tmp_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return tmp_dir


def is_huge_statistics_bundle_path(huge_statistics_file):
    lower = huge_statistics_file.lower()
    return lower.endswith(".tar.gz") or lower.endswith(".tgz") or lower.endswith(".tar")


def write_prefixed_tar_bundle(
    out_path,
    *,
    prefix_basename,
    write_prefix_fn,
    is_bundle_path_fn=None,
    option_name="--bundle-out",
    temp_prefix="bundle_out_",
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if is_bundle_path_fn is None:
        is_bundle_path_fn = is_huge_statistics_bundle_path

    if not is_bundle_path_fn(out_path):
        write_prefix_fn(out_path)
        return out_path

    tar_mode = get_tar_write_mode_for_bundle_path(
        out_path,
        option_name=option_name,
        bail_fn=bail_fn,
    )
    ensure_parent_dir_for_file(out_path)

    with tempfile.TemporaryDirectory(prefix=temp_prefix) as stage_dir:
        staged_prefix = os.path.join(stage_dir, prefix_basename)
        write_prefix_fn(staged_prefix)
        staged_names = sorted(
            name for name in os.listdir(stage_dir)
            if name.startswith(prefix_basename + ".")
        )
        if len(staged_names) == 0:
            bail_fn("Cannot write %s: no staged files with prefix %s." % (option_name, prefix_basename))
        with tarfile.open(out_path, tar_mode) as tar_fh:
            for name in staged_names:
                tar_fh.add(os.path.join(stage_dir, name), arcname=name)
    return out_path


def read_prefixed_tar_bundle(
    in_path,
    *,
    required_suffix,
    read_prefix_fn,
    is_bundle_path_fn=None,
    bundle_flag_name="--bundle-in",
    temp_prefix="bundle_in_",
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if is_bundle_path_fn is None:
        is_bundle_path_fn = is_huge_statistics_bundle_path

    if not is_bundle_path_fn(in_path):
        return read_prefix_fn(in_path)

    extract_dir = safe_extract_tar_to_temp(
        in_path,
        temp_prefix=temp_prefix,
        bundle_flag_name=bundle_flag_name,
        bail_fn=bail_fn,
    )
    try:
        marker_files = sorted(
            name for name in os.listdir(extract_dir)
            if name.endswith(required_suffix)
        )
        if len(marker_files) == 0:
            bail_fn("%s bundle did not contain a %s file" % (bundle_flag_name, required_suffix))
        if len(marker_files) > 1:
            bail_fn(
                "%s bundle contained multiple %s files: %s"
                % (bundle_flag_name, required_suffix, ", ".join(marker_files))
            )
        prefix = os.path.join(extract_dir, marker_files[0][:-len(required_suffix)])
        return read_prefix_fn(prefix)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


@dataclass
class BundleManifest:
    manifest: dict
    bundle_path: str | None = None
    extract_dir: str | None = None
    default_inputs: dict = field(default_factory=dict)

    @classmethod
    def load_defaults(
        cls,
        bundle_path,
        expected_schema,
        allowed_default_inputs,
        *,
        bundle_flag_name="--bundle-in",
        manifest_name="manifest.json",
        temp_prefix="bundle_in_",
        bail_fn=None,
    ):
        if bail_fn is None:
            bail_fn = _default_bail

        extract_dir, manifest = load_bundle_manifest(
            bundle_path,
            expected_schema,
            bundle_flag_name=bundle_flag_name,
            manifest_name=manifest_name,
            temp_prefix=temp_prefix,
            bail_fn=bail_fn,
        )
        default_inputs = resolve_bundle_default_inputs(
            manifest.get("default_inputs"),
            extract_dir,
            allowed_default_inputs,
            bundle_flag_name=bundle_flag_name,
            bail_fn=bail_fn,
        )
        return cls(
            manifest=manifest,
            bundle_path=bundle_path,
            extract_dir=extract_dir,
            default_inputs=default_inputs,
        )

    @classmethod
    def build(
        cls,
        schema,
        source_tool,
        source_mode,
        source_argv,
        default_inputs,
        files_metadata,
    ):
        return cls(
            manifest={
                "schema": schema,
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": {
                    "tool": source_tool,
                    "mode": source_mode,
                    "argv": list(source_argv),
                },
                "default_inputs": dict(default_inputs),
                "files": dict(files_metadata),
            },
            default_inputs=dict(default_inputs),
        )

    def write_manifest(self, stage_dir, manifest_name="manifest.json"):
        manifest_path = os.path.join(stage_dir, manifest_name)
        with open(manifest_path, "w", encoding="utf-8") as out_fh:
            json.dump(self.manifest, out_fh, indent=2, sort_keys=True)
            out_fh.write("\n")
        return manifest_path

    def write_archive(self, out_path, tar_mode, stage_dir, staged_file_names, *, manifest_name="manifest.json"):
        manifest_path = os.path.join(stage_dir, manifest_name)
        with tarfile.open(out_path, tar_mode) as tar_fh:
            tar_fh.add(manifest_path, arcname=manifest_name)
            for bundle_name in sorted(staged_file_names):
                tar_fh.add(os.path.join(stage_dir, bundle_name), arcname=bundle_name)


@dataclass
class BundleDefaultsApplication:
    bundle: BundleManifest
    applied_defaults: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "bundle_path": self.bundle.bundle_path,
            "extract_dir": self.bundle.extract_dir,
            "schema": self.bundle.manifest.get("schema") if isinstance(self.bundle.manifest, dict) else None,
            "manifest": self.bundle.manifest,
            "default_inputs": self.bundle.default_inputs,
            "applied_defaults": self.applied_defaults,
        }


def load_bundle_manifest(
    bundle_path,
    expected_schema,
    *,
    bundle_flag_name="--bundle-in",
    manifest_name="manifest.json",
    temp_prefix="bundle_in_",
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if not os.path.exists(bundle_path):
        bail_fn("Could not find %s bundle %s" % (bundle_flag_name, bundle_path))

    extract_dir = safe_extract_tar_to_temp(
        bundle_path,
        temp_prefix=temp_prefix,
        bundle_flag_name=bundle_flag_name,
        bail_fn=bail_fn,
    )
    manifest_path = os.path.join(extract_dir, manifest_name)
    if not os.path.exists(manifest_path):
        bail_fn("%s bundle is missing %s: %s" % (bundle_flag_name, manifest_name, bundle_path))

    with open(manifest_path) as in_fh:
        manifest = json.load(in_fh)
    if not isinstance(manifest, dict):
        bail_fn("%s manifest must be a JSON object: %s" % (bundle_flag_name, bundle_path))
    if manifest.get("schema") != expected_schema:
        bail_fn(
            "Unsupported %s schema '%s' in %s (expected %s)"
            % (bundle_flag_name, manifest.get("schema"), bundle_path, expected_schema)
        )
    return extract_dir, manifest


def resolve_bundle_default_inputs(
    raw_default_inputs,
    extract_dir,
    allowed_default_inputs,
    *,
    bundle_flag_name="--bundle-in",
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail

    if not isinstance(raw_default_inputs, dict):
        bail_fn("%s manifest missing required object key 'default_inputs'" % bundle_flag_name)

    resolved_default_inputs = {}
    abs_extract_dir = os.path.abspath(extract_dir)
    for key, rel_path in raw_default_inputs.items():
        if key not in allowed_default_inputs:
            continue
        if not isinstance(rel_path, str) or len(rel_path.strip()) == 0:
            bail_fn("Invalid bundle path for default input '%s'" % key)
        joined = os.path.normpath(os.path.join(extract_dir, rel_path))
        abs_joined = os.path.abspath(joined)
        if os.path.commonpath([abs_extract_dir, abs_joined]) != abs_extract_dir:
            bail_fn("Refusing to resolve path outside %s bundle for key '%s': %s" % (bundle_flag_name, key, rel_path))
        if not os.path.exists(joined):
            bail_fn("%s manifest path for '%s' does not exist: %s" % (bundle_flag_name, key, rel_path))
        resolved_default_inputs[key] = joined
    return resolved_default_inputs


def apply_bundle_defaults_to_options(
    options,
    bundle_manifest,
    *,
    x_source_option_names=None,
    x_default_key="X_in",
    x_target_option_name="X_in",
    scalar_default_option_names=None,
):
    defaults = bundle_manifest.default_inputs
    applied = {}

    if x_source_option_names is None:
        x_source_option_names = ["X_in", "X_list", "Xd_in", "Xd_list"]
    if scalar_default_option_names is None:
        scalar_default_option_names = []

    has_explicit_x_source = any(getattr(options, key, None) is not None for key in x_source_option_names)
    if x_default_key in defaults and not has_explicit_x_source:
        setattr(options, x_target_option_name, [defaults[x_default_key]])
        applied[x_target_option_name] = defaults[x_default_key]

    for key in scalar_default_option_names:
        if key not in defaults:
            continue
        if getattr(options, key, None) is None:
            setattr(options, key, defaults[key])
            applied[key] = defaults[key]

    return BundleDefaultsApplication(bundle=bundle_manifest, applied_defaults=applied)


def load_and_apply_bundle_defaults(
    options,
    *,
    bundle_path,
    expected_schema,
    allowed_default_inputs,
    bundle_flag_name="--bundle-in",
    manifest_name="manifest.json",
    temp_prefix="bundle_in_",
    x_source_option_names=None,
    x_default_key="X_in",
    x_target_option_name="X_in",
    scalar_default_option_names=None,
    bail_fn=None,
):
    bundle = BundleManifest.load_defaults(
        bundle_path=bundle_path,
        expected_schema=expected_schema,
        allowed_default_inputs=allowed_default_inputs,
        bundle_flag_name=bundle_flag_name,
        manifest_name=manifest_name,
        temp_prefix=temp_prefix,
        bail_fn=bail_fn,
    )
    return apply_bundle_defaults_to_options(
        options,
        bundle,
        x_source_option_names=x_source_option_names,
        x_default_key=x_default_key,
        x_target_option_name=x_target_option_name,
        scalar_default_option_names=scalar_default_option_names,
    )


def ensure_parent_dir_for_file(path):
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def require_existing_nonempty_file(
    path,
    label,
    suggestion,
    *,
    option_name="--bundle-out",
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    bail_fn("Cannot write %s: missing %s (%s)" % (option_name, label, suggestion))


def stage_file_into_dir(source_path, stage_dir, bundle_name, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if source_path is None or not os.path.exists(source_path):
        bail_fn("Cannot stage missing file into bundle: %s" % source_path)
    staged_path = os.path.join(stage_dir, bundle_name)
    with open(source_path, "rb") as in_fh:
        with open(staged_path, "wb") as out_fh:
            shutil.copyfileobj(in_fh, out_fh)
    return staged_path


def write_bundle_from_specs(
    out_path,
    *,
    schema,
    source_tool,
    source_mode,
    source_argv,
    generated_file_specs,
    optional_existing_files=None,
    option_name="--bundle-out",
    temp_prefix="bundle_out_",
    manifest_name="manifest.json",
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail

    tar_mode = get_tar_write_mode_for_bundle_path(
        out_path,
        option_name=option_name,
        bail_fn=bail_fn,
    )
    ensure_parent_dir_for_file(out_path)

    with tempfile.TemporaryDirectory(prefix=temp_prefix) as stage_dir:
        file_map = {}
        file_meta = {}

        for (default_key, bundle_name, write_fn, label, suggestion) in generated_file_specs:
            staged_path = os.path.join(stage_dir, bundle_name)
            write_fn(staged_path)
            require_existing_nonempty_file(
                staged_path,
                label,
                suggestion,
                option_name=option_name,
                bail_fn=bail_fn,
            )
            file_map[default_key] = bundle_name
            file_meta[bundle_name] = collect_file_metadata(staged_path)

        for (default_key, source_path, bundle_name) in optional_existing_files or []:
            if source_path is None or not os.path.exists(source_path):
                continue
            staged_path = stage_file_into_dir(
                source_path,
                stage_dir,
                bundle_name,
                bail_fn=bail_fn,
            )
            file_map[default_key] = bundle_name
            file_meta[bundle_name] = collect_file_metadata(staged_path)

        manifest = BundleManifest.build(
            schema=schema,
            source_tool=source_tool,
            source_mode=source_mode,
            source_argv=source_argv,
            default_inputs=file_map,
            files_metadata=file_meta,
        )
        manifest.write_manifest(stage_dir, manifest_name=manifest_name)
        manifest.write_archive(
            out_path,
            tar_mode,
            stage_dir,
            file_meta.keys(),
            manifest_name=manifest_name,
        )

    return out_path


def hash_file_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def collect_file_metadata(path):
    return {
        "size_bytes": int(os.path.getsize(path)),
        "sha256": hash_file_sha256(path),
    }
