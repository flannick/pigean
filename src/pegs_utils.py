import json
import os

import numpy as np


def _default_bail(message):
    raise ValueError(message)


def merge_dicts(base_value, override_value):
    if not isinstance(base_value, dict):
        base_value = {}
    merged = dict(base_value)
    for key, value in override_value.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json_config(config_path, bail_fn=None, seen_paths=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if seen_paths is None:
        seen_paths = set()

    abs_path = os.path.abspath(config_path)
    if abs_path in seen_paths:
        bail_fn("Detected circular config include at %s" % abs_path)
    seen_paths.add(abs_path)

    with open(abs_path) as cfg_fh:
        cfg = json.load(cfg_fh)

    if not isinstance(cfg, dict):
        bail_fn("Config file must contain a JSON object: %s" % abs_path)

    includes = cfg.get("include")
    if includes is None:
        return cfg

    include_list = includes if isinstance(includes, list) else [includes]
    merged = {}
    cfg_dir = os.path.dirname(abs_path)
    for include_file in include_list:
        if not isinstance(include_file, str):
            bail_fn("Config include entries must be strings in %s" % abs_path)
        include_path = include_file
        if not os.path.isabs(include_path):
            include_path = os.path.normpath(os.path.join(cfg_dir, include_path))
        include_cfg = load_json_config(include_path, bail_fn=bail_fn, seen_paths=seen_paths)
        merged = merge_dicts(merged, include_cfg)

    cfg = dict(cfg)
    del cfg["include"]
    return merge_dicts(merged, cfg)


def is_remote_path(value):
    if not isinstance(value, str):
        return False
    lower = value.lower()
    return lower.startswith("http:") or lower.startswith("https:") or lower.startswith("ftp:")


def resolve_config_path_value(value, config_dir):
    if not isinstance(value, str):
        return value
    if value == "":
        return value
    if is_remote_path(value):
        return value
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(config_dir, expanded))


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]
    if isinstance(value, set):
        return [json_safe(x) for x in sorted(value)]
    if isinstance(value, tuple):
        return [json_safe(x) for x in value]
    if isinstance(value, list):
        return [json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return value
