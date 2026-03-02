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


def iter_parser_options(parser):
    for option in parser.option_list:
        if option is not None and option.dest is not None:
            yield option
    for group in parser.option_groups:
        for option in group.option_list:
            if option is not None and option.dest is not None:
                yield option


def collect_cli_specified_dests(argv, parser):
    option_lookup = {}
    for option in iter_parser_options(parser):
        for long_opt in option._long_opts:
            option_lookup[long_opt] = option

    specified_dests = set()
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            break
        if arg.startswith("--"):
            opt_token = arg.split("=", 1)[0]
            if opt_token in option_lookup:
                opt_obj = option_lookup[opt_token]
                if opt_obj.dest is not None:
                    specified_dests.add(opt_obj.dest)
                if "=" not in arg and opt_obj.takes_value() and i + 1 < len(argv):
                    i += 1
        i += 1
    return specified_dests


def coerce_config_value(option, raw_value, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail

    def _cast_scalar(scalar):
        if scalar is None:
            return None
        if option.type == "int":
            return int(scalar)
        if option.type == "float":
            return float(scalar)
        return scalar

    if option.action == "append":
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        return [_cast_scalar(v) for v in values]

    if option.action in ("store_true", "store_false"):
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str):
            lower = raw_value.strip().lower()
            if lower in ("1", "true", "yes", "y", "on"):
                return True
            if lower in ("0", "false", "no", "n", "off"):
                return False
        bail_fn("Config value for %s must be boolean" % (option.dest))

    if option.action == "callback":
        return raw_value

    return _cast_scalar(raw_value)
