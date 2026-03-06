import json
import os
import sys

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


def is_path_like_dest(dest):
    if dest is None:
        return False
    dest_lower = dest.lower()
    return (
        dest_lower.endswith("_in")
        or dest_lower.endswith("_out")
        or dest_lower.endswith("_file")
        or "_file_" in dest_lower
        or dest_lower in ("log_file", "warnings_file", "config")
    )


def emit_stderr_warning(message):
    sys.stderr.write("Warning: %s\n" % message)
    sys.stderr.flush()


def callback_set_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(","))


def callback_set_comma_separated_args_as_float(option, opt, value, parser):
    setattr(parser.values, option.dest, [float(x) for x in value.split(",")])


def callback_set_comma_separated_args_as_set(option, opt, value, parser):
    setattr(parser.values, option.dest, set(value.split(",")))


def open_optional_log_handle(filepath, default_stream=None, mode="w"):
    if filepath is not None:
        return open(filepath, mode)
    if default_stream is not None:
        return default_stream
    return sys.stderr


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


def _format_moved_tool_name(replacement):
    if not isinstance(replacement, str):
        return None
    if not replacement.startswith("__MOVED_TO_"):
        return None
    tool = replacement[len("__MOVED_TO_"):].strip("_")
    if len(tool) == 0:
        return None
    return tool.lower()


def format_removed_option_message(option_name, replacement, context, config_path=None):
    moved_tool = _format_moved_tool_name(replacement)
    if context == "cli":
        if moved_tool is not None:
            return "Error: option %s moved to %s.py after repository split; run this in the %s repository" % (
                option_name,
                moved_tool,
                moved_tool,
            )
        if replacement is None:
            return "Error: option %s has been removed and is no longer supported" % option_name
        return "Error: option %s has been removed; use %s instead" % (option_name, replacement)

    if context == "config":
        if moved_tool is not None:
            return "Config key '%s' moved to %s.py after repository split; run this in the %s repository" % (
                option_name,
                moved_tool,
                moved_tool,
            )
        if replacement is None:
            return "Config key '%s' has been removed in %s and is no longer supported" % (option_name, config_path)
        replacement_config_key = replacement[2:].replace("-", "_") if isinstance(replacement, str) and replacement.startswith("--") else replacement
        return "Config key '%s' has been removed in %s; use '%s' (CLI: %s) instead" % (
            option_name,
            config_path,
            replacement_config_key,
            replacement,
        )

    raise ValueError("Unknown removed-option message context '%s'" % context)


def fail_removed_cli_aliases(
    argv,
    removed_option_replacements,
    *,
    format_removed_option_message_fn=None,
    stderr_write_fn=None,
    exit_fn=None,
):
    formatter = format_removed_option_message if format_removed_option_message_fn is None else format_removed_option_message_fn
    write_fn = sys.stderr.write if stderr_write_fn is None else stderr_write_fn
    terminate_fn = sys.exit if exit_fn is None else exit_fn

    for arg in argv:
        if not isinstance(arg, str) or not arg.startswith("--"):
            continue
        flag = arg.split("=", 1)[0]
        normalized = flag[2:].replace("-", "_")
        if normalized in removed_option_replacements:
            replacement = removed_option_replacements[normalized]
            write_fn("%s\n" % formatter(flag, replacement, context="cli"))
            terminate_fn(2)


def apply_cli_config_overrides(
    options_obj,
    args,
    parser,
    argv,
    *,
    resolve_path_fn,
    is_path_like_dest_fn,
    early_warn_fn,
    bail_fn,
    removed_option_replacements,
    format_removed_option_message_fn,
    track_config_specified_dests=False,
):
    cli_specified_dests = collect_cli_specified_dests(argv, parser)
    config_specified_dests = set() if track_config_specified_dests else None
    config_mode = None

    if getattr(options_obj, "config", None) is not None:
        config_path = resolve_path_fn(options_obj.config, os.getcwd())
        options_obj.config = config_path
        config_dir = os.path.dirname(config_path)
        config_data = load_json_config(config_path, bail_fn=bail_fn, seen_paths=None)
        config_mode = config_data["mode"] if "mode" in config_data else None

        if "options" in config_data:
            config_options = config_data["options"]
            if not isinstance(config_options, dict):
                bail_fn("Config key 'options' must be a JSON object")
        else:
            config_options = dict(config_data)
            config_options.pop("mode", None)
            config_options.pop("include", None)

        apply_config_option_overrides(
            options_obj,
            parser,
            config_options,
            config_path,
            config_dir,
            cli_specified_dests,
            resolve_path_fn=resolve_path_fn,
            is_path_like_dest_fn=is_path_like_dest_fn,
            early_warn_fn=early_warn_fn,
            bail_fn=bail_fn,
            removed_option_replacements=removed_option_replacements,
            format_removed_option_message_fn=format_removed_option_message_fn,
            config_specified_dests=config_specified_dests,
        )

    return options_obj, args, config_mode, cli_specified_dests, config_specified_dests


def harmonize_cli_mode_args(args, config_mode, *, early_warn_fn=None):
    resolved_args = list(args)
    if config_mode is None:
        return resolved_args
    if len(resolved_args) < 1:
        return [config_mode]
    if resolved_args[0] != config_mode and early_warn_fn is not None:
        early_warn_fn("Config mode '%s' differs from CLI mode '%s'; using CLI mode" % (config_mode, resolved_args[0]))
    return resolved_args


def coerce_option_int_list(values, option_name, bail_fn):
    try:
        return [int(x) for x in values]
    except Exception:
        bail_fn("option %s: invalid integer list %s" % (option_name, values))


def initialize_cli_logging(options_obj, *, stderr_stream=None, default_debug_level=1):
    if stderr_stream is None:
        stderr_stream = sys.stderr

    log_fh = open_optional_log_handle(
        getattr(options_obj, "log_file", None),
        default_stream=stderr_stream,
        mode="w",
    )
    warnings_fh = open_optional_log_handle(
        getattr(options_obj, "warnings_file", None),
        default_stream=stderr_stream,
        mode="w",
    )

    NONE = 0
    INFO = 1
    DEBUG = 2
    TRACE = 3
    debug_level = default_debug_level if getattr(options_obj, "debug_level", None) is None else options_obj.debug_level

    def log(message, level=INFO, end_char="\n"):
        if level <= debug_level:
            log_fh.write("%s%s" % (message, end_char))
            log_fh.flush()

    def warn(message):
        if warnings_fh is not None:
            warnings_fh.write("Warning: %s\n" % message)
            warnings_fh.flush()
        log(message, level=INFO)

    return {
        "NONE": NONE,
        "INFO": INFO,
        "DEBUG": DEBUG,
        "TRACE": TRACE,
        "debug_level": debug_level,
        "log_fh": log_fh,
        "warnings_fh": warnings_fh,
        "log": log,
        "warn": warn,
    }


def configure_random_seed(options_obj, random_module, numpy_module, log_fn=None, info_level=None):
    if getattr(options_obj, "deterministic", False) and getattr(options_obj, "seed", None) is None:
        options_obj.seed = 0

    if getattr(options_obj, "seed", None) is not None:
        random_module.seed(options_obj.seed)
        numpy_module.random.seed(options_obj.seed)
        if log_fn is not None:
            if info_level is None:
                log_fn("Using deterministic random seed %d" % options_obj.seed)
            else:
                log_fn("Using deterministic random seed %d" % options_obj.seed, info_level)


def apply_config_option_overrides(
    options_obj,
    parser,
    config_options,
    config_path,
    config_dir,
    cli_specified_dests,
    *,
    resolve_path_fn,
    is_path_like_dest_fn,
    early_warn_fn,
    bail_fn,
    removed_option_replacements=None,
    format_removed_option_message_fn=None,
    config_specified_dests=None,
):
    dest_to_option = {}
    long_key_to_dest = {}
    for opt in iter_parser_options(parser):
        dest_to_option[opt.dest] = opt
        for long_opt in opt._long_opts:
            key = long_opt.lstrip("-")
            long_key_to_dest[key] = opt.dest
            long_key_to_dest[key.replace("-", "_")] = opt.dest

    for raw_key, raw_value in config_options.items():
        if raw_key in ("mode", "options", "include"):
            continue

        normalized_config_key = raw_key
        if isinstance(normalized_config_key, str):
            if normalized_config_key.startswith("--"):
                normalized_config_key = normalized_config_key[2:]
            normalized_config_key = normalized_config_key.replace("-", "_")

        if (
            removed_option_replacements is not None
            and normalized_config_key in removed_option_replacements
        ):
            replacement = removed_option_replacements[normalized_config_key]
            if format_removed_option_message_fn is not None:
                bail_fn(
                    format_removed_option_message_fn(
                        raw_key,
                        replacement,
                        context="config",
                        config_path=config_path,
                    )
                )
            if replacement is None:
                bail_fn(
                    "Config key '%s' has been removed in %s and is no longer supported"
                    % (raw_key, config_path)
                )
            replacement_config_key = (
                replacement[2:].replace("-", "_")
                if isinstance(replacement, str) and replacement.startswith("--")
                else replacement
            )
            bail_fn(
                "Config key '%s' has been removed in %s; use '%s' (CLI: %s) instead"
                % (raw_key, config_path, replacement_config_key, replacement)
            )

        key = raw_key[2:] if isinstance(raw_key, str) and raw_key.startswith("--") else raw_key
        key_norm = key.replace("-", "_") if isinstance(key, str) else key
        if key in dest_to_option:
            dest = key
        elif key_norm in dest_to_option:
            dest = key_norm
        elif key in long_key_to_dest:
            dest = long_key_to_dest[key]
        elif key_norm in long_key_to_dest:
            dest = long_key_to_dest[key_norm]
        else:
            early_warn_fn("Ignoring unknown config key '%s' in %s" % (raw_key, config_path))
            continue

        if dest in cli_specified_dests:
            continue

        opt = dest_to_option[dest]
        coerced_value = coerce_config_value(opt, raw_value, bail_fn=bail_fn)
        if is_path_like_dest_fn(dest):
            if isinstance(coerced_value, list):
                coerced_value = [
                    resolve_path_fn(v, config_dir) if isinstance(v, str) else v
                    for v in coerced_value
                ]
            elif isinstance(coerced_value, str):
                coerced_value = resolve_path_fn(coerced_value, config_dir)

        setattr(options_obj, dest, coerced_value)
        if config_specified_dests is not None:
            config_specified_dests.add(dest)


def get_tar_write_mode_for_bundle_path(bundle_path, option_name="--eaggl-bundle-out", bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    lower = bundle_path.lower()
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        return "w:gz"
    if lower.endswith(".tar"):
        return "w"
    bail_fn("Option %s must end with .tar, .tar.gz, or .tgz" % option_name)
