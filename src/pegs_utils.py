import json
import os
import shutil
import tarfile
import tempfile
import hashlib
import csv
import gzip
import io
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import scipy.stats

EAGGL_BUNDLE_SCHEMA = "pigean_eaggl_bundle/v1"
EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS = set([
    "X_in",
    "gene_stats_in",
    "gene_set_stats_in",
    "gene_phewas_bfs_in",
    "gene_set_phewas_stats_in",
])

DIG_OPEN_DATA_PREFIX = "dig-open-data:"
DIG_OPEN_DATA_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")


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


def urlopen_with_retry(
    file,
    flag=None,
    tries=5,
    delay=60,
    backoff=2,
    *,
    log_fn=None,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail

    while tries > 1:
        try:
            if flag is not None:
                return urllib.request.urlopen(file, flag)
            return urllib.request.urlopen(file)
        except urllib.error.URLError as e:
            if log_fn is not None:
                log_fn("%s, Retrying in %d seconds..." % (str(e), delay))
            time.sleep(delay)
            tries -= 1
            delay *= backoff
    bail_fn("Couldn't open file after too many retries")


def is_dig_open_data_uri(filepath):
    return isinstance(filepath, str) and filepath.startswith(DIG_OPEN_DATA_PREFIX)


def is_dig_open_data_ancestry_trait_spec(spec):
    if not isinstance(spec, str):
        return False
    if spec.count(":") != 1:
        return False
    ancestry, trait = spec.split(":", 1)
    if len(ancestry) == 0 or len(trait) == 0:
        return False
    if "/" in ancestry or "/" in trait:
        return False
    if not DIG_OPEN_DATA_TOKEN_RE.match(ancestry):
        return False
    if not DIG_OPEN_DATA_TOKEN_RE.match(trait):
        return False
    return True


def open_dig_open_data(uri, flag=None, *, log_fn=None, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail

    if flag is not None and "w" in flag:
        bail_fn("dig-open-data sources are read-only and cannot be opened for writing")

    spec = uri[len(DIG_OPEN_DATA_PREFIX):]
    if len(spec.strip()) == 0:
        bail_fn("Invalid dig-open-data source '%s'; expected dig-open-data:<ancestry>:<trait>" % uri)

    try:
        from dig_open_data import open_text, open_trait
    except ImportError:
        bail_fn("dig_open_data is required to read '%s'. Install https://github.com/flannick/dig-open-data/" % uri)

    if is_dig_open_data_ancestry_trait_spec(spec):
        ancestry, trait = spec.split(":", 1)
        if log_fn is not None:
            log_fn("Reading dig-open-data trait ancestry=%s trait=%s" % (ancestry, trait))
        return open_trait(ancestry, trait)

    if log_fn is not None:
        log_fn("Reading dig-open-data source %s" % spec)
    return open_text(spec)


def is_gz_file(filepath, is_remote, flag=None, *, urlopen_with_retry_fn=None):
    open_url_fn = urlopen_with_retry if urlopen_with_retry_fn is None else urlopen_with_retry_fn

    if len(filepath) >= 3 and (filepath[-3:] == ".gz" or filepath[-4:] == ".bgz") and (flag is None or "w" not in flag):
        try:
            if is_remote:
                test_fh = open_url_fn(filepath)
            else:
                test_fh = gzip.open(filepath, "rb")

            try:
                test_fh.readline()
                test_fh.close()
                return True
            except Exception:
                return False
        except FileNotFoundError:
            return True

    elif flag is None or "w" not in flag:
        test_flag = "rb"
        if is_remote:
            test_fh = open_url_fn(filepath, test_flag)
        else:
            test_fh = open(filepath, test_flag)

        gz_magic = test_fh.read(2) == b"\x1f\x8b"
        test_fh.close()
        return gz_magic

    return filepath[-3:] == ".gz" or filepath[-4:] == ".bgz"


def open_text_auto(
    file,
    flag=None,
    *,
    log_fn=None,
    bail_fn=None,
    urlopen_with_retry_fn=None,
    is_gz_file_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    open_url_fn = urlopen_with_retry if urlopen_with_retry_fn is None else urlopen_with_retry_fn
    detect_gz_fn = is_gz_file if is_gz_file_fn is None else is_gz_file_fn

    if is_dig_open_data_uri(file):
        return open_dig_open_data(file, flag=flag, log_fn=log_fn, bail_fn=bail_fn)

    is_remote = is_remote_path(file)

    try:
        is_gz = detect_gz_fn(
            file,
            is_remote,
            flag=flag,
            urlopen_with_retry_fn=open_url_fn,
        )
    except TypeError:
        # Backward-compatible path for legacy wrapper call signatures.
        is_gz = detect_gz_fn(file, is_remote, flag=flag)

    if is_gz:
        open_fun = gzip.open
        if flag is not None and len(flag) > 0 and not flag.endswith("t"):
            flag = "%st" % flag
        elif flag is None:
            flag = "rt"
    else:
        open_fun = open

    if is_remote:
        if flag is not None:
            if open_fun is open:
                fh = io.TextIOWrapper(open_url_fn(file, flag))
            else:
                fh = open_fun(open_url_fn(file), flag)
        else:
            if open_fun is open:
                fh = io.TextIOWrapper(open_url_fn(file))
            else:
                fh = open_fun(open_url_fn(file))
    else:
        if flag is not None:
            try:
                fh = open_fun(file, flag, encoding="utf-8")
            except LookupError:
                fh = open_fun(file, flag)
        else:
            try:
                fh = open_fun(file, encoding="utf-8")
            except LookupError:
                fh = open_fun(file)

    return fh


class TsvTable(object):
    def __init__(self, columns, rows, key_column=None, by_key=None):
        self.columns = columns
        self.rows = rows
        self.key_column = key_column
        self.by_key = by_key


def read_tsv(path, key_column=None, required_columns=None, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    required = set(required_columns or [])

    with open_text_auto(str(path), "rt", bail_fn=bail_fn) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            bail_fn("No header found in TSV: %s" % path)

        missing = required.difference(reader.fieldnames)
        if missing:
            missing_fmt = ", ".join(sorted(missing))
            bail_fn("Missing required columns (%s) in %s" % (missing_fmt, path))

        rows = []
        by_key = {} if key_column else None
        for row in reader:
            rows.append(row)
            if key_column:
                key = row.get(key_column, "")
                if key in by_key:
                    bail_fn("Duplicate key '%s' in %s (%s)" % (key, path, key_column))
                by_key[key] = row

    return TsvTable(columns=list(reader.fieldnames), rows=rows, key_column=key_column, by_key=by_key)


def write_tsv(path, columns, rows):
    path = str(path)
    ensure_parent_dir_for_file(path)
    cols = list(columns)
    if path.endswith(".gz"):
        with gzip.open(path, "wt", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def read_gene_stats(path, *, bail_fn=None):
    return read_tsv(path, key_column="Gene", required_columns=["Gene"], bail_fn=bail_fn)


def read_gene_set_stats(path, *, bail_fn=None):
    return read_tsv(path, key_column="Gene_Set", required_columns=["Gene_Set"], bail_fn=bail_fn)


def read_gene_phewas_stats(path, *, bail_fn=None):
    return read_tsv(path, key_column="Gene", required_columns=["Gene"], bail_fn=bail_fn)


def read_gene_set_phewas_stats(path, *, bail_fn=None):
    return read_tsv(path, key_column="Gene_Set", required_columns=["Gene_Set"], bail_fn=bail_fn)


def read_factor_phewas_stats(path, *, bail_fn=None):
    return read_tsv(path, required_columns=[], bail_fn=bail_fn)


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


def resolve_column_index(col_name_or_index, header_cols, require_match=True, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail

    try:
        col_ind = int(col_name_or_index)
    except (TypeError, ValueError):
        col_ind = None

    if col_ind is not None:
        if col_ind <= 0:
            bail_fn("All column ids specified as indices are 1-based")
        return col_ind - 1

    matching_cols = [i for i in range(0, len(header_cols)) if header_cols[i] == col_name_or_index]
    if len(matching_cols) == 0:
        if require_match:
            bail_fn("Could not find match for column %s in header: %s" % (col_name_or_index, "\t".join(header_cols)))
        else:
            return None
    if len(matching_cols) > 1:
        bail_fn("Found two matches for column %s in header: %s" % (col_name_or_index, "\t".join(header_cols)))
    return matching_cols[0]


def clean_chrom_name(chrom):
    if chrom is None:
        return chrom
    if len(chrom) >= 3 and chrom[:3] == "chr":
        return chrom[3:]
    return chrom


def construct_map_to_ind(values):
    return dict([(values[i], i) for i in range(len(values))])


def complete_p_beta_se(p, beta, se, *, warn_fn=None):
    if warn_fn is None:
        warn_fn = lambda _message: None

    p_none_mask = np.logical_or(p == None, np.isnan(p))
    beta_none_mask = np.logical_or(beta == None, np.isnan(beta))
    se_none_mask = np.logical_or(se == None, np.isnan(se))

    se_zero_mask = np.logical_and(~se_none_mask, se == 0)
    se_zero_beta_non_zero_mask = np.logical_and(se_zero_mask, np.logical_and(~beta_none_mask, beta != 0))

    if np.sum(se_zero_beta_non_zero_mask) != 0:
        warn_fn("%d variants had zero SEs; setting these to beta zero and se 1" % (np.sum(se_zero_beta_non_zero_mask)))
        beta[se_zero_beta_non_zero_mask] = 0
    se[se_zero_mask] = 1

    bad_mask = np.logical_and(np.logical_and(p_none_mask, beta_none_mask), se_none_mask)
    if np.sum(bad_mask) > 0:
        warn_fn("Couldn't infer p/beta/se at %d positions; setting these to beta zero and se 1" % (np.sum(bad_mask)))
        p[bad_mask] = 1
        beta[bad_mask] = 0
        se[bad_mask] = 1
        p_none_mask[bad_mask] = False
        beta_none_mask[bad_mask] = False
        se_none_mask[bad_mask] = False

    if np.sum(p_none_mask) > 0:
        p[p_none_mask] = 2 * scipy.stats.norm.pdf(-np.abs(beta[p_none_mask] / se[p_none_mask]))
    if np.sum(beta_none_mask) > 0:
        z = np.abs(scipy.stats.norm.ppf(np.array(p[beta_none_mask] / 2)))
        beta[beta_none_mask] = z * se[beta_none_mask]
    if np.sum(se_none_mask) > 0:
        z = np.abs(scipy.stats.norm.ppf(np.array(p[se_none_mask] / 2)))
        z[z == 0] = 1
        se[se_none_mask] = np.abs(beta[se_none_mask] / z)
    return (p, beta, se)


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


def load_bundle_defaults(
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
    return extract_dir, manifest, default_inputs


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


def build_bundle_manifest(
    schema,
    source_tool,
    source_mode,
    source_argv,
    default_inputs,
    files_metadata,
):
    return {
        "schema": schema,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "tool": source_tool,
            "mode": source_mode,
            "argv": list(source_argv),
        },
        "default_inputs": dict(default_inputs),
        "files": dict(files_metadata),
    }


def write_bundle_manifest_file(stage_dir, manifest, manifest_name="manifest.json"):
    manifest_path = os.path.join(stage_dir, manifest_name)
    with open(manifest_path, "w", encoding="utf-8") as out_fh:
        json.dump(manifest, out_fh, indent=2, sort_keys=True)
        out_fh.write("\n")
    return manifest_path


def write_bundle_archive(out_path, tar_mode, stage_dir, staged_file_names, *, manifest_name="manifest.json"):
    manifest_path = os.path.join(stage_dir, manifest_name)
    with tarfile.open(out_path, tar_mode) as tar_fh:
        tar_fh.add(manifest_path, arcname=manifest_name)
        for bundle_name in sorted(staged_file_names):
            tar_fh.add(os.path.join(stage_dir, bundle_name), arcname=bundle_name)


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
