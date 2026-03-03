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
import copy
from dataclasses import dataclass, field

import numpy as np
import scipy.stats
import scipy.sparse as sparse

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


@dataclass
class XData:
    X_orig: object = None
    X_orig_missing_genes: object = None
    X_orig_missing_gene_sets: object = None
    X_orig_missing_genes_missing_gene_sets: object = None
    genes: list = field(default_factory=list)
    genes_missing: list = field(default_factory=list)
    gene_sets: list = field(default_factory=list)
    gene_sets_missing: list = field(default_factory=list)
    gene_sets_ignored: list = field(default_factory=list)
    gene_to_ind: dict = field(default_factory=dict)
    gene_set_to_ind: dict = field(default_factory=dict)
    scale_factors: object = None
    mean_shifts: object = None
    gene_set_batches: object = None
    gene_set_labels: object = None
    is_dense_gene_set: object = None


@dataclass
class XInputPlan:
    initial_ps: object
    X_ins: list
    batches: list
    labels: list
    orig_files: list
    is_dense: list


@dataclass
class ParsedGeneSetStats:
    need_to_take_log: bool
    has_beta_tilde: bool
    has_p_or_se: bool
    has_beta: bool
    has_beta_uncorrected: bool
    records: dict


@dataclass
class ParsedGeneBfs:
    gene_in_bfs: dict
    gene_in_combined: object
    gene_in_priors: object


@dataclass
class ParsedGenePhewasBfs:
    phenos: list
    pheno_to_ind: dict
    row: object
    col: object
    Ys: object
    combineds: object
    priors: object
    num_filtered: int


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


def open_text_with_retry(filepath, flag=None, *, log_fn=None, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _msg: None

    open_url_with_retry_fn = lambda _file, _flag=None: urlopen_with_retry(
        _file,
        flag=_flag,
        log_fn=log_fn,
        bail_fn=bail_fn,
    )
    return open_text_auto(
        filepath,
        flag=flag,
        log_fn=log_fn,
        bail_fn=bail_fn,
        urlopen_with_retry_fn=open_url_with_retry_fn,
    )


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


def parse_gene_set_statistics_file(
    stats_in,
    *,
    stats_id_col,
    stats_exp_beta_tilde_col,
    stats_beta_tilde_col,
    stats_p_col,
    stats_se_col,
    stats_beta_col,
    stats_beta_uncorrected_col,
    ignore_negative_exp_beta,
    max_gene_set_p,
    min_gene_set_beta,
    min_gene_set_beta_uncorrected,
    open_text_fn,
    get_col_fn,
    log_fn=None,
    warn_fn=None,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _msg: None
    if warn_fn is None:
        warn_fn = lambda _msg: None

    if stats_in is None:
        bail_fn("Require --stats-in for this operation")

    log_fn("Reading --stats-in file %s" % stats_in)
    need_to_take_log = False
    records = {}

    with open_text_fn(stats_in) as stats_fh:
        header_cols = stats_fh.readline().strip("\n").split()
        id_col = get_col_fn(stats_id_col, header_cols)
        beta_tilde_col = None

        if stats_beta_tilde_col is not None:
            beta_tilde_col = get_col_fn(stats_beta_tilde_col, header_cols, False)
        if beta_tilde_col is not None:
            log_fn("Using col %s for beta_tilde values" % stats_beta_tilde_col)
        elif stats_exp_beta_tilde_col is not None:
            beta_tilde_col = get_col_fn(stats_exp_beta_tilde_col, header_cols)
            need_to_take_log = True
            if beta_tilde_col is not None:
                log_fn("Using %s for exp(beta_tilde) values" % stats_exp_beta_tilde_col)
            else:
                bail_fn(
                    "Could not find beta_tilde column %s or %s in header: %s"
                    % (stats_beta_tilde_col, stats_exp_beta_tilde_col, "\t".join(header_cols))
                )

        p_col = None
        if stats_p_col is not None:
            p_col = get_col_fn(stats_p_col, header_cols, False)

        se_col = None
        if stats_se_col is not None:
            se_col = get_col_fn(stats_se_col, header_cols, False)

        beta_col = None
        if stats_beta_col is not None:
            beta_col = get_col_fn(stats_beta_col, header_cols, True)
        else:
            beta_col = get_col_fn("beta", header_cols, False)

        beta_uncorrected_col = None
        if stats_beta_uncorrected_col is not None:
            beta_uncorrected_col = get_col_fn(stats_beta_uncorrected_col, header_cols, True)
        else:
            beta_uncorrected_col = get_col_fn("beta_uncorrected", header_cols, False)

        if (
            se_col is None
            and p_col is None
            and beta_tilde_col is None
            and beta_col is None
            and beta_uncorrected_col is None
        ):
            bail_fn("Require at least something to read from --gene-set-stats-in")

        for line in stats_fh:
            beta_tilde = None
            p = None
            se = None
            z = None
            beta = None
            beta_uncorrected = None

            cols = line.strip("\n").split()
            if (
                id_col > len(cols)
                or (beta_tilde_col is not None and beta_tilde_col > len(cols))
                or (p_col is not None and p_col > len(cols))
                or (se_col is not None and se_col > len(cols))
            ):
                warn_fn("Skipping due to too few columns in line: %s" % line)
                continue

            gene_set = cols[id_col]
            if gene_set in records:
                warn_fn("Already seen gene set %s; only considering first instance" % (gene_set))
                continue

            if beta_tilde_col is not None:
                try:
                    beta_tilde = float(cols[beta_tilde_col])
                except ValueError:
                    if cols[beta_tilde_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible beta_tilde value %s for gene_set %s"
                            % (cols[beta_tilde_col], gene_set)
                        )
                    continue

                if need_to_take_log:
                    if beta_tilde < 0:
                        if ignore_negative_exp_beta:
                            continue
                        bail_fn(
                            "Exp(beta) value %s for gene set %s is < 0; did you mean to specify --stats-beta-col? Otherwise, specify --ignore-negative-exp-beta to ignore these"
                            % (beta_tilde, gene_set)
                        )
                    beta_tilde = np.log(beta_tilde)

            if se_col is not None:
                try:
                    se = float(cols[se_col])
                except ValueError:
                    if cols[se_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible se value %s for gene_set %s"
                            % (cols[se_col], gene_set)
                        )
                    continue

                if beta_tilde_col is not None:
                    z = beta_tilde / se
                    p = 2 * scipy.stats.norm.cdf(-np.abs(z))
                    if max_gene_set_p is not None and p > max_gene_set_p:
                        continue
            elif p_col is not None:
                try:
                    p = float(cols[p_col])
                    if max_gene_set_p is not None and p > max_gene_set_p:
                        continue
                except ValueError:
                    if cols[p_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible p value %s for gene_set %s"
                            % (cols[p_col], gene_set)
                        )
                    continue

                z = np.abs(scipy.stats.norm.ppf(p / 2))
                if z == 0:
                    warn_fn("Skipping gene_set %s due to 0 z-score" % (gene_set))
                    continue

                if beta_tilde_col is not None:
                    se = np.abs(beta_tilde) / z

            if beta_col is not None:
                try:
                    beta = float(cols[beta_col])
                    if min_gene_set_beta is not None and beta < min_gene_set_beta:
                        continue
                except ValueError:
                    if cols[beta_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible beta value %s for gene_set %s"
                            % (cols[beta_col], gene_set)
                        )
                    continue

            if beta_uncorrected_col is not None:
                try:
                    beta_uncorrected = float(cols[beta_uncorrected_col])
                    if (
                        min_gene_set_beta_uncorrected is not None
                        and beta_uncorrected < min_gene_set_beta_uncorrected
                    ):
                        continue
                except ValueError:
                    if cols[beta_uncorrected_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible beta_uncorrected value %s for gene_set %s"
                            % (cols[beta_uncorrected_col], gene_set)
                        )
                    continue

            records[gene_set] = (
                beta_tilde,
                p,
                se,
                z,
                beta,
                beta_uncorrected,
            )

    return ParsedGeneSetStats(
        need_to_take_log=need_to_take_log,
        has_beta_tilde=beta_tilde_col is not None,
        has_p_or_se=(p_col is not None or se_col is not None),
        has_beta=beta_col is not None,
        has_beta_uncorrected=beta_uncorrected_col is not None,
        records=records,
    )


def parse_gene_bfs_file(
    gene_bfs_in,
    *,
    gene_bfs_id_col,
    gene_bfs_log_bf_col,
    gene_bfs_combined_col,
    gene_bfs_prob_col,
    gene_bfs_prior_col,
    background_log_bf,
    gene_label_map,
    open_text_fn,
    get_col_fn,
    log_fn=None,
    warn_fn=None,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _msg: None
    if warn_fn is None:
        warn_fn = lambda _msg: None

    if gene_bfs_in is None:
        bail_fn("Require --gene-stats-in for this operation")

    log_fn("Reading --gene-stats-in file %s" % gene_bfs_in)
    gene_in_bfs = {}
    gene_in_combined = None
    gene_in_priors = None
    with open_text_fn(gene_bfs_in) as gene_bfs_fh:
        header_cols = gene_bfs_fh.readline().strip("\n").split()
        if gene_bfs_id_col is None:
            gene_bfs_id_col = "Gene"

        id_col = get_col_fn(gene_bfs_id_col, header_cols)

        prob_col = None
        if gene_bfs_prob_col is not None:
            prob_col = get_col_fn(gene_bfs_prob_col, header_cols, True)

        bf_col = None
        if gene_bfs_log_bf_col is not None:
            bf_col = get_col_fn(gene_bfs_log_bf_col, header_cols)
        else:
            if prob_col is None:
                bf_col = get_col_fn("log_bf", header_cols)

        if bf_col is None and prob_col is None:
            bail_fn("--gene-stats-log-bf-col or --gene-stats-prob-col required for this operation")

        combined_col = None
        if gene_bfs_combined_col is not None:
            combined_col = get_col_fn(gene_bfs_combined_col, header_cols, True)
        else:
            combined_col = get_col_fn("combined", header_cols, False)

        prior_col = None
        if gene_bfs_prior_col is not None:
            prior_col = get_col_fn(gene_bfs_prior_col, header_cols, True)
        else:
            prior_col = get_col_fn("prior", header_cols, False)

        if combined_col is not None or prob_col is not None:
            gene_in_combined = {}
        if prior_col is not None:
            gene_in_priors = {}

        for line in gene_bfs_fh:
            cols = line.strip("\n").split()
            if (
                id_col >= len(cols)
                or (bf_col is not None and bf_col >= len(cols))
                or (combined_col is not None and combined_col >= len(cols))
                or (prob_col is not None and prob_col >= len(cols))
                or (prior_col is not None and prior_col >= len(cols))
            ):
                warn_fn("Skipping due to too few columns in line: %s" % line)
                continue

            gene = cols[id_col]
            if gene_label_map is not None and gene in gene_label_map:
                gene = gene_label_map[gene]

            if bf_col is not None:
                try:
                    bf = float(cols[bf_col])
                except ValueError:
                    if cols[bf_col] != "NA":
                        warn_fn("Skipping unconvertible bf value %s for gene %s" % (cols[bf_col], gene))
                    continue
            elif prob_col is not None:
                try:
                    prob = float(cols[prob_col])
                except ValueError:
                    if cols[prob_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible prob value %s for gene %s"
                            % (cols[prob_col], gene)
                        )
                    continue
                if prob <= 0 or prob >= 1:
                    warn_fn("Skipping probability %.3g outside of (0,1)" % (prob))
                    continue
                bf = np.log(prob / (1 - prob)) - background_log_bf

            gene_in_bfs[gene] = bf

            if combined_col is not None:
                try:
                    combined = float(cols[combined_col])
                except ValueError:
                    if cols[combined_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible combined value %s for gene %s"
                            % (cols[combined_col], gene)
                        )
                    continue
                gene_in_combined[gene] = combined

            if prior_col is not None:
                try:
                    prior = float(cols[prior_col])
                except ValueError:
                    if cols[prior_col] != "NA":
                        warn_fn(
                            "Skipping unconvertible prior value %s for gene %s"
                            % (cols[prior_col], gene)
                        )
                    continue
                gene_in_priors[gene] = prior

    return ParsedGeneBfs(
        gene_in_bfs=gene_in_bfs,
        gene_in_combined=gene_in_combined,
        gene_in_priors=gene_in_priors,
    )


def parse_gene_phewas_bfs_file(
    gene_phewas_bfs_in,
    *,
    gene_phewas_bfs_id_col,
    gene_phewas_bfs_pheno_col,
    gene_phewas_bfs_log_bf_col,
    gene_phewas_bfs_combined_col,
    gene_phewas_bfs_prior_col,
    min_value,
    max_num_entries_at_once,
    existing_phenos,
    existing_pheno_to_ind,
    gene_to_ind,
    gene_label_map,
    phewas_gene_to_x_gene,
    open_text_fn,
    get_col_fn,
    bail_fn=None,
    warn_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if warn_fn is None:
        warn_fn = lambda _msg: None

    if max_num_entries_at_once is None:
        max_num_entries_at_once = 200 * 10000

    success = False
    num_filtered = 0
    final_phenos = list(existing_phenos) if existing_phenos is not None else []
    final_pheno_to_ind = copy.copy(existing_pheno_to_ind) if existing_pheno_to_ind is not None else {}
    final_row = np.array([], dtype=np.int32)
    final_col = np.array([], dtype=np.int32)
    final_Ys = None
    final_combineds = None
    final_priors = None

    for delim in [None, "\t"]:
        success = True
        Ys = None
        combineds = None
        priors = None

        row = []
        col = []
        row_chunks = []
        col_chunks = []
        Y_chunks = []
        combined_chunks = []
        prior_chunks = []

        with open_text_fn(gene_phewas_bfs_in) as gene_phewas_bfs_fh:
            header_cols = gene_phewas_bfs_fh.readline().strip("\n").split(delim)
            id_col_name = gene_phewas_bfs_id_col if gene_phewas_bfs_id_col is not None else "Gene"
            pheno_col_name = gene_phewas_bfs_pheno_col if gene_phewas_bfs_pheno_col is not None else "Pheno"

            id_col = get_col_fn(id_col_name, header_cols)
            pheno_col = get_col_fn(pheno_col_name, header_cols)

            if gene_phewas_bfs_log_bf_col is not None:
                bf_col = get_col_fn(gene_phewas_bfs_log_bf_col, header_cols)
            else:
                bf_col = get_col_fn("log_bf", header_cols, False)

            if gene_phewas_bfs_combined_col is not None:
                combined_col = get_col_fn(gene_phewas_bfs_combined_col, header_cols, True)
            else:
                combined_col = get_col_fn("combined", header_cols, False)

            if gene_phewas_bfs_prior_col is not None:
                prior_col = get_col_fn(gene_phewas_bfs_prior_col, header_cols, True)
            else:
                prior_col = get_col_fn("prior", header_cols, False)

            if bf_col is not None:
                Ys = []
            if combined_col is not None:
                combineds = []
            if prior_col is not None:
                priors = []

            def _flush_chunks():
                if len(row) == 0:
                    return
                row_chunks.append(np.array(row, dtype=np.int32))
                col_chunks.append(np.array(col, dtype=np.int32))
                if Ys is not None:
                    Y_chunks.append(np.array(Ys, dtype=np.float64))
                    Ys[:] = []
                if combineds is not None:
                    combined_chunks.append(np.array(combineds, dtype=np.float64))
                    combineds[:] = []
                if priors is not None:
                    prior_chunks.append(np.array(priors, dtype=np.float64))
                    priors[:] = []
                row[:] = []
                col[:] = []

            phenos = list(existing_phenos) if existing_phenos is not None else []
            pheno_to_ind = (
                copy.copy(existing_pheno_to_ind) if existing_pheno_to_ind is not None else {}
            )
            num_filtered = 0

            for line in gene_phewas_bfs_fh:
                cols = line.strip("\n").split(delim)
                if len(cols) != len(header_cols):
                    success = False
                    continue

                if (
                    id_col >= len(cols)
                    or pheno_col >= len(cols)
                    or (bf_col is not None and bf_col >= len(cols))
                    or (combined_col is not None and combined_col >= len(cols))
                    or (prior_col is not None and prior_col >= len(cols))
                ):
                    warn_fn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene = cols[id_col]
                pheno = cols[pheno_col]

                cur_combined = None
                if combined_col is not None:
                    try:
                        combined = float(cols[combined_col])
                    except ValueError:
                        if cols[combined_col] != "NA":
                            warn_fn(
                                "Skipping unconvertible value %s for gene_set %s"
                                % (cols[combined_col], gene)
                            )
                        continue

                    if min_value is not None and combined < min_value:
                        num_filtered += 1
                        continue
                    cur_combined = combined

                cur_Y = None
                if bf_col is not None:
                    try:
                        bf = float(cols[bf_col])
                    except ValueError:
                        if cols[bf_col] != "NA":
                            warn_fn(
                                "Skipping unconvertible value %s for gene %s and pheno %s"
                                % (cols[bf_col], gene, pheno)
                            )
                        continue

                    if min_value is not None and combined_col is None and bf < min_value:
                        num_filtered += 1
                        continue
                    cur_Y = bf

                cur_prior = None
                if prior_col is not None:
                    try:
                        prior = float(cols[prior_col])
                    except ValueError:
                        if cols[prior_col] != "NA":
                            warn_fn(
                                "Skipping unconvertible value %s for gene %s"
                                % (cols[prior_col], gene)
                            )
                        continue

                    if min_value is not None and combined_col is None and bf_col is None and prior < min_value:
                        num_filtered += 1
                        continue
                    cur_prior = prior

                if pheno not in pheno_to_ind:
                    pheno_to_ind[pheno] = len(phenos)
                    phenos.append(pheno)
                pheno_ind = pheno_to_ind[pheno]

                if gene_label_map is not None and gene in gene_label_map:
                    gene = gene_label_map[gene]

                mapped_genes = [gene]
                if phewas_gene_to_x_gene is not None and gene in phewas_gene_to_x_gene:
                    mapped_genes = list(phewas_gene_to_x_gene[gene])

                for cur_gene in mapped_genes:
                    if cur_gene not in gene_to_ind:
                        continue
                    if combineds is not None:
                        combineds.append(cur_combined)
                    if Ys is not None:
                        Ys.append(cur_Y)
                    if priors is not None:
                        priors.append(cur_prior)

                    col.append(pheno_ind)
                    row.append(gene_to_ind[cur_gene])
                    if len(row) >= max_num_entries_at_once:
                        _flush_chunks()

            _flush_chunks()

        if success:
            final_phenos = phenos
            final_pheno_to_ind = pheno_to_ind
            if len(row_chunks) > 0:
                row = np.concatenate(row_chunks)
                col = np.concatenate(col_chunks)
            else:
                row = np.array([], dtype=np.int32)
                col = np.array([], dtype=np.int32)

            if len(row) > 0:
                key = row.astype(np.int64) * int(len(phenos)) + col.astype(np.int64)
                _, unique_indices = np.unique(key, return_index=True)
            else:
                unique_indices = np.array([], dtype=np.int64)

            if len(unique_indices) < len(row):
                warn_fn("Found %d duplicate values; ignoring duplicates" % (len(row) - len(unique_indices)))

            final_row = row[unique_indices]
            final_col = col[unique_indices]

            if combineds is not None:
                if len(combined_chunks) > 0:
                    final_combineds = np.concatenate(combined_chunks)[unique_indices]
                else:
                    final_combineds = np.array([], dtype=np.float64)
            else:
                final_combineds = None

            if Ys is not None:
                if len(Y_chunks) > 0:
                    final_Ys = np.concatenate(Y_chunks)[unique_indices]
                else:
                    final_Ys = np.array([], dtype=np.float64)
            else:
                final_Ys = None

            if priors is not None:
                if len(prior_chunks) > 0:
                    final_priors = np.concatenate(prior_chunks)[unique_indices]
                else:
                    final_priors = np.array([], dtype=np.float64)
            else:
                final_priors = None
            break

    if not success:
        bail_fn("Error: different number of columns in header row and non header rows")

    return ParsedGenePhewasBfs(
        phenos=final_phenos,
        pheno_to_ind=final_pheno_to_ind,
        row=final_row,
        col=final_col,
        Ys=final_Ys,
        combineds=final_combineds,
        priors=final_priors,
        num_filtered=num_filtered,
    )


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


def remove_tag_from_input(x_in, tag_separator=":"):
    tag = None
    if tag_separator in x_in:
        tag_index = x_in.index(tag_separator)
        tag = x_in[:tag_index]
        x_in = x_in[tag_index + 1 :]
        if len(tag) == 0:
            tag = None
    return (x_in, tag)


def add_tag_to_input(x_in, tag, tag_separator=":"):
    if tag is None:
        return x_in
    return tag_separator.join([tag, x_in])


def assign_default_batches(batches, orig_files, batch_all_for_hyper, first_for_hyper):
    batches = list(batches)
    used_batches = set([str(b) for b in batches if b is not None])
    next_batch_num = 1

    def _generate_new_batch(new_batch_num):
        new_batch = "BATCH%d" % new_batch_num
        while new_batch in used_batches:
            new_batch_num += 1
            new_batch = "BATCH%d" % new_batch_num
        used_batches.add(new_batch)
        return new_batch, new_batch_num

    for i in range(len(batches)):
        if batches[i] is None:
            batches[i], next_batch_num = _generate_new_batch(next_batch_num)

            if batch_all_for_hyper:
                for j in range(i + 1, len(batches)):
                    batches[j] = batches[i]
                break
            for j in range(i + 1, len(batches)):
                if batches[j] is None and orig_files[i] == orig_files[j]:
                    batches[j] = batches[i]

        if first_for_hyper:
            for j in range(i + 1, len(batches)):
                if batches[j] != batches[i]:
                    batches[j] = None
            break
    return batches


def _normalize_input_specs(input_specs):
    if input_specs is None:
        return ([], [])
    if type(input_specs) == str:
        return ([input_specs], [input_specs])
    if type(input_specs) == list:
        return (input_specs, copy.copy(input_specs))
    return ([], [])


def _append_initial_p_indices(initial_ps, input_specs, xin_to_p_noninf_ind):
    if initial_ps is None:
        return
    for input_spec in input_specs:
        assert(input_spec in xin_to_p_noninf_ind)
        initial_ps.append(xin_to_p_noninf_ind[input_spec])


def _map_initial_p_indices_to_values(initial_ps, initial_p):
    if initial_ps is None:
        return
    assert(type(initial_p) is list)
    for i in range(len(initial_ps)):
        assert(initial_ps[i]) >= 0 and initial_ps[i] < len(initial_p)
        initial_ps[i] = initial_p[initial_ps[i]]


def _expand_x_inputs(x_inputs, orig_files, batch_separator="@", file_separator=None):
    expanded_inputs = []
    batches = []
    labels = []
    expanded_orig_files = []
    for i in range(len(x_inputs)):
        x_input = x_inputs[i]
        orig_file = orig_files[i]
        batch = None
        label = os.path.basename(orig_file)
        if "." in label:
            label = ".".join(label.split(".")[:-1])
        if batch_separator in x_input:
            batch = x_input.split(batch_separator)[-1]
            label = batch
            x_input = batch_separator.join(x_input.split(batch_separator)[:-1])

        (x_input, tag) = remove_tag_from_input(x_input)
        if tag is not None:
            label = tag

        if file_separator is not None:
            x_to_add = x_input.split(file_separator)
        else:
            x_to_add = [x_input]

        expanded_inputs += x_to_add
        batches += [batch] * len(x_to_add)
        labels += [label] * len(x_to_add)
        expanded_orig_files += [orig_file] * len(x_to_add)
    return (expanded_inputs, batches, labels, expanded_orig_files)


def _append_inputs_from_list_files(
    list_specs,
    dest_inputs,
    dest_orig_files,
    list_open_fn,
    strip_fn,
    resolve_relative_paths=False,
    skip_empty_lines=True,
    initial_ps=None,
    xin_to_p_noninf_ind=None,
    batch_separator="@",
):
    if list_specs is None:
        return

    if type(list_specs) == str:
        list_specs = [list_specs]

    for list_spec in list_specs:
        batch = None
        if batch_separator in list_spec:
            batch = list_spec.split(batch_separator)[-1]
            list_spec = batch_separator.join(list_spec.split(batch_separator)[:-1])

        list_dir = os.path.dirname(os.path.abspath(list_spec))
        with list_open_fn(list_spec) as list_fh:
            for raw_line in list_fh:
                line = strip_fn(raw_line)
                if skip_empty_lines and len(line) == 0:
                    continue

                if resolve_relative_paths:
                    (path, label) = remove_tag_from_input(line)
                    if path and not os.path.isabs(path):
                        path = os.path.normpath(os.path.join(list_dir, path))
                    line = add_tag_to_input(path, label)

                if batch is not None and batch_separator not in line:
                    line = "%s%s%s" % (line, batch_separator, batch)

                dest_inputs.append(line)
                if initial_ps is not None:
                    assert(list_spec in xin_to_p_noninf_ind)
                    initial_ps.append(xin_to_p_noninf_ind[list_spec])
                dest_orig_files.append(list_spec)


def prepare_read_x_inputs(
    X_in,
    X_list,
    Xd_in,
    Xd_list,
    initial_p,
    xin_to_p_noninf_ind,
    batch_separator,
    file_separator,
    *,
    sparse_list_open_fn,
    dense_list_open_fn,
):
    initial_ps = None
    if initial_p is not None:
        if type(initial_p) is not list:
            initial_p = [initial_p]
        initial_ps = []
        assert(xin_to_p_noninf_ind is not None)

    (X_ins, orig_files) = _normalize_input_specs(X_in)
    _append_initial_p_indices(initial_ps, X_ins, xin_to_p_noninf_ind)
    _append_inputs_from_list_files(
        list_specs=X_list,
        dest_inputs=X_ins,
        dest_orig_files=orig_files,
        list_open_fn=sparse_list_open_fn,
        strip_fn=lambda line: line.strip(),
        resolve_relative_paths=True,
        skip_empty_lines=True,
        initial_ps=initial_ps,
        xin_to_p_noninf_ind=xin_to_p_noninf_ind,
        batch_separator=batch_separator,
    )
    X_ins, batches, labels, orig_files = _expand_x_inputs(
        X_ins,
        orig_files,
        batch_separator=batch_separator,
        file_separator=file_separator,
    )
    is_dense = [False for _ in X_ins]

    (Xd_ins, orig_dfiles) = _normalize_input_specs(Xd_in)
    _append_initial_p_indices(initial_ps, Xd_ins, xin_to_p_noninf_ind)
    _append_inputs_from_list_files(
        list_specs=Xd_list,
        dest_inputs=Xd_ins,
        dest_orig_files=orig_dfiles,
        list_open_fn=dense_list_open_fn,
        strip_fn=lambda line: line.strip("\n"),
        resolve_relative_paths=False,
        skip_empty_lines=False,
        initial_ps=initial_ps,
        xin_to_p_noninf_ind=xin_to_p_noninf_ind,
        batch_separator=batch_separator,
    )
    Xd_ins, batches2, labels2, orig_dfiles = _expand_x_inputs(
        Xd_ins,
        orig_dfiles,
        batch_separator=batch_separator,
        file_separator=file_separator,
    )

    _map_initial_p_indices_to_values(initial_ps, initial_p)

    X_ins += Xd_ins
    batches += batches2
    labels += labels2
    orig_files += orig_dfiles
    is_dense += [True for _ in Xd_ins]
    return XInputPlan(
        initial_ps=initial_ps,
        X_ins=X_ins,
        batches=batches,
        labels=labels,
        orig_files=orig_files,
        is_dense=is_dense,
    )


def xdata_from_input_plan(input_plan):
    return XData(
        gene_set_batches=np.array(input_plan.batches),
        gene_set_labels=np.array(input_plan.labels),
        is_dense_gene_set=np.array(input_plan.is_dense, dtype=bool),
    )


def infer_columns_from_table_file(filename, open_text_fn, *, log_fn=None, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _msg: None

    log_fn("Trying to determine columns from headers and data for %s..." % filename)
    header = None
    with open_text_fn(filename) as fh:
        header = fh.readline().strip("\n")
        orig_header_cols = header.split()

        first_line = fh.readline().strip("\n")
        first_cols = first_line.split()

        if len(orig_header_cols) > len(first_cols):
            orig_header_cols = header.split("\t")

        header_cols = [x.strip('"').strip("'").strip("\n") for x in orig_header_cols]

        def __get_possible_from_headers(_header_cols, possible_headers1, possible_headers2=None):
            possible = np.full(len(_header_cols), False)
            possible_inds = [i for i in range(len(_header_cols)) if _header_cols[i].lower().strip('_"') in possible_headers1]
            if len(possible_inds) == 0 and possible_headers2 is not None:
                possible_inds = [i for i in range(len(_header_cols)) if _header_cols[i].lower() in possible_headers2]
            possible[possible_inds] = True
            return possible

        possible_gene_id_headers = set(["gene", "id"])
        possible_var_id_headers = set(["var", "id", "rs", "varid"])
        possible_chrom_headers = set(["chr", "chrom", "chromosome", "#chrom"])
        possible_pos_headers = set(["pos", "bp", "position", "base_pair_location"])
        possible_locus_headers = set(["variant"])
        possible_p_headers = set(["p-val", "p_val", "pval", "p.value", "p-value", "p_value"])
        possible_p_headers2 = set(["p"])
        possible_beta_headers = set(["beta", "effect"])
        possible_se_headers = set(["se", "std", "stderr", "standard_error"])
        possible_freq_headers = set(["maf", "freq"])
        possible_freq_headers2 = set(["af", "effect_allele_frequency"])
        possible_n_headers = set(["sample", "neff", "TotalSampleSize", "n_samples"])
        possible_n_headers2 = set(["n"])

        possible_gene_id_cols = __get_possible_from_headers(header_cols, possible_gene_id_headers)
        possible_var_id_cols = __get_possible_from_headers(header_cols, possible_var_id_headers)
        possible_chrom_cols = __get_possible_from_headers(header_cols, possible_chrom_headers)
        possible_locus_cols = __get_possible_from_headers(header_cols, possible_locus_headers)
        possible_pos_cols = __get_possible_from_headers(header_cols, possible_pos_headers)
        possible_p_cols = __get_possible_from_headers(header_cols, possible_p_headers, possible_p_headers2)
        possible_beta_cols = __get_possible_from_headers(header_cols, possible_beta_headers)
        possible_se_cols = __get_possible_from_headers(header_cols, possible_se_headers)
        possible_freq_cols = __get_possible_from_headers(header_cols, possible_freq_headers, possible_freq_headers2)
        possible_n_cols = __get_possible_from_headers(header_cols, possible_n_headers, possible_n_headers2)

        missing_vals = set(["", ".", "-", "na"])
        num_read = 0
        max_to_read = 1000

        for line in fh:
            cols = line.strip("\n").split()
            seen_non_missing = False
            if len(cols) != len(header_cols):
                cols = line.strip("\n").split("\t")

            if len(cols) != len(header_cols):
                bail_fn("Error: couldn't parse line into same number of columns as header (%d vs. %d)" % (len(cols), len(header_cols)))

            for i in range(len(cols)):
                token = cols[i].lower()

                if token.lower() in missing_vals:
                    continue

                seen_non_missing = True

                if possible_gene_id_cols[i]:
                    try:
                        val = float(cols[i])
                        if not int(val) == val:
                            possible_gene_id_cols[i] = False
                    except ValueError:
                        pass
                if possible_var_id_cols[i]:
                    if len(token) < 4:
                        possible_var_id_cols[i] = False

                    if "chr" in token or ":" in token or "rs" in token or "_" in token or "-" in token or "var" in token:
                        pass
                    else:
                        possible_var_id_cols[i] = False
                if possible_chrom_cols[i]:
                    if "chr" in token or "x" in token or "y" in token or "m" in token:
                        pass
                    else:
                        try:
                            val = int(cols[i])
                            if val < 1 or val > 26:
                                possible_chrom_cols[i] = False
                        except ValueError:
                            possible_chrom_cols[i] = False
                if possible_locus_cols[i]:
                    if "chr" in token or "x" in token or "y" in token or "m" in token:
                        pass
                    else:
                        try:
                            locus = None
                            for delim in [":", "_"]:
                                if delim in cols[i]:
                                    locus = cols[i].split(delim)
                            if locus is not None and len(locus) >= 2:
                                chrom = int(locus[0])
                                _pos = int(locus[1])
                                if chrom < 1 or chrom > 26:
                                    possible_locus_cols[i] = False
                        except ValueError:
                            possible_locus_cols[i] = False
                if possible_pos_cols[i]:
                    try:
                        if len(token) < 3:
                            possible_pos_cols[i] = False
                        val = float(cols[i])
                        if not int(val) == val:
                            possible_pos_cols[i] = False
                    except ValueError:
                        possible_pos_cols[i] = False

                if possible_p_cols[i]:
                    try:
                        val = float(cols[i])
                        if val > 1 or val < 0:
                            possible_p_cols[i] = False
                    except ValueError:
                        possible_p_cols[i] = False
                if possible_beta_cols[i]:
                    try:
                        _val = float(cols[i])
                    except ValueError:
                        possible_beta_cols[i] = False
                if possible_se_cols[i]:
                    try:
                        val = float(cols[i])
                        if val < 0:
                            possible_se_cols[i] = False
                    except ValueError:
                        possible_se_cols[i] = False
                if possible_freq_cols[i]:
                    try:
                        val = float(cols[i])
                        if val > 1 or val < 0:
                            possible_freq_cols[i] = False
                    except ValueError:
                        possible_freq_cols[i] = False
                if possible_n_cols[i]:
                    if len(token) < 3:
                        possible_n_cols[i] = False
                    else:
                        try:
                            val = float(cols[i])
                            if val < 0:
                                possible_n_cols[i] = False
                        except ValueError:
                            possible_n_cols[i] = False
            if seen_non_missing:
                num_read += 1
                if num_read >= max_to_read:
                    break

    possible_beta_cols[possible_p_cols] = False
    possible_beta_cols[possible_se_cols] = False
    possible_beta_cols[possible_pos_cols] = False

    total_possible = (
        possible_gene_id_cols.astype(int)
        + possible_var_id_cols.astype(int)
        + possible_chrom_cols.astype(int)
        + possible_pos_cols.astype(int)
        + possible_p_cols.astype(int)
        + possible_beta_cols.astype(int)
        + possible_se_cols.astype(int)
        + possible_freq_cols.astype(int)
        + possible_n_cols.astype(int)
    )
    for possible_cols in [
        possible_gene_id_cols,
        possible_var_id_cols,
        possible_chrom_cols,
        possible_pos_cols,
        possible_p_cols,
        possible_beta_cols,
        possible_se_cols,
        possible_freq_cols,
        possible_n_cols,
    ]:
        possible_cols[total_possible > 1] = False

    orig_header_cols = np.array(orig_header_cols)
    return (
        orig_header_cols[possible_gene_id_cols],
        orig_header_cols[possible_var_id_cols],
        orig_header_cols[possible_chrom_cols],
        orig_header_cols[possible_pos_cols],
        orig_header_cols[possible_locus_cols],
        orig_header_cols[possible_p_cols],
        orig_header_cols[possible_beta_cols],
        orig_header_cols[possible_se_cols],
        orig_header_cols[possible_freq_cols],
        orig_header_cols[possible_n_cols],
        header,
    )


def needs_gwas_column_detection(
    gwas_pos_col,
    gwas_chrom_col,
    gwas_locus_col,
    gwas_p_col,
    gwas_beta_col,
    gwas_se_col,
    gwas_n_col,
    gwas_n,
):
    if (gwas_pos_col is None or gwas_chrom_col is None) and gwas_locus_col is None:
        return True

    has_se = gwas_se_col is not None or gwas_n_col is not None or gwas_n is not None
    if (gwas_p_col is not None and gwas_beta_col is not None) or (gwas_p_col is not None and has_se) or (gwas_beta_col is not None and has_se):
        return False
    return True


def autodetect_gwas_columns(
    gwas_in,
    gwas_pos_col,
    gwas_chrom_col,
    gwas_locus_col,
    gwas_p_col,
    gwas_beta_col,
    gwas_se_col,
    gwas_freq_col,
    gwas_n_col,
    gwas_n,
    *,
    infer_columns_fn,
    log_fn=None,
    bail_fn=None,
    debug_just_check_header=False,
):
    if bail_fn is None:
        bail_fn = _default_bail
    if log_fn is None:
        log_fn = lambda _msg: None

    (
        _possible_gene_id_cols,
        _possible_var_id_cols,
        possible_chrom_cols,
        possible_pos_cols,
        possible_locus_cols,
        possible_p_cols,
        possible_beta_cols,
        possible_se_cols,
        possible_freq_cols,
        possible_n_cols,
        header,
    ) = infer_columns_fn(gwas_in)

    if gwas_pos_col is None:
        if len(possible_pos_cols) == 1:
            gwas_pos_col = possible_pos_cols[0]
            log_fn("Using %s for position column; change with --gwas-pos-col if incorrect" % gwas_pos_col)
        else:
            log_fn("Could not determine position column from header %s; specify with --gwas-pos-col" % header)
    if gwas_chrom_col is None:
        if len(possible_chrom_cols) == 1:
            gwas_chrom_col = possible_chrom_cols[0]
            log_fn("Using %s for chrom column; change with --gwas-chrom-col if incorrect" % gwas_chrom_col)
        else:
            log_fn("Could not determine chrom column from header %s; specify with --gwas-chrom-col" % header)
    if (gwas_pos_col is None or gwas_chrom_col is None) and gwas_locus_col is None:
        if len(possible_locus_cols) == 1:
            gwas_locus_col = possible_locus_cols[0]
            log_fn("Using %s for locus column; change with --gwas-locus-col if incorrect" % gwas_locus_col)
        else:
            bail_fn("Could not determine chrom and pos columns from header %s; specify with --gwas-chrom-col and --gwas-pos-col or with --gwas-locus-col" % header)

    if gwas_p_col is None:
        if len(possible_p_cols) == 1:
            gwas_p_col = possible_p_cols[0]
            log_fn("Using %s for p column; change with --gwas-p-col if incorrect" % gwas_p_col)
        else:
            log_fn("Could not determine p column from header %s; if desired specify with --gwas-p-col" % header)
    if gwas_se_col is None:
        if len(possible_se_cols) == 1:
            gwas_se_col = possible_se_cols[0]
            log_fn("Using %s for se column; change with --gwas-se-col if incorrect" % gwas_se_col)
        else:
            log_fn("Could not determine se column from header %s; if desired specify with --gwas-se-col" % header)
    if gwas_beta_col is None:
        if len(possible_beta_cols) == 1:
            gwas_beta_col = possible_beta_cols[0]
            log_fn("Using %s for beta column; change with --gwas-beta-col if incorrect" % gwas_beta_col)
        else:
            log_fn("Could not determine beta column from header %s; if desired specify with --gwas-beta-col" % header)

    if gwas_n_col is None:
        if len(possible_n_cols) == 1:
            gwas_n_col = possible_n_cols[0]
            log_fn("Using %s for N column; change with --gwas-n-col if incorrect" % gwas_n_col)
        else:
            log_fn("Could not determine N column from header %s; if desired specify with --gwas-n-col" % header)

    if gwas_freq_col is None:
        if len(possible_freq_cols) == 1:
            gwas_freq_col = possible_freq_cols[0]
            log_fn("Using %s for freq column; change with --gwas-freq-col if incorrect" % gwas_freq_col)

    has_se = gwas_se_col is not None
    has_n = gwas_n_col is not None or gwas_n is not None
    if (gwas_p_col is not None and gwas_beta_col is not None) or (gwas_p_col is not None and (has_se or has_n)) or (gwas_beta_col is not None and has_se):
        pass
    else:
        bail_fn("Require information about p-value and se or N or beta, or beta and se; specify with --gwas-p-col, --gwas-beta-col, --gwas-se-col, and --gwas-n-col")

    if debug_just_check_header:
        bail_fn("Done checking headers")

    return (
        gwas_pos_col,
        gwas_chrom_col,
        gwas_locus_col,
        gwas_p_col,
        gwas_beta_col,
        gwas_se_col,
        gwas_freq_col,
        gwas_n_col,
    )


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


def is_huge_statistics_bundle_path(huge_statistics_file):
    lower = huge_statistics_file.lower()
    return lower.endswith(".tar.gz") or lower.endswith(".tgz") or lower.endswith(".tar")


def coerce_runtime_state_dict(runtime_state, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if isinstance(runtime_state, dict):
        return runtime_state
    if hasattr(runtime_state, "__dict__"):
        return runtime_state.__dict__
    bail_fn("Internal error: unsupported runtime state container for HuGE cache IO")


def get_huge_statistics_paths_for_prefix(prefix):
    return {
        "meta": "%s.huge.meta.json.gz" % prefix,
        "cache_genes": "%s.huge.cache_genes.tsv.gz" % prefix,
        "extra_scores": "%s.huge.extra_scores.tsv.gz" % prefix,
        "matrix_row_genes": "%s.huge.matrix_row_genes.tsv.gz" % prefix,
        "gene_scores": "%s.huge.gene_scores.tsv.gz" % prefix,
        "gene_covariates": "%s.huge.gene_covariates.tsv.gz" % prefix,
        "bfs_data": "%s.huge_signal_bfs.data.tsv.gz" % prefix,
        "bfs_indices": "%s.huge_signal_bfs.indices.tsv.gz" % prefix,
        "bfs_indptr": "%s.huge_signal_bfs.indptr.tsv.gz" % prefix,
        "bfs_reg_data": "%s.huge_signal_bfs_for_regression.data.tsv.gz" % prefix,
        "bfs_reg_indices": "%s.huge_signal_bfs_for_regression.indices.tsv.gz" % prefix,
        "bfs_reg_indptr": "%s.huge_signal_bfs_for_regression.indptr.tsv.gz" % prefix,
        "signal_posteriors": "%s.huge_signal_posteriors.tsv.gz" % prefix,
        "signal_posteriors_for_regression": "%s.huge_signal_posteriors_for_regression.tsv.gz" % prefix,
        "signal_sum_gene_cond_probabilities": "%s.huge_signal_sum_gene_cond_probabilities.tsv.gz" % prefix,
        "signal_sum_gene_cond_probabilities_for_regression": "%s.huge_signal_sum_gene_cond_probabilities_for_regression.tsv.gz" % prefix,
        "signal_mean_gene_pos": "%s.huge_signal_mean_gene_pos.tsv.gz" % prefix,
        "signal_mean_gene_pos_for_regression": "%s.huge_signal_mean_gene_pos_for_regression.tsv.gz" % prefix,
    }


def write_numeric_vector_file(out_file, values, *, open_text_fn, value_type=float):
    with open_text_fn(out_file, "w") as out_fh:
        if values is None:
            return
        values = np.ravel(np.array(values))
        for value in values:
            if value_type == int:
                out_fh.write("%d\n" % int(value))
            else:
                out_fh.write("%.18g\n" % float(value))


def read_numeric_vector_file(in_file, *, open_text_fn, value_type=float):
    values = []
    with open_text_fn(in_file) as in_fh:
        for line in in_fh:
            line = line.strip()
            if line == "":
                continue
            if value_type == int:
                values.append(int(line))
            else:
                values.append(float(line))
    if value_type == int:
        return np.array(values, dtype=int)
    return np.array(values, dtype=float)


def build_huge_statistics_matrix_row_genes(cache_genes, extra_genes, num_matrix_rows, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if len(cache_genes) > 0:
        if num_matrix_rows < len(cache_genes):
            bail_fn("Error writing HuGE statistics cache: matrix rows %d < number of genes %d" % (num_matrix_rows, len(cache_genes)))
        num_extra_matrix_rows = num_matrix_rows - len(cache_genes)
        if num_extra_matrix_rows > len(extra_genes):
            bail_fn("Error writing HuGE statistics cache: matrix rows require %d extra genes but only %d were provided" % (num_extra_matrix_rows, len(extra_genes)))
        return cache_genes + extra_genes[:num_extra_matrix_rows]
    if num_matrix_rows > len(extra_genes):
        bail_fn("Error writing HuGE statistics cache: matrix rows %d > number of extra genes %d" % (num_matrix_rows, len(extra_genes)))
    return extra_genes[:num_matrix_rows]


def build_huge_statistics_score_maps(runtime_state, cache_genes, extra_genes, gene_bf, extra_gene_bf, gene_bf_for_regression, extra_gene_bf_for_regression):
    gene_to_score = {}
    if runtime_state.get("gene_to_gwas_huge_score") is not None:
        gene_to_score = dict(runtime_state["gene_to_gwas_huge_score"])

    gene_to_score_uncorrected = {}
    if runtime_state.get("gene_to_gwas_huge_score_uncorrected") is not None:
        gene_to_score_uncorrected = dict(runtime_state["gene_to_gwas_huge_score_uncorrected"])

    gene_to_score_for_regression = {}

    for i, gene in enumerate(cache_genes):
        if gene_bf is not None and i < len(gene_bf):
            gene_to_score[gene] = float(gene_bf[i])
        if gene_bf_for_regression is not None and i < len(gene_bf_for_regression):
            gene_to_score_for_regression[gene] = float(gene_bf_for_regression[i])
        elif gene in gene_to_score:
            gene_to_score_for_regression[gene] = float(gene_to_score[gene])

    for i, gene in enumerate(extra_genes):
        if i < len(extra_gene_bf):
            gene_to_score[gene] = float(extra_gene_bf[i])
        if i < len(extra_gene_bf_for_regression):
            gene_to_score_for_regression[gene] = float(extra_gene_bf_for_regression[i])
        elif gene in gene_to_score:
            gene_to_score_for_regression[gene] = float(gene_to_score[gene])
        if gene not in gene_to_score_uncorrected:
            gene_to_score_uncorrected[gene] = float(extra_gene_bf[i])

    return (gene_to_score, gene_to_score_uncorrected, gene_to_score_for_regression)


def build_huge_statistics_meta(runtime_state, huge_signal_bfs, huge_signal_bfs_for_regression, *, json_safe_fn=None):
    if json_safe_fn is None:
        json_safe_fn = json_safe
    return {
        "version": 1,
        "huge_signal_bfs_shape": [int(huge_signal_bfs.shape[0]), int(huge_signal_bfs.shape[1])],
        "huge_signal_bfs_for_regression_shape": [int(huge_signal_bfs_for_regression.shape[0]), int(huge_signal_bfs_for_regression.shape[1])],
        "huge_signal_max_closest_gene_prob": (None if runtime_state.get("huge_signal_max_closest_gene_prob") is None else float(runtime_state.get("huge_signal_max_closest_gene_prob"))),
        "huge_cap_region_posterior": bool(runtime_state.get("huge_cap_region_posterior", True)),
        "huge_scale_region_posterior": bool(runtime_state.get("huge_scale_region_posterior", False)),
        "huge_phantom_region_posterior": bool(runtime_state.get("huge_phantom_region_posterior", False)),
        "huge_allow_evidence_of_absence": bool(runtime_state.get("huge_allow_evidence_of_absence", False)),
        "huge_sparse_mode": bool(runtime_state.get("huge_sparse_mode", False)),
        "huge_signals": [] if runtime_state.get("huge_signals") is None else [[str(x[0]), int(x[1]), float(x[2]), x[3]] for x in runtime_state.get("huge_signals")],
        "gene_covariate_names": (None if runtime_state.get("gene_covariate_names") is None else list(runtime_state.get("gene_covariate_names"))),
        "gene_covariate_directions": (None if runtime_state.get("gene_covariate_directions") is None else list(np.array(runtime_state.get("gene_covariate_directions"), dtype=float))),
        "gene_covariate_intercept_index": runtime_state.get("gene_covariate_intercept_index"),
        "gene_covariate_slope_defaults": (None if runtime_state.get("gene_covariate_slope_defaults") is None else list(np.array(runtime_state.get("gene_covariate_slope_defaults"), dtype=float))),
        "total_qc_metric_betas_defaults": (None if runtime_state.get("total_qc_metric_betas_defaults") is None else list(np.array(runtime_state.get("total_qc_metric_betas_defaults"), dtype=float))),
        "total_qc_metric_intercept_defaults": (None if runtime_state.get("total_qc_metric_intercept_defaults") is None else float(runtime_state.get("total_qc_metric_intercept_defaults"))),
        "total_qc_metric2_betas_defaults": (None if runtime_state.get("total_qc_metric2_betas_defaults") is None else list(np.array(runtime_state.get("total_qc_metric2_betas_defaults"), dtype=float))),
        "total_qc_metric2_intercept_defaults": (None if runtime_state.get("total_qc_metric2_intercept_defaults") is None else float(runtime_state.get("total_qc_metric2_intercept_defaults"))),
        "recorded_params": json_safe_fn(runtime_state.get("params")),
        "recorded_param_keys": json_safe_fn(runtime_state.get("param_keys")),
    }


def write_huge_statistics_text_tables(
    paths,
    runtime_state,
    cache_genes,
    extra_genes,
    extra_gene_bf,
    extra_gene_bf_for_regression,
    matrix_row_genes,
    gene_to_score,
    gene_to_score_uncorrected,
    gene_to_score_for_regression,
    *,
    open_text_fn,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    with open_text_fn(paths["cache_genes"], "w") as out_fh:
        for gene in cache_genes:
            out_fh.write("%s\n" % gene)

    with open_text_fn(paths["extra_scores"], "w") as out_fh:
        out_fh.write("Gene\tlog_bf\tlog_bf_for_regression\n")
        for i in range(len(extra_genes)):
            bf = np.nan
            if i < len(extra_gene_bf):
                bf = extra_gene_bf[i]
            bf_for_regression = np.nan
            if i < len(extra_gene_bf_for_regression):
                bf_for_regression = extra_gene_bf_for_regression[i]
            out_fh.write("%s\t%.18g\t%.18g\n" % (extra_genes[i], bf, bf_for_regression))

    with open_text_fn(paths["matrix_row_genes"], "w") as out_fh:
        for gene in matrix_row_genes:
            out_fh.write("%s\n" % gene)

    ordered_genes = []
    seen = set()
    for gene in cache_genes + extra_genes + list(gene_to_score.keys()) + list(gene_to_score_uncorrected.keys()):
        if gene not in seen:
            seen.add(gene)
            ordered_genes.append(gene)

    with open_text_fn(paths["gene_scores"], "w") as out_fh:
        out_fh.write("Gene\tlog_bf\tlog_bf_uncorrected\tlog_bf_for_regression\n")
        for gene in ordered_genes:
            score = gene_to_score.get(gene, np.nan)
            score_uncorrected = gene_to_score_uncorrected.get(gene, np.nan)
            score_for_regression = gene_to_score_for_regression.get(gene, np.nan)
            out_fh.write("%s\t%.18g\t%.18g\t%.18g\n" % (gene, score, score_uncorrected, score_for_regression))

    gene_covariates = runtime_state.get("gene_covariates")
    if gene_covariates is not None:
        if gene_covariates.shape[0] != len(matrix_row_genes):
            bail_fn("Error writing HuGE statistics cache: gene covariates have %d rows but matrix has %d rows" % (gene_covariates.shape[0], len(matrix_row_genes)))
        with open_text_fn(paths["gene_covariates"], "w") as out_fh:
            out_fh.write("Gene\t%s\n" % ("\t".join(runtime_state.get("gene_covariate_names"))))
            for i in range(len(matrix_row_genes)):
                out_fh.write("%s\t%s\n" % (matrix_row_genes[i], "\t".join(["%.18g" % x for x in gene_covariates[i, :]])))


def read_huge_statistics_text_tables(paths, *, open_text_fn):
    cache_genes = []
    with open_text_fn(paths["cache_genes"]) as in_fh:
        for line in in_fh:
            gene = line.strip()
            if gene != "":
                cache_genes.append(gene)

    extra_genes = []
    extra_gene_bf = []
    extra_gene_bf_for_regression = []
    with open_text_fn(paths["extra_scores"]) as in_fh:
        _header = in_fh.readline()
        for line in in_fh:
            cols = line.strip("\n").split("\t")
            if len(cols) < 3:
                continue
            extra_genes.append(cols[0])
            extra_gene_bf.append(float(cols[1]))
            extra_gene_bf_for_regression.append(float(cols[2]))

    matrix_row_genes = []
    with open_text_fn(paths["matrix_row_genes"]) as in_fh:
        for line in in_fh:
            gene = line.strip()
            if gene != "":
                matrix_row_genes.append(gene)

    gene_to_score = {}
    gene_to_score_uncorrected = {}
    gene_to_score_for_regression = {}
    with open_text_fn(paths["gene_scores"]) as in_fh:
        _header = in_fh.readline()
        for line in in_fh:
            cols = line.strip("\n").split("\t")
            if len(cols) < 4:
                continue
            gene = cols[0]
            gene_to_score[gene] = float(cols[1])
            gene_to_score_uncorrected[gene] = float(cols[2])
            gene_to_score_for_regression[gene] = float(cols[3])

    return (
        cache_genes,
        extra_genes,
        extra_gene_bf,
        extra_gene_bf_for_regression,
        matrix_row_genes,
        gene_to_score,
        gene_to_score_uncorrected,
        gene_to_score_for_regression,
    )


def resolve_huge_statistics_gene_vectors(
    runtime_state,
    cache_genes,
    extra_genes,
    matrix_row_genes,
    gene_to_score,
    gene_to_score_for_regression,
    *,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail
    genes = runtime_state.get("genes")
    if genes is None:
        if len(cache_genes) > 0:
            bail_fn("HuGE cache was generated with preloaded genes but this run has no preloaded genes")
        if len(matrix_row_genes) > 0 and matrix_row_genes != extra_genes[:len(matrix_row_genes)]:
            bail_fn("HuGE cache is inconsistent: matrix rows do not match extra gene ordering")
        return (np.array([]), np.array([]))

    if cache_genes != genes:
        bail_fn("HuGE cache gene ordering does not match current run. Rebuild cache for this run setup.")
    if len(matrix_row_genes) < len(genes) or matrix_row_genes[:len(genes)] != genes:
        bail_fn("HuGE cache matrix row ordering does not match current run genes")

    gene_bf = np.array([gene_to_score.get(gene, np.nan) for gene in genes])
    gene_bf_for_regression = np.array([gene_to_score_for_regression.get(gene, np.nan) for gene in genes])
    return (gene_bf, gene_bf_for_regression)


def read_huge_statistics_covariates_if_present(runtime_state, paths, *, open_text_fn, exists_fn=None):
    if exists_fn is None:
        exists_fn = os.path.exists
    if not exists_fn(paths["gene_covariates"]):
        return
    covariate_rows = []
    with open_text_fn(paths["gene_covariates"]) as in_fh:
        header = in_fh.readline().strip("\n").split("\t")
        if len(header) > 1 and runtime_state.get("gene_covariate_names") is None:
            runtime_state["gene_covariate_names"] = header[1:]
        for line in in_fh:
            cols = line.strip("\n").split("\t")
            if len(cols) <= 1:
                continue
            covariate_rows.append([float(x) for x in cols[1:]])
    runtime_state["gene_covariates"] = np.array(covariate_rows)


def load_huge_statistics_sparse_and_vectors(runtime_state, paths, meta, *, read_vector_fn):
    huge_signal_bfs_shape = tuple(meta["huge_signal_bfs_shape"])
    huge_signal_bfs_for_regression_shape = tuple(meta["huge_signal_bfs_for_regression_shape"])

    sparse_components = {
        "bfs_data": read_vector_fn(paths["bfs_data"], value_type=float),
        "bfs_indices": read_vector_fn(paths["bfs_indices"], value_type=int),
        "bfs_indptr": read_vector_fn(paths["bfs_indptr"], value_type=int),
        "bfs_reg_data": read_vector_fn(paths["bfs_reg_data"], value_type=float),
        "bfs_reg_indices": read_vector_fn(paths["bfs_reg_indices"], value_type=int),
        "bfs_reg_indptr": read_vector_fn(paths["bfs_reg_indptr"], value_type=int),
    }

    runtime_state["huge_signal_bfs"] = sparse.csc_matrix(
        (sparse_components["bfs_data"], sparse_components["bfs_indices"], sparse_components["bfs_indptr"]),
        shape=huge_signal_bfs_shape,
    )
    runtime_state["huge_signal_bfs_for_regression"] = sparse.csc_matrix(
        (sparse_components["bfs_reg_data"], sparse_components["bfs_reg_indices"], sparse_components["bfs_reg_indptr"]),
        shape=huge_signal_bfs_for_regression_shape,
    )

    runtime_vector_map = (
        ("huge_signal_posteriors", "signal_posteriors"),
        ("huge_signal_posteriors_for_regression", "signal_posteriors_for_regression"),
        ("huge_signal_sum_gene_cond_probabilities", "signal_sum_gene_cond_probabilities"),
        ("huge_signal_sum_gene_cond_probabilities_for_regression", "signal_sum_gene_cond_probabilities_for_regression"),
        ("huge_signal_mean_gene_pos", "signal_mean_gene_pos"),
        ("huge_signal_mean_gene_pos_for_regression", "signal_mean_gene_pos_for_regression"),
    )
    for state_key, path_key in runtime_vector_map:
        runtime_state[state_key] = read_vector_fn(paths[path_key], value_type=float)


def apply_huge_statistics_meta_to_runtime(runtime_state, meta):
    runtime_state["huge_signal_max_closest_gene_prob"] = meta["huge_signal_max_closest_gene_prob"]
    runtime_state["huge_cap_region_posterior"] = bool(meta["huge_cap_region_posterior"])
    runtime_state["huge_scale_region_posterior"] = bool(meta["huge_scale_region_posterior"])
    runtime_state["huge_phantom_region_posterior"] = bool(meta["huge_phantom_region_posterior"])
    runtime_state["huge_allow_evidence_of_absence"] = bool(meta["huge_allow_evidence_of_absence"])
    runtime_state["huge_sparse_mode"] = bool(meta["huge_sparse_mode"])
    runtime_state["huge_signals"] = [tuple(x) for x in meta["huge_signals"]]

    runtime_state["gene_covariates"] = None
    runtime_state["gene_covariates_mask"] = None
    runtime_state["gene_covariate_names"] = meta.get("gene_covariate_names")
    runtime_state["gene_covariate_directions"] = None
    if meta.get("gene_covariate_directions") is not None:
        runtime_state["gene_covariate_directions"] = np.array(meta["gene_covariate_directions"], dtype=float)
    runtime_state["gene_covariate_intercept_index"] = meta.get("gene_covariate_intercept_index")
    runtime_state["gene_covariates_mat_inv"] = None
    runtime_state["gene_covariate_zs"] = None
    runtime_state["gene_covariate_adjustments"] = None

    runtime_state["gene_covariate_slope_defaults"] = None if meta.get("gene_covariate_slope_defaults") is None else np.array(meta.get("gene_covariate_slope_defaults"), dtype=float)
    runtime_state["total_qc_metric_betas_defaults"] = None if meta.get("total_qc_metric_betas_defaults") is None else np.array(meta.get("total_qc_metric_betas_defaults"), dtype=float)
    runtime_state["total_qc_metric_intercept_defaults"] = meta.get("total_qc_metric_intercept_defaults")
    runtime_state["total_qc_metric2_betas_defaults"] = None if meta.get("total_qc_metric2_betas_defaults") is None else np.array(meta.get("total_qc_metric2_betas_defaults"), dtype=float)
    runtime_state["total_qc_metric2_intercept_defaults"] = meta.get("total_qc_metric2_intercept_defaults")


def combine_runtime_huge_scores(runtime_state):
    if runtime_state.get("gene_to_gwas_huge_score") is not None and runtime_state.get("gene_to_exomes_huge_score") is not None:
        runtime_state["gene_to_huge_score"] = {}
        genes = list(set().union(runtime_state["gene_to_gwas_huge_score"], runtime_state["gene_to_exomes_huge_score"]))
        for gene in genes:
            runtime_state["gene_to_huge_score"][gene] = 0
            if gene in runtime_state["gene_to_gwas_huge_score"]:
                runtime_state["gene_to_huge_score"][gene] += runtime_state["gene_to_gwas_huge_score"][gene]
            if gene in runtime_state["gene_to_exomes_huge_score"]:
                runtime_state["gene_to_huge_score"][gene] += runtime_state["gene_to_exomes_huge_score"][gene]


def validate_huge_statistics_loaded_shapes(runtime_state, matrix_row_genes, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    if runtime_state["huge_signal_bfs"].shape[1] != len(runtime_state["huge_signals"]):
        bail_fn("HuGE cache is inconsistent: huge_signal_bfs has %d columns but found %d signals" % (runtime_state["huge_signal_bfs"].shape[1], len(runtime_state["huge_signals"])))
    if runtime_state["huge_signal_bfs"].shape[0] != len(matrix_row_genes):
        bail_fn("HuGE cache is inconsistent: huge_signal_bfs has %d rows but found %d matrix-row genes" % (runtime_state["huge_signal_bfs"].shape[0], len(matrix_row_genes)))


def write_huge_statistics_runtime_vectors(paths, runtime_state, *, write_vector_fn):
    runtime_vector_map = (
        ("signal_posteriors", "huge_signal_posteriors"),
        ("signal_posteriors_for_regression", "huge_signal_posteriors_for_regression"),
        ("signal_sum_gene_cond_probabilities", "huge_signal_sum_gene_cond_probabilities"),
        ("signal_sum_gene_cond_probabilities_for_regression", "huge_signal_sum_gene_cond_probabilities_for_regression"),
        ("signal_mean_gene_pos", "huge_signal_mean_gene_pos"),
        ("signal_mean_gene_pos_for_regression", "huge_signal_mean_gene_pos_for_regression"),
    )
    for path_key, state_key in runtime_vector_map:
        write_vector_fn(paths[path_key], runtime_state.get(state_key), value_type=float)


def write_huge_statistics_sparse_components(paths, huge_signal_bfs, huge_signal_bfs_for_regression, *, write_vector_fn):
    sparse_vector_map = (
        ("bfs_data", huge_signal_bfs.data, float),
        ("bfs_indices", huge_signal_bfs.indices, int),
        ("bfs_indptr", huge_signal_bfs.indptr, int),
        ("bfs_reg_data", huge_signal_bfs_for_regression.data, float),
        ("bfs_reg_indices", huge_signal_bfs_for_regression.indices, int),
        ("bfs_reg_indptr", huge_signal_bfs_for_regression.indptr, int),
    )
    for path_key, values, value_type in sparse_vector_map:
        write_vector_fn(paths[path_key], values, value_type=value_type)
