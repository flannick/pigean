import csv
import gzip
import io
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np

from pegs_shared.bundle import ensure_parent_dir_for_file
from pegs_shared.cli import _default_bail, is_remote_path


DIG_OPEN_DATA_PREFIX = "dig-open-data:"
DIG_OPEN_DATA_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")


class TsvTable(object):
    def __init__(self, columns, rows, key_column=None, by_key=None):
        self.columns = columns
        self.rows = rows
        self.key_column = key_column
        self.by_key = by_key


class GeneStatsTable(TsvTable):
    KEY_COLUMN = "Gene"
    REQUIRED_COLUMNS = ["Gene"]

    @classmethod
    def read(cls, path, *, bail_fn=None):
        table = read_tsv(
            path,
            key_column=cls.KEY_COLUMN,
            required_columns=cls.REQUIRED_COLUMNS,
            bail_fn=bail_fn,
        )
        return cls(
            columns=table.columns,
            rows=table.rows,
            key_column=table.key_column,
            by_key=table.by_key,
        )


class GeneSetStatsTable(TsvTable):
    KEY_COLUMN = "Gene_Set"
    REQUIRED_COLUMNS = ["Gene_Set"]

    @classmethod
    def read(cls, path, *, bail_fn=None):
        table = read_tsv(
            path,
            key_column=cls.KEY_COLUMN,
            required_columns=cls.REQUIRED_COLUMNS,
            bail_fn=bail_fn,
        )
        return cls(
            columns=table.columns,
            rows=table.rows,
            key_column=table.key_column,
            by_key=table.by_key,
        )


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


def parse_gene_map_file(
    gene_map_in,
    *,
    gene_map_orig_gene_col=1,
    gene_map_new_gene_col=2,
    allow_multi=False,
    bail_fn=None,
):
    if bail_fn is None:
        bail_fn = _default_bail

    gene_label_map = {}
    gene_map_orig_gene_col -= 1
    if gene_map_orig_gene_col < 0:
        bail_fn("--gene-map-orig-gene-col must be greater than 1")
    gene_map_new_gene_col -= 1
    if gene_map_new_gene_col < 0:
        bail_fn("--gene-map-new-gene-col must be greater than 1")

    with open(gene_map_in) as map_fh:
        for line in map_fh:
            cols = line.strip('\n').split()
            if len(cols) <= gene_map_orig_gene_col or len(cols) <= gene_map_new_gene_col:
                bail_fn("Not enough columns in --gene-map-in:\n\t%s" % line)

            orig_gene = cols[0]
            new_gene = cols[1]
            if allow_multi:
                if orig_gene not in gene_label_map:
                    gene_label_map[orig_gene] = set()
                gene_label_map[orig_gene].add(new_gene)
            else:
                gene_label_map[orig_gene] = new_gene
    return gene_label_map


def read_loc_file_with_gene_map(
    loc_file,
    *,
    gene_label_map=None,
    return_intervals=False,
    hold_out_chrom=None,
    clean_chrom_fn=None,
    warn_fn=None,
    bail_fn=None,
    split_gene_length=1000000,
):
    if clean_chrom_fn is None:
        clean_chrom_fn = clean_chrom_name
    if warn_fn is None:
        warn_fn = lambda _msg: None
    if bail_fn is None:
        bail_fn = _default_bail

    gene_to_chrom = {}
    gene_to_pos = {}
    gene_chrom_name_pos = {}
    chrom_interval_to_gene = {}

    with open(loc_file) as loc_fh:
        for line in loc_fh:
            cols = line.strip('\n').split()
            if len(cols) != 6:
                bail_fn(
                    "Format for --gene-loc-file is:\n\tgene_id\tchrom\tstart\tstop\tstrand\tgene_name\nOffending line:\n\t%s"
                    % line
                )
            gene = cols[5]
            if gene_label_map is not None and gene in gene_label_map:
                gene = gene_label_map[gene]
            chrom = clean_chrom_fn(cols[1])
            if hold_out_chrom is not None and chrom == hold_out_chrom:
                continue
            pos1 = int(cols[2])
            pos2 = int(cols[3])

            if gene in gene_to_chrom and gene_to_chrom[gene] != chrom:
                warn_fn("Gene %s appears multiple times with different chromosomes; keeping only first" % gene)
                continue

            if gene in gene_to_pos and np.abs(np.mean(gene_to_pos[gene]) - np.mean((pos1, pos2))) > 1e7:
                warn_fn("Gene %s appears multiple times with far away positions; keeping only first" % gene)
                continue

            gene_to_chrom[gene] = chrom
            gene_to_pos[gene] = (pos1, pos2)

            if chrom not in gene_chrom_name_pos:
                gene_chrom_name_pos[chrom] = {}
            if gene not in gene_chrom_name_pos[chrom]:
                gene_chrom_name_pos[chrom][gene] = set()
            gene_chrom_name_pos[chrom][gene].add(pos1)
            gene_chrom_name_pos[chrom][gene].add(pos2)

            if pos2 < pos1:
                pos1, pos2 = pos2, pos1

            if return_intervals:
                if chrom not in chrom_interval_to_gene:
                    chrom_interval_to_gene[chrom] = {}
                if (pos1, pos2) not in chrom_interval_to_gene[chrom]:
                    chrom_interval_to_gene[chrom][(pos1, pos2)] = []
                chrom_interval_to_gene[chrom][(pos1, pos2)].append(gene)

            if pos2 > pos1:
                for posm in range(pos1, pos2, split_gene_length)[1:]:
                    gene_chrom_name_pos[chrom][gene].add(posm)

    if return_intervals:
        return chrom_interval_to_gene
    return (gene_chrom_name_pos, gene_to_chrom, gene_to_pos)


def construct_map_to_ind(values):
    return dict([(values[i], i) for i in range(len(values))])
