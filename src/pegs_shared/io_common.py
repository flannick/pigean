import csv
import gzip

from pegs_shared.bundle import ensure_parent_dir_for_file
from pegs_shared.cli import _default_bail


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


def read_tsv(path, key_column=None, required_columns=None, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail
    required = set(required_columns or [])

    from pegs_utils import open_text_auto

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
