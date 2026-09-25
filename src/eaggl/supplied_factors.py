"""Input adapters for externally supplied fixed factor bases."""

import csv

import numpy as np


def read_transposed_factors(input_fh, domain):
    """Read Factor x Gene TSV and expose the usual Gene x Factor row contract.

    Row order defines Factor1..FactorK in outputs; input factor names become labels.
    The numeric matrix is retained once; row dictionaries are generated on demand.
    """
    reader = csv.reader(input_fh, delimiter="\t")
    header = next(reader, None)
    if not header or header[0] != "Factor" or len(header) < 2:
        domain.bail("Transposed factor table must start with Factor followed by gene columns")
    genes = header[1:]
    if any(not gene for gene in genes) or len(set(genes)) != len(genes):
        domain.bail("Transposed factor table requires unique, nonempty gene columns")
    names, values = [], []
    seen = set()
    for line_number, row in enumerate(reader, 2):
        if len(row) != len(header):
            domain.bail("Transposed factor table row %d has the wrong number of columns" % line_number)
        name = row[0]
        if not name or name in seen:
            domain.bail("Transposed factor table requires unique, nonempty Factor names")
        seen.add(name)
        names.append(name)
        try:
            values.append(np.asarray(row[1:], dtype=float))
        except ValueError:
            domain.bail("Invalid loading in transposed factor table row %d" % line_number)
    if not values:
        domain.bail("Transposed factor table contains no factors")
    matrix = np.asarray(values).T
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        domain.bail("Gene-factor loadings must be finite and nonnegative")
    columns = ["Factor%d" % (i + 1) for i in range(len(names))]
    rows = (dict(zip(["Gene"] + columns, [gene] + list(matrix[i]))) for i, gene in enumerate(genes))
    return ["Gene"] + columns, rows, dict(enumerate(names))
