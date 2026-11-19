"""Gene provenance from the signal-gene links in a run's HuGE archive."""
import gzip
import json
import tarfile
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix


def load_loci(path: Path, genes, gene_stats):
    with tarfile.open(path, "r:*") as archive:
        files = {member.name: member for member in archive.getmembers() if member.isfile()}

        def content(suffix):
            matches = [entry for name, entry in files.items() if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"{path}: expected one {suffix} entry")
            with archive.extractfile(matches[0]) as stream:
                return gzip.decompress(stream.read()).decode("utf-8")

        meta = json.loads(content(".huge.meta.json.gz"))
        symbols = content(".huge.matrix_row_genes.tsv.gz").splitlines()
        def vector(key, dtype):
            return np.fromstring(content(".huge_signal_bfs."+key+".tsv.gz"), sep=" ", dtype=dtype)
        links = csc_matrix((vector("data", float), vector("indices", int), vector("indptr", int)),
                           shape=meta["huge_signal_bfs_shape"]).tocsr()
    signals = meta["huge_signals"]
    if links.shape != (len(symbols), len(signals)) or len(set(symbols)) != len(symbols):
        raise ValueError(f"Inconsistent signal/gene axes in {path}")
    lookup = {g:i for i,g in enumerate(symbols)}
    by_gene = {}
    for gene in genes:
        i = lookup.get(gene["gene"])
        if i is None:
            continue
        start, end = links.indptr[i:i+2]
        hits = []
        for j, weight in zip(links.indices[start:end],links.data[start:end]):
            chrom, pos, p, *_ = signals[j]
            if weight > 0:
                hits.append([str(chrom), int(pos), float(p), float(weight)])
        if hits:
            hits.sort(key=lambda r:(r[2], -r[3], r[0], r[1]))
            by_gene[gene["gene"]] = hits
    intervals = {}
    for row in gene_stats:
        if row["chrom"] and row["start"] is not None and row["end"] is not None:
            intervals.setdefault(row["chrom"], []).append([row["gene"],int(row["start"]),int(row["end"])])
    for rows in intervals.values():
        rows.sort(key=lambda r:(r[1],r[2],r[0]))
    return {"signals":by_gene, "intervals":intervals, "source":str(path)}
