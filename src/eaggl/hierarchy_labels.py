"""Name and identify nodes of a saved single-trait EAGGL hierarchy."""
import argparse
import base64
import csv
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from . import labeling


def fit_signature(h, views):
    """Bind names to topology, node order, axis IDs and the two loading matrices."""
    topology = [(i, n["factor"], n["children"]) for i, n in sorted(h["nodes"].items())]
    digest = hashlib.sha256(json.dumps([topology, views["keys"]]).encode())
    for field in ("genes", "gene_sets"):
        digest.update(json.dumps(views[field]["rows"]).encode())
        digest.update(views[field]["data"].encode("ascii"))
    return digest.hexdigest()


def open_table(path, mode):
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, mode, encoding="utf-8", newline="")


def read_label_tables(paths):
    tables = {}
    for path in paths:
        with open_table(path, "rt") as source:
            reader = csv.DictReader(source, delimiter="\t")
            required = {"trait", "node_id", "mechanism", "fit_signature"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing hierarchy label columns in {path}: {sorted(required)}")
            for row in reader:
                trait, key = row["trait"], row["node_id"]
                name = (row["mechanism"] or "").strip()
                rows = tables.setdefault(trait, {})
                if not name or key in rows:
                    raise ValueError(f"Empty or duplicate node label for {trait}/{key}")
                rows[key] = (row["fit_signature"], name)
            if reader.line_num <= 1:
                raise ValueError(f"Empty hierarchy label table: {path}")
    return tables


def node_evidence(view, sources, membership):
    from .set_evidence import SetEvidence
    from scipy.sparse import load_npz

    def unpack(field):
        axis = view[field]
        matrix = np.frombuffer(base64.b64decode(axis["data"]), dtype="<f4")
        return np.array(axis["rows"]), matrix.reshape(len(axis["rows"]), len(view["keys"]))

    genes, w = unpack("genes")
    sets, a = unpack("gene_sets")
    overlap = None
    if membership is not None and "gene-sets" in sources:
        matrix = load_npz(membership)
        if matrix.shape != (len(genes), len(sets)):
            raise ValueError("Membership matrix must match the saved gene and gene-set axes")
        overlap = SetEvidence(matrix, w, genes)
    evidence = {}
    for i, key in enumerate(view["keys"]):
        gene_ids = np.lexsort((genes, -w[:, i]))[:5] if "genes" in sources else []
        set_ids = np.lexsort((sets, -a[:, i]))[:5] if "gene-sets" in sources else []
        names = [str(genes[j]) for j in gene_ids if w[j, i] > 0]
        set_ids = [j for j in set_ids if a[j, i] > 0]
        set_names = [str(sets[j]).replace("_", " ") for j in set_ids]
        entries = []
        for j, name in zip(set_ids, set_names):
            if overlap is None:
                entries.append(f"{name} (loading {a[j,i]:.2g})")
            else:
                _, (coverage, enrichment, _, supporters) = overlap.describe(j, [i])
                entries.append(labeling.evidence_entry(name, a[j,i], coverage[0], enrichment[0], supporters[0]))
        evidence[key] = (labeling.label_payload(entries, names), labeling.label_payload(set_names, names),
                         next(iter(set_names or names), ""))
    return evidence


def read_leaves(path, expected):
    if path is None:
        return {}
    with open_table(path, "rt") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        field = "Factor" if "Factor" in (reader.fieldnames or []) else "factor"
        if not {field, "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Expected Factor and label columns in the saved leaf table")
        leaves = {}
        for row in reader:
            key = "f" + row[field].removeprefix("Factor")
            name = (row["label"] or "").strip()
            if not name or name == "NA" or key in leaves:
                raise ValueError(f"Empty or duplicate frozen label for {key}")
            leaves[key] = name
    if leaves.keys() != set(expected):
        raise ValueError("Frozen factor IDs must exactly match the saved leaves, including QC flags")
    return leaves


def generate(h, evidence, frozen, request, context=None, strategy="cross-factor"):
    leaves = [k for k in h["views"]["keys"] if k.startswith("f")]
    labels = dict(frozen)
    if not frozen:
        prompts = labeling.leaf_prompts([evidence[k][0] for k in leaves], [evidence[k][1] for k in leaves],
                                        context=context, strategy=strategy)
        for key, prompt in zip(leaves, prompts):
            labels[key] = request(key, prompt, evidence[key][2]) if prompt else "Unlabeled"
    members = {}
    for i, node in sorted(h["nodes"].items()):
        if node["factor"]:
            members[i] = [f"f{node['factor']}"]
            continue
        children = node["children"]
        keys = [f"f{h['nodes'][c]['factor']}" if h['nodes'][c]['factor'] else f"n{c}" for c in children]
        members[i] = [leaf for c in children for leaf in members[c]]
        if len(set(members[i])) != len(members[i]) or len(members[i]) != node["count"]:
            raise ValueError(f"Inconsistent descendant counts at family {i}")
        child_evidence = [(labels[k], evidence[k][1] or "No positive selected loadings") for k in keys]
        rosters = [[labels[leaf] for leaf in members[c]] for c in children]
        key = f"n{i}"
        prompt = labeling.family_prompt(evidence[key][0], child_evidence, descendant_labels=rosters, context=context)
        caption = " / ".join(dict.fromkeys(evidence[k][2] for k in keys if evidence[k][2]))
        labels[key] = request(key, prompt, caption or "Unlabeled")
    return labels


def atomic_text(path, text):
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def run(args):
    from pigean.dashboard_hierarchy import load_hierarchy
    h = load_hierarchy(args.hierarchy, args.hierarchy_source_root, include_graph=False)
    view = h["views"]
    if args.label_sources not in ("genes", "gene-sets", "genes,gene-sets", "gene-sets,genes"):
        raise ValueError("The saved hierarchy supplies genes and gene-sets as label sources")
    evidence = node_evidence(view, args.label_sources.split(","), args.membership_npz)
    frozen = read_leaves(args.leaf_labels_in, [k for k in view["keys"] if k.startswith("f")])
    def fail(message):
        raise ValueError(message)
    client = (labeling.build_labeling_client(args.lmm_auth_key, args.lmm_model, bail_fn=fail)
              if args.lmm_auth_key else None)
    signature = fit_signature(h, view)
    cache = args.checkpoint_dir
    if cache and client:
        cache.mkdir(parents=True, exist_ok=True)
    def request(key, prompt, caption):
        if not client:
            return caption
        digest = hashlib.sha256(json.dumps([client.model, signature, key, prompt]).encode()).hexdigest()
        receipt = cache / (digest + ".json") if cache else None
        if receipt and receipt.exists():
            result = labeling.parse_label(json.loads(receipt.read_text())["label"])
            if result:
                return result
        for suffix in ("", " Return only a two- or three-word umbrella name."):
            name = labeling.parse_label(client.query(prompt + suffix))
            if name:
                if receipt:
                    atomic_text(receipt, json.dumps({"label": name, "model": client.model}) + "\n")
                return name
        raise ValueError(f"No valid provider label for {key}. Rerun with the same checkpoint directory.")
    labels = generate(h, evidence, frozen, request, args.label_context, args.label_strategy)
    output = args.labels_out
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=output.suffix, delete=False) as stream:
        temp = Path(stream.name)
    try:
        with open_table(temp, "wt") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(("trait", "node_id", "mechanism", "fit_signature", "source"))
            for k in view["keys"]:
                source = "evidence-caption"
                if k in frozen:
                    source = "frozen"
                elif client and (k.startswith("n") or evidence[k][0]):
                    source = client.model
                writer.writerow((h["trait"], k, labels[k], signature, source))
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy", required=True, type=Path, help="Single-trait saved hierarchy directory")
    parser.add_argument("--hierarchy-source-root", type=Path, default=Path("."))
    parser.add_argument("--labels-out", required=True, type=Path)
    parser.add_argument("--leaf-labels-in", type=Path, help="Freeze Factor and label columns from the same fit")
    parser.add_argument("--membership-npz", type=Path, help="Optional sparse gene-by-set matrix in saved axis order")
    parser.add_argument("--label-sources", default="genes,gene-sets")
    parser.add_argument("--label-context")
    parser.add_argument("--label-strategy", choices=("cross-factor", "independent"), default="cross-factor")
    parser.add_argument("--lmm-auth-key", choices=["env"], help="Use OpenAI via OPENAI_API_KEY")
    parser.add_argument("--lmm-model")
    parser.add_argument("--checkpoint-dir", type=Path, help="Reuse successful hosted responses for matching prompts")
    try:
        run(parser.parse_args(argv))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0
