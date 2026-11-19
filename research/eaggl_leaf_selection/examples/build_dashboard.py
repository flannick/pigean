"""Generate a small synthetic selected-fit handoff and render its hierarchy dashboard."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH / "src"))

from eaggl_leaf_quality_research.preselection import factor_qc
from eaggl_leaf_quality_research.selection import select


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    # The first two leaves overlap in five genes. The third is separate.
    w = np.zeros((36, 3))
    w[0:8, 0] = .7
    w[3:11, 1] = .8
    w[19:31, 2] = .6
    h = np.array([[.2, .6, 0], [.1, .2, .8], [.9, .1, 0]])
    genes = np.array([f"G{i+1:02}" for i in range(len(w))])
    annotations = np.array(["Set A", "Set B", "Set C"])
    phi = .01
    # Synthetic restart profiles exercise the qualification code without fitting GWAS data.
    rng = np.random.default_rng(19)
    starts = [w * rng.uniform(.95, 1.05, w.shape) for _ in range(5)]
    rows, summary = factor_qc(w, starts)
    for row in rows:
        row["original_consensus_factor"] = row["factor"]
    # The example target is exactly the union prediction of the supplied fit.
    candidate = dict(id="0.01", phi=phi, numerically_qualified=True, global_error=0.,
                     balanced_error=0., distortion=0., **summary)
    decision = select([candidate])
    if decision["selected"] != candidate["id"]:
        raise ValueError("Synthetic leaf fit failed qualification")
    files = {"axes": "axes.npz", "loadings": "fit.npz", "qc": "qc.tsv.gz", "decision": "selection.json"}
    np.savez_compressed(output / files["axes"], genes=genes, gene_sets=annotations)
    np.savez_compressed(output / files["loadings"], W=w, H=h, original_factor_indices=np.arange(w.shape[1]))
    pd.DataFrame(rows).to_csv(output / files["qc"], sep="\t", index=False)
    pd.DataFrame([candidate]).to_csv(output / "phi_summary.tsv.gz", sep="\t", index=False)
    (output / files["decision"]).write_text(json.dumps(decision))
    files = {key: {"path": name, "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
             for key, name in files.items()}
    record = dict(phi=phi, fitted=len(rows), passing=sum(not r["flagged"] for r in rows), files=files)
    manifest = output / "inputs.json"
    manifest.write_text(json.dumps({"traits": {"Example": record}}, indent=2))
    hierarchy = output / "hierarchy"
    subprocess.run([sys.executable, str(RESEARCH / "scripts/build_factor_hierarchy.py"),
                    "--manifest", str(manifest), "--input-root", str(output),
                    "--trait", "Example", "--output", str(hierarchy)], check=True)
    subprocess.run([sys.executable, "-m", "pigean.dashboard", "--eaggl-hierarchy", str(hierarchy),
                    "--hierarchy-source-root", str(output), "--html-out", str(output / "index.html")], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New output directory")
    build(parser.parse_args().output.resolve())
