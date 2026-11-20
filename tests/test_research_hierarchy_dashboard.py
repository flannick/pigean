"""Portable hierarchy integration and optional Chromium interaction checks."""
import base64
import csv
import gzip
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    out = tmp_path_factory.mktemp("research") / "example"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, "research/eaggl_leaf_selection/examples/build_dashboard.py", str(out)],
                   cwd=ROOT, env=env, check=True)
    return out


def test_portable_example(example):
    html = (example / "index.html").read_text()
    encoded = re.search(r'DATA_PAYLOAD_GZIP_BASE64 = "([^"]+)"', html)[1]
    tree = json.loads(gzip.decompress(base64.b64decode(encoded)))["hierarchies"]["Example"]
    assert len(tree["leaves"]) == 3
    assert len(tree["nodes"]) == 5
    with gzip.open(example / "hierarchy/hierarchy_cuts.tsv.gz", "rt") as stream:
        assert {r["leaf_factors"] for r in csv.DictReader(stream, delimiter="\t")} == {"1,2", "3"}


@pytest.mark.skipif(not os.environ.get("CHROME_BIN"), reason="Set CHROME_BIN to run headless browser checks")
def test_example_browser(example, tmp_path):
    # Run assertions after the dashboard has decoded its offline payload and bound its controls.
    script = """
    <script>
    async function checkDashboard() {
      const el = s => document.querySelector(s);
      const event = (s, value, kind) => {
        el(s).value=value;
        el(s).dispatchEvent(new Event(kind,{bubbles:true}));
      };
      function check(ok,message){if(!ok) throw Error(message)}
      for(let i=0;i<100 && !el('#nscope');i++) await new Promise(r=>setTimeout(r,50));
      event('#nscope','leaves','change');
      const button = tab => el('#eaggl-table-region [data-open-row-tab][data-tab="'+tab+'"]');
      for(const [tab,title] of [['genes','Leading genes'],['gene_sets','Leading gene sets']]){
        button(tab).click();
        check(el('.details-row h3')?.textContent===title,'Expand '+title);
      }
      button('gene_sets').click();
      check(!el('.details-row'),'Collapse detail');
      for(const [tab,pattern,order] of [['genes','^G(0[4-8]|09)$','1,0,2'],['gene_sets','^Set B$','2,1,0']]){
        el('[data-eaggl-section="'+tab+'"]').click();
        const columns=()=>Array.from(document.querySelectorAll('[data-nsort]'),b=>b.dataset.nsort).join(',');
        const original=columns();
        el('[data-weight-order]').click();
        event('#nquery',pattern,'input');
        check(columns()===order,'Order by matching '+tab);
        event('#nmeasure','cosine','change');
        check(columns()===order,'Use raw weights for '+tab);
        el('[data-weight-order]').click();
        check(columns()===original,'Restore '+tab);
      }
    }
    checkDashboard().then(()=>document.body.dataset.testResult='passed')
       .catch(e=>document.body.dataset.testResult=e.message);
    </script>"""
    page = tmp_path / "browser-check.html"
    page.write_text((example / "index.html").read_text().replace("</body>\n</html>", script + "</body>\n</html>"))
    run = subprocess.run([os.environ["CHROME_BIN"], "--headless", "--disable-gpu", "--no-first-run",
                          "--user-data-dir=" + str(tmp_path / "profile"), "--virtual-time-budget=10000",
                          "--dump-dom", page.as_uri()], capture_output=True, text=True, timeout=60, check=True)
    result = re.search(r'<body[^>]*data-test-result="([^"]*)"', run.stdout)
    assert result and result[1] == "passed", result[1] if result else "Browser check did not finish"


def test_label_hierarchy_to_dashboard(example, tmp_path):
    """Exercise both public entry points with frozen leaf names and no provider."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    frozen = tmp_path / "leaves.tsv"
    frozen.write_text("Factor\tlabel\nFactor3\tThird process\nFactor1\tFirst process\nFactor2\tFirst process\n")
    names = tmp_path / "nodes.tsv.gz"
    subprocess.run([sys.executable, "-m", "eaggl", "label-hierarchy", "--hierarchy", str(example/"hierarchy"),
                    "--hierarchy-source-root", str(example), "--leaf-labels-in", str(frozen),
                    "--labels-out", str(names)], env=env, check=True)
    with gzip.open(names, "rt") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [r["node_id"] for r in rows] == ["f1", "f2", "f3", "n3", "n4"]
    assert [r["mechanism"] for r in rows[:3]] == ["First process", "First process", "Third process"]
    assert all(r["source"] == "evidence-caption" for r in rows[3:])
    html = tmp_path / "labeled.html"
    cmd = [sys.executable, "-m", "pigean.dashboard", "--eaggl-hierarchy", str(example/"hierarchy"),
           "--hierarchy-source-root", str(example), "--hierarchy-labels-in", str(names), "--html-out", str(html)]
    subprocess.run(cmd, env=env, check=True)
    encoded = re.search(r'DATA_PAYLOAD_GZIP_BASE64 = "([^"]+)"', html.read_text())[1]
    data = json.loads(gzip.decompress(base64.b64decode(encoded)))
    h = data["hierarchies"]["Example"]
    root = rows[-1]["mechanism"]
    assert h["leafLabels"]["f1"] == "First process" and h["familyLabels"]["n4"] == root
    assert h["views"]["labels"][-1] == f"{root} · Family 4"
    graph_html = data["eaggl_runs"]["Example::hierarchy"]["factor_graph_html"]
    encoded_graph = re.search(r'<script[^>]*id="eaggl-factor-graph-data"[^>]*>(.*?)</script>', graph_html, re.S)[1]
    graph = json.loads(encoded_graph)
    root_node = next(n for n in graph["nodes"] if n["id"] == "n4")
    assert root_node["display_label"] == root_node["label"] == f"{root} · Family 4"
    rows[-1]["fit_signature"] = "unrelated"
    with gzip.open(names, "wt") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    bad = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert bad.returncode and "do not match the fit" in bad.stderr


def test_hosted_family_checkpoint(example, tmp_path, monkeypatch):
    from eaggl import hierarchy_labels, labeling
    from unittest.mock import Mock

    client = Mock(model="test-model")
    client.query.side_effect = ["Leaf A", "Leaf B", "Leaf C", "Family A", "Family B"]
    monkeypatch.setattr(labeling, "build_labeling_client", lambda *args, **kwargs: client)
    saved = tmp_path / "labels.tsv.gz"
    args = ["--hierarchy", str(example/"hierarchy"), "--hierarchy-source-root", str(example),
            "--checkpoint-dir", str(tmp_path/"receipts"), "--label-sources", "genes",
            "--labels-out", str(saved), "--lmm-auth-key", "env"]
    assert hierarchy_labels.main(args) == 0
    assert client.query.call_count == 5
    prompts = [call.args[0] for call in client.query.call_args_list]
    assert all("Set A" not in p and "Example" not in p for p in prompts)
    assert all(name in prompts[-1] for name in ["Leaf A", "Leaf B", "Leaf C", "Family A"])
    assert hierarchy_labels.main(args) == 0
    assert client.query.call_count == 5
