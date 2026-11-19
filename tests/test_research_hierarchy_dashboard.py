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
