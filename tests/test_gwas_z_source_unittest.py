"""Local-only CLI and HuGE tests with deliberately discordant association columns."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pigean.huge import select_explicit_gwas_p, select_p_derived_z_mask


class GwasZSourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.loc = self.path / "genes.loc"
        self.loc.write_text("ENSG1 1 100000 110000 + GENE1\nENSG2 1 1000000 1010000 + GENE2\nENSG3 1 2000000 2010000 + GENE3\n")
        self.frame = pd.DataFrame(dict(CHR=[1]*3, POS=[100100,1000100,2000100],
                                       P=[1e-12,0.5,1e-9], BETA=[0.2,1.,0.7], SE=[0.1]*3, N=[10000]*3))
        self.counter = 0

    def cli(self, *args):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONHASHSEED="0",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        return subprocess.run([sys.executable,"-m","pigean",*args], cwd=ROOT, env=env,
                              text=True,capture_output=True,timeout=90)

    def fit(self, frame=None, source=None, extra=(), ok=True):
        self.counter += 1
        prefix = self.path / str(self.counter)
        input_path = prefix.with_suffix(".tsv.gz")
        (self.frame if frame is None else frame).to_csv(input_path,sep="\t",index=False,na_rep="NA")
        args = ["huge","--gwas-in",str(input_path),"--gene-loc-file-huge",str(self.loc),
                "--no-correct-huge","--deterministic","--hide-opts","--output-detail","full","--min-n-ratio","0",
                "--min-gwas-inverse-variance-ratio","0","--gene-stats-out",str(prefix.with_suffix(".genes.gz")),
                "--params-out",str(prefix.with_suffix(".params.gz"))]
        if source is not None:args += ["--gwas-z-source",source]
        proc = self.cli(*args,*extra)
        if not ok:
            self.assertNotEqual(proc.returncode,0,proc.stdout+proc.stderr)
            return proc.stdout+proc.stderr
        self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
        frame = pd.read_csv(prefix.with_suffix(".genes.gz"),sep="\t").set_index("Gene")
        payload=prefix.with_suffix(".params.gz").read_bytes()
        params=(gzip.decompress(payload) if payload.startswith(b"\x1f\x8b") else payload).decode()
        return frame,proc.stdout+proc.stderr,params

    def test_explicit_selection_and_invalid_evidence(self):
        self.assertAlmostEqual(select_explicit_gwas_p(.9,1.,.2,"beta-se"),2*norm.sf(5))
        self.assertEqual(select_explicit_gwas_p(.9,1.,.2,"p"),.9)
        for p in [None,np.nan,np.inf,-.1,1.1]:self.assertIsNone(select_explicit_gwas_p(p,1.,.2,"p"))
        for beta,se in [(None,.1),(np.nan,.1),(1.,None),(1.,0),(1.,np.inf)]:
            self.assertIsNone(select_explicit_gwas_p(1e-8,beta,se,"beta-se"))
        np.testing.assert_array_equal(select_p_derived_z_mask(np.array([.1,np.nan]),"beta-se"),[False,False])
        np.testing.assert_array_equal(select_p_derived_z_mask(np.array([.1,np.nan])),[True,False])

    def test_default_p_precedence_even_with_explicit_beta_se(self):
        a,_,_=self.fit()
        b,_,_=self.fit(source="auto",extra=["--gwas-beta-col","BETA","--gwas-se-col","SE"])
        c,_,params=self.fit(source="p")
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        pd.testing.assert_frame_equal(a,c,check_exact=True)
        self.assertIn("gwas_z_source",params)

    def test_beta_se_ignores_detected_p_including_early_filter(self):
        extra=["--gwas-ignore-p-threshold","1e-5","--gwas-beta-col","BETA","--gwas-se-col","SE"]
        a,log,_=self.fit(source="beta-se",extra=extra)
        changed=self.frame.copy();changed.P=[.8,1e-40,.9]
        b,_,_=self.fit(frame=changed,source="beta-se",extra=extra)
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        without_p=self.frame.drop(columns="P")
        c,_,_=self.fit(frame=without_p,source="beta-se",extra=extra)
        pd.testing.assert_frame_equal(a,c,check_exact=True)
        default,_,_=self.fit(extra=extra)
        self.assertGreater(np.max(np.abs(a.huge_score_gwas-default.huge_score_gwas)),.01)
        self.assertIn("association source: beta-se",log)
        self.assertIn("concordance among observed columns",log)

    def test_missing_required_columns_fail_not_infer(self):
        log=self.fit(frame=self.frame.drop(columns="SE"),source="beta-se",ok=False)
        self.assertIn("requires observed beta and SE",log)
        log=self.fit(frame=self.frame.drop(columns="P"),source="p",ok=False)
        self.assertIn("requires a reported p-value column",log)

    def test_missing_required_rows_warn_and_do_not_fallback(self):
        frame=self.frame.copy();frame.loc[0,"SE"]=np.nan
        a,log,params=self.fit(frame=frame,source="beta-se")
        b,_,_=self.fit(frame=frame.iloc[1:],source="beta-se")
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        self.assertIn("Skipped 1 variants missing valid required evidence",log)
        self.assertIn("gwas_z_source_missing_variants",params)
        frame=self.frame.copy();frame.loc[0,"P"]=np.nan
        a,log,_=self.fit(frame=frame,source="p")
        b,_,_=self.fit(frame=frame.iloc[1:],source="p")
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        self.assertIn("Skipped 1 variants missing valid required evidence",log)

    def test_negative_se_uses_magnitude(self):
        frame=self.frame.copy();frame.SE=-frame.SE
        a,log,_=self.fit(frame=frame,source="beta-se")
        b,_,_=self.fit(source="beta-se")
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        self.assertIn("negative standard errors",log)

    def test_beta_se_mismatch_warning_reports_actual_source(self):
        frame=pd.DataFrame(dict(CHR=[1]*120,POS=np.arange(100100,100220),
                                P=np.linspace(.1,.8,120),BETA=np.linspace(.8,1.2,120),SE=[.1]*120,N=[10000]*120))
        _,log,_=self.fit(frame=frame,source="beta-se")
        self.assertIn("GWAS association columns materially disagree",log)
        self.assertIn("will use beta/absolute-SE-derived Z for HuGE Bayes factors",log)
        self.assertNotIn("will use p-derived Z magnitude",log)

    def test_cli_configuration_and_cached_input_guard(self):
        for mode in ["auto","p","beta-se"]:
            proc=self.cli("huge","--gwas-in","local.tsv","--gwas-z-source",mode,"--print-effective-config")
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["options"]["gwas_z_source"],mode)
        proc=self.cli("huge","--gwas-z-source","invalid")
        self.assertNotEqual(proc.returncode,0)
        proc=self.cli("huge","--gwas-in","local.tsv","--huge-statistics-in","cache.tar.gz","--gwas-z-source","beta-se")
        self.assertNotEqual(proc.returncode,0)
        self.assertIn("cached scores cannot be reinterpreted",proc.stdout+proc.stderr)


if __name__ == "__main__":unittest.main()
