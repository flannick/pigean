from __future__ import annotations
import csv
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np
from scipy import sparse,stats,integrate

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from eaggl.snp_factor.data import Variant,Weights,Source,fetch_selected,resolve_sources,allele_sign
from eaggl.snp_factor.ld import LD,load_ld,from_bed
from eaggl.snp_factor.weights import construct,apply_anchor,export_scores
from eaggl.snp_factor.runner import run
from eaggl.snp_factor.statistics import mixture_sf,signed_associations,unsigned_associations
from pegs_shared.snp_effectors import signal_gene_allocations
from eaggl import cli


def options(*args):return cli.parser.parse_args(['factor',*args])[0]


def blocks(R):
    def factory():
        yield np.arange(len(R)),R,R*R,True
    return factory


def independent(n):
    def factory():
        for i in range(n):yield np.array([i]),np.ones((1,1)),np.ones((1,1)),True
    return factory


class StatisticsTest(unittest.TestCase):
    def test_signed_joint_matches_individual_regression(self):
        rng=np.random.default_rng(712);x=rng.normal(size=(30000,5));x[:,1]+=.5*x[:,0]
        x=(x-x.mean(0))/x.std(0);R=x.T@x/len(x)
        U=np.array([[1,0],[.5,.4],[0,1],[.3,.2],[.1,.4]])
        scores=x@U;scores/=scores.std(0)
        y=scores@np.array([.03,-.04])+rng.normal(size=len(x));y-=y.mean()
        b=x.T@y/len(x);se=np.full((5,1),y.std()/np.sqrt(len(x)))
        result=signed_associations(sparse.csr_matrix(U),b[:,None],se,blocks(R),['a','b'])
        joint=[r['beta'] for r in result if r['scope']=='joint']
        np.testing.assert_allclose(joint,np.linalg.lstsq(scores,y,rcond=None)[0],atol=1e-12)
        marginal=[r['beta'] for r in result if r['scope']=='marginal']
        np.testing.assert_allclose(marginal,scores.T@y/len(x),atol=1e-12)
        self.assertGreater(abs(joint[0]-marginal[0]),.005)

    def test_orthogonal_and_duplicate_scores(self):
        U=sparse.eye(3,format='csr');b=np.array([[.1],[.2],[-.1]]);se=np.full((3,1),.02)
        rows=signed_associations(U,b,se,blocks(np.eye(3)),['a','b','c'])
        np.testing.assert_allclose([r['beta'] for r in rows[:3]],[r['beta'] for r in rows[3:]])
        U=sparse.csr_matrix([[1,1,0],[0,0,1],[0,0,0]])
        rows=signed_associations(U,b,se,blocks(np.eye(3)),['a','copy','c'])
        joint=[r for r in rows if r['scope']=='joint']
        self.assertEqual(joint[0]['status'],'not_identifiable')
        self.assertEqual(joint[1]['status'],'not_identifiable')
        self.assertEqual(joint[2]['status'],'ok')

    def test_unsigned_gls_and_batch_parity(self):
        rng=np.random.default_rng(19);A=rng.uniform(size=(40,2));y=.3+A@np.array([.7,-.2])+rng.normal(scale=.03,size=40)
        z2=(y+1)[:,None];rows=unsigned_associations(sparse.csr_matrix(A),z2,independent(40),['a','b'])
        expected=np.linalg.lstsq(np.column_stack((np.ones(40),A)),y,rcond=None)[0]
        joint=[r for r in rows if r['scope']=='joint']
        np.testing.assert_allclose([r['beta'] for r in joint],expected[1:],atol=1e-12)
        self.assertTrue(all(r['se']>0 for r in joint))
        batch=unsigned_associations(sparse.csr_matrix(A),np.column_stack((z2,z2)),independent(40),['a','b'])
        for r in rows:
            matches=[v for v in batch if v['method']==r['method'] and v['scope']==r['scope'] and v['factor']==r['factor']]
            self.assertEqual(len(matches),2)
            for v in matches:np.testing.assert_allclose(v['p'],r['p'],atol=1e-10)

    def test_quadratic_null_and_reference_integral(self):
        eig=np.linspace(.3,1.2,20);q=25.
        # Independent Imhof inversion, not the implementation under test.
        def integrand(t):
            if t==0:return eig.sum()-q
            theta=.5*np.arctan(2*t*eig).sum()-t*q
            rho=np.exp(.25*np.log1p(4*t*t*eig*eig).sum())
            return np.sin(theta)/(t*rho)
        reference=.5+integrate.quad(integrand,0,100,epsabs=1e-10,limit=500)[0]/np.pi
        value,method=mixture_sf(q,eig)
        self.assertEqual(method,'saddlepoint');self.assertLess(abs(value-reference),.003)
        rng=np.random.default_rng(31);qs=(rng.normal(size=(1000,len(eig)))**2)@eig
        p=np.array([mixture_sf(v,eig)[0] for v in qs])
        self.assertTrue(.025<np.mean(p<.05)<.075)
        self.assertAlmostEqual(mixture_sf(5,[2,2])[0],stats.chi2.sf(2.5,2))
        self.assertAlmostEqual(mixture_sf(5,[2,0])[0],stats.chi2.sf(2.5,1))

    def test_evidence_rank_and_jackknife_status(self):
        A=sparse.csr_matrix([[1,1],[0,0],[.4,.4]])
        rows=unsigned_associations(A,np.ones((3,1)),independent(3),['a','b'])
        self.assertTrue(all(r['status']=='not_identifiable' for r in rows if r['scope']=='joint'))

    def test_correlated_unsigned_gls_uncertainty_and_recoding(self):
        rng=np.random.default_rng(941);n=30
        A=rng.uniform(size=(n,2));A[:,1]=.7*A[:,0]+.3*A[:,1]
        R=np.kron(np.eye(n//2),np.array([[1.,.6],[.6,1.]]))
        x=np.column_stack((np.ones(n),A));y=x@np.array([.1,.6,-.2])+rng.normal(0,.1,n)
        def factory(matrix):
            def iterate():
                for i in range(0,n,2):
                    ix=np.arange(i,i+2);block=matrix[np.ix_(ix,ix)]
                    yield ix,block,block*block,True
            return iterate
        rows=unsigned_associations(sparse.csr_matrix(A),(y+1)[:,None],factory(R),['a','b'])
        covariance=2*R*R
        def fit(keep):
            xx=x[keep];yy=y[keep];cc=covariance[np.ix_(keep,keep)]
            return np.linalg.solve(xx.T@np.linalg.solve(cc,xx),xx.T@np.linalg.solve(cc,yy))
        expected=fit(np.arange(n));deleted=np.array([fit(np.delete(np.arange(n),[i,i+1])) for i in range(0,n,2)])
        error=np.sqrt((len(deleted)-1)*np.var(deleted,axis=0))
        joint=[r for r in rows if r['scope']=='joint']
        np.testing.assert_allclose([r['beta'] for r in joint],expected[1:],atol=1e-10)
        np.testing.assert_allclose([r['se'] for r in joint],error[1:],atol=1e-10)
        signs=rng.choice([-1,1],n);recoded=signs[:,None]*R*signs[None,:]
        flipped=unsigned_associations(sparse.csr_matrix(A),(y+1)[:,None],factory(recoded),['a','b'])
        np.testing.assert_allclose([r['p'] for r in rows],[r['p'] for r in flipped],atol=1e-12)

    def test_signal_weights_not_association_existence(self):
        indices=np.array([[0,0],[1,1]])
        full=np.array([.1,.3]);linkage=np.array([[.8,.2],[.8,.3]])
        q=signal_gene_allocations(indices,linkage*full,full,{'A':0,'B':1})
        q2=signal_gene_allocations(indices,linkage*full*2,full*2,{'A':0,'B':1})
        self.assertEqual(q,q2);self.assertAlmostEqual(q['A'],.25*.5+.75*.2)
        self.assertAlmostEqual(q['B'],.25*.5+.75*.3)
        self.assertLess(sum(q.values()),1)


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.basis=self.root/'factors.tsv';self.eff=self.root/'effectors.tsv';self.gwas=self.root/'target.tsv'
        self.basis.write_text('Gene\tFactor1\tFactor2\nA\t0.8\t0.1\nB\t0.1\t0.9\nC\t0.4\t0.3\n')
        self.eff.write_text('Gene\tsignal_gene_weight\tlead_variant\tgenome_build\treference_frequency\n'+''.join(f'{g}\t1\t1:{100+i*100}:A:G\tGRCh37\t0.3\n' for i,g in enumerate(['A','B','C','A','B','C'])))
        self.gwas.write_text('chromosome\tposition\treference\talt\tbeta\tstdErr\tpValue\teaf\tinfo\n'+''.join(f'1\t{100+i*100}\tA\tG\t{.1*(i+1)}\t0.1\t{.05/(i+1)}\t.3\t.99\n' for i in range(6)))
    def tearDown(self):self.temp.cleanup()
    def base(self,*extra):return options('--factor-gene-clusters-in',str(self.basis),'--snp-factor-gene-effectors-in',str(self.eff),*extra)
    def test_weight_only_native_roundtrip_and_exports(self):
        native=self.root/'weights.tsv.gz'
        w=run(self.base('--snp-factor-weights-out',str(native),'--snp-factor-phewas-gwas-in','DOES_NOT_EXIST'))
        loaded=Weights.read(native)
        np.testing.assert_array_equal(w.membership.toarray(),loaded.membership.toarray())
        self.assertFalse((self.root/'snp_factor_phewas.tsv.gz').exists())
        ld=LD();sign=apply_anchor(w,str(self.gwas),'sign',ld)
        plink=self.root/'plink.tsv.gz';pgs=self.root/'pgs'
        export_scores(sign,str(plink),str(pgs))
        dosage=np.loadtxt(plink,skiprows=1,usecols=(2,3))
        x=np.array([[0,1,2,1,0,2],[2,2,0,0,1,1]])
        f=sign.frequencies;internal=((x-2*f)/np.sqrt(2*f*(1-f)))@sign.coefficients.toarray()
        offset=np.array(json.loads(Path(str(plink)+'.json').read_text())['centering_offset'])
        np.testing.assert_allclose(x@dosage+offset,internal)
        self.assertEqual(len(json.loads(Path(str(pgs)+'.manifest.json').read_text())['files']),2)
        with self.assertRaises(ValueError):export_scores(w,str(plink))

    def test_missing_anchor_roundtrip_and_saved_export(self):
        w=construct(self.base());w.frequencies[:]=np.nan
        # The final SNP has neither an anchor record nor an independent frequency.
        lines=self.gwas.read_text().splitlines();self.gwas.write_text('\n'.join(lines[:-1])+'\n')
        sign=apply_anchor(w,str(self.gwas),'sign',LD())
        self.assertEqual(sign.coefficients.getrow(5).nnz,0)
        self.assertTrue(np.isnan(sign.frequencies[5]))
        native=self.root/'signed.gz';sign.write(native)
        out=self.root/'saved.plink.gz'
        run(options('--snp-factor-weights-in',str(native),'--snp-factor-weights-plink-out',str(out)))
        with gzip.open(out,'rt') as fh:self.assertEqual(len(fh.read().splitlines()),6)
        with self.assertRaises(ValueError):run(options('--snp-factor-weights-in',str(native),'--snp-factor-gene-weight','cond_prob_huge'))

    def test_sign_only_invariant_to_magnitude(self):
        w=construct(self.base());first=apply_anchor(w,str(self.gwas),'sign',LD())
        rows=self.gwas.read_text().splitlines();updated=[rows[0]]
        for line in rows[1:]:
            fields=line.split('\t');fields[4]=str(float(fields[4])*10);updated.append('\t'.join(fields))
        self.gwas.write_text('\n'.join(updated)+'\n')
        second=apply_anchor(w,str(self.gwas),'sign',LD());magnitude=apply_anchor(w,str(self.gwas),'effect',LD())
        np.testing.assert_array_equal(first.coefficients.toarray(),second.coefficients.toarray())
        self.assertFalse(np.allclose(second.coefficients.toarray(),magnitude.coefficients.toarray()))

    def test_cli_weight_only_and_reload_phewas(self):
        native=self.root/'w.gz';out=self.root/'out.gz'
        from eaggl.app import main
        self.assertEqual(main(['factor','--factor-gene-clusters-in',str(self.basis),'--snp-factor-gene-effectors-in',str(self.eff),'--snp-factor-weights-out',str(native)]),0)
        self.assertEqual(main(['factor','--snp-factor-weights-in',str(native),'--run-snp-factor-phewas','--snp-factor-phewas-gwas-in',str(self.gwas),'--snp-factor-phewas-stats-out',str(out)]),0)
        with gzip.open(out,'rt') as fh:rows=list(csv.DictReader(fh,delimiter='\t'))
        self.assertEqual(len(rows),6)
        self.assertEqual({r['scope'] for r in rows},{'quadratic','marginal','joint'})
        self.assertTrue(all('q' in r for r in rows))

    def test_covered_joint_output_roster_and_single_factor_equivalence(self):
        # Remove a SNP carrying little Factor1 but substantial Factor2 weight.
        lines=self.gwas.read_text().splitlines();del lines[5]
        self.gwas.write_text('\n'.join(lines)+'\n')
        out=self.root/'covered.gz'
        run(self.base('--run-snp-factor-phewas','--snp-factor-phewas-gwas-in',str(self.gwas),
            '--snp-factor-phewas-stats-out',str(out),'--snp-factor-min-coverage','.85',
            '--snp-factor-joint-factor-policy','covered'))
        with gzip.open(out,'rt') as fh:rows=list(csv.DictReader(fh,delimiter='\t'))
        for row in rows:
            self.assertEqual(row['joint_factor_count'],'1')
            self.assertEqual(json.loads(row['joint_factors']),['Factor1'])
            self.assertEqual(row['joint_full_roster'],'False')
            self.assertEqual(row['n_selected_variants'],'5')
        marginal=next(r for r in rows if r['factor']=='Factor1' and r['scope']=='marginal')
        joint=next(r for r in rows if r['factor']=='Factor1' and r['scope']=='joint')
        for field in ['beta','se','p','intercept']:self.assertEqual(marginal[field],joint[field])
        excluded=next(r for r in rows if r['factor']=='Factor2' and r['scope']=='joint')
        self.assertEqual(excluded['status'],'excluded_joint_coverage')
        self.assertEqual(excluded['p'],'NA')

    def test_selected_stream_duplicate_and_negative_se(self):
        with self.gwas.open('a') as fh:fh.write('1\t999\tA\tG\t.1\t-.1\t.02\t.3\t.9\n')
        with fetch_selected(Source(str(self.gwas)),{('1',999)}) as records:
            self.assertEqual(records.qc['selected'],1)
            v=Variant('1',999,'A','G')
            self.assertIsNotNone(records.lookup(v))
            self.assertIsNone(records.lookup(v,signed=True))

    @unittest.skipUnless(shutil.which('tabix') and shutil.which('bgzip'),'HTSlib executable required')
    def test_real_indexed_stream_parity(self):
        compressed=self.root/'target.tsv.gz'
        try:
            with compressed.open('wb') as fh:subprocess.run(['bgzip','-c',str(self.gwas)],stdout=fh,check=True)
        except OSError as exc:
            self.skipTest(f'Installed HTSlib executable is incompatible with this host: {exc}')
        subprocess.run(['tabix','-S','1','-s','1','-b','2','-e','2',str(compressed)],check=True)
        coords={('1',100),('1',400)}
        with fetch_selected(Source(str(compressed)),coords) as indexed,fetch_selected(Source(str(self.gwas)),coords) as streamed:
            self.assertEqual(indexed.access,'tabix')
            for pos in (100,400):self.assertEqual(indexed.lookup(Variant('1',pos,'A','G')),streamed.lookup(Variant('1',pos,'A','G')))

    def test_proxies_rank_phase_and_collapse(self):
        w=construct(self.base());ld=LD()
        original=w.variants[0];p1=Variant('1',700,'A','G');p2=Variant('1',800,'A','G')
        ld.variants.update({v.id:v for v in (original,p1,p2)})
        ld.add_edge(original.id,p2.id,-.9,.81);ld.add_edge(original.id,p1.id,.95,.9025)
        self.assertEqual(ld.candidates(original)[1][0],p1)
        self.assertEqual(ld.candidates(original)[2][1],-.9)
        self.assertIsNone(allele_sign(Variant('1',9,'A','T'),Variant('1',9,'A','T'),signed=True))
        self.assertEqual(allele_sign(Variant('1',9,'G','A'),Variant('1',9,'A','G'),signed=True),-1)

    def test_memory_limit(self):
        o=self.base('--run-snp-factor-phewas','--snp-factor-phewas-gwas-in',str(self.gwas),'--snp-factor-phewas-stats-out',str(self.root/'o.gz'))
        o.snp_factor_memory_mb=.0001
        with self.assertRaises(ValueError):run(o)


class AdapterTest(unittest.TestCase):
    def test_bed_correlation_and_signed_proxies(self):
        with tempfile.TemporaryDirectory() as temp:
            prefix=Path(temp)/'ref'
            npanel=np.array([[0,1,2,0,1,2,0,2],[2,1,0,2,1,0,2,0],[0,0,1,2,1,2,0,1]],dtype=int)
            Path(str(prefix)+'.fam').write_text(''.join(f'F I{i} 0 0 0 -9\n' for i in range(8)))
            Path(str(prefix)+'.bim').write_text('1 rs1 0 100 G A\n1 rs2 0 200 G A\n1 rs3 0 300 G A\n')
            codes={0:3,1:2,2:0}
            with open(str(prefix)+'.bed','wb') as fh:
                fh.write(b'\x6c\x1b\x01')
                for row in npanel:
                    for start in (0,4):fh.write(bytes([sum(codes[int(row[start+j])]<<(2*j) for j in range(4))]))
            ld=from_bed(str(prefix),[Variant('1',100,'A','G')],64,1000,'European')
            self.assertAlmostEqual(ld.pair('1:100:A:G','1:200:A:G')[0],-1)
            self.assertAlmostEqual(ld.frequencies['1:100:A:G'],npanel[0].mean()/2)

    def test_pysam_adapter_queries_only_requested_coordinates(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'x.tsv';source.write_text('chromosome position reference alt pValue\n1 100 A G .1\n1 200 A G .2\n')
            Path(str(source)+'.tbi').write_bytes(b'fixture')
            calls=[]
            class FakeTabix:
                contigs=['1']
                def __init__(self,*args,**kwargs):pass
                def fetch(self,c,start,end):
                    calls.append((c,start,end));return iter(['1 100 A G .1'])
                def close(self):pass
            with patch.dict(sys.modules,{'pysam':SimpleNamespace(TabixFile=FakeTabix)}):
                with fetch_selected(Source(str(source)),{('1',100)}) as records:
                    self.assertEqual(records.access,'tabix');self.assertEqual(records.qc['selected'],1)
            self.assertEqual(calls,[('1',99,100)])

    def test_selectors_and_ancestry_deduplicate(self):
        with tempfile.TemporaryDirectory() as temp:
            ids=Path(temp)/'ids.txt';ids.write_text('T2D\nBMI\n')
            o=options('--snp-factor-phewas-dig-id','T2D','--snp-factor-phewas-dig-id-list',str(ids),'--snp-factor-phewas-dig-ancestry','European','--snp-factor-phewas-dig-ancestry','Mixed')
            sources=resolve_sources(o)
            self.assertEqual(len(sources),4)
            self.assertEqual({s.ancestry for s in sources},{'European','Mixed'})

    def test_stream_memory_does_not_scale_with_rows(self):
        import tracemalloc
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'large.tsv'
            with path.open('w') as fh:
                fh.write('chromosome position reference alt pValue\n')
                for i in range(1,50001):fh.write(f'1 {i} A G .5\n')
            tracemalloc.start()
            with fetch_selected(Source(str(path)),{('1',1),('1',50000)}) as records:
                self.assertEqual(records.qc['selected'],2)
                _,peak=tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.assertLess(peak,4*1024**2)

class HugeEffectorIntegrationTest(unittest.TestCase):
    def test_new_allocations_survive_huge_cache(self):
        repo=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);loc=root/'genes.loc';gwas=root/'gwas.tsv';cache=root/'huge.tar.gz'
            loc.write_text('ENSG1 1 100000 110000 + A\nENSG2 1 200000 210000 + B\nENSG3 1 300000 310000 + C\n')
            gwas.write_text('CHR\tPOS\tP\tBETA\tSE\tN\tREF\tALT\tAF\n1\t100100\t1e-9\t.2\t.03\t10000\tA\tG\t.3\n1\t200100\t1e-10\t.3\t.04\t10000\tC\tT\t.2\n')
            env=dict(os.environ,PYTHONPATH=str(repo/'src'),PYTHONHASHSEED='0')
            common=[sys.executable,'-m','pigean','huge','--deterministic','--hide-opts','--snp-factor-metadata']
            direct=root/'direct.gz';cached=root/'cached.gz'
            direct_args=['--gwas-in',str(gwas),'--gwas-chrom-col','CHR','--gwas-pos-col','POS','--gwas-p-col','P','--gwas-beta-col','BETA','--gwas-se-col','SE','--gwas-n-col','N','--gwas-freq-col','AF','--gwas-build','GRCh37','--gene-loc-file-huge',str(loc),'--no-correct-huge','--huge-statistics-out',str(cache)]
            for extra,output in [(direct_args,direct),(['--huge-statistics-in',str(cache)],cached)]:
                proc=subprocess.run(common+extra+['--gene-effectors-out',str(output),'--gene-stats-out',str(output)+'.genes.gz'],cwd=repo,env=env,text=True,capture_output=True)
                self.assertEqual(proc.returncode,0,proc.stderr)
            with gzip.open(direct,'rt') as fh:rows=list(csv.DictReader(fh,delimiter='\t'))
            with gzip.open(cached,'rt') as fh:reloaded=list(csv.DictReader(fh,delimiter='\t'))
            self.assertTrue(rows);self.assertEqual(rows,reloaded)
            legacy=root/'legacy.gz'
            old_common=[x for x in common if x!='--snp-factor-metadata']
            proc=subprocess.run(old_common+['--huge-statistics-in',str(cache),'--gene-effectors-out',str(legacy)],cwd=repo,env=env,text=True,capture_output=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            with gzip.open(legacy,'rt') as fh: oldrows=list(csv.DictReader(fh,delimiter='\t'))
            self.assertNotIn('signal_gene_weight',oldrows[0])
            self.assertEqual([{k:r[k] for k in oldrows[0]} for r in rows],oldrows)

            self.assertTrue(all(r['genome_build']=='GRCh37' for r in rows))
            self.assertTrue(all(r['lead_variant']!='NA' for r in rows))
            totals={}
            for row in rows:
                value=float(row['signal_gene_weight']);self.assertGreaterEqual(value,0)
                totals[row['signal_id']]=totals.get(row['signal_id'],0)+value
            self.assertTrue(all(v<=1+1e-10 for v in totals.values()))
            # Main's beta/SE mode appends observed p after sample size; allele
            # metadata must not replace either field when extending the tuple.
            beta_se=root/'beta_se.gz'
            args=direct_args[:-2]+['--gwas-z-source','beta-se','--gene-effectors-out',str(beta_se)]
            proc=subprocess.run(common+args,cwd=repo,env=env,text=True,capture_output=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            with gzip.open(beta_se,'rt') as fh: se_rows=list(csv.DictReader(fh,delimiter='\t'))
            self.assertTrue(se_rows)
            self.assertTrue(all(r['lead_variant'] in ('1:100100:A:G','1:200100:C:T') for r in se_rows))


class FullFactorWorkflowTest(unittest.TestCase):
    def test_fit_and_phewas_with_or_without_export(self):
        repo=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);genes=[f'G{i}' for i in range(20)]
            (d/'sets.gmt').write_text(''.join(f'S{k}\tNA\t'+'\t'.join(genes[k*4:(k+1)*4+2])+'\n' for k in range(4)))
            (d/'genes.tsv').write_text('Gene\tlog_bf\tprior\tcombined\n'+''.join(f'{g}\t3\t1\t4\n' for g in genes))
            (d/'sets.tsv').write_text('Gene_Set\tbeta\tbeta_uncorrected\tp_value\n'+''.join(f'S{k}\t1\t1\t.0001\n' for k in range(4)))
            (d/'eff.tsv').write_text('Gene\tsignal_gene_weight\tlead_variant\tgenome_build\n'+''.join(f'{g}\t1\t1:{i+100}:A:G\tGRCh37\n' for i,g in enumerate(genes)))
            (d/'target.tsv').write_text('CHR\tPOS\tP\n'+''.join(f'1\t{i+100}\t{.5/(i+1)}\n' for i in range(20)))
            common=[sys.executable,'-m','eaggl','factor','--hide-opts','--seed','4',
                '--X-in',str(d/'sets.gmt'),'--gene-stats-in',str(d/'genes.tsv'),
                '--gene-set-stats-in',str(d/'sets.tsv'),'--max-num-factors','2','--phi','0.001','--no-trait-linkage',
                '--snp-factor-gene-effectors-in',str(d/'eff.tsv'),'--run-snp-factor-phewas',
                '--snp-factor-phewas-gwas-in',str(d/'target.tsv')]
            env=dict(os.environ,PYTHONPATH=str(repo/'src'),PYTHONHASHSEED='0')
            for name,extra in [('export',['--snp-factor-weights-out',str(d/'weights.gz')]),('no_export',[])]:
                p=subprocess.run(common+extra+['--snp-factor-phewas-stats-out',str(d/(name+'.gz'))],cwd=repo,env=env,capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr[-4000:])
            with gzip.open(d/'export.gz','rt') as a,gzip.open(d/'no_export.gz','rt') as b:self.assertEqual(a.read(),b.read())
            self.assertTrue((d/'weights.gz').exists())

    def test_palindromic_anchor_frequency_orients_sign(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'anchor.tsv'
            path.write_text('CHR POS REF ALT BETA P EAF\n1 100 T A .2 .1 .7\n')
            v=Variant('1',100,'A','T')
            w=Weights([v],['Factor1'],sparse.csr_matrix([[1.]]),sparse.csr_matrix([[1.]]),np.array([.3]),dict(mode='unsigned'))
            signed=apply_anchor(w,str(path),'sign',LD())
            self.assertEqual(signed.coefficients[0,0],-1.)



class ProxySelectionIntegrationTest(unittest.TestCase):
    def test_shared_proxy_is_one_observation_and_phase_transfers(self):
        from eaggl.snp_factor.runner import prepare_target
        import io
        with tempfile.TemporaryDirectory() as tmp:
            v1=Variant('1',100,'A','G');v2=Variant('1',200,'A','G');proxy=Variant('1',300,'A','G')
            w=Weights([v1,v2],['a','b'],sparse.eye(2,format='csr'),sparse.eye(2,format='csr'),np.array([.3,.3]),dict(mode='sign',build='GRCh37'))
            ld=LD();ld.variants.update({v.id:v for v in (v1,v2,proxy)});ld.frequencies[proxy.id]=.3
            ld.add_edge(v1.id,proxy.id,.95,.9025);ld.add_edge(v2.id,proxy.id,-.9,.81)
            path=Path(tmp)/'target.tsv';path.write_text('CHR POS REF ALT BETA SE P EAF\n1 300 A G .2 .1 .05 .3\n')
            candidates=[ld.candidates(v) for v in w.variants]
            audit=csv.writer(io.StringIO(),delimiter='\t')
            item=prepare_target(Source(str(path)),w,ld,candidates,{v.coordinate for group in candidates for v,r,r2 in group},options(),audit)
            self.assertEqual(len(item['variants']),1)
            np.testing.assert_allclose(item['A'].toarray(),[[1.,1.]])
            np.testing.assert_allclose(item['U'].toarray(),[[.95,-.9]])
            np.testing.assert_allclose(item['coverage'],[1.,1.])
            rows=signed_associations(item['U'],item['b'][:,None],item['se'][:,None],blocks(np.eye(1)),w.factors)
            self.assertTrue(all(r['status']=='not_identifiable' for r in rows if r['scope']=='joint'))

if __name__=='__main__':unittest.main()


class ResourceAndFailureTest(unittest.TestCase):
    def test_ld_metadata_budget_and_conflicting_pairs(self):
        ld=LD(memory_mb=.01)
        ld.variants={str(i):Variant('1',i+1,'A','G') for i in range(20)}
        with self.assertRaises(MemoryError):ld.check_budget()
        ld=LD();ld.add_edge('a','b',.8,.64)
        ld.add_edge('a','b',.8,.64)
        self.assertEqual(len(ld.proxies['a']),1)
        with self.assertRaises(ValueError):ld.add_edge('a','b',-.8,.64)

    def test_dig_patch_region_iterator_and_discovery(self):
        import importlib.util,types,io
        from unittest.mock import patch
        repo=Path(__file__).resolve().parents[2]
        package=types.ModuleType('fixture_dig');package.__path__=[]
        api=types.ModuleType('fixture_dig.api')
        api.open_text=lambda *a,**k:io.StringIO(json.dumps({'version':1,'traits':{'Mixed:T':{'uri':'s3://bucket/data.gz','index_uri':'s3://bucket/data.gz.csi'}}}))
        catalog=types.ModuleType('fixture_dig.catalog');catalog.DEFAULT_BUCKET='bucket';catalog.DEFAULT_PREFIX='bottom-line'
        calls=[]
        class Tabix:
            contigs=('1',)
            def __init__(self,*a,**k):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def fetch(self,*args):calls.append(args);return iter(['record'])
        fake=types.ModuleType('pysam');fake.TabixFile=Tabix
        with patch.dict(sys.modules,{'fixture_dig':package,'fixture_dig.api':api,'fixture_dig.catalog':catalog,'pysam':fake}):
            spec=importlib.util.spec_from_file_location('fixture_dig.indexed',repo/'contrib/dig-open-data/indexed.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.assertIsNone(module.indexed_resource('European','absent'))
            self.assertEqual(list(module.iter_trait_regions('Mixed','T',[('1',0,10),('1',5,20),('2',0,1)])),['record'])
            self.assertEqual(calls,[('1',0,20)])


class MultipleTestingTest(unittest.TestCase):
    def test_scope_families_include_failed_tests(self):
        from eaggl.snp_factor.runner import Results
        with tempfile.TemporaryDirectory() as tmp:
            result=Results(str(Path(tmp)/'r.sqlite'))
            try:
                for scope,ps in [('marginal',[.01,.02,None]),('joint',[.1,.3,None])]:
                    for j,p in enumerate(ps):result.add(dict(trait=str(j),factor='A',method='unsigned_evidence',scope=scope,p=p))
                out=Path(tmp)/'out.gz';result.write(out)
                with gzip.open(out,'rt') as fh:rows=list(csv.DictReader(fh,delimiter='\t'))
                np.testing.assert_allclose([float(r['q']) for r in rows if r['scope']=='marginal' and r['p']!='NA'],[.03,.03])
                np.testing.assert_allclose([float(r['q']) for r in rows if r['scope']=='joint' and r['p']!='NA'],[.3,.45])
                self.assertEqual(sum(r['p']=='NA' for r in rows),2)
            finally:result.close()


class CoveredJointTest(unittest.TestCase):
    def test_covered_joint_refits_columns_preserving_rows_and_intercept(self):
        from eaggl.snp_factor.runner import associate_group
        rng=np.random.default_rng(8506);a=rng.uniform(size=(30,3));y=.4+a@np.array([.6,.9,-.2])+rng.normal(0,.05,30)
        variants=[Variant('1',i+1,'A','G') for i in range(30)]
        w=Weights(variants,['a','b','c'],sparse.csr_matrix(a),sparse.csr_matrix(a),np.full(30,.3),{'mode':'unsigned'})
        item=dict(A=w.membership,U=w.coefficients,z2=y+1,variants=variants,coverage=np.array([.9,.7,.86]))
        o=options('--snp-factor-min-coverage','.85','--snp-factor-joint-factor-policy','covered')
        rows=associate_group([item],w,LD(),o)
        joint=[r for r in rows if r['scope']=='joint' and r['status']=='ok']
        x=np.column_stack((np.ones(30),a[:,[0,2]]));expected=np.linalg.lstsq(x,y,rcond=None)[0]
        deleted=np.array([np.linalg.lstsq(np.delete(x,i,axis=0),np.delete(y,i),rcond=None)[0] for i in range(30)])
        se=np.sqrt(29*np.var(deleted,axis=0))
        np.testing.assert_allclose([r['beta'] for r in joint],expected[1:],atol=1e-10)
        np.testing.assert_allclose([r['se'] for r in joint],se[1:],atol=1e-10)
        np.testing.assert_allclose([r['intercept'] for r in joint],expected[0],atol=1e-10)
        excluded=[r for r in rows if r['scope']=='joint' and r['factor']=='b'][0]
        self.assertEqual(excluded['status'],'excluded_joint_coverage')
        full=unsigned_associations(w.membership,(y+1)[:,None],independent(30),w.factors)
        np.testing.assert_allclose([r['beta'] for r in rows if r['scope']=='marginal'],[r['beta'] for r in full if r['scope']=='marginal'])
        item['coverage']=np.zeros(3)
        empty=associate_group([item],w,LD(),o)
        self.assertEqual(sum(r['status']=='no_eligible_joint_factors' for r in empty),3)
        o.snp_factor_joint_factor_policy='all'
        default=associate_group([item],w,LD(),o)
        np.testing.assert_allclose([r['beta'] for r in default if r['scope']=='joint'],[r['beta'] for r in full if r['scope']=='joint'])
