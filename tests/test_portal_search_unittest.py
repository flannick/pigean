from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test_portal_unittest import GENE_STATS, GENE_SET_STATS, PHENOTYPES_FLAT, _write
from pigean import portal, portal_db, portal_search, portal_server


class PortalSearchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "portal.sqlite"
        self.genes = self.root / "genes.tsv"
        self.sets = self.root / "sets.tsv"
        _write(self.genes, GENE_STATS)
        _write(self.sets, GENE_SET_STATS)
        _write(self.root / "phenotypes.tsv", PHENOTYPES_FLAT)
        self.graph = {
            "schema": "eaggl_factor_graph/v1",
            "nodes": [
                {"id": "Factor1", "kind": "factor", "label": "Immune response", "relevance": 0.8,
                 "provenance": {"relevance_by_anchor": [{"anchor": "T2D", "beta": 0.6, "joint_fraction": 0.75},
                                                         {"anchor": "UnloadedTrait", "beta": 0.2}]}},
                {"id": "Factor2", "kind": "factor", "label": "Metabolic", "relevance": 0.4},
                {"id": "T2D", "kind": "trait"},
            ],
            "edges": [{"from": "Factor1", "to": "T2D", "kind": "factor_trait", "weight": 0.75}],
            "candidate_nodes": [{"id": "IBD", "kind": "trait"}],
            "candidate_edges": [{"from": "Factor1", "to": "IBD", "kind": "factor_trait", "weight": 0.2}],
        }
        graph_path = self.root / "graph.html"
        self.write_graph(graph_path, self.graph)
        args = ["build", "--db", str(self.db), "--phenotype-file", str(self.root / "phenotypes.tsv"),
                "--gene-filter", "combined>1", "--gene-set-filter", "beta>0.01"]
        for model, trait, seed in [("large", "T2D", "s1"), ("large", "T2D", "s2"), ("small", "IBD", "s1")]:
            genes, sets = self.genes, self.sets
            if model == 'small':
                genes, sets = self.root / 'small_genes.tsv', self.root / 'small_sets.tsv'
                _write(genes, GENE_STATS + 'SMALL_ONLY\t2\t3\t1\t0.5\t5\t1\t100\t200\n')
                _write(sets, GENE_SET_STATS + 'SMALL_SET\tsmall library\t10\t0.5\t1.5\t1e-5\t4.4\n')
            args += ["--package", f"model={model},trait={trait},run={seed},gene_stats={genes},gene_set_stats={sets},factor_graph={graph_path}"]
        self.assertEqual(portal.main(args), 0)
        self.state = portal_server.PortalState(self.db, title="test", plotly_src="about:blank")
        self.conn = self.state.connection()
        self.addCleanup(self.conn.close)

    def write_graph(self, path, graph):
        path.write_text('<html><script id="eaggl-factor-graph-data" type="application/json">' + json.dumps(graph) + '</script></html>')

    def api(self, endpoint, **params):
        return portal_server.handle_api(self.state, endpoint, {k: [str(v)] for k, v in params.items()})

    def test_search_across_models_counts_distinct_traits_and_respects_filters(self):
        status, body = self.api('/api/search', kind='gene', q='gene1')
        self.assertEqual(status, 200)
        self.assertEqual(body['matches'], [{'id': 'GENE1', 'label': '', 'n_runs': 3, 'n_traits': 2}])
        self.assertEqual(self.api('/api/search', kind='gene', q='gene1', model='large')[1]['matches'][0]['n_traits'], 1)
        self.assertEqual(self.api('/api/search', kind='gene', q='GENELOW')[1]['matches'], [])
        self.assertEqual(self.api('/api/search', kind='gene', q='gene1', model='missing')[1]['matches'], [])
        rows = self.api('/api/gene_across', id='GENE1')[1]['rows']
        self.assertEqual(next(r for r in rows if r['trait'] == 'T2D')['phenotype_name'], 'Type 2 diabetes (T2D)')

    def test_geneset_identifier_library_literal_search_and_limits(self):
        self.assertEqual([m['id'] for m in self.api('/api/search', kind='gene_set', q='set a')[1]['matches']], ['SET_A'])
        self.assertEqual([m['id'] for m in self.api('/api/search', kind='gene_set', q='set_')[1]['matches']], ['SET_A', 'SET_B'])
        for query in ['%', "' OR 1=1 --", 'SET_LOW', '']:
            self.assertEqual(self.api('/api/search', kind='gene_set', q=query)[1]['matches'], [])
        self.assertEqual(len(self.api('/api/search', kind='gene', q='GENE', limit=1)[1]['matches']), 1)
        self.assertEqual(self.api('/api/search', kind='unknown', q='a')[0], 400)
        self.assertEqual(self.api('/api/search', kind='gene', q='a', limit='invalid')[0], 400)

    def test_search_finds_entries_absent_from_the_initial_run(self):
        for kind, ident in [('gene', 'SMALL_ONLY'), ('gene_set', 'SMALL_SET')]:
            match = self.api('/api/search', kind=kind, q=ident.lower())[1]['matches'][0]
            self.assertEqual((match['id'], match['n_traits'], match['n_runs']), (ident, 1, 1))
            self.assertEqual(self.api('/api/search', kind=kind, q=ident, model='large')[1]['matches'], [])
            endpoint = '/api/gene_across' if kind == 'gene' else '/api/gene_set_across'
            self.assertEqual(self.api(endpoint, id=ident)[1]['rows'][0]['run_id'], 'small__IBD__s1')

    def test_factor_search_retains_run_identity_and_model_scope(self):
        status, body = self.api('/api/search', kind='factor', q='immune')
        self.assertEqual(status, 200)
        matches = body['matches']
        self.assertEqual(len(matches), 3)
        self.assertEqual(len({m['run_id'] for m in matches}), 3)
        self.assertTrue(all(m['id'] == 'Factor1' and m['n_traits'] == 3 for m in matches))
        self.assertNotIn('html', json.dumps(body))
        self.assertEqual(len(self.api('/api/search', kind='factor', q='factor1', model='small')[1]['matches']), 1)
        self.assertEqual(len(self.api('/api/search', kind='factor', q='metab')[1]['matches']), 3)

    def test_factor_traits_merge_provenance_and_graph_without_inventing_links(self):
        status, detail = self.api('/api/factor', run='large__T2D__s1', id='Factor1')
        self.assertEqual(status, 200)
        rows = {r['trait']: r for r in detail['rows']}
        self.assertEqual(set(rows), {'T2D', 'IBD', 'UnloadedTrait'})
        self.assertEqual(rows['T2D']['beta'], 0.6)
        self.assertEqual(rows['T2D']['graph_weight'], 0.75)
        self.assertEqual(rows['T2D']['phenotype_name'], 'Type 2 diabetes (T2D)')
        self.assertEqual(rows['T2D']['run_ids'], ['large__T2D__s1', 'large__T2D__s2'])
        self.assertEqual(rows['IBD']['run_ids'], [])  # other model is not silently substituted
        self.assertEqual(rows['UnloadedTrait']['run_ids'], [])
        self.assertEqual(self.api('/api/factor', run='large__T2D__s1', id='Factor2')[1]['rows'], [])
        self.assertEqual(self.api('/api/factor', run='large__T2D__s1', id='Factor3')[0], 404)
        self.assertEqual(self.api('/api/factor', run='large__T2D__s1')[0], 400)
        self.assertEqual(self.api('/api/factor', run='missing', id='Factor1')[0], 404)

    def test_factor_content_changes_and_non_eaggl_html(self):
        # Existing graph storage can be read without migration; changing its contents invalidates the parse cache.
        self.assertEqual(len(portal_search.factor_catalog(self.conn)), 6)
        writer = portal_db.open_database(self.db)
        try:
            writer.execute("UPDATE run_factor_graphs SET html='<p>No structured EAGGL data</p>'")
            writer.commit()
            self.assertEqual(portal_search.factor_catalog(self.conn), [])
            writer.execute("DROP TABLE run_factor_graphs")
            writer.commit()
            self.assertEqual(portal_search.factor_catalog(self.conn), [])
            self.assertIsNone(portal_search.factor_detail(self.conn, 'large__T2D__s1', 'Factor1'))
            self.assertEqual(len(self.api('/api/search', kind='gene', q='gene1')[1]['matches']), 1)
        finally:
            writer.close()

    def test_malformed_embedded_graph_is_not_executable_or_a_search_failure(self):
        for content in ['not json', 'null', '[]', '{"schema":"unrecognized"}',
                        '{"schema":"eaggl_factor_graph/v1","nodes":null,"edges":{}}']:
            self.assertEqual(portal_search._graph_factors('<script id="eaggl-factor-graph-data">' + content + '</script>'), [])


if __name__ == '__main__':
    unittest.main()
