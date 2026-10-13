import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from eaggl import app, cli, factor, factor_runtime as runtime, main_support, trait_linkage
from eaggl.symnmf_numerics import project_nonnegative_rows, profiled_gradient, profiled_objective
from eaggl.state import EagglState


def silent(*args, **kwargs):
    pass


def make_state(W):
    W = np.asarray(W, dtype=float)
    n, k = W.shape
    state = EagglState()
    state.discovery_model = 'gene_by_gene'
    state.genes = ['G%d' % i for i in range(n)]
    state.gene_sets = ['GS%d' % i for i in range(n)]
    state.phenos = ['Trait']
    state.X_orig = sparse.eye(n, format='csr')
    state.gene_in_discovery_mask = np.ones(n, dtype=bool)
    state.gene_set_in_discovery_mask = np.ones(n, dtype=bool)
    state.exp_gene_factors = W.copy()
    state.exp_gene_set_factors = W.copy()
    state.exp_pheno_factors = np.arange(1, k + 1)[None] / 10
    state.exp_lambdak = np.ones(k)
    state.factor_relevance = np.arange(k, 0, -1, dtype=float) / (k + 1)
    state.factor_anchor_relevance = state.factor_relevance[:, None].copy()
    state.factor_marginal_relevance = state.factor_relevance.copy()
    state.factor_anchor_marginal_relevance = state.factor_anchor_relevance.copy()
    state.gene_prob_factor_vector = np.ones((n, 1))
    state.gene_set_prob_factor_vector = np.ones((n, 1))
    state.pheno_prob_factor_vector = np.ones((1, 1))
    return state


FINAL = dict(factor_gene_set_x_pheno=False, factor_top_loading_type='raw',
             lmm_auth_key=None, lmm_model=None, lmm_provider='openai',
             label_gene_sets_only=False, label_include_phenos=False, label_individually=False,
             bail_fn=lambda x: (_ for _ in ()).throw(ValueError(x)), warn_fn=silent,
             log_fn=silent, info_level=1,
             labeling_module=SimpleNamespace(populate_factor_labels=silent))


def test_child_seed_reaches_solver_and_root_rng_is_restored():
    calls = []
    def fit(state, **kwargs):
        calls.append(kwargs['seed'])
        return {}
    np.random.seed(21)
    before = np.random.get_state()
    with mock.patch.object(runtime, '_run_factor_single', side_effect=fit):
        runtime._run_factor_with_seed(object(), seed=1234, run_index=0, factor_kwargs={'seed': 0})
    assert calls == [1234]
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])


def test_sorting_moves_every_exported_linkage_field(tmp_path):
    state = make_state([[.9, 0], [0, .8]])
    state.factor_relevance = np.array([.1, .9])
    state.trait_linkage_nnls = np.array([[.2, .8]])
    state.exp_pheno_factors = state.trait_linkage_nnls.copy()
    state.trait_linkage_joint = state.trait_linkage_nnls.copy()
    state.trait_linkage_cosine = trait_linkage.factor_basis_cosines(state.trait_linkage_nnls)
    state.trait_linkage_euclidean = trait_linkage.factor_basis_euclidean_distances(state.trait_linkage_nnls)
    expected_cosine = state.trait_linkage_cosine[:, ::-1].copy()
    expected_euclidean = state.trait_linkage_euclidean[:, ::-1].copy()
    state.trait_linkage_factor_num_genes = np.array([1, 2])
    state.trait_linkage_factor_weight_sum = np.array([.9, 1.8])
    runtime._finalize_factor_outputs(state, **FINAL)
    np.testing.assert_allclose(state.trait_linkage_nnls, [[.8, .2]])
    np.testing.assert_allclose(state.trait_linkage_cosine, expected_cosine)
    np.testing.assert_allclose(state.trait_linkage_euclidean, expected_euclidean)
    np.testing.assert_array_equal(state.trait_linkage_factor_num_genes, [2, 1])
    runtime._write_final_factor_gmt(state, {'factor_gmt_out': str(tmp_path/'final.gmt'), 'factor_output_scope': 'all'})
    assert (tmp_path/'final.gmt').read_text().splitlines()[0].split('\t')[2] == 'G1'


def test_final_sparsification_updates_cached_discovery_rows():
    state = make_state([[.9, 1e-8], [1e-8, .8]])
    state.exp_gene_factors_full_projected = state.exp_gene_factors.copy()
    runtime._finalize_factor_outputs(state, **FINAL)
    np.testing.assert_array_equal(state.exp_gene_factors_full_projected, state.exp_gene_factors)


def test_final_sparsification_keeps_trait_geometry_consistent():
    state = make_state([[.9, 0], [0, .8]])
    state.exp_pheno_factors = np.array([[.8, 1e-9]])
    state.trait_linkage_nnls = state.exp_pheno_factors.copy()
    state.trait_linkage_cosine = trait_linkage.factor_basis_cosines(state.trait_linkage_nnls)
    state.trait_linkage_euclidean = trait_linkage.factor_basis_euclidean_distances(state.trait_linkage_nnls)
    runtime._finalize_factor_outputs(state, **FINAL)
    np.testing.assert_array_equal(state.trait_linkage_nnls, state.exp_pheno_factors)
    np.testing.assert_allclose(state.trait_linkage_cosine, trait_linkage.factor_basis_cosines(state.exp_pheno_factors))
    np.testing.assert_allclose(state.trait_linkage_euclidean, trait_linkage.factor_basis_euclidean_distances(state.exp_pheno_factors))


def test_consensus_reprojects_final_basis_and_caps_rows():
    matrices = [[[.6, .4, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, .6, .4], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[.4, 0, .6], [1, 0, 0], [0, 1, 0], [0, 0, 1]]]
    states = [make_state(W) for W in matrices]
    for state in states:
        state.exp_gene_factors_full_projected = state.exp_gene_factors.copy()
        state.project_full_gene_factors_gene_by_gene = lambda retained_discovery_gene_factors, cap_genes: retained_discovery_gene_factors.copy()
    summaries = [dict(run_index=i, evidence=i, num_factors=3, factor_gene_set_x_pheno=False) for i in range(3)]
    kwargs = {k: v for k, v in FINAL.items() if k != 'factor_gene_set_x_pheno'}
    kwargs.update(phi=.01, rel_tol=1e-5, no_trait_linkage=True, gene_gene_row_sum_cap=True)
    result, _ = runtime._apply_consensus_solution(states[0], states, summaries,
        min_run_support=.5, aggregation='median', min_factor_cosine=.5, factor_kwargs=kwargs)
    assert np.max(result.exp_gene_factors.sum(axis=1)) <= 1 + 1e-12
    np.testing.assert_allclose(result.exp_gene_factors_full_projected, result.exp_gene_factors)
    np.testing.assert_allclose(result.factor_anchor_marginal_relevance, result.factor_anchor_relevance)


def test_profiled_gradient_matches_finite_difference():
    rng = np.random.default_rng(11)
    W = rng.uniform(.1, .3, (3, 2))
    target = rng.uniform(0, 1, (3, 3)); target = (target + target.T) / 2
    weights = rng.uniform(0, 1, (3, 3))
    args = (target, weights, .07, 12., .1, .02)
    analytic = profiled_gradient(W, *args)
    numeric = np.zeros_like(W)
    for index in np.ndindex(W.shape):
        a, b = W.copy(), W.copy()
        a[index] += 1e-6; b[index] -= 1e-6
        numeric[index] = (profiled_objective(a, *args)[0] - profiled_objective(b, *args)[0]) / 2e-6
    np.testing.assert_allclose(analytic, numeric, atol=1e-7)


def test_projected_solver_removes_two_cycle_and_descends():
    state = EagglState()
    state.internal_symnmf_update_method = 'projected_gradient'
    state.internal_symnmf_stopping = 'fixed'
    W, _, _, _, error = state._bayes_sym_nmf_l2_extension(
        [[0, .25], [.25, 0]], K0=1, phi=0, seed=123, max_iter=2000, rel_tol=1e-6)
    # With an unweighted diagonal, the optimum fixes the product, not each loading.
    np.testing.assert_allclose(W[0, 0] * W[1, 0], .25, atol=1e-6)
    assert error < 1e-10
    assert state.last_factorization_converged
    assert np.max(np.diff(state.last_factorization_objective_trace)) <= 1e-10


def test_good_short_fit_does_not_bypass_stability_window():
    state = EagglState()
    state.internal_symnmf_update_method = 'projected_gradient'
    W, _, _, _, error = state._bayes_sym_nmf_l2_extension(
        [[0, .25], [.25, 0]], K0=1, phi=0, seed=123, max_iter=1000, rel_tol=1e-6)
    np.testing.assert_allclose(W[0, 0] * W[1, 0], .25, atol=1e-6)
    assert error < 1e-10
    assert not state.last_factorization_converged


def test_fixed_projection_is_deterministic_and_permutation_equivariant():
    state = EagglState()
    W = np.array([[.8, .1], [.2, .9]])
    V = np.array([[.3, .1], [.5, .7]])
    first = state._project_H_with_fixed_W(W, V, None, None)
    np.random.seed(987)
    second = state._project_H_with_fixed_W(W, V, None, None)
    np.testing.assert_array_equal(first, second)
    swapped = state._project_H_with_fixed_W(W[:, ::-1], V, None, None)
    np.testing.assert_allclose(first, swapped[:, ::-1], atol=1e-12)


def test_extreme_log_scores_do_not_overflow():
    p = trait_linkage._as_probability_surface([[1000, -1000]], 'combined', background_log_bf=-3)
    np.testing.assert_array_equal(p, [[1, 0]])


def test_supplied_ids_and_labels(tmp_path):
    domain = SimpleNamespace(open_gz=lambda p, mode: open(p, mode),
        bail=lambda x: (_ for _ in ()).throw(ValueError(x)),
        options=SimpleNamespace(factor_gene_clusters_layout='genes-by-factors'), log=silent, INFO=1)
    gene = tmp_path/'gene.tsv'
    gene.write_text('Gene\tcluster\tlabel\tFactor3\tFactor6\nG1\tFactor3\timmune\t.8\t.2\nG2\tFactor6\tmetabolic\t.1\t.9\n')
    state = EagglState()
    factor.load_existing_factor_gene_clusters(domain, state, str(gene))
    assert state.factor_labels == ['immune', 'metabolic']
    sets = tmp_path/'sets.tsv'
    sets.write_text('Gene_Set\tFactor3\tFactor7\nGS1\t.8\t.2\n')
    with pytest.raises(ValueError, match='IDs disagree'):
        factor.load_existing_factor_gene_set_clusters(domain, state, str(sets))


@pytest.mark.parametrize('value', ['nan', 'inf', '-1'])
def test_invalid_supplied_gene_set_loadings_are_rejected(tmp_path, value):
    domain = SimpleNamespace(open_gz=lambda p, mode: open(p, mode),
        bail=lambda x: (_ for _ in ()).throw(ValueError(x)))
    path = tmp_path/'sets.tsv'; path.write_text('Gene_Set\tFactor1\nGS1\t%s\n' % value)
    with pytest.raises(ValueError, match='finite and nonnegative'):
        factor.load_existing_factor_gene_set_clusters(domain, EagglState(), str(path))


def test_extra_export_does_not_change_discovery_mask_or_trait_scores(tmp_path):
    genes = tmp_path/'genes.tsv'; genes.write_text(
        'Gene\tin_discovery\tcluster\tlabel\tFactor1\tFactor2\n'
        'G1\tTrue\tFactor1\tfirst\t.8\t0\nG2\tTrue\tFactor2\tsecond\t0\t.8\n'
        'G3\tFalse\tFactor2\tsecond\t0\t.8\n')
    phenos = tmp_path/'phenos.tsv'; phenos.write_text(
        'Gene\tPheno\tcombined\tlog_bf\tprior\nG1\tTrait\t4\t4\t0\nG2\tTrait\t0\t0\t0\nG3\tTrait\t8\t8\t0\n')
    annotations = tmp_path/'X.txt'; annotations.write_text('GS1 G1 G3\nGS2 G2\n')
    results = []
    for extra in (False, True):
        command = ['factor', '--factor-gene-clusters-in', str(genes), '--gene-phewas-bfs-in', str(phenos),
                   '--trait-factor-links-out', str(tmp_path/('links%s.tsv' % extra)), '--seed', '7']
        if extra:
            command += ['--X-in', str(annotations), '--gene-set-clusters-out', str(tmp_path/'sets.tsv')]
        assert main_support._bootstrap_cli(command)
        state = app.run_main_pipeline(cli.options).state
        results.append((state.gene_in_discovery_mask.copy(), state.trait_linkage_nnls.copy()))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_allclose(results[0][1], results[1][1], atol=1e-12)


def test_fresh_fit_projects_extra_traits_without_changing_discovery(tmp_path):
    genes = tmp_path/'gene_stats.tsv'
    genes.write_text('Gene\tcombined\tlog_bf\tprior\nG1\t5\t3\t2\nG2\t4\t2\t2\nG3\t4\t2\t2\n')
    sets = tmp_path/'gene_set_stats.tsv'
    sets.write_text('Gene_Set\tbeta\tbeta_uncorrected\tp\nGS1\t.5\t.5\t.001\nGS2\t.4\t.4\t.001\n')
    annotations = tmp_path/'X.txt'; annotations.write_text('GS1 G1 G2\nGS2 G2 G3\n')
    phenos = tmp_path/'phenos.tsv'
    phenos.write_text('Gene\tPheno\tcombined\tlog_bf\tprior\nG1\tTestTrait\t5\t3\t2\nG2\tTestTrait\t4\t2\t2\n')
    states = []
    for extra in (False, True):
        command = ['factor', '--X-in', str(annotations), '--gene-stats-in', str(genes),
                   '--gene-set-stats-in', str(sets), '--discovery-model', 'gene_by_gene',
                   '--seed', '7', '--max-num-factors', '2', '--phi', '.01']
        if extra:
            command += ['--gene-phewas-bfs-in', str(phenos), '--trait-factor-links-out', str(tmp_path/'links.tsv')]
        assert main_support._bootstrap_cli(command)
        states.append(app.run_main_pipeline(cli.options).state)
    np.testing.assert_array_equal(states[0].exp_gene_factors, states[1].exp_gene_factors)
    np.testing.assert_array_equal(states[0].factor_anchor_relevance, states[1].factor_anchor_relevance)
    assert states[1].trait_linkage_nnls is not None
    assert (tmp_path/'links.tsv').exists()
    np.testing.assert_allclose(states[1].exp_pheno_factors, states[1].trait_linkage_nnls)


def test_no_consensus_support_is_explicitly_flagged():
    states = [make_state([[1, 0], [0, 1]]), make_state([[.5, .5], [.5, .5]])]
    summaries = [dict(run_index=i, evidence=i, num_factors=2, factor_gene_set_x_pheno=False) for i in range(2)]
    _, diagnostics = runtime._apply_consensus_solution(states[0], states, summaries,
        min_run_support=1., aggregation='median', min_factor_cosine=.99, factor_kwargs={})
    assert diagnostics['mode'] == 'consensus_fallback_reference'
    assert diagnostics['fallback_to_reference']
