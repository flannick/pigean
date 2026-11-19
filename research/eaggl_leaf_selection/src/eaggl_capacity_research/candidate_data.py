"""Frozen Main input adapter; no reconstruction from graph-only annotation panels."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.special import expit, logit


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class SavedInputs:
    def __init__(self, folder):
        self.folder = Path(folder)
        paths = [self.folder / n for n in ['native_factor_state.npz',
                 'inputs/metadata.npz', 'inputs/X.npz', 'inputs/X_missing.npz',
                 'params.out.gz', 'receipt.json']]
        self.hashes = {str(p.resolve()): sha(p) for p in paths}
        with np.load(paths[0], allow_pickle=False) as f:
            z = {k: f[k] for k in f.files}
        with np.load(paths[1], allow_pickle=False) as f:
            m = {k: f[k] for k in f.files}
        self.X = sparse.load_npz(paths[2]).tocsr()
        self.X_missing = sparse.load_npz(paths[3]).tocsr()
        self.genes = m['genes']; self.genes_missing = m['genes_missing']
        self.gene_sets = m['gene_sets']; self.mask = m['mask'].astype(bool)
        self.beta = m['betas'].reshape(-1) / np.where(m['scales'] == 0, 1, m['scales']).reshape(-1)
        self.pg = z['gene_prob_factor_vector']; self.ps = z['gene_set_prob_factor_vector']
        np.testing.assert_array_equal(self.genes, z['genes'])
        np.testing.assert_array_equal(self.gene_sets, z['gene_sets'])
        assert self.X.shape == (len(self.genes), len(self.gene_sets))
        assert self.X_missing.shape == (len(self.genes_missing), len(self.gene_sets))
        assert len(set(self.genes) & set(self.genes_missing)) == 0
        import pandas as pd
        p = pd.read_csv(paths[4], sep='\t').drop_duplicates('Parameter', keep='last').set_index('Parameter').Value
        self.prior = float(p['gene_gene_pair_prior'])
        self.floor = float(p['gene_gene_matrix_floor'])
        if str(p['gene_gene_excess_probability']) != 'True' or str(p['gene_gene_logbf_base']) != 'natural':
            raise ValueError('This adapter requires the frozen default excess-probability target')
        inv = np.argsort(z['_phi_selection_target_gene_indices'])
        self.target = z['_phi_selection_target_matrix'][np.ix_(inv, inv)].copy()
        np.fill_diagonal(self.target, 0.)
        check = self.cross_target(self.X); np.fill_diagonal(check, 0.)
        np.testing.assert_allclose(check, self.target, atol=1e-14, rtol=1e-12)
        self.target_hash = array_hash(self.target)
        self.genes_hash = hashlib.sha256(json.dumps(self.genes.tolist()).encode()).hexdigest()

    def cross_target(self, x):
        evidence = (x[:, self.mask].multiply(self.beta[self.mask]) @ self.X[:, self.mask].T).toarray()
        t = np.clip((expit(logit(self.prior) + evidence) - self.prior) / (1 - self.prior), 0, 1)
        t[t < self.floor] = 0.
        return t
