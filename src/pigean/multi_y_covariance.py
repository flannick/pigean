"""Explicit, trait-labelled residual correlations for the factor-GMT workflow."""
import csv,gzip,hashlib,zipfile
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
import numpy as np
from pegs_shared.covariance import validate_correlation


def load_batch_correlations(manifest, traits, genes, *, response_scale, memory_bytes):
    root=Path(manifest).resolve().parent
    with open(manifest) as f:
        reader=csv.DictReader(f,delimiter='\t')
        if not {'trait','matrix','genes','response_scale'} <= set(reader.fieldnames or []):
            raise ValueError('Correlation manifest needs trait, matrix, genes, response_scale columns')
        entries={}
        for row in reader:
            if row['trait'] in entries:
                raise ValueError('Duplicate trait in correlation manifest: '+row['trait'])
            entries[row['trait']]=row
    matrices=[];provenance=[];used=0
    for trait in traits:
        if trait not in entries:
            raise ValueError('Missing correlation matrix for trait '+trait)
        row=entries[trait]
        if row['response_scale'] != response_scale:
            raise ValueError('Correlation response_scale must match the fitted response: '+response_scale)
        matrix_path=root/row['matrix'];gene_path=root/row['genes']
        op=gzip.open if str(gene_path).endswith('.gz') else open
        with op(gene_path,'rt') as f:labels=[line.strip() for line in f if line.strip()]
        if len(set(labels)) != len(labels):
            raise ValueError('Duplicate gene labels in correlation input')
        index={g:i for i,g in enumerate(labels)}
        if any(g not in index for g in genes):
            raise ValueError('Correlation input does not cover the complete gene universe')
        with zipfile.ZipFile(matrix_path) as archive:
            size=sum(entry.file_size for entry in archive.infolist())
        # Allow for the loaded CSR, reordered copy, and symmetry validation workspace.
        validation_workspace = 24*len(genes)*8 + (2*len(genes)**2*8 if len(genes) <= 128 else 0)
        if used+4*size+validation_workspace > memory_bytes:
            raise ValueError('Correlation matrices exceed the memory budget; reduce --multi-y-max-phenos-per-batch')
        full=sparse.load_npz(matrix_path).tocsr()
        if full.shape != (len(labels),len(labels)):
            raise ValueError('Correlation matrix and gene labels have different dimensions')
        indices=np.array([index[g] for g in genes]);r=validate_correlation(full[indices,:][:,indices],len(genes))
        # Reject indefinite inputs before any reported coefficient is tested.
        try:
            smallest = (np.linalg.eigvalsh(r.toarray())[0] if len(genes) <= 128 else
                        eigsh(r, k=1, which='SA', return_eigenvectors=False,
                              v0=np.random.default_rng(0).normal(size=len(genes)),
                              tol=1e-7, maxiter=1000)[0])
        except ArpackNoConvergence as exc:
            raise ValueError('Could not verify positive semidefiniteness of correlation input') from exc
        if smallest < -1e-7:
            raise ValueError('Correlation input must be positive semidefinite')
        used += r.data.nbytes+r.indices.nbytes+r.indptr.nbytes
        matrices.append(r)
        def digest(p):
            h=hashlib.sha256()
            with open(p,'rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            return h.hexdigest()
        provenance.append(dict(trait=trait,matrix=str(matrix_path),matrix_sha256=digest(matrix_path),genes=str(gene_path),genes_sha256=digest(gene_path),response_scale=response_scale))
    return matrices,provenance
