"""Optional fused scalar conditional arithmetic; residual products stay separate."""
import numpy as np

try:
    from numba import njit
except ImportError as error:
    raise ImportError('inner-fused requires the optional numba dependency') from error


@njit(cache=True, fastmath=False, error_model='numpy')
def conditional_kernel(indices, residual, alpha, uniforms, normals, shrink,
                       c_const, d_const, hdmpn, se2, scale, sampled,
                       probability, mean, saved_residual, gauss_seidel):
    width = sampled.shape[2]
    for chain in range(sampled.shape[0]):
        for k in range(len(indices)):
            parallel = indices[k] // width
            gene_set = indices[k] % width
            r = residual[chain, k]
            b2 = r * r
            d = d_const[parallel, gene_set] * np.exp(-b2 / (se2[parallel, gene_set] * 2.0))
            numerator = c_const[parallel, gene_set] * np.exp(-b2 / (2.0 * hdmpn[parallel, gene_set]))
            denominator = numerator + d
            if numerator == 0:
                denominator = 1.0
            postp = numerator / denominator
            postmean = shrink[parallel, gene_set] * (postp * r)
            probability[chain, parallel, gene_set] = postp
            mean[chain, parallel, gene_set] = postmean
            if gauss_seidel:
                proposal = postmean
            else:
                proposal = shrink[parallel, gene_set] * r + scale[parallel, gene_set] * normals[chain, parallel, gene_set]
                if uniforms[chain, parallel, gene_set] >= postp * alpha:
                    proposal = 0.0
            sampled[chain, parallel, gene_set] = proposal
            saved_residual[chain, parallel, gene_set] = r


def prepare_inner_compiled():
    """Compile the normal float64/C-array signature without drawing randomness."""
    matrix = np.ones((1, 1), dtype=np.float64)
    tensor = np.ones((1, 1, 1), dtype=np.float64)
    conditional_kernel(np.array([0], dtype=np.int64), matrix, 1.0,
        tensor, tensor, matrix, matrix, matrix, matrix, matrix, matrix,
        tensor.copy(), tensor.copy(), tensor.copy(), tensor.copy(), False)
