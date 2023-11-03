from sourcesep.models.nmf_variants import vectorized_grad_sparsity_ratio_X, grad_sparsity_ratio_x
import numpy as np


def test_grad_sparsity(verbose=False):
    X = np.random.rand(3,2)

    grad = vectorized_grad_sparsity_ratio_X(X, axis=0)
    grad_ = grad_sparsity_ratio_x(X[:,0])
    assert np.allclose(grad_, grad[:,0]), "mismatch!"

    grad = vectorized_grad_sparsity_ratio_X(X, axis=1)
    grad_ = grad_sparsity_ratio_x(X[0,:])
    assert np.allclose(grad_, grad[0,:]), "mismatch!"

    if verbose: print('passed')
    return