import numpy as np
from tqdm import tqdm

class Hyperparameter:
    def __init__(self, name, init_val, schedule):
        self.name = name
        self.init_val = init_val
        self.schedule = schedule

    def value(self, iteration):
        return self.schedule(self.init_val, iteration)

def exp_decay(init_val, iteration, rate=0.01):
    return init_val * np.exp(-rate * iteration)

def exp_sat(init_val, iteration, rate=0.01):
    return init_val * (1 - np.exp(-rate * iteration))


def init_AX(Y, A=None, X=None, rank=None, A_keep=None, X_keep=None):
    if (X is not None) and (A is not None) and (rank is not None):
        print('X and A are initialized, ignoring rank.')

    if A is None:
        assert rank is not None, 'rank must be specified if A is not provided'
        A = np.random.rand(Y.shape[0], rank)

    if X is None:
        assert rank is not None, 'rank must be specified if X is not provided'
        X = np.random.rand(rank, Y.shape[1])

    if A_keep is None:
        A_keep = np.ones_like(A, dtype=bool)

    if X_keep is None:
        X_keep = np.ones_like(X, dtype=bool)
    
    return A, X, A_keep, X_keep


def fro_norm_grad_A(Y, A, X):
    return -1 * (Y - A @ X) @ X.T

def fro_norm_grad_X(Y, A, X):
    return -1 * A.T @ (Y - A @ X)

def fro_norm_mu_stepsize_A(Y, A, X):
    step = A / ((A @ X @ X.T) + 1e-8)
    step[step>1.0] = 1.0
    return step

def fro_norm_mu_stepsize_X(Y, A, X):
    step = X / ((A.T @ A @ X) + 1e-8)
    step[step>1.0] = 1.0
    return step

def grad_2norm_x(x):
    return 2 * x

def grad_sparsity_ratio_x(x):
    # Grad wrt x for $L = \frac{\sum_k(|x_k|)}{\sum_k(x_k^2)}$
    assert np.all(x>=0), 'x must be non-negative, assumption violated'
    l1norm = np.sum(np.abs(x))
    l2norm = np.linalg.norm(x, ord=2)
    grad = (1/l2norm)*(1 - (l1norm*(x))/(l2norm**2))
    return grad

def vectorized_grad_sparsity_ratio_X(X, axis=None):
    # Each vector is along the specified axis
    assert X.ndim==2, 'expecting X to be a matrix'
    l1norm = np.linalg.norm(X, ord=1, axis=axis, keepdims=True)
    l2norm = np.linalg.norm(X, ord=2, axis=axis, keepdims=True)
    l1norm = np.broadcast_to(l1norm, X.shape)
    l2norm = np.broadcast_to(l2norm, X.shape)
    grad = (1/l2norm) - (l1norm*X)/(l2norm**3)
    return grad

def deviation_grad(x, ref):
    # L = |x-a|_2. Gradient as per matrix cookbook eqn 129
    grad = (x.ravel() - ref.ravel()) / np.linalg.norm(x.ravel() - ref.ravel(), ord=2)
    return grad

def set_zeros(M, keep):
    M[~keep] = 0
    return M

def set_norm(M, axis=0, c=1.0):
    norm = np.linalg.norm(M, axis=axis, keepdims=True)
    return c * (M / norm), norm/c

def nmf_recipe_01(Y, A=None, X=None, rank=None,
                  A_keep=None, X_keep=None,
                  grad_type=None,
                  step_size_A=0.01, 
                  step_size_X=0.01,
                  lam_sparsity_x=0.0,
                  lam_2norm_x = 0.0, 
                  lam_deviation_x = 0.0,
                  tol=1e-4,
                  exit_tol=1e-4, 
                  max_iter=1000):

    A, X, A_keep, X_keep = init_AX(Y, A=A, X=X, rank=rank, A_keep=A_keep, X_keep=X_keep)

    err_list = []
    init = {'A':A.copy(), 'X':X.copy()}
    for _ in tqdm(range(max_iter)):
        # gradient step for frobenius norm part
        dX_fro_norm = fro_norm_grad_X(Y, A, X)
        dA_fro_norm = fro_norm_grad_A(Y, A, X)

        if grad_type == 'mu':
            mu_step_X = fro_norm_mu_stepsize_X(Y, A, X)
            mu_step_A = fro_norm_mu_stepsize_A(Y, A, X)
        else:
            mu_step_X = 1.0
            mu_step_A = 1.0

        X += -step_size_X * np.multiply(mu_step_X, dX_fro_norm)
        A += -step_size_A * np.multiply(mu_step_A, dA_fro_norm)

        # gradient step for deviation
        A[:,0] += -lam_deviation_x * step_size_A * deviation_grad(A[:,0], init['A'][:,0])

        # project to non-negative orthant
        A[A < tol] = tol
        X[X < tol] = tol

        # gradient step for smoothing. remove mean before using this. 
        X[0,:] += -lam_2norm_x * step_size_X * grad_2norm_x(X[0,:] - np.mean(X[0,:]))

        # project to non-negative orthant
        A[A < tol] = tol
        X[X < tol] = tol

        # gradient step for sparsity across columns of A
        A += -lam_sparsity_x * step_size_A * vectorized_grad_sparsity_ratio_X(A, axis=1)

        # project to non-negative orthant
        A[A < tol] = tol
        X[X < tol] = tol

        # setting known zero constraints
        A = set_zeros(A, A_keep)
        X = set_zeros(X, X_keep)

        # force columns of A to fixed norm
        A, factor = set_norm(A, axis=0, c=1.0)

        # rescale rows of X by the same factor
        X = np.diag(factor.ravel()) @ X
        err = np.linalg.norm(Y - A @ X, 'fro')
        err_list.append(err)

        if len(err_list)>1:
            del_err = np.abs(err_list[-1] - err_list[-2])
            if del_err<exit_tol:
                break

    return A, X, err_list, init


def test_nmf_recipe_01():
    import pickle
    from sourcesep.utils.config import load_config
    paths = load_config(dataset_key='all')
    fname = paths['root'] / 'sims' / 'nmf_test_01.pkl'
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    A_init = data['A'].copy() + 0.1 * np.random.rand(*data['A'].shape)
    A_init = A_init + np.min(A_init) + 1e-2
    A_init, _ = set_norm(A_init, axis=0, c=1.0)
    
    A, X, err, init = nmf_recipe_01(data['Y'].copy(), A=A_init, X=None, rank=2,
                  A_keep=None, X_keep=None,
                  grad_type=None, step_size_A=0.01, step_size_X=0.01,
                  lam_2norm_x = 0.0, lam_deviation_x = 0.0,
                  tol=1e-4, exit_tol=1e-4, 
                  max_iter=9999)

    results = {'A':A, 'X':X, 'err':err}
    return data, results, init


if __name__ == '__main__':
    test_nmf_recipe_01()
    pass

