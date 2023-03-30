import itertools
import numpy as np
from scipy.fft import fft, fftfreq, ifft


def lowpass(xt, sampling_interval, pass_below, axis=-1):
    """A simple low pass filter

    Args:
        xt (np.array): 1d time series
        sampling_interval (float): sampling rate in Hz for xt
        pass_below (float): low pass frequency threshold in Hz

    Returns:
        xt_filtered: Filtered time series
    """
    xf = fft(xt, axis=axis)
    f = fftfreq(xt.shape[axis], sampling_interval)
    ind = np.abs(f) > pass_below

    new_shape = np.ones((np.ndim(xf),), dtype=int)
    new_shape[axis] = np.shape(xf)[axis]
    ind_array = np.broadcast_to(ind.reshape(new_shape),xf.shape)
    xf[ind_array] = 0
    xt_filtered = np.real_if_close(ifft(xf, axis=axis))
    return xt_filtered


def gauss_lambda(mu, sigma):
    return lambda x: np.exp(-(x-mu)**2/(sigma**2))


def perm_avgabscorr(X, Y, dim=1):
    """Compute the absolute correlation between X and Y over all channels
    
    Args:
        X (np.array): default shape is [time, channels]
        Y (np.array): default shape is [time, channels]
        dim (int, optional): Dimension to permute over. Defaults to 1.
    """

    n_dims = X.shape[dim]
    assert n_dims == Y.shape[dim], 'X and Y must have the same number of channels'
    assert n_dims < 8, 'Aborting. Number of permutations is too large'
    # perm_idx contains all permutations of [0,1,2]: e.g. [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]
    perm_idx = np.array(list(itertools.permutations(range(n_dims))))
    corrs = np.zeros(perm_idx.shape[0])
    for i, perm in enumerate(perm_idx):
        corrs[i] = np.nanmean(np.abs([np.corrcoef(X[:, perm[j]], Y[:, j])[0, 1] for j in range(Y.shape[1])]))
    return perm_idx, corrs


def test_perm_avgabscorr():
    # test_perm_abscorr()
    x = np.random.rand(1000, 3)
    x_perm = x[:,[1,2,0]].copy()
    perms, z = perm_avgabscorr(x, x_perm)
    
    assert tuple(perms[np.argmax(z)]) == (1,2,0), 'permuted correlations failed test'
    return