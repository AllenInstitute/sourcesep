import itertools
import numpy as np
import xarray as xr
import torch
from scipy.fft import fft, fftfreq, ifft
from tqdm import tqdm


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
    ind_array = np.broadcast_to(ind.reshape(new_shape), xf.shape)
    xf[ind_array] = 0
    xt_filtered = np.real_if_close(ifft(xf, axis=axis))
    return xt_filtered


def custom_sigmoid(x, bottom=0, top=1.0, beta=1.0):
    return bottom + (top - bottom) / (1 + np.exp(-beta * x))


def softplus(x, beta=40, thr=10.0):
    """Soft version of ReLU.
    (1/beta)log(1+exp(beta*x))
    Large beta makes it closer to ReLU.
    Depending on choice of beta, there will be a (small) discontinuity at x=thr.

    Args:
        x (np.array): 1d array
        beta (float): (1/beta)np.log1p(exp(beta*x)
        thr (float): beta*x > thr are returned as is

    Returns:
        np.array: same shape as x
    """
    ind = beta * x < thr
    exp_x = np.where(ind, np.exp(beta * x), 0)
    return np.where(ind, (1 / beta) * np.log1p(exp_x), x)


def gauss_lambda(mu, sigma):
    """Returns a gaussian function with mean mu and standard deviation sigma"""
    return lambda x: np.exp(-((x - mu) ** 2) / (sigma**2))


def perm_avgabscorr(X, Y, dim=1):
    """Compute the absolute correlation between X and Y over all channels

    Args:
        X (np.array): default shape is [time, channels]
        Y (np.array): default shape is [time, channels]
        dim (int, optional): Dimension to permute over. Defaults to 1.
    """

    n_dims = X.shape[dim]
    assert n_dims == Y.shape[dim], "X and Y must have the same number of channels"
    assert n_dims < 8, "Aborting. Number of permutations is too large"
    # perm_idx contains all permutations of [0,1,2]: e.g. [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]
    perm_idx = np.array(list(itertools.permutations(range(n_dims))))
    corrs = np.zeros(perm_idx.shape[0])
    for i, perm in enumerate(perm_idx):
        corrs[i] = np.nanmean(np.abs([np.corrcoef(X[:, perm[j]], Y[:, j])[0, 1] for j in range(Y.shape[1])]))
    return perm_idx, corrs


def test_perm_avgabscorr():
    """test for perm_avgabscorr"""
    # test_perm_abscorr()
    x = np.random.rand(1000, 3)
    x_perm = x[:, [1, 2, 0]].copy()
    perms, z = perm_avgabscorr(x, x_perm)

    assert tuple(perms[np.argmax(z)]) == (1, 2, 0), "permuted correlations failed test"
    return


def welch_psd(signal, n_per_segment=100, n_overlap=None, sampling_freq=20, verbose=False):
    """Calculate power spectral density for a 1d signal using Welch's method
    1. Split 1d `signal` into segments of length `n_per_segment`
    2. Apply Hann window to each segment
    3. Take FFT, and average the power spectrum across segments

    Args:
        signal: 1d time series
        n_per_segment: number of samples to use per
        n_overlap: determines if segments overlap
        sampling_freq: to keep calculations in physical units

    Returns:
        freqs_welch: array of frequencies corresponding to psd
        psd_welch: psd
    """

    if n_overlap is None:
        n_overlap = n_per_segment // 2
    assert n_per_segment <= signal.shape[0], "n_per_segment must be less than signal length"
    assert n_per_segment > n_overlap, "n_per_segment must be greater than n_overlap"

    segments = signal.unfold(0, n_per_segment, n_per_segment - n_overlap)
    windows = torch.hann_window(n_per_segment)
    windowed_segments = segments * windows
    fft_segments = torch.fft.rfft(windowed_segments)

    if verbose:
        print(f"Segment shape: {segments.shape}")

    # TODO: revisit normalization here:
    psd_welch = torch.mean(torch.abs(fft_segments) ** 2, dim=0) / (torch.sum(windows**2))
    freqs_welch = torch.fft.rfftfreq(n_per_segment, d=1 / sampling_freq)

    return freqs_welch, psd_welch


def quadratic_bezier(p0, p1, p2, t):
    """Quadratic Bezier curve

    Args:
        p0 (tuple): (x,y) control point
        p1 (tuple): (x,y) control point
        p2 (tuple): (x,y) control point
        t (float): parameter along the curve.

    Returns:
        (x,y): Value at given t
    """
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def solve_quadratic_bezier(p0x, p1x, p2x, x):
    """Solves for t in a quadratic Bezier curve given x values of control points.

    Args:
        p0x (float): x value of control point
        p1x (float): x value of control point
        p2x (float): x value of control point
        t (float): parameter along the curve.
    """
    a = p0x - 2 * p1x + p2x
    b = 2 * (p1x - p0x)
    c = p0x - x
    t = np.roots([a, b, c])
    t = t[(t >= 0) & (t <= 1)]
    return t


def cubic_bezier(p0, p1, p2, p3, t):
    """Cubic Bezier curve

    Args:
        p0 (tuple): (x,y) control point
        p1 (tuple): (x,y) control point
        p2 (tuple): (x,y) control point
        p3 (tuple): (x,y) control point
        t (float): parameter along the curve.

    Returns:
        (x,y): Value at given t
    """
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def detrend_1d(signal):
    x = np.arange(len(signal))
    p = np.polyfit(x, signal, 4)
    polyfit = np.polyval(p, x)
    detrended = signal - polyfit + np.mean(signal)
    return detrended, polyfit


def detrend_signal(F):
    """Independently fit each entry in F along the time axis.

    Args:
        F (xr.DataArray): n-dim float array with fluoresence signal
    """
    assert isinstance(
        F, xr.core.dataarray.DataArray
    ), "Expecting F to be an xarray DataArray with a named time dimension"
    F_detrended = xr.full_like(F, 0, dtype=np.double)
    F_detrended = F_detrended.transpose("time", ...)
    for i in tqdm(range(F.shape[1])):
        for j in range(F.shape[2]):
            F_detrended[:, i, j] = detrend_1d(F[:, i, j])[0]

    F_detrended = F_detrended.transpose(*F.dims)
    return F_detrended
