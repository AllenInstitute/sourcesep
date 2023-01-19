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