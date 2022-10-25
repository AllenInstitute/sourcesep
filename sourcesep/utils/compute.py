import numpy as np
from scipy.fft import fft, fftfreq, ifft


def lowpass(xt, sampling_interval, pass_below):
    """A simple low pass filter

    Args:
        xt (np.array): 1d time series
        sampling_interval (float): sampling rate in Hz for xt
        pass_below (float): low pass frequency threshold in Hz

    Returns:
        xt_filtered: Filtered time series
    """
    xf = fft(xt)
    f = fftfreq(xt.size, sampling_interval)
    xf[np.abs(f) > pass_below] = 0
    xt_filtered = np.real_if_close(ifft(xf))
    return xt_filtered


def gauss_lambda(mu, sigma):
    return lambda x: np.exp(-(x-mu)**2/(sigma**2))