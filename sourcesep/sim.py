
import numpy as np
from numpy.random import default_rng

def emission_spectrum(lam_peak, lam_sigma, lam_range):
    """Returns a simulated indicator emission spectrum (gaussian shape) \n
    Spectrum is normalized to have peak value = 1

    Args:
        lam_peak (float): emission peak
        lam_sigma (float): emission width
        lam_range (np.array): measured wavelengths

    Returns:
        si(lambda,)
    """
    return np.exp(-(lam_range-lam_peak)**2/(lam_sigma**2))



def laser_spectrum(lam_peak, lam_sigma, lam_range):
    """Returns a laser spectrum (gaussian shape) \n
    Spectrum is normalized to have peak value = 1

    Args:
        lam_peak (float): emission peak
        lam_sigma (float): emission width
        lam_range (np.array): measured wavelengths
    Returns:
        ej(lambda,)
    """
    return np.exp(-(lam_range-lam_peak)**2/(lam_sigma**2))


def emission_efficiency(n_lasers, n_indicators, noise_level=0.2):
    """Simulates emission efficiency (identity + random noise)

    Args:
        n_lasers (int):
        n_indicators (int):

    Returns:
        W(ji)
    """
    rng=default_rng()
    W = np.eye((n_lasers, n_indicators)) + noise_level*rng.random((n_lasers, n_indicators))
    return W