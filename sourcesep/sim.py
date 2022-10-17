import numpy as np
import pandas as pd
import toml
from dysts.flows import Lorenz
from sourcesep.utils.config import load_config
from sourcesep.utils.compute import lowpass


class SimData():
    def __init__(self, T=None, cfg_path=None):
        self.cfg = toml.load(cfg_path)
        
        self.J = len(self.cfg['laser'])            # Number of excitation lasers (input channels)
        self.L = self.cfg['sensor']['n_channels']  # Number of sensor pixels (output channels)
        self.I = len(self.cfg['indicator'])        # Number of indicators
        self.T = T                                 # Number of samples --> TODO: rename?

        self.set_arrays()                          # set time and wavelength arrays

        self.E = None
        self.W = None
        self.S = None
        self.Mu_ox = None
        self.Mu_dox = None

        self.A_model = Lorenz()                    # Activity model
        self.H_ox_model = Lowpass_Gauss()          # Hemodynamics model (oxy hemo)
        self.H_dox_model = Lowpass_Gauss()         # Hemodynamics model (deoxy hemo)
        self.N_model = Lowpass_Gauss()             # Laser noise model
        self.M_model = Lowpass_Gauss()             # Motion artifact model

        self.paths = load_config(dataset_key='all') # Paths for data files
        self.rng = np.random.default_rng()

    def set_arrays(self):
        # time stamps
        self.T_arr = np.linspace(0,
                                 (self.T-1) / self.cfg['sensor']['sampling_freq_Hz'],
                                 self.T)

        # wavelengths measured
        self.L_arr = np.linspace(self.cfg['sensor']['lambda_min'],
                                 self.cfg['sensor']['lambda_max'],
                                 self.cfg['sensor']['n_channels'])
        return

    def get_S(self):
        if self.S is None:
            icfg = self.cfg['indicator']
            S = []
            for i, k in enumerate(icfg.keys()):
                if icfg[k]['from_path']:
                    df = pd.read_csv(
                        self.paths['spectra'] / icfg[k]['spectrum_path'])
                    df.rename(columns=dict(zip(df.columns, ['wavelength', 'exc', 'em', '2p'])), inplace=True)
                    df['em'] = df['em'].fillna(0)
                    S.append(np.interp(x=self.L_arr, xp=df['wavelength'], fp=df['em']))
                else:
                    # Implement synthetic spectrum
                    pass
            S = np.vstack(S)
            assert S.shape == (self.I, self.L), 'check spectra shape'
            self.S = S

        return self.S

    def get_W(self):
        if self.W is None:
            icfg = self.cfg['indicator']
            lcfg = self.cfg['laser']

            laser_freq = [lcfg[name]['em_wavelength_nm'] for name in lcfg.keys()]
            laser_freq = np.array(laser_freq)

            indicator = []
            W = np.zeros((self.I, self.J))
            for i, k in enumerate(icfg.keys()):
                if icfg[k]['from_path']:
                    df = pd.read_csv(
                        self.paths['spectra'] / icfg[k]['spectrum_path'])
                    df.rename(columns=dict(zip(df.columns, ['wavelength', 'exc', 'em', '2p'])),
                              inplace=True)
                    df['exc'] = df['exc'].fillna(0)
                    W[i, :] = np.interp(
                        x=laser_freq, xp=df['wavelength'], fp=df['exc'])
                    indicator.append(k)
            W_df = pd.DataFrame(
                W, columns=laser_freq.astype(int), index=indicator)
            self.W = W
            self.W_df = W_df
        return self.W

    def get_E(self):
        if self.E is None:
            # laser spectrum is approximately a delta function
            # sigma is a small fixed value
            lcfg = self.cfg['laser']
            sigma = 1
            E = []
            for k in lcfg.keys():
                mu = lcfg[k]['em_wavelength_nm']
                y = np.exp(-(self.L_arr-mu)**2/(sigma**2))
                y = y/np.max(y)
                E.append(y)
            E = np.vstack(E)
            assert E.shape == (self.J, self.L), 'check laser spectrum (E) shape'
            self.E = E

        return self.E

    def get_Mu(self):
        if (self.Mu_ox is None) and (self.Mu_dox is None):
            df = pd.read_csv(self.paths['spectra']/ self.cfg['hemodynamics']['spectrum_path'])
            scale = np.max([df['Hb02 (cm-1/M)'].max(),df['Hb (cm-1/M)'].max()])
            self.Mu_ox = np.interp(self.L_arr, df['wavelength'], df['Hb02 (cm-1/M)']) / scale
            self.Mu_dox = np.interp(self.L_arr, df['wavelength'], df['Hb (cm-1/M)']) / scale

        Mu_ox = self.Mu_ox
        Mu_dox = self.Mu_dox
        return Mu_ox, Mu_dox

    def A_model(self):
        self.model.ic = self.rng.integers(1, high=100)*self.rng.random(3)
        A = self.model.make_trajectory(self.T, pts_per_period=100, resample=True, standardize=True)
        A = A[:,:self.I]
        assert A.shape == (self.T, self.I), 'check generated shape of A'
        return A

    def compose_obs(self):
        # These are all constants:
        S = self.get_S()
        W = self.get_W()
        E = self.get_E()
        Mu_ox, Mu_dox = self.get_Mu()

        A = self.A_model()

        AS = np.einsum('ti,il->til', A, S)
        ASW = np.einsum('til,ij->tjl', AS, W)
        E = np.einsum('td,djl -> tjl', np.ones((self.T,1)), E[np.newaxis,...])
        ASWE = ASW + E

        # 2nd term
        H_ox = self.rng.random((self.T,))
        H_dox = self.rng.random((self.T,))

        HD = np.einsum('td,dl -> tl', H_ox[...,np.newaxis], Mu_ox[np.newaxis,...]) \
            + np.einsum('td,dl -> tl', H_dox[...,np.newaxis], Mu_dox[np.newaxis,...])

        HD = np.einsum('j,tl -> tjl', np.ones((self.J,)), HD)

        M = self.rng.random((self.T,))
        B = self.rng.random((self.J,self.L))

        HDM = np.einsum('tjl,t -> tjl', HD, M)
        B = np.einsum('t,jl -> tjl', np.ones((self.T,)), B)
        H = HDM + B

        # 3rd term
        N = self.rng.random((self.T,self.J))
        N = np.einsum('l,tj -> tjl', np.ones((self.L,)), N)
        assert np.array_equal(N[:,:,0],N[:,:,1]), 'broadcast check'

        O = np.einsum('tjl,tjl -> tjl', ASWE, H)
        O = np.einsum('tjl,tjl -> tjl', O, N)
        return O


    def gen_H(self):
        hdyn = self.cfg['hemodynamics']
        sampling_interval = 1/self.cfg['sensor']['sampling_freq_Hz']

        H_ox_pre = hdyn['H_ox_amp'] * self.rng.standard_normal(size=(self.T,))
        H_ox = lowpass(xt=H_ox_pre,
                       sampling_interval=sampling_interval,
                       pass_below=hdyn['lowpass_thr_Hz'])

        H_dox_pre = hdyn['H_ox_amp'] * self.rng.standard_normal(size=(self.T,))
        H_dox = lowpass(xt=H_dox_pre,
                        sampling_interval=sampling_interval,
                        pass_below=hdyn['lowpass_thr_Hz'])
        return H_ox, H_dox

    def gen_N(self):
        # Simulating N - this can be estimated from diffraction pattern around saturated pixels. 
        # For now we can treat this as a known 'constant' (i.e. doesn't need to be fit) in the model
        # Simulated with amplitude of 2% compared to that for activity
        0.02* self.rng.standard_normal(size=(self.T,))


def gauss_lambda(mu, sigma):
    return lambda x: np.exp(-(x-mu)**2/(sigma**2))


def Lowpass_Gauss():
    return