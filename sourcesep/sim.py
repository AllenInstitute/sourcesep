import numpy as np
import pandas as pd
import toml
from scipy import signal
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
        self.paths = load_config(dataset_key='all')# Paths for data files
        self.rng = np.random.default_rng()

    def set_arrays(self):
        # time stamps
        self.T_arr = np.linspace(0,
                                 (self.T-1) / self.cfg['sensor']['sampling_freq_Hz'],
                                 self.T)

        # wavelengths measured
        self.L_arr = np.linspace(self.cfg['sensor']['lambda_min_nm'],
                                 self.cfg['sensor']['lambda_max_nm'],
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
        if (self.Mu_ox is None) or (self.Mu_dox is None):
            df = pd.read_csv(self.paths['spectra']/ self.cfg['hemodynamics']['spectrum_path'])
            self.eps_ox = np.interp(self.L_arr, df['wavelength'], df['Hb02 (cm-1/M)'])
            self.eps_dox = np.interp(self.L_arr, df['wavelength'], df['Hb (cm-1/M)']) # cm-1 g-1
            self.MHg = 64500 # Hemoglobin grams per mole
            self.blood_concentration = 150 # grams per Liter

            df = pd.read_csv(self.paths['spectra']/ self.cfg['hemodynamics']['pathlength_path'])
            self.pathlength = np.interp(x=self.L_arr, xp=df['Wavelength (nm)'], fp=df['Estimated average pathlength (cm)']) #
            
            self.Mu_dox = self.blood_concentration * (self.pathlength * self.eps_dox) / self.MHg
            self.Mu_ox = self.blood_concentration * (self.pathlength * self.eps_ox) / self.MHg

        return self.Mu_ox, self.Mu_dox

    def gen_A_slow(self):
        A = []
        T_conv = int(0.2*self.T)
        p = self.cfg['sensor']['sampling_freq_Hz'] * 1/self.cfg['activity']['dominant_freq_Hz']
        p = p*2 # for the lorenz attractor, the period is roughly around both orbits.
        p = self.rng.integers(low=int(0.8*p), high=int(1.2*p))
        assert p > 1, 'Check number of points per period'
        for i in range(self.I % self.A_model.embedding_dimension + 1):
            self.A_model.ic = self.rng.integers(low=-100, high=100)*self.rng.random(3)
            A.append(self.A_model.make_trajectory(self.T + T_conv,
                                                pts_per_period=p,
                                                resample=True,
                                                standardize=True))

        A = np.hstack(A)
        ind = np.arange(A.shape[1])
        np.random.shuffle(ind) # inplace op.
        A = A[T_conv:,ind[:self.I]]
        assert A.shape == (self.T, self.I), 'check generated shape of A'
        return A

    def gen_A_fast(self):
        icfg = self.cfg['indicator']
        sampling_interval = 1/self.cfg['sensor']['sampling_freq_Hz']
        A_fast = np.zeros((self.T,self.I))
        for i,k in enumerate(icfg.keys()):
            # instantiate spikes
            x_unif = self.rng.random(self.T)
            p_spike = icfg[k]['modulator_spiking_f_Hz'] * sampling_interval
            A_fast[x_unif <= p_spike, i] = 1

            # convolve with exponential decay kernel
            t_window = np.arange(0, icfg[k]['exp_decay_const_s']*10, sampling_interval)
            kernel = 1 * np.exp(-(1/icfg[k]['exp_decay_const_s'])*t_window)
            A_fast[:,i] = signal.convolve(A_fast[:,i], kernel, mode='same')

        return A_fast

    def gen_H(self):
        # Should be always +ve
        hdyn = self.cfg['hemodynamics']
        cfg = self.cfg['amplitude']
        sampling_interval = 1/self.cfg['sensor']['sampling_freq_Hz']

        H_total = lowpass(xt=self.rng.standard_normal(size=(self.T,)),
                       sampling_interval=sampling_interval,
                       pass_below=hdyn['lowpass_thr_Hz'])
        
        # rescaling H_total
        H_total = H_total - np.min(H_total)
        H_total = H_total / np.max(H_total)
        H_total = cfg['H_total_range']*(H_total - np.mean(H_total)) + 1.0

        f = lowpass(xt=self.rng.standard_normal(size=(self.T,)),
                       sampling_interval=sampling_interval,
                       pass_below=hdyn['lowpass_thr_Hz'])

        # rescaling f
        f = f - np.min(f)
        f = f / np.max(f)
        f = cfg['f_range'] * (f - np.mean(f)) + 0.7

        H_ox = f*H_total
        H_dox = (1-f)*H_total

        assert H_ox.shape == (self.T,), 'check H_ox shape'
        assert H_dox.shape == (self.T,), 'check H_dox shape'
        return H_ox, H_dox, H_total, f

    def gen_N(self):
        # Simulating N - this can be estimated from diffraction pattern around saturated pixels. 
        # For now we can treat this as a known 'constant' (i.e. doesn't need to be fit) in the model
        # Simulated with amplitude of 2% compared to that for activity
        return self.rng.standard_normal(size=(self.T,self.J))

    def gen_M(self):
        return self.rng.standard_normal(size=(self.T,))

    def gen_B(self):
        return self.rng.random((self.J,self.L))

    def compose(self):
        amp = self.cfg['amplitude']

        # These are all constants:
        S = self.get_S()
        W = self.get_W()
        E = self.get_E()
        Mu_ox, Mu_dox = self.get_Mu()

        A = amp['A_slow'] * self.gen_A_slow() + amp['A_fast'] * self.gen_A_fast()
        AS = np.einsum('ti,il->til', A, S)
        ASW = np.einsum('til,ij->tjl', AS, W)
        E = np.einsum('td,djl -> tjl', np.ones((self.T,1)), E[np.newaxis,...])
        ASWE = ASW + E

        # 2nd term
        H_ox, H_dox, H_total, f  = self.gen_H()
        H_ox = amp['H_ox'] * H_ox
        H_dox = amp['H_dox'] * H_dox
        HD = np.einsum('td,dl -> tl', H_ox[...,np.newaxis], Mu_ox[np.newaxis,...]) \
            + np.einsum('td,dl -> tl', H_dox[...,np.newaxis], Mu_dox[np.newaxis,...])
        HD = np.einsum('j,tl -> tjl', np.ones((self.J,)), HD)
        HD = np.exp(-1 * HD)
        M = 1 + amp['M'] * self.gen_M()         # multiplicative; setting mean = 1
        B = amp['B'] * self.gen_B()
        HDM = np.einsum('tjl,t -> tjl', HD, M)
        B_ = np.einsum('t,jl -> tjl', np.ones((self.T,)), B)
        H = HDM + B_

        # 3rd term
        N = 1 + amp['N'] * self.gen_N()         # multiplicative; setting mean = 1
        N_ = np.einsum('l,tj -> tjl', np.ones((self.L,)), N)
        O = np.einsum('tjl,tjl -> tjl', ASWE, H)
        O = np.einsum('tjl,tjl -> tjl', O, N_)

        dat = dict(O=O,
                   E=E,
                   A=A,
                   N=N,
                   B=B,
                   M=M,
                   H_ox=H_ox,
                   H_dox=H_dox)
        return dat