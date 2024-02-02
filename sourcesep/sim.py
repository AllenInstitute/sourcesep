import numpy as np
import pandas as pd
import toml
from scipy import signal
from scipy.ndimage import convolve, uniform_filter1d
from dysts.flows import Lorenz
from sourcesep.utils.config import load_config
from sourcesep.utils.compute import lowpass, softplus, custom_sigmoid


class SimData():
    def __init__(self, n_samples=None, cfg_path=None):
        """Class to generate samples for the simulation

        Args:
            n_samples (int): Number of samples in time
            cfg_path (Path or str): config for simulation parameters
        """
        self.cfg = toml.load(cfg_path)
        
        self.J = len(self.cfg['laser'])            # Number of excitation lasers (input channels)
        self.L = self.cfg['sensor']['n_channels']  # Number of sensor pixels (output channels)
        self.I = len(self.cfg['indicator'])        # Number of indicators
        self.n_samples = n_samples                 # Duration of signal in seconds

        self.set_arrays()                          # set time and wavelength arrays

        self.E = None   # spectra
        self.W = None   # relative excitation efficiency of different lasers for each indicator
        self.S = None   
        self.X = None
        self.notches = None
        self.Mu_HbO = None
        self.Mu_HbR = None
        self.notch = None

        self.paths = load_config(dataset_key='all')# Paths for data files
        self.rng = np.random.default_rng()


    def set_arrays(self):
        # time stamps
        self.T_arr = np.linspace(0,
                                 (self.n_samples-1) / self.cfg['sensor']['sampling_freq_Hz'],
                                 self.n_samples)

        # wavelengths measured
        self.L_arr = np.linspace(self.cfg['sensor']['lambda_min_nm'],
                                 self.cfg['sensor']['lambda_max_nm'],
                                 self.cfg['sensor']['n_channels'])
        return

    def get_S(self):
        """Populates self.S with the indicator spectra
        """

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

            S = np.vstack(S)
            assert S.shape == (self.I, self.L), 'check spectra shape'
            self.S = S

        return self.S
    
    def get_X(self):
        """Populates self.Xex and self.Xem with pathlengths from Zhang et al. 2022
        """
        if self.X is None:
            df = pd.read_csv(self.paths['spectra']/self.cfg['sensor']['pathlength_path'])
            df.columns = ['wavelength', 'Xex', 'Xem']
            self.X = np.interp(x=self.L_arr, xp=df['wavelength'], fp=df['em'])

        return self.X
    
    
    def get_notches(self):
        """Populates self.notches with the effective transmission coefficient over all 
        notch filters

        """
        
        if self.notch is None:
            ncfg = self.cfg['notch']
            self.notch = np.ones((self.L,))
            if (len(ncfg.keys()) > 0):
                notch_dict = {}
                for key in ncfg.keys():
                    lam = ncfg[key]['block_freq_nm']
                    df = pd.read_excel(self.paths['spectra'] / f'Semrock_{lam}nm_notch.xlsx', skiprows=13)  # Skip the first 10 rows if they contain headers or metadata
                    df.columns = ['wavelength', 'trans']
                    df_interp = pd.DataFrame({'wavelength': self.L_arr})
                    df_interp['trans'] = np.interp(self.L_arr, df['wavelength'], df['trans'])
                    notch_dict[key] = df_interp

                for key in notch_dict:
                    self.notch = self.notch * notch_dict[key]['trans'].values

        return self.notch

    def get_W(self):
        """Populates self.W with the excitation efficiency
        """

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
            W_df = pd.DataFrame(W, columns=laser_freq.astype(int), index=indicator)
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
        if (self.Mu_HbO is None) or (self.Mu_HbR is None):
            df = pd.read_csv(self.paths['spectra']/ self.cfg['hemodynamics']['spectrum_path'])
            self.eps_ox = np.interp(self.L_arr, df['wavelength'], df['Hb02 (cm-1/M)'])
            self.eps_dox = np.interp(self.L_arr, df['wavelength'], df['Hb (cm-1/M)']) # cm-1 g-1
            self.MHg = 64500 # Hemoglobin grams per mole
            self.blood_concentration = 150 # grams per Liter

            #df = pd.read_csv(self.paths['spectra']/ self.cfg['hemodynamics']['pathlength_path'])
            #self.pathlength = np.interp(x=self.L_arr, xp=df['Wavelength (nm)'], fp=df['Estimated average pathlength (cm)'])
            self.pathlength = self.cfg['hemodynamics']['pathlength']

            self.Mu_HbR = self.blood_concentration * (self.pathlength * self.eps_dox) / self.MHg
            self.Mu_HbO = self.blood_concentration * (self.pathlength * self.eps_ox) / self.MHg

        return self.Mu_HbO, self.Mu_HbR

    @staticmethod
    def _dyn_slow(n_samples, sampling_interval, lowpass_thr_Hz, 
                  bottom, top, beta, rng=None):
        """_summary_

        Args:
            n_samples (int): _description_
            sampling_interval (float) 
            lowpass_thr_Hz (float)
            bottom (float): roughly the min value of the output
            top (float): roughly the max value of the output
            beta (float): controls how saturated the output is
            rng: Defaults to None.

        Returns:
            np.array: slowly changing signal
        """
        
        if rng is None:
            rng = np.random.default_rng()

        x = lowpass(xt=rng.standard_normal(size=(n_samples,)),
                     sampling_interval=sampling_interval,
                     pass_below=lowpass_thr_Hz,
                     axis=0)

        x = x - np.mean(x)
        x = x / np.std(x)

        x = custom_sigmoid(x, bottom=bottom, top=top, beta=beta)
        return x

    def gen_f_bound_slow(self):
        """Generates slow component of the activity signal. 
        """
        sampling_interval = 1/self.cfg['sensor']['sampling_freq_Hz']

        icfg = self.cfg['indicator']
        sampling_freq_Hz = self.cfg['sensor']['sampling_freq_Hz']
        sampling_interval = 1/sampling_freq_Hz

        A_slow = np.zeros((self.n_samples, self.I))
        for i, key in enumerate(icfg.keys()):
            lowpass_thr_Hz = icfg[key]['modulator_lowpass_f_Hz']
            min_amplitude = 0.0
            max_amplitude = icfg[key]['modulator_slow_amplitude']
            assert icfg[key]['modulator_slow_amplitude'] + \
                icfg[key]['modulator_fast_amplitude'] <= 1.0, \
                f'slow + fast amplitude for {key} should be <= 1.0'

            A_slow[:, i] = self._dyn_slow(n_samples=self.n_samples,
                                          sampling_interval=sampling_interval,
                                          lowpass_thr_Hz=lowpass_thr_Hz,
                                          bottom=min_amplitude,
                                          top=max_amplitude,
                                          beta=1.0,
                                          rng=self.rng)

        assert A_slow.shape == (self.n_samples, self.I), \
            'check generated shape of A_slow'
        return A_slow

    @staticmethod
    def _dyn_fast(n_samples,
                  spiking_freq_Hz,
                  sampling_interval,
                  decay_const_s,
                  spike_min=0.64,
                  spike_max=0.8,
                  rng=None):
        if rng is None:
            rng = np.random.default_rng()
        A_fast = np.zeros((n_samples,))
        x_unif = rng.random(n_samples)
        p_spike = spiking_freq_Hz * sampling_interval
        # replace ones with randomly sampled values uniformly between 0.64-0.8
        A_fast = np.where(x_unif <= p_spike, rng.uniform(
            spike_min, spike_max, size=(n_samples,)), A_fast)

        # convolve with exponential decay kernel
        t_window = np.arange(0, decay_const_s*10, sampling_interval)
        kernel = 1 * np.exp(-(1/decay_const_s)*t_window)
        A_fast = signal.convolve(A_fast, kernel, mode='same')
        return A_fast

    def gen_f_bound_fast(self):
        """Generates fast component of the activity signal
        """

        icfg = self.cfg['indicator']
        sampling_freq_Hz = self.cfg['sensor']['sampling_freq_Hz']
        sampling_interval = 1/sampling_freq_Hz

        A_fast = np.zeros((self.n_samples,self.I))
        for i,key in enumerate(icfg.keys()):
            # instantiate spikes
            spiking_freq_Hz = icfg[key]['modulator_spiking_f_Hz']
            spike_max = icfg[key]['modulator_fast_amplitude']
            decay_const_s = icfg[key]['exp_decay_const_s']
            A_fast[:,i] = self._dyn_fast(n_samples=self.n_samples,
                    spiking_freq_Hz=spiking_freq_Hz,
                    sampling_interval=sampling_interval,
                    decay_const_s=decay_const_s,
                    spike_min = 0.8*spike_max,
                    spike_max = spike_max,
                    rng=self.rng)
        return A_fast
    
    def fluorescing_population(self):
        """Fluorescing population at each time decays due to bleaching. 
        
        Args:
        """
        Ct = np.zeros((self.I,self.n_samples))
        icfg = self.cfg['indicator']
        
        for i,k in enumerate(icfg.keys()):
            bcfg = icfg[k]['bleaching']    
            if bcfg['bleach']:
                Ct[i,:] = (bcfg['C0_slow']*np.exp(-self.T_arr/bcfg['tau_slow_s']) \
                    + bcfg['C0_fast']*np.exp(-self.T_arr/bcfg['tau_fast_s']) \
                    + bcfg['C0_const'])
        return Ct

    def gen_H(self):
        """Generate hemodynamic activity signal.
        
        Args: 

        Returns:
            HbO (np.array): with shape (n_samples,)
            HbR (np.array): with shape (n_samples,)
            HbT (np.array): with shape (n_samples,)
            f_HbO (np.array): with shape (n_samples,)
        """
        hdyn = self.cfg['hemodynamics']
        cfg = self.cfg['amplitude']
        sampling_interval = 1/self.cfg['sensor']['sampling_freq_Hz']

        HbT = self._dyn_slow(n_samples=self.n_samples,
                             sampling_interval=sampling_interval,
                             lowpass_thr_Hz=0.2,
                             bottom=hdyn['HbT_min'],
                             top=hdyn['HbT_max'],
                             beta=1.0,
                             rng=self.rng)

        f_HbO = self._dyn_slow(n_samples=self.n_samples,
                               sampling_interval=sampling_interval,
                               lowpass_thr_Hz=hdyn['lowpass_thr_Hz'],
                               bottom=hdyn['HbT_min'],
                               top=hdyn['HbT_max'],
                               beta=1.0,
                               rng=self.rng)

        HbO = f_HbO*HbT
        HbR = (1-f_HbO)*HbT

        assert HbO.shape == (self.n_samples,), 'check HbO shape'
        assert HbR.shape == (self.n_samples,), 'check HbR shape'
        return HbO, HbR, HbT, f_HbO

    def gen_P(self, mean, var):
        """Simulate laser power as i.i.d Gaussian distributed samples
        
        Returns:
            P (np.array): with shape (n_samples, J)"""
        return self.rng.normal(loc=mean, scale=var**0.5, size=(self.n_samples,self.J))

    def gen_M(self): 
        """Simulate motion artifacts as i.i.d Gaussian distributed samples
        
        Returns:
            M (np.array): with shape (n_samples, J)"""
        return self.rng.standard_normal(size=(self.n_samples,))

    def gen_B(self):
        return self.rng.random((self.J,self.L))
    
    @staticmethod
    def arr_lookup(arr, x):
        return np.argmin(np.abs(arr -x))

    def to_disk(self, filepath=None):
        """
        Generate and save data to disk.

        Args:
            filepath (str or Path): Full path to save the h5 file.
        """
        
        import os
        import h5py 

        dat = self.compose()
        dat['L_arr'] = self.L_arr
        dat['T_arr'] = self.T_arr
        assert filepath is not None, 'filepath is None'
        max_chunk_size = 1000
        if os.path.exists(filepath):
                print(f"Removing {filepath}")
                os.remove(filepath)

        with h5py.File(filepath, "a") as f:
            for key in dat.keys():
                print(f"Creating {key}")
                chunk_size = min(max_chunk_size, dat[key].shape[0])
                f.create_dataset(key, data=dat[key], 
                                shape=dat[key].shape, 
                                maxshape=dat[key].shape, 
                                chunks=(chunk_size, *dat[key].shape[1:]), 
                                dtype='float')
        print(f"Saved to {filepath}")

        # save the config with toml
        with open(filepath.replace('.h5', '.toml'), 'w') as f:
            f.write(toml.dumps(self.cfg))

        print(f"Saved config to {filepath.replace('.h5', '.toml')}")
        return

    def compose(self):
        amp = self.cfg['amplitude']

        # Read in known constants:
        S = self.get_S()
        W = self.get_W()
        E = self.get_E()
        Mu_HbO, Mu_HbR = self.get_Mu()
        notches = self.get_notches()
        
        # simulate bound fraction of indicator
        f_bound = self.gen_f_bound_slow() + self.gen_f_bound_fast()
        f_bound = 1.0 - softplus(1.0 - f_bound, beta=40, thr=1.0) # ensures f_bound in [0,1]

        # simlulate laser power
        P = self.gen_P(mean = amp['P_mean'], var = amp['P_var'])

        # multiply by known spectra, indicator concentration, and laser power
        fSW = np.einsum('ti,il,ij->tjl', f_bound, S, W)
        fSWP = np.einsum('tjl,tj->tjl', fSW, P)

        # hemodynamics absorption
        HbO, HbR, HbT, f_HbO = self.gen_H()
        H = np.einsum('td,dl -> tl', HbO[..., np.newaxis], Mu_HbO[np.newaxis, ...]) \
            + np.einsum('td,dl -> tl', HbR[..., np.newaxis], Mu_HbR[np.newaxis, ...])
        H = np.einsum('j,tl -> tjl', np.ones((self.J,)), np.exp(-H))

        # M = amp['M_mean'] + (amp['M_var'])**0.5 * self.gen_M()  # multiplicative, with specified mean and variance
        # M = clip(M, threshold=0.1)                              # fix to ensure M is positive
        
        B = amp['B'] * self.gen_B()
        HDM = np.einsum('tjl,t -> tjl', HD, M)
        B_ = np.einsum('t,jl -> tjl', np.ones((self.n_samples,)), B)
        H = HDM + B_

        # 3rd term
        N = amp['N_mean'] + (amp['N_var'])**0.5 * self.gen_N()   # multiplicative, with specified mean and variance
        N = clip(N, threshold=0.1)                               # fix to ensure N is positive
        N_ = np.einsum('l,tj -> tjl', np.ones((self.L,)), N)
        O = np.einsum('tjl,tjl -> tjl', fSWP, H)
        O = np.einsum('tjl,tjl -> tjl', O, N_)

        O = np.einsum('tjl,l -> tjl', O, notches)                # notch filter attenuates each wavelength

        # image formation; K is the kernel
        K = np.ones_like(np.arange(0,self.cfg['image']['window_nm'], np.mean(np.diff(self.L_arr))))
        K = K / np.sum(K)
        K = K.reshape(1,1,-1)
        O = convolve(O, K, mode='reflect')

        dat = dict(O=O,
                   f_bound=f_bound,
                   N=N,
                   B=B,
                   HbO=HbO,
                   HbR=HbR,
                   S=S,
                   W=W,
                   E=E,
                   Mu_HbO=Mu_HbO,
                   Mu_HbR=Mu_HbR)
        return dat

    def make_signal():
        # f_b(t)S_b(λ)w_b(j)C(t)P(j,t)+(1−fb(t))S f(λ)wf(j)C(t)P(j,t)
        pass
        return


def clip(x, threshold=0.0):
    """Clip values below threshold to threshold
    """
    x[x < threshold] = threshold
    return x


if __name__ == '__main__':
    from sourcesep.sim import SimData
    from sourcesep.utils.config import load_config
    paths = load_config(dataset_key='all')
    sim = SimData(T=500, cfg_path=paths['root'] / "sim_config.toml")
    sim.get_W()
    #sim.compose_torch()
