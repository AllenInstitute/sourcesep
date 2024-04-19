import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
from scipy import signal
from scipy.ndimage import convolve
from scipy.stats import norm

from sourcesep.utils.compute import custom_sigmoid, lowpass, softplus
from sourcesep.utils.config import load_config


class SimData:
    def __init__(self, n_samples=None, cfg_path=None, rng_seed=None):
        """Class to generate samples for the simulation

        Args:
            n_samples (int): Number of samples in time
            cfg_path (Path or str): config for simulation parameters
        """
        self.cfg = toml.load(cfg_path)

        self.J = len(self.cfg["laser"])  # Number of excitation lasers (input channels)
        self.L = self.cfg["sensor"][
            "n_channels"
        ]  # Number of sensor pixels (output channels)
        self.I = len(self.cfg["indicator"])  # Number of indicators
        self.n_samples = n_samples  # Duration of signal in seconds

        self.set_arrays()  # set time and wavelength arrays

        self.E = None  # spectra
        self.W = None  # relative excitation efficiency of different lasers for each indicator
        self.S = None
        self.X = None
        self.S_autofl = None
        self.W_autofl = None
        self.Mu_HbO = None
        self.Mu_HbR = None
        self.notch = None

        self.paths = load_config(dataset_key="all")  # Paths for data files
        if rng_seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(rng_seed)

    def set_arrays(self):
        # time stamps
        self.T_arr = np.linspace(
            0,
            (self.n_samples - 1) / self.cfg["sensor"]["sampling_freq_Hz"],
            self.n_samples,
        )

        # wavelengths measured
        self.L_arr = np.linspace(
            self.cfg["sensor"]["lambda_min_nm"],
            self.cfg["sensor"]["lambda_max_nm"],
            self.cfg["sensor"]["n_channels"],
        )
        return

    def get_L_ind(self, lam):
        """Find the index of the closest wavelength in the L_arr"""
        return np.argmin(np.abs(self.L_arr - lam))

    def get_S_names(self):
        return np.array([k for k in self.cfg["indicator"].keys()])

    def get_S(self):
        """Populates self.S with the indicator spectra"""

        if self.S is None:
            icfg = self.cfg["indicator"]
            S = []
            for i, k in enumerate(icfg.keys()):
                if icfg[k]["from_path"]:
                    df = pd.read_csv(self.paths["spectra"] / icfg[k]["spectrum_path"])
                    df.rename(
                        columns=dict(
                            zip(df.columns, ["wavelength", "exc", "em", "2p"])
                        ),
                        inplace=True,
                    )
                    df["em"] = df["em"].fillna(0)
                    S.append(np.interp(x=self.L_arr, xp=df["wavelength"], fp=df["em"]))

            S = np.vstack(S)
            assert S.shape == (self.I, self.L), "check spectra shape"
            self.S = S
        return self.S

    def get_S_synthetic(self):
        """Synthetic excitation and emission specspectra for bound and free to test simulation

        Returns:
            S (dict): with keys 'bound_ex', 'free_ex', 'bound_em', 'free_em'
        """
        df = pd.read_csv(self.paths["spectra"] / "EGFP.csv")
        df.fillna(0, inplace=True)

        S = {}
        S["bound_ex"] = np.interp(x=self.L_arr, xp=df["wavelength"], fp=df["EGFP ex"])
        S["bound_ex"] = S["bound_ex"] / np.max(S["bound_ex"])
        S["free_ex"] = norm.pdf(self.L_arr, loc=400, scale=40)
        S["free_ex"] = S["free_ex"] / np.max(S["free_ex"]) * 0.3
        S["bound_em"] = np.interp(x=self.L_arr, xp=df["wavelength"], fp=df["EGFP em"])
        S["bound_em"] = S["bound_em"] / np.max(S["bound_em"])
        S["free_em"] = S["bound_em"]
        return S

    def get_X(self):
        """Populates self.Xex and self.Xem with pathlengths from Zhang et al. 2022"""
        if self.X is None:
            df = pd.read_csv(
                self.paths["spectra"] / self.cfg["sensor"]["pathlength_path"]
            )
            df.columns = ["wavelength", "Xex", "Xem"]
            self.X = np.interp(x=self.L_arr, xp=df["wavelength"], fp=df["em"])

        return self.X

    def get_notches(self):
        """Populates self.notches with the effective transmission coefficient over all
        notch filters

        """
        if self.notch is None:
            ncfg = self.cfg["notch"]
            self.notch = np.ones((self.L,))
            if len(ncfg.keys()) > 0:
                notch_dict = {}
                for key in ncfg.keys():
                    lam = ncfg[key]["block_freq_nm"]
                    # Skip the first 10 rows if they contain headers or metadata
                    df = pd.read_excel(
                        self.paths["spectra"] / f"Semrock_{lam}nm_notch.xlsx",
                        skiprows=13,
                    )
                    df.columns = ["wavelength", "trans"]
                    df_interp = pd.DataFrame({"wavelength": self.L_arr})
                    df_interp["trans"] = np.interp(
                        self.L_arr, df["wavelength"], df["trans"]
                    )
                    notch_dict[key] = df_interp

                for key in notch_dict:
                    self.notch = self.notch * notch_dict[key]["trans"].values

        return self.notch

    def get_W(self):
        """Populates self.W with the excitation efficiency"""

        if self.W is None:
            icfg = self.cfg["indicator"]
            lcfg = self.cfg["laser"]

            laser_freq = [lcfg[name]["em_wavelength_nm"] for name in lcfg.keys()]
            laser_freq = np.array(laser_freq)

            indicator = []
            W = np.zeros((self.I, self.J))
            for i, k in enumerate(icfg.keys()):
                if icfg[k]["from_path"]:
                    df = pd.read_csv(self.paths["spectra"] / icfg[k]["spectrum_path"])
                    df.rename(
                        columns=dict(
                            zip(df.columns, ["wavelength", "exc", "em", "2p"])
                        ),
                        inplace=True,
                    )
                    df["exc"] = df["exc"].fillna(0)
                    W[i, :] = np.interp(x=laser_freq, xp=df["wavelength"], fp=df["exc"])
                    indicator.append(k)
            W_df = pd.DataFrame(W, columns=laser_freq.astype(int), index=indicator)
            self.W = W
            self.W_df = W_df
        return self.W

    def get_E(self):
        if self.E is None:
            # laser spectrum is approximately a delta function
            # sigma is a small fixed value
            lcfg = self.cfg["laser"]
            sigma = 1
            E = []
            for k in lcfg.keys():
                mu = lcfg[k]["em_wavelength_nm"]
                y = np.exp(-((self.L_arr - mu) ** 2) / (sigma**2))
                y = y / np.max(y)
                E.append(y)
            E = np.vstack(E)
            assert E.shape == (self.J, self.L), "check laser spectrum (E) shape"
            self.E = E

        return self.E

    def get_Mu(self):
        if (self.Mu_HbO is None) or (self.Mu_HbR is None):
            df = pd.read_csv(
                self.paths["spectra"] / self.cfg["hemodynamics"]["spectrum_path"]
            )
            self.eps_HbO = np.interp(self.L_arr, df["wavelength"], df["Hb02 (cm-1/M)"])
            self.eps_HbR = np.interp(
                self.L_arr, df["wavelength"], df["Hb (cm-1/M)"]
            )  # cm-1 g-1
            self.MHg = 64500  # Hemoglobin grams per mole
            self.blood_concentration = 150  # grams per Liter
            self.blood_concentration_M = (
                self.blood_concentration / self.MHg
            )  # mol per Liter

            # df = pd.read_csv(self.paths['spectra']/ self.cfg['hemodynamics']['pathlength_path'])
            # self.pathlength = np.interp(x=self.L_arr, xp=df['Wavelength (nm)'], fp=df['Estimated average pathlength (cm)'])
            self.pathlength = self.cfg["hemodynamics"]["pathlength"]

            self.Mu_HbO = self.blood_concentration_M * (self.pathlength * self.eps_HbO)
            self.Mu_HbR = self.blood_concentration_M * (self.pathlength * self.eps_HbR)
        return self.Mu_HbO, self.Mu_HbR

    def get_S_autofl(self):
        if self.S_autofl is None:
            acfg = self.cfg["autofl"]
            n_autofl = len(acfg.keys())
            S_autofl = []
            for i, k in enumerate(acfg.keys()):
                df = pd.read_csv(self.paths["spectra"] / acfg[k]["spectrum_path"])
                df.rename(
                    columns=dict(zip(df.columns, ["wavelength", "exc", "em", "2p"])),
                    inplace=True,
                )
                df["em"] = df["em"].fillna(0)
                df["em"] = df["em"] / 100  # autofluoresence spectra are in %
                S_autofl.append(
                    np.interp(x=self.L_arr, xp=df["wavelength"], fp=df["em"])
                )

            S_autofl = np.vstack(S_autofl)
            assert S_autofl.shape == (n_autofl, self.L), "check spectra shape"
            self.S_autofl = S_autofl

        return self.S_autofl

    def get_W_autofl(self):
        """Populates self.W_autofl with the excitation efficiency"""

        if self.W_autofl is None:
            acfg = self.cfg["autofl"]
            lcfg = self.cfg["laser"]

            laser_freq = [lcfg[name]["em_wavelength_nm"] for name in lcfg.keys()]
            laser_freq = np.array(laser_freq)
            n_autofl = len(acfg.keys())

            aufl_src = []
            W = np.zeros((n_autofl, self.J))
            for i, k in enumerate(acfg.keys()):
                df = pd.read_csv(self.paths["spectra"] / acfg[k]["spectrum_path"])
                df.rename(
                    columns=dict(zip(df.columns, ["wavelength", "exc", "em", "2p"])),
                    inplace=True,
                )
                df["exc"] = df["exc"].fillna(0)
                df["exc"] = df["exc"] / 100  # autofluoresence spectra are in %
                W[i, :] = np.interp(x=laser_freq, xp=df["wavelength"], fp=df["exc"])
                aufl_src.append(k)
            W_df = pd.DataFrame(W, columns=laser_freq.astype(int), index=aufl_src)
            self.W_autofl = W
            self.W_autofl_df = W_df
        return self.W_autofl

    @staticmethod
    def _dyn_slow(
        n_samples, sampling_interval, lowpass_thr_Hz, bottom, top, beta, rng=None
    ):
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

        x = lowpass(
            xt=rng.standard_normal(size=(n_samples,)),
            sampling_interval=sampling_interval,
            pass_below=lowpass_thr_Hz,
            axis=0,
        )

        x = x - np.mean(x)
        x = x / np.std(x)

        x = custom_sigmoid(x, bottom=bottom, top=top, beta=beta)
        return x

    def gen_f_bound_slow(self):
        """Generates slow component of the activity signal."""
        sampling_interval = 1 / self.cfg["sensor"]["sampling_freq_Hz"]

        icfg = self.cfg["indicator"]
        sampling_freq_Hz = self.cfg["sensor"]["sampling_freq_Hz"]
        sampling_interval = 1 / sampling_freq_Hz

        A_slow = np.zeros((self.n_samples, self.I))
        for i, key in enumerate(icfg.keys()):
            lowpass_thr_Hz = icfg[key]["modulator_lowpass_f_Hz"]
            min_amplitude = 0.0
            max_amplitude = icfg[key]["modulator_slow_amplitude"]
            assert (
                icfg[key]["modulator_slow_amplitude"]
                + icfg[key]["modulator_fast_amplitude"]
                <= 1.0
            ), f"slow + fast amplitude for {key} should be <= 1.0"

            A_slow[:, i] = self._dyn_slow(
                n_samples=self.n_samples,
                sampling_interval=sampling_interval,
                lowpass_thr_Hz=lowpass_thr_Hz,
                bottom=min_amplitude,
                top=max_amplitude,
                beta=1.0,
                rng=self.rng,
            )

        assert A_slow.shape == (
            self.n_samples,
            self.I,
        ), "check generated shape of A_slow"
        return A_slow

    @staticmethod
    def _dyn_fast(
        n_samples,
        spiking_freq_Hz,
        sampling_interval,
        decay_const_s,
        spike_min=0.64,
        spike_max=0.8,
        rng=None,
    ):
        if rng is None:
            rng = np.random.default_rng()
        A_fast = np.zeros((n_samples,))
        x_unif = rng.random(n_samples)
        p_spike = spiking_freq_Hz * sampling_interval
        # replace ones with randomly sampled values uniformly between 0.64-0.8
        A_fast = np.where(
            x_unif <= p_spike,
            rng.uniform(spike_min, spike_max, size=(n_samples,)),
            A_fast,
        )

        # convolve with exponential decay kernel
        t_window = np.arange(0, decay_const_s * 10, sampling_interval)
        kernel = 1 * np.exp(-(1 / decay_const_s) * t_window)
        A_fast = signal.convolve(A_fast, kernel, mode="same")
        return A_fast

    def gen_f_bound_fast(self):
        """Generates fast component of the activity signal"""

        icfg = self.cfg["indicator"]
        sampling_freq_Hz = self.cfg["sensor"]["sampling_freq_Hz"]
        sampling_interval = 1 / sampling_freq_Hz

        A_fast = np.zeros((self.n_samples, self.I))
        for i, key in enumerate(icfg.keys()):
            # instantiate spikes
            spiking_freq_Hz = icfg[key]["modulator_spiking_f_Hz"]
            spike_max = icfg[key]["modulator_fast_amplitude"]
            decay_const_s = icfg[key]["exp_decay_const_s"]
            A_fast[:, i] = self._dyn_fast(
                n_samples=self.n_samples,
                spiking_freq_Hz=spiking_freq_Hz,
                sampling_interval=sampling_interval,
                decay_const_s=decay_const_s,
                spike_min=0.8 * spike_max,
                spike_max=spike_max,
                rng=self.rng,
            )
        return A_fast

    def fluorescing_population(self):
        """Fluorescing population at each time decays due to bleaching.

        Args:
        """
        Ct = np.zeros((self.I, self.n_samples))
        icfg = self.cfg["indicator"]

        for i, k in enumerate(icfg.keys()):
            bcfg = icfg[k]["bleaching"]
            if bcfg["bleach"]:
                Ct[i, :] = (
                    bcfg["C0_slow"] * np.exp(-self.T_arr / bcfg["tau_slow_s"])
                    + bcfg["C0_fast"] * np.exp(-self.T_arr / bcfg["tau_fast_s"])
                    + bcfg["C0_const"]
                )
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
        hdyn = self.cfg["hemodynamics"]
        sampling_interval = 1 / self.cfg["sensor"]["sampling_freq_Hz"]

        HbT = self._dyn_slow(
            n_samples=self.n_samples,
            sampling_interval=sampling_interval,
            lowpass_thr_Hz=0.2,
            bottom=hdyn["HbT_min"],
            top=hdyn["HbT_max"],
            beta=1.0,
            rng=self.rng,
        )

        f_HbO = self._dyn_slow(
            n_samples=self.n_samples,
            sampling_interval=sampling_interval,
            lowpass_thr_Hz=hdyn["lowpass_thr_Hz"],
            bottom=hdyn["f_HbO_min"],
            top=hdyn["f_HbO_max"],
            beta=1.0,
            rng=self.rng,
        )

        HbO = f_HbO * HbT
        HbR = (1 - f_HbO) * HbT

        assert HbO.shape == (self.n_samples,), "check HbO shape"
        assert HbR.shape == (self.n_samples,), "check HbR shape"
        return HbO, HbR, HbT, f_HbO

    def gen_autofl(self):
        """Simulate autofluorescence components"""

        acfg = self.cfg["autofl"]
        sampling_interval = 1 / self.cfg["sensor"]["sampling_freq_Hz"]
        f_autofl = np.zeros((self.n_samples, len(acfg.keys())))
        for i, k in enumerate(acfg.keys()):
            f_autofl[:, i] = self._dyn_slow(
                n_samples=self.n_samples,
                sampling_interval=sampling_interval,
                lowpass_thr_Hz=acfg[k]["lowpass_thr_Hz"],
                bottom=acfg[k]["autofl_min"],
                top=acfg[k]["autofl_max"],
                beta=1.0,
                rng=self.rng,
            )

        assert f_autofl.shape == (
            self.n_samples,
            len(acfg.keys()),
        ), "check f_autofl shape"
        return f_autofl

    def gen_P(self):
        """Simulate laser power as i.i.d Gaussian distributed samples

        Returns:
            P (np.array): with shape (n_samples, J)"""

        P = np.ones((self.n_samples, self.J))
        lcfg = self.cfg["laser"]
        for j, key in enumerate(lcfg.keys()):
            mean_ = lcfg[key]["power_mean"]
            var_ = lcfg[key]["power_var"]
            sd_ = var_**0.5
            P[:, j] = self.rng.normal(loc=mean_, scale=sd_, size=(self.n_samples,))
        return P

    def gen_M(self):
        """Simulate motion artifacts as i.i.d Gaussian distributed samples

        Returns:
            M (np.array): with shape (n_samples, J)"""
        return self.rng.standard_normal(size=(self.n_samples,))

    def gen_B(self):
        return self.rng.random((self.J, self.L))

    @staticmethod
    def arr_lookup(arr, x):
        return np.argmin(np.abs(arr - x))

    def to_disk(self, filepath=None):
        """
        Generate and save data to disk.

        Args:
            filepath (str or Path): Full path to save the h5 file.
        """

        import os

        import h5py

        dat = self.compose()
        dat["L_arr"] = self.L_arr
        dat["T_arr"] = self.T_arr
        assert filepath is not None, "filepath is None"
        max_chunk_size = 1000
        if os.path.exists(filepath):
            print(f"Removing {filepath}")
            os.remove(filepath)

        with h5py.File(filepath, "a") as f:
            for key in dat.keys():
                print(f"Creating {key}")
                chunk_size = min(max_chunk_size, dat[key].shape[0])
                f.create_dataset(
                    key,
                    data=dat[key],
                    shape=dat[key].shape,
                    maxshape=dat[key].shape,
                    chunks=(chunk_size, *dat[key].shape[1:]),
                    dtype="float",
                )
        print(f"Saved to {filepath}")

        # save the config with toml
        with open(filepath.replace(".h5", ".toml"), "w") as f:
            f.write(toml.dumps(self.cfg))

        print(f"Saved config to {filepath.replace('.h5', '.toml')}")
        return

    def compose(self):
        S = self.get_S_synthetic()
        lam_ = [
            self.cfg["laser"][key]["em_wavelength_nm"]
            for key in self.cfg["laser"].keys()
        ]

        # W: excitation efficiency
        Wb = np.zeros((1, self.J))
        Wf = np.zeros((1, self.J))
        for j in range(self.J):
            Wb[0, j] = S["bound_ex"][self.get_L_ind([lam_[j]])]
            Wf[0, j] = S["free_ex"][self.get_L_ind([lam_[j]])]

        # S: emission spectra
        Wb_full = S["bound_ex"].reshape(1, -1)
        Wf_full = S["free_ex"].reshape(1, -1)
        Sb = S["bound_em"].reshape(1, -1)
        Sf = S["free_em"].reshape(1, -1)
        P = self.gen_P()

        # H: hemodynamics
        fudge = 1.0
        C_HbO, C_HbR, C_HbT, f_HbO = self.gen_H()
        Mu_HbO, Mu_HbR = self.get_Mu()
        H = np.einsum("t,l->tl", C_HbO, Mu_HbO) + np.einsum("t,l->tl", C_HbR, Mu_HbR)
        H = np.exp(-fudge * H)

        # Indicator activity free, bound
        f_rest = 0.7
        f_dynamic_range = 0.2 * f_rest
        fb = self.gen_f_bound_slow() + self.gen_f_bound_fast()
        fb = 1.0 - softplus(1.0 - fb, beta=40, thr=1.0)
        fb = f_rest + fb * f_dynamic_range
        ff = 1.0 - fb

        # Compile indicator signal component
        fSW_b = np.einsum("ti,il,ij->tjl", fb, Sb, Wb)
        fSWP_b = np.einsum("tjl,tj->tjl", fSW_b, P)
        fSW_f = np.einsum("ti,il,ij->tjl", ff, Sf, Wf)
        fSWP_f = np.einsum("tjl,tj->tjl", fSW_f, P)
        fSWP = fSWP_b + fSWP_f
        fSWPH = np.einsum("tjl,tl->tjl", fSWP.copy(), H)

        # Compile autofluorescence signal component
        S_autofl = self.get_S_autofl()
        W_autofl = self.get_W_autofl()
        f_autofl = self.gen_autofl()
        fSW_autofl = np.einsum("ti,il,ij->tjl", f_autofl, S_autofl, W_autofl)
        fSWP_autofl = np.einsum("tjl,tj->tjl", fSW_autofl.copy(), P)
        fSWPH_autofl = np.einsum("tjl,tl->tjl", fSWP_autofl.copy(), H)

        # Compile total signal
        Obs = fSWPH + fSWPH_autofl
        notches = self.get_notches()
        Obs_notches = np.einsum("tjl,l->tjl", Obs.copy(), notches)

        # true f_rest <-- only used for checks
        f0SW_b = np.einsum("ti,il,ij->tjl", np.ones_like(fb) * f_rest, Sb, Wb)
        f0SWP_b = np.einsum("tjl,tj->tjl", f0SW_b, P)
        f0SW_f = np.einsum("ti,il,ij->tjl", (1.0 - np.ones_like(fb) * f_rest), Sf, Wf)
        f0SWP_f = np.einsum("tjl,tj->tjl", f0SW_f, P)
        f0SWP = f0SWP_b + f0SWP_f
        f0SWPH = np.einsum("tjl,tl->tjl", f0SWP, H)

        # image formation; K is the kernel
        K = np.ones_like(
            np.arange(0, self.cfg["image"]["window_nm"], np.mean(np.diff(self.L_arr)))
        )
        K = K / np.sum(K)
        K = K.reshape(1, 1, -1)
        OK = convolve(Obs_notches, K, mode="reflect")

        dat = dict(
            O=Obs,
            OK=OK,
            fb=fb,
            ff=ff,
            f_autofl=f_autofl,
            f_rest=f_rest,
            f_dynamic_range=f_dynamic_range,
            Wb=Wb,
            Wf=Wf,
            Wb_full=Wb_full,
            Wf_full=Wf_full,
            Sb=Sb,
            Sf=Sf,
            S_autofl=S_autofl,
            P=P,
            H=H,
            fSWP=fSWP,
            fSWP_autofl=fSWP_autofl,
            fSWPH=fSWPH,
            fSWPH_autofl=fSWPH_autofl,
            f0SWPH=f0SWPH,
            C_HbO=C_HbO,
            C_HbR=C_HbR,
            C_HbT=C_HbT,
            f_HbO=f_HbO,
            Mu_HbO=Mu_HbO,
            Mu_HbR=Mu_HbR,
            notches=notches,
            T_arr=self.T_arr,
            L_arr=self.L_arr,
        )
        return dat

    @staticmethod
    def plot2d(X, J, L_range, L_arr, T_range, T_arr, ax=None):
        """
        Args:
            X (np.array): shape (t,j,l)
            J (int): index of laser
            L_range: (np.array): shape (2,) in nm
            L_arr (np.array): shape (l,) in nm
            T_range (np.array): shape (2,) in seconds
            T_arr (np.array): shape (t,) in seconds
        """
        L_arr = L_arr.reshape(
            -1,
        )
        T_arr = T_arr.reshape(
            -1,
        )
        l_ind = [np.argmin(np.abs(L_arr - l)) for l in L_range]
        t_ind = [np.argmin(np.abs(T_arr - t)) for t in T_range]

        def fmt_y_(x, _):
            if x < L_arr.size:
                return f"{L_arr[int(x)]:0.1f} nm"
            else:
                return "NA"

        fmt_y = mpl.ticker.FuncFormatter(fmt_y_)

        def fmt_x_(x, _):
            if x < T_arr.size:
                return f"{T_arr[int(x)]:0.1f} s"
            else:
                return "NA"

        fmt_x = mpl.ticker.FuncFormatter(fmt_x_)

        if ax == None:
            f, ax = plt.subplots(1, 1, figsize=(8, 3))
        img = ax.imshow(
            X[:, J, :].T,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=0.4,
            interpolation="nearest",
        )
        ax.xaxis.set(major_formatter=fmt_x)
        ax.yaxis.set(major_formatter=fmt_y)
        ax.set(
            xlim=(t_ind),
            ylim=(l_ind),
            xlabel="time",
            ylabel="wavelength",
            title=f"exc. laser {J}",
        )
        return f, ax, img

    @staticmethod
    def plot1d_time(X, J, L_, L_arr, T_arr, ax=None):
        L_arr = L_arr.reshape(
            -1,
        )
        T_arr = T_arr.reshape(
            -1,
        )
        if isinstance(L_, list):
            l_ind = [np.argmin(np.abs(L_arr - L)) for L in L_]
        else:
            l_ind = [np.argmin(np.abs(L_arr - L_))]
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(8, 3))
        for l in l_ind:
            ax.plot(T_arr, X[:, J, l], label=f"{L_arr[l]:0.1f} nm")
        ax.set(xlabel="time", ylabel="intensity", title=f"exc. laser {J}")
        ax.legend()
        return f, ax


if __name__ == "__main__":
    from sourcesep.sim import SimData
    from sourcesep.utils.config import load_config

    paths = load_config(dataset_key="all")
    sim = SimData(n_samples=500, cfg_path=paths["root"] / "sim_config.toml")
    dat = sim.compose()
