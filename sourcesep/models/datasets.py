import numpy as np
import matplotlib.pyplot as plt
from rich import print as pprint
import xarray as xr
from sourcesep.sim import SimData
from sourcesep.utils.config import load_config


def test_data_v1(verbose=True, plots=True):
    paths = load_config(dataset_key="all")

    # make dataset with custom parameters:
    sim = SimData(
        n_samples=6000, cfg_path=paths["root"] / "sim_config.toml", rng_seed=42
    )

    # change simulation parameters from default values to custom ones.
    h_mean = 0.0
    h_limit = 0.0
    sim.cfg["hemodynamics"]["HbT_min"] = h_mean - h_limit
    sim.cfg["hemodynamics"]["HbT_max"] = h_mean + h_limit

    for k in sim.cfg["laser"].keys():
        sim.cfg["laser"][k]["power_var"] = 0.005**2

    dat = sim.compose()
    S = sim.get_S_synthetic()
    S_autofl = sim.get_S_autofl()

    # create array with physical units for the excitation wavelength
    lam_ = [
        sim.cfg["laser"][key]["em_wavelength_nm"] for key in sim.cfg["laser"].keys()
    ]

    # create a dataset with xarray
    xdat = xr.DataArray(
        dat["fSWP"] + dat["fSWP_autofl"],
        dims=("time", "laser", "wavelength"),
        coords={"time": dat["T_arr"], "laser": lam_, "wavelength": dat["L_arr"]},
    )

    if plots:
        # fmt: off
        f, ax = plt.subplots(figsize=(6, 3))
        ax.plot(dat["L_arr"], dat["Wb_full"].reshape(-1, ), "--g", label="ex. bound")
        ax.plot(dat["L_arr"], dat["Wf_full"].reshape(-1, ), "--b", label="ex. free")
        ax.plot(dat["L_arr"], dat["Sf"].reshape(-1,),"g", label="em. bound")
        ax.plot(dat["L_arr"], 0.9 * dat["Sb"].reshape(-1, ), "b", label="em. free")
        ax.plot(dat["L_arr"], S_autofl[0, :], "r", label="em. FAD")
        ax.plot(dat["L_arr"], S_autofl[1, :], "orange", label="em. NADH")
        ax.set(xlabel="Wavelength (nm)", ylabel="AU")
        plt.legend()
        plt.show()
        # fmt: on

    W_df = sim.W_autofl_df
    W_df.loc["Wb"] = dat["Wb"].reshape(
        -1,
    )
    W_df.loc["Wf"] = dat["Wf"].reshape(
        -1,
    )

    if verbose:
        pprint(W_df)

    if plots:
        f, ax = plt.subplots(4, 1, figsize=(8, 6))
        xdat.sel(wavelength=510, laser=405, method="nearest").plot(
            label="free dominant", ax=ax[0]
        )
        xdat.sel(wavelength=510, laser=473, method="nearest").plot(
            label="bound dominant", ax=ax[1]
        )
        xdat.sel(wavelength=575, laser=445, method="nearest").plot(
            label="FAD dominant", c="r", ax=ax[2]
        )
        ax[3].plot(dat["T_arr"], dat["fb"], label="fb")
        ax[3].plot(dat["T_arr"], dat["ff"], label="ff")
        ax[3].plot(dat["T_arr"], dat["f_autofl"], label="f_autofl")
        for a in ax:
            a.set(xlim=(0, 90), ylim=(0, 0.8))
            a.legend()

        plt.tight_layout()
        plt.show()

    if verbose:
        pprint(dat.keys())

    def concat_fn(xdat):
        n_samples = xdat.shape[0]
        n_channels = xdat.shape[1]
        n_wavelengths = xdat.shape[2]
        xdat_concat = np.concatenate(
            [np.squeeze(xdat[:, j, :]) for j in range(n_channels)], axis=1
        )
        xdat_concat = xdat_concat.T
        assert xdat_concat.shape == (
            n_wavelengths * n_channels,
            n_samples,
        ), "shape mismatch"
        return xdat_concat

    xdat_concat = concat_fn(xdat)

    return dat, sim, S, S_autofl, xdat, xdat_concat, W_df
