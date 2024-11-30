import h5py
import toml
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset
from pathlib import Path


class H5Dataset(Dataset):
    def __init__(self, h5_filename, n_timesamples):
        self.n_timesamples = n_timesamples
        self.f = h5py.File(h5_filename, "r")
        cfg_filename = h5_filename.replace(".h5", ".toml")
        if Path.exists(Path(cfg_filename)):
            self.cfg = toml.load(cfg_filename)
        else:
            print(f"{cfg_filename} not found!")
            self.cfg = None

    def __len__(self):
        return self.f["O"].shape[0] - self.n_timesamples

    def _proc(self, x, idx):
        return np.swapaxes(x[idx : idx + self.n_timesamples].reshape(self.n_timesamples, -1), axis1=0, axis2=1).astype(
            np.float32
        )

    def __getitem__(self, idx):
        return dict(
            O=self._proc(self.f["O"], idx),
            A=self._proc(self.f["A"], idx),
            N=self._proc(self.f["N"], idx),
            M=self._proc(self.f["M"], idx),
            H_ox=self._proc(self.f["H_ox"], idx),
            H_dox=self._proc(self.f["H_dox"], idx),
        )


class H5DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_h5_filename,
        val_h5_filename,
        n_timesamples,
        train_samples_per_epoch=400,
        val_samples_per_epoch=20,
        train_seed=None,
        val_seed=None,
        batch_size=20,
        num_workers=8,
    ):
        super().__init__()
        self.train_h5_filename = train_h5_filename
        self.val_h5_filename = val_h5_filename
        self.n_timesamples = n_timesamples
        self.batch_size = batch_size
        self.train_samples_per_epoch = train_samples_per_epoch
        self.val_samples_per_epoch = val_samples_per_epoch
        self.dataset = None
        self.train_seed = train_seed
        self.val_seed = val_seed
        self.num_workers = num_workers

    def get_sim_arrays(self):
        f = h5py.File(self.train_h5_filename, "r")
        S = f["S"][:]
        W = f["W"][:]
        E = f["E"][:]
        Mu_ox = f["Mu_ox"][:]
        Mu_dox = f["Mu_dox"][:]
        B = f["B"][:]
        return S, W, E, Mu_ox, Mu_dox, B

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = H5Dataset(h5_filename=self.train_h5_filename, n_timesamples=self.n_timesamples)
            self.val_dataset = H5Dataset(h5_filename=self.val_h5_filename, n_timesamples=self.n_timesamples)

    def train_dataloader(self):
        rng = np.random.RandomState(self.train_seed)
        subset_idx = rng.choice(a=len(self.train_dataset), size=self.train_samples_per_epoch, replace=False)
        sub_dataset = Subset(self.train_dataset, subset_idx)
        return DataLoader(
            sub_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        rng = np.random.RandomState(self.val_seed)
        subset_idx = rng.choice(a=len(self.val_dataset), size=self.val_samples_per_epoch, replace=False)
        sub_dataset = Subset(self.val_dataset, subset_idx)
        return DataLoader(
            sub_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
