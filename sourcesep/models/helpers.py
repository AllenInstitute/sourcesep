import h5py
import toml
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset
from pathlib import Path


class H5Dataset(Dataset):
    def __init__(self, h5_filename, n_per_sample):
        self.f = h5py.File(h5_filename, "r")
        cfg_filename = h5_filename.replace('.h5', '.toml')
        if Path.exists(Path(cfg_filename)):
            self.cfg = toml.load(cfg_filename)
        else:
            print(f'{cfg_filename} not found!')
            self.cfg = None
        self.n_per_sample = n_per_sample

    def __len__(self):
        return self.f['O'].shape[0] - self.n_per_sample
    
    def _proc(self, x, idx):
        return np.swapaxes(x[idx:idx+self.n_per_sample].reshape(self.n_per_sample, -1),axis1=0,axis2=1).astype(np.float32)

    def __getitem__(self, idx):
        return dict(O=self._proc(self.f['O'], idx),
                    A=self._proc(self.f['A'], idx),
                    N=self._proc(self.f['N'], idx),
                    M=self._proc(self.f['M'], idx),
                    H_ox=self._proc(self.f['H_ox'], idx),
                    H_dox=self._proc(self.f['H_dox'], idx))


class H5DataModule(pl.LightningDataModule):
    def __init__(self, h5_filename, n_per_sample, train_steps_per_epoch=400, val_steps_per_epoch=20, batch_size=20, num_workers=8):
        super().__init__()
        self.h5_filename = h5_filename
        self.n_per_sample = n_per_sample
        self.batch_size = batch_size
        self.train_steps_per_epoch = train_steps_per_epoch
        self.val_steps_per_epoch = val_steps_per_epoch
        self.dataset = None
        self.val_sub_dataloader = None
        self.num_workers = num_workers

    def get_sim_arrays(self):
        f = h5py.File(self.h5_filename, "r")
        S = f['S'][:]
        W = f['W'][:] 
        E = f['E'][:] 
        Mu_ox = f['Mu_ox'][:]
        Mu_dox = f['Mu_dox'][:]
        B = f['B'][:]
        return S, W, E, Mu_ox, Mu_dox, B

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.dataset = H5Dataset(h5_filename=self.h5_filename, n_per_sample=self.n_per_sample)

            # validation remains the same across epochs
            val_subset_idx = np.random.choice(len(self.dataset), self.val_steps_per_epoch, replace=False)
            val_sub_dataset = Subset(self.dataset, val_subset_idx)
            self.val_sub_dataloader = DataLoader(val_sub_dataset, batch_size=self.batch_size, 
                                                shuffle=False, num_workers=8, pin_memory=True)

        if stage == "test":
            # use when ground truth available
            pass

        if stage == "predict":
            # use when ground truth absent
            pass

    def train_dataloader(self):
        subset_idx = np.random.choice(len(self.dataset), self.train_steps_per_epoch, replace=False)
        sub_dataset = Subset(self.dataset, subset_idx)
        return DataLoader(sub_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return self.val_sub_dataloader

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass