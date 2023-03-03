import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset


class H5Dataset(Dataset):
    def __init__(self, filename, n_per_sample):
        self.f = h5py.File(filename, "r")
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
    def __init__(self, filename, n_per_sample, train_steps_per_epoch=400, val_steps_per_epoch=20, batch_size=20):
        super().__init__()
        self.filename = filename
        self.n_per_sample = n_per_sample
        self.batch_size = batch_size
        self.train_steps_per_epoch = train_steps_per_epoch
        self.val_steps_per_epoch = val_steps_per_epoch
        self.dataset = None
        self.val_sub_dataloader = None

    def prepare_data(self, dataset_id):
        # paths = load_config(dataset_key='all')
        # sim = SimData(T=36000*6, cfg_path=paths['root'] / "sim_config.toml")
        # sim.to_disk(filepath=str(paths['root'] / "sims" / f"{dataset_id}.h5"))
        pass
        return

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # training dataset will be sampled from `self.dataset`
            self.dataset = H5Dataset(filename=self.filename, n_per_sample=self.n_per_sample)

            # validation remains the same across epochs
            val_subset_idx = np.random.choice(len(self.dataset), self.val_steps_per_epoch, replace=False)
            val_sub_dataset = Subset(self.dataset, val_subset_idx)
            self.val_sub_dataloader = DataLoader(val_sub_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        if stage == "test":
            pass

        if stage == "predict":
            pass

    def train_dataloader(self):
        subset_idx = np.random.choice(len(self.dataset), self.train_steps_per_epoch, replace=False)
        sub_dataset = Subset(self.dataset, subset_idx)
        print('here', subset_idx[0])
        return DataLoader(sub_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return self.val_sub_dataloader

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass