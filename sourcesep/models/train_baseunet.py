from sourcesep.models.baseunet import BaseUnet
from sourcesep.models.helpers import H5Dataset
from sourcesep.utils.config import load_config
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np

def main():
    n_epochs = 2
    steps_per_epoch = 400

    #data 
    paths = load_config(dataset_key='all')
    h5_file=str(paths['root'] / "sims" / "2023-02-24.h5")
    dataset = H5Dataset(filename=h5_file, n_per_sample=2000)

    for epoch in range(n_epochs):
        subset_idx = np.random.choice(len(dataset), steps_per_epoch, replace=False)
        sub_dataset = torch.utils.data.Subset(dataset, subset_idx)
        sub_dataloader = DataLoader(sub_dataset, batch_size=20, shuffle=True)

        for step, batch in enumerate(iter(sub_dataloader)):
            print(batch['O'].shape)
            pass
        print('epoch done')

    print('training done')
    return

if __name__ == '__main__':
    main()