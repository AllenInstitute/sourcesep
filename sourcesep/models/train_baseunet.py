# %%
from sourcesep.models.baseunet import LitBaseUnet
from sourcesep.models.helpers import H5DataModule
from sourcesep.utils.config import load_config

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from timebudget import timebudget
import toml

def main():

    n_epochs = 2000
    train_steps_per_epoch = 1000
    val_steps_per_epoch = 20
    batch_size = 100

    # data paths
    paths = load_config(dataset_key='all')
    sim_name = '2023-03-08'
    h5_filename = str(paths['root'] / "sims" / f"{sim_name}.h5")
    cfg_filename = str(paths['root'] / "sims" / f"{sim_name}.toml")
    sim_cfg = toml.load(cfg_filename)
    datamodule = H5DataModule(h5_filename=h5_filename,
                              n_per_sample=2048,
                              train_steps_per_epoch=train_steps_per_epoch,
                              val_steps_per_epoch=20,
                              batch_size=batch_size)
    S, W, E, Mu_ox, Mu_dox, B = datamodule.get_sim_arrays()
    logger = TensorBoardLogger(paths['root'] / 'results', name='lt_logs', version='version_102',
                               log_graph=False, default_hp_metric=True, prefix='', sub_dir='DEBUG')
    trainer = Trainer(logger=logger, log_every_n_steps=1, max_epochs=n_epochs,
                      reload_dataloaders_every_n_epochs=1, accelerator='gpu', devices=1, check_val_every_n_epoch=2)
    model = LitBaseUnet(in_channels=300, out_channels=8, 
                        S=S, W=W, E=E, Mu_ox=Mu_ox, Mu_dox=Mu_dox, B=B).float()

    with timebudget('training'):
        trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
