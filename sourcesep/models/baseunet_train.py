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
    #train_samples_per_epoch = 100
    #val_samples_per_epoch = 20
    batch_size = 2
    train_samples_per_epoch = batch_size
    val_samples_per_epoch = batch_size

    # data paths
    paths = load_config(dataset_key='all')
    sim_name = '2023-03-08'
    h5_filename = str(paths['root'] / "sims" / f"{sim_name}.h5")
    cfg_filename = str(paths['root'] / "sims" / f"{sim_name}.toml")
    datamodule = H5DataModule(train_h5_filename=h5_filename,
                              val_h5_filename=h5_filename,
                              n_timesamples=2048,
                              train_samples_per_epoch=train_samples_per_epoch,
                              val_samples_per_epoch=val_samples_per_epoch,
                              batch_size=batch_size,
                              train_seed=0,
                              val_seed=0)
    
    #Get constants relevant for reconstruction
    S, W, E, Mu_ox, Mu_dox, B = datamodule.get_sim_arrays()
    
    logger = TensorBoardLogger(paths['root'] / 'results',
                               name='lt_logs',
                               version='tests',
                               prefix='',
                               sub_dir='DEBUG',
                               log_graph=False, default_hp_metric=True)

    trainer = Trainer(logger=logger, 
                      log_every_n_steps=1, 
                      max_epochs=n_epochs,
                      accelerator='gpu',
                      devices=1, overfit_batches=1.0)

    model = LitBaseUnet(in_channels=300,
                        S=S,
                        W=W,
                        E=E,
                        Mu_ox=Mu_ox,
                        Mu_dox=Mu_dox,
                        B=B).float()

    with timebudget('training'):
        trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
