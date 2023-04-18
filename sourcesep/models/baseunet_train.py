# %%
from sourcesep.models.baseunet import LitBaseUnet
from sourcesep.models.helpers import H5DataModule
from sourcesep.utils.config import load_config

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from timebudget import timebudget
import toml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--expt', type=str, default='unet')

def main(expt='unet'):

    n_epochs = 10
    train_samples_per_epoch = 1000
    val_samples_per_epoch = 100
    batch_size = 20

    # data paths
    paths = load_config(dataset_key='all')
    expt_path = paths['root'] / 'results' / expt 
    sim_name = '2023-03-25'
    train_h5_filename = str(paths['root'] / "sims" / f"{sim_name}_train.h5")
    val_h5_filename = str(paths['root'] / "sims" / f"{sim_name}_val.h5")
    datamodule = H5DataModule(train_h5_filename=train_h5_filename,
                              val_h5_filename=val_h5_filename,
                              n_timesamples=2048,
                              train_samples_per_epoch=train_samples_per_epoch,
                              val_samples_per_epoch=val_samples_per_epoch,
                              batch_size=batch_size,
                              train_seed=None,
                              val_seed=0)
    
    #Get constants relevant for reconstruction
    S, W, E, Mu_ox, Mu_dox, B = datamodule.get_sim_arrays()

    ckpt_best_val_loss = ModelCheckpoint(monitor="val_loss", dirpath=expt_path, filename='best-val-loss', verbose=True)
    ckpt_best_val_exp_var = ModelCheckpoint(monitor="val_exp_var", dirpath=expt_path, filename='best-val-exp-var', verbose=True)
    
    tb_logger = TensorBoardLogger(paths['root'] / 'results',
                               name='lt_logs',
                               version=expt,
                               prefix='',
                               log_graph=False, 
                               default_hp_metric=True)

    trainer = Trainer(max_epochs=n_epochs,
                      accelerator='gpu',
                      devices=1,
                      logger=[tb_logger], 
                      log_every_n_steps=1, 
                      reload_dataloaders_every_n_epochs=1,
                      check_val_every_n_epoch=1,
                      callbacks=[ckpt_best_val_loss])

    model = LitBaseUnet(in_channels=300,
                        S=S,
                        W=W,
                        E=E,
                        Mu_ox=Mu_ox,
                        Mu_dox=Mu_dox,
                        B=B).float()

    with timebudget('training'):
        trainer.fit(model, datamodule=datamodule)
    save_path = str(expt_path / 'final-model.ckpt')
    print(f'Saving model at end of training to {save_path}')
    trainer.save_checkpoint(expt_path / 'exit-model.ckpt')


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))