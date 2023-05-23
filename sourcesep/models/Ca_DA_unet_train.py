# %%
from sourcesep.models.Ca_DA_unet import LitBaseUnet
from sourcesep.models.helpers import H5DataModule
from sourcesep.utils.config import load_config

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import tensorboard_trace_handler, schedule

from timebudget import timebudget
import toml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--expt', type=str, default='ca_da_debug')

def main(expt='ca_da_debug'):

    n_epochs = 10000
    max_time = '00:01:00:00' # in DD:HH:MM:SS format
    train_samples_per_epoch = 1000
    val_samples_per_epoch = 200
    batch_size = 20

    # data paths
    paths = load_config(dataset_key='all')
    expt_path = paths['root'] / 'results' / expt 
    sim_name = '2023-05-23_ca-da'
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

    ckpt_best_val_loss = ModelCheckpoint(monitor="val_loss", mode='min', dirpath=expt_path, filename='best-val-loss_{epoch}', verbose=True)
    ckpt_best_val_exp_var = ModelCheckpoint(monitor="val_metric_exp_var", mode='max', dirpath=expt_path, filename='best-val-exp-var_{epoch}', verbose=True)
    
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
                      callbacks=[ckpt_best_val_loss, ckpt_best_val_exp_var, RichProgressBar()],
                      max_time=max_time)

    model = LitBaseUnet(in_channels=180,
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