# write a script using optuna to optimize the parameters of nmf
import logging
import sys
import itertools
import optuna
import numpy as np
import h5py
from sklearn.decomposition import NMF
from sourcesep.utils.compute import perm_avgabscorr
from sourcesep.utils.config import load_config


def get_data():
    paths = load_config(dataset_key='all')
    f = h5py.File(paths['root'] / 'sims' / '2023-03-08.h5', 'r')
    for key in f.keys():
        print(key, f[key].shape)

    # prepare data and ground truth
    O = f['O'][:]
    A = f['A'][:]

    # flatten channels
    O = O.reshape(O.shape[0], -1)
    O.shape
    return O, A


def do_study(study_name, n_trials):
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(
        logging.StreamHandler(sys.stdout))

    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize')

    O, A = get_data()

    def objective(trial):

        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        alpha_W = trial.suggest_float('alpha_W', 0.0, 1.0)
        alpha_H = trial.suggest_float('alpha_H', 0.0, 1.0)

        model = NMF(n_components=3, init='nndsvd',
                    solver='cd', beta_loss='frobenius',
                    tol=1e-4, max_iter=500, random_state=None,
                    alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio,
                    verbose=0, shuffle=False)
        W = model.fit_transform(O.T)
        H = model.components_

        # add small random noise to H to issues with all zeros
        H += 1e-6 * np.random.randn(*H.shape)

        perm, corr = perm_avgabscorr(H.T, A)
        best_ind = np.argmax(corr)
        best_perm = perm[best_ind]
        best_corr = corr[best_ind]

        return best_corr

    study.optimize(objective, n_trials=n_trials)
    print('Best parameter set')
    print(study.best_params)

    print('Best objective value')
    print(study.best_value)
    return


if __name__ == '__main__':
    do_study(study_name='nmf-optimization', n_trials=10000)
