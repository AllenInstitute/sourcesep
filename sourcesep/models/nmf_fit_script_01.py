
from sourcesep.models.nmf_variants import nmf_recipe_02
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sourcesep.utils.config import load_config

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", font_scale=0.8, rc=custom_params)


def main():
    paths = load_config(dataset_key='all')
    fname = paths['root'] / 'sims' / 'nmf_single_indicator_test_data.pkl'
    with open(fname, 'rb') as f:
        dat = pickle.load(f)

    A, X, err, init = nmf_recipe_02(dat['Y'].copy(), A=dat['A_init_0'].copy(), X=dat['X_init_0'].copy(), rank=5,
                                    A_keep=dat['A_keep'].copy(), X_keep=None, grad_type=None,
                                    step_size_A=1.0, step_size_X=1.0,
                                    lam_sparsity_x=0.0, lam_2norm_x=0.00, lam_deviation_x=0.0,
                                    fit_A=False, fit_X=True,
                                    tol=1e-2, exit_tol=1e-7, max_iter=100)

    res_warmup = {'A': A.copy(), 'X': X.copy(), 'err': err}
    print(f'warmup: {np.log10(err[-1]):0.5f}')

    # -------------------------------------------------------------------------------------------------------------
    A, X, err, init = nmf_recipe_02(dat['Y'].copy(), A=res_warmup['A'].copy(), X=res_warmup['X'].copy(), rank=5,
                                    A_keep=dat['A_keep'].copy(), X_keep=None, grad_type=None,
                                    step_size_A=0.1, step_size_X=0.1,
                                    lam_sparsity_x=0.0, lam_2norm_x=0.00, lam_deviation_x=0.0,
                                    fit_A=True, fit_X=True,
                                    tol=1e-4, exit_tol=1e-7, max_iter=400)

    res_fine_tune_0 = {'A': A.copy(), 'X': X.copy(), 'err': err}
    print(f'fine_tune_0: {np.log10(err[-1]):0.5f}')

    # -------------------------------------------------------------------------------------------------------------
    A, X, err, init = nmf_recipe_02(dat['Y'].copy(), A=res_fine_tune_0['A'].copy(), X=res_fine_tune_0['X'].copy(), rank=5,
                                    A_keep=dat['A_keep'].copy(), X_keep=None, grad_type=None,
                                    step_size_A=0.05, step_size_X=0.05,
                                    lam_sparsity_x=0.0, lam_2norm_x=0.00, lam_deviation_x=0.0,
                                    fit_A=True, fit_X=True,
                                    tol=1e-5, exit_tol=1e-7, max_iter=800)

    res_fine_tune_1 = {'A': A.copy(), 'X': X.copy(), 'err': err}
    print(f'fine_tune_1: {np.log10(err[-1]):0.5f}')

    # -------------------------------------------------------------------------------------------------------------
    A, X, err, init = nmf_recipe_02(dat['Y'].copy(), A=res_fine_tune_1['A'].copy(), X=res_fine_tune_1['X'].copy(), rank=5,
                                    A_keep=dat['A_keep'].copy(), X_keep=None, grad_type=None,
                                    step_size_A=0.01, step_size_X=0.01,
                                    lam_sparsity_x=0.0, lam_2norm_x=0.00, lam_deviation_x=0.0,
                                    fit_A=True, fit_X=True,
                                    tol=1e-6, exit_tol=1e-7, max_iter=1600)

    res_fine_tune_2 = {'A': A.copy(), 'X': X.copy(), 'err': err}
    print(f'fine_tune_2: {np.log10(err[-1]):0.5f}')

    # -------------------------------------------------------------------------------------------------------------
    A, X, err, init = nmf_recipe_02(dat['Y'].copy(), A=res_fine_tune_2['A'].copy(), X=res_fine_tune_2['X'].copy(), rank=5,
                                    A_keep=dat['A_keep'].copy(), X_keep=None, grad_type=None,
                                    step_size_A=0.005, step_size_X=0.005,
                                    lam_sparsity_x=0.0, lam_2norm_x=0.00, lam_deviation_x=0.0,
                                    fit_A=True, fit_X=True,
                                    tol=1e-7, exit_tol=1e-7, max_iter=3200)

    res_fine_tune_3 = {'A': A.copy(), 'X': X.copy(), 'err': err}
    print(f'fine_tune_3: {np.log10(err[-1]):0.5f}')

    # -------------------------------------------------------------------------------------------------------------
    A, X, err, init = nmf_recipe_02(dat['Y'].copy(), A=res_fine_tune_3['A'].copy(), X=res_fine_tune_3['X'].copy(), rank=5,
                                    A_keep=dat['A_keep'].copy(), X_keep=None, grad_type=None,
                                    step_size_A=0.001, step_size_X=0.001,
                                    lam_sparsity_x=0.0, lam_2norm_x=0.00, lam_deviation_x=0.0,
                                    fit_A=True, fit_X=True,
                                    tol=1e-8, exit_tol=1e-7, max_iter=6400)

    res_fine_tune_4 = {'A': A.copy(), 'X': X.copy(), 'err': err}
    print(f'fine_tune_4: {np.log10(err[-1]):0.5f}')

    # -------------------------------------------------------------------------------------------------------------
    A, X, err, init = nmf_recipe_02(dat['Y'].copy(), A=res_fine_tune_4['A'].copy(), X=res_fine_tune_4['X'].copy(), rank=5,
                                    A_keep=dat['A_keep'].copy(), X_keep=None, grad_type=None,
                                    step_size_A=0.0005, step_size_X=0.0005,
                                    lam_sparsity_x=0.0, lam_2norm_x=0.00, lam_deviation_x=0.0,
                                    fit_A=True, fit_X=True,
                                    tol=1e-9, exit_tol=1e-9, max_iter=12800)

    res_fine_tune_5 = {'A': A.copy(), 'X': X.copy(), 'err': err}
    print(f'fine_tune_5: {np.log10(err[-1]):0.5f}')

    results = {'res_warmup': res_warmup,
               'res_fine_tune_0': res_fine_tune_0,
               'res_fine_tune_1': res_fine_tune_1,
               'res_fine_tune_2': res_fine_tune_2,
               'res_fine_tune_3': res_fine_tune_3,
               'res_fine_tune_4': res_fine_tune_4,
               'res_fine_tune_5': res_fine_tune_5}

    fname = paths['root'] / 'data/results/00_nmf_single_indicator_fit.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved results to {fname}')
    return


if __name__ == '__main__':
    main()
