'''
Set of benchmarks for different holdout randomization testing algorithms.
'''
import numpy as np
import time
from scipy.stats import norm, binom
from collections import defaultdict
from timings import DataGeneratingModel, fit_bayes_ridge, fit_lasso, fit_rf, fit_tabpfn, fit_xgboost, fit_ols, fit_keras_nn, create_train_test, create_folds, basic_binom_hrt



if __name__ == '__main__':
    # N samples, P covariates, 4 non-null, repeat nfolds indepent times, with error rate alpha
    N = 100
    P = 50
    nruns = 50
    nfolds = 5
    ntested = 50
    nsignals = 3
    response_structure = 'tanh'


    nfactors = 5
    sigma = 1
    ntrials = [1, 5, 10, 30, 50, 100, 500, 10000]

    # reproducibility
    np.random.seed(42)

    # Quieter sklearn
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    np.set_printoptions(precision=2, suppress=True)

    # Consider a few different predictive models
    fit_fn =  fit_bayes_ridge#[fit_ols, fit_bayes_ridge, fit_lasso, fit_rf, fit_xgboost, fit_keras_nn, fit_tabpfn]
    model = 'bayes_ridge'
    testers = [f'binom{x}' for x in ntrials]
    ntesters = len(testers)

    p_values = np.zeros((nruns, ntested, ntesters))
    timings = np.zeros_like(p_values)
    for run in range(nruns):
        print('Trial {}'.format(run+1))

        # Generate the data
        dgm = DataGeneratingModel(N, P, sigma=sigma, nfactors=nfactors, nsignals=nsignals)

        # Split the data into folds (share folds between tests for consistency)
        folds = create_folds(dgm.X, nfolds)

        # Look at the first three features to see how good our power is using diff methods
        for signal_idx in range(ntested):
            method_idx = 0
            for tr in ntrials: 
                start = time.time()
                p_values[run, signal_idx, method_idx] = basic_binom_hrt(dgm, signal_idx, fit_fn, test_fold=folds[0], ntrials=tr)
                end = time.time()
                timings[run, signal_idx, method_idx] = end - start
                method_idx += 1

    flat_data = np.stack([p_values.flatten(), timings.flatten()], axis=1)
    np.savetxt(f"csv/ntrials/N{N}_p{P}_{model}_{response_structure}_nsignal{nsignals}_ntested{ntested}_nruns{nruns}_nfolds{nfolds}.csv", flat_data, delimiter=",", header="p_value,timing", comments="")

























