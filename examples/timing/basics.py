'''
Set of benchmarks for different holdout randomization testing algorithms.
'''
import numpy as np
import time
from scipy.stats import norm, binom
from timings import DataGeneratingModel, fit_bayes_ridge, fit_lasso, fit_rf, fit_tabpfn, fit_xgboost, fit_ols, fit_keras_nn, fit_xgboost_GPU, create_train_test, create_folds,basic_hrt,basic_hpt, basic_binom_hrt, basic_hgt, basic_cpi, invalid_cpi




if __name__ == '__main__':
    # N samples, P covariates, 4 non-null, repeat nfolds indepent times, with error rate alpha
    N = 200
    P = 50
    nruns = 100
    nfolds = 5
    ntested = 50
    nsignals = 20
    response_structure = 'linear'

    nsamples = 10000
    nfactors = 5
    sigma = 1
    # reproducibility
    np.random.seed(42)

    # Quieter sklearn
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    np.set_printoptions(precision=2, suppress=True)

    # Consider a few different predictive models
    fit_fn =  fit_tabpfn#[fit_ols, fit_bayes_ridge, fit_lasso, fit_rf, fit_xgboost, fit_keras_nn, fit_tabpfn, fit_xgboost_GPU]
    model = 'tabpfn'
    testers = [#'Naive CRT', 'Naive CV-CRT',
               'Basic HRT', #'CV-HRT', #'Invalid CV-HRT',
               'Basic HGT', #'CV-HGT',
               'Basic HPT', #'CV-HPT',
               'Basic Binom_CRT',
               'Basic CPI',
               'Invalid CPI'
               ]
    ntesters = len(testers)
   

    p_values = np.zeros((nruns, ntested, ntesters))
    timings = np.zeros_like(p_values)
    for run in range(nruns):
        print('Trial {}'.format(run+1))

        # Generate the data
        dgm = DataGeneratingModel(N, P, sigma=sigma, nfactors=nfactors)

        # Split the data into folds (share folds between tests for consistency)
        folds = create_folds(dgm.X, nfolds)

        # Look at the first three features to see how good our power is using diff methods
        for signal_idx in range(ntested):
            method_idx = 0
            # start = time.time()
            # p_values[run, signal_idx, method_idx] = vanilla_crt(dgm, signal_idx, fit_fn, ntrials=nsamples)
            # end = time.time()
            # timings[run, signal_idx, method_idx] = end - start
            # method_idx += 1

            # start = time.time()
            # p_values[run, signal_idx, method_idx] = vanilla_cv_crt(dgm, signal_idx, fit_fn, nfolds=nfolds, ntrials=nsamples)
            # end = time.time()
            # timings[run, signal_idx, method_idx] = end - start
            # method_idx += 1

            start = time.time()
            p_values[run, signal_idx, method_idx] = basic_hrt(dgm, signal_idx, fit_fn, test_fold=folds[0], ntrials=nsamples)
            end = time.time()
            timings[run, signal_idx, method_idx] = end - start
            method_idx += 1

            # start = time.time()
            # p_values[run, signal_idx, method_idx] = valid_cv_hrt(dgm, signal_idx, fit_fn, folds=folds, ntrials=nsamples)
            # end = time.time()
            # timings[run, signal_idx, method_idx] = end - start
            # method_idx += 1

            # start = time.time()
            # p_values[run, signal_idx, method_idx] = invalid_cv_hrt(dgm, signal_idx, fit_fn, folds=folds, ntrials=nsamples)
            # end = time.time()
            # timings[run, signal_idx, method_idx] = end - start
            # method_idx += 1

            start = time.time()
            p_values[run, signal_idx, method_idx] = basic_hgt(dgm, signal_idx, fit_fn, test_fold=folds[0], ntrials=nsamples)
            end = time.time()
            timings[run, signal_idx, method_idx] = end - start
            method_idx += 1

            # start = time.time()
            # p_values[run, signal_idx, method_idx] = cv_hgt(dgm, signal_idx, fit_fn, folds=folds, ntrials=nsamples)
            # end = time.time()
            # timings[run, signal_idx, method_idx] = end - start
            # method_idx += 1

            start = time.time()
            p_values[run, signal_idx, method_idx] = basic_hpt(dgm, signal_idx, fit_fn, test_fold=folds[0], ntrials=nsamples)
            end = time.time()
            timings[run, signal_idx, method_idx] = end - start
            method_idx += 1

            start = time.time()
            p_values[run, signal_idx, method_idx] = basic_binom_hrt(dgm, signal_idx, fit_fn, test_fold=folds[0], ntrials=30)
            end = time.time()
            timings[run, signal_idx, method_idx] = end - start
            method_idx += 1

            start = time.time()
            p_values[run, signal_idx, method_idx] = basic_cpi(dgm, signal_idx, fit_fn, test_fold=folds[0], ntrials=100)
            end = time.time()
            timings[run, signal_idx, method_idx] = end - start
            method_idx += 1

            start = time.time()
            p_values[run, signal_idx, method_idx] = invalid_cpi(dgm, signal_idx, fit_fn, test_fold=folds[0], ntrials=100)
            end = time.time()
            timings[run, signal_idx, method_idx] = end - start
            method_idx += 1
            
    flat_data = np.stack([p_values.flatten(), timings.flatten()], axis=1)
    np.savetxt(f"csv/basics/N{N}_p{P}_{model}_{response_structure}_nsignal{nsignals}_ntested{ntested}_nruns{nruns}_nfolds{nfolds}.csv", flat_data, delimiter=",", header="p_value,timing", comments="")

