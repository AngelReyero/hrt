'''
Set of benchmarks for different holdout randomization testing algorithms.
'''
import numpy as np
import time
from scipy.stats import norm, binom
from collections import defaultdict
from timings import DataGeneratingModel, fit_bayes_ridge, fit_lasso, fit_rf, fit_tabpfn, fit_xgboost, fit_ols, fit_keras_nn,fit_fast_xgboost, basic_binom_hrt



if __name__ == '__main__':
    # N samples, P covariates, 4 non-null, repeat nfolds indepent times, with error rate alpha
    N = 200
    P = 100
    nruns = 50
    nfolds = 5
    ntested = 100
    nsignals = 3
    response_structure = 'tanh'

    nfactors = 5
    sigma = 1
    test_percentage = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    # reproducibility
    np.random.seed(42)

    # Quieter sklearn
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    np.set_printoptions(precision=2, suppress=True)

    # Consider a few different predictive models
    fit_fn =  fit_ols#[fit_ols, fit_bayes_ridge, fit_lasso, fit_rf, fit_xgboost, fit_keras_nn, fit_tabpfn, fit_fast_xgboost]
    model = 'ols'
    testers = [f'binom{x}' for x in test_percentage]
    ntesters = len(testers)
    

    p_values = np.zeros((nruns, ntested, ntesters))
    timings = np.zeros_like(p_values)
    for run in range(nruns):
        print('Trial {}'.format(run+1))

        # Generate the data
        dgm = DataGeneratingModel(N, P, sigma=sigma, nfactors=nfactors, nsignals=nsignals)


        # Look at the first three features to see how good our power is using diff methods
        for signal_idx in range(ntested):
            method_idx = 0
            for percent in test_percentage: 
                start = time.time()
                p_values[run, signal_idx, method_idx] = basic_binom_hrt(dgm, signal_idx, fit_fn, test_fold=percent, ntrials=5)
                end = time.time()
                timings[run, signal_idx, method_idx] = end - start
                method_idx += 1

    flat_data = np.stack([p_values.flatten(), timings.flatten()], axis=1)

    # Save to CSV
    np.savetxt(f"csv/sample_size/N{N}_p{P}_{model}_{response_structure}_nsignal{nsignals}_ntested{ntested}_nruns{nruns}_nfolds{nfolds}.csv", flat_data, delimiter=",", header="p_value,timing", comments="")

    