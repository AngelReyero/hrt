'''
Set of benchmarks for different holdout randomization testing algorithms.
'''
import numpy as np
import time
from scipy.stats import norm, binom
from collections import defaultdict

# Benjamini-hochberg
def bh(p, fdr):
    p_orders = np.argsort(p)
    discoveries = []
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (k+1) / m * fdr:
            discoveries.append(s)
        else:
            break
    return np.array(discoveries)

class DataGeneratingModel:
    def __init__(self, N, P, sigma=1, nfactors=5, response_structure='tanh', nsignals = 3):
        self.N = N
        self.P = P
        self.sigma = sigma
        self.nfactors = nfactors
        self.response_structure=response_structure
        self.nsignals = nsignals
        self.covariates()
        self.response()

    def covariates(self):
        # Make a simple correlated errors model via low-rank non-linear factor model
        self.Z = np.random.gamma(1, 1, size=(self.N, self.nfactors))
        self.W = np.random.normal(0, 1/np.sqrt(self.nfactors), size=(self.P, self.nfactors))
        self.X = np.random.normal(self.Z.dot(self.W.T), self.sigma)

    def response(self):
        # Nonlinear model in a few covariates
        if self.response_structure=='tanh':
            self.signals = np.array([0.1,0.1,0.1])
            self.y = np.tanh(self.X[:,0]) * self.signals[0] + \
                    5*np.tanh(self.X[:,1] * self.signals[1] + self.X[:,2] * self.signals[2]) + \
                    np.random.normal(0, self.sigma, size=self.X.shape[0])
            self.nsignals = 3

        elif self.response_structure=='linear':
            self.signals = np.zeros(N)
            self.signals[:self.nsignals] = np.random.choice([-3, -2, -1, 1, 2, 3], size=self.nsignals)
            self.y = self.X[:,:len(self.signals)].dot(self.signals)


    def conditional_samples(self, rows, idx, nsamples=1):
        # Sample from the complete conditional distribution
        #print(self.Z[rows].dot(self.W[idx]).reshape(-1, 1))
        #print(np.random.normal(self.Z[rows].dot(self.W[idx]).reshape(-1, 1), self.sigma, size=(len(rows), nsamples)))
        return np.squeeze(np.random.normal(self.Z[rows].dot(self.W[idx]).reshape(-1, 1), self.sigma, size=(len(rows), nsamples)))

    def conditional_grid(self, rows, idx, ngrid, tol=1e-8):
        # Conditional mean for each sample in rows
        mu = self.Z[rows].dot(self.W[idx])

        # Complete conditional probabilities on a uniform grid covering nearly 
        # all the conditional support of the feature
        grid_start = norm.ppf(tol, mu, scale=self.sigma)
        grid_end = norm.ppf(1-tol, mu, scale=self.sigma)
        grid = np.array([np.linspace(start, end, ngrid-1) for start, end in zip(grid_start, grid_end)])
        
        # Include the real value
        grid = np.concatenate([grid, self.X[rows, idx:idx+1]], axis=1)

        # Get the probability of each grid point
        grid_probs = norm.pdf(grid, mu[:,None], scale=self.sigma)
        grid_probs = grid_probs / grid_probs.sum(axis=1, keepdims=True)

        return grid, grid_probs


    def permutation_probs(self, rows, idx):
        # Complete conditional probabilities for all unique feature values
        unique_vals = np.unique(self.X[rows,idx])
        logprobs = norm.logpdf(unique_vals[None], self.Z[rows].dot(self.W[idx])[:,None], scale=self.sigma)
        logprobs -= logprobs.max(axis=1, keepdims=True)
        numerator = np.exp(logprobs)
        return numerator / numerator.sum(axis=1, keepdims=True)

def fit_lasso(X, y):
    ''' Fit a lasso model '''
    from sklearn.linear_model import LassoCV
    lasso = LassoCV(cv=5)
    lasso.fit(X, y)
    return lasso

def fit_bayes_ridge(X, y):
    ''' Fit a Bayesian linear model with ARD '''
    from sklearn.linear_model import ARDRegression
    ard = ARDRegression()
    ard.fit(X, y)
    return ard

def fit_rf(X, y):
    ''' Fit a random forest model '''
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=20)
    rf.fit(X,y)
    return rf

def fit_xgboost(X, y):
    ''' Fit a gradient boosted trees model using XGBoost  '''
    import xgboost as xgb

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X, y, verbose=False)
    return model

class XGBoostGPU:
    def __init__(self, max_depth=6, learning_rate=0.1, n_estimators=100):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.model = None

    def fit(self, X, y):
        import xgboost as xgb

        """ Fit the XGBoost model on GPU """
        # Convert input data into DMatrix, which is required for XGBoost
        dtrain = xgb.DMatrix(X, label=y)
        
        # Define parameters for GPU training
        params = {
            'objective': 'reg:squarederror',
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'tree_method': 'hist',  # Histogram-based algorithm (faster for large datasets)
            'device': 'cuda',  # Use GPU for training
        }

        # Train the model
        self.model = xgb.train(params, dtrain, num_boost_round=self.n_estimators)

    def predict(self, X):
        """ Predict using the trained XGBoost model """

        import xgboost as xgb

        # Convert test data into DMatrix
        dtest = xgb.DMatrix(X)
        
        # Use the trained model to predict
        return self.model.predict(dtest)

def fit_xgboost_GPU(X, y):
    """ Fit function for XGBoost with GPU """
    xgb_model = XGBoostGPU()
    xgb_model.fit(X, y)
    return xgb_model


def fit_fast_xgboost(X, y):
    ''' Fit a lightweight and fast XGBoost regressor '''
    import xgboost as xgb

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=20,          # fewer trees
        max_depth=3,              # shallower trees
        learning_rate=0.2,        # faster convergence
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method='auto',       # can try 'exact' if your data is small
        verbosity=0,
        random_state=42
    )

    model.fit(X, y)
    return model

def fit_tabpfn(X, y):
    ''' Fit a TabPFN regressor on the full dataset '''
    from tabpfn import TabPFNRegressor

    # Convert to NumPy arrays with correct types
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Initialize model
    model = TabPFNRegressor()

    # Fit on full data
    model.fit(X, y)

    return model

class OLS:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.coef_ = np.linalg.solve(X.T.dot(X), X.T.dot(y))

    def predict(self, X):
        return X.dot(self.coef_)

def fit_ols(X, y):
    ols = OLS()
    ols.fit(X, y)
    return ols

def fit_keras_nn(X, y):
    ''' Fit a simple neural network using Keras with GPU support '''
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("No GPU found. Please ensure TensorFlow is configured with GPU support.")
    
    # Optional: Set memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)

    X = np.asarray(X)
    y = np.asarray(y)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(0.001), loss='mse')
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    return model

def create_train_test(X, test_fold):
    # Split the data into train/test
    if isinstance(test_fold, float):
        test_fold = np.random.choice(X.shape[0], size=int(np.round(X.shape[0]*test_fold)), replace=False)
    mask = np.ones(X.shape[0], dtype=bool)
    mask[test_fold] = False
    train_fold = np.arange(X.shape[0])[mask]
    return train_fold, test_fold

def create_folds(X, k):
    if isinstance(X, int) or isinstance(X, np.integer):
        indices = np.arange(X)
    elif hasattr(X, '__len__'):
        indices = np.arange(len(X))
    else:
        indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    folds = []
    start = 0
    end = 0
    for f in range(k):
        start = end
        end = start + len(indices) // k + (1 if (len(indices) % k) > f else 0)
        folds.append(indices[start:end])
    return folds

def cv_mse(X, y, folds, fit_fn):
    # Calculate cross-validation mean-squared error
    mse = 0
    for fold in folds:
        # Split into train and test for this fold
        mask = np.ones(X.shape[0], dtype=bool)
        mask[fold] = False
        X_train, y_train = X[mask], y[mask]
        X_test, y_test = X[fold], y[fold]

        model = fit_fn(X_train, y_train)
        pred = model.predict(X_test)
        mse += ((y_test - pred)**2).sum()
    return mse / len(folds)

def vanilla_crt(dgm, idx, fit_fn, ntrials=1000, **kwargs):
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Split the data into train/test
    train_fold, test_fold = create_folds(X, 2)
    X_train, y_train = X[train_fold], y[train_fold]
    X_test, y_test = X[test_fold], y[test_fold]

    # Get the test MSE
    model = fit_fn(X_train, y_train)
    pred = model.predict(X_test)
    t_true = ((y_test - pred)**2).sum()

    # Run the vanilla CRT that refits the model for every null sample
    rows = np.arange(X.shape[0])
    p_value = 0
    for trial in range(ntrials):
        # Sample a null column for the target feature
        X[:,idx] = dgm.conditional_samples(rows, idx)
        X_train, X_test = X[train_fold], X[test_fold]

        # Get the null test MSE
        model = fit_fn(X_train, y_train)
        pred = model.predict(X_test)
        t_null = ((y_test - pred)**2).sum()

        # Add 1 if the null was at least as good as the true feature
        p_value += int(t_true >= t_null)

    # Return the one-sided p-value
    return (1+p_value) / (1+ntrials)


def vanilla_cv_crt(dgm, idx, fit_fn, folds=5, ntrials=1000, **kwargs):
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Split the data into folds (if they are not already provided)
    if isinstance(folds, int):
        folds = create_folds(X, folds)
    nfolds = len(folds)

    # Get the true cross-validation error
    t_true = cv_mse(X, y, folds, fit_fn)

    # Run the vanilla CRT that refits the model for every null sample
    rows = np.arange(X.shape[0])
    p_value = 0
    for trial in range(ntrials):
        # Sample a null column for the target feature
        X[:,idx] = dgm.conditional_samples(rows, idx)

        # Get the nul CV MSE score
        t_null = cv_mse(X, y, folds, fit_fn)

        # Add 1 if the null was at least as good as the true feature
        p_value += int(t_true >= t_null)

    # Return the one-sided p-value
    return (1+p_value) / (1+ntrials)

def basic_hrt(dgm, idx, fit_fn, test_fold=0.2, ntrials=1000, **kwargs):
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Get the train and test indices
    train_fold, test_fold = create_train_test(X, test_fold)
    X_train, y_train = X[train_fold], y[train_fold]
    X_test, y_test = X[test_fold], y[test_fold]

    # Get the test MSE
    model = fit_fn(X_train, y_train)
    pred = model.predict(X_test)
    t_true = ((y_test - pred)**2).sum()

    # Run the basic HRT that avoids refitting on the test set
    p_value = 0
    for trial in range(ntrials):
        # Sample a null column for the target feature
        X_test[:,idx] = dgm.conditional_samples(test_fold, idx)

        # Get the null test MSE
        pred = model.predict(X_test)
        t_null = ((y_test - pred)**2).sum()

        # Add 1 if the null was at least as good as the true feature
        p_value += int(t_true >= t_null)

    # Return the one-sided p-value
    return (1+p_value) / (1+ntrials)


def basic_binom_hrt(dgm, idx, fit_fn, test_fold=0.2, ntrials=1, **kwargs):
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Get the train and test indices
    train_fold, test_fold = create_train_test(X, test_fold)
    X_train, y_train = X[train_fold], y[train_fold]
    X_test, y_test = X[test_fold], y[test_fold]

    # Get the test MSE
    model = fit_fn(X_train, y_train)
    pred = model.predict(X_test)
    loss_difference = -((y_test - pred)**2)
    for trial in range(ntrials):
        # Sample a null column for the target feature
        X_test[:,idx] = dgm.conditional_samples(test_fold, idx)

        # Get the null test MSE
        pred = model.predict(X_test)
        loss_difference += ((y_test - pred)**2)/ntrials

    binom_stat=(loss_difference>0).sum()
    # Return the one-sided p-value
    return binom.sf(binom_stat-1, y_test.shape[0], 0.5)


def invalid_binom_hrt(dgm, idx, fit_fn, test_fold=0.2, ntrials=1, **kwargs):
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Get the train and test indices
    train_fold, test_fold = create_train_test(X, test_fold)
    X_train, y_train = X[train_fold], y[train_fold]
    X_test, y_test = X[test_fold], y[test_fold]

    # Get the test MSE
    model = fit_fn(X_train, y_train)
    pred = model.predict(X_test)
    true_loss = ((y_test - pred)**2)
    binom_stat=0
    for trial in range(ntrials):
        # Sample a null column for the target feature
        X_test[:,idx] = dgm.conditional_samples(test_fold, idx)

        # Get the null test MSE
        pred = model.predict(X_test)
        binom_stat += (((y_test - pred)**2-true_loss)>0).sum()


    # Return the one-sided p-value
    return binom.sf(binom_stat-1, ntrials*y_test.shape[0], 0.5)

def basic_cpi(dgm, idx, fit_fn, test_fold=0.2, ntrials=100, **kwargs):
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Get the train and test indices
    train_fold, test_fold = create_train_test(X, test_fold)
    X_train, y_train = X[train_fold], y[train_fold]
    X_test, y_test = X[test_fold], y[test_fold]

    # Get the test MSE
    model = fit_fn(X_train, y_train)
    pred = model.predict(X_test)
    t_true = ((y_test - pred)**2).sum()

    # Run the basic HRT that avoids refitting on the test set
    cpi = []
    for trial in range(ntrials):
        # Sample a null column for the target feature
        X_test[:,idx] = dgm.conditional_samples(test_fold, idx)

        # Get the null test MSE
        pred = model.predict(X_test)
        t_null = ((y_test - pred)**2).sum()
        cpi.append(t_null-t_true)

    cpi=np.array(cpi)
    # Return the one-sided p-value
    return norm.sf(np.mean(cpi)/(np.std(cpi)+np.var(y_test)/np.sqrt(y_test.shape[0])))

def invalid_cpi(dgm, idx, fit_fn, test_fold=0.2, ntrials=100, **kwargs):
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Get the train and test indices
    train_fold, test_fold = create_train_test(X, test_fold)
    X_train, y_train = X[train_fold], y[train_fold]
    X_test, y_test = X[test_fold], y[test_fold]

    # Get the test MSE
    model = fit_fn(X_train, y_train)
    pred = model.predict(X_test)
    t_true = ((y_test - pred)**2).sum()

    # Run the basic HRT that avoids refitting on the test set
    cpi = []
    for trial in range(ntrials):
        # Sample a null column for the target feature
        X_test[:,idx] = dgm.conditional_samples(test_fold, idx)

        # Get the null test MSE
        pred = model.predict(X_test)
        t_null = ((y_test - pred)**2).sum()
        cpi.append(t_null-t_true)

    cpi=np.array(cpi)
    # Return the one-sided p-value
    return norm.sf(np.mean(cpi)/(np.std(cpi)+10e-6))

def fisher(p_values, axis=None):
    '''Implements Fisher's method for combining p-values.'''
    from scipy.stats import chi2
    # Convert to numpy
    if not isinstance(p_values, np.ndarray):
        p_values = np.array(p_values)
    # Check for hard zeroes
    zeroes = p_values.min(axis=axis) == 0
    if axis is None and zeroes:
        return 0
    # Get the number of p-values on the axis of interest
    N = np.prod(p_values.shape) if axis is None else p_values.shape[axis]

    # Fisher merge
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        results = chi2.sf(-2 * np.log(p_values).sum(axis=axis), 2*N)

    if axis is not None:
        # Fix any hard-zeros
        results[zeroes] = 0
    return results

def valid_cv_hrt(dgm, idx, fit_fn, folds=5, ntrials=1000, **kwargs):
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Split the data into folds (if they are not already provided)
    if isinstance(folds, int):
        folds = create_folds(X, folds)
    nfolds = len(folds)

    # Build separate p-values for each CV fold
    p_values = np.zeros(nfolds)
    for fold_idx in range(nfolds):
        # Split into train and test for this fold
        fold = folds[fold_idx]

        # Get the fold-specific p-value
        p_values[fold_idx] = basic_hrt(dgm, idx, fit_fn, test_fold=fold, ntrials=ntrials, **kwargs)

    # Correct using Holm-Bonferroni method
    p_values = np.maximum.accumulate(((p_values.shape[0] - np.arange(p_values.shape[0]))*p_values[np.argsort(p_values)]).clip(0,1))

    # Return the smallest p-value after correction
    return p_values[0]

    # Correct using the adjusted geometric mean (Mattner)
    # from scipy.stats import gmean
    # return np.e * gmean(p_values)

    # Correct using the adjusted harmonic mean (Vovk and Wang 2020, Biometrika)
    # from scipy.stats import hmean
    # return np.e * np.log(nfolds) * hmean(p_values)

    # Correct using Hommel (1983)
    # return np.min([1,np.sum(1/np.arange(1,nfolds+1)) * np.min(nfolds / np.arange(1,nfolds+1) * p_values[np.argsort(p_values)])])


def invalid_cv_hrt(dgm, idx, fit_fn, folds=5, ntrials=1000, **kwargs):
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Split the data into folds (if they are not already provided)
    if isinstance(folds, int):
        folds = create_folds(X, folds)

    # Build separate p-values for each CV fold
    t_true = 0
    t_null = np.zeros(ntrials)
    for fold_idx, fold in enumerate(folds):
        # Split into train and test for this fold
        mask = np.ones(X.shape[0], dtype=bool)
        mask[fold] = False
        X_train, y_train = X[mask], y[mask]
        X_test, y_test = X[fold], y[fold]

        # Get the true MSE on the held out fold
        model = fit_fn(X_train, y_train)
        pred = model.predict(X_test)
        t_true += ((y_test - pred)**2).sum()

        # Calculate the fold-specific p-value
        X_test_cv = np.copy(X_test)
        for trial in range(ntrials):
            # Sample a null column for the target feature
            X_test_cv[:,idx] = dgm.conditional_samples(fold, idx)
            
            # Get the null test MSE
            pred = model.predict(X_test_cv)
            t_null[trial] += ((y_test - pred)**2).sum()


    # Return the one-sided p-value
    return (1+(t_true >= t_null).sum()) / (1+ntrials)

def valid_cv_cpi(dgm, idx, fit_fn, folds=5, ntrials=1000, **kwargs):
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Split the data into folds (if they are not already provided)
    if isinstance(folds, int):
        folds = create_folds(X, folds)

    # Build separate p-values for each CV fold
    cpi = []
    for fold_idx, fold in enumerate(folds):
        # Split into train and test for this fold
        mask = np.ones(X.shape[0], dtype=bool)
        mask[fold] = False
        X_train, y_train = X[mask], y[mask]
        X_test, y_test = X[fold], y[fold]

        # Get the true MSE on the held out fold
        model = fit_fn(X_train, y_train)
        pred = model.predict(X_test)
        t_true = ((y_test - pred)**2).sum()

        # Calculate the fold-specific p-value
        X_test_cv = np.copy(X_test)
        for trial in range(ntrials):
            # Sample a null column for the target feature
            X_test_cv[:,idx] = dgm.conditional_samples(fold, idx)
            
            # Get the null test MSE
            pred = model.predict(X_test_cv)
            cpi.append(((y_test - pred)**2).sum()-t_true) 

    cpi = np.array(cpi)
    # Return the one-sided p-value
    return norm.sf(np.mean(cpi)/(np.std(cpi)+np.var(y)/np.sqrt(y.shape[0])))

def invalid_cv_cpi(dgm, idx, fit_fn, folds=5, ntrials=1000, **kwargs):
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Split the data into folds (if they are not already provided)
    if isinstance(folds, int):
        folds = create_folds(X, folds)

    # Build separate p-values for each CV fold
    cpi = []
    for fold_idx, fold in enumerate(folds):
        # Split into train and test for this fold
        mask = np.ones(X.shape[0], dtype=bool)
        mask[fold] = False
        X_train, y_train = X[mask], y[mask]
        X_test, y_test = X[fold], y[fold]

        # Get the true MSE on the held out fold
        model = fit_fn(X_train, y_train)
        pred = model.predict(X_test)
        t_true = ((y_test - pred)**2).sum()

        # Calculate the fold-specific p-value
        X_test_cv = np.copy(X_test)
        for trial in range(ntrials):
            # Sample a null column for the target feature
            X_test_cv[:,idx] = dgm.conditional_samples(fold, idx)
            
            # Get the null test MSE
            pred = model.predict(X_test_cv)
            cpi.append(((y_test - pred)**2).sum()-t_true) 

    cpi = np.array(cpi)
    # Return the one-sided p-value
    return norm.sf(np.mean(cpi)/(np.std(cpi)+10e-6))

def invalid_cv_binom_hrt(dgm, idx, fit_fn, folds=5, ntrials=1000, **kwargs):
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Split the data into folds (if they are not already provided)
    if isinstance(folds, int):
        folds = create_folds(X, folds)

    binom_stat = 0
    for fold_idx, fold in enumerate(folds):
        # Split into train and test for this fold
        mask = np.ones(X.shape[0], dtype=bool)
        mask[fold] = False
        X_train, y_train = X[mask], y[mask]
        X_test, y_test = X[fold], y[fold]

        # Get the true MSE on the held out fold
        model = fit_fn(X_train, y_train)
        pred = model.predict(X_test)

        loss_difference = -((y_test - pred)**2)
        X_test_cv = np.copy(X_test)
        for trial in range(ntrials):
            # Sample a null column for the target feature
            X_test_cv[:,idx] = dgm.conditional_samples(fold, idx)

            # Get the null test MSE
            pred = model.predict(X_test_cv)
            loss_difference += ((y_test - pred)**2)/ntrials

        binom_stat += (loss_difference>0).sum()
        # Return the one-sided p-value
    return binom.sf(binom_stat-1, y.shape[0], 0.5)


def grid_predictions(X, grid, idx, model):
    X_grid = np.repeat(X, grid.shape[1], axis=0)
    X_grid[:,idx] = grid.flatten()
    return model.predict(X_grid).reshape(grid.shape)

def basic_hgt(dgm, idx, fit_fn, test_fold=0.2, ntrials=1000, ngrid=50, **kwargs):
    '''A faster HRT that picks points on a grid along the support of the null distribution
    then randomly samples with probability proportional to the point in the null'''
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Split the data into train/test
    train_fold, test_fold = create_train_test(X, test_fold)
    X_train, y_train = X[train_fold], y[train_fold]
    X_test, y_test = X[test_fold], y[test_fold]

    # Get the grid and relative probabilities
    grid, grid_probs = dgm.conditional_grid(test_fold, idx, ngrid)

    # Fit the model and get predictions for every test grid point
    model = fit_fn(X_train, y_train)
    pred = grid_predictions(X_test, grid, idx, model)

    # Get the grid test errors
    t_grid = ((y_test[:,None] - pred)**2)
    t_true = t_grid[:,-1].sum() # The last column in the grid is the true X value

    # Run the basic HGT that avoids refitting on the test set
    t_null = np.zeros(ntrials)
    for s in range(X_test.shape[0]):
        t_null += np.random.choice(t_grid[s], p=grid_probs[s], replace=True, size=ntrials)

    # Return the one-sided p-value
    return (1+(np.logical_or(np.isclose(t_null, t_true), (t_null <= t_true))).sum()) / (1+ntrials)

def cv_hgt(dgm, idx, fit_fn, folds=5, ntrials=1000, **kwargs):
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Split the data into folds (if they are not already provided)
    if isinstance(folds, int):
        folds = create_folds(X, folds)
    nfolds = len(folds)

    # Build separate p-values for each CV fold
    p_values = np.zeros(nfolds)
    for fold_idx in range(nfolds):
        # Split into train and test for this fold
        fold = folds[fold_idx]
        
        # Run the HPT on this CV fold
        p_values[fold_idx] = basic_hgt(dgm, idx, fit_fn, test_fold=fold, ntrials=ntrials, **kwargs)

    # Correct using Holm-Bonferroni method
    p_values = np.maximum.accumulate(((p_values.shape[0] - np.arange(p_values.shape[0]))*p_values[np.argsort(p_values)]).clip(0,1))
    
    # Return the smallest p-value after correction
    return p_values[0]


def permutation_predictions(X, idx, model):
    # Predictions for all unique values of X
    unique_vals = np.unique(X[:,idx])
    arrs = []
    if idx > 0:
        arrs.append(np.repeat(X[:,:idx], X.shape[0], axis=0))
    arrs.append(np.tile(unique_vals[:,None], (X.shape[0],1)))
    if idx < X.shape[1]-1:
        arrs.append(np.repeat(X[:,idx+1:], X.shape[0], axis=0))
    X_perms = np.concatenate(arrs, axis=1)
    perm_preds = model.predict(X_perms).reshape((X.shape[0], unique_vals.shape[0]))
    return perm_preds

def hpt_mcmc(perm, probs, errors, mcmc_steps=50):
    # The MCMC algorithm from the CPT paper for sampling from the permutation distribution
    # Each step randomly swaps at most n // 2 indices
    even_stop = perm.shape[0] - (perm.shape[0] % 2)

    for step in range(mcmc_steps):
        # Choose n // 2 disjoint pairs of indices to try swapping
        indices = np.arange(perm.shape[0])
        np.random.shuffle(indices)
        pairs = indices[:even_stop].reshape((even_stop//2, 2))

        i, j = pairs.T # Original index is i, swapped index is j
        a, b = perm[i], perm[j] # Original value is a, swapped value is b
        swap_likelihood = probs[i, b] * probs[j, a] # prob of swapped data
        orig_likelihood = probs[i, a] * probs[j, b] # prob of original data
        odds = swap_likelihood / orig_likelihood # odds ratio
        swap_probs = odds / (1+odds) # Metropolis-Hasting probability of acceptance
        to_swap = np.random.random(size=len(swap_probs)) <= swap_probs # swap coin flips
        
        # Swap the chosen indices
        i, j = i[to_swap], j[to_swap]
        temp = perm.copy()
        perm[i] = perm[j]
        perm[j] = temp[i]

    return perm

def basic_hpt(dgm, idx, fit_fn, test_fold=0.2, ntrials=1000, mcmc_steps=50, **kwargs):
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Split the data into train/test
    train_fold, test_fold = create_train_test(X, test_fold)
    X_train, y_train = X[train_fold], y[train_fold]
    X_test, y_test = X[test_fold], y[test_fold]

    # Get the test MSE
    model = fit_fn(X_train, y_train)
    pred = model.predict(X_test)
    t_true = ((y_test - pred)**2).sum()

    # We only need to worry about calculating probabilities and errors for
    # the unique set of X_idx values
    unique_vals = np.unique(X_test[:,idx])

    # Get the N x M conditional probability matrix with M unique values
    probs = dgm.permutation_probs(test_fold, idx)

    # Get the N x M squared errors matrix
    preds = permutation_predictions(X_test, idx, model)
    errors = (y_test[:,None] - preds)**2

    # Get the true permutation ordering
    ranks = (X_test[:,idx:idx+1] > unique_vals[None]).sum(axis=1).astype(int)
    row_indices = np.arange(X_test.shape[0])

    # Run the basic HPT that avoids refitting on the test set
    p_value = 0
    perm = ranks.copy()
    for trial in range(ntrials):
        # Sample a null permutation
        perm = hpt_mcmc(perm, probs, errors, mcmc_steps=mcmc_steps)

        # Get the squared error
        t_null = errors[row_indices,perm].sum()

        # Add to the p-value if the null performed at least as well as the truth
        p_value += int(t_true >= t_null)

    # Return the one-sided p-value
    return (1+p_value) / (1+ntrials)


def cv_hpt(dgm, idx, fit_fn, folds=5, ntrials=1000, **kwargs):
    # Do not modify the original X
    X = np.copy(dgm.X)
    y = np.copy(dgm.y)

    # Split the data into folds (if they are not already provided)
    if isinstance(folds, int):
        folds = create_folds(X, folds)
    nfolds = len(folds)

    # Build separate p-values for each CV fold
    p_values = np.zeros(nfolds)
    for fold_idx in range(nfolds):
        # Split into train and test for this fold
        fold = folds[fold_idx]
        
        # Run the HPT on this CV fold
        p_values[fold_idx] = basic_hpt(dgm, idx, fit_fn, test_fold=fold, ntrials=ntrials, **kwargs)

    # Correct using Holm-Bonferroni method
    p_values = np.maximum.accumulate(((p_values.shape[0] - np.arange(p_values.shape[0]))*p_values[np.argsort(p_values)]).clip(0,1))
    
    # Return the smallest p-value after correction
    return p_values[0]


if __name__ == '__main__':
    # N samples, P covariates, 4 non-null, repeat nfolds indepent times, with error rate alpha
    N = 500
    P = 100
    nsamples = 10000
    ntested = 10
    nruns = 5
    alpha = 0.05
    nfolds = 5
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
    fit_fn =  fit_ols#[fit_ols, fit_bayes_ridge, fit_lasso, fit_rf, fit_xgboost, fit_keras_nn, fit_tabpfn]
    model = 'ols'
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
            p_values[run, signal_idx, method_idx] = basic_binom_hrt(dgm, signal_idx, fit_fn, test_fold=folds[0], ntrials=10)
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
            
            print(timings[run, signal_idx], '\t\t', p_values[run, signal_idx])
            print()
        print()

    import matplotlib.pyplot as plt
    import seaborn as sns
    grid = np.linspace(0,1,1000)
    print(p_values.shape)

    # Colorblind-friendly color cycle
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00',
                  'black']
    fig = plt.figure()
    for tester_idx, (label, color) in enumerate(zip(testers, CB_color_cycle)):
        cdf_signals = (p_values[:,:nsignals,tester_idx].flatten()[:,None] <= grid[None]).mean(axis=0)
        cdf_nulls = (p_values[:,nsignals:,tester_idx].flatten()[:,None] <= grid[None]).mean(axis=0)
        plt.plot(grid, cdf_signals, label=label, ls='-', color=color)
        plt.plot(grid, cdf_nulls, ls=':', color=color)
    plt.plot([0,1],[0,1], color='black')
    
    fig.legend(loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.65)   

    # plt.legend(loc='lower right', ncol=2)
    plt.savefig(f'plots/example-N{N}_p{P}_{model}_basics.pdf', bbox_inches='tight')
    plt.close()

    grid = np.linspace(0,0.05,1000)
    fig = plt.figure()
    for tester_idx, (label, color) in enumerate(zip(testers, CB_color_cycle)):
        cdf_signals = (p_values[:,:nsignals,tester_idx].flatten()[:,None] <= grid[None]).mean(axis=0)
        cdf_nulls = (p_values[:,nsignals:,tester_idx].flatten()[:,None] <= grid[None]).mean(axis=0)
        plt.plot(grid, cdf_signals, label=label, ls='-', color=color)
        plt.plot(grid, cdf_nulls, ls=':', color=color)
    plt.plot([0,0.05],[0,0.05], color='black')
    
    fig.legend(loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.65)   

    # plt.legend(loc='lower right', ncol=2)
    plt.savefig(f'plots/example-zoomed-N{N}_p{P}_{model}_basics.pdf', bbox_inches='tight')
    plt.close()
       
























