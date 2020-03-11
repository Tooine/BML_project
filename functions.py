### BML - Project 

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as sps
import scipy.optimize as spo
import sklearn.gaussian_process as sklgp
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
import time

def eval_objective(target, x, nu=0, countTime=False):
    x = np.asarray(x)
    n = x.shape[0]
    eval = []
    Times = []
    for i in range(x.shape[0]):
        begin = time.time()
        eval.append(target(x[i]) + nu * npr.rand())
        end = time.time()
        Times.append(end-begin)
    if countTime:
        return np.asarray(eval).reshape(n, 1), np.asarray(Times).reshape(n, 1)
    return np.asarray(eval).reshape(n, 1)

def generate_data(target, nu, N0, space, seed, countTime=False):
    """Generate simulated data."""
    npr.seed(seed)
    X = random_point(N0, space)
    X = np.sort(X, axis=0)
    if countTime:
        y, t = eval_objective(target, X, nu=nu, countTime=countTime)
        return X, y, t
    y = np.array(eval_objective(target, X, nu=nu, countTime=countTime))
    return X, y

def random_point(n, space):
    X = np.zeros((n, space.shape[0]))
    for i in range(space.shape[0]):
        X[:, i] = npr.uniform(space[i, 0], space[i, 1], n)
    return X

class BayesianOptimizationEI:
    '''The classical BO with GP approach, with EI as acquisition function. '''

    def __init__(self, X, y, space, kernel=Matern(nu=5/2)):
        self.X, self.y = X.copy(), y.copy() # Initial training set
        self.sample_size = X.shape[0]
        self.dimension = X.shape[1]
        self.space = space
        
        self.gp = sklgp.GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    def fit(self):
        self.gp.fit(self.X, self.y)

    def predict(self, X_test):
        y_mean, y_std = self.gp.predict(X_test, return_std=True)
        return y_mean, y_std.reshape(y_mean.shape)

    def sample_y(self, X_test, random_state):
        return self.gp.sample_y(X_test, 1, random_state=random_state)

    def acquisition_criterion(self, X_test):
        """implement of acquisition criterion: EI"""
        m = np.min(self.y) # current minimum value
        y_mean, y_std = self.predict(X_test)
        u = (m - y_mean)/(y_std)
        ei = y_std * (u*sps.norm.cdf(u) + sps.norm.pdf(u))
        return ei

    def find_next_point(self, number_of_restarts):
        """maximize acquisition criterion"""
        # Set things up
        criterion = lambda x: -self.acquisition_criterion(x.reshape((1, self.dimension))).flatten()[0]

        # Optimize once and then restart
        x0 = random_point(1, self.space)
        res = spo.minimize(criterion, x0=x0, bounds=self.space)
        xbest, fbest = [res[key] for key in ['x', 'fun']]
        for _ in range(number_of_restarts):
            # Restart strategy to avoid local behaviour of the optimizer
            x0 = random_point(1, self.space)
            res = spo.minimize(criterion, x0=x0, bounds=self.space)
            xopt, fopt = [res[key] for key in ['x', 'fun']]
            if fopt < fbest:
                # the current restart has found a better point
                xbest, fbest = xopt, fopt

        return xbest

    def update(self, x, y):
        self.X = np.concatenate((self.X, x.reshape(1, self.dimension)), 0)
        self.y = np.concatenate((self.y, y.reshape(1, 1)), 0)


class BayesianOptimizationEI_MCMC:
    '''Strategy 1: the kernel parameters are seen with a Bayesian point of view. The acquisition function is the integrated EI criterion (see report). '''

    def __init__(self, X, y, space, kernel=Matern(nu=5/2)):
        self.X, self.y = X.copy(), y.copy() # Initial training set
        self.sample_size = X.shape[0]
        self.dimension = X.shape[1]
        self.space = space

        self.gp = sklgp.GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    def fit(self):
        self.gp.fit(self.X, self.y)

    def predict(self, X_test):
        y_mean, y_std = self.gp.predict(X_test, return_std=True)
        return y_mean, y_std.reshape(y_mean.shape)

    def sample_y(self, X_test, random_state):
        return self.gp.sample_y(X_test, 1, random_state=random_state)

    def acquisition_criterion_theta(self, X_test, theta):
        """implement of acquisition criterion EI with fixed kernel parameters"""
        m = np.min(self.y) # current minimum value
        self.gp.kernel_.theta = theta
        y_mean, y_std = self.predict(X_test)
        u = (m - y_mean)/(y_std+1e-10)
        ei = y_std * (u*sps.norm.cdf(u) + sps.norm.pdf(u))
        return ei
    
    def random_theta(self):
        ## TODO: here we need to compute the posterior distribution ... in the following we do it uniquely with a uniform distribution ...
        size = self.gp.kernel_.theta.shape[0]
        theta = np.zeros(size)
        for s in range(size):
            theta[s] = npr.uniform(bo.gp.kernel_.hyperparameters[s].bounds[0, 0], bo.gp.kernel_.hyperparameters[s].bounds[0, 1])
        return theta
    
    def acquisition_criterion(self, X_test):
        """implement of acquisition criterion: the integrated EI (see report). """
        
        pass

    def find_next_point(self, number_of_restarts):
        """maximize acquisition criterion"""
        # Set things up
        criterion = lambda x: -self.acquisition_criterion(x.reshape((1, self.dimension))).flatten()[0]

        # Optimize once and then restart
        x0 = random_point(1, self.space)
        res = spo.minimize(criterion, x0=x0, bounds=self.space)
        xbest, fbest = [res[key] for key in ['x', 'fun']]
        for _ in range(number_of_restarts):
            # Restart strategy to avoid local behaviour of the optimizer
            x0 = random_point(1, self.space)
            res = spo.minimize(criterion, x0=x0, bounds=self.space)
            xopt, fopt = [res[key] for key in ['x', 'fun']]
            if fopt < fbest:
                # the current restart has found a better point
                xbest, fbest = xopt, fopt

        return xbest

    def update(self, x, y):
        self.X = np.concatenate((self.X, x.reshape(1, self.dimension)), 0)
        self.y = np.concatenate((self.y, y.reshape(1, 1)), 0)

class BayesianOptimizationEIperS:
    '''Strategy 2: we replace acquisition function EI by EI per second. '''
    
    def __init__(self, X, y, t, space, kernel=Matern(nu=5/2)):
        self.X, self.y = X.copy(), y.copy() # Initial training set
        self.sample_size = X.shape[0]
        self.dimension = X.shape[1]
        self.space = space
        self.lnTime = np.log(t)     # the time values are stored at a log scale.

        self.gp_target = sklgp.GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.gp_lnTime = sklgp.GaussianProcessRegressor(kernel=kernel, normalize_y=True)    # maybe we can use two different kernels

    def fit(self):
        self.gp_target.fit(self.X, self.y)
        self.gp_lnTime.fit(self.X, self.lnTime)

    def predict_target(self, X_test):
        y_mean, y_std = self.gp_target.predict(X_test, return_std=True)
        return y_mean, y_std.reshape(y_mean.shape)
        
    def predict_lnTime(self, X_test):
        t_mean, t_std = self.gp_lnTime.predict(X_test, return_std=True)
        return t_mean, t_std.reshape(t_mean.shape)
    
    def sample_y(self, X_test, random_state):
        return self.gp_target.sample_y(X_test, 1, random_state=random_state)
    
    def sample_lnt(self, X_test, random_state):
        return self.gp_lnTime.sample_y(X_test, 1, random_state=random_state)

    def acquisition_criterion(self, X_test):
        """implement of acquisition criterion: EI per sec."""
        m = np.min(self.y) # current minimum value
        y_mean, y_std = self.predict_target(X_test)
        u = (m - y_mean)/(y_std+1e-10)
        ei = y_std * (u*sps.norm.cdf(u) + sps.norm.pdf(u))
        
        lnt_mean, _ = self.predict_lnTime(X_test)
        return ei / np.exp(lnt_mean)

    def find_next_point(self, number_of_restarts=3):
        """maximize acquisition criterion."""
        # Set things up
        criterion = lambda x: -self.acquisition_criterion(x.reshape((1, self.dimension))).flatten()[0]

        # Optimize once and then restart
        x0 = random_point(1, self.space)
        res = spo.minimize(criterion, x0=x0, bounds=self.space)
        xbest, fbest = [res[key] for key in ['x', 'fun']]
        for _ in range(number_of_restarts):
            # Restart strategy to avoid local behaviour of the optimizer
            x0 = random_point(1, self.space)
            res = spo.minimize(criterion, x0=x0, bounds=self.space)
            xopt, fopt = [res[key] for key in ['x', 'fun']]
            if fopt < fbest:
                # the current restart has found a better point
                xbest, fbest = xopt, fopt

        return xbest

    def update(self, x, y, t):
        self.X = np.concatenate((self.X, x.reshape(1, self.dimension)), 0)
        self.y = np.concatenate((self.y, y.reshape(1, 1)), 0)
        self.lnTime = np.concatenate((self.lnTime, np.log(t).reshape(1, 1)), 0)
