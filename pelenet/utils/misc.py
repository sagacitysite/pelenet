import numpy as np
import scipy.linalg as la
from scipy.stats.stats import pearsonr
import statsmodels.api as sm
from types import SimpleNamespace

from lib.helper.exceptions import ArgumentNotValid

"""
@desc: Trains ordinary least square model, includes filtering and regularization
@pars: trainSpikes: has dimensions N (number of neurons) x T (number of time steps per trial) x B (number of trials)
       testSpikes: has dimensions N (number of neurons) x T (number of time steps per trial)
       targetFunction: has dimensions T (number of time steps per trial)
       filter: filter method as string, can be: 'single exponential', 'double exponential' or 'gaussian' (symmetric)
"""
def trainOLS(self, trainSpikes, testSpikes, targetFunction, filter='single exponential'):

    # Preprocess if B i does not exist
    if (len(np.shape(trainSpikes)) == 2):
        trainSpikes = trainSpikes[:,:,np.newaxis]

    # Get shapes (N: num neurons, T: num time steps, B: num trials)
    N, T, B = np.shape(trainSpikes)
    Nt, Tt = np.shape(testSpikes)

    # Some checks
    if (len(targetFunction) != T):
        raise ArgumentNotValid('Length of target function and length of train spikes is not equal.')
    if (len(targetFunction) != Tt):
        raise ArgumentNotValid('Length of target function and length of test spikes is not equal.')
    if (Nt != N or Tt != T):
        raise ArgumentNotValid('Number of neurons or number of time steps in train and test spikes is not equal.')

    # Get filtered spike trains for train and test spikes
    x = self.getFilteredSpikes(trainSpikes.reshape(N, T*B), filter)
    xe = self.getFilteredSpikes(testSpikes, filter)

    # Get target function for all trials
    y = np.tile(targetFunction, B)

    # Train the parameters
    model = sm.OLS(y, x.T)
    params = model.fit().params
    #params = model.fit_regularized().params  # https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit_regularized.html

    # Estimate target function for test spike train
    ye = np.dot(xe.T, params)

    # Calculate performance
    mse = np.mean(np.square(y - ye))  # MSE error
    cor = pearsonr(y, ye)[0]  # Pearson correlaton coefficient

    # Join performance measures
    performance = SimpleNamespace(**{ 'mse': mse, 'cor': cor })

    return params, ye, performance

"""
@desc: Caluclates PCA out of given data
@pars: data as 2D NumPy array
@return: data transformed in 2 dims/columns + regenerated original data
@link: https://stackoverflow.com/a/13224592/2692283
"""
def pca(data, dims_rescaled_data=2):
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = la.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T, evals, evecs
