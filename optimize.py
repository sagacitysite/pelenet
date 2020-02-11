import os
import sys
import numpy as np
import statsmodels.api as sm
from scipy.stats.stats import pearsonr

from pelenet.utils import Utils

utils = Utils.instance()

# Get dir and file
dirPath = os.path.dirname(os.path.realpath(__file__))
filePath = dirPath + '/data_2020-02-06_19-10.npy'

# Load data
data = np.load(filePath)

# Define paramaters to scan for
alphas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
weights = [0.0005, 0.001, 0.002, 0.005, 0.01]
#alphas = [0.001]
#weights = [0.0005]

# Define empty arrays to fill
cors = np.zeros((len(alphas), len(weights)))
numZeros = np.zeros((len(alphas), len(weights)))
yes = []

# Define training and testing data
trainSpikes = data[:24]
testSpikes = data[24]
#trainSpikes = data[:2]
#testSpikes = data[1]

targetFunction = 0.5+0.5*np.sin((np.pi/(0.5*400))*np.arange(400))

# Get shapes of data
B, N, T = np.shape(trainSpikes)
Nt, Tt = np.shape(testSpikes)

# Get filtered data
x = np.array([utils.getFilteredSpikes(trainSpikes[i,...], 'single exponential') for i in range(B)]).reshape(N, T*B)
xe = utils.getFilteredSpikes(testSpikes, 'single exponential')
print('Filtering finished')

# Get target function for all trials
y = np.tile(targetFunction, B)

# Iterate over alphas
idx = 0
allSteps = len(alphas)*len(weights)
for i, a in enumerate(alphas):
    _ye = []
    # Iterate over l1 weights
    for j, w in enumerate(weights):
        idx += 1
        print('({}/{}) a = {}, w = {}'.format(idx, allSteps, a, w))
        # Estimate
        model = sm.OLS(y, x.T)
        params = model.fit_regularized(alpha=a, L1_wt=w).params
        ye = np.dot(xe.T, params)

        # Fill arrays
        cors[i,j] = pearsonr(targetFunction, ye)[0]
        numZeros[i,j] = len(params[params==0])
        _ye.append(ye)
    yes.append(_ye)

yes = np.array(yes)

res = { 'correlations': cors, 'zeropars': numZeros, 'yestimated': yes }

np.save(dirPath+'res.npy', res)
