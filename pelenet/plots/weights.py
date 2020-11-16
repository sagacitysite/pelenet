import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

"""
@desc: Plot weight distribution and show spectral radius
"""
def weightDistribution(self, weights, yscale="linear", title=True, figsize=None):
    # Get only data from sparse matrix (equals flattend matrix)
    wsf = weights.data

    # Calculate spectral radius (largest eigenvalue)
    maxeigval = np.absolute(sparse.linalg.eigs(weights.asfptype() / 255., k=1, which='LM', return_eigenvectors=False)[0])
    maxeigval_round = np.round(maxeigval*1000)/1000.

    # Calculate mean weight from non zero weights
    wnonzero = wsf[np.nonzero(wsf)]
    meanweight = np.round(np.mean(wnonzero)*1000)/1000.

    # Reservoir weights histogram
    if figsize is not None: plt.figure(figsize=figsize)
    plt.xlim((0, np.max(wsf)))
    plt.xlabel('size of weight')
    plt.ylabel('frequency')
    if title: plt.title('Spectral radius: ' + str(maxeigval_round) + ', Mean weight: ' + str(meanweight))
    plt.yscale(yscale)
    plt.hist(wsf[np.nonzero(wsf)], bins=np.arange(np.max(wsf)))
    plt.savefig(self.plotDir + 'weights_distribution.' + self.p.pltFileType)
    p = plt.show()

"""
@desc: Plot weight distribution of initial excitatory weights and show spectral radius
"""
def initialExWeightDistribution(self, *args, **kwargs):
    weightDistribution(self, self.obj.initialWeights.exex, *args, **kwargs)

"""
@desc: Plot weight distribution of initial excitatory weights and show spectral radius
"""
def trainedExWeightDistribution(self, *args, **kwargs):
    weightDistribution(self, self.obj.trainedWeightsExex, *args, **kwargs)

"""
@desc: Plot weight matrix
"""
def weightMatrix(self, sparseMatrix, filename, title):
    # If weight matrix is too large, matrix should not be transformed into dense matrix
    if sparseMatrix.shape[0] > self.p.maxSparseToDenseLimit:
        warnings.warn('excitatoryWeightMatrix was not plotted, since weight matrix is too large')
        return None

    denseMatrix = sparseMatrix.toarray()

    max_weight = 20
    plt.figure(figsize=(6, 6))
    plt.imshow(denseMatrix, vmin=0, vmax=max_weight, interpolation=None)
    plt.title(title)
    plt.colorbar()
    plt.savefig(self.plotDir + 'weights_'+filename+'.' + self.p.pltFileType)
    p = plt.show()

"""
@desc: Plot initial excitatory weight matrix
"""
def initialExWeightMatrix(self):
    weightMatrix(self, self.obj.initialWeights.exex, 'initial', 'Initial weights')

"""
@desc: Plot trained excitatory weight matrix
"""
def trainedExWeightMatrix(self, trainedWeightsExex=None):
    if trainedWeightsExex is None:
        # If no trained weight matrix for plotting is given, plot the last one
        weightMatrix(self, self.obj.trainedWeightsExex[-1], 'trained', 'Trained weights')
    else:
        # If a specific matrix is given, plot it
        weightMatrix(self, trainedWeightsExex, 'trained', 'Trained weights')

"""
@desc: Plots the whole weight matrix with weights sorted according to support weights
@params:
    supportMask: is calculated by getSupportWeightsMask() in utils
"""
def weightsSortedBySupport(self, supportMask, trainedWeightsExex):
    nCs = self.p.inputNumTargetNeurons
    nC = self.p.inputVaryNum

    # Get parts to sort of weight matrix
    top = trainedWeightsExex[:nC*nCs,:].toarray()  # top
    bottom = trainedWeightsExex[nC*nCs:,:].toarray()  # bottom

    # Remove no-support neurons
    supportMask = supportMask[:-1]

    # Get sorted indices
    indices = np.lexsort(supportMask[::-1])[::-1]

    # Define sorted matrix
    sortedMatrix = np.zeros(trainedWeightsExex.shape)

    # Fill
    sortedMatrix[:nC*nCs,:] = top  # fill top
    sortedMatrix[nC*nCs:,:] = bottom[indices,...] # fill bottom

    # Plot
    weightMatrix(self, sparse.csr_matrix(sortedMatrix), 'sorted', 'Sorted weights')
