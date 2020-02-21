import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

"""
@desc: Plot weight distribution and show spectral radius
"""
def weightDistribution(self, weights, yscale):
    # Get only data from sparse matrix (equals flattend matrix)
    wsf = weights.data

    # Calculate spectral radius (largest eigenvalue)
    maxeigval = np.absolute(sparse.linalg.eigs(weights.asfptype() / 255., k=1, which='LM', return_eigenvectors=False)[0])
    maxeigval_round = np.round(maxeigval*1000)/1000.

    # Calculate mean weight from non zero weights
    wnonzero = wsf[np.nonzero(wsf)]
    meanweight = np.round(np.mean(wnonzero)*1000)/1000.

    # Reservoir weights histogram
    plt.figure(figsize=(16, 4))
    plt.xlim((0, np.max(wsf)))
    plt.xlabel('weights')
    plt.ylabel('#')
    plt.title('Spectral radius: ' + str(maxeigval_round) + ', Mean weight: ' + str(meanweight))
    plt.yscale(yscale)
    plt.savefig(self.plotDir + 'weights_distribution.' + self.p.pltFileType)
    p = plt.hist(wsf[np.nonzero(wsf)], bins=np.arange(np.max(wsf)))

"""
@desc: Plot weight distribution of initial excitatory weights and show spectral radius
"""
def initialExWeightDistribution(self, yscale="linear"):
    weightDistribution(self, self.obj.initialWeights.exex, yscale)

"""
@desc: Plot weight distribution of initial excitatory weights and show spectral radius
"""
def trainedExWeightDistribution(self, yscale="linear"):
    weightDistribution(self, self.obj.trainedWeightsExex, yscale)

"""
@desc: Plot weight matrix
"""
def weightMatrix(self, sparseMatrix):
    # If weight matrix is too large, matrix should not be transformed into dense matrix
    if sparseMatrix.shape[0] > self.p.maxSparseToDenseLimit:
        warnings.warn('excitatoryWeightMatrix was not plotted, since weight matrix is too large')
        return None

    denseMatrix = sparseMatrix.toarray()

    max_weight = 20
    plt.figure(figsize=(6, 6))
    plt.imshow(denseMatrix, vmin=0, vmax=max_weight, interpolation=None)
    plt.title('Initial weights')
    plt.savefig(self.plotDir + 'weights_matrix.' + self.p.pltFileType)
    p = plt.colorbar()

"""
@desc: Plot initial excitatory weight matrix
"""
def initialExWeightMatrix(self):
    weightMatrix(self, self.obj.initialWeights.exex)

"""
@desc: Plot trained excitatory weight matrix
"""
def trainedExWeightMatrix(self):
    weightMatrix(self, self.obj.trainedWeightsExex)

"""
@desc: Plots the whole weight matrix with weights sorted according to support weights
"""
def weightsSortedBySupport(self, mask):
    nCs = self.p.traceClusterSize
    nC = self.p.traceClusters

    matrix = self.obj.initialWeights.exex
    top = matrix[:nC*nCs,:]  # top
    bottom = matrix[nC*nCs:,:]  # bottom

    # Get sorted indices
    indices = np.lexsort(mask[::-1])[::-1]

    # Define sorted matrix
    sorted_matrix = np.zeros(matrix.shape)

    # Fill
    sorted_matrix[:nC*nCs,:] = top  # fill top
    sorted_matrix[nC*nCs:,:] = bottom[indices,...] # fill bottom

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(sorted_matrix, interpolation=None, vmin=0, vmax=20)
    plt.title('Sorted weights')
    plt.colorbar()
    plt.savefig(self.plotDir + 'weights_support.' + self.p.pltFileType)
    p = plt.show()

"""
@desc: Shows average input weights for cue input, especially show were input is given in topology
"""
def cueWeightMean(self):
    # Define some variables
    topsize = int(np.sqrt(self.p.reservoirExSize))

    # Get average over cue input weights 
    inputWeightMean = np.mean(self.obj.cueWeights, axis=1)
    # Reshape to topology
    wgt = inputWeightMean.reshape((topsize,topsize))
    # Show and save result
    plt.savefig(self.plotDir + 'weights_cue_mean.' + self.p.pltFileType)
    p = plt.imshow(wgt)
