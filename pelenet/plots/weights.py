import warnings
import string
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

"""
@desc: Plot weight distribution and show spectral radius
"""
def weightDistribution(self, weights, yscale="linear", title=True, figsize=None, xlim=None):
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
    if xlim is None:
        plt.xlim((0, np.max(wsf)))
    else:
        plt.xlim(xlim)
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
    nC = self.p.inputAlternatingNum

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

"""
@desc:  Share of support neurons
"""
def supportShare(self, supportMasks, figsize=(10,4), filename=None):
    # Calculate share of support neurons
    supportShare = np.array([ np.sum(supportMasks[i], axis=1) for i in range(self.p.trials+1)]).T

    # Number of background neurons (potential support neurons) 
    numBackroundNeurons = np.shape(supportMasks)[2]

    # x values for plot
    xvals = np.arange(1,self.p.trials+1)

    # Letters A to Z
    alphabet = list(string.ascii_uppercase)

    # Set size of figure
    plt.figure(figsize=figsize)

    # Plot support share for all clusters
    for i in range(np.shape(supportShare)[0]-1):
        plt.plot(supportShare[i]/numBackroundNeurons, label=alphabet[i])
    # Plot support share for not assigned neurons
    plt.plot(supportShare[-1]/numBackroundNeurons, label='not assigned')
    # Add legend
    plt.legend()

    # Set limits and labels
    plt.ylim((0,1))
    plt.xlim((0,self.p.trials))
    plt.ylabel('support neuron share')
    plt.xlabel('trials', labelpad=25)
    plt.xticks(xvals)

    # Add input labels to x-axis indicating stimulated cluster
    inputs = self.utils.getInputLetterList(self.obj.inputTrials)
    for i in range(self.p.trials):
        plt.text(i+0.845, -0.165, inputs[i])

    # Save plot in datalog
    filename = '_'+filename if filename is not None else ''
    plt.savefig(self.plotDir + 'weights_support_share' + filename + '.' + self.p.pltFileType)

    # Show plot
    pl = plt.show()

"""
@desc:  Turnovers in support neurons
"""
def supportTurnover(self, supportMasks, figsize=(10,4), filename=None):

    # Get support turnovers
    turnover = self.utils.getSupportNeuronTurnovers(supportMasks)

    # x values for plot
    xvals = np.arange(1,self.p.trials+1)

    # Letters A to Z
    alphabet = list(string.ascii_uppercase)

    # Set size of figure
    plt.figure(figsize=figsize)

    # Plot support share for all clusters
    for i in range(np.shape(turnover)[0]-1):
        plt.plot(xvals, turnover[i], label=alphabet[i])
    # Plot support share for not assigned neurons
    plt.plot(xvals, turnover[-1], label='not assigned')
    # Add legend
    plt.legend()

    # Set limits and labels
    plt.ylim((0,50))
    plt.xlim((1,self.p.trials+1))
    plt.ylabel('support neuron turnover')
    plt.xlabel('trials', labelpad=25)
    plt.xticks(xvals)
    
    # Add input labels to x-axis indicating stimulated cluster
    inputs = self.utils.getInputLetterList(self.obj.inputTrials)
    for i in range(self.p.trials):
        plt.text(i+0.845, -8, inputs[i])

    # Save plot in datalog
    filename = '_'+filename if filename is not None else ''
    plt.savefig(self.plotDir + 'weights_support_turnover' + filename + '.' + self.p.pltFileType)

    # Show plot
    pl = plt.show()
