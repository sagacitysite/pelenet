import numpy as np
from scipy import sparse

"""
Calculate spectral radius of whole weight matrix
"""
def getSpectralRadius(self, weights):
    # Stack top and bottom row of weight matrix horizontally
    top = sparse.hstack([weights.exex, weights.inex])
    bottom = sparse.hstack([weights.exin, weights.inin])

    # Stack vertically
    wgs = sparse.vstack([top, bottom])

    # Calculate and return rounded spectral radius
    maxeigval = np.absolute(sparse.linalg.eigs(wgs.asfptype() / 255., k=1, which='LM', return_eigenvectors=False)[0])
    return np.round(maxeigval*1000)/1000.

"""
Recombine weight matrix from excitatory probe chunks
"""
def recombineExWeightMatrix(self, initialExWeights, exWeightProbes):
    # Get shorthand for some variables
    init = initialExWeights
    nPerCore = self.p.neuronsPerCore
    # Array which finally contains all weight matrices
    weightMatrices = []
    # Iterate over number of probes over time
    numProbes = len(exWeightProbes[0][0][0][0].data)
    for p in range(numProbes):
        # Calculate trained weight matrix from weight probes
        weightMatrix = []
        # Iterate over connection chunks between cores
        n, m = np.shape(exWeightProbes)
        for i in range(n):
            # Define from/to indices for indexing
            ifr, ito = i*nPerCore, (i+1)*nPerCore
            chunks = []
            for j in range(m):
                # Define from/to indices for indexing
                jfr, jto = j*nPerCore, (j+1)*nPerCore
                # Get number of synapses in current probe
                numSyn = np.shape(exWeightProbes[i][j])[0]
                # Iterate over number of synapses in current probe (connections from one core to another)
                data = []
                for k in range(numSyn):
                    # Get weights data from probe index p and append to data array
                    data.append(exWeightProbes[i][j][k][0].data[p])
                # Get chunk from initial matrix for defining sparse matrix of the current chunk (need indices and index pointer)
                ic = init[jfr:jto, ifr:ito]
                # Define sparse matrix, using initial weight matrix indices and index pointerm, as well as shape of chunk
                chunks.append(sparse.csr_matrix((data, ic.indices, ic.indptr), shape=np.shape(ic)))
            # Stack list of chunks together to column
            column = sparse.vstack(chunks)
            # Append column to weight matrix
            weightMatrix.append(column)
        # Stack list of columns together to the whole trained weight matrix
        wmcsr = sparse.hstack(weightMatrix).tocsr()  # transform to csr, since stacking returns coo format
        # Add weight matrix of current 
        weightMatrices.append(wmcsr)

    return weightMatrices

"""
@desc: Get mask of support weights for every cluster in the assembly
@return: Mask of the bottom-left area of the matrix
"""
def getSupportWeightsMask(self, exWeightMatrix):
    nCs = self.p.inputNumTargetNeurons
    nEx = self.p.reservoirExSize
    nC = self.p.inputAlternatingNum
    matrix = exWeightMatrix

    # Get areas in matrix
    #left = matrix[:,:nC*nCs].toarray()  # left
    #top = matrix[:nC*nCs,:].toarray()  # top
    #bottom = matrix[nC*nCs:,:].toarray()  # bottom
    bottomLeft = matrix[nC*nCs:,:nC*nCs].toarray()  # bottom-left

    # Get single cluster colums in bottom-left area (candidates for support weights)
    cols = np.array([ bottomLeft[:,i*nCs:(i+1)*nCs] for i in range(nC)])

    # Calculate means for every column in bottom-left
    col_rowmeans = np.array([np.mean(cols[i], axis=1) for i in range(nC)])

    # Condition 1: Get only rows their mean is greater than total mean
    greaterMeanIndices = col_rowmeans > np.mean(bottomLeft)

    # Condition 2: Get for every row the column which has max value
    col_argmax = np.argmax(col_rowmeans, axis=0)
    maxRowIndices = np.array(col_argmax[:,None] == range(nC)).T

    # Get final mask in combining both conditions
    supportMasks = np.logical_and(greaterMeanIndices, maxRowIndices)
    
    # Create a "false" column, which is necessary if only one column (one input) exists
    falseCol = np.zeros((supportMasks.shape[1])).astype(bool)

    # Get mask for other neurons
    othersMask = np.logical_not(np.logical_or(falseCol, *supportMasks))

    # Combine masks for support neurons and other neurons
    return np.array([*supportMasks, othersMask])

"""
@desc: Get support masks for weight matrices of all trials
"""
def getSupportMasksForAllTrials(self, initialweightsExex, trainedWeightsExex):
    supportMasks = []

    # First add initial weights
    swm = self.getSupportWeightsMask(initialweightsExex)
    supportMasks.append(swm)

    # Add all trained weight matrices 
    for i in range(len(trainedWeightsExex)):
        swm = self.getSupportWeightsMask(trainedWeightsExex[i])
        supportMasks.append(swm)

    return np.array(supportMasks)

"""
@desc: Get turnovers of support neurons
"""
def getSupportNeuronTurnovers(self, supportMasks):
    numTrials = np.shape(supportMasks)[0]-1
    turnover = []

    # Get turonover between all trials
    for i in range(numTrials):
        # Get difference between support masks
        diff = np.subtract(supportMasks[i].astype(int), supportMasks[i+1].astype(int))
        # Absolute sum of difference
        turnover.append(np.sum(np.abs(diff), axis=1))
    
    return np.array(turnover).T
