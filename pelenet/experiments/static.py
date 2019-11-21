# Official modules
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tabulate import tabulate
from copy import deepcopy

# Own modules
from parameters import Parameters
from utils import Utils
from plots import Plot
from network.reservoir.reservoir import ReservoirNetwork

"""
@desc: Class for running an experiment, usually contains performing
       several networks (e.g. for training and testing)
"""
class Experiment():

    """
    @desc: Initiates the experiment
    """
    def __init__(self):
        self.p = Parameters()  # Parameters

        self.net = None
        self.trainSpikes = None

        # Instantiate utils and plot
        self.utils = Utils()
        self.plot = Plot(self)
    
    """
    @desc: Build network
    """
    def build(self):
        # Instanciate innate network
        self.net = ReservoirNetwork()

        # Draw random mask and weights
        self.drawMaskAndWeights()

        # Build the network structure
        self.net.build()

        # Plot histogram of weights and calc spectral radius
        self.net.plot.initialExWeightDistribution()

        # Plot weight matrix
        self.net.plot.initialExWeightMatrix()
    """
    @desc: Run network
    """
    def run(self):
        # Run network
        self.net.run()

    """
    @desc: Draw mask and weights
    """
    def drawMaskAndWeights(self):
        # Draw and store mask matrix
        nAll = self.p.reservoirExSize + self.p.reservoirInSize
        mask = self.net.drawSparseMaskMatrix(self.p.reservoirDens, nAll, nAll, avoidSelf=True)

        # Draw and store weight matrix
        self.net.drawSparseWeightMatrix(mask)
    
    """
    @desc: Call several function in order to evaluate results
    """
    def evaluateExperiment(self):
        idx = self.p.cueSteps + self.p.relaxationSteps

        # Prepare spikes for evaluation
        self.trainSpikes = np.array([net.reservoirProbe[2].data for net in self.trainNets])
        trainSpikesMean = np.mean(self.trainSpikes, axis=0)
        self.testSpikes = self.testNet.reservoirProbe[2].data

        # Plot autocorrelation function
        self.plot.autocorrelation(trainSpikesMean[:,idx:], numNeuronsToShow=1)
        # TODO is it a good idea to mean over trials? maybe use index selection like for fano factors

        # Plot crosscorrelation function
        self.plot.crosscorrelation(self.testSpikes)

        # Plot spike missmatch
        self.plot.spikesMissmatch(self.trainSpikes[i,:,fr:to], self.testSpikes[:,fr:to])

        # Plot fano factors of spike counts (test spikes)
        self.plot.ffSpikeCounts(self.testSpikes, neuronIdx = [1,5,10,15])

        # Plot first 2 components of PCA of all spikes 
        self.plot.pca(self.testSpikes)

        # Evalulate assembly weights
        self.evaluateAssemblyWeights()

        # Plot pre synaptic trace
        self.trainNet.plot.preSynapticTrace()
