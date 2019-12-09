# Official modules
import numpy as np
import logging
from copy import deepcopy
import gc

# Importing anisotropic network
from .anisotropic import AnisotropicExperiment

# Own modules
from ..network.reservoir.reservoir import ReservoirNetwork

"""
@desc: Class for running an experiment, usually contains performing
       several networks (e.g. for training and testing)
"""
class ReadoutExperiment(AnisotropicExperiment):

    """
    @desc: Initiates the experiment
    """
    def __init__(self):
        # Init super object
        super().__init__()

        # Define some further variables
        self.targetFunction = self.getTargetFunction()

    """
    @desc: Run whole experiment
    """
    def run(self):
        # Run network
        self.net.run()

    """
    @desc: Build all networks
    """
    def build(self):
        # Instanciate innate network
        self.net = ReservoirNetwork(self.p)
        self.net.landscape = None

        # Draw anisotropic mask and weights
        self.drawMaskAndWeights()

        # Connect network
        self.net.addReservoirNetworkDistributed()

        # Add cue
        self.net.addRepeatedCueGenerator()

        # Add stop generator
        self.net.addRepeatedStopGenerator()

        # Add background noise
        #self.net.addNoiseGenerator()

        # Build the network structure
        self.net.build()

    """
    @desc: Define function to learn as ouput
    @params:
            clusterIndex: index of cluster the function is defined for
            type: 'sin', 'revsin', 'lin'
    """
    def getTargetFunction(self, type = 'sin'):
        nTs = self.p.movementSteps

        # Define function values
        if type == 'sin': return 0.5+0.5*np.sin((np.pi/(0.5*nTs))*np.arange(nTs))
        elif type == 'revsin': return 0.5-0.5*np.sin((np.pi/(0.5*nTs))*np.arange(nTs))
        elif type == 'lin': return np.concatenate((0.5-(1/nTs)*np.arange(nTs/2), (1/nTs)*np.arange(nTs/2)))
        else: raise ValueError('Chosen function type is not available')
    
    """
    TODO: Shift to plots/misc.py
    @desc: Plots either target function or estimated function
    """
    def plotActionSequenceFunction(self, y, title="Target function"):
        # Plot function
        plt.figure(figsize=(16, 4))
        plt.title(title)
        plt.xlabel('Time')
        for i in range(self.p.traceClusters):
            fr, to = i*self.p.traceSteps, (i+1)*self.p.traceSteps
            plt.plot(np.arange(fr, to), y[i], label="Cluster"+str(i))
        plt.legend()
        p = plt.show()
