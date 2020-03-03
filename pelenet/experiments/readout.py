# Loihi modules
import nxsdk.api.n2a as nx

# Official modules
import numpy as np
import logging
from copy import deepcopy
import os

# Pelenet modules
from ..system import System
from ..system.datalog import Datalog
from ..parameters import Parameters
from ..utils import Utils
from ..plots import Plot
from .anisotropic import AnisotropicExperiment
from ..network import ReservoirNetwork

"""
@desc: Class for running an experiment, usually contains performing
       several networks (e.g. for training and testing)
"""
class ReadoutExperiment(AnisotropicExperiment):

    """
    @desc: Initiates the experiment
    """
    def __init__(self):
        # Parameters
        self.p = Parameters(update = self.updateParameters())

        self.net = None

        # Instantiate system singleton and add datalog object
        self.system = System.instance()
        datalog = Datalog(self.p)
        self.system.setDatalog(datalog)

        # Instantiate utils and plot
        self.utils = Utils.instance()
        self.plot = Plot(self)

        # Define some further variables
        self.targetFunction = self.getTargetFunction()

    """
    @desc: Overwrite parameters for this experiment
    """
    def updateParameters(self):
        # Update patameters from parent
        p = super().updateParameters()

        return {
            # Parameters from parent
            **p,
            # Experiment
            'trials': 25,
            'stepsPerTrial': 100,
            # Probes
            'isExSpikeProbe': True,
            'isOutSpikeProbe': True
        }
        # Experiment
        #self.p.trials = 25
        #self.p.stepsPerTrial = 100

        # Probes
        #self.p.isExSpikeProbe = True
        #self.p.isOutSpikeProbe = True
    
    """
    @desc: Build all networks
    """
    def build(self):
        # Instanciate innate network
        self.net = ReservoirNetwork(self.p)
        self.net.landscape = None

        # Draw anisotropic mask and weights
        self.drawMaskAndWeights()

        # Draw output weights
        self.net.drawOutputMaskAndWeights()

        # Connect ex-in reservoir
        self.net.connectReservoir()

        # Connect reservoir to output
        self.net.connectOutput()

        # Add cue
        self.net.addRepeatedPatchGenerator()

        # Add background noise
        #self.net.addNoiseGenerator()

        # Build the network structure
        self.net.build()
    
    """
    @desc: Run whole experiment
    """
    def run(self):
        # Compile network
        compiler = nx.N2Compiler()
        board = compiler.compile(self.net.nxNet)
        logging.info('Network successfully compiled')

        # Add snips and channel
        resetInitSnips = self.net.addResetSnips(board)  # add snips
        resetInitChannels = self.net.createAndConnectResetInitChannels(board, resetInitSnips)  # create channels for transfering initial values for the reset SNIP
        
        # Start board
        board.start()
        logging.info('Board successfully started')

        # Write initial data to channels
        for i in range(self.p.numChips):
            resetInitChannels[i].write(3, [
                self.p.neuronsPerCore,  # number of neurons per core
                self.p.totalTrialSteps,  # reset interval
                self.p.resetSteps  # number of steps to clear voltages/currents
            ])
        logging.info('Initial values transfered to SNIPs via channel')

        # Run and disconnect board
        board.run(self.p.totalSteps)
        board.disconnect()

        # Perform postprocessing
        self.net.postProcessing()

    """
    @desc: Define function to learn as ouput
    @params:
            clusterIndex: index of cluster the function is defined for
            type: 'sin', 'revsin', 'lin'
    """
    def getTargetFunction(self, type = 'sin', steps = None):
        if steps is None:
            nTs = self.p.stepsPerTrial
        else:
            nTs = steps

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
