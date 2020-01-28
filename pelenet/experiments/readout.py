# Loihi modules
import nxsdk.api.n2a as nx
from nxsdk.graph.processes.phase_enums import Phase

# Official modules
import numpy as np
import logging
from copy import deepcopy
import os

# Pelenet modules
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
        # Init super object
        super().__init__()

        # Define some further variables
        self.targetFunction = self.getTargetFunction()
    
    """
    @desc: Run whole experiment
    """
    def run(self):
        # Compile network
        compiler = nx.N2Compiler()
        board = compiler.compile(self.net.nxNet)
        logging.info('Network successfully compiled')

        # Add snips
        resetInitSnips = self.addSnips(board)
        logging.info('SNIPs added to chips')

        # Create channels for transfering initial values for the reset SNIP
        resetInitChannels = self.createAndConnectInitChannels(board, resetInitSnips)
        logging.info('Channels added')
        
        # Start board
        board.start()
        logging.info('Board successfully started')

        # Write initial data to channels
        for i in range(self.p.numChips):
            resetInitChannels[i].write(3, [
                self.p.neuronsPerCore,  # number of neurons per core
                self.p.totalTrialSteps,  # reset interval
                self.p.stopSteps  # number of steps to clear voltages/currents
            ])
        logging.info('Initial values transfered to SNIPs via channel')

        # Run and disconnect board
        board.run(self.p.totalSteps)
        board.disconnect()

        # Perform postprocessing
        self.net.postProcessing()

    """
    @desc: Add SNIPs to the chips of the system

    TODO: Only add SNIPs to used chips, not to all available chips on the system
    """
    def addSnips(self, board):
        snipDir = os.path.abspath(os.path.join('pelenet', 'snips'))
        print(snipDir)
        print(self.p.snipsPath)

        # Add one SNIPs to every chip
        resetInitSnips = []
        for i in range(self.p.numChips):
            # SNIP for initializing some values for the reset SNIP
            resetInitSnips.append(board.createSnip(
                name='init'+str(i),
                cFilePath=self.p.snipsPath + "/reset_init.c",
                includeDir=self.p.snipsPath,
                funcName='initialize_reset',
                phase=Phase.EMBEDDED_INIT,
                lmtId=0,
                chipId=i))

        # SNIPs for resetting the voltages and currents
        resetSnips = []
        for i in range(self.p.numChips):
            # Add one SNIP for every chip
            board.createSnip(
                name='reset'+str(i),
                cFilePath=self.p.snipsPath + "/reset.c",
                includeDir=self.p.snipsPath,
                guardName='do_reset',
                funcName='reset',
                phase=Phase.EMBEDDED_MGMT,
                lmtId=0,
                chipId=i)

        return resetInitSnips

    """
    @desc: Create and connect channels for initializing values for the reset SNIPs
    """
    def createAndConnectInitChannels(self, board, resetInitSnips):
        resetInitChannels = []
        # Add one channel to every chip
        for i in range(self.p.numChips):
            # Create channel for init data with buffer size of 3
            initResetChannel = board.createChannel(bytes('initreset'+str(i), 'utf-8'), "int", 3)
            
            # Connect channel to init snip
            initResetChannel.connect(None, resetInitSnips[i])

            # Add channel to list
            resetInitChannels.append(initResetChannel)

        return resetInitChannels


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
        #self.net.drawOutputMaskAndWeights()

        # Connect network
        self.net.addReservoirNetworkDistributed()

        # Add cue
        self.net.addRepeatedCueGenerator()

        # Add stop generator
        #self.net.addRepeatedStopGenerator()

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
