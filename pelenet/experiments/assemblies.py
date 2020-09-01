# Official modules
import numpy as np

# Pelenet modules
from ..system import System
from ..system.datalog import Datalog
from ..parameters import Parameters
from ..utils import Utils
from ..plots import Plot
from ..network import ReservoirNetwork

"""
@desc: Creating cell assemblies in a reservoir network
"""
class AssemblyExperiment():

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

    """
    @desc: Overwrite parameters for this experiment
    """
    def updateParameters(self):
        return {
            # Experiment
            'trials': 5,
            'stepsPerTrial': 90,
            # Network
            'reservoirExSize': 2048,
            'reservoirConnProb': None,
            'reservoirConnPerNeuron': 45,
            'isLearningRule': True,
            'learningRule': '2^-2*x1*y0 - 2^-2*y1*x0 + 2^-4*x1*y1*y0 - 2^-3*y0*w*w',
            # Probes
            'isExSpikeProbe': True
        }

    """
    @desc: Build all networks
    """
    def build(self):
        # Instanciate innate network
        self.net = ReservoirNetwork(self.p)

        # Draw mask and log-normal weights
        self.net.drawMaskAndWeights()

        # Connect ex-in reservoir
        self.net.connectReservoir()

        # Add patch input
        self.net.addAssemlyPatchGenerator()  # TODO

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
