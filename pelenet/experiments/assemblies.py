# Official modules
import numpy as np

# Pelenet modules
from ._abstract import Experiment
from ..network import ReservoirNetwork

"""
@desc: Creating cell assemblies in a reservoir network
"""
class AssemblyExperiment(Experiment):

    """
    @desc: Define parameters for this experiment
    """
    def defineParameters(self):
        return {
            # # Experiment
            # 'seed': 1,
            # 'stepsPerTrial': 90,
            # # Network
            # 'reservoirExSize': 2048,
            # 'reservoirConnProb': None,
            # 'reservoirConnPerNeuron': 45,
            # 'isLearningRule': True,
            # 'learningRule': '2^-2*x1*y0 - 2^-2*y1*x0 + 2^-4*x1*y1*y0 - 2^-3*y0*w*w',
            # # Probes
            # 'isExSpikeProbe': True
        }

    """
    @desc: Build reservoir network with given mask and weights
    """
    def buildWithGivenMaskAndWeights(self, mask, weights):
        # Instanciate innate network
        self.net = ReservoirNetwork(self.p)

        # Set mask and weights
        self.net.initialMasks = mask
        self.net.initialWeights = weights

        # Connect ex-in reservoir
        self.net.connectReservoir()

        # Add cue
        self.net.addInput()

        # Add background noise
        if self.p.isNoise:
            self.net.addNoiseGenerator()

        # Add Probes
        self.net.addProbes()

        # Call afterBuild
        self.afterBuild()
